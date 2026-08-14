"""CLI reproducible para la investigacion independiente Matching-Moment IPSA.

El periodo 2024-2026 se trata como validacion de desarrollo porque ya fue
inspeccionado. Ningun resultado de este pipeline se etiqueta como test sellado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mm_ipsa.config import load_config, objective_weights, target_parameters
from mm_ipsa.lineage import assert_lineage_current, validate_lineage, write_lineage


def _discover_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_checkout = Path(__file__).resolve().parents[2]
    if source_checkout not in candidates:
        candidates.append(source_checkout)
    for candidate in candidates:
        if (candidate / "config.yaml").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "No se encontro la raiz del proyecto. Ejecute el comando dentro de un checkout "
        "que contenga config.yaml y pyproject.toml."
    )


ROOT = _discover_project_root()
PACKAGE_ROOT = ROOT / "src" / "mm_ipsa"


def banner(message: str) -> None:
    """Imprime un separador de etapa en la salida de progreso."""
    print(f"\n[step] {message}")


def _ensure_dirs(cfg: dict) -> None:
    for key in ("data_raw", "figures", "tables", "runs"):
        if key in cfg["paths"]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["data_raw"]) / "lineage").mkdir(parents=True, exist_ok=True)


def _lineage_path(cfg: dict, stage: str) -> Path:
    return Path(cfg["paths"]["data_raw"]) / "lineage" / f"{stage}.json"


def _common_lineage_inputs() -> list[Path]:
    return [
        ROOT / "run.py",
        ROOT / "config.yaml",
        PACKAGE_ROOT / "pipeline.py",
        PACKAGE_ROOT / "config.py",
        PACKAGE_ROOT / "lineage.py",
    ]


def _require_stage(cfg: dict, stage: str) -> Path:
    manifest = _lineage_path(cfg, stage)
    assert_lineage_current(manifest, root=ROOT)
    return manifest


def _write_stage(
    cfg: dict,
    stage: str,
    inputs: list[Path],
    outputs: list[Path],
    metadata: dict[str, object] | None = None,
) -> Path:
    manifest = write_lineage(
        _lineage_path(cfg, stage),
        stage,
        [*_common_lineage_inputs(), *inputs],
        outputs,
        root=ROOT,
        metadata=metadata,
    )
    print(f"  [lineage] {manifest}")
    return manifest


def _read_returns(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute los pasos previos.")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.rename(columns={column: column.replace(".SN", "") for column in frame.columns}, inplace=True)
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"Faltan activos en {path.name}: {missing}")
    result = frame.loc[:, labels].sort_index()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{path.name} contiene valores no finitos")
    return result


def _load_is_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(cfg["paths"]["data_raw"])
    H = int(cfg["data"]["H"])
    labels = cfg["asset_labels"]
    terminal = _read_returns(out / f"hist_terminal_returns_H{H}.csv", labels)
    daily = _read_returns(out / "daily_returns.csv", labels)
    cutoff = pd.Timestamp(cfg["data"]["end_train"])
    terminal = terminal.loc[terminal.index <= cutoff]
    daily = daily.loc[daily.index <= cutoff]
    if terminal.empty or daily.empty:
        raise RuntimeError("La muestra in-sample esta vacia")
    return terminal, daily


def _compute_targets(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    from mm_ipsa.mm.targets import compute_targets, save_targets

    terminal, daily = _load_is_data(cfg)
    parameters = target_parameters(cfg["mm"])
    moments, covariance, daily_mean = compute_targets(terminal, daily, **parameters)
    save_targets(moments, covariance, cfg["asset_labels"], cfg["paths"]["data_raw"])
    metadata = {
        **parameters,
        "ewma_half_life_weeks": cfg["mm"].get("ewma_half_life_weeks"),
        "observations_per_week": cfg["mm"].get("observations_per_week", 5.0),
        "target_rows": len(terminal),
        "target_start": str(terminal.index.min().date()),
        "target_end": str(terminal.index.max().date()),
    }
    (Path(cfg["paths"]["data_raw"]) / "targets_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return moments, covariance, daily_mean, terminal


def step_download(cfg: dict):
    """Descarga precios crudos y guarda mascara de disponibilidad y calidad."""
    banner("1/10 Ingestion y calidad de precios")
    _ensure_dirs(cfg)
    from mm_ipsa.data.download import download_prices

    prices = download_prices(cfg)
    out = Path(cfg["paths"]["data_raw"])
    _write_stage(
        cfg,
        "download",
        [PACKAGE_ROOT / "data" / "download.py"],
        [
            out / "adj_close_prices_raw.csv",
            out / "price_observation_mask.csv",
            out / "data_quality_prices.csv",
            out / "adj_close_prices.csv",
            out / "data_download_metadata.json",
        ],
        {"provider": "Yahoo Finance via yfinance"},
    )
    return prices


def step_reuse_download(cfg: dict):
    """Revalida una descarga local exacta sin consultar nuevamente al proveedor."""
    banner("Revalidacion de descarga local mediante hashes")
    _ensure_dirs(cfg)
    out = Path(cfg["paths"]["data_raw"])
    metadata_path = out / "data_download_metadata.json"
    required = [
        out / "adj_close_prices_raw.csv",
        out / "price_observation_mask.csv",
        out / "data_quality_prices.csv",
        out / "adj_close_prices.csv",
        metadata_path,
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"No se puede reutilizar la descarga; faltan archivos: {missing}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("requested_tickers") != list(cfg["assets"]):
        raise RuntimeError("La descarga local no corresponde al universo configurado")
    if metadata.get("end_exclusive") != cfg["data"]["end_oos"]:
        raise RuntimeError("La descarga local no corresponde al periodo configurado")
    stored_hashes = metadata.get("sha256", {})
    if not stored_hashes:
        raise RuntimeError("La metadata local no contiene hashes internos")
    mismatches = [
        name
        for name, expected in stored_hashes.items()
        if not (out / name).is_file() or _sha256(out / name) != expected
    ]
    if mismatches:
        raise RuntimeError(f"Hashes internos invalidos: {mismatches}")
    _write_stage(
        cfg,
        "download",
        [PACKAGE_ROOT / "data" / "download.py"],
        required,
        {
            "provider": "Yahoo Finance via yfinance",
            "source": "verified_existing_download",
            "network_request": False,
        },
    )
    print("  [ok] descarga existente revalidada; no se realizo una solicitud de red")
    return out / "adj_close_prices.csv"


def step_transform(cfg: dict):
    """Construye retornos diarios y terminales H, y separa entrenamiento de OOS."""
    banner("2/10 Retornos y split temporal")
    _ensure_dirs(cfg)
    from mm_ipsa.data.transform import build_returns

    price_path = Path(cfg["paths"]["data_raw"]) / "adj_close_prices.csv"
    if not price_path.exists():
        raise FileNotFoundError("Ejecute primero: mm-ipsa run --step download")
    upstream = [_require_stage(cfg, "download")]
    source_status = "verified_download_lineage"
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    mask_path = Path(cfg["paths"]["data_raw"]) / "price_observation_mask.csv"
    observation_mask = (
        pd.read_csv(mask_path, index_col=0, parse_dates=True) if mask_path.exists() else None
    )
    result = build_returns(cfg, prices, observation_mask=observation_mask)
    out = Path(cfg["paths"]["data_raw"])
    H = int(cfg["data"]["H"])
    _write_stage(
        cfg,
        "transform",
        [
            PACKAGE_ROOT / "data" / "transform.py",
            price_path,
            *([mask_path] if mask_path.exists() else []),
            *upstream,
        ],
        [
            out / "daily_returns.csv",
            out / "daily_returns_OOS.csv",
            out / f"hist_terminal_returns_H{H}.csv",
            out / f"terminal_returns_H{H}.csv",
            out / f"terminal_returns_H{H}_nonoverlap.csv",
            out / f"terminal_returns_H{H}_OOS.csv",
            out / f"terminal_returns_H{H}_OOS_nonoverlap.csv",
            out / "data_quality_returns.csv",
            out / "returns_metadata.json",
        ],
        {"source_status": source_status},
    )
    return result


def step_mm(cfg: dict):
    """Calibra la distribucion discreta MM resolviendo el BCD multi-start."""
    banner("3/10 Calibracion Matching-Moment con BCD validado")
    _ensure_dirs(cfg)
    from mm_ipsa.mm.bcd import BCDSolver
    from mm_ipsa.mm.diagnostics import MMDiagnostics
    from mm_ipsa.mm.objective import MMObjective

    transform_manifest = _require_stage(cfg, "transform")
    moments, covariance, _, terminal = _compute_targets(cfg)
    mm_cfg = cfg["mm"]
    objective = MMObjective(
        moments,
        covariance,
        objective_weights(mm_cfg),
        int(mm_cfg["N_scenarios"]),
    )
    solver = BCDSolver(objective, mm_cfg, warm_noise=float(mm_cfg.get("warm_noise", 0.15)))
    scenarios, probabilities = solver.solve(out_path=cfg["paths"]["data_raw"])

    out = Path(cfg["paths"]["data_raw"])
    labels = cfg["asset_labels"]
    np.save(out / "mm_scenarios_x.npy", scenarios)
    np.save(out / "mm_probabilities_p.npy", probabilities)
    pd.DataFrame(scenarios, columns=labels).to_csv(out / "mm_scenarios_x.csv", index=False)
    pd.DataFrame({"scenario": np.arange(len(probabilities)), "probability": probabilities}).to_csv(
        out / "mm_probabilities_p.csv", index=False
    )
    solver.summary_table().to_csv(Path(cfg["paths"]["tables"]) / "bcd_starts.csv", index=False)
    selected_start = next(
        record for record in solver.all_starts if record["start_id"] == solver.best_start_id
    )
    calibration = {
        "F": objective.evaluate(scenarios, probabilities),
        "G": solver.regularized_objective(scenarios, probabilities),
        "KL_uniform": solver._kl_uniform(probabilities),
        "N_eff": float(1.0 / np.sum(probabilities**2)),
        "stationarity": solver.stationarity_metrics(scenarios, probabilities),
        "best_start_converged": bool(selected_start["converged"]),
        "best_start_stationary": bool(selected_start["stationarity_pass"]),
        "selected_start_id": solver.best_start_id,
        "selection_requires_stationarity": True,
        "lowest_G_any_start": float(
            min(record["G_fin"] for record in solver.all_starts)
        ),
        "all_starts_hit_max_iter": bool(
            all(record["stop_reason"] == "max_iter" for record in solver.all_starts)
        ),
        "solver_events": solver.solver_events,
        "components": objective.components(scenarios, probabilities),
    }
    (out / "mm_calibration_metrics.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    MMDiagnostics(scenarios, probabilities, moments, covariance, labels, cfg).run_all(
        solver.history, solver.all_starts, terminal
    )
    H = int(cfg["data"]["H"])
    _write_stage(
        cfg,
        "mm",
        [
            PACKAGE_ROOT / "mm" / "objective.py",
            PACKAGE_ROOT / "mm" / "bcd.py",
            PACKAGE_ROOT / "mm" / "targets.py",
            transform_manifest,
            out / "daily_returns.csv",
            out / f"hist_terminal_returns_H{H}.csv",
        ],
        [
            out / "mm_scenarios_x.npy",
            out / "mm_probabilities_p.npy",
            out / "mm_calibration_metrics.json",
            out / "objective_history.csv",
            out / "regularized_objective_history.csv",
            out / "targets_moments.csv",
            out / "targets_cov.csv",
            Path(cfg["paths"]["tables"]) / "bcd_starts.csv",
        ],
        {
            "best_start_converged": calibration["best_start_converged"],
            "best_start_stationary": calibration["best_start_stationary"],
            "selected_start_id": calibration["selected_start_id"],
        },
    )
    print(f"  [ok] escenarios={scenarios.shape}, F={calibration['F']:.6g}, G={calibration['G']:.6g}")
    return scenarios, probabilities


def step_benchmarks(cfg: dict):
    """Genera los controles terminales con la misma media, covarianza y horizonte."""
    banner("4/10 Benchmarks terminales comparables")
    _ensure_dirs(cfg)
    from mm_ipsa.mm.targets import _ewma_weights
    from mm_ipsa.models.benchmarks import generate_benchmarks, resolve_student_t_df

    transform_manifest = _require_stage(cfg, "transform")
    moments, covariance, _, terminal = _compute_targets(cfg)
    parameters = target_parameters(cfg["mm"])
    history_weights = _ewma_weights(len(terminal), parameters["decay_lambda"])
    benchmark_cfg = cfg["benchmarks"]
    student_t_df, student_t_report = resolve_student_t_df(
        benchmark_cfg,
        terminal.to_numpy(),
        moments[0],
        covariance,
        history_weights,
    )
    out = Path(cfg["paths"]["data_raw"])
    labels = cfg["asset_labels"]
    print(
        f"  student_t: nu={student_t_df:.3f} (modo={student_t_report['student_t_df_mode']})"
    )
    daily_is = _read_returns(out / "daily_returns.csv", labels)
    models, benchmark_diagnostics = generate_benchmarks(
        moments,
        covariance,
        terminal.to_numpy(),
        history_weights,
        n_scenarios=int(benchmark_cfg["N_scenarios"]),
        seed=int(benchmark_cfg["seed"]),
        student_t_df=student_t_df,
        daily_returns=daily_is.to_numpy(),
        horizon=int(cfg["data"]["H"]),
        include=list(benchmark_cfg["include"]),
    )
    if benchmark_diagnostics:
        student_t_report.update(benchmark_diagnostics)
        print(
            "  dcc_garch: a={dcc_a:.4f}, b={dcc_b:.4f}, "
            "persistencia={dcc_persistence:.4f}, nu_innovaciones={innovation_df:.2f}".format(
                **benchmark_diagnostics
            )
        )
    # Se escribe despues de incorporar el diagnostico del ajuste DCC-GARCH.
    (out / "student_t_df_estimation.json").write_text(
        json.dumps(student_t_report, indent=2), encoding="utf-8"
    )
    labels = cfg["asset_labels"]
    for name, (scenarios, probabilities) in models.items():
        np.save(out / f"{name}_scenarios.npy", scenarios)
        np.save(out / f"{name}_probabilities.npy", probabilities)
        pd.DataFrame(scenarios, columns=labels).to_csv(out / f"{name}_scenarios.csv", index=False)
    print("  [ok] " + ", ".join(f"{name}={values[0].shape}" for name, values in models.items()))
    H = int(cfg["data"]["H"])
    benchmark_outputs: list[Path] = []
    for name in cfg["benchmarks"]["include"]:
        benchmark_outputs.extend([out / f"{name}_scenarios.npy", out / f"{name}_probabilities.npy"])
    _write_stage(
        cfg,
        "benchmarks",
        [
            PACKAGE_ROOT / "models" / "benchmarks.py",
            PACKAGE_ROOT / "mm" / "targets.py",
            transform_manifest,
            out / "daily_returns.csv",
            out / f"hist_terminal_returns_H{H}.csv",
        ],
        [
            *benchmark_outputs,
            out / "targets_moments.csv",
            out / "targets_cov.csv",
            out / "student_t_df_estimation.json",
        ],
        {
            "models": list(cfg["benchmarks"]["include"]),
            "student_t_df": float(student_t_df),
            "student_t_df_mode": str(student_t_report["student_t_df_mode"]),
        },
    )
    return models


def _load_models(cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = Path(cfg["paths"]["data_raw"])
    models = {
        "MM": (
            np.load(out / "mm_scenarios_x.npy"),
            np.load(out / "mm_probabilities_p.npy"),
        )
    }
    for name in cfg["benchmarks"]["include"]:
        models[name] = (
            np.load(out / f"{name}_scenarios.npy"),
            np.load(out / f"{name}_probabilities.npy"),
        )
    return models


def step_evaluate(cfg: dict):
    """Puntua todos los modelos OOS y produce contrastes, MCS y calibracion."""
    banner("5/10 Evaluacion probabilistica OOS")
    _ensure_dirs(cfg)
    from mm_ipsa.evaluation.comparison import compare_focal_model
    from mm_ipsa.evaluation.scoring import evaluate_scenarios_detailed

    mm_manifest = _require_stage(cfg, "mm")
    benchmark_manifest = _require_stage(cfg, "benchmarks")
    out = Path(cfg["paths"]["data_raw"])
    tables = Path(cfg["paths"]["tables"])
    H = int(cfg["data"]["H"])
    labels = cfg["asset_labels"]
    oos_path = out / f"terminal_returns_H{H}_OOS_nonoverlap.csv"
    observations = _read_returns(oos_path, labels)
    marginal_frames, aggregate_frames, observation_frames = [], [], []
    for idx, (name, (scenarios, probabilities)) in enumerate(_load_models(cfg).items()):
        marginal, aggregate, by_observation = evaluate_scenarios_detailed(
            name,
            scenarios,
            probabilities,
            observations.to_numpy(),
            labels,
            observation_ids=observations.index,
            alpha=float(cfg["portfolio"]["alpha_cvar"]),
            seed=int(cfg["evaluation"]["seed"]) + idx * 10_000,
            energy_pair_samples=int(cfg["evaluation"]["energy_pair_samples"]),
        )
        marginal_frames.append(marginal)
        aggregate_frames.append(aggregate)
        observation_frames.append(by_observation)
    marginal_all = pd.concat(marginal_frames, ignore_index=True)
    aggregate_all = pd.concat(aggregate_frames, ignore_index=True).sort_values("mean_crps")
    observation_all = pd.concat(observation_frames, ignore_index=True)
    score_differences = compare_focal_model(
        observation_all,
        focal_model="MM",
        benchmark_models=list(cfg["benchmarks"]["include"]),
        block_size=int(cfg["evaluation"]["score_bootstrap_block_size"]),
        block_size_mode=str(
            cfg["evaluation"].get("score_bootstrap_block_size_mode", "auto")
        ),
        samples=int(cfg["evaluation"]["score_bootstrap_samples"]),
        confidence_level=float(cfg["evaluation"]["score_bootstrap_confidence"]),
        seed=int(cfg["evaluation"]["seed"]),
        inference_status=str(cfg["evaluation"]["status"]),
    )
    confidence_sets = _model_confidence_sets(
        observation_all,
        block_sizes=dict(
            zip(
                score_differences["metric"],
                score_differences["block_size"],
                strict=False,
            )
        ),
        alpha=1.0 - float(cfg["evaluation"]["score_bootstrap_confidence"]),
        samples=int(cfg["evaluation"]["score_bootstrap_samples"]),
        seed=int(cfg["evaluation"]["seed"]) + 500_000,
    )
    calibration_detail, calibration_summary = _calibration_frames(
        cfg,
        _load_models(cfg),
        observations.to_numpy(),
        labels,
        int(cfg["evaluation"]["seed"]) + 600_000,
    )
    marginal_all.to_csv(tables / "probabilistic_scores_by_asset.csv", index=False)
    aggregate_all.to_csv(tables / "probabilistic_scores_summary.csv", index=False)
    observation_all.to_csv(tables / "probabilistic_scores_by_observation.csv", index=False)
    score_differences.to_csv(tables / "probabilistic_score_differences.csv", index=False)
    confidence_sets.to_csv(tables / "model_confidence_set.csv", index=False)
    calibration_detail.to_csv(tables / "calibration_pit_by_asset.csv", index=False)
    calibration_summary.to_csv(tables / "calibration_pit_summary.csv", index=False)
    model_inputs = [out / "mm_scenarios_x.npy", out / "mm_probabilities_p.npy"]
    for name in cfg["benchmarks"]["include"]:
        model_inputs.extend([out / f"{name}_scenarios.npy", out / f"{name}_probabilities.npy"])
    _write_stage(
        cfg,
        "evaluation",
        [
            PACKAGE_ROOT / "evaluation" / "scoring.py",
            PACKAGE_ROOT / "evaluation" / "comparison.py",
            mm_manifest,
            benchmark_manifest,
            oos_path,
            *model_inputs,
        ],
        [
            tables / "probabilistic_scores_by_asset.csv",
            tables / "probabilistic_scores_summary.csv",
            tables / "probabilistic_scores_by_observation.csv",
            tables / "probabilistic_score_differences.csv",
            tables / "model_confidence_set.csv",
            tables / "calibration_pit_by_asset.csv",
            tables / "calibration_pit_summary.csv",
        ],
        {
            "status": cfg["evaluation"]["status"],
            "oos_rows": len(observations),
            "score_bootstrap_block_size": cfg["evaluation"]["score_bootstrap_block_size"],
            "score_bootstrap_samples": cfg["evaluation"]["score_bootstrap_samples"],
            "multiple_testing": cfg["evaluation"]["score_multiple_testing"],
        },
    )
    print("\n" + aggregate_all.to_string(index=False))
    print(f"  estado_evaluacion={cfg['evaluation']['status']} (no es test sellado)")
    return aggregate_all


def _calibration_frames(
    cfg: dict,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    observations: np.ndarray,
    labels: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostico PIT con soporte igualado segun configuracion."""
    from mm_ipsa.evaluation.calibration import calibration_report

    evaluation = cfg["evaluation"]
    support = (
        min(len(scenarios) for scenarios, _ in models.values())
        if bool(evaluation.get("calibration_equalise_support", True))
        else None
    )
    return calibration_report(
        models,
        observations,
        labels,
        seed=seed,
        bins=int(evaluation.get("calibration_bins", 10)),
        support_size=support,
    )


def _model_confidence_sets(
    observation_scores: pd.DataFrame,
    *,
    block_sizes: dict[str, int],
    alpha: float,
    samples: int,
    seed: int,
    group_column: str | None = None,
    metrics: Sequence[str] = ("mean_crps", "energy_score", "variogram_score"),
) -> pd.DataFrame:
    """Construye un Model Confidence Set por metrica sobre las mismas fechas.

    Reutiliza el ancho de bloque ya resuelto para cada metrica en la tabla de
    contrastes pareados, de modo que ambas inferencias describan la misma
    estructura de dependencia temporal.
    """
    from mm_ipsa.evaluation.model_confidence import model_confidence_set

    index: list[str] = (
        ["observation"] if group_column is None else [group_column, "observation"]
    )
    frames: list[pd.DataFrame] = []
    for position, metric in enumerate(metrics):
        pivot = observation_scores.pivot_table(
            index=index, columns="model", values=metric
        )
        groups = (
            None
            if group_column is None
            else pivot.index.get_level_values(group_column).to_numpy()
        )
        frame = model_confidence_set(
            pivot,
            alpha=alpha,
            block_size=int(block_sizes.get(metric, 1)),
            samples=samples,
            seed=seed + position * 1_000,
            groups=groups,
        )
        frame.insert(0, "metric", metric)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def step_portfolio(cfg: dict):
    """Construye carteras derivadas de cada modelo y los baselines robustos."""
    banner("6/10 Portafolios robustos y baselines ingenuos")
    _ensure_dirs(cfg)
    from mm_ipsa.portfolio.optimization import (
        equal_weight,
        hierarchical_risk_parity,
        inverse_variance,
        maximum_sharpe,
        minimum_cvar,
        minimum_variance,
        portfolio_diagnostics,
        weighted_mean_cov,
    )

    mm_manifest = _require_stage(cfg, "mm")
    benchmark_manifest = _require_stage(cfg, "benchmarks")
    labels = cfg["asset_labels"]
    portfolio_cfg = cfg["portfolio"]
    max_weight = float(portfolio_cfg["max_weight"])
    l2_penalty = float(portfolio_cfg.get("l2_penalty", 0.0))
    alpha = float(portfolio_cfg["alpha_cvar"])
    models = _load_models(cfg)
    weights_by_name: dict[str, np.ndarray] = {"EqualWeight": equal_weight(len(labels))}

    target_covariance = pd.read_csv(
        Path(cfg["paths"]["data_raw"]) / "targets_cov.csv", index_col=0
    ).loc[labels, labels].to_numpy()
    weights_by_name["InverseVariance"] = inverse_variance(target_covariance)
    weights_by_name["HRP"] = hierarchical_risk_parity(target_covariance)

    for model, (scenarios, probabilities) in models.items():
        mean, covariance = weighted_mean_cov(scenarios, probabilities)
        weights_by_name[f"{model}_MinVariance"] = minimum_variance(covariance, max_weight, l2_penalty)
        weights_by_name[f"{model}_MinCVaR"] = minimum_cvar(scenarios, probabilities, alpha, max_weight)
        weights_by_name[f"{model}_MaxSharpe"] = maximum_sharpe(
            mean,
            covariance,
            max_weight=max_weight,
            risk_free_rate=float(portfolio_cfg["rf"]),
            l2_penalty=l2_penalty,
            seed=int(cfg["evaluation"]["seed"]),
        )

    weight_rows, metric_rows = [], []
    reference_scenarios, reference_p = models["MM"]
    for name, weights in weights_by_name.items():
        for label, value in zip(labels, weights):
            weight_rows.append({"portfolio": name, "asset": label, "weight": float(value)})
        metric_rows.append(
            {
                "portfolio": name,
                **portfolio_diagnostics(
                    weights,
                    reference_scenarios,
                    reference_p,
                    alpha,
                    risk_free_rate=float(portfolio_cfg["rf"]),
                ),
            }
        )
    tables = Path(cfg["paths"]["tables"])
    pd.DataFrame(weight_rows).to_csv(tables / "portfolio_weights_research.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(tables / "portfolio_metrics_in_sample.csv", index=False)
    # Formato transitorio para graficos/documentos historicos.
    pd.DataFrame(
        [
            {"Portafolio": row["portfolio"], "Activo": row["asset"], "Peso_pct": row["weight"] * 100}
            for row in weight_rows
        ]
    ).to_csv(tables / "portfolio_weights.csv", index=False)
    out = Path(cfg["paths"]["data_raw"])
    model_inputs = [out / "mm_scenarios_x.npy", out / "mm_probabilities_p.npy"]
    for model_name in cfg["benchmarks"]["include"]:
        model_inputs.extend([out / f"{model_name}_scenarios.npy", out / f"{model_name}_probabilities.npy"])
    _write_stage(
        cfg,
        "portfolio",
        [
            PACKAGE_ROOT / "portfolio" / "optimization.py",
            mm_manifest,
            benchmark_manifest,
            out / "targets_cov.csv",
            *model_inputs,
        ],
        [tables / "portfolio_weights_research.csv", tables / "portfolio_metrics_in_sample.csv"],
        {"strategies": len(weights_by_name)},
    )
    print(f"  [ok] {len(weights_by_name)} estrategias; EqualWeight/IV/HRP incluidos")
    return weights_by_name


def _adjust_portfolio_multiplicity(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica Holm dentro de cada familia de contrastes de portafolio.

    Los contrastes like-for-like de H4 y los del efecto de rebalanceo responden
    preguntas distintas, por lo que se corrigen por separado. Sin este ajuste,
    afirmar que "algun portafolio mejora el Sharpe" sobre seis o mas
    comparaciones simultaneas infla la probabilidad de un falso positivo.
    """
    from mm_ipsa.evaluation.comparison import holm_adjust

    if frame.empty:
        return frame
    adjusted = frame.copy()
    adjusted["pvalue_holm"] = np.nan
    for family, group in adjusted.groupby("evaluation_design"):
        adjusted.loc[group.index, "pvalue_holm"] = holm_adjust(group["pvalue_raw"])
    adjusted["reject_holm_5pct"] = adjusted["pvalue_holm"] < 0.05
    adjusted["ci_excludes_zero"] = (adjusted["ci_low"] > 0.0) | (
        adjusted["ci_high"] < 0.0
    )
    adjusted["multiple_testing"] = "holm_within_evaluation_design"
    return adjusted


def step_backtest(cfg: dict):
    """Simula las carteras OOS con deriva de pesos, turnover y costos explicitos."""
    banner("7/10 Backtest real con costos y walk-forward")
    _ensure_dirs(cfg)
    from mm_ipsa.backtest.walk_forward import (
        moving_block_bootstrap_sharpe_difference,
        performance_metrics,
        simulate_strategy,
        walk_forward_weights,
    )
    from mm_ipsa.portfolio.optimization import (
        equal_weight,
        hierarchical_risk_parity,
        inverse_variance,
    )

    transform_manifest = _require_stage(cfg, "transform")
    portfolio_manifest = _require_stage(cfg, "portfolio")
    out = Path(cfg["paths"]["data_raw"])
    tables = Path(cfg["paths"]["tables"])
    labels = cfg["asset_labels"]
    daily_is = _read_returns(out / "daily_returns.csv", labels)
    daily_oos = _read_returns(out / "daily_returns_OOS.csv", labels)
    full = pd.concat([daily_is, daily_oos]).sort_index()
    start_oos = pd.Timestamp(cfg["data"]["start_oos"])
    if daily_oos.index.min() < start_oos:
        raise RuntimeError("El archivo OOS contiene fechas previas a start_oos")

    weights_frame = pd.read_csv(tables / "portfolio_weights_research.csv")
    static_weights: dict[str, np.ndarray] = {}
    for name, group in weights_frame.groupby("portfolio"):
        selected_weights = group.set_index("asset").loc[labels, ["weight"]]
        static_weights[str(name)] = selected_weights["weight"].to_numpy(dtype=float)
    first_oos_date = pd.Timestamp(str(daily_oos.index[0]))
    schedules: dict[str, dict[pd.Timestamp, np.ndarray]] = {
        name: {first_oos_date: weights} for name, weights in static_weights.items()
    }
    designs = {name: "static_single_shot_oos" for name in schedules}

    def estimator_equal(training: pd.DataFrame) -> np.ndarray:
        return equal_weight(training.shape[1])

    def estimator_iv(training: pd.DataFrame) -> np.ndarray:
        return inverse_variance(training.cov(ddof=0).to_numpy())

    def estimator_hrp(training: pd.DataFrame) -> np.ndarray:
        return hierarchical_risk_parity(training.cov(ddof=0).to_numpy())

    for name, estimator in {
        "WF_EqualWeight": estimator_equal,
        "WF_InverseVariance": estimator_iv,
        "WF_HRP": estimator_hrp,
    }.items():
        schedules[name] = walk_forward_weights(
            full,
            start_oos,
            estimator,
            frequency="Q",
            min_history=252,
        )
        designs[name] = "expanding_window_walk_forward"

    cost_bps = float(cfg["portfolio"].get("transaction_cost_bps", 0.0))
    results, executions = {}, {}
    rows, schedule_rows = [], []
    for name, schedule in schedules.items():
        wealth, execution, execution_info = simulate_strategy(daily_oos, schedule, cost_bps)
        metrics = performance_metrics(wealth, execution)
        rows.append({
            "portfolio": name,
            "evaluation_design": designs[name],
            **metrics,
            **execution_info,
            "transaction_cost_bps": cost_bps,
        })
        results[name], executions[name] = wealth, execution
        for date, weights in schedule.items():
            for asset, weight in zip(labels, weights):
                schedule_rows.append({"portfolio": name, "rebalance_date": date, "asset": asset, "weight": weight})
        execution.to_csv(tables / f"execution_{name}.csv")

    # Cada estrategia se contrasta contra el Equal Weight de su MISMO diseno de
    # evaluacion. Comparar una cartera congelada contra un baseline que se
    # recalibra cada trimestre mezcla dos efectos distintos -metodo de
    # construccion y politica de rebalanceo- y hace que H4 no sea interpretable.
    benchmark_by_design = {
        "static_single_shot_oos": "EqualWeight",
        "expanding_window_walk_forward": "WF_EqualWeight",
    }
    missing_benchmarks = sorted(
        {
            benchmark_by_design[design]
            for design in designs.values()
            if benchmark_by_design[design] not in executions
        }
    )
    if missing_benchmarks:
        raise RuntimeError(
            f"Faltan baselines comparables por diseno: {missing_benchmarks}"
        )

    bootstrap_rows = []
    for name, execution in executions.items():
        design = designs[name]
        benchmark_name = benchmark_by_design[design]
        if name == benchmark_name:
            continue
        interval = moving_block_bootstrap_sharpe_difference(
            execution["net_return"].to_numpy(),
            executions[benchmark_name]["net_return"].to_numpy(),
            block_size=int(cfg["evaluation"]["bootstrap_block_size"]),
            samples=int(cfg["evaluation"]["bootstrap_samples"]),
            seed=int(cfg["evaluation"]["seed"]),
        )
        bootstrap_rows.append({
            "portfolio": name,
            "evaluation_design": design,
            "benchmark": benchmark_name,
            "comparison_is_like_for_like": True,
            **interval,
        })

    # Pregunta separada y legitima: cuanto aporta rebalancear. Se reporta
    # aparte y etiquetado, nunca mezclado con el contraste de H4.
    for name in ("WF_EqualWeight", "WF_InverseVariance", "WF_HRP"):
        static_counterpart = name.removeprefix("WF_")
        if name not in executions or static_counterpart not in executions:
            continue
        interval = moving_block_bootstrap_sharpe_difference(
            executions[name]["net_return"].to_numpy(),
            executions[static_counterpart]["net_return"].to_numpy(),
            block_size=int(cfg["evaluation"]["bootstrap_block_size"]),
            samples=int(cfg["evaluation"]["bootstrap_samples"]),
            seed=int(cfg["evaluation"]["seed"]),
        )
        bootstrap_rows.append({
            "portfolio": name,
            "evaluation_design": "rebalancing_effect",
            "benchmark": static_counterpart,
            "comparison_is_like_for_like": False,
            **interval,
        })

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame = _adjust_portfolio_multiplicity(bootstrap_frame)

    metrics_frame = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    metrics_frame.to_csv(tables / "backtest_metrics.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(tables / "walk_forward_weights.csv", index=False)
    bootstrap_frame.to_csv(tables / "bootstrap_sharpe_differences.csv", index=False)
    wealth_frame = pd.DataFrame(results)
    wealth_frame.to_csv(tables / "backtest_wealth.csv")
    execution_outputs = [tables / f"execution_{name}.csv" for name in schedules]
    _write_stage(
        cfg,
        "backtest",
        [
            PACKAGE_ROOT / "backtest" / "walk_forward.py",
            PACKAGE_ROOT / "portfolio" / "optimization.py",
            transform_manifest,
            portfolio_manifest,
            out / "daily_returns.csv",
            out / "daily_returns_OOS.csv",
            tables / "portfolio_weights_research.csv",
        ],
        [
            tables / "backtest_metrics.csv",
            tables / "bootstrap_sharpe_differences.csv",
            tables / "backtest_wealth.csv",
            tables / "walk_forward_weights.csv",
            *execution_outputs,
        ],
        {
            "transaction_cost_bps": cost_bps,
            "benchmark_by_design": benchmark_by_design,
            "multiple_testing": "holm_within_evaluation_design",
        },
    )
    print("\n" + metrics_frame.to_string(index=False))
    print("  [ok] resultados netos de costos; sin fallback sintetico")
    return metrics_frame


def step_liquidity_robustness(cfg: dict):
    """Repite el analisis sobre el universo liquido seleccionado solo con datos IS."""
    banner("8/10 Robustez de liquidez con filtro exclusivamente in-sample")
    _ensure_dirs(cfg)
    from mm_ipsa.analysis.liquidity_robustness import (
        load_liquidity_config,
        run_liquidity_robustness,
    )

    transform_manifest = _require_stage(cfg, "transform")
    evaluation_manifest = _require_stage(cfg, "evaluation")
    backtest_manifest = _require_stage(cfg, "backtest")
    experiment_path = ROOT / "research" / "liquidity_robustness.yaml"
    experiment_cfg = load_liquidity_config(experiment_path)
    source = Path(cfg["paths"]["data_raw"])
    output_dir = source / "robustness" / "liquidity"
    result = run_liquidity_robustness(cfg, experiment_cfg, output_dir)

    H = int(cfg["data"]["H"])
    _write_stage(
        cfg,
        "liquidity_robustness",
        [
            PACKAGE_ROOT / "analysis" / "liquidity_robustness.py",
            PACKAGE_ROOT / "mm" / "objective.py",
            PACKAGE_ROOT / "mm" / "bcd.py",
            PACKAGE_ROOT / "mm" / "targets.py",
            PACKAGE_ROOT / "models" / "benchmarks.py",
            PACKAGE_ROOT / "evaluation" / "scoring.py",
            PACKAGE_ROOT / "evaluation" / "comparison.py",
            PACKAGE_ROOT / "portfolio" / "optimization.py",
            PACKAGE_ROOT / "backtest" / "walk_forward.py",
            experiment_path,
            transform_manifest,
            evaluation_manifest,
            backtest_manifest,
            source / "data_quality_returns.csv",
            source / "daily_returns.csv",
            source / "daily_returns_OOS.csv",
            source / f"hist_terminal_returns_H{H}.csv",
            source / f"terminal_returns_H{H}_OOS_nonoverlap.csv",
            Path(cfg["paths"]["tables"]) / "probabilistic_scores_summary.csv",
            Path(cfg["paths"]["tables"]) / "backtest_metrics.csv",
        ],
        result["artifacts"],
        {
            "experiment_id": result["metadata"]["experiment_id"],
            "selection_sample": result["metadata"]["selection_sample"],
            "selection_uses_oos": result["metadata"]["selection_uses_oos"],
            "selected_assets": result["metadata"]["selected_assets"],
            "excluded_assets": result["metadata"]["excluded_assets"],
            "primary_universe_unchanged": True,
        },
    )
    print(
        f"  [ok] universo liquido={result['metadata']['selected_count']}/"
        f"{result['metadata']['full_count']}; seleccion solo IS"
    )
    print("\n" + result["scores"].to_string(index=False))
    return result


def step_rolling_origin(cfg: dict):
    """Ejecuta la validacion rolling-origin recalibrando todos los modelos por fold."""
    banner("9/10 Validacion rolling-origin con recalibracion por fold")
    _ensure_dirs(cfg)
    from mm_ipsa.analysis.rolling_origin import (
        load_rolling_origin_config,
        run_rolling_origin,
    )

    transform_manifest = _require_stage(cfg, "transform")
    experiment_path = ROOT / "research" / "rolling_origin.yaml"
    experiment_cfg = load_rolling_origin_config(experiment_path)
    source = Path(cfg["paths"]["data_raw"])
    daily_is_path = source / "daily_returns.csv"
    daily_oos_path = source / "daily_returns_OOS.csv"
    daily_is = _read_returns(daily_is_path, cfg["asset_labels"])
    daily_oos = _read_returns(daily_oos_path, cfg["asset_labels"])
    output_dir = source / "robustness" / "rolling_origin"
    result = run_rolling_origin(
        cfg,
        experiment_cfg,
        daily_is,
        daily_oos,
        output_dir,
    )
    _write_stage(
        cfg,
        "rolling_origin",
        [
            PACKAGE_ROOT / "analysis" / "rolling_origin.py",
            PACKAGE_ROOT / "data" / "transform.py",
            PACKAGE_ROOT / "mm" / "objective.py",
            PACKAGE_ROOT / "mm" / "bcd.py",
            PACKAGE_ROOT / "mm" / "targets.py",
            PACKAGE_ROOT / "models" / "benchmarks.py",
            PACKAGE_ROOT / "evaluation" / "scoring.py",
            PACKAGE_ROOT / "evaluation" / "comparison.py",
            experiment_path,
            transform_manifest,
            daily_is_path,
            daily_oos_path,
        ],
        result["artifacts"],
        {
            "experiment_id": result["metadata"]["experiment_id"],
            "window_type": result["metadata"]["window_type"],
            "fold_count": result["metadata"]["fold_count"],
            "total_evaluation_windows": result["metadata"][
                "total_evaluation_windows"
            ],
            "refit_all_models_each_fold": True,
            "temporal_leakage_detected": False,
        },
    )
    print("\n" + result["pooled_scores"].to_string(index=False))
    print(
        f"  [ok] folds={result['metadata']['fold_count']}, "
        f"ventanas={result['metadata']['total_evaluation_windows']}; sin look-ahead"
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_fingerprint() -> tuple[str, list[str]]:
    files = sorted(
        [ROOT / "run.py", ROOT / "config.yaml", ROOT / "requirements.txt"]
        + list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest(), [str(path.relative_to(ROOT)) for path in files]


def step_snapshot(cfg: dict):
    """Congela configuracion, entorno y hashes de artefactos para reproducibilidad."""
    banner("10/10 Snapshot auditable")
    _ensure_dirs(cfg)
    required_stages = (
        "download", "transform", "mm", "benchmarks", "evaluation", "portfolio",
        "backtest", "liquidity_robustness", "rolling_origin",
    )
    current_lineages = [_require_stage(cfg, stage) for stage in required_stages]
    source = Path(cfg["paths"]["data_raw"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(cfg["paths"]["runs"]) / run_id
    destination.mkdir(parents=True, exist_ok=False)

    root_artifacts = {
        "adj_close_prices_raw.csv", "price_observation_mask.csv",
        "data_quality_prices.csv", "data_download_metadata.json",
        "adj_close_prices.csv", "daily_returns.csv", "daily_returns_OOS.csv",
        "data_quality_returns.csv", "returns_metadata.json",
        f"hist_terminal_returns_H{cfg['data']['H']}.csv",
        f"terminal_returns_H{cfg['data']['H']}.csv",
        f"terminal_returns_H{cfg['data']['H']}_nonoverlap.csv",
        f"terminal_returns_H{cfg['data']['H']}_OOS.csv",
        f"terminal_returns_H{cfg['data']['H']}_OOS_nonoverlap.csv",
        "targets_moments.csv", "targets_cov.csv", "targets_metadata.json",
        "mm_scenarios_x.npy", "mm_probabilities_p.npy",
        "mm_calibration_metrics.json", "objective_history.csv",
        "regularized_objective_history.csv",
    }
    for model in ("gaussian_terminal", "student_t_terminal", "historical_weighted"):
        root_artifacts.update({f"{model}_scenarios.npy", f"{model}_probabilities.npy"})
    table_artifacts = {
        "bcd_starts.csv", "bcd_multistart_ranking.csv", "moments_comparison.csv",
        "mae_moments.csv", "scenario_probs.csv",
        "probabilistic_scores_by_asset.csv", "probabilistic_scores_summary.csv",
        "probabilistic_scores_by_observation.csv", "probabilistic_score_differences.csv",
        "portfolio_weights_research.csv", "portfolio_metrics_in_sample.csv",
        "backtest_metrics.csv", "backtest_wealth.csv",
        "walk_forward_weights.csv", "bootstrap_sharpe_differences.csv",
    }
    candidates = [source / name for name in sorted(root_artifacts) if (source / name).is_file()]
    tables_dir = source / "tables"
    if tables_dir.exists():
        candidates.extend(tables_dir / name for name in sorted(table_artifacts) if (tables_dir / name).is_file())
        candidates.extend(sorted(tables_dir.glob("execution_*.csv")))
    robustness_dir = source / "robustness" / "liquidity"
    if robustness_dir.exists():
        candidates.extend(sorted(path for path in robustness_dir.iterdir() if path.is_file()))
    rolling_dir = source / "robustness" / "rolling_origin"
    if rolling_dir.exists():
        candidates.extend(sorted(path for path in rolling_dir.rglob("*") if path.is_file()))
    candidates.extend(current_lineages)
    copied = []
    for path in candidates:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({"path": str(relative).replace("\\", "/"), "sha256": _sha256(target), "bytes": target.stat().st_size})
    for name in ("config.yaml", "requirements.txt", "requirements-lock.txt", "pyproject.toml"):
        path = ROOT / name
        if path.exists():
            target = destination / name
            shutil.copy2(path, target)
            copied.append({"path": name, "sha256": _sha256(target), "bytes": target.stat().st_size})

    research_sources = (
        ROOT / "research" / "PROTOCOL.md",
        ROOT / "research" / "REFERENCES.md",
        ROOT / "research" / "RESULTS_20260810.md",
        ROOT / "research" / "liquidity_robustness.yaml",
        ROOT / "research" / "rolling_origin.yaml",
        ROOT / "research" / "MM_Research_Report.tex",
        ROOT / "research" / "build" / "MM_Research_Report.pdf",
    )
    for path in research_sources:
        if path.exists():
            relative = Path("research") / path.name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append({
                "path": relative.as_posix(),
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
            })

    packages = {}
    for package in ("numpy", "pandas", "scipy", "yfinance", "matplotlib", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    code_hash, code_files = _code_fingerprint()
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    calibration = {}
    calibration_path = source / "mm_calibration_metrics.json"
    if calibration_path.exists():
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = {
            key: calibration_payload.get(key)
            for key in ("F", "G", "N_eff", "best_start_converged", "all_starts_hit_max_iter")
        }
    download_metadata_available = (source / "data_download_metadata.json").exists()
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": cfg["evaluation"]["status"],
        "sealed_holdout": False,
        "artifact_scope": "allowlist_current_pipeline_only",
        "download_metadata_available": download_metadata_available,
        "calibration_status": calibration,
        "lineage_stages": list(required_stages),
        "config": cfg,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": packages},
        "git_commit": git_commit,
        "code_sha256": code_hash,
        "code_files": code_files,
        "artifacts": copied,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [ok] {destination} ({len(copied)} artefactos con SHA-256)")
    return destination


STEP_LINEAGE = {
    "download": "download",
    "transform": "transform",
    "mm": "mm",
    "benchmarks": "benchmarks",
    "evaluate": "evaluation",
    "portfolio": "portfolio",
    "backtest": "backtest",
    "liquidity": "liquidity_robustness",
    "rolling": "rolling_origin",
}


def _step_is_current(cfg: dict, step: str) -> tuple[bool, list[str]]:
    stage = STEP_LINEAGE.get(step)
    if stage is None:
        return False, ["always_run"]
    result = validate_lineage(_lineage_path(cfg, stage), root=ROOT)
    return bool(result["valid"]), list(result["errors"])


def _execution_sequence(step: str) -> tuple[str, ...]:
    if step == "all":
        return (
            "download",
            "transform",
            "mm",
            "benchmarks",
            "evaluate",
            "portfolio",
            "backtest",
            "liquidity",
            "rolling",
            "snapshot",
        )
    return (step,)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del CLI; devuelve 0 si la etapa solicitada termina bien."""
    parser = argparse.ArgumentParser(description="Pipeline de investigacion MM-BCD IPSA")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "download", "transform", "mm", "benchmarks", "evaluate",
            "reuse-download", "portfolio", "backtest", "liquidity", "rolling", "snapshot", "all",
        ],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="omite etapas cuyo manifiesto, entradas y salidas siguen vigentes",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="muestra el estado de las etapas sin ejecutar ni modificar artefactos",
    )
    args = parser.parse_args(argv)
    cfg = load_config(ROOT / "config.yaml")
    steps = {
        "download": step_download,
        "reuse-download": step_reuse_download,
        "transform": step_transform,
        "mm": step_mm,
        "benchmarks": step_benchmarks,
        "evaluate": step_evaluate,
        "portfolio": step_portfolio,
        "backtest": step_backtest,
        "liquidity": step_liquidity_robustness,
        "rolling": step_rolling_origin,
        "snapshot": step_snapshot,
    }
    sequence = _execution_sequence(args.step)
    if args.plan:
        print("\n[plan] estado de ejecucion")
        for step in sequence:
            current, errors = _step_is_current(cfg, step)
            status = "current" if current else "run"
            detail = "" if current or errors == ["always_run"] else f" ({', '.join(errors)})"
            print(f"  {step:16s} {status}{detail}")
        return 0

    for step in sequence:
        current, _ = _step_is_current(cfg, step)
        if args.resume and current:
            print(f"\n[skip] {step}: linaje y artefactos vigentes")
            continue
        steps[step](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
