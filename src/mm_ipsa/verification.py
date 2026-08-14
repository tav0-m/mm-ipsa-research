"""Verificacion dura de contratos; cualquier fallo critico produce exit code 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mm_ipsa.analysis.liquidity_robustness import load_liquidity_config, select_liquid_universe
from mm_ipsa.analysis.rolling_origin import load_rolling_origin_config
from mm_ipsa.config import load_config, objective_weights, target_parameters
from mm_ipsa.lineage import validate_lineage
from mm_ipsa.mm.objective import MMObjective
from mm_ipsa.mm.targets import compute_targets


def sha256(path: Path) -> str:
    """Digest SHA-256 leyendo por bloques para no cargar el archivo completo."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_sizes_are_valid(
    differences: pd.DataFrame, cfg: dict, n_observations: int
) -> bool:
    """Comprueba que el ancho de bloque respete el modo declarado.

    Con seleccion automatica el ancho varia por contraste, de modo que exigir
    un valor unico ya no describe el contrato. Lo que debe cumplirse es que el
    ancho sea utilizable, que el modo registrado coincida con la configuracion
    y que el diagnostico de Politis-White acompane a cada fila cuando el modo
    es automatico.
    """
    sizes = differences["block_size"].astype(int)
    if not bool(((sizes >= 1) & (sizes <= n_observations)).all()):
        return False
    mode = str(cfg["evaluation"].get("score_bootstrap_block_size_mode", "auto"))
    if mode == "fixed":
        expected = int(cfg["evaluation"]["score_bootstrap_block_size"])
        return bool((sizes == expected).all())
    if "block_size_mode_auto" not in differences.columns:
        return False
    if not bool((differences["block_size_mode_auto"] == 1.0).all()):
        return False
    return "block_size_politis_white" in differences.columns


def _rolling_block_sizes_are_valid(
    differences: pd.DataFrame, inference_cfg: dict, smallest_fold: int
) -> bool:
    """Igual que el contrato del split unico, pero acotado por el fold mas corto.

    Ningun bloque puede exceder el largo del fold menor, porque los bloques no
    cruzan fronteras temporales y no habria de donde tomarlos.
    """
    sizes = differences["block_size"].astype(int)
    if not bool(((sizes >= 1) & (sizes <= smallest_fold)).all()):
        return False
    mode = str(inference_cfg.get("block_size_mode", "auto"))
    if mode == "fixed":
        return bool((sizes == int(inference_cfg["block_size"])).all())
    return bool(
        "block_size_mode_auto" in differences.columns
        and (differences["block_size_mode_auto"] == 1.0).all()
    )


class Verifier:
    """Acumula fallos de contrato e imprime cada comprobacion a medida que ocurre."""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        symbol = "OK" if condition else "FAIL"
        print(f"  [{symbol}] {label}" + (f": {detail}" if detail else ""))
        if not condition:
            self.failures.append(label + (f": {detail}" if detail else ""))


def main(argv: Sequence[str] | None = None) -> int:
    """Verifica artefactos y contratos; devuelve 1 ante cualquier fallo critico."""
    parser = argparse.ArgumentParser(description="Verifica contratos y artefactos MM-IPSA")
    parser.add_argument("--scope", choices=["core", "full"], default="full")
    args = parser.parse_args(argv)
    cfg = load_config("config.yaml")
    out = Path(cfg["paths"]["data_raw"])
    tables = Path(cfg["paths"]["tables"])
    labels = cfg["asset_labels"]
    H = int(cfg["data"]["H"])
    verifier = Verifier()

    print("\n[verify] datos y split")
    required_data = [
        out / "adj_close_prices_raw.csv",
        out / "price_observation_mask.csv",
        out / "data_quality_prices.csv",
        out / "data_download_metadata.json",
        out / "adj_close_prices.csv",
        out / "returns_metadata.json",
        out / "data_quality_returns.csv",
        out / "daily_returns.csv",
        out / f"hist_terminal_returns_H{H}.csv",
        out / "daily_returns_OOS.csv",
        out / f"terminal_returns_H{H}_OOS_nonoverlap.csv",
    ]
    for path in required_data:
        verifier.check(path.exists(), path.name)
    if verifier.failures:
        print("\n[verify] no es posible continuar sin regenerar datos")
        return 1

    download_metadata = json.loads((out / "data_download_metadata.json").read_text(encoding="utf-8"))
    price_quality = pd.read_csv(out / "data_quality_prices.csv")
    observation_mask = pd.read_csv(out / "price_observation_mask.csv", index_col=0)
    verifier.check(
        download_metadata.get("requested_tickers") == list(cfg["assets"]),
        "universo descargado corresponde a config",
    )
    verifier.check(
        download_metadata.get("end_exclusive") == cfg["data"]["end_oos"],
        "fin exclusivo de descarga corresponde a config",
    )
    verifier.check(
        len(price_quality) == len(labels)
        and set(price_quality["asset"]) == set(labels)
        and (price_quality["coverage_raw"] >= float(cfg["data"]["min_coverage"])).all(),
        "cobertura raw por activo",
    )
    verifier.check(
        list(observation_mask.columns) == labels,
        "mascara de observacion conserva orden de activos",
    )
    internal_hashes = download_metadata.get("sha256", {})
    verifier.check(
        bool(internal_hashes)
        and all(
            (out / name).is_file() and sha256(out / name) == expected
            for name, expected in internal_hashes.items()
        ),
        "hashes internos de descarga",
    )

    daily = pd.read_csv(out / "daily_returns.csv", index_col=0, parse_dates=True)[labels]
    terminal = pd.read_csv(out / f"hist_terminal_returns_H{H}.csv", index_col=0, parse_dates=True)[labels]
    oos = pd.read_csv(out / f"terminal_returns_H{H}_OOS_nonoverlap.csv", index_col=0, parse_dates=True)[labels]
    verifier.check(daily.index.max() <= pd.Timestamp(cfg["data"]["end_train"]), "IS respeta end_train")
    verifier.check(oos.index.min() >= pd.Timestamp(cfg["data"]["start_oos"]), "OOS respeta start_oos")
    verifier.check(daily.index.max() < oos.index.min(), "IS/OOS disjuntos")
    verifier.check(not daily.isna().any().any() and not oos.isna().any().any(), "retornos completos")
    returns_metadata = json.loads((out / "returns_metadata.json").read_text(encoding="utf-8"))
    verifier.check(
        returns_metadata.get("observation_mask_status") == "available",
        "transformacion usa mascara raw",
    )
    verifier.check(
        returns_metadata.get("exclude_imputed_return_endpoints") is True,
        "endpoints imputados se excluyen",
        f"filas={returns_metadata.get('rows_dropped_for_imputed_endpoints')}",
    )

    print("\n[verify] MM y gradientes")
    mm_paths = [out / "mm_scenarios_x.npy", out / "mm_probabilities_p.npy"]
    for path in mm_paths:
        verifier.check(path.exists(), path.name)
    if all(path.exists() for path in mm_paths):
        x = np.load(mm_paths[0])
        p = np.load(mm_paths[1])
        parameters = target_parameters(cfg["mm"])
        moments, covariance, _ = compute_targets(terminal, daily, **parameters)
        objective = MMObjective(moments, covariance, objective_weights(cfg["mm"]), int(cfg["mm"]["N_scenarios"]))
        verifier.check(x.shape == (int(cfg["mm"]["N_scenarios"]), len(labels)), "shape escenarios", str(x.shape))
        verifier.check(p.shape == (len(x),), "shape probabilidades", str(p.shape))
        verifier.check(np.isfinite(x).all() and np.isfinite(p).all(), "escenarios finitos")
        verifier.check(bool(np.all(p >= 0) and np.isclose(p.sum(), 1.0)), "simplex de probabilidades")
        if x.shape == (objective.N, objective.n) and p.shape == (objective.N,):
            rng = np.random.default_rng(99)
            direction = rng.normal(size=x.shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-6
            numerical = (objective.evaluate(x + epsilon * direction, p) - objective.evaluate(x - epsilon * direction, p)) / (2 * epsilon)
            analytic = float(np.sum(objective.grad_x(x, p) * direction))
            relative = abs(numerical - analytic) / max(1.0, abs(numerical), abs(analytic))
            verifier.check(relative < 1e-5, "gradiente x por diferencia finita", f"error={relative:.2e}")
            stored_targets = pd.read_csv(out / "targets_moments.csv", index_col=0).loc[labels].to_numpy().T
            verifier.check(np.allclose(stored_targets, moments, rtol=1e-8, atol=1e-12), "targets corresponden a config actual")
        history_path = out / "regularized_objective_history.csv"
        verifier.check(history_path.exists(), history_path.name)
        if history_path.exists():
            history = pd.read_csv(history_path)
            objective_columns = [column for column in history.columns if column.startswith("start_")]
            monotone = bool(objective_columns) and all(
                np.all(np.diff(history[column].dropna().to_numpy()) <= 1e-9)
                for column in objective_columns
            )
            verifier.check(monotone, "G no creciente en todos los starts")
        calibration_path = out / "mm_calibration_metrics.json"
        verifier.check(calibration_path.exists(), calibration_path.name)
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            verifier.check(
                calibration.get("best_start_converged") is True,
                "mejor start alcanza criterio de convergencia",
            )
            stationarity = calibration.get("stationarity", {})
            x_residual = float(stationarity.get("x_gradient_inf", np.inf))
            p_residual = float(stationarity.get("p_tangent_gradient_inf", np.inf))
            verifier.check(
                np.isfinite(x_residual) and x_residual <= float(cfg["mm"]["x_stationarity_tol"]),
                "residuo de estacionariedad X",
                f"{x_residual:.2e}",
            )
            verifier.check(
                np.isfinite(p_residual) and p_residual <= float(cfg["mm"]["p_stationarity_tol"]),
                "residuo KKT de probabilidades",
                f"{p_residual:.2e}",
            )

    if args.scope == "full":
        print("\n[verify] evaluacion, portafolios y robustez temporal/liquidez")
        for path in (
            tables / "probabilistic_scores_summary.csv",
            tables / "probabilistic_scores_by_observation.csv",
            tables / "probabilistic_score_differences.csv",
            tables / "portfolio_weights_research.csv",
            tables / "backtest_metrics.csv",
            tables / "bootstrap_sharpe_differences.csv",
        ):
            verifier.check(path.exists(), path.name)
        robustness = out / "robustness" / "liquidity"
        robustness_paths = (
            robustness / "universe_selection.csv",
            robustness / "experiment_metadata.json",
            robustness / "mm_scenarios_x.npy",
            robustness / "mm_probabilities_p.npy",
            robustness / "mm_calibration_metrics.json",
            robustness / "probabilistic_scores_summary.csv",
            robustness / "probabilistic_scores_by_observation.csv",
            robustness / "probabilistic_score_differences.csv",
            robustness / "probabilistic_rank_comparison.csv",
            robustness / "portfolio_weights.csv",
            robustness / "backtest_metrics.csv",
            robustness / "bootstrap_sharpe_differences.csv",
            robustness / "portfolio_comparison_full_vs_liquid.csv",
        )
        for path in robustness_paths:
            verifier.check(path.exists(), f"liquidez: {path.name}")

        rolling = out / "robustness" / "rolling_origin"
        rolling_paths = (
            rolling / "fold_definitions.csv",
            rolling / "fold_calibration.csv",
            rolling / "probabilistic_scores_by_fold.csv",
            rolling / "probabilistic_scores_by_observation.csv",
            rolling / "probabilistic_scores_pooled.csv",
            rolling / "probabilistic_score_differences_pooled.csv",
            rolling / "model_stability_summary.csv",
            rolling / "experiment_metadata.json",
        )
        for path in rolling_paths:
            verifier.check(path.exists(), f"rolling-origin: {path.name}")

        score_paths = (
            (
                "completo",
                tables / "probabilistic_scores_summary.csv",
                tables / "probabilistic_scores_by_observation.csv",
                tables / "probabilistic_score_differences.csv",
            ),
            (
                "liquido",
                robustness / "probabilistic_scores_summary.csv",
                robustness / "probabilistic_scores_by_observation.csv",
                robustness / "probabilistic_score_differences.csv",
            ),
        )
        for universe, summary_path, observation_path, difference_path in score_paths:
            if not all(path.exists() for path in (summary_path, observation_path, difference_path)):
                continue
            summary_scores = pd.read_csv(summary_path).set_index("model")
            observation_scores = pd.read_csv(observation_path)
            differences = pd.read_csv(difference_path)
            grouped = observation_scores.groupby("model")[
                ["mean_crps", "energy_score", "variogram_score"]
            ].mean()
            verifier.check(
                len(observation_scores)
                == len(summary_scores) * len(oos)
                and not observation_scores.duplicated(["model", "observation"]).any(),
                f"scores pareados completos: {universe}",
            )
            verifier.check(
                all(
                    np.allclose(
                        grouped.loc[summary_scores.index, metric],
                        summary_scores[metric],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    for metric in ("mean_crps", "energy_score", "variogram_score")
                ),
                f"scores por observacion agregan al resumen: {universe}",
            )
            # Tres scoring rules por cada control declarado. Fijar el numero en
            # nueve ataba el contrato a tener exactamente tres controles.
            expected_contrasts = 3 * len(cfg["benchmarks"]["include"])
            verifier.check(
                len(differences) == expected_contrasts
                and int(differences["registered_primary"].sum()) == 1
                and (differences["difference_direction"] == "focal_minus_benchmark").all()
                and (differences["n_observations"] == len(oos)).all(),
                f"contrastes pareados ({expected_contrasts}) y primario unico: {universe}",
            )
            verifier.check(
                bool(differences[["pvalue_raw", "pvalue_holm"]].apply(
                    lambda column: column.between(0.0, 1.0).all()
                ).all()
                and (differences["pvalue_holm"] + 1e-15 >= differences["pvalue_raw"]).all()
                and _block_sizes_are_valid(differences, cfg, len(oos))
                and (
                    differences["bootstrap_samples"]
                    == int(cfg["evaluation"]["score_bootstrap_samples"])
                ).all()),
                f"bootstrap temporal y Holm validos: {universe}",
            )

        if all(path.exists() for path in robustness_paths):
            experiment_cfg = load_liquidity_config("research/liquidity_robustness.yaml")
            metadata = json.loads(
                (robustness / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            stored_selection = pd.read_csv(robustness / "universe_selection.csv")
            expected_assets, expected_selection = select_liquid_universe(
                pd.read_csv(out / "data_quality_returns.csv"),
                labels,
                experiment_cfg["selection"],
            )
            stored_assets = stored_selection.loc[stored_selection["selected"], "asset"].tolist()
            verifier.check(
                metadata.get("selection_uses_oos") is False
                and metadata.get("selection_sample") == "in_sample_only",
                "filtro de liquidez no usa OOS",
            )
            verifier.check(
                metadata.get("primary_universe_unchanged") is True,
                "universo principal permanece inalterado",
            )
            verifier.check(
                stored_assets == expected_assets
                and stored_selection["selected"].tolist()
                == expected_selection["selected"].tolist(),
                "seleccion liquida reproducible desde metricas IS",
                f"seleccionados={len(expected_assets)}",
            )
            liquid_x = np.load(robustness / "mm_scenarios_x.npy")
            liquid_p = np.load(robustness / "mm_probabilities_p.npy")
            verifier.check(
                liquid_x.shape == (int(cfg["mm"]["N_scenarios"]), len(expected_assets))
                and liquid_p.shape == (len(liquid_x),),
                "shape MM del universo liquido",
                f"{liquid_x.shape}",
            )
            verifier.check(
                bool(np.isfinite(liquid_x).all()
                and np.isfinite(liquid_p).all()
                and np.all(liquid_p >= 0)
                and np.isclose(liquid_p.sum(), 1.0)),
                "MM liquido finito y probabilidades en simplex",
            )
            liquid_scores = pd.read_csv(robustness / "probabilistic_scores_summary.csv")
            expected_models = {"MM", *cfg["benchmarks"]["include"]}
            verifier.check(
                set(liquid_scores["model"]) == expected_models,
                "todos los modelos se reevaluan en universo liquido",
            )
            liquid_calibration = json.loads(
                (robustness / "mm_calibration_metrics.json").read_text(encoding="utf-8")
            )
            stationarity = liquid_calibration.get("stationarity", {})
            verifier.check(
                liquid_calibration.get("best_start_converged") is True
                and float(stationarity.get("x_gradient_inf", np.inf))
                <= float(cfg["mm"]["x_stationarity_tol"])
                and float(stationarity.get("p_tangent_gradient_inf", np.inf))
                <= float(cfg["mm"]["p_stationarity_tol"]),
                "convergencia y estacionariedad MM del universo liquido",
            )

        if all(path.exists() for path in rolling_paths):
            rolling_cfg = load_rolling_origin_config("research/rolling_origin.yaml")
            metadata = json.loads(
                (rolling / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            definitions = pd.read_csv(rolling / "fold_definitions.csv")
            calibrations = pd.read_csv(rolling / "fold_calibration.csv")
            fold_scores = pd.read_csv(rolling / "probabilistic_scores_by_fold.csv")
            observation_scores = pd.read_csv(
                rolling / "probabilistic_scores_by_observation.csv"
            )
            pooled_scores = pd.read_csv(rolling / "probabilistic_scores_pooled.csv")
            differences = pd.read_csv(
                rolling / "probabilistic_score_differences_pooled.csv"
            )
            stability = pd.read_csv(rolling / "model_stability_summary.csv")
            expected_fold_ids = [str(fold["fold_id"]) for fold in rolling_cfg["folds"]]
            actual_fold_ids = definitions["fold_id"].astype(str).tolist()
            total_windows = int(definitions["evaluation_terminal_rows"].sum())
            smallest_fold_windows = int(definitions["evaluation_terminal_rows"].min())
            expected_models = {"MM", *cfg["benchmarks"]["include"]}
            verifier.check(
                actual_fold_ids == expected_fold_ids
                and len(definitions) == len(expected_fold_ids)
                and not definitions["temporal_leakage"].astype(bool).any(),
                "folds rolling-origin completos, ordenados y sin fuga",
            )
            configured_temporal_order = all(
                pd.Timestamp(fold["train_end"])
                < pd.Timestamp(fold["evaluation_start"])
                <= pd.Timestamp(fold["evaluation_end"])
                for fold in rolling_cfg["folds"]
            )
            observed_temporal_order = all(
                pd.Timestamp(str(train_end))
                < pd.Timestamp(str(evaluation_start))
                <= pd.Timestamp(str(evaluation_end))
                for train_end, evaluation_start, evaluation_end in zip(
                    definitions["train_end_actual"],
                    definitions["evaluation_start_actual"],
                    definitions["evaluation_end_actual"],
                )
            )
            verifier.check(
                configured_temporal_order and observed_temporal_order,
                "orden temporal rolling-origin verificable",
            )
            verifier.check(
                metadata.get("refit_all_models_each_fold") is True
                and metadata.get("temporal_leakage_detected") is False
                and int(metadata.get("fold_count", 0)) == len(expected_fold_ids)
                and int(metadata.get("total_evaluation_windows", 0)) == total_windows,
                "metadata rolling-origin declara recalibracion y conteos",
            )
            verifier.check(
                len(calibrations) == len(expected_fold_ids)
                and calibrations["best_start_converged"].astype(bool).all()
                and calibrations["best_start_stationary"].astype(bool).all()
                and (
                    calibrations["x_gradient_inf"]
                    <= float(cfg["mm"]["x_stationarity_tol"])
                ).all()
                and (
                    calibrations["p_tangent_gradient_inf"]
                    <= float(cfg["mm"]["p_stationarity_tol"])
                ).all(),
                "MM converge y es estacionario en todos los folds",
            )
            verifier.check(
                set(fold_scores["model"]) == expected_models
                and len(fold_scores) == len(expected_fold_ids) * len(expected_models)
                and not fold_scores.duplicated(["fold_id", "model"]).any(),
                "scores agregados completos por fold y modelo",
            )
            verifier.check(
                len(observation_scores) == total_windows * len(expected_models)
                and not observation_scores.duplicated(
                    ["fold_id", "model", "observation"]
                ).any(),
                "scores rolling-origin pareados por observacion",
            )
            grouped = observation_scores.groupby("model")[[
                "mean_crps", "energy_score", "variogram_score"
            ]].mean()
            pooled = pooled_scores.set_index("model")
            verifier.check(
                set(pooled.index) == expected_models
                and all(
                    np.allclose(
                        grouped.loc[pooled.index, metric],
                        pooled[metric],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    for metric in ("mean_crps", "energy_score", "variogram_score")
                ),
                "scores pooled reproducen perdidas por observacion",
            )
            inference_cfg = rolling_cfg["inference"]
            verifier.check(
                bool(len(differences) == 3 * len(cfg["benchmarks"]["include"])
                and int(differences["registered_primary"].sum()) == 1
                and (differences["n_groups"] == len(expected_fold_ids)).all()
                and (differences["n_observations"] == total_windows).all()
                and (differences["group_column"] == "fold_id").all()
                and _rolling_block_sizes_are_valid(
                    differences, inference_cfg, smallest_fold_windows
                )
                and (
                    differences["bootstrap_samples"]
                    == int(inference_cfg["bootstrap_samples"])
                ).all()
                and differences[["pvalue_raw", "pvalue_holm"]].apply(
                    lambda column: column.between(0.0, 1.0).all()
                ).all()
                and (differences["pvalue_holm"] >= differences["pvalue_raw"]).all()),
                "inferencia pooled respeta folds, bloques y Holm",
            )
            verifier.check(
                len(stability) == 3 * len(expected_models)
                and set(stability["model"]) == expected_models
                and (stability["folds"] == len(expected_fold_ids)).all(),
                "resumen de estabilidad cubre todos los modelos y scores",
            )

        print("\n[verify] linaje y snapshot")
        lineage_root = out / "lineage"
        required_stages = (
            "download", "transform", "mm", "benchmarks", "evaluation", "portfolio",
            "backtest", "liquidity_robustness", "rolling_origin",
        )
        for stage in required_stages:
            lineage_result = validate_lineage(lineage_root / f"{stage}.json", root=Path.cwd())
            verifier.check(
                bool(lineage_result["valid"]),
                f"linaje vigente: {stage}",
                ", ".join(lineage_result["errors"]),
            )
        run_root = Path(cfg["paths"]["runs"])
        manifests = sorted(run_root.glob("*/manifest.json")) if run_root.exists() else []
        verifier.check(bool(manifests), "snapshot con manifest")
        if manifests:
            manifest_path = manifests[-1]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verifier.check(manifest.get("sealed_holdout") is False, "estado holdout declarado")
            verifier.check(
                manifest.get("lineage_stages") == list(required_stages),
                "snapshot declara cadena de linaje completa",
            )
            snapshot = manifest_path.parent
            valid_hashes = all(
                (snapshot / item["path"]).exists() and sha256(snapshot / item["path"]) == item["sha256"]
                for item in manifest.get("artifacts", [])
            )
            verifier.check(valid_hashes, "hashes SHA-256 del snapshot")

    print("\n[verify] resultado")
    if verifier.failures:
        print(f"  FAIL: {len(verifier.failures)} contrato(s) incumplido(s)")
        for failure in verifier.failures:
            print(f"    - {failure}")
        return 1
    print("  OK: todos los contratos solicitados se cumplen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
