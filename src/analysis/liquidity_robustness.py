"""Sensibilidad de liquidez con seleccion congelada exclusivamente in-sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.backtest.walk_forward import (
    moving_block_bootstrap_sharpe_difference,
    performance_metrics,
    simulate_strategy,
    walk_forward_weights,
)
from src.config import objective_weights, target_parameters
from src.evaluation.comparison import compare_focal_model
from src.evaluation.scoring import evaluate_scenarios_detailed
from src.mm.bcd import BCDSolver
from src.mm.objective import MMObjective
from src.mm.targets import _ewma_weights, compute_targets, save_targets
from src.models.benchmarks import generate_benchmarks
from src.portfolio.optimization import (
    equal_weight,
    hierarchical_risk_parity,
    inverse_variance,
    maximum_sharpe,
    minimum_cvar,
    minimum_variance,
    portfolio_diagnostics,
    weighted_mean_cov,
)


def load_liquidity_config(path: str | Path) -> dict[str, Any]:
    """Carga y valida el protocolo; OOS queda prohibido como criterio de seleccion."""
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or not cfg.get("experiment_id"):
        raise ValueError("El experimento requiere experiment_id")
    selection = cfg.get("selection", {})
    if selection.get("sample") != "in_sample_only" or selection.get("uses_oos") is not False:
        raise ValueError("La seleccion de liquidez debe usar exclusivamente in-sample")
    rate = float(selection.get("max_zero_return_rate_is", -1.0))
    run = int(selection.get("max_consecutive_zero_returns_is", -1))
    minimum = int(selection.get("min_assets", 0))
    if not 0.0 <= rate <= 1.0 or run < 0 or minimum < 2:
        raise ValueError("Umbrales de liquidez invalidos")
    if cfg.get("primary_universe_unchanged") is not True:
        raise ValueError("El experimento no puede reemplazar el universo principal")
    return cfg


def select_liquid_universe(
    quality: pd.DataFrame,
    labels: list[str],
    selection_cfg: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Selecciona por tasa y racha de ceros IS, sin consultar campos OOS."""
    required = {"asset", "zero_return_rate_is", "max_zero_run_is"}
    missing_columns = sorted(required.difference(quality.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas de calidad IS: {missing_columns}")
    if quality["asset"].duplicated().any():
        raise ValueError("data_quality_returns contiene activos duplicados")

    indexed = quality.set_index("asset", drop=False)
    missing_assets = [label for label in labels if label not in indexed.index]
    if missing_assets:
        raise ValueError(f"Faltan activos en calidad de retornos: {missing_assets}")

    # Esta proyeccion deliberada impide que una columna OOS participe en la regla.
    selection_inputs = indexed.loc[labels, ["asset", "zero_return_rate_is", "max_zero_run_is"]].copy()
    max_rate = float(selection_cfg["max_zero_return_rate_is"])
    max_run = int(selection_cfg["max_consecutive_zero_returns_is"])
    selection_inputs["passes_zero_rate_is"] = (
        selection_inputs["zero_return_rate_is"].astype(float) <= max_rate
    )
    selection_inputs["passes_zero_run_is"] = (
        selection_inputs["max_zero_run_is"].astype(int) <= max_run
    )
    selection_inputs["selected"] = (
        selection_inputs["passes_zero_rate_is"] & selection_inputs["passes_zero_run_is"]
    )

    def reason(row: pd.Series) -> str:
        failed: list[str] = []
        if not bool(row["passes_zero_rate_is"]):
            failed.append("zero_return_rate_is")
        if not bool(row["passes_zero_run_is"]):
            failed.append("max_zero_run_is")
        return "included" if not failed else "excluded:" + "+".join(failed)

    selection_inputs["decision_reason"] = selection_inputs.apply(reason, axis=1)
    selection_inputs["selection_sample"] = "in_sample_only"
    selection_inputs["selection_uses_oos"] = False

    # Las metricas OOS se adjuntan solo como diagnostico ex post, despues de decidir.
    diagnostics = [column for column in ("zero_return_rate_oos", "max_zero_run_oos") if column in indexed]
    if diagnostics:
        selection_inputs = selection_inputs.join(indexed.loc[labels, diagnostics])
    selection_inputs.reset_index(drop=True, inplace=True)
    selected = selection_inputs.loc[selection_inputs["selected"], "asset"].tolist()
    minimum = int(selection_cfg["min_assets"])
    if len(selected) < minimum:
        raise RuntimeError(
            f"El filtro deja {len(selected)} activos, menos que min_assets={minimum}"
        )
    return selected, selection_inputs


def _read_returns(path: Path, labels: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.rename(columns={column: column.replace(".SN", "") for column in frame.columns}, inplace=True)
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"Faltan activos en {path.name}: {missing}")
    result = frame.loc[:, labels].sort_index()
    if result.empty or result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{path.name} no contiene una matriz completa y finita")
    return result


def _calibration_payload(
    solver: BCDSolver,
    objective: MMObjective,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    best = next(
        record for record in solver.all_starts if record["start_id"] == solver.best_start_id
    )
    return {
        "F": objective.evaluate(scenarios, probabilities),
        "G": solver.regularized_objective(scenarios, probabilities),
        "KL_uniform": solver._kl_uniform(probabilities),
        "N_eff": float(1.0 / np.sum(probabilities**2)),
        "stationarity": solver.stationarity_metrics(scenarios, probabilities),
        "best_start_converged": bool(best["converged"]),
        "best_start_stationary": bool(best["stationarity_pass"]),
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


def _portfolio_weights(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    target_covariance: np.ndarray,
    main_cfg: dict[str, Any],
    n_assets: int,
) -> dict[str, np.ndarray]:
    portfolio_cfg = main_cfg["portfolio"]
    max_weight = float(portfolio_cfg["max_weight"])
    l2_penalty = float(portfolio_cfg.get("l2_penalty", 0.0))
    alpha = float(portfolio_cfg["alpha_cvar"])
    weights: dict[str, np.ndarray] = {
        "EqualWeight": equal_weight(n_assets),
        "InverseVariance": inverse_variance(target_covariance),
        "HRP": hierarchical_risk_parity(target_covariance),
    }
    for model, (scenarios, probabilities) in models.items():
        mean, covariance = weighted_mean_cov(scenarios, probabilities)
        weights[f"{model}_MinVariance"] = minimum_variance(
            covariance, max_weight, l2_penalty
        )
        weights[f"{model}_MinCVaR"] = minimum_cvar(
            scenarios, probabilities, alpha, max_weight
        )
        weights[f"{model}_MaxSharpe"] = maximum_sharpe(
            mean,
            covariance,
            max_weight=max_weight,
            risk_free_rate=float(portfolio_cfg["rf"]),
            l2_penalty=l2_penalty,
            seed=int(main_cfg["evaluation"]["seed"]),
        )
    return weights


def _score_comparison(
    full_path: Path,
    liquid: pd.DataFrame,
    primary_score: str,
) -> pd.DataFrame:
    full = pd.read_csv(full_path)
    if primary_score not in full or primary_score not in liquid:
        raise ValueError(f"Score primario inexistente: {primary_score}")
    full = full.copy()
    liquid = liquid.copy()
    full["full_rank"] = full[primary_score].rank(method="min")
    liquid["liquid_rank"] = liquid[primary_score].rank(method="min")
    keep_full = full[["model", primary_score, "full_rank"]].rename(
        columns={primary_score: f"full_{primary_score}"}
    )
    keep_liquid = liquid[["model", primary_score, "liquid_rank"]].rename(
        columns={primary_score: f"liquid_{primary_score}"}
    )
    return keep_full.merge(keep_liquid, on="model", how="outer").sort_values("liquid_rank")


def _run_backtest(
    main_cfg: dict[str, Any],
    labels: list[str],
    daily_is: pd.DataFrame,
    daily_oos: pd.DataFrame,
    static_weights: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    full = pd.concat([daily_is, daily_oos]).sort_index()
    start_oos = pd.Timestamp(main_cfg["data"]["start_oos"])
    schedules = {
        name: {daily_oos.index[0]: weights} for name, weights in static_weights.items()
    }
    designs = {name: "static_single_shot_oos" for name in schedules}

    estimators = {
        "WF_EqualWeight": lambda training: equal_weight(training.shape[1]),
        "WF_InverseVariance": lambda training: inverse_variance(
            training.cov(ddof=0).to_numpy()
        ),
        "WF_HRP": lambda training: hierarchical_risk_parity(
            training.cov(ddof=0).to_numpy()
        ),
    }
    for name, estimator in estimators.items():
        schedules[name] = walk_forward_weights(
            full, start_oos, estimator, frequency="Q", min_history=252
        )
        designs[name] = "expanding_window_walk_forward"

    cost_bps = float(main_cfg["portfolio"].get("transaction_cost_bps", 0.0))
    rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    executions: dict[str, pd.DataFrame] = {}
    wealth: dict[str, pd.Series] = {}
    execution_paths: list[Path] = []
    for name, schedule in schedules.items():
        wealth_series, execution, execution_info = simulate_strategy(
            daily_oos, schedule, cost_bps
        )
        rows.append(
            {
                "portfolio": name,
                "evaluation_design": designs[name],
                **performance_metrics(wealth_series, execution),
                **execution_info,
                "transaction_cost_bps": cost_bps,
            }
        )
        executions[name] = execution
        wealth[name] = wealth_series
        for date, weights in schedule.items():
            for asset, weight in zip(labels, weights):
                schedule_rows.append(
                    {
                        "portfolio": name,
                        "rebalance_date": date,
                        "asset": asset,
                        "weight": float(weight),
                    }
                )
        execution_path = output_dir / f"execution_{name}.csv"
        execution.to_csv(execution_path)
        execution_paths.append(execution_path)

    benchmark_name = "WF_EqualWeight"
    benchmark_returns = executions[benchmark_name]["net_return"].to_numpy()
    bootstrap_rows = []
    for name, execution in executions.items():
        interval = moving_block_bootstrap_sharpe_difference(
            execution["net_return"].to_numpy(),
            benchmark_returns,
            block_size=int(main_cfg["evaluation"]["bootstrap_block_size"]),
            samples=int(main_cfg["evaluation"]["bootstrap_samples"]),
            seed=int(main_cfg["evaluation"]["seed"]),
        )
        bootstrap_rows.append(
            {
                "portfolio": name,
                "evaluation_design": designs[name],
                "benchmark": benchmark_name,
                **interval,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(output_dir / "backtest_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_sharpe_differences.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(output_dir / "walk_forward_weights.csv", index=False)
    pd.DataFrame(wealth).to_csv(output_dir / "backtest_wealth.csv")
    return metrics, bootstrap, execution_paths


def run_liquidity_robustness(
    main_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Recalibra y reevalua todos los modelos en el universo liquido IS."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(main_cfg["paths"]["data_raw"])
    tables = Path(main_cfg["paths"]["tables"])
    H = int(main_cfg["data"]["H"])

    quality = pd.read_csv(source / "data_quality_returns.csv")
    labels, selection = select_liquid_universe(
        quality, list(main_cfg["asset_labels"]), experiment_cfg["selection"]
    )
    selection.to_csv(output / "universe_selection.csv", index=False)

    terminal_is = _read_returns(source / f"hist_terminal_returns_H{H}.csv", labels)
    daily_is = _read_returns(source / "daily_returns.csv", labels)
    terminal_oos = _read_returns(
        source / f"terminal_returns_H{H}_OOS_nonoverlap.csv", labels
    )
    daily_oos = _read_returns(source / "daily_returns_OOS.csv", labels)

    parameters = target_parameters(main_cfg["mm"])
    moments, covariance, _ = compute_targets(
        terminal_is, daily_is, **parameters
    )
    save_targets(moments, covariance, labels, output)

    mm_cfg = main_cfg["mm"]
    objective = MMObjective(
        moments,
        covariance,
        objective_weights(mm_cfg),
        int(mm_cfg["N_scenarios"]),
    )
    solver = BCDSolver(
        objective, mm_cfg, warm_noise=float(mm_cfg.get("warm_noise", 0.15))
    )
    mm_scenarios, mm_probabilities = solver.solve(out_path=output)
    np.save(output / "mm_scenarios_x.npy", mm_scenarios)
    np.save(output / "mm_probabilities_p.npy", mm_probabilities)
    solver.summary_table().to_csv(output / "bcd_starts.csv", index=False)
    calibration = _calibration_payload(
        solver, objective, mm_scenarios, mm_probabilities
    )
    (output / "mm_calibration_metrics.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    history_weights = _ewma_weights(len(terminal_is), parameters["decay_lambda"])
    benchmark_cfg = main_cfg["benchmarks"]
    benchmarks = generate_benchmarks(
        moments,
        covariance,
        terminal_is.to_numpy(),
        history_weights,
        n_scenarios=int(benchmark_cfg["N_scenarios"]),
        seed=int(benchmark_cfg["seed"]),
        student_t_df=float(benchmark_cfg["student_t_df"]),
    )
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "MM": (mm_scenarios, mm_probabilities),
        **benchmarks,
    }
    for name, (scenarios, probabilities) in benchmarks.items():
        np.save(output / f"{name}_scenarios.npy", scenarios)
        np.save(output / f"{name}_probabilities.npy", probabilities)

    marginal_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []
    for index, (name, (scenarios, probabilities)) in enumerate(models.items()):
        marginal, aggregate, by_observation = evaluate_scenarios_detailed(
            name,
            scenarios,
            probabilities,
            terminal_oos.to_numpy(),
            labels,
            observation_ids=terminal_oos.index,
            alpha=float(main_cfg["portfolio"]["alpha_cvar"]),
            seed=int(main_cfg["evaluation"]["seed"]) + index * 10_000,
            energy_pair_samples=int(main_cfg["evaluation"]["energy_pair_samples"]),
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
        benchmark_models=list(main_cfg["benchmarks"]["include"]),
        block_size=int(main_cfg["evaluation"]["score_bootstrap_block_size"]),
        samples=int(main_cfg["evaluation"]["score_bootstrap_samples"]),
        confidence_level=float(main_cfg["evaluation"]["score_bootstrap_confidence"]),
        seed=int(main_cfg["evaluation"]["seed"]),
        inference_status=str(main_cfg["evaluation"]["status"]),
    )
    marginal_all.to_csv(output / "probabilistic_scores_by_asset.csv", index=False)
    aggregate_all.to_csv(output / "probabilistic_scores_summary.csv", index=False)
    observation_all.to_csv(output / "probabilistic_scores_by_observation.csv", index=False)
    score_differences.to_csv(output / "probabilistic_score_differences.csv", index=False)

    primary_score = str(experiment_cfg["comparison"]["primary_score"])
    score_comparison = _score_comparison(
        tables / "probabilistic_scores_summary.csv", aggregate_all, primary_score
    )
    score_comparison.to_csv(output / "probabilistic_rank_comparison.csv", index=False)

    weights_by_name = _portfolio_weights(
        models, covariance, main_cfg, len(labels)
    )
    weight_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for name, weights in weights_by_name.items():
        for asset, weight in zip(labels, weights):
            weight_rows.append(
                {"portfolio": name, "asset": asset, "weight": float(weight)}
            )
        diagnostic_rows.append(
            {
                "portfolio": name,
                **portfolio_diagnostics(
                    weights, mm_scenarios, mm_probabilities,
                    float(main_cfg["portfolio"]["alpha_cvar"]),
                ),
            }
        )
    pd.DataFrame(weight_rows).to_csv(output / "portfolio_weights.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "portfolio_metrics_in_sample.csv", index=False
    )

    backtest_metrics, _, _ = _run_backtest(
        main_cfg, labels, daily_is, daily_oos, weights_by_name, output
    )
    full_backtest = pd.read_csv(tables / "backtest_metrics.csv")
    comparable_columns = [
        "portfolio", "evaluation_design", "cagr", "annual_volatility", "sharpe",
        "max_drawdown", "final_wealth", "total_turnover", "total_cost_fraction",
    ]
    portfolio_comparison = full_backtest[comparable_columns].merge(
        backtest_metrics[comparable_columns],
        on=["portfolio", "evaluation_design"],
        how="outer",
        suffixes=("_full", "_liquid"),
    )
    portfolio_comparison["delta_sharpe_liquid_minus_full"] = (
        portfolio_comparison["sharpe_liquid"] - portfolio_comparison["sharpe_full"]
    )
    portfolio_comparison.to_csv(
        output / "portfolio_comparison_full_vs_liquid.csv", index=False
    )

    excluded = selection.loc[~selection["selected"], "asset"].tolist()
    metadata = {
        "experiment_id": experiment_cfg["experiment_id"],
        "research_status": experiment_cfg["research_status"],
        "selection_sample": experiment_cfg["selection"]["sample"],
        "selection_uses_oos": False,
        "primary_universe_unchanged": True,
        "selection_thresholds": {
            "max_zero_return_rate_is": float(
                experiment_cfg["selection"]["max_zero_return_rate_is"]
            ),
            "max_consecutive_zero_returns_is": int(
                experiment_cfg["selection"]["max_consecutive_zero_returns_is"]
            ),
            "min_assets": int(experiment_cfg["selection"]["min_assets"]),
        },
        "selected_assets": labels,
        "excluded_assets": excluded,
        "selected_count": len(labels),
        "full_count": len(main_cfg["asset_labels"]),
        "is_rows_terminal": len(terminal_is),
        "oos_rows_terminal_nonoverlap": len(terminal_oos),
        "oos_rows_daily": len(daily_oos),
        "comparison_warning": experiment_cfg["comparison"]["warning"],
        "same_main_hyperparameters": True,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    artifacts = sorted(path for path in output.iterdir() if path.is_file())
    return {
        "artifacts": artifacts,
        "metadata": metadata,
        "scores": aggregate_all,
        "backtest": backtest_metrics,
        "calibration": calibration,
    }
