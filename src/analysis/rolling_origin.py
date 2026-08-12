"""Validacion temporal multi-origen con recalibracion completa por fold."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import objective_weights, target_parameters
from src.data.transform import _rolling_terminal_fast, non_overlapping_windows
from src.evaluation.comparison import compare_focal_model_by_group
from src.evaluation.scoring import evaluate_scenarios_detailed
from src.mm.bcd import BCDSolver
from src.mm.objective import MMObjective
from src.mm.targets import _ewma_weights, compute_targets
from src.models.benchmarks import generate_benchmarks


def load_rolling_origin_config(path: str | Path) -> dict:
    """Carga y valida el protocolo congelado del experimento."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    folds = cfg.get("folds", [])
    if not folds:
        raise ValueError("rolling_origin requiere al menos un fold")
    fold_ids = [str(fold.get("fold_id")) for fold in folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("fold_id debe ser unico")
    previous_evaluation_end: pd.Timestamp | None = None
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        evaluation_start = pd.Timestamp(fold["evaluation_start"])
        evaluation_end = pd.Timestamp(fold["evaluation_end"])
        if not train_end < evaluation_start <= evaluation_end:
            raise ValueError(f"Orden temporal invalido en fold {fold['fold_id']}")
        if previous_evaluation_end is not None and evaluation_start <= previous_evaluation_end:
            raise ValueError("Las ventanas de evaluacion deben ser disjuntas y crecientes")
        previous_evaluation_end = evaluation_end
    validation = cfg.get("validation", {})
    if int(validation.get("minimum_training_daily_rows", 0)) <= 0:
        raise ValueError("minimum_training_daily_rows debe ser positivo")
    if int(validation.get("minimum_evaluation_terminal_rows", 0)) < 2:
        raise ValueError("minimum_evaluation_terminal_rows debe ser al menos dos")
    inference = cfg.get("inference", {})
    if int(inference.get("bootstrap_samples", 0)) < 100:
        raise ValueError("bootstrap_samples debe ser al menos 100")
    if int(inference.get("block_size", 0)) <= 0:
        raise ValueError("block_size debe ser positivo")
    if inference.get("multiple_testing") != "holm":
        raise ValueError("multiple_testing debe ser holm")
    return cfg


def combine_daily_returns(
    daily_is: pd.DataFrame,
    daily_oos: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Reconstruye el panel diario completo sin duplicados ni columnas extra."""
    combined = pd.concat([daily_is[labels], daily_oos[labels]]).sort_index()
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise ValueError("El panel diario combinado debe tener indice unico y creciente")
    if combined.isna().any().any() or not np.isfinite(combined.to_numpy()).all():
        raise ValueError("El panel diario combinado contiene valores no finitos")
    return combined


def build_fold_samples(
    daily: pd.DataFrame,
    fold: dict,
    horizon: int,
    *,
    minimum_training_daily_rows: int,
    minimum_evaluation_terminal_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye train rolling y evaluacion no solapada sin cruzar el origen."""
    train_end = pd.Timestamp(fold["train_end"])
    evaluation_start = pd.Timestamp(fold["evaluation_start"])
    evaluation_end = pd.Timestamp(fold["evaluation_end"])
    training_daily = daily.loc[daily.index <= train_end].copy()
    evaluation_daily = daily.loc[
        (daily.index >= evaluation_start) & (daily.index <= evaluation_end)
    ].copy()
    if len(training_daily) < minimum_training_daily_rows:
        raise ValueError(
            f"Fold {fold['fold_id']} tiene solo {len(training_daily)} filas de entrenamiento"
        )
    if training_daily.index.max() >= evaluation_daily.index.min():
        raise RuntimeError(f"Fold {fold['fold_id']} presenta fuga temporal")
    training_terminal = _rolling_terminal_fast(training_daily, horizon)
    evaluation_terminal = non_overlapping_windows(
        _rolling_terminal_fast(evaluation_daily, horizon), horizon
    )
    if len(evaluation_terminal) < minimum_evaluation_terminal_rows:
        raise ValueError(
            f"Fold {fold['fold_id']} tiene solo {len(evaluation_terminal)} ventanas OOS"
        )
    return training_daily, training_terminal, evaluation_terminal


def _calibrate_mm(
    main_cfg: dict,
    moments: np.ndarray,
    covariance: np.ndarray,
    fold_index: int,
    fold_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mm_cfg = deepcopy(main_cfg["mm"])
    mm_cfg["seed"] = int(mm_cfg["seed"]) + fold_index * 1_000
    objective = MMObjective(
        moments,
        covariance,
        objective_weights(mm_cfg),
        int(mm_cfg["N_scenarios"]),
    )
    solver = BCDSolver(
        objective,
        mm_cfg,
        warm_noise=float(mm_cfg.get("warm_noise", 0.15)),
    )
    scenarios, probabilities = solver.solve(out_path=fold_dir)
    selected = next(
        record for record in solver.all_starts if record["start_id"] == solver.best_start_id
    )
    stationarity = solver.stationarity_metrics(scenarios, probabilities)
    calibration: dict[str, object] = {
        "seed": mm_cfg["seed"],
        "F": objective.evaluate(scenarios, probabilities),
        "G": solver.regularized_objective(scenarios, probabilities),
        "N_eff": float(1.0 / np.sum(probabilities**2)),
        "selected_start_id": solver.best_start_id,
        "best_start_converged": bool(selected["converged"]),
        "best_start_stationary": bool(selected["stationarity_pass"]),
        "stationarity": stationarity,
    }
    np.save(fold_dir / "mm_scenarios_x.npy", scenarios)
    np.save(fold_dir / "mm_probabilities_p.npy", probabilities)
    (fold_dir / "mm_calibration_metrics.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    solver.summary_table().to_csv(fold_dir / "bcd_starts.csv", index=False)
    return scenarios, probabilities, calibration


def _stability_summary(fold_scores: pd.DataFrame) -> pd.DataFrame:
    metric_rows: list[pd.DataFrame] = []
    for metric in ("mean_crps", "energy_score", "variogram_score"):
        ranked = fold_scores[["fold_id", "model", metric]].copy()
        ranked["rank"] = ranked.groupby("fold_id")[metric].rank(
            method="min", ascending=True
        )
        summary = ranked.groupby("model").agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            fold_wins=("rank", lambda values: int(np.sum(values == 1.0))),
            folds=("rank", "size"),
        ).reset_index()
        summary.insert(0, "metric", metric)
        metric_rows.append(summary)
    return pd.concat(metric_rows, ignore_index=True)


def run_rolling_origin(
    main_cfg: dict,
    experiment_cfg: dict,
    daily_is: pd.DataFrame,
    daily_oos: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, object]:
    """Recalibra todos los modelos en cada origen y agrega evidencia pareada."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(main_cfg["asset_labels"])
    daily = combine_daily_returns(daily_is, daily_oos, labels)
    horizon = int(main_cfg["data"]["H"])
    validation_cfg = experiment_cfg["validation"]
    inference_cfg = experiment_cfg["inference"]
    target_cfg = target_parameters(main_cfg["mm"])
    benchmark_cfg = main_cfg["benchmarks"]

    definition_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    fold_score_frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(experiment_cfg["folds"]):
        fold_id = str(fold["fold_id"])
        fold_dir = output / "folds" / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_daily, training_terminal, evaluation_terminal = build_fold_samples(
            daily,
            fold,
            horizon,
            minimum_training_daily_rows=int(
                validation_cfg["minimum_training_daily_rows"]
            ),
            minimum_evaluation_terminal_rows=int(
                validation_cfg["minimum_evaluation_terminal_rows"]
            ),
        )
        moments, covariance, _ = compute_targets(
            training_terminal,
            training_daily,
            **target_cfg,
        )
        mm_scenarios, mm_probabilities, calibration = _calibrate_mm(
            main_cfg,
            moments,
            covariance,
            fold_index,
            fold_dir,
        )
        historical_weights = _ewma_weights(
            len(training_terminal), target_cfg["decay_lambda"]
        )
        models = {
            "MM": (mm_scenarios, mm_probabilities),
            **generate_benchmarks(
                moments,
                covariance,
                training_terminal.to_numpy(),
                historical_weights,
                n_scenarios=int(benchmark_cfg["N_scenarios"]),
                seed=int(benchmark_cfg["seed"]) + fold_index * 1_000,
                student_t_df=float(benchmark_cfg["student_t_df"]),
            ),
        }

        per_model: list[pd.DataFrame] = []
        for model_index, (model_name, (scenarios, probabilities)) in enumerate(models.items()):
            _, aggregate, by_observation = evaluate_scenarios_detailed(
                model_name,
                scenarios,
                probabilities,
                evaluation_terminal.to_numpy(),
                labels,
                observation_ids=evaluation_terminal.index,
                alpha=float(main_cfg["portfolio"]["alpha_cvar"]),
                seed=int(main_cfg["evaluation"]["seed"])
                + fold_index * 100_000
                + model_index * 10_000,
                energy_pair_samples=int(main_cfg["evaluation"]["energy_pair_samples"]),
            )
            aggregate.insert(0, "fold_id", fold_id)
            aggregate["train_end"] = str(pd.Timestamp(fold["train_end"]).date())
            aggregate["evaluation_start"] = str(
                pd.Timestamp(fold["evaluation_start"]).date()
            )
            aggregate["evaluation_end"] = str(
                pd.Timestamp(fold["evaluation_end"]).date()
            )
            aggregate["n_observations"] = len(evaluation_terminal)
            per_model.append(aggregate)
            by_observation.insert(0, "fold_id", fold_id)
            observation_frames.append(by_observation)

        fold_scores = pd.concat(per_model, ignore_index=True)
        fold_scores.to_csv(fold_dir / "probabilistic_scores_summary.csv", index=False)
        fold_score_frames.append(fold_scores)
        calibration_rows.append(
            {
                "fold_id": fold_id,
                "train_end": str(pd.Timestamp(fold["train_end"]).date()),
                "seed": calibration["seed"],
                "F": calibration["F"],
                "G": calibration["G"],
                "N_eff": calibration["N_eff"],
                "selected_start_id": calibration["selected_start_id"],
                "best_start_converged": calibration["best_start_converged"],
                "best_start_stationary": calibration["best_start_stationary"],
                "x_gradient_inf": calibration["stationarity"]["x_gradient_inf"],
                "p_tangent_gradient_inf": calibration["stationarity"][
                    "p_tangent_gradient_inf"
                ],
            }
        )
        definition_rows.append(
            {
                "fold_id": fold_id,
                "train_start_actual": str(training_daily.index.min().date()),
                "train_end_configured": str(pd.Timestamp(fold["train_end"]).date()),
                "train_end_actual": str(training_daily.index.max().date()),
                "evaluation_start_configured": str(
                    pd.Timestamp(fold["evaluation_start"]).date()
                ),
                "evaluation_start_actual": str(evaluation_terminal.index.min().date()),
                "evaluation_end_configured": str(
                    pd.Timestamp(fold["evaluation_end"]).date()
                ),
                "evaluation_end_actual": str(evaluation_terminal.index.max().date()),
                "training_daily_rows": len(training_daily),
                "training_terminal_rows": len(training_terminal),
                "evaluation_terminal_rows": len(evaluation_terminal),
                "temporal_leakage": False,
            }
        )

    definitions = pd.DataFrame(definition_rows)
    calibrations = pd.DataFrame(calibration_rows)
    fold_scores_all = pd.concat(fold_score_frames, ignore_index=True)
    observations_all = pd.concat(observation_frames, ignore_index=True)
    pooled_scores = (
        observations_all.groupby("model")[[
            "mean_crps", "energy_score", "variogram_score"
        ]]
        .mean()
        .reset_index()
    )
    pooled_scores["n_observations"] = observations_all.groupby("model").size().to_numpy()
    pooled_scores = pooled_scores.sort_values("mean_crps")
    differences = compare_focal_model_by_group(
        observations_all,
        group_column="fold_id",
        focal_model="MM",
        benchmark_models=list(main_cfg["benchmarks"]["include"]),
        block_size=int(inference_cfg["block_size"]),
        samples=int(inference_cfg["bootstrap_samples"]),
        confidence_level=float(inference_cfg["confidence_level"]),
        seed=int(main_cfg["evaluation"]["seed"]) + 900_000,
        inference_status=str(experiment_cfg["experiment"]["status"]),
    )
    stability = _stability_summary(fold_scores_all)

    definitions.to_csv(output / "fold_definitions.csv", index=False)
    calibrations.to_csv(output / "fold_calibration.csv", index=False)
    fold_scores_all.to_csv(output / "probabilistic_scores_by_fold.csv", index=False)
    observations_all.to_csv(output / "probabilistic_scores_by_observation.csv", index=False)
    pooled_scores.to_csv(output / "probabilistic_scores_pooled.csv", index=False)
    differences.to_csv(output / "probabilistic_score_differences_pooled.csv", index=False)
    stability.to_csv(output / "model_stability_summary.csv", index=False)

    metadata = {
        "experiment_id": experiment_cfg["experiment"]["id"],
        "status": experiment_cfg["experiment"]["status"],
        "window_type": experiment_cfg["experiment"]["window_type"],
        "refit_all_models_each_fold": True,
        "temporal_leakage_detected": False,
        "horizon": horizon,
        "fold_count": len(definitions),
        "total_evaluation_windows": int(definitions["evaluation_terminal_rows"].sum()),
        "models": ["MM", *main_cfg["benchmarks"]["include"]],
        "inference": inference_cfg,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    return {
        "artifacts": artifacts,
        "metadata": metadata,
        "fold_scores": fold_scores_all,
        "pooled_scores": pooled_scores,
        "differences": differences,
        "stability": stability,
    }
