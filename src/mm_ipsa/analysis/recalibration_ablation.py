"""Ablacion controlada del efecto de recalibrar en cada origen.

El proyecto observaba que DCC-GARCH pasa de ultimo a primero en CRPS segun se
evalue con un ajuste unico proyectado o recalibrandolo por fold. Ese contraste no
identificaba el efecto: entre ambos disenos cambiaban tambien la muestra evaluada
y parte del calendario de ventanas, de modo que el cambio de ranking podia
provenir de cualquiera de las tres cosas.

Esta ablacion aisla la recalibracion. Ambas variantes se puntuan sobre
exactamente las mismas ventanas de evaluacion:

- ``frozen``: todos los modelos se calibran una sola vez, con los datos previos a
  la primera ventana evaluada, y se proyectan sin actualizar.
- ``refit``: los modelos se recalibran al inicio de cada fold, que es el
  comportamiento del analisis principal.

Como la muestra y el calendario quedan fijos, cualquier diferencia es atribuible
a la recalibracion. En el primer fold ambas variantes coinciden por construccion,
lo que funciona como comprobacion interna del montaje.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mm_ipsa.analysis.rolling_origin import build_fold_samples, combine_daily_returns
from mm_ipsa.config import objective_weights, target_parameters
from mm_ipsa.evaluation.comparison import (
    grouped_moving_block_bootstrap_loss_difference,
    holm_adjust,
    resolve_block_size,
)
from mm_ipsa.evaluation.scoring import evaluate_scenarios_detailed
from mm_ipsa.mm.bcd import BCDSolver
from mm_ipsa.mm.objective import MMObjective
from mm_ipsa.mm.targets import _ewma_weights, compute_targets
from mm_ipsa.models.benchmarks import generate_benchmarks, resolve_student_t_df

METRICS = ("mean_crps", "energy_score", "variogram_score")


def combine_daily_for_ablation(
    daily_is: pd.DataFrame, daily_oos: pd.DataFrame, labels: list[str]
) -> pd.DataFrame:
    """Une los tramos diarios con la misma validacion temporal del rolling-origin."""
    return combine_daily_returns(daily_is, daily_oos, labels)


def _calibrate_origin(
    training_daily: pd.DataFrame,
    training_terminal: pd.DataFrame,
    main_cfg: dict[str, Any],
    seed_offset: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calibra los cinco generadores con la informacion de un origen."""
    mm_cfg = main_cfg["mm"]
    parameters = target_parameters(mm_cfg)
    moments, covariance, _ = compute_targets(
        training_terminal, training_daily, **parameters
    )
    solver = BCDSolver(
        MMObjective(
            moments, covariance, objective_weights(mm_cfg), int(mm_cfg["N_scenarios"])
        ),
        mm_cfg,
        n_workers=1,
    )
    scenarios, probabilities = solver.solve()

    benchmark_cfg = main_cfg["benchmarks"]
    history_weights = _ewma_weights(len(training_terminal), parameters["decay_lambda"])
    degrees, _ = resolve_student_t_df(
        benchmark_cfg,
        training_terminal.to_numpy(),
        moments[0],
        covariance,
        history_weights,
    )
    controls, _ = generate_benchmarks(
        moments,
        covariance,
        training_terminal.to_numpy(),
        history_weights,
        n_scenarios=int(benchmark_cfg["N_scenarios"]),
        seed=int(benchmark_cfg["seed"]) + seed_offset,
        student_t_df=degrees,
        daily_returns=training_daily.to_numpy(),
        horizon=int(main_cfg["data"]["H"]),
        include=list(benchmark_cfg["include"]),
    )
    return {"MM": (scenarios, probabilities), **controls}


def _score_variant(
    variant: str,
    fold_id: str,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    evaluation: pd.DataFrame,
    labels: list[str],
    main_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Puntua todos los modelos de una variante sobre las ventanas de un fold."""
    frames: list[pd.DataFrame] = []
    for index, (name, (scenarios, probabilities)) in enumerate(models.items()):
        _, _, by_observation = evaluate_scenarios_detailed(
            name,
            scenarios,
            probabilities,
            evaluation.to_numpy(),
            labels,
            observation_ids=evaluation.index,
            alpha=float(main_cfg["portfolio"]["alpha_cvar"]),
            seed=seed + index * 1_000,
            energy_pair_samples=int(main_cfg["evaluation"]["energy_pair_samples"]),
        )
        by_observation.insert(0, "variant", variant)
        by_observation.insert(1, "fold_id", fold_id)
        frames.append(by_observation)
    return pd.concat(frames, ignore_index=True)


def run_recalibration_ablation(
    main_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    daily: pd.DataFrame,
    output: str | Path,
) -> dict[str, Any]:
    """Contrasta ``frozen`` y ``refit`` sobre ventanas identicas.

    El calendario de evaluacion, el universo y las semillas de puntuacion son los
    mismos en ambas variantes; solo cambia si los modelos se actualizan al llegar
    a cada origen.
    """
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    labels = list(main_cfg["asset_labels"])
    horizon = int(main_cfg["data"]["H"])
    validation = experiment_cfg["validation"]
    folds = list(experiment_cfg["folds"])
    if len(folds) < 2:
        raise ValueError("La ablacion requiere al menos dos folds")

    observations: list[pd.DataFrame] = []
    frozen_models: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    frozen_origin: str | None = None

    for position, fold in enumerate(folds):
        fold_id = str(fold["fold_id"])
        training_daily, training_terminal, evaluation = build_fold_samples(
            daily,
            fold,
            horizon,
            minimum_training_daily_rows=int(
                validation["minimum_training_daily_rows"]
            ),
            minimum_evaluation_terminal_rows=int(
                validation["minimum_evaluation_terminal_rows"]
            ),
        )
        refit_models = _calibrate_origin(
            training_daily, training_terminal, main_cfg, position * 1_000
        )
        if frozen_models is None:
            # La variante congelada es la calibracion del primer origen, que ya
            # respeta la separacion temporal de todas las ventanas evaluadas.
            frozen_models = refit_models
            frozen_origin = str(training_daily.index.max().date())

        seed = int(main_cfg["evaluation"]["seed"]) + position * 100_000
        observations.append(
            _score_variant(
                "refit", fold_id, refit_models, evaluation, labels, main_cfg, seed
            )
        )
        observations.append(
            _score_variant(
                "frozen", fold_id, frozen_models, evaluation, labels, main_cfg, seed
            )
        )

    scores = pd.concat(observations, ignore_index=True)
    scores.to_csv(target / "ablation_scores_by_observation.csv", index=False)

    pooled = (
        scores.groupby(["variant", "model"])[list(METRICS)].mean().reset_index()
    )
    pooled.to_csv(target / "ablation_scores_pooled.csv", index=False)

    # El primer fold es identico por construccion; incluirlo diluiria el efecto
    # con ceros exactos y dejaria ese tramo sin varianza para el remuestreo.
    first_fold = str(folds[0]["fold_id"])
    comparable = scores.loc[scores["fold_id"] != first_fold]
    differences = _paired_effect(comparable, main_cfg)
    differences.to_csv(target / "ablation_recalibration_effect.csv", index=False)

    identical = _first_fold_is_identical(scores, first_fold)
    metadata = {
        "experiment_id": "recalibration_ablation_v1",
        "status": str(main_cfg["evaluation"]["status"]),
        "frozen_calibration_end": frozen_origin,
        "folds": [str(fold["fold_id"]) for fold in folds],
        "folds_compared": sorted(comparable["fold_id"].unique().tolist()),
        "windows_total": _windows_per_model(scores),
        "windows_compared": _windows_per_model(comparable),
        "first_fold_variants_identical": identical,
        "models": sorted(scores["model"].unique().tolist()),
    }
    (target / "ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"scores": scores, "pooled": pooled, "differences": differences,
            "metadata": metadata}


def _windows_per_model(scores: pd.DataFrame) -> int:
    """Ventanas evaluadas por modelo; identico para todos por construccion."""
    refit = scores.loc[scores["variant"] == "refit"]
    if refit.empty:
        return 0
    first_model = str(refit["model"].iloc[0])
    return int((refit["model"] == first_model).sum())


def _first_fold_is_identical(scores: pd.DataFrame, first_fold: str) -> bool:
    """Comprueba el montaje: en el primer origen ambas variantes coinciden."""
    subset = scores.loc[scores["fold_id"] == first_fold]
    pivot = subset.pivot_table(
        index=["model", "observation"], columns="variant", values="mean_crps"
    )
    if "frozen" not in pivot or "refit" not in pivot:
        return False
    return bool(np.allclose(pivot["frozen"], pivot["refit"], atol=1e-12))


def _paired_effect(scores: pd.DataFrame, main_cfg: dict[str, Any]) -> pd.DataFrame:
    """Diferencia pareada ``frozen - refit`` por modelo y metrica.

    Un valor positivo significa que congelar la calibracion empeora el score, es
    decir que recalibrar aporta.
    """
    inference = main_cfg["evaluation"]
    rows: list[dict[str, object]] = []
    index = 0
    for model in sorted(scores["model"].unique()):
        subset = scores.loc[scores["model"] == model]
        for metric in METRICS:
            pivot = subset.pivot_table(
                index=["fold_id", "observation"], columns="variant", values=metric
            )
            if pivot[["frozen", "refit"]].isna().any().any():
                raise ValueError(f"Ventanas no pareadas para {model}/{metric}")
            groups = pivot.index.get_level_values("fold_id").to_numpy()
            smallest = min(
                int((groups == group).sum())
                for group in dict.fromkeys(groups.tolist())
            )
            frozen = pivot["frozen"].to_numpy()
            refit = pivot["refit"].to_numpy()
            block, diagnostics = resolve_block_size(
                frozen - refit,
                mode=str(inference.get("score_bootstrap_block_size_mode", "auto")),
                configured=int(inference["score_bootstrap_block_size"]),
                max_block=smallest,
                groups=groups,
            )
            result = grouped_moving_block_bootstrap_loss_difference(
                frozen,
                refit,
                groups,
                block_size=block,
                samples=int(inference["score_bootstrap_samples"]),
                confidence_level=float(inference["score_bootstrap_confidence"]),
                seed=int(inference["seed"]) + index * 10_000,
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "difference_direction": "frozen_minus_refit",
                    **result,
                    "block_size": block,
                    **diagnostics,
                    "n_observations": len(pivot),
                }
            )
            index += 1

    frame = pd.DataFrame(rows)
    frame["pvalue_holm"] = holm_adjust(frame["pvalue_raw"])
    frame["reject_holm_5pct"] = frame["pvalue_holm"] < 0.05
    frame["recalibration_helps"] = frame["reject_holm_5pct"] & (
        frame["mean_difference"] > 0
    )
    return frame
