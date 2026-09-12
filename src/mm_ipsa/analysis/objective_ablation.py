"""Ablacion de los terminos del objetivo, contrastada fuera de muestra.

El objetivo pondera cuatro momentos marginales y un termino de dependencia. Esas
ponderaciones se fijaron al disenar el metodo y durante mucho tiempo no se
contrastaron: se sabia que el ajuste en muestra era casi exacto, pero no si cada
termino compraba algo predictivo.

Este modulo responde esa pregunta de la unica forma que decide. Recalibra el
generador con distintas ponderaciones, puntua cada version contra las mismas
ventanas fuera de muestra, y contrasta cada variante contra la publicada con el
mismo aparato inferencial que el resto del proyecto: bootstrap por bloques con
ancho elegido por Politis-White y correccion de Holm sobre las tres reglas.

La distincion entre ajuste y desempeno es el punto. Un peso mas alto siempre
mejora el ajuste en muestra del termino que pondera, porque eso es lo que el
optimizador persigue. Si ese ajuste mejor no se traduce en mejor prediccion, el
termino estaba ajustando error de estimacion y no estructura.

Las ventanas se toman disjuntas. Un retorno terminal calculado cada dia comparte
jornadas con su vecino, y tratarlas como observaciones independientes estrecharia
los intervalos sin justificacion.
"""

from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

SCORING_RULES = ("mean_crps", "energy_score", "variogram_score")


class AblationReport(TypedDict):
    fit: pd.DataFrame
    tests: pd.DataFrame
    windows: int

# Ablaciones que responden las dos preguntas de diseno abiertas: si el termino de
# dependencia admite mas precision, y si los momentos superiores compran algo.
DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "publicada": {},
    "dependencia_alta": {"cov_weight": 20.0},
    "sin_momentos_superiores": {"moment_weights": {"k3": 0.0, "k4": 0.0}},
    "momentos_superiores_altos": {"moment_weights": {"k3": 2.0, "k4": 1.5}},
}


def _apply_overrides(
    settings: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Sobreescribe parametros sin mutar la configuracion de origen.

    ``moment_weights`` se funde clave a clave para poder alterar un solo momento
    sin tener que repetir los otros tres.
    """
    updated = copy.deepcopy(settings)
    for key, value in overrides.items():
        if key == "moment_weights":
            weights = dict(updated["moment_weights"])
            weights.update(value)
            updated["moment_weights"] = weights
        else:
            updated[key] = value
    return updated


def _relative_error(fitted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.max(np.abs(fitted - target) / np.maximum(np.abs(target), 1e-12))
    )


def calibrate_and_score(
    moment_targets: np.ndarray,
    covariance_target: np.ndarray,
    observations: pd.DataFrame,
    main_cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calibra una variante y devuelve su ajuste y sus perdidas por ventana.

    ``strict_solver`` se desactiva porque una variante deliberadamente degradada
    puede no alcanzar los umbrales de estacionariedad, y detener la medicion por
    eso impediria justamente contrastarla.
    """
    from mm_ipsa.config import objective_weights
    from mm_ipsa.evaluation.scoring import evaluate_scenarios_detailed
    from mm_ipsa.mm.bcd import BCDSolver
    from mm_ipsa.mm.objective import MMObjective

    settings = _apply_overrides(main_cfg["mm"], overrides)
    settings["strict_solver"] = False

    objective = MMObjective(
        moment_targets, covariance_target, objective_weights(settings),
        int(settings["N_scenarios"]),
    )
    with redirect_stdout(io.StringIO()):
        scenarios, probabilities = BCDSolver(
            objective, settings, n_workers=1
        ).solve()
        _, _, by_window = evaluate_scenarios_detailed(
            "MM", scenarios, probabilities, observations.to_numpy(),
            list(observations.columns), observation_ids=observations.index,
            alpha=float(main_cfg["portfolio"]["alpha_cvar"]),
            seed=int(main_cfg["evaluation"]["seed"]),
            energy_pair_samples=int(main_cfg["evaluation"]["energy_pair_samples"]),
        )

    fitted, _, fitted_covariance = objective.compute_moments(
        scenarios, probabilities
    )
    off_diagonal = ~np.eye(moment_targets.shape[1], dtype=bool)
    fit = {
        "objective": float(objective.evaluate(scenarios, probabilities)),
        "mean_error": _relative_error(fitted[0], moment_targets[0]),
        "variance_error": _relative_error(fitted[1], moment_targets[1]),
        "third_moment_error": _relative_error(fitted[2], moment_targets[2]),
        "fourth_moment_error": _relative_error(fitted[3], moment_targets[3]),
        "covariance_error": _relative_error(
            fitted_covariance[off_diagonal], covariance_target[off_diagonal]
        ),
        **{rule: float(by_window[rule].mean()) for rule in SCORING_RULES},
    }
    return fit, by_window


def compare_against_reference(
    losses: dict[str, pd.DataFrame],
    reference: str,
    *,
    samples: int = 5000,
    seed: int = 11,
) -> pd.DataFrame:
    """Contrasta cada variante contra la referencia, regla por regla.

    Holm se aplica dentro de cada variante sobre sus tres reglas, porque la
    pregunta es si esa variante cambia el desempeno y no cual de todos los
    contrastes resulto mas extremo.
    """
    from mm_ipsa.evaluation.comparison import (
        holm_adjust,
        moving_block_bootstrap_loss_difference,
        resolve_block_size,
    )

    if reference not in losses:
        raise ValueError(f"La referencia {reference} no esta entre las variantes")

    rows: list[dict[str, Any]] = []
    baseline = losses[reference]
    for variant, frame in losses.items():
        if variant == reference:
            continue
        results = []
        for rule in SCORING_RULES:
            focal = frame[rule].to_numpy()
            benchmark = baseline[rule].to_numpy()
            with redirect_stdout(io.StringIO()):
                block, _ = resolve_block_size(
                    focal - benchmark, mode="auto", configured=5, max_block=20
                )
                results.append(
                    moving_block_bootstrap_loss_difference(
                        focal, benchmark, block_size=int(block),
                        samples=samples, confidence_level=0.95, seed=seed,
                    )
                )
        adjusted = holm_adjust([item["pvalue_raw"] for item in results])
        for rule, item, pvalue in zip(SCORING_RULES, results, adjusted):
            rows.append(
                {
                    "variant": variant,
                    "rule": rule,
                    "mean_difference": item["mean_difference"],
                    "relative_pct": item["relative_difference_pct"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                    "pvalue_holm": float(pvalue),
                    "worse_than_reference": bool(
                        pvalue < 0.05 and item["mean_difference"] > 0.0
                    ),
                    "better_than_reference": bool(
                        pvalue < 0.05 and item["mean_difference"] < 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_objective_ablation(
    main_cfg: dict[str, Any],
    prices: pd.DataFrame,
    observations: pd.DataFrame,
    output: str | Path,
    *,
    variants: dict[str, dict[str, Any]] | None = None,
    reference: str = "publicada",
) -> AblationReport:
    """Ejecuta la ablacion completa y escribe ajuste y contrastes."""
    from mm_ipsa.config import target_parameters
    from mm_ipsa.data.transform import _rolling_terminal_fast
    from mm_ipsa.mm.targets import compute_targets

    chosen = DEFAULT_VARIANTS if variants is None else variants
    if reference not in chosen:
        raise ValueError(f"La referencia {reference} no esta entre las variantes")

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    horizon = int(main_cfg["data"]["H"])

    daily = prices.pct_change().dropna(how="all")
    with redirect_stdout(io.StringIO()):
        moments, covariance, _ = compute_targets(
            _rolling_terminal_fast(daily, horizon), daily,
            **target_parameters(main_cfg["mm"]),
        )

    windows = observations.iloc[::horizon]
    fits: list[dict[str, Any]] = []
    losses: dict[str, pd.DataFrame] = {}
    for variant, overrides in chosen.items():
        fit, by_window = calibrate_and_score(
            moments, covariance, windows, main_cfg, overrides
        )
        fits.append({"variant": variant, **fit})
        losses[variant] = by_window

    fit_table = pd.DataFrame(fits)
    comparison = compare_against_reference(losses, reference)
    fit_table.to_csv(target / "objective_ablation_fit.csv", index=False)
    comparison.to_csv(target / "objective_ablation_tests.csv", index=False)
    return {"fit": fit_table, "tests": comparison, "windows": len(windows)}


def _fold_groups(losses: dict[str, pd.DataFrame]) -> np.ndarray:
    """Etiqueta de fold por observacion, tomada de cualquiera de las variantes.

    Todas comparten el mismo calendario por construccion, de modo que cualquiera
    sirve de referencia y la coincidencia se comprueba al ensamblarlas.
    """
    return next(iter(losses.values()))["fold_id"].to_numpy()


def compare_across_folds(
    losses: dict[str, pd.DataFrame],
    reference: str,
    *,
    samples: int = 5000,
    seed: int = 11,
) -> pd.DataFrame:
    """Contrasta variantes sobre folds, sin remuestrear a traves de sus fronteras.

    Un bloque que cruzara de un fold al siguiente uniria observaciones separadas
    por un reajuste completo del modelo, que es justamente la discontinuidad que
    el diseno rolling-origin introduce a proposito.
    """
    from mm_ipsa.evaluation.comparison import (
        grouped_moving_block_bootstrap_loss_difference,
        holm_adjust,
        resolve_block_size,
    )

    if reference not in losses:
        raise ValueError(f"La referencia {reference} no esta entre las variantes")

    groups = _fold_groups(losses)
    baseline = losses[reference]
    rows: list[dict[str, Any]] = []

    for variant, frame in losses.items():
        if variant == reference:
            continue
        if len(frame) != len(baseline):
            raise ValueError(
                f"{variant} tiene {len(frame)} ventanas y la referencia "
                f"{len(baseline)}"
            )
        results = []
        for rule in SCORING_RULES:
            focal = frame[rule].to_numpy()
            benchmark = baseline[rule].to_numpy()
            with redirect_stdout(io.StringIO()):
                block, _ = resolve_block_size(
                    focal - benchmark, mode="auto", configured=5,
                    max_block=20, groups=groups,
                )
                results.append(
                    grouped_moving_block_bootstrap_loss_difference(
                        focal, benchmark, groups, block_size=int(block),
                        samples=samples, confidence_level=0.95, seed=seed,
                    )
                )
        adjusted = holm_adjust([item["pvalue_raw"] for item in results])
        for rule, item, pvalue in zip(SCORING_RULES, results, adjusted):
            rows.append(
                {
                    "variant": variant,
                    "rule": rule,
                    "mean_difference": item["mean_difference"],
                    "relative_pct": item["relative_difference_pct"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                    "pvalue_holm": float(pvalue),
                    "worse_than_reference": bool(
                        pvalue < 0.05 and item["mean_difference"] > 0.0
                    ),
                    "better_than_reference": bool(
                        pvalue < 0.05 and item["mean_difference"] < 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_rolling_objective_ablation(
    main_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    daily: pd.DataFrame,
    output: str | Path,
    *,
    variants: dict[str, dict[str, Any]] | None = None,
    reference: str = "publicada",
) -> AblationReport:
    """Repite la ablacion recalibrando cada variante en cada origen.

    La version de origen unico calibra una sola vez y puntua todo el periodo
    posterior, de modo que un resultado favorable podria deberse a que esa unica
    calibracion cayo bien. Aqui cada variante se recalibra al inicio de cada fold
    con datos exclusivamente anteriores, que es el diseno bajo el que el proyecto
    contrasta todo lo demas.
    """
    from mm_ipsa.analysis.rolling_origin import build_fold_samples
    from mm_ipsa.config import target_parameters
    from mm_ipsa.mm.targets import compute_targets

    chosen = DEFAULT_VARIANTS if variants is None else variants
    if reference not in chosen:
        raise ValueError(f"La referencia {reference} no esta entre las variantes")

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    horizon = int(main_cfg["data"]["H"])
    validation = experiment_cfg["validation"]

    fits: list[dict[str, Any]] = []
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in chosen}

    for fold in experiment_cfg["folds"]:
        training_daily, training_terminal, evaluation = build_fold_samples(
            daily, fold, horizon,
            minimum_training_daily_rows=int(
                validation["minimum_training_daily_rows"]
            ),
            minimum_evaluation_terminal_rows=int(
                validation["minimum_evaluation_terminal_rows"]
            ),
        )
        with redirect_stdout(io.StringIO()):
            moments, covariance, _ = compute_targets(
                training_terminal, training_daily,
                **target_parameters(main_cfg["mm"]),
            )
        for variant, overrides in chosen.items():
            fit, by_window = calibrate_and_score(
                moments, covariance, evaluation, main_cfg, overrides
            )
            by_window = by_window.copy()
            by_window["fold_id"] = str(fold["fold_id"])
            collected[variant].append(by_window)
            fits.append(
                {"fold_id": str(fold["fold_id"]), "variant": variant, **fit}
            )

    losses = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in collected.items()
    }
    fit_table = pd.DataFrame(fits)
    comparison = compare_across_folds(losses, reference)

    fit_table.to_csv(target / "rolling_ablation_fit.csv", index=False)
    comparison.to_csv(target / "rolling_ablation_tests.csv", index=False)
    return {
        "fit": fit_table,
        "tests": comparison,
        "windows": len(losses[reference]),
    }
