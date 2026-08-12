"""Inferencia pareada de diferencias de scoring rules con dependencia temporal."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("mean_crps", "energy_score", "variogram_score")


def moving_block_bootstrap_loss_difference(
    focal_losses: np.ndarray,
    benchmark_losses: np.ndarray,
    *,
    block_size: int,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """IC y p-valor bilateral para media(focal - benchmark).

    Un valor negativo favorece al modelo focal porque todos los scores son
    perdidas. El p-valor remuestrea la serie centrada para imponer H0: E[d]=0.
    """
    focal = np.asarray(focal_losses, dtype=float)
    benchmark = np.asarray(benchmark_losses, dtype=float)
    if focal.ndim != 1 or focal.shape != benchmark.shape or len(focal) < 2:
        raise ValueError("Las perdidas deben ser vectores pareados del mismo largo")
    if not np.isfinite(focal).all() or not np.isfinite(benchmark).all():
        raise ValueError("Las perdidas contienen valores no finitos")
    n = len(focal)
    if not 1 <= block_size <= n:
        raise ValueError("block_size invalido")
    if samples < 100:
        raise ValueError("samples debe ser al menos 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level debe pertenecer a (0, 1)")

    difference = focal - benchmark
    observed = float(difference.mean())
    centered = difference - observed
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    null_means = np.empty(samples, dtype=float)
    max_start = n - block_size + 1
    for sample in range(samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, max_start))
            indices.extend(range(start, start + block_size))
        selected = np.asarray(indices[:n])
        bootstrap_means[sample] = float(difference[selected].mean())
        null_means[sample] = float(centered[selected].mean())

    alpha = 1.0 - confidence_level
    null_low, null_high = np.quantile(
        null_means, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    low = observed - null_high
    high = observed - null_low
    median = float(np.median(bootstrap_means))
    lower_tail = (1.0 + float(np.sum(null_means <= observed))) / (samples + 1.0)
    upper_tail = (1.0 + float(np.sum(null_means >= observed))) / (samples + 1.0)
    pvalue = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "focal_mean_loss": float(focal.mean()),
        "benchmark_mean_loss": float(benchmark.mean()),
        "mean_difference": observed,
        "relative_difference_pct": float(
            100.0 * observed / max(abs(float(benchmark.mean())), 1e-15)
        ),
        "ci_low": float(low),
        "bootstrap_median": float(median),
        "ci_high": float(high),
        "probability_focal_better": float(np.mean(bootstrap_means < 0.0)),
        "pvalue_raw": float(min(max(pvalue, 0.0), 1.0)),
    }


def grouped_moving_block_bootstrap_loss_difference(
    focal_losses: np.ndarray,
    benchmark_losses: np.ndarray,
    groups: Iterable[object],
    *,
    block_size: int,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """Bootstrap por bloques que nunca cruza fronteras entre folds.

    Cada fold conserva su largo original y se remuestrea de forma independiente.
    La media final pondera naturalmente cada observacion, no cada fold por igual.
    """
    focal = np.asarray(focal_losses, dtype=float)
    benchmark = np.asarray(benchmark_losses, dtype=float)
    group_values = np.asarray(list(groups), dtype=object)
    if (
        focal.ndim != 1
        or focal.shape != benchmark.shape
        or group_values.shape != focal.shape
        or len(focal) < 2
    ):
        raise ValueError("Perdidas y grupos deben ser vectores pareados del mismo largo")
    if not np.isfinite(focal).all() or not np.isfinite(benchmark).all():
        raise ValueError("Las perdidas contienen valores no finitos")
    if samples < 100:
        raise ValueError("samples debe ser al menos 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level debe pertenecer a (0, 1)")

    unique_groups = list(dict.fromkeys(group_values.tolist()))
    group_indices = [np.flatnonzero(group_values == group) for group in unique_groups]
    if not unique_groups or any(len(indices) < 2 for indices in group_indices):
        raise ValueError("Cada fold debe contener al menos dos observaciones")
    if block_size < 1 or any(block_size > len(indices) for indices in group_indices):
        raise ValueError("block_size no puede superar el largo de ningun fold")

    difference = focal - benchmark
    observed = float(difference.mean())
    centered = difference - observed
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    null_means = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected_parts: list[np.ndarray] = []
        for indices in group_indices:
            n_group = len(indices)
            sampled_local: list[int] = []
            max_start = n_group - block_size + 1
            while len(sampled_local) < n_group:
                start = int(rng.integers(0, max_start))
                sampled_local.extend(range(start, start + block_size))
            selected_parts.append(indices[np.asarray(sampled_local[:n_group])])
        selected = np.concatenate(selected_parts)
        bootstrap_means[sample] = float(difference[selected].mean())
        null_means[sample] = float(centered[selected].mean())

    alpha = 1.0 - confidence_level
    null_low, null_high = np.quantile(
        null_means, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    lower_tail = (1.0 + float(np.sum(null_means <= observed))) / (samples + 1.0)
    upper_tail = (1.0 + float(np.sum(null_means >= observed))) / (samples + 1.0)
    return {
        "focal_mean_loss": float(focal.mean()),
        "benchmark_mean_loss": float(benchmark.mean()),
        "mean_difference": observed,
        "relative_difference_pct": float(
            100.0 * observed / max(abs(float(benchmark.mean())), 1e-15)
        ),
        "ci_low": float(observed - null_high),
        "bootstrap_median": float(np.median(bootstrap_means)),
        "ci_high": float(observed - null_low),
        "probability_focal_better": float(np.mean(bootstrap_means < 0.0)),
        "pvalue_raw": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "n_groups": float(len(unique_groups)),
    }


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    """Ajuste step-down de Holm para controlar FWER."""
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues invalidos")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def compare_focal_model(
    observation_scores: pd.DataFrame,
    *,
    focal_model: str,
    benchmark_models: list[str],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    block_size: int = 4,
    samples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    inference_status: str = "development_validation",
) -> pd.DataFrame:
    """Compara un modelo contra controles usando exactamente las mismas fechas."""
    required = {"model", "observation", *metrics}
    missing = sorted(required.difference(observation_scores.columns))
    if missing:
        raise ValueError(f"Faltan columnas de scores: {missing}")
    if observation_scores.duplicated(["model", "observation"]).any():
        raise ValueError("Hay scores duplicados por modelo y observacion")
    available = set(observation_scores["model"])
    missing_models = [
        model for model in [focal_model, *benchmark_models] if model not in available
    ]
    if missing_models:
        raise ValueError(f"Faltan modelos para comparar: {missing_models}")

    rows: list[dict[str, object]] = []
    comparison_index = 0
    for metric in metrics:
        pivot = observation_scores.pivot(index="observation", columns="model", values=metric)
        needed = [focal_model, *benchmark_models]
        if pivot[needed].isna().any().any():
            raise ValueError(f"Las fechas pareadas no coinciden para {metric}")
        for benchmark in benchmark_models:
            result = moving_block_bootstrap_loss_difference(
                pivot[focal_model].to_numpy(),
                pivot[benchmark].to_numpy(),
                block_size=block_size,
                samples=samples,
                confidence_level=confidence_level,
                seed=seed + comparison_index * 10_000,
            )
            rows.append(
                {
                    "metric": metric,
                    "focal_model": focal_model,
                    "benchmark_model": benchmark,
                    "difference_direction": "focal_minus_benchmark",
                    **result,
                    "n_observations": len(pivot),
                    "block_size": block_size,
                    "bootstrap_samples": samples,
                    "confidence_level": confidence_level,
                    "inference_status": inference_status,
                }
            )
            comparison_index += 1

    frame = pd.DataFrame(rows)
    frame["pvalue_holm"] = holm_adjust(frame["pvalue_raw"])
    frame["reject_holm_5pct"] = frame["pvalue_holm"] < 0.05
    frame["ci_excludes_zero"] = (frame["ci_low"] > 0.0) | (frame["ci_high"] < 0.0)
    frame["registered_primary"] = (
        (frame["metric"] == "mean_crps")
        & (frame["benchmark_model"] == "gaussian_terminal")
    )
    return frame


def compare_focal_model_by_group(
    observation_scores: pd.DataFrame,
    *,
    group_column: str,
    focal_model: str,
    benchmark_models: list[str],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    block_size: int = 4,
    samples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    inference_status: str = "development_validation",
) -> pd.DataFrame:
    """Compara modelos en varios folds sin formar bloques entre folds."""
    required = {"model", "observation", group_column, *metrics}
    missing = sorted(required.difference(observation_scores.columns))
    if missing:
        raise ValueError(f"Faltan columnas de scores: {missing}")
    key = [group_column, "observation"]
    if observation_scores.duplicated(["model", *key]).any():
        raise ValueError("Hay scores duplicados por modelo, fold y observacion")
    available = set(observation_scores["model"])
    missing_models = [
        model for model in [focal_model, *benchmark_models] if model not in available
    ]
    if missing_models:
        raise ValueError(f"Faltan modelos para comparar: {missing_models}")

    rows: list[dict[str, object]] = []
    comparison_index = 0
    for metric in metrics:
        pivot = observation_scores.pivot(index=key, columns="model", values=metric)
        needed = [focal_model, *benchmark_models]
        if pivot[needed].isna().any().any():
            raise ValueError(f"Las observaciones pareadas no coinciden para {metric}")
        groups = pivot.index.get_level_values(group_column).to_numpy()
        for benchmark in benchmark_models:
            result = grouped_moving_block_bootstrap_loss_difference(
                pivot[focal_model].to_numpy(),
                pivot[benchmark].to_numpy(),
                groups,
                block_size=block_size,
                samples=samples,
                confidence_level=confidence_level,
                seed=seed + comparison_index * 10_000,
            )
            rows.append(
                {
                    "metric": metric,
                    "focal_model": focal_model,
                    "benchmark_model": benchmark,
                    "difference_direction": "focal_minus_benchmark",
                    **result,
                    "n_observations": len(pivot),
                    "group_column": group_column,
                    "block_size": block_size,
                    "bootstrap_samples": samples,
                    "confidence_level": confidence_level,
                    "inference_status": inference_status,
                }
            )
            comparison_index += 1

    frame = pd.DataFrame(rows)
    frame["pvalue_holm"] = holm_adjust(frame["pvalue_raw"])
    frame["reject_holm_5pct"] = frame["pvalue_holm"] < 0.05
    frame["ci_excludes_zero"] = (frame["ci_low"] > 0.0) | (frame["ci_high"] < 0.0)
    frame["registered_primary"] = (
        (frame["metric"] == "mean_crps")
        & (frame["benchmark_model"] == "gaussian_terminal")
    )
    return frame
