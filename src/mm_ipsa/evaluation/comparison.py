"""Inferencia pareada de diferencias de scoring rules con dependencia temporal."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_METRICS = ("mean_crps", "energy_score", "variogram_score")


def sample_autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocovarianzas muestrales con normalizador 1/n para lags 0..max_lag."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if x.ndim != 1 or n < 2:
        raise ValueError("series debe ser un vector de largo al menos dos")
    if not 0 <= max_lag < n:
        raise ValueError("max_lag debe pertenecer a [0, n)")
    centered = x - x.mean()
    # El normalizador 1/n (no 1/(n-k)) mantiene la secuencia semidefinida
    # positiva, requisito de los estimadores espectrales de ventana plana.
    return np.array(
        [float(centered[: n - lag] @ centered[lag:]) / n for lag in range(max_lag + 1)]
    )


def pooled_autocovariance(
    segments: list[np.ndarray], max_lag: int
) -> tuple[np.ndarray, int]:
    """Autocovarianzas agrupadas sin cruzar fronteras entre segmentos.

    Los productos rezagados se acumulan solo dentro de cada segmento, pero el
    centrado y el normalizador usan la muestra completa. Estimar la ACF fold a
    fold con pocas decenas de observaciones produce anchos muy inestables; al
    agrupar se usa toda la evidencia disponible sin que el remuestreo llegue a
    concatenar el final de un fold con el inicio de otro.
    """
    if not segments:
        raise ValueError("segments no puede estar vacio")
    arrays = [np.asarray(segment, dtype=float) for segment in segments]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("cada segmento debe ser un vector")
    total = int(sum(len(array) for array in arrays))
    if total < 2:
        raise ValueError("se requieren al menos dos observaciones")
    if max_lag < 0:
        raise ValueError("max_lag debe ser no negativo")
    grand_mean = float(np.concatenate(arrays).mean())

    accumulated = np.zeros(max_lag + 1, dtype=float)
    for array in arrays:
        centered = array - grand_mean
        n = len(centered)
        for lag in range(min(max_lag, n - 1) + 1):
            accumulated[lag] += float(centered[: n - lag] @ centered[lag:])
    return accumulated / total, total


def _flat_top_weight(ratio: float) -> float:
    """Ventana trapezoidal de Politis-Romano: plana hasta 1/2 y lineal hasta 1."""
    magnitude = abs(ratio)
    if magnitude <= 0.5:
        return 1.0
    if magnitude <= 1.0:
        return 2.0 * (1.0 - magnitude)
    return 0.0


def politis_white_block_length(
    series: np.ndarray | None = None,
    *,
    segments: list[np.ndarray] | None = None,
    bootstrap_kind: str = "circular",
    max_block: int | None = None,
) -> dict[str, float]:
    """Block length automatico de Politis-White con correccion de 2009.

    Implementa el criterio de Politis y White (2004), "Automatic Block-Length
    Selection for the Dependent Bootstrap", con la correccion publicada por
    Patton, Politis y White (2009). El ancho de banda M se determina por la
    primera region donde la autocorrelacion cae dentro de la banda de ruido
    ``2 sqrt(log10(n)/n)`` durante ``K_n`` lags consecutivos.

    Se pasa ``series`` para una muestra contigua o ``segments`` cuando la
    muestra esta partida en folds; en el segundo caso las autocovarianzas se
    agrupan sin cruzar fronteras.

    Un block length fijado por analogia puede quedar corto frente a la
    dependencia real; en ese caso el bootstrap subestima la varianza de la
    media y produce intervalos y p-valores anti-conservadores.
    """
    if bootstrap_kind not in {"circular", "stationary"}:
        raise ValueError("bootstrap_kind debe ser circular o stationary")
    if (series is None) == (segments is None):
        raise ValueError("Debe entregarse exactamente uno de series o segments")

    if segments is None:
        x = np.asarray(series, dtype=float)
        if x.ndim != 1 or len(x) < 4:
            raise ValueError("series debe ser un vector de largo al menos cuatro")
        if not np.isfinite(x).all():
            raise ValueError("series contiene valores no finitos")
        pieces = [x]
        longest = len(x)
    else:
        pieces = [np.asarray(segment, dtype=float) for segment in segments]
        if any(not np.isfinite(piece).all() for piece in pieces):
            raise ValueError("segments contiene valores no finitos")
        pieces = [piece for piece in pieces if len(piece) >= 2]
        if not pieces:
            raise ValueError("segments no contiene ningun tramo utilizable")
        longest = max(len(piece) for piece in pieces)

    n = int(sum(len(piece) for piece in pieces))
    if n < 4:
        raise ValueError("se requieren al menos cuatro observaciones")

    hard_cap = int(max_block) if max_block is not None else longest
    hard_cap = max(1, min(hard_cap, longest))

    values = np.concatenate(pieces)
    # Una serie constante no tiene varianza exactamente nula: la media acumula
    # error de redondeo y deja desviaciones del orden de eps. Comparar contra
    # cero absoluto haria que el criterio interprete ruido numerico como
    # dependencia, por lo que el umbral es relativo a la escala de los datos.
    scale = max(float(np.max(np.abs(values))), 1.0)
    degenerate_tolerance = (1e-12 * scale) ** 2
    if float(np.var(values)) <= degenerate_tolerance:
        return {
            "block_length": 1.0,
            "block_length_raw": 1.0,
            "m_hat": 0.0,
            "bandwidth": 0.0,
            "n_observations": float(n),
            "degenerate": 1.0,
        }

    # El ancho de busqueda no puede exceder el tramo mas corto utilizable.
    search_lag = min(
        min(len(piece) for piece in pieces) - 1,
        max(5, int(np.ceil(np.sqrt(n)))) + 10,
    )
    search_lag = max(search_lag, 1)
    autocovariance, _ = pooled_autocovariance(pieces, search_lag)
    if autocovariance[0] <= degenerate_tolerance:
        return {
            "block_length": 1.0,
            "block_length_raw": 1.0,
            "m_hat": 0.0,
            "bandwidth": 0.0,
            "n_observations": float(n),
            "degenerate": 1.0,
        }
    correlation = autocovariance / autocovariance[0]

    threshold = 2.0 * np.sqrt(np.log10(n) / n)
    consecutive = max(5, int(np.ceil(np.sqrt(np.log10(n)))))

    m_hat = 0
    for candidate in range(1, search_lag + 1):
        window = correlation[candidate : candidate + consecutive + 1]
        if len(window) == 0:
            break
        if np.all(np.abs(window) < threshold):
            m_hat = candidate - 1
            break
    else:
        m_hat = search_lag
    m_hat = max(m_hat, 1)

    bandwidth = min(2 * m_hat, search_lag)
    lags = np.arange(-bandwidth, bandwidth + 1)
    weights = np.array([_flat_top_weight(lag / bandwidth) for lag in lags])
    covariances = autocovariance[np.abs(lags)]

    g_hat = float(np.sum(weights * np.abs(lags) * covariances))
    spectral = float(np.sum(weights * covariances))
    factor = 4.0 / 3.0 if bootstrap_kind == "circular" else 2.0
    d_hat = factor * spectral**2

    if d_hat <= 0.0 or not np.isfinite(g_hat) or g_hat == 0.0:
        raw = 1.0
    else:
        raw = float((2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0))
    if not np.isfinite(raw) or raw <= 0.0:
        raw = 1.0

    block = int(min(max(1, int(np.ceil(raw))), hard_cap))
    return {
        "block_length": float(block),
        "block_length_raw": float(raw),
        "m_hat": float(m_hat),
        "bandwidth": float(bandwidth),
        "n_observations": float(n),
        "degenerate": 0.0,
    }


def newey_west_long_run_variance(
    series: np.ndarray, lag: int | None = None
) -> tuple[float, int]:
    """Varianza de largo plazo con kernel de Bartlett y pesos no negativos.

    El kernel de Bartlett garantiza una estimacion no negativa aun cuando las
    autocovarianzas muestrales cambian de signo. Sin truncamiento explicito se
    usa la regla automatica de Newey-West (1994), ``floor(4 (n/100)^(2/9))``.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if x.ndim != 1 or n < 2:
        raise ValueError("series debe ser un vector de largo al menos dos")
    if not np.isfinite(x).all():
        raise ValueError("series contiene valores no finitos")

    truncation = (
        int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lag is None else int(lag)
    )
    truncation = max(0, min(truncation, n - 1))

    centered = x - x.mean()
    variance = float(centered @ centered) / n
    for k in range(1, truncation + 1):
        weight = 1.0 - k / (truncation + 1.0)
        covariance = float(centered[: n - k] @ centered[k:]) / n
        variance += 2.0 * weight * covariance
    return max(variance, 0.0), truncation


def grouped_newey_west_long_run_variance(
    segments: list[np.ndarray], lag: int | None = None
) -> tuple[float, int]:
    """Varianza de largo plazo agrupada sin productos cruzados entre folds.

    Aplicar el estimador HAC sobre la serie concatenada introduciria
    autocovarianzas entre el final de un fold y el inicio del siguiente, que el
    diseno rolling-origin excluye por construccion.
    """
    arrays = [np.asarray(segment, dtype=float) for segment in segments]
    arrays = [array for array in arrays if len(array) >= 2]
    if not arrays:
        raise ValueError("segments no contiene ningun tramo utilizable")
    n = int(sum(len(array) for array in arrays))
    shortest = min(len(array) for array in arrays)

    truncation = (
        int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lag is None else int(lag)
    )
    truncation = max(0, min(truncation, shortest - 1))

    autocovariance, _ = pooled_autocovariance(arrays, truncation)
    variance = float(autocovariance[0])
    for k in range(1, truncation + 1):
        weight = 1.0 - k / (truncation + 1.0)
        variance += 2.0 * weight * float(autocovariance[k])
    return max(variance, 0.0), truncation


def diebold_mariano(
    focal_losses: np.ndarray,
    benchmark_losses: np.ndarray,
    *,
    lag: int | None = None,
    small_sample_correction: bool = True,
    segments: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Contraste de Diebold-Mariano con varianza HAC y correccion de Harvey.

    Provee una ruta de inferencia independiente del bootstrap por bloques. Que
    ambas coincidan es evidencia de que la conclusion no depende del mecanismo
    de remuestreo; que discrepen indica que la dependencia temporal domina y la
    conclusion es fragil.

    La correccion de Harvey, Leybourne y Newbold (1997) ajusta el sesgo de la
    varianza en muestras cortas y contrasta contra una t de Student con n-1
    grados de libertad en vez de una normal.
    """
    from scipy import stats

    focal = np.asarray(focal_losses, dtype=float)
    benchmark = np.asarray(benchmark_losses, dtype=float)
    if focal.ndim != 1 or focal.shape != benchmark.shape or len(focal) < 3:
        raise ValueError("Las perdidas deben ser vectores pareados de largo >= 3")
    if not np.isfinite(focal).all() or not np.isfinite(benchmark).all():
        raise ValueError("Las perdidas contienen valores no finitos")

    difference = focal - benchmark
    n = len(difference)
    mean_difference = float(difference.mean())
    if segments is None:
        long_run, truncation = newey_west_long_run_variance(difference, lag)
    else:
        long_run, truncation = grouped_newey_west_long_run_variance(segments, lag)
    if long_run <= 0.0:
        return {
            "mean_difference": mean_difference,
            "dm_statistic": float("nan"),
            "pvalue": float("nan"),
            "long_run_variance": 0.0,
            "hac_lag": float(truncation),
            "n_observations": float(n),
            "degenerate": 1.0,
        }

    statistic = mean_difference / np.sqrt(long_run / n)
    horizon = truncation + 1
    if small_sample_correction:
        factor = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
        statistic *= np.sqrt(max(factor, 1e-12))
    pvalue = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return {
        "mean_difference": mean_difference,
        "dm_statistic": float(statistic),
        "pvalue": pvalue,
        "long_run_variance": float(long_run),
        "hac_lag": float(truncation),
        "n_observations": float(n),
        "degenerate": 0.0,
    }


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


def resolve_block_size(
    difference: np.ndarray,
    *,
    mode: str,
    configured: int,
    max_block: int,
    groups: np.ndarray | None = None,
) -> tuple[int, dict[str, float]]:
    """Resuelve el block length efectivo y devuelve su diagnostico.

    En modo ``auto`` se aplica Politis-White una sola vez sobre las
    autocovarianzas agrupadas dentro de folds. Estimar el ancho fold a fold y
    quedarse con el maximo dejaba que un unico tramo corto y ruidoso fijara el
    ancho de todo el contraste.
    """
    if mode not in {"auto", "fixed"}:
        raise ValueError("block_size_mode debe ser auto o fixed")
    ceiling = max(1, int(max_block))
    if mode == "fixed":
        chosen = int(min(max(1, int(configured)), ceiling))
        return chosen, {
            "block_size_mode_auto": 0.0,
            "block_size_configured": float(configured),
            "block_size_ceiling": float(ceiling),
        }

    values = np.asarray(difference, dtype=float)
    if groups is None:
        segments = [values]
    else:
        group_values = np.asarray(list(groups), dtype=object)
        segments = [
            values[group_values == group]
            for group in dict.fromkeys(group_values.tolist())
        ]

    try:
        diagnostics = politis_white_block_length(segments=segments, max_block=ceiling)
    except ValueError:
        chosen = int(min(max(1, int(configured)), ceiling))
        return chosen, {
            "block_size_mode_auto": 1.0,
            "block_size_configured": float(configured),
            "block_size_ceiling": float(ceiling),
            "block_size_politis_white": float("nan"),
        }

    raw = float(diagnostics["block_length_raw"])
    chosen = int(min(max(1, int(diagnostics["block_length"])), ceiling))
    # Si el criterio pide mas dependencia de la que cabe en el tramo mas corto,
    # el remuestreo no puede reproducirla y el intervalo resultante sigue
    # siendo optimista. Se marca para que la lectura no lo pase por alto.
    return chosen, {
        "block_size_mode_auto": 1.0,
        "block_size_configured": float(configured),
        "block_size_ceiling": float(ceiling),
        "block_size_politis_white": float(diagnostics["block_length"]),
        "block_size_politis_white_raw": raw,
        "block_size_bandwidth": float(diagnostics["bandwidth"]),
        "block_size_capped": float(raw > ceiling),
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
    block_size_mode: str = "auto",
    samples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    inference_status: str = "development_validation",
) -> pd.DataFrame:
    """Compara un modelo contra controles usando exactamente las mismas fechas.

    ``block_size`` actua como valor de respaldo: con ``block_size_mode='auto'``
    el ancho efectivo lo determina Politis-White sobre cada serie de
    diferencias y queda registrado en la columna ``block_size``.
    """
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
            focal_losses = pivot[focal_model].to_numpy()
            benchmark_losses = pivot[benchmark].to_numpy()
            effective_block, block_diagnostics = resolve_block_size(
                focal_losses - benchmark_losses,
                mode=block_size_mode,
                configured=block_size,
                max_block=len(pivot),
            )
            result = moving_block_bootstrap_loss_difference(
                focal_losses,
                benchmark_losses,
                block_size=effective_block,
                samples=samples,
                confidence_level=confidence_level,
                seed=seed + comparison_index * 10_000,
            )
            dm = diebold_mariano(focal_losses, benchmark_losses)
            rows.append(
                {
                    "metric": metric,
                    "focal_model": focal_model,
                    "benchmark_model": benchmark,
                    "difference_direction": "focal_minus_benchmark",
                    **result,
                    "dm_statistic": dm["dm_statistic"],
                    "dm_pvalue": dm["pvalue"],
                    "dm_hac_lag": dm["hac_lag"],
                    "n_observations": len(pivot),
                    "block_size": effective_block,
                    **block_diagnostics,
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
    block_size_mode: str = "auto",
    samples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    inference_status: str = "development_validation",
) -> pd.DataFrame:
    """Compara modelos en varios folds sin formar bloques entre folds.

    El ancho automatico se evalua dentro de cada fold y se toma el maximo, de
    modo que ningun fold quede sub-bloqueado y los bloques sigan sin cruzar
    fronteras temporales.
    """
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
        smallest_group = int(
            min(int((groups == group).sum()) for group in dict.fromkeys(groups.tolist()))
        )
        for benchmark in benchmark_models:
            focal_losses = pivot[focal_model].to_numpy()
            benchmark_losses = pivot[benchmark].to_numpy()
            effective_block, block_diagnostics = resolve_block_size(
                focal_losses - benchmark_losses,
                mode=block_size_mode,
                configured=block_size,
                max_block=smallest_group,
                groups=groups,
            )
            result = grouped_moving_block_bootstrap_loss_difference(
                focal_losses,
                benchmark_losses,
                groups,
                block_size=effective_block,
                samples=samples,
                confidence_level=confidence_level,
                seed=seed + comparison_index * 10_000,
            )
            difference_series = focal_losses - benchmark_losses
            dm = diebold_mariano(
                focal_losses,
                benchmark_losses,
                segments=[
                    difference_series[groups == group]
                    for group in dict.fromkeys(groups.tolist())
                ],
            )
            rows.append(
                {
                    "metric": metric,
                    "focal_model": focal_model,
                    "benchmark_model": benchmark,
                    "difference_direction": "focal_minus_benchmark",
                    **result,
                    "dm_statistic": dm["dm_statistic"],
                    "dm_pvalue": dm["pvalue"],
                    "dm_hac_lag": dm["hac_lag"],
                    "n_observations": len(pivot),
                    "group_column": group_column,
                    "block_size": effective_block,
                    **block_diagnostics,
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


def block_size_sensitivity(
    observation_scores: pd.DataFrame,
    *,
    focal_model: str,
    benchmark_models: list[str],
    block_sizes: Iterable[int],
    group_column: str | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    samples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Repite la comparacion completa para varios block lengths fijos.

    Una conclusion que solo sobrevive con un ancho concreto no es robusta. La
    tabla resultante permite verificar si el signo, el intervalo y el p-valor
    ajustado se mantienen al variar el unico parametro que gobierna cuanta
    dependencia temporal preserva el remuestreo.
    """
    frames: list[pd.DataFrame] = []
    for block in block_sizes:
        if group_column is None:
            frame = compare_focal_model(
                observation_scores,
                focal_model=focal_model,
                benchmark_models=benchmark_models,
                metrics=metrics,
                block_size=int(block),
                block_size_mode="fixed",
                samples=samples,
                confidence_level=confidence_level,
                seed=seed,
            )
        else:
            frame = compare_focal_model_by_group(
                observation_scores,
                group_column=group_column,
                focal_model=focal_model,
                benchmark_models=benchmark_models,
                metrics=metrics,
                block_size=int(block),
                block_size_mode="fixed",
                samples=samples,
                confidence_level=confidence_level,
                seed=seed,
            )
        frame.insert(0, "requested_block_size", int(block))
        frames.append(frame)
    if not frames:
        raise ValueError("block_sizes no puede estar vacio")
    return pd.concat(frames, ignore_index=True)
