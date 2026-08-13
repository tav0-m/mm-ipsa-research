"""Diagnosticos de calibracion por transformada integral de probabilidad.

El CRPS es una regla propia que agrega calibracion y sharpness en un unico
numero: un modelo puede perder por estar mal calibrado o por ser innecesariamente
difuso, y el score no distingue cual de las dos cosas ocurre. El PIT separa
ambas propiedades y convierte "MM no gana" en una afirmacion sobre que parte de
la distribucion falla.

Lectura del histograma:

- uniforme        -> predictiva calibrada;
- forma de U      -> demasiado estrecha, el modelo es sobreconfiado;
- forma de joroba -> demasiado ancha;
- inclinado       -> sesgo sistematico de localizacion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

UNIFORM_VARIANCE = 1.0 / 12.0


def randomized_pit(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """PIT aleatorizado para predictivas discretas, con shape ``(T, n)``.

    Una predictiva discreta asigna masa positiva a puntos concretos, de modo
    que ``F`` salta y el PIT simple deja de ser uniforme cuando la observacion
    coincide con un punto del soporte. La version aleatorizada reparte la masa
    del salto, ``u = F(y^-) + V (F(y) - F(y^-))`` con ``V ~ U(0,1)``, y
    restituye uniformidad exacta en esos empates. Con observaciones continuas
    los empates tienen probabilidad nula y la correccion queda inactiva.

    Esto no corrige un efecto distinto y mas relevante aqui: ``F`` solo toma
    ``N+1`` valores, por lo que un soporte pequeno produce un histograma
    escalonado que los contrastes de uniformidad interpretan como mala
    calibracion. Ese sesgo se controla igualando el numero de escenarios entre
    modelos con ``resample_support``, no aleatorizando.
    """
    x = np.asarray(scenarios, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(observations, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("scenarios y observations deben ser matrices")
    if x.shape[1] != y.shape[1]:
        raise ValueError("scenarios y observations deben compartir el numero de activos")
    if p.shape != (x.shape[0],):
        raise ValueError("probabilities debe tener un valor por escenario")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-8):
        raise ValueError("probabilities debe ser no negativo y sumar uno")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("scenarios u observations contienen valores no finitos")

    rng = np.random.default_rng(seed)
    n_observations, n_assets = y.shape
    output = np.empty((n_observations, n_assets), dtype=float)

    for asset in range(n_assets):
        order = np.argsort(x[:, asset], kind="stable")
        sorted_support = x[order, asset]
        cumulative = np.concatenate([[0.0], np.cumsum(p[order])])
        # searchsorted 'left'/'right' entrega la masa estrictamente menor y la
        # masa menor o igual, que son exactamente F(y^-) y F(y).
        lower = cumulative[np.searchsorted(sorted_support, y[:, asset], side="left")]
        upper = cumulative[np.searchsorted(sorted_support, y[:, asset], side="right")]
        uniform = rng.random(n_observations)
        output[:, asset] = lower + uniform * (upper - lower)

    return np.clip(output, 0.0, 1.0)


def reliability_index(values: np.ndarray, bins: int = 10) -> float:
    """Suma de desviaciones absolutas del histograma respecto del uniforme.

    Vale cero para un histograma perfectamente plano y crece con cualquier
    desviacion, sin importar su forma.
    """
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    frequencies = counts / max(len(values), 1)
    return float(np.sum(np.abs(frequencies - 1.0 / bins)))


def classify_dispersion(ratio: float, tolerance: float = 0.05) -> str:
    """Traduce la razon de dispersion a una lectura interpretable."""
    if ratio > 1.0 + tolerance:
        return "subdispersa"
    if ratio < 1.0 - tolerance:
        return "sobredispersa"
    return "calibrada"


def pit_diagnostics(
    pit: np.ndarray,
    labels: list[str],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Diagnostico de uniformidad por activo.

    Los contrastes se aplican activo por activo. Agrupar todas las columnas en
    una sola prueba invalidaria el p-valor, porque las observaciones de una
    misma fecha comparten el estado del mercado y no son independientes.
    """
    values = np.asarray(pit, dtype=float)
    if values.ndim != 2:
        raise ValueError("pit debe ser una matriz (T, n)")
    if values.shape[1] != len(labels):
        raise ValueError("labels debe tener un nombre por activo")
    if bins < 2:
        raise ValueError("bins debe ser al menos dos")

    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        column = values[:, index]
        variance = float(np.var(column))
        ratio = variance / UNIFORM_VARIANCE
        kolmogorov = stats.kstest(column, "uniform")
        rows.append(
            {
                "asset": label,
                "n_observations": int(len(column)),
                "pit_mean": float(column.mean()),
                "pit_variance": variance,
                # >1 indica predictiva demasiado estrecha: las observaciones
                # caen en las colas mas seguido de lo que el modelo admite.
                "dispersion_ratio": float(ratio),
                "dispersion": classify_dispersion(float(ratio)),
                "reliability_index": reliability_index(column, bins),
                "ks_statistic": float(kolmogorov.statistic),
                "ks_pvalue": float(kolmogorov.pvalue),
            }
        )
    return pd.DataFrame(rows)


def resample_support(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    support_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remuestrea una predictiva discreta a un soporte de tamano fijo.

    Sirve para igualar la resolucion del ensemble entre modelos. Sin esto, un
    modelo con menos escenarios exhibe error de muestreo finito que el contraste
    de uniformidad interpreta como mala calibracion.
    """
    x = np.asarray(scenarios, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if support_size < 2:
        raise ValueError("support_size debe ser al menos dos")
    if support_size >= len(x):
        return x, p / p.sum()
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(x), size=support_size, replace=True, p=p / p.sum())
    return x[picked], np.full(support_size, 1.0 / support_size)


def calibration_report(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    observations: np.ndarray,
    labels: list[str],
    *,
    seed: int = 0,
    bins: int = 10,
    support_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostico PIT comparable para todos los modelos.

    Devuelve el detalle por activo y un resumen por modelo. El resumen agrega
    los contrastes por activo mediante Holm, controlando la familia de pruebas
    dentro de cada modelo en lugar de reportar quince p-valores sueltos.

    ``support_size`` iguala el numero de escenarios de todos los modelos antes
    de calcular el PIT. Es imprescindible para comparar calibracion entre
    modelos de distinta resolucion: bajo especificacion correcta, un ensemble
    de 500 puntos rechaza uniformidad cerca del 27% de las veces frente al 5%
    nominal, mientras que uno de 10.000 alcanza el nivel correcto. Sin igualar
    el soporte, el contraste mide cuantos escenarios tiene cada modelo antes que
    su calidad distributiva.
    """
    from mm_ipsa.evaluation.comparison import holm_adjust

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for position, (name, (scenarios, probabilities)) in enumerate(models.items()):
        native_size = len(scenarios)
        if support_size is not None:
            scenarios, probabilities = resample_support(
                scenarios, probabilities, support_size, seed + position
            )
        pit = randomized_pit(scenarios, probabilities, observations, seed + position)
        detail = pit_diagnostics(pit, labels, bins=bins)
        detail.insert(0, "model", name)
        detail["ks_pvalue_holm"] = holm_adjust(detail["ks_pvalue"])
        detail["rejects_uniformity_5pct"] = detail["ks_pvalue_holm"] < 0.05
        detail_frames.append(detail)

        summary_rows.append(
            {
                "model": name,
                "assets": len(labels),
                "native_support": int(native_size),
                "effective_support": int(len(scenarios)),
                "mean_dispersion_ratio": float(detail["dispersion_ratio"].mean()),
                "median_dispersion_ratio": float(detail["dispersion_ratio"].median()),
                "mean_reliability_index": float(detail["reliability_index"].mean()),
                "assets_underdispersed": int((detail["dispersion"] == "subdispersa").sum()),
                "assets_overdispersed": int((detail["dispersion"] == "sobredispersa").sum()),
                "assets_calibrated": int((detail["dispersion"] == "calibrada").sum()),
                "assets_rejecting_uniformity": int(
                    detail["rejects_uniformity_5pct"].sum()
                ),
                "pooled_pit_mean": float(np.mean(pit)),
            }
        )

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("mean_reliability_index")
    return detail_all, summary.reset_index(drop=True)
