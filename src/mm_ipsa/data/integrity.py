"""Integridad estructural de la serie de precios, previa a cualquier modelo.

Los controles existentes miden cobertura: cuantos datos faltan, cuanto se imputo,
que rachas de ausencia quedan. Son necesarios y no son suficientes. Un ajuste de
evento corporativo mal aplicado -un split que el proveedor no propago hacia atras-
produce una serie con cobertura perfecta y un salto de nivel espurio que ningun
control de cobertura puede ver. El retorno resultante entra al pipeline como si
fuera movimiento de mercado.

El protocolo declara desde la version 0.5.0 que Yahoo Finance es una fuente
conveniente y no un feed institucional, y que una fase posterior debe reconciliar
los eventos corporativos. La reconciliacion contra un segundo proveedor sigue
pendiente por falta de una fuente independiente que cubra este universo. Este
modulo cubre la parte que no requiere segunda fuente: detectar por consistencia
interna los patrones que delatan un ajuste fallido.

El discriminante util no es el tamano del salto. Un desplome de mercado y un
split no ajustado producen retornos igual de grandes. Lo que los separa es la
persistencia: un split mal aplicado desplaza el nivel de la serie de forma
permanente, mientras que un movimiento de mercado -por violento que sea- deja un
nivel que sigue evolucionando. Por eso cada candidato se evalua comparando el
nivel mediano antes y despues del salto, y no solo su magnitud.
"""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import pandas as pd


class CalendarSummary(TypedDict):
    rows: int
    duplicated_dates: int
    monotonic_increasing: bool
    weekend_rows: int
    largest_gap_days: int


class PriceAudit(TypedDict):
    calendar: CalendarSummary
    extreme_returns: pd.DataFrame
    corporate_action_candidates: pd.DataFrame
    stale_prices: pd.DataFrame
    blocking: list[str]
    passed: bool

# Razones de precio que produce un split o split inverso habitual. Un ajuste
# fallido deja el retorno pegado a una de estas, no en un valor cualquiera.
SPLIT_RATIOS = (
    0.1,
    0.2,
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
)

MAD_TO_SIGMA = 1.4826


def _as_date(value: object) -> str:
    """Fecha del indice en formato ISO, sin depender del tipo declarado."""
    return str(pd.Timestamp(cast(str, value)).date())


def robust_scale(values: np.ndarray) -> float:
    """Desviacion tipica estimada por MAD, inmune a los propios atipicos.

    La desviacion muestral usa los atipicos que se quieren detectar y por eso
    infla su propio umbral. La MAD no.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    deviation = np.median(np.abs(finite - np.median(finite)))
    return float(deviation * MAD_TO_SIGMA)


def extreme_returns(
    returns: pd.DataFrame, threshold: float = 8.0
) -> pd.DataFrame:
    """Retornos que exceden ``threshold`` escalas robustas respecto de su mediana.

    No es un detector de errores: en una muestra que contiene marzo de 2020 se
    espera que marque movimientos genuinos. Sirve para acotar el conjunto que
    merece inspeccion, no para descartar observaciones.
    """
    records: list[dict[str, object]] = []
    for asset in returns.columns:
        series = returns[asset].dropna()
        if series.empty:
            continue
        values = series.to_numpy()
        scale = robust_scale(values)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        center = float(np.median(values))
        scores = np.abs(values - center) / scale
        for position in np.flatnonzero(scores > threshold):
            records.append(
                {
                    "asset": asset,
                    "date": _as_date(series.index[position]),
                    "log_return": float(values[position]),
                    "robust_z": float(scores[position]),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=["asset", "date", "log_return", "robust_z"]
    )


def _level_persistence(
    log_prices: np.ndarray, position: int, window: int
) -> float:
    """Fraccion del salto que sobrevive como desplazamiento de nivel.

    Cerca de uno, el salto movio la serie y se quedo ahi, que es la firma de un
    ajuste fallido. Cerca de cero, la serie volvio, que es un movimiento de
    mercado con reversion.
    """
    jump = log_prices[position] - log_prices[position - 1]
    if jump == 0.0:
        return 0.0
    before = log_prices[max(0, position - window) : position]
    after = log_prices[position : position + window]
    if before.size == 0 or after.size == 0:
        return float("nan")
    return float((np.median(after) - np.median(before)) / jump)


CANDIDATE_COLUMNS = {
    "asset": "object",
    "date": "object",
    "ratio": "float64",
    "nearest_split_ratio": "float64",
    "distance_in_scales": "float64",
    "level_persistence": "float64",
    "suspected": "bool",
}


def corporate_action_candidates(
    prices: pd.DataFrame,
    *,
    scale_tolerance: float = 3.0,
    persistence_window: int = 10,
    minimum_persistence: float = 0.8,
) -> pd.DataFrame:
    """Saltos compatibles con un split no ajustado por el proveedor.

    Un candidato debe cumplir dos condiciones a la vez: que la razon de precios
    caiga cerca de una razon de split habitual, y que el salto persista como
    desplazamiento de nivel. La primera sola marca coincidencias numericas sin
    significado; la segunda las separa de la reversion de mercado.

    La cercania se mide en escalas robustas del propio retorno diario, no como
    porcentaje fijo. La razon es que un split no ajustado nunca produce el valor
    exacto: el precio observado incorpora tambien el movimiento de mercado de
    ese dia, de modo que un split dos por uno aparece como ``0.5 * exp(r_t)``.
    Una tolerancia fija en porcentaje es simultaneamente demasiado estrecha para
    un activo volatil y demasiado ancha para uno tranquilo; con un umbral de dos
    por ciento sobre una volatilidad diaria de uno y medio, uno de cada cinco
    splits reales pasaria sin marcarse. Expresarla en escalas de la distribucion
    del activo la vuelve comparable entre activos.

    El coste de ensanchar la banda es bajo porque el filtro de persistencia, y
    no la banda, es lo que separa un ajuste fallido de un movimiento de mercado.
    """
    if persistence_window < 2:
        raise ValueError("persistence_window debe ser al menos dos")
    if scale_tolerance <= 0.0:
        raise ValueError("scale_tolerance debe ser positivo")

    log_split_ratios = np.log(SPLIT_RATIOS)
    records: list[dict[str, object]] = []
    for asset in prices.columns:
        series = prices[asset].dropna()
        if len(series) < 2:
            continue
        values = series.to_numpy(dtype=float)
        if np.any(values <= 0.0):
            raise ValueError(f"{asset} contiene precios no positivos")

        log_prices = np.log(values)
        log_returns = np.diff(log_prices)
        scale = robust_scale(log_returns)
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        for offset, log_ratio in enumerate(log_returns):
            distances = np.abs(log_ratio - log_split_ratios) / scale
            best = int(np.argmin(distances))
            if distances[best] > scale_tolerance:
                continue
            position = offset + 1
            persistence = _level_persistence(
                log_prices, position, persistence_window
            )
            records.append(
                {
                    "asset": asset,
                    "date": _as_date(series.index[position]),
                    "ratio": float(np.exp(log_ratio)),
                    "nearest_split_ratio": float(SPLIT_RATIOS[best]),
                    "distance_in_scales": float(distances[best]),
                    "level_persistence": persistence,
                    "suspected": bool(
                        np.isfinite(persistence)
                        and persistence >= minimum_persistence
                    ),
                }
            )
    return pd.DataFrame.from_records(
        records, columns=list(CANDIDATE_COLUMNS)
    ).astype(CANDIDATE_COLUMNS)


def _longest_run(flags: np.ndarray) -> int:
    """Racha maxima de valores verdaderos consecutivos."""
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def stale_price_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Dias sin variacion de precio, por activo.

    Un precio que no se mueve porque no hubo transacciones no es informacion de
    mercado, pero entra al estimador como un retorno cero: sesga la volatilidad
    a la baja e induce autocorrelacion espuria en el horizonte compuesto. La
    magnitud pertenece al informe, no a una nota al pie.
    """
    records: list[dict[str, object]] = []
    for asset in prices.columns:
        series = prices[asset].dropna()
        if len(series) < 2:
            continue
        unchanged = (series.diff().to_numpy()[1:] == 0.0)
        records.append(
            {
                "asset": asset,
                "observations": int(len(series)),
                "unchanged_days": int(unchanged.sum()),
                "unchanged_share": float(unchanged.mean()),
                "longest_unchanged_run": _longest_run(unchanged),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "asset",
            "observations",
            "unchanged_days",
            "unchanged_share",
            "longest_unchanged_run",
        ],
    )


def calendar_integrity(prices: pd.DataFrame) -> CalendarSummary:
    """Comprobaciones del indice temporal, independientes de los valores."""
    index = cast(pd.DatetimeIndex, pd.DatetimeIndex(prices.index))
    gaps = index.to_series().diff().dt.days.dropna()
    return {
        "rows": int(len(index)),
        "duplicated_dates": int(index.duplicated().sum()),
        "monotonic_increasing": bool(index.is_monotonic_increasing),
        "weekend_rows": int((index.dayofweek >= 5).sum()),
        "largest_gap_days": int(gaps.max()) if not gaps.empty else 0,
    }


def audit_prices(
    prices: pd.DataFrame,
    *,
    extreme_threshold: float = 8.0,
    split_tolerance: float = 3.0,
) -> PriceAudit:
    """Reune las cuatro comprobaciones en un solo informe.

    ``blocking`` senala unicamente lo que invalida la serie: un indice temporal
    inconsistente o un salto que aparenta un split no ajustado. Los retornos
    extremos y los precios estancados se reportan sin bloquear, porque describen
    el mercado que se esta estudiando y no un defecto de la ingesta.
    """
    returns = cast(pd.DataFrame, np.log(prices / prices.shift(1))).dropna(how="all")
    calendar = calendar_integrity(prices)
    candidates = corporate_action_candidates(
        prices, scale_tolerance=split_tolerance
    )
    suspected = candidates.loc[candidates["suspected"]]

    blocking: list[str] = []
    if calendar["duplicated_dates"]:
        blocking.append(f"{calendar['duplicated_dates']} fechas duplicadas")
    if not calendar["monotonic_increasing"]:
        blocking.append("el indice no es monotono creciente")
    if calendar["weekend_rows"]:
        blocking.append(f"{calendar['weekend_rows']} filas en fin de semana")
    if len(suspected):
        detail = ", ".join(
            f"{row.asset} {row.date} ratio={row.ratio:.4f}"
            for row in suspected.itertuples()
        )
        blocking.append(f"posible split no ajustado: {detail}")

    return {
        "calendar": calendar,
        "extreme_returns": extreme_returns(returns, extreme_threshold),
        "corporate_action_candidates": candidates,
        "stale_prices": stale_price_report(prices),
        "blocking": blocking,
        "passed": not blocking,
    }
