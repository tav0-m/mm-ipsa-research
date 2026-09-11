"""Concentracion de los excesos de riesgo segun el regimen de volatilidad.

Un modelo de riesgo puede estar bien calibrado en promedio y fallar justo cuando
importa. Es la critica central a los modelos internos: se estiman sobre historia
reciente, de modo que en calma producen limites laxos y cuando llega la tension
los excesos se acumulan en el peor momento. El marco regulatorio responde a esa
propiedad exigiendo una calibracion sobre periodo de estres, precisamente porque
la calibracion corriente no basta.

El diagnostico aqui no construye un estres sintetico: lo busca en los datos. El
periodo de evaluacion contiene un cambio de regimen al alza, de modo que la
pregunta se puede contestar directamente. Si los excesos se reparten en
proporcion al numero de ventanas de cada regimen, el modelo es aciclico. Si se
concentran en el regimen tensionado, es prociclico, y su ES describe la calma en
vez del riesgo.

La clasificacion usa unicamente informacion anterior al inicio de cada ventana.
Un retorno terminal a ``H`` jornadas fechado en ``t`` cubre ``[t - H + 1, t]``,
por lo que la volatilidad que lo clasifica se mide sobre jornadas anteriores a
``t - H + 1``. Sin ese desfase la clasificacion usaria el propio movimiento que
pretende anticipar.
"""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CALM = "calmado"
STRESSED = "tensionado"


class RegimeBreaches(TypedDict):
    regime: str
    windows: int
    breaches: int
    breach_rate: float
    mean_depth: float


class ProcyclicalityTest(TypedDict):
    var: float
    alpha: float
    calm: RegimeBreaches
    stressed: RegimeBreaches
    rate_ratio: float
    pvalue: float
    concentrates_in_stress: bool
    total_breaches: int
    reliable: bool


# Con menos excesos que esto la tabla de contingencia no discrimina.
MINIMUM_BREACHES = 8


def trailing_volatility(
    daily: pd.DataFrame,
    window_ends: pd.DatetimeIndex,
    horizon: int,
    lookback: int = 63,
) -> pd.Series:
    """Volatilidad realizada previa al inicio de cada ventana terminal.

    Devuelve ``NaN`` mientras no existan ``lookback`` jornadas anteriores, de modo
    que las primeras ventanas quedan sin clasificar en vez de clasificarse con
    informacion incompleta.
    """
    if horizon < 1:
        raise ValueError("horizon debe ser positivo")
    if lookback < 2:
        raise ValueError("lookback debe ser al menos dos")

    index = pd.DatetimeIndex(daily.index)
    portfolio = daily.mean(axis=1).to_numpy()
    values: list[float] = []

    for end in window_ends:
        position = int(cast(int, index.searchsorted(end, side="right")))
        start = position - horizon
        if start - lookback < 0:
            values.append(float("nan"))
            continue
        history = portfolio[start - lookback : start]
        values.append(float(np.std(history, ddof=1)))

    return pd.Series(values, index=window_ends, name="trailing_volatility")


def classify_regimes(volatility: pd.Series, quantile: float = 0.75) -> pd.Series:
    """Etiqueta cada ventana como calmada o tensionada.

    El umbral es el cuantil muestral de la propia volatilidad previa. Es una
    particion descriptiva y no una regla operable: separa el periodo en dos
    estados comparables, sin pretender que el umbral pudiera haberse fijado de
    antemano.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile debe pertenecer a (0, 1)")
    finite = volatility.dropna()
    if finite.empty:
        raise ValueError("No hay volatilidad previa suficiente para clasificar")

    threshold = float(finite.quantile(quantile))
    labels = pd.Series(pd.NA, index=volatility.index, dtype="object")
    labels[volatility <= threshold] = CALM
    labels[volatility > threshold] = STRESSED
    labels[volatility.isna()] = pd.NA
    return labels


def _summarise(
    losses: np.ndarray, var: float, regime: str
) -> RegimeBreaches:
    breached = losses < var
    count = int(breached.sum())
    depth = float((var - losses[breached]).mean()) if count else float("nan")
    return {
        "regime": regime,
        "windows": int(losses.size),
        "breaches": count,
        "breach_rate": float(count / losses.size) if losses.size else float("nan"),
        "mean_depth": depth,
    }


def breach_concentration(
    observations: np.ndarray,
    regimes: pd.Series,
    var: float,
    alpha: float,
) -> ProcyclicalityTest:
    """Contrasta si los excesos se concentran en el regimen tensionado.

    ``var`` es el limite que el modelo predijo una sola vez, sin recalibrar, que
    es la situacion que el diagnostico quiere examinar. La prueba exacta de
    Fisher sobre la tabla de dos por dos evita la aproximacion asintotica, que
    con pocos excesos no es fiable.
    """
    series = np.asarray(observations, dtype=float).ravel()
    labels = regimes.to_numpy()
    if series.size != labels.size:
        raise ValueError("observations y regimes deben tener el mismo tamano")

    calm_mask = labels == CALM
    stressed_mask = labels == STRESSED
    if not calm_mask.any() or not stressed_mask.any():
        raise ValueError("Se requieren ventanas de ambos regimenes")

    calm = _summarise(series[calm_mask], var, CALM)
    stressed = _summarise(series[stressed_mask], var, STRESSED)

    table = [
        [stressed["breaches"], stressed["windows"] - stressed["breaches"]],
        [calm["breaches"], calm["windows"] - calm["breaches"]],
    ]
    # Los stubs de scipy declaran un retorno generico; el segundo elemento
    # es el valor p en todas las formas de llamada.
    pvalue = float(np.asarray(fisher_exact(table, alternative="greater"))[1])

    ratio = (
        stressed["breach_rate"] / calm["breach_rate"]
        if calm["breach_rate"] > 0.0
        else float("inf")
    )
    total = stressed["breaches"] + calm["breaches"]

    return {
        "var": float(var),
        "alpha": float(alpha),
        "calm": calm,
        "stressed": stressed,
        "rate_ratio": float(ratio),
        "pvalue": pvalue,
        "concentrates_in_stress": pvalue < 0.05,
        "total_breaches": total,
        "reliable": total >= MINIMUM_BREACHES,
    }
