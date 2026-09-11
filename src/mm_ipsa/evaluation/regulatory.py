"""Semaforo de Basilea y adecuacion de la muestra para clasificarlo.

El Comite de Basilea clasifica un modelo interno de VaR segun cuantas veces la
perdida realizada supero la prediccion en doscientas cincuenta jornadas al
noventa y nueve por ciento: hasta cuatro excepciones es zona verde, de cinco a
nueve amarilla, diez o mas roja. La zona determina un recargo sobre el
multiplicador de capital.

Esos numeros no son arbitrarios ni transportables. Salen de la binomial con
``n = 250`` y ``p = 0.01``, y trasladarlos tal cual a una muestra de otro tamano
cambia el error de tipo I sin avisar. Con ciento veinte observaciones el limite
de cuatro excepciones deja de significar lo que significa con doscientas
cincuenta.

Este modulo define las zonas por la probabilidad binomial acumulada, que es de
donde salieron, y se reduce exactamente a la tabla de Basilea cuando la muestra
es la suya. Ademas informa si la muestra alcanza para clasificar: con pocas
excepciones esperadas el semaforo carece de potencia y su color dice mas del
tamano muestral que del modelo.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.stats import binom

BASEL_OBSERVATIONS = 250
BASEL_LEVEL = 0.01

# Fronteras implicitas en la tabla original, leidas como probabilidad acumulada.
GREEN_UPPER_PROBABILITY = 0.95
YELLOW_UPPER_PROBABILITY = 0.9999

# Recargos del Comite sobre el multiplicador, definidos solo para su configuracion.
BASEL_YELLOW_ADD_ON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
BASEL_RED_ADD_ON = 1.00
BASEL_BASE_MULTIPLIER = 3.00

# Por debajo de este numero de excepciones esperadas el semaforo no discrimina.
MINIMUM_EXPECTED_EXCEPTIONS = 2.0


class TrafficLight(TypedDict):
    observations: int
    level: float
    exceptions: int
    expected_exceptions: float
    green_upper: int
    yellow_upper: int
    zone: str
    pvalue: float
    capital_multiplier: float | None
    sample_is_adequate: bool
    note: str


def zone_boundaries(observations: int, level: float) -> tuple[int, int]:
    """Mayor conteo de cada zona, derivado de la binomial acumulada.

    Devuelve ``(verde_max, amarillo_max)``. Reproduce ``(4, 9)`` en la
    configuracion del Comite, que es la comprobacion de que la generalizacion no
    inventa umbrales nuevos.
    """
    if observations < 1:
        raise ValueError("observations debe ser positivo")
    if not 0.0 < level < 1.0:
        raise ValueError("level debe pertenecer a (0, 1)")

    counts = np.arange(observations + 1)
    cumulative = binom.cdf(counts, observations, level)
    # Cada zona termina en el ultimo conteo que queda por debajo del umbral, de
    # modo que la siguiente empieza justo donde la acumulada lo alcanza.
    green = int(np.searchsorted(cumulative, GREEN_UPPER_PROBABILITY, "left")) - 1
    yellow = int(np.searchsorted(cumulative, YELLOW_UPPER_PROBABILITY, "left")) - 1
    green = min(max(green, 0), observations)
    return green, min(max(yellow, green), observations)


def basel_traffic_light(
    exceptions: int, observations: int, level: float = BASEL_LEVEL
) -> TrafficLight:
    """Clasifica un modelo de VaR y declara si la muestra permite hacerlo.

    ``pvalue`` es la cola superior binomial: la probabilidad de observar al menos
    tantas excepciones si el modelo fuera correcto.

    ``capital_multiplier`` se informa unicamente en la configuracion del Comite.
    Sus recargos estan calibrados para doscientas cincuenta jornadas al noventa y
    nueve por ciento, y trasladarlos a otra muestra seria inventar una tabla que
    el marco no define.
    """
    if exceptions < 0:
        raise ValueError("exceptions no puede ser negativo")
    if exceptions > observations:
        raise ValueError("exceptions no puede superar observations")

    green_upper, yellow_upper = zone_boundaries(observations, level)
    expected = observations * level

    if exceptions <= green_upper:
        zone = "verde"
    elif exceptions <= yellow_upper:
        zone = "amarilla"
    else:
        zone = "roja"

    multiplier: float | None = None
    if observations == BASEL_OBSERVATIONS and level == BASEL_LEVEL:
        if zone == "verde":
            multiplier = BASEL_BASE_MULTIPLIER
        elif zone == "amarilla":
            multiplier = BASEL_BASE_MULTIPLIER + BASEL_YELLOW_ADD_ON.get(exceptions, BASEL_RED_ADD_ON)
        else:
            multiplier = BASEL_BASE_MULTIPLIER + BASEL_RED_ADD_ON

    adequate = expected >= MINIMUM_EXPECTED_EXCEPTIONS
    if not adequate:
        note = (
            f"Muestra insuficiente: {expected:.2f} excepciones esperadas. "
            "El color refleja el tamano muestral mas que el modelo."
        )
    elif multiplier is None:
        note = (
            "Zonas reescaladas por binomial; el recargo de capital solo esta "
            f"definido para {BASEL_OBSERVATIONS} jornadas al "
            f"{1 - BASEL_LEVEL:.0%}."
        )
    else:
        note = "Configuracion del Comite."

    return {
        "observations": int(observations),
        "level": float(level),
        "exceptions": int(exceptions),
        "expected_exceptions": float(expected),
        "green_upper": green_upper,
        "yellow_upper": yellow_upper,
        "zone": zone,
        "pvalue": float(binom.sf(exceptions - 1, observations, level)),
        "capital_multiplier": multiplier,
        "sample_is_adequate": adequate,
        "note": note,
    }


def observations_for_power(
    level: float = BASEL_LEVEL,
    *,
    detectable_ratio: float = 2.0,
    power: float = 0.80,
    size: float = 0.05,
    maximum: int = 10_000,
) -> int:
    """Jornadas necesarias para detectar una tasa ``detectable_ratio`` veces mayor.

    Responde a la pregunta que el semaforo deja implicita: cuanta historia hace
    falta para que un modelo que subestima el riesgo al doble sea clasificado
    fuera de la zona verde con probabilidad ``power``.
    """
    if detectable_ratio <= 1.0:
        raise ValueError("detectable_ratio debe ser mayor que uno")
    alternative = min(level * detectable_ratio, 1.0)

    for observations in range(10, maximum + 1):
        critical = int(binom.isf(size, observations, level))
        if binom.sf(critical, observations, alternative) >= power:
            return observations
    return maximum
