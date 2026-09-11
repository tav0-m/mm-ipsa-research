"""Contraste de Expected Shortfall por simulacion, segun Acerbi y Szekely (2014).

El pipeline calcula ES desde la version inicial y nunca lo contrasta. Para VaR
existen Kupiec, Christoffersen y el test conjunto de cobertura condicional; para
ES no habia nada. La asimetria importa porque Basilea III, en su revision del
marco de riesgo de mercado, sustituyo VaR por Expected Shortfall como medida
regulatoria: contrastar solo VaR valida la medida que la industria dejo atras.

El obstaculo historico fue que ES no es *elicitable*: no existe una funcion de
perdida cuyo minimizador sea ES, de modo que no admite un backtest por conteo de
excedencias como el de VaR. Acerbi y Szekely resuelven el problema por otra via:
construyen estadisticos con esperanza nula bajo la hipotesis de que la predictiva
es correcta, y obtienen su distribucion simulando desde esa misma predictiva.

Aqui esa simulacion es directa. La predictiva del proyecto ya es una
distribucion discreta -el conjunto de escenarios con sus probabilidades-, de modo
que muestrear bajo la hipotesis nula es remuestrear los escenarios.

Convencion de signos
--------------------
El proyecto expresa VaR y ES en espacio de retornos, como cuantiles negativos, y
no como perdidas positivas. Los estadisticos se definen de modo que conserven la
lectura de la literatura original:

``Z < 0`` el riesgo esta subestimado; las perdidas de cola realizadas superaron
a las predichas. Es la direccion que importa para un regulador.

``Z > 0`` el riesgo esta sobreestimado, lo que es conservador y no penalizable.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from mm_ipsa.evaluation.scoring import lower_tail_mean, weighted_quantile


class ShortfallBacktest(TypedDict):
    alpha: float
    observations: int
    exceedances: int
    exceedance_rate: float
    var: float
    predicted_es: float
    realised_tail_mean: float
    z1: float
    pvalue_z1: float
    z2: float
    pvalue_z2: float
    simulations_used_z1: int
    verdict: str


def _sample_indices(
    probabilities: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices muestreados de una discreta, por inversion de la acumulada.

    Mas rapido que ``rng.choice`` cuando el numero de extracciones supera con
    holgura el tamano del soporte, que es el caso en cualquier simulacion util.
    """
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, rng.random(shape), side="right")


def _z1_statistic(
    sample: np.ndarray, var: float, expected_shortfall: float
) -> np.ndarray:
    """Estadistico condicional a que haya ocurrido una excedencia.

    Compara la media de las perdidas que excedieron VaR contra el ES predicho.
    Queda indefinido cuando no hubo ninguna excedencia, y esos casos se propagan
    como ``nan`` para excluirlos de la nula.
    """
    breaches = sample < var
    counts = breaches.sum(axis=-1)
    totals = np.where(breaches, sample, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        average = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return 1.0 - average / expected_shortfall


def _z2_statistic(
    sample: np.ndarray, var: float, expected_shortfall: float, alpha: float
) -> np.ndarray:
    """Estadistico incondicional, definido siempre.

    Bajo la predictiva correcta la masa esperada de la cola es ``T * alpha * ES``.
    A diferencia de ``Z1`` no necesita que hayan ocurrido excedencias, de modo que
    conserva potencia en muestras cortas o colas muy finas.
    """
    breaches = sample < var
    totals = np.where(breaches, sample, 0.0).sum(axis=-1)
    horizon = sample.shape[-1]
    return 1.0 - totals / (horizon * alpha * expected_shortfall)


class DisjointShortfallBacktest(TypedDict):
    horizon: int
    subsamples: list[ShortfallBacktest]
    worst_pvalue_z1: float
    worst_pvalue_z2: float
    median_pvalue_z2: float
    verdicts: dict[str, int]
    consistent: bool


def disjoint_subsample_backtests(
    observations: np.ndarray,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    horizon: int,
    *,
    n_simulations: int = 10_000,
    seed: int = 0,
) -> DisjointShortfallBacktest:
    """Contrasta ES sobre ventanas solapadas, separandolas en submuestras validas.

    Un retorno terminal a ``horizon`` dias calculado cada dia comparte ``horizon
    - 1`` jornadas con su vecino. Los estadisticos de Acerbi y Szekely suponen
    observaciones independientes, y aplicarlos directamente a esa serie produce
    valores p muy por debajo de los correctos: la muestra efectiva es del orden
    de ``T / horizon``, no de ``T``.

    Tomar una de cada ``horizon`` observaciones restaura la independencia, pero
    el desplazamiento inicial es arbitrario y existen ``horizon`` submuestras
    igualmente legitimas. En vez de elegir una, se contrastan todas y se reporta
    el rango. Si el veredicto cambia segun el desplazamiento, la evidencia no es
    concluyente, y esa es informacion que una sola submuestra ocultaria.
    """
    if horizon < 1:
        raise ValueError("horizon debe ser positivo")

    series = np.asarray(observations, dtype=float).ravel()
    if series.size < horizon:
        raise ValueError("La serie es mas corta que el horizonte")

    subsamples = [
        expected_shortfall_backtest(
            series[offset::horizon],
            scenarios,
            probabilities,
            alpha,
            n_simulations=n_simulations,
            seed=seed + offset,
        )
        for offset in range(horizon)
    ]

    z1_values = [r["pvalue_z1"] for r in subsamples if np.isfinite(r["pvalue_z1"])]
    z2_values = [r["pvalue_z2"] for r in subsamples]
    verdicts: dict[str, int] = {}
    for report in subsamples:
        verdicts[report["verdict"]] = verdicts.get(report["verdict"], 0) + 1

    return {
        "horizon": int(horizon),
        "subsamples": subsamples,
        "worst_pvalue_z1": float(min(z1_values)) if z1_values else float("nan"),
        "worst_pvalue_z2": float(min(z2_values)),
        "median_pvalue_z2": float(np.median(z2_values)),
        "verdicts": verdicts,
        "consistent": len(verdicts) == 1,
    }


def _monte_carlo_pvalue(null: np.ndarray, observed: float) -> float:
    """Valor p de una cola con la correccion habitual de Monte Carlo.

    La forma ingenua ``mean(null <= observed)`` es sesgada cuando la nula es
    discreta, porque los empates cuentan entero. Contar el propio estadistico
    observado como una realizacion mas de la nula, ``(1 + k) / (1 + M)``, corrige
    el sesgo y garantiza que el valor p nunca sea cero.
    """
    return float((1 + np.count_nonzero(null <= observed)) / (1 + null.size))


# Dos contrastes sobre la misma predictiva, de modo que el umbral por estadistico
# se reparte para que el error de tipo I conjunto siga siendo del cinco por ciento.
SHORTFALL_LEVEL = 0.05
PER_STATISTIC_LEVEL = SHORTFALL_LEVEL / 2.0


def _verdict(pvalue_z1: float, pvalue_z2: float) -> str:
    """Lectura conjunta de ambos estadisticos.

    Ninguno domina al otro. ``Z2`` detecta mejor una predictiva sistematicamente
    estrecha; ``Z1`` detecta mejor una cola mas pesada de lo predicho, porque esa
    produce excedencias menos frecuentes pero mas profundas y los dos efectos se
    compensan en el estadistico incondicional. Un veredicto basado en uno solo
    dejaria pasar justo el modo de fallo que el otro cubre.
    """
    finite = [value for value in (pvalue_z1, pvalue_z2) if np.isfinite(value)]
    if not finite:
        return "indeterminado"
    if min(finite) < PER_STATISTIC_LEVEL:
        return "riesgo subestimado"
    if min(finite) > 1.0 - PER_STATISTIC_LEVEL:
        return "riesgo sobreestimado"
    return "compatible"


def expected_shortfall_backtest(
    observations: np.ndarray,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    alpha: float = 0.05,
    *,
    n_simulations: int = 10_000,
    seed: int = 0,
) -> ShortfallBacktest:
    """Contrasta el ES de una predictiva discreta contra lo realizado.

    ``scenarios`` y ``probabilities`` describen la predictiva de un activo o de
    una cartera; ``observations`` son los retornos realizados que se le enfrentan.

    El valor p es de una cola y responde a la pregunta que importa en riesgo:
    cual es la probabilidad de observar un estadistico al menos tan adverso como
    el realizado si la predictiva fuera correcta. Un valor pequeno indica que el
    modelo subestima la cola.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe pertenecer a (0, 1)")
    if n_simulations < 1:
        raise ValueError("n_simulations debe ser positivo")

    realised = np.asarray(observations, dtype=float).ravel()
    support = np.asarray(scenarios, dtype=float).ravel()
    if realised.size == 0:
        raise ValueError("Se requiere al menos una observacion")
    if support.size == 0:
        raise ValueError("La predictiva no tiene soporte")

    weights = np.asarray(probabilities, dtype=float).ravel()
    if weights.size != support.size:
        raise ValueError("probabilities debe tener un valor por escenario")
    if np.any(weights < 0.0):
        raise ValueError("probabilities contiene valores negativos")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("probabilities suma cero")
    weights = weights / total

    var = weighted_quantile(support, weights, alpha)
    predicted = lower_tail_mean(support, weights, alpha)
    if not np.isfinite(predicted) or predicted >= 0.0:
        raise ValueError(
            "El ES predicho debe ser negativo en espacio de retornos; "
            f"recibido {predicted}"
        )

    breaches = realised < var
    exceedances = int(breaches.sum())
    realised_tail = (
        float(realised[breaches].mean()) if exceedances else float("nan")
    )

    observed_z1 = float(_z1_statistic(realised, var, predicted))
    observed_z2 = float(_z2_statistic(realised, var, predicted, alpha))

    rng = np.random.default_rng(seed)
    draws = support[
        _sample_indices(weights, (n_simulations, realised.size), rng)
    ]
    null_z1 = _z1_statistic(draws, var, predicted)
    null_z2 = _z2_statistic(draws, var, predicted, alpha)

    usable = np.isfinite(null_z1)
    if np.isfinite(observed_z1) and usable.any():
        pvalue_z1 = _monte_carlo_pvalue(null_z1[usable], observed_z1)
    else:
        pvalue_z1 = float("nan")
    pvalue_z2 = _monte_carlo_pvalue(null_z2, observed_z2)

    verdict = _verdict(pvalue_z1, pvalue_z2)

    return {
        "alpha": float(alpha),
        "observations": int(realised.size),
        "exceedances": exceedances,
        "exceedance_rate": float(exceedances / realised.size),
        "var": float(var),
        "predicted_es": float(predicted),
        "realised_tail_mean": realised_tail,
        "z1": observed_z1,
        "pvalue_z1": pvalue_z1,
        "z2": observed_z2,
        "pvalue_z2": pvalue_z2,
        "simulations_used_z1": int(usable.sum()),
        "verdict": verdict,
    }
