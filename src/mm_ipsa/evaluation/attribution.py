"""Descomposicion del Expected Shortfall por posicion.

Los contrastes de calibracion responden si el ES predicho acerto en magnitud.
No responden si acerto en composicion. Un modelo puede anticipar correctamente
cuanto se pierde en la cola y equivocarse por completo sobre que posiciones la
producen, y para una funcion de riesgo esa segunda pregunta es la accionable:
determina que se cubre.

El ES es positivamente homogeneo de grado uno en los pesos, de modo que el
teorema de Euler lo descompone de forma exacta y unica::

    ES(w) = sum_i w_i * d ES / d w_i = sum_i w_i * E[X_i | R <= VaR(R)]

Tasche mostro que esta es la unica asignacion compatible con el ordenamiento por
desempeno ajustado a riesgo, y es la que usa la practica bajo el nombre de
contribucion componente. Cada sumando es lo que esa posicion aporta a la cola de
la cartera, no su riesgo aislado: una posicion volatil que se mueve en contra del
resto puede contribuir negativamente.

La aditividad exacta no es automatica en una distribucion discreta. Requiere que
la cola se pondere con la misma masa fraccionaria en el cuantil que usa el ES
agregado, y este modulo la reutiliza para que la descomposicion cierre por
construccion.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.stats import spearmanr


class ComponentShortfall(TypedDict):
    alpha: float
    total: float
    components: np.ndarray
    shares: np.ndarray
    conditional: np.ndarray
    effective_tail_mass: float
    tail_observations: int


class AttributionComparison(TypedDict):
    composition_error: float
    conditional_rank_correlation: float
    share_rank_correlation: float
    largest_predicted: int
    largest_realised: int
    identifies_main_driver: bool
    realised_tail_observations: int
    reliable: bool


# Con menos eventos de cola que esto la composicion realizada es ruido.
MINIMUM_TAIL_OBSERVATIONS = 10


def tail_weights(
    losses: np.ndarray, probabilities: np.ndarray, alpha: float
) -> np.ndarray:
    """Reparto de la masa ``alpha`` sobre los peores escenarios.

    El escenario que cruza el cuantil recibe solo la fraccion que cabe, de modo
    que la masa sumada es exactamente ``alpha`` aunque el soporte sea discreto.
    Sin ese reparto la descomposicion de Euler no cierra.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha debe pertenecer a (0, 1]")

    values = np.asarray(losses, dtype=float).ravel()
    weights = np.asarray(probabilities, dtype=float).ravel()
    if values.size != weights.size:
        raise ValueError("losses y probabilities deben tener el mismo tamano")
    if np.any(weights < 0.0):
        raise ValueError("probabilities contiene valores negativos")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("probabilities suma cero")
    weights = weights / total

    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    taken = np.clip(alpha - (cumulative - weights[order]), 0.0, weights[order])

    allocated = np.zeros_like(values)
    allocated[order] = taken
    return allocated


def component_expected_shortfall(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> ComponentShortfall:
    """Contribucion de cada activo al ES de la cartera.

    ``shares`` expresa cada contribucion como fraccion del total. Suman uno por
    la aditividad de Euler, y una fraccion negativa senala una posicion que
    reduce la cola de la cartera en vez de alimentarla.
    """
    x = np.asarray(scenarios, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] != w.size:
        raise ValueError("scenarios debe tener una columna por peso")

    portfolio = x @ w
    allocated = tail_weights(portfolio, probabilities, alpha)
    mass = float(allocated.sum())
    if mass <= 0.0:
        raise ValueError("La cola no recibio masa de probabilidad")

    conditional = (allocated @ x) / mass
    components = w * conditional
    total = float(components.sum())
    if total == 0.0:
        raise ValueError("El ES de la cartera es cero; no admite descomposicion")

    return {
        "alpha": float(alpha),
        "total": total,
        "components": components,
        "shares": components / total,
        "conditional": conditional,
        "effective_tail_mass": mass,
        "tail_observations": int(np.count_nonzero(allocated)),
    }


def realised_component_shortfall(
    observations: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> ComponentShortfall:
    """Descomposicion sobre lo efectivamente observado.

    Cada observacion pesa igual, de modo que la cola son las peores ``alpha`` por
    ciento de las jornadas realizadas. Con pocas observaciones la composicion
    resultante es ruidosa, y ``tail_observations`` permite verificarlo antes de
    leerla.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2:
        raise ValueError("observations debe ser bidimensional")
    uniform = np.full(len(y), 1.0 / len(y))
    return component_expected_shortfall(y, uniform, weights, alpha)


def compare_attribution(
    predicted: ComponentShortfall, realised: ComponentShortfall
) -> AttributionComparison:
    """Contrasta la composicion predicha de la cola contra la observada.

    ``composition_error`` es la mitad de la distancia absoluta entre ambas
    reparticiones, que con fracciones no negativas se lee como la porcion de la
    cola atribuida a las posiciones equivocadas.

    ``conditional_rank_correlation`` mide el acierto del modelo sin el peso de la
    cartera. Es la correlacion entre las esperanzas condicionales predichas y las
    observadas, que es lo unico que el modelo aporta: los pesos son identicos en
    ambas descomposiciones.

    ``share_rank_correlation`` se reporta aparte porque esta confundida con la
    concentracion. Al compartir el vector de pesos, una cartera concentrada
    produce correlacion cercana a uno aunque el modelo no acierte nada sobre el
    comportamiento de cola. Sirve para describir la cartera, no para evaluar el
    modelo.
    """
    predicted_shares = predicted["shares"]
    realised_shares = realised["shares"]
    if predicted_shares.size != realised_shares.size:
        raise ValueError("Las descomposiciones tienen distinto numero de activos")

    error = 0.5 * float(np.abs(predicted_shares - realised_shares).sum())

    def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
        if first.size < 3:
            return float("nan")
        # Los stubs de scipy declaran un retorno generico; el primer elemento es
        # el coeficiente en todas las formas de llamada de un solo par.
        return float(np.asarray(spearmanr(first, second))[0])

    conditional_correlation = _rank_correlation(
        predicted["conditional"], realised["conditional"]
    )
    share_correlation = _rank_correlation(predicted_shares, realised_shares)

    largest_predicted = int(np.argmax(predicted_shares))
    largest_realised = int(np.argmax(realised_shares))
    observations = realised["tail_observations"]

    return {
        "composition_error": error,
        "conditional_rank_correlation": conditional_correlation,
        "share_rank_correlation": share_correlation,
        "largest_predicted": largest_predicted,
        "largest_realised": largest_realised,
        "identifies_main_driver": largest_predicted == largest_realised,
        "realised_tail_observations": observations,
        "reliable": observations >= MINIMUM_TAIL_OBSERVATIONS,
    }
