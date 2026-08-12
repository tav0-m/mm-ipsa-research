"""Optimizadores con baselines ingenuos y controles de concentracion."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from src.evaluation.scoring import lower_tail_mean
from src.models.benchmarks import nearest_psd


def weighted_mean_cov(scenarios: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scenarios, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    mean = p @ x
    dev = x - mean
    covariance = (p[:, None] * dev).T @ dev
    return mean, nearest_psd(covariance)


def equal_weight(n_assets: int) -> np.ndarray:
    if n_assets <= 0:
        raise ValueError("n_assets debe ser positivo")
    return np.full(n_assets, 1.0 / n_assets)


def inverse_variance(covariance: np.ndarray) -> np.ndarray:
    variance = np.maximum(np.diag(covariance), 1e-15)
    weights = 1.0 / variance
    return weights / weights.sum()


def _cluster_variance(covariance: np.ndarray, indices: list[int]) -> float:
    sub = covariance[np.ix_(indices, indices)]
    weights = inverse_variance(sub)
    return float(weights @ sub @ weights)


def hierarchical_risk_parity(covariance: np.ndarray) -> np.ndarray:
    """HRP long-only mediante single linkage y biseccion recursiva."""
    covariance = nearest_psd(covariance)
    standard = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    correlation = covariance / np.outer(standard, standard)
    correlation = np.clip(correlation, -1.0, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    order = leaves_list(linkage(squareform(distance, checks=False), method="single")).tolist()
    weights = np.ones(len(order), dtype=float)
    clusters = [order]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_left = _cluster_variance(covariance, left)
            var_right = _cluster_variance(covariance, right)
            allocation_left = 1.0 - var_left / (var_left + var_right)
            weights[left] *= allocation_left
            weights[right] *= 1.0 - allocation_left
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights / weights.sum()


def _feasible_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, max_weight)
    # Proyeccion iterativa simple porque n es pequeno y sum(max_weight)>=1.
    for _ in range(100):
        deficit = 1.0 - weights.sum()
        if abs(deficit) < 1e-12:
            break
        free = weights < max_weight - 1e-12 if deficit > 0 else weights > 1e-12
        if not np.any(free):
            raise ValueError("max_weight hace inviable el presupuesto")
        weights[free] += deficit / int(free.sum())
        weights = np.clip(weights, 0.0, max_weight)
    return weights / weights.sum()


def minimum_variance(
    covariance: np.ndarray,
    max_weight: float = 1.0,
    l2_penalty: float = 0.0,
) -> np.ndarray:
    covariance = nearest_psd(covariance)
    n = len(covariance)
    reference = equal_weight(n)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights + l2_penalty * np.sum((weights - reference) ** 2))

    result = minimize(
        objective,
        _feasible_weights(reference, max_weight),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Minimum variance fallo: {result.message}")
    return _feasible_weights(result.x, max_weight)


def maximum_sharpe(
    mean: np.ndarray,
    covariance: np.ndarray,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.0,
    l2_penalty: float = 0.0,
    n_starts: int = 20,
    seed: int = 42,
) -> np.ndarray:
    covariance = nearest_psd(covariance)
    mean = np.asarray(mean, dtype=float)
    n = len(mean)
    reference = equal_weight(n)

    def objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(float(weights @ covariance @ weights), 1e-15))
        sharpe = (float(weights @ mean) - risk_free_rate) / volatility
        return -sharpe + l2_penalty * float(np.sum((weights - reference) ** 2))

    rng = np.random.default_rng(seed)
    starts = [reference] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]
    best = None
    for start in starts:
        result = minimize(
            objective,
            _feasible_weights(start, max_weight),
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"ftol": 1e-11, "maxiter": 500},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("Maximum Sharpe no encontro una solucion factible")
    return _feasible_weights(best.x, max_weight)


def minimum_cvar(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    alpha: float = 0.05,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Minimiza CVaR de perdidas con el LP de Rockafellar-Uryasev."""
    x = np.asarray(scenarios, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    N, n = x.shape
    variables = n + 1 + N  # pesos, umbral eta, excesos
    objective = np.zeros(variables)
    objective[n] = 1.0
    objective[n + 1 :] = p / alpha
    inequalities = np.zeros((N, variables))
    inequalities[:, :n] = -x
    inequalities[:, n] = -1.0
    inequalities[np.arange(N), n + 1 + np.arange(N)] = -1.0
    equality = np.zeros((1, variables))
    equality[0, :n] = 1.0
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(N),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * N,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Minimum CVaR fallo: {result.message}")
    return _feasible_weights(result.x[:n], max_weight)


def portfolio_diagnostics(
    weights: np.ndarray,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    mean, covariance = weighted_mean_cov(scenarios, probabilities)
    returns = np.asarray(scenarios) @ weights
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    volatility = np.sqrt(max(float(weights @ covariance @ weights), 0.0))
    return {
        "expected_return": float(weights @ mean),
        "volatility": float(volatility),
        "sharpe": float(weights @ mean / max(volatility, 1e-15)),
        "lower_es": lower_tail_mean(returns, p, alpha),
        "herfindahl": float(np.sum(weights**2)),
        "effective_assets": float(1.0 / np.sum(weights**2)),
        "max_weight": float(np.max(weights)),
    }
