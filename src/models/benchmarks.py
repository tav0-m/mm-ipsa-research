"""Benchmarks terminales comparables para evaluar Matching-Moment."""

from __future__ import annotations

import numpy as np


def nearest_psd(covariance: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Proyeccion espectral simetrica a una matriz definida positiva."""
    covariance = np.asarray(covariance, dtype=float)
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    return (eigenvectors * clipped) @ eigenvectors.T


def gaussian_terminal(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normal multivariada estimada directamente al horizonte H."""
    if n_scenarios <= 1:
        raise ValueError("n_scenarios debe ser mayor que uno")
    rng = np.random.default_rng(seed)
    scenarios = rng.multivariate_normal(mean, nearest_psd(covariance), size=n_scenarios)
    return scenarios, np.full(n_scenarios, 1.0 / n_scenarios)


def student_t_terminal(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_scenarios: int,
    degrees_of_freedom: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """t multivariada eliptica con covarianza exactamente parametrizada."""
    if degrees_of_freedom <= 4:
        raise ValueError("degrees_of_freedom debe ser >4 para cuarto momento finito")
    rng = np.random.default_rng(seed)
    scale = nearest_psd(covariance) * (degrees_of_freedom - 2.0) / degrees_of_freedom
    gaussian = rng.multivariate_normal(np.zeros(len(mean)), scale, size=n_scenarios)
    chi_square = rng.chisquare(degrees_of_freedom, size=n_scenarios)
    scenarios = np.asarray(mean) + gaussian / np.sqrt(chi_square[:, None] / degrees_of_freedom)
    return scenarios, np.full(n_scenarios, 1.0 / n_scenarios)


def historical_weighted(
    observations: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distribucion empirica ponderada sin ruido Monte Carlo adicional."""
    scenarios = np.asarray(observations, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if scenarios.ndim != 2 or probabilities.shape != (len(scenarios),):
        raise ValueError("Shapes incompatibles para el benchmark historico")
    if np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("Probabilidades historicas invalidas")
    probabilities = probabilities / probabilities.sum()
    return scenarios.copy(), probabilities


def generate_benchmarks(
    moments: np.ndarray,
    covariance: np.ndarray,
    historical: np.ndarray,
    historical_weights: np.ndarray,
    n_scenarios: int = 10_000,
    seed: int = 42,
    student_t_df: float = 6.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Genera el conjunto estandar de controles con identico target H."""
    mean = np.asarray(moments, dtype=float)[0]
    return {
        "gaussian_terminal": gaussian_terminal(mean, covariance, n_scenarios, seed),
        "student_t_terminal": student_t_terminal(
            mean, covariance, n_scenarios, student_t_df, seed + 1
        ),
        "historical_weighted": historical_weighted(historical, historical_weights),
    }
