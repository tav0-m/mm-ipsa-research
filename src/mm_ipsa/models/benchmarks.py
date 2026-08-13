"""Benchmarks terminales comparables para evaluar Matching-Moment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

# Cota inferior dura: el cuarto momento de una t existe solo si nu > 4, y el
# protocolo compara curtosis. Se deja margen porque la varianza del estimador
# explota cuando nu se aproxima a la frontera.
MIN_DEGREES_OF_FREEDOM = 4.5
MAX_DEGREES_OF_FREEDOM = 60.0


def nearest_psd(
    covariance: np.ndarray,
    floor: float = 1e-12,
    relative_floor: float = 1e-10,
) -> np.ndarray:
    """Proyeccion espectral simetrica a una matriz definida positiva.

    El piso efectivo es ``max(floor, relative_floor * lambda_max)``. Un piso
    puramente absoluto no regulariza matrices cuyas entradas son O(1e-4), como
    las covarianzas de retornos a cinco dias: recorta los autovalores negativos
    a casi cero y deja el numero de condicion practicamente intacto.
    """
    covariance = np.asarray(covariance, dtype=float)
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues.max())
    effective_floor = max(float(floor), relative_floor * largest) if largest > 0 else float(floor)
    clipped = np.maximum(eigenvalues, effective_floor)
    return (eigenvectors * clipped) @ eigenvectors.T


def student_t_log_likelihood(
    squared_distances: np.ndarray,
    weights: np.ndarray,
    log_determinant: float,
    dimension: int,
    degrees_of_freedom: float,
) -> float:
    """Log-verosimilitud ponderada de una t eliptica con covarianza fijada.

    ``squared_distances`` son las distancias de Mahalanobis respecto de la
    covarianza objetivo Sigma, no respecto de la matriz de escala. Al imponer
    ``S = Sigma (nu - 2) / nu`` los terminos en nu se simplifican y la
    dependencia queda concentrada en ``log(1 + d_t / (nu - 2))``, de modo que
    las distancias se calculan una sola vez fuera de la optimizacion.
    """
    nu = float(degrees_of_freedom)
    if nu <= 2.0:
        return -np.inf
    p = int(dimension)
    shifted = nu - 2.0
    constant = (
        gammaln((nu + p) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * p * np.log(np.pi * shifted)
        - 0.5 * log_determinant
    )
    quadratic = -0.5 * (nu + p) * np.log1p(squared_distances / shifted)
    return float(np.sum(weights * (constant + quadratic)))


def estimate_student_t_df(
    observations: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray | None = None,
    bounds: tuple[float, float] = (MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM),
) -> dict[str, float]:
    """Estima nu por verosimilitud de perfil dejando fijas media y covarianza.

    Solo se ajusta el parametro de cola. La media y la covarianza se mantienen
    en los mismos targets EWMA que recibe Matching-Moment, de modo que el
    control conserva exactamente el mismo conjunto de informacion y la
    comparacion no se contamina con una estimacion distinta de los dos primeros
    momentos.

    Las ventanas terminales rolling se solapan, por lo que esto es una
    pseudo-verosimilitud: el punto estimado es consistente como M-estimador,
    pero sus errores estandar nominales subestimarian la incertidumbre y no se
    reportan como inferencia.
    """
    x = np.asarray(observations, dtype=float)
    mu = np.asarray(mean, dtype=float)
    if x.ndim != 2:
        raise ValueError("observations debe ser una matriz (T, n)")
    T, p = x.shape
    if mu.shape != (p,):
        raise ValueError("mean debe tener shape (n,)")
    if T <= p:
        raise ValueError("Se requieren mas observaciones que activos para estimar nu")
    low, high = float(bounds[0]), float(bounds[1])
    if not 2.0 < low < high:
        raise ValueError("bounds debe cumplir 2 < low < high")

    if weights is None:
        w = np.full(T, 1.0 / T)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (T,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights debe ser no negativo, no nulo y de largo T")
        w = w / w.sum()

    sigma = nearest_psd(covariance)
    cholesky = np.linalg.cholesky(sigma)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    # Resolver el sistema triangular es mas estable que invertir Sigma.
    solved = np.linalg.solve(cholesky, (x - mu).T)
    squared_distances = np.sum(solved**2, axis=0)

    def negative(nu: float) -> float:
        value = student_t_log_likelihood(squared_distances, w, log_determinant, p, nu)
        return -value if np.isfinite(value) else np.inf

    result = minimize_scalar(negative, bounds=(low, high), method="bounded",
                             options={"xatol": 1e-4})
    if not result.success:
        raise RuntimeError(f"La estimacion de nu no convergio: {result.message}")

    estimated = float(result.x)
    log_likelihood = -float(result.fun)
    # Referencia gaussiana: nu -> infinito. Se evalua en la cota superior para
    # cuantificar cuanta evidencia de cola aporta realmente el control t.
    gaussian_reference = student_t_log_likelihood(
        squared_distances, w, log_determinant, p, high
    )
    at_bound = bool(estimated >= high - 1e-3 or estimated <= low + 1e-3)
    return {
        "degrees_of_freedom": estimated,
        "log_likelihood": log_likelihood,
        "log_likelihood_at_upper_bound": float(gaussian_reference),
        "log_likelihood_gain": float(log_likelihood - gaussian_reference),
        "effective_observations": float(1.0 / np.sum(w**2)),
        "at_bound": float(at_bound),
        "bound_low": low,
        "bound_high": high,
    }


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


def resolve_student_t_df(
    benchmark_cfg: dict,
    observations: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Resuelve nu segun configuracion y devuelve el diagnostico auditable.

    El modo ``mle`` estima nu con la informacion del origen correspondiente; el
    modo ``fixed`` conserva el valor declarado. En ambos casos se retorna la
    procedencia para que el linaje registre si el control t fue calibrado o
    impuesto.
    """
    mode = str(benchmark_cfg.get("student_t_df_mode", "mle"))
    if mode not in {"mle", "fixed"}:
        raise ValueError("student_t_df_mode debe ser mle o fixed")

    if mode == "fixed":
        value = float(benchmark_cfg["student_t_df"])
        if value <= 4.0:
            raise ValueError("student_t_df debe ser >4 para cuarto momento finito")
        return value, {"student_t_df": value, "student_t_df_mode": "fixed"}

    raw_bounds = benchmark_cfg.get(
        "student_t_df_bounds", [MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM]
    )
    bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
    diagnostics = estimate_student_t_df(observations, mean, covariance, weights, bounds)
    value = float(diagnostics["degrees_of_freedom"])
    report: dict[str, float | str] = {"student_t_df_mode": "mle"}
    report.update({f"student_t_{key}": val for key, val in diagnostics.items()})
    report["student_t_df"] = value
    return value, report


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
