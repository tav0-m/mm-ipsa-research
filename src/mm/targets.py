"""Estimacion reproducible de momentos y dependencia para Matching-Moment.

La parametrizacion temporal se expresa en unidades interpretables. Una vida
media de ``h`` semanas aplicada a retornos rolling diarios usa
``lambda = 0.5 ** (1 / (h * observaciones_por_semana))``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def decay_from_half_life(
    half_life_weeks: float,
    observations_per_week: float = 5.0,
) -> float:
    """Convierte vida media semanal en el factor de decaimiento por fila."""
    if half_life_weeks <= 0 or observations_per_week <= 0:
        raise ValueError("La vida media y observaciones_por_semana deben ser positivas")
    return float(0.5 ** (1.0 / (half_life_weeks * observations_per_week)))


def effective_sample_size(weights: np.ndarray) -> float:
    """Tamano efectivo de Kish: 1/sum(w_t^2) para pesos normalizados."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights debe ser un vector finito y no vacio")
    total = float(w.sum())
    if total <= 0 or np.any(w < 0):
        raise ValueError("weights debe ser no negativo y sumar un valor positivo")
    w = w / total
    return float(1.0 / np.sum(w**2))


def resolve_decay_lambda(mm_cfg: Mapping, observations_per_week: float = 5.0) -> float:
    """Resuelve lambda, priorizando la vida media explicita sobre el legado."""
    half_life = mm_cfg.get("ewma_half_life_weeks")
    if half_life is not None:
        return decay_from_half_life(float(half_life), observations_per_week)
    return float(mm_cfg.get("decay_lambda", 1.0))


def _ewma_weights(T: int, decay_lambda: float) -> np.ndarray:
    """Construye pesos EWMA estables; la ultima fila recibe el mayor peso."""
    if T <= 0:
        raise ValueError("T debe ser positivo")
    if not 0 < decay_lambda <= 1:
        raise ValueError("decay_lambda debe pertenecer a (0, 1]")
    if np.isclose(decay_lambda, 1.0):
        return np.full(T, 1.0 / T)
    ages = np.arange(T - 1, -1, -1, dtype=float)
    log_weights = ages * np.log(decay_lambda)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def _validate_inputs(terminal: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    if terminal.empty or daily.empty:
        raise ValueError("terminal y daily no pueden estar vacios")
    labels = terminal.columns.tolist()
    if not labels or any(label not in daily.columns for label in labels):
        raise ValueError("daily debe contener todas las columnas de terminal")
    if terminal[labels].isna().any().any() or daily[labels].isna().any().any():
        raise ValueError("Los retornos de entrada contienen valores faltantes")
    if not np.isfinite(terminal[labels].to_numpy(dtype=float)).all():
        raise ValueError("terminal contiene NaN o infinitos")
    if not np.isfinite(daily[labels].to_numpy(dtype=float)).all():
        raise ValueError("daily contiene NaN o infinitos")
    return labels


def compute_targets(
    terminal: pd.DataFrame,
    daily: pd.DataFrame,
    decay_lambda: float = 1.0,
    covariance_shrinkage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula cuatro momentos centrales y covarianza ponderados.

    ``covariance_shrinkage`` aplica contraccion lineal hacia la matriz diagonal:
    ``(1-a) Sigma + a diag(Sigma)``. No se presenta como Ledoit-Wolf exacto;
    es un control transparente de condicionamiento cuya intensidad debe fijarse
    en validacion temporal, nunca usando el periodo de evaluacion final.
    """
    labels = _validate_inputs(terminal, daily)
    if not 0 <= covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage debe pertenecer a [0, 1]")

    X = terminal[labels].to_numpy(dtype=float)
    T, n = X.shape
    weights = _ewma_weights(T, float(decay_lambda))
    mu = weights @ X
    dev = X - mu

    M = np.vstack(
        [
            mu,
            weights @ dev**2,
            weights @ dev**3,
            weights @ dev**4,
        ]
    )
    Sigma_raw = (weights[:, None] * dev).T @ dev
    diagonal = np.diag(np.diag(Sigma_raw))
    Sigma = (1.0 - covariance_shrinkage) * Sigma_raw + covariance_shrinkage * diagonal
    Sigma = (Sigma + Sigma.T) / 2.0

    # Mantiene la firma historica. Los benchmarks nuevos usan directamente
    # M[0] y Sigma, garantizando el mismo horizonte y ponderacion que MM.
    mu_daily = daily[labels].mean().to_numpy(dtype=float)

    half_life_rows = (
        np.inf if np.isclose(decay_lambda, 1.0)
        else float(np.log(0.5) / np.log(decay_lambda))
    )
    print("\n[targets] momentos objetivo")
    print(
        f"  T={T}, n={n}, lambda={decay_lambda:.8f}, "
        f"vida_media_filas={half_life_rows:.1f}, N_eff={effective_sample_size(weights):.1f}"
    )
    print(f"  covariance_shrinkage={covariance_shrinkage:.3f}")

    eigenvalues = np.linalg.eigvalsh(Sigma)
    if eigenvalues.min() < -1e-12:
        raise RuntimeError(
            f"La covarianza no es semidefinida positiva: lambda_min={eigenvalues.min():.3e}"
        )
    if eigenvalues.min() <= 0:
        epsilon = abs(float(eigenvalues.min())) + 1e-10
        Sigma += epsilon * np.eye(n)
        print(f"  sigma_target=regularizada_numericamente, eps={epsilon:.2e}")
    else:
        condition = float(eigenvalues.max() / eigenvalues.min())
        print(f"  sigma_target=PD, lambda_min={eigenvalues.min():.2e}, condicion={condition:.2e}")

    return M, Sigma, mu_daily


def save_targets(
    M: np.ndarray,
    Sigma: np.ndarray,
    labels: list[str],
    out_path: str | Path,
) -> None:
    """Guarda targets con nombres de columnas auditables."""
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        M.T,
        index=labels,
        columns=["M1_media", "M2_varianza", "M3_central", "M4_central"],
    ).to_csv(out / "targets_moments.csv")
    pd.DataFrame(Sigma, index=labels, columns=labels).to_csv(out / "targets_cov.csv")
    print(f"  targets guardados en {out}/")
