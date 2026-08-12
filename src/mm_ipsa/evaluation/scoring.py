"""Scores propios y pruebas de cobertura para escenarios ponderados."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _normalise_weights(weights: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    if p.shape != (n,) or np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("Probabilidades invalidas")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Las probabilidades deben sumar un valor positivo")
    return p / total


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Cuantil empirico izquierdo de una distribucion discreta ponderada."""
    if not 0 <= q <= 1:
        raise ValueError("q debe pertenecer a [0, 1]")
    x = np.asarray(values, dtype=float)
    p = _normalise_weights(weights, len(x))
    order = np.argsort(x)
    idx = np.searchsorted(np.cumsum(p[order]), q, side="left")
    return float(x[order[min(idx, len(x) - 1)]])


def lower_tail_mean(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Expected shortfall inferior usando masa fraccionaria en el cuantil."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha debe pertenecer a (0, 1]")
    x = np.asarray(values, dtype=float)
    p = _normalise_weights(weights, len(x))
    order = np.argsort(x)
    remaining = alpha
    total = 0.0
    for value, probability in zip(x[order], p[order]):
        take = min(remaining, float(probability))
        total += take * float(value)
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / alpha


def crps_ensemble(values: np.ndarray, weights: np.ndarray, observation: float) -> float:
    """CRPS exacto O(N log N) para una distribucion discreta ponderada."""
    x = np.asarray(values, dtype=float)
    p = _normalise_weights(weights, len(x))
    first = float(np.sum(p * np.abs(x - observation)))
    order = np.argsort(x)
    xs, ps = x[order], p[order]
    cumulative_p = np.cumsum(ps) - ps
    cumulative_px = np.cumsum(ps * xs) - ps * xs
    pair_half = float(np.sum(ps * (xs * cumulative_p - cumulative_px)))
    return first - pair_half


def energy_score(
    scenarios: np.ndarray,
    weights: np.ndarray,
    observation: np.ndarray,
    seed: int = 0,
    pair_samples: int = 20_000,
) -> float:
    """Energy score; aproxima el termino entre escenarios por Monte Carlo."""
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observation, dtype=float)
    p = _normalise_weights(weights, len(x))
    first = float(np.sum(p * np.linalg.norm(x - y, axis=1)))
    rng = np.random.default_rng(seed)
    left = rng.choice(len(x), size=pair_samples, p=p)
    right = rng.choice(len(x), size=pair_samples, p=p)
    pair = float(np.mean(np.linalg.norm(x[left] - x[right], axis=1)))
    return first - 0.5 * pair


def _crps_matrix(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """CRPS por observacion y activo, reutilizando el termino entre escenarios."""
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observations, dtype=float)
    p = _normalise_weights(probabilities, len(x))
    result = np.empty((len(y), x.shape[1]), dtype=float)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j])
        xs, ps = x[order, j], p[order]
        cumulative_p = np.cumsum(ps) - ps
        cumulative_px = np.cumsum(ps * xs) - ps * xs
        pair_half = float(np.sum(ps * (xs * cumulative_p - cumulative_px)))
        first = np.sum(
            p[:, None] * np.abs(x[:, j, None] - y[None, :, j]), axis=0
        )
        result[:, j] = first - pair_half
    return result


def _observation_score_components(
    model_name: str,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    observation_ids: Iterable[object] | None,
    seed: int,
    energy_pair_samples: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observations, dtype=float)
    p = _normalise_weights(probabilities, len(x))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Shapes incompatibles para scores por observacion")
    if len(y) < 2:
        raise ValueError("Se requieren al menos dos observaciones OOS")
    if energy_pair_samples <= 0:
        raise ValueError("energy_pair_samples debe ser positivo")
    ids = list(range(len(y))) if observation_ids is None else list(observation_ids)
    if len(ids) != len(y) or len(set(map(str, ids))) != len(ids):
        raise ValueError("observation_ids debe ser unico y coincidir con OOS")

    crps_values = _crps_matrix(x, p, y)
    rng = np.random.default_rng(seed)
    left = rng.choice(len(x), size=energy_pair_samples, p=p)
    right = rng.choice(len(x), size=energy_pair_samples, p=p)
    pair_term = float(np.mean(np.linalg.norm(x[left] - x[right], axis=1)))
    energy_values = np.asarray(
        [float(p @ np.linalg.norm(x - observation, axis=1)) - 0.5 * pair_term for observation in y]
    )
    variogram_values = np.asarray(
        [variogram_score(x, p, observation) for observation in y]
    )
    frame = pd.DataFrame(
        {
            "model": model_name,
            "observation": [str(value) for value in ids],
            "mean_crps": crps_values.mean(axis=1),
            "energy_score": energy_values,
            "variogram_score": variogram_values,
        }
    )
    return frame, crps_values


def score_scenarios_by_observation(
    model_name: str,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    observation_ids: Iterable[object] | None = None,
    seed: int = 0,
    energy_pair_samples: int = 20_000,
) -> pd.DataFrame:
    """Devuelve perdidas propias pareadas para cada origen OOS.

    El termino entre pares del Energy Score se estima una sola vez por modelo.
    Asi la variacion temporal refleja las observaciones y no nuevos sorteos Monte
    Carlo en cada fecha.
    """
    frame, _ = _observation_score_components(
        model_name,
        scenarios,
        probabilities,
        observations,
        observation_ids,
        seed,
        energy_pair_samples,
    )
    return frame


def variogram_score(
    scenarios: np.ndarray,
    weights: np.ndarray,
    observation: np.ndarray,
    order: float = 0.5,
) -> float:
    """Variogram score multivariado con peso unitario por par."""
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observation, dtype=float)
    p = _normalise_weights(weights, len(x))
    score = 0.0
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            observed = abs(float(y[i] - y[j])) ** order
            expected = float(p @ (np.abs(x[:, i] - x[:, j]) ** order))
            score += (observed - expected) ** 2
    return float(score)


def kupiec_test(exceedances: np.ndarray, alpha: float) -> dict[str, float]:
    """Likelihood-ratio de cobertura incondicional de VaR."""
    hits = np.asarray(exceedances, dtype=bool)
    n, failures = len(hits), int(hits.sum())
    if n == 0:
        raise ValueError("Se requiere al menos una observacion")
    rate = failures / n
    eps = np.finfo(float).eps
    a = np.clip(alpha, eps, 1 - eps)
    r = np.clip(rate, eps, 1 - eps)
    log_null = failures * np.log(a) + (n - failures) * np.log(1 - a)
    log_alt = failures * np.log(r) + (n - failures) * np.log(1 - r)
    statistic = max(0.0, -2.0 * (log_null - log_alt))
    return {
        "n": float(n),
        "exceedances": float(failures),
        "rate": float(rate),
        "lr_uc": float(statistic),
        "pvalue_uc": float(chi2.sf(statistic, 1)),
    }


def christoffersen_test(exceedances: np.ndarray) -> dict[str, float]:
    """Prueba de independencia de hits mediante transiciones de Markov."""
    hits = np.asarray(exceedances, dtype=int)
    if len(hits) < 2:
        raise ValueError("Se requieren al menos dos observaciones")
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))
    eps = np.finfo(float).eps
    pi = np.clip((n01 + n11) / max(1, n00 + n01 + n10 + n11), eps, 1 - eps)
    pi01 = np.clip(n01 / max(1, n00 + n01), eps, 1 - eps)
    pi11 = np.clip(n11 / max(1, n10 + n11), eps, 1 - eps)
    log_independent = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_markov = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    statistic = max(0.0, -2.0 * (log_independent - log_markov))
    return {
        "n00": float(n00), "n01": float(n01), "n10": float(n10), "n11": float(n11),
        "lr_ind": float(statistic), "pvalue_ind": float(chi2.sf(statistic, 1)),
    }


def evaluate_scenarios_detailed(
    model_name: str,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    labels: list[str],
    observation_ids: Iterable[object] | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    energy_pair_samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalua margenes, dependencia y devuelve perdidas por observacion OOS."""
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observations, dtype=float)
    p = _normalise_weights(probabilities, len(x))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(labels) != x.shape[1]:
        raise ValueError("Shapes incompatibles en evaluate_scenarios")

    observation_scores, crps_values = _observation_score_components(
        model_name,
        x,
        p,
        y,
        observation_ids,
        seed,
        energy_pair_samples,
    )
    rows = []
    for j, label in enumerate(labels):
        var = weighted_quantile(x[:, j], p, alpha)
        es = lower_tail_mean(x[:, j], p, alpha)
        hits = y[:, j] < var
        uc = kupiec_test(hits, alpha)
        ind = christoffersen_test(hits)
        rows.append(
            {
                "model": model_name,
                "asset": label,
                "crps": float(crps_values[:, j].mean()),
                "var_alpha": var,
                "es_alpha": es,
                **uc,
                **ind,
                "lr_cc": uc["lr_uc"] + ind["lr_ind"],
                "pvalue_cc": float(chi2.sf(uc["lr_uc"] + ind["lr_ind"], 2)),
            }
        )
    marginal = pd.DataFrame(rows)

    scenario_corr = _weighted_corr(x, p)
    oos_corr = np.corrcoef(y, rowvar=False)
    mask = ~np.eye(x.shape[1], dtype=bool)
    aggregate = pd.DataFrame(
        [
            {
                "model": model_name,
                "mean_crps": float(marginal["crps"].mean()),
                "energy_score": float(observation_scores["energy_score"].mean()),
                "variogram_score": float(observation_scores["variogram_score"].mean()),
                "correlation_mae_oos": float(np.mean(np.abs(scenario_corr - oos_corr)[mask])),
                "mean_exceedance_rate": float(marginal["rate"].mean()),
                "coverage_rejections_5pct": int((marginal["pvalue_uc"] < 0.05).sum()),
            }
        ]
    )
    return marginal, aggregate, observation_scores


def evaluate_scenarios(
    model_name: str,
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    observations: np.ndarray,
    labels: list[str],
    alpha: float = 0.05,
    seed: int = 0,
    energy_pair_samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """API compatible para evaluacion agregada y marginal."""
    marginal, aggregate, _ = evaluate_scenarios_detailed(
        model_name,
        scenarios,
        probabilities,
        observations,
        labels,
        observation_ids=None,
        alpha=alpha,
        seed=seed,
        energy_pair_samples=energy_pair_samples,
    )
    return marginal, aggregate


def _weighted_corr(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weights @ values
    dev = values - mean
    covariance = (weights[:, None] * dev).T @ dev
    standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    result = covariance / np.maximum(np.outer(standard, standard), 1e-15)
    np.fill_diagonal(result, 1.0)
    return result
