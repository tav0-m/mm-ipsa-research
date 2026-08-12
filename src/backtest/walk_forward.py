"""Motor walk-forward con deriva de pesos y costos de transaccion."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


Estimator = Callable[[pd.DataFrame], np.ndarray]


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "Q") -> pd.DatetimeIndex:
    """Primer dia observado de cada nuevo periodo de rebalanceo."""
    marker = pd.Series(index=index, data=index.to_period(frequency))
    return pd.DatetimeIndex(marker.groupby(marker).apply(lambda values: values.index[0]).to_list())


def simulate_strategy(
    daily_returns: pd.DataFrame,
    target_weights: dict[pd.Timestamp, np.ndarray],
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    """Simula ejecucion: rebalancea antes del retorno y deja derivar pesos."""
    if daily_returns.empty or not target_weights:
        raise ValueError("Se requieren retornos y al menos un rebalanceo")
    n = daily_returns.shape[1]
    current = np.full(n, 1.0 / n)
    wealth = 1.0
    values, records = [], []
    cost_rate = transaction_cost_bps / 10_000.0
    total_turnover = total_cost_fraction = 0.0
    schedule = {pd.Timestamp(date): np.asarray(w, dtype=float) for date, w in target_weights.items()}

    for date, row in daily_returns.iterrows():
        turnover = cost = 0.0
        if pd.Timestamp(date) in schedule:
            target = schedule[pd.Timestamp(date)]
            if target.shape != (n,) or np.any(target < -1e-12) or not np.isclose(target.sum(), 1.0):
                raise ValueError(f"Pesos invalidos para {date}")
            turnover = 0.5 * float(np.sum(np.abs(target - current)))
            cost = turnover * cost_rate
            wealth *= 1.0 - cost
            current = target.copy()
            total_turnover += turnover
            total_cost_fraction += cost

        asset_returns = row.to_numpy(dtype=float)
        portfolio_return = float(current @ asset_returns)
        wealth *= 1.0 + portfolio_return
        denominator = 1.0 + portfolio_return
        if denominator <= 0:
            raise RuntimeError("El portafolio perdio 100% o mas en un periodo")
        current = current * (1.0 + asset_returns) / denominator
        values.append(wealth)
        net_return = (1.0 - cost) * (1.0 + portfolio_return) - 1.0
        records.append({
            "date": date,
            "turnover": turnover,
            "cost_fraction": cost,
            "gross_return": portfolio_return,
            "net_return": net_return,
        })

    wealth_series = pd.Series(values, index=daily_returns.index, name="wealth")
    execution = pd.DataFrame(records).set_index("date")
    return wealth_series, execution, {
        "total_turnover": total_turnover,
        "total_cost_fraction": total_cost_fraction,
    }


def walk_forward_weights(
    full_returns: pd.DataFrame,
    evaluation_start: str | pd.Timestamp,
    estimator: Estimator,
    frequency: str = "Q",
    min_history: int = 252,
) -> dict[pd.Timestamp, np.ndarray]:
    """Ajusta pesos solo con filas estrictamente anteriores a cada fecha."""
    evaluation = full_returns.loc[full_returns.index >= pd.Timestamp(evaluation_start)]
    schedule: dict[pd.Timestamp, np.ndarray] = {}
    for date in rebalance_dates(evaluation.index, frequency):
        training = full_returns.loc[full_returns.index < date]
        if len(training) < min_history:
            continue
        schedule[pd.Timestamp(date)] = np.asarray(estimator(training.copy()), dtype=float)
    if not schedule:
        raise RuntimeError("No se generaron rebalanceos; revise min_history y fechas")
    return schedule


def performance_metrics(wealth: pd.Series, execution: pd.DataFrame, annual_periods: int = 252) -> dict[str, float]:
    returns = execution["net_return"].to_numpy(dtype=float)
    years = len(returns) / annual_periods
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(np.std(returns, ddof=1) * np.sqrt(annual_periods))
    sharpe = float(np.mean(returns) / max(np.std(returns, ddof=1), 1e-15) * np.sqrt(annual_periods))
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "final_wealth": float(wealth.iloc[-1]),
    }


def moving_block_bootstrap_sharpe_difference(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    block_size: int = 8,
    samples: int = 2000,
    seed: int = 0,
    annual_periods: int = 252,
) -> dict[str, float]:
    """IC percentil del diferencial de Sharpe anualizado con bloques móviles."""
    strategy = np.asarray(strategy_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if strategy.shape != benchmark.shape or strategy.ndim != 1:
        raise ValueError("Las series deben ser vectores del mismo largo")
    n = len(strategy)
    if not 1 <= block_size <= n:
        raise ValueError("block_size invalido")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_size + 1))
            indices.extend(range(start, start + block_size))
        idx = np.asarray(indices[:n])
        s, b = strategy[idx], benchmark[idx]
        sr_s = np.mean(s) / max(np.std(s, ddof=1), 1e-15)
        sr_b = np.mean(b) / max(np.std(b, ddof=1), 1e-15)
        differences.append((sr_s - sr_b) * np.sqrt(annual_periods))
    low, median, high = np.quantile(differences, [0.025, 0.5, 0.975])
    return {"ci_low": float(low), "median": float(median), "ci_high": float(high)}
