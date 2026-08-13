"""Motor walk-forward con deriva de pesos y costos de transaccion."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

Estimator = Callable[[pd.DataFrame], np.ndarray]


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "Q") -> pd.DatetimeIndex:
    """Primer dia observado de cada nuevo periodo de rebalanceo."""
    periods = index.to_period(frequency)
    return pd.DatetimeIndex(index[~periods.duplicated()])


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
    schedule = {
        pd.Timestamp(date.to_pydatetime()): np.asarray(weights, dtype=float)
        for date, weights in target_weights.items()
    }
    dates = pd.DatetimeIndex(pd.to_datetime(daily_returns.index))
    daily_returns = daily_returns.copy()
    daily_returns.index = dates

    # Una fecha de rebalanceo ausente del indice se omitiria en silencio y la
    # cartera conservaria la asignacion previa sin dejar rastro. Un desajuste de
    # calendario -un feriado local, un reindexado- degradaria la estrategia sin
    # que ninguna metrica lo delate, por lo que se exige coincidencia exacta.
    unscheduled = sorted(date for date in schedule if date not in set(dates))
    if unscheduled:
        raise ValueError(
            "Fechas de rebalanceo ausentes del indice de retornos: "
            f"{[str(date.date()) for date in unscheduled]}"
        )

    for date_value, row in daily_returns.iterrows():
        date = pd.Timestamp(str(date_value))
        turnover = cost = 0.0
        if date in schedule:
            target = schedule[date]
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
    full_returns = full_returns.copy()
    full_returns.index = pd.DatetimeIndex(pd.to_datetime(full_returns.index))
    evaluation = full_returns.loc[full_returns.index >= pd.Timestamp(evaluation_start)]
    schedule: dict[pd.Timestamp, np.ndarray] = {}
    evaluation_dates = cast(pd.DatetimeIndex, pd.to_datetime(evaluation.index))
    for date in rebalance_dates(evaluation_dates, frequency):
        training = full_returns.loc[full_returns.index < date]
        if len(training) < min_history:
            continue
        schedule[pd.Timestamp(date)] = np.asarray(estimator(training.copy()), dtype=float)
    if not schedule:
        raise RuntimeError("No se generaron rebalanceos; revise min_history y fechas")
    return schedule


def performance_metrics(wealth: pd.Series, execution: pd.DataFrame, annual_periods: int = 252) -> dict[str, float]:
    """CAGR, volatilidad, Sharpe y drawdown a partir de la serie de riqueza neta."""
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


def _annualised_sharpe(returns: np.ndarray, annual_periods: int) -> float:
    """Sharpe anualizado sin tasa libre de riesgo, con denominador protegido."""
    return float(
        np.mean(returns)
        / max(float(np.std(returns, ddof=1)), 1e-15)
        * np.sqrt(annual_periods)
    )


def moving_block_bootstrap_sharpe_difference(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    block_size: int = 8,
    samples: int = 2000,
    seed: int = 0,
    annual_periods: int = 252,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """IC percentil y p-valor del diferencial de Sharpe anualizado.

    Las dos series se remuestrean con los MISMOS indices en cada replica, de
    modo que la dependencia contemporanea entre estrategia y baseline se
    conserva y el intervalo describe el diferencial, no dos series
    independientes.

    El p-valor usa la aproximacion bootstrap basica: la distribucion de
    ``delta* - delta`` estima la del error de estimacion, y bajo H0 el
    verdadero diferencial es cero.
    """
    strategy = np.asarray(strategy_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if strategy.shape != benchmark.shape or strategy.ndim != 1:
        raise ValueError("Las series deben ser vectores del mismo largo")
    if not np.isfinite(strategy).all() or not np.isfinite(benchmark).all():
        raise ValueError("Las series contienen valores no finitos")
    n = len(strategy)
    if not 1 <= block_size <= n:
        raise ValueError("block_size invalido")
    if samples < 100:
        raise ValueError("samples debe ser al menos 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level debe pertenecer a (0, 1)")

    observed = _annualised_sharpe(strategy, annual_periods) - _annualised_sharpe(
        benchmark, annual_periods
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for sample in range(samples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_size + 1))
            indices.extend(range(start, start + block_size))
        idx = np.asarray(indices[:n])
        differences[sample] = _annualised_sharpe(
            strategy[idx], annual_periods
        ) - _annualised_sharpe(benchmark[idx], annual_periods)

    alpha = 1.0 - confidence_level
    low, median, high = np.quantile(
        differences, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]
    )
    centered = differences - observed
    pvalue = (1.0 + float(np.sum(np.abs(centered) >= abs(observed)))) / (samples + 1.0)
    return {
        "sharpe_difference": observed,
        "ci_low": float(low),
        "median": float(median),
        "ci_high": float(high),
        "pvalue_raw": float(min(pvalue, 1.0)),
        "bootstrap_samples": float(samples),
        "block_size": float(block_size),
    }
