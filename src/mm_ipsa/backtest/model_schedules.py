"""Carteras derivadas de modelos, recalibradas en cada fecha de rebalanceo.

Hasta la version 0.8.0 los portafolios derivados de MM y de los controles se
calibraban una sola vez al inicio del periodo de evaluacion y quedaban
congelados, mientras que los baselines ingenuos se recalibraban cada trimestre.
La comparacion de H4 se corrigio emparejando cada estrategia con el baseline de
su mismo diseno, pero la limitacion de fondo permanecia: ningun portafolio
derivado de un modelo se evaluaba bajo recalibracion periodica.

Este modulo cierra esa brecha. En cada fecha de rebalanceo reconstruye los
targets, recalibra los cinco generadores de escenarios usando exclusivamente
observaciones anteriores, y deriva de cada uno las tres carteras del protocolo.

El costo era el impedimento real: antes de vectorizar el calculo de momentos,
una sola recalibracion de MM tomaba del orden de un minuto y medio.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from mm_ipsa.backtest.walk_forward import rebalance_dates


def _origin_models(
    training: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Recalibra los cinco generadores con datos anteriores a la fecha.

    Reproduce exactamente la secuencia del analisis principal: targets EWMA con
    contraccion de Ledoit-Wolf, calibracion MM por descenso en bloques y los
    cuatro controles con sus propios parametros reestimados.
    """
    from mm_ipsa.config import objective_weights, target_parameters
    from mm_ipsa.data.transform import _rolling_terminal_fast
    from mm_ipsa.mm.bcd import BCDSolver
    from mm_ipsa.mm.objective import MMObjective
    from mm_ipsa.mm.targets import _ewma_weights, compute_targets
    from mm_ipsa.models.benchmarks import generate_benchmarks, resolve_student_t_df

    horizon = int(cfg["data"]["H"])
    mm_cfg = cfg["mm"]
    parameters = target_parameters(mm_cfg)
    terminal = _rolling_terminal_fast(training, horizon)
    moments, covariance, _ = compute_targets(terminal, training, **parameters)

    solver = BCDSolver(
        MMObjective(
            moments,
            covariance,
            objective_weights(mm_cfg),
            int(mm_cfg["N_scenarios"]),
        ),
        mm_cfg,
        n_workers=1,
    )
    scenarios, probabilities = solver.solve()

    benchmark_cfg = cfg["benchmarks"]
    history_weights = _ewma_weights(len(terminal), parameters["decay_lambda"])
    degrees, _ = resolve_student_t_df(
        benchmark_cfg, terminal.to_numpy(), moments[0], covariance, history_weights
    )
    controls, _ = generate_benchmarks(
        moments,
        covariance,
        terminal.to_numpy(),
        history_weights,
        n_scenarios=int(benchmark_cfg["N_scenarios"]),
        seed=int(benchmark_cfg["seed"]),
        student_t_df=degrees,
        daily_returns=training.to_numpy(),
        horizon=horizon,
        include=list(benchmark_cfg["include"]),
    )
    return {"MM": (scenarios, probabilities), **controls}


def _origin_portfolios(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Deriva las tres carteras del protocolo desde cada predictiva."""
    from mm_ipsa.portfolio.optimization import (
        maximum_sharpe,
        minimum_cvar,
        minimum_variance,
        weighted_mean_cov,
    )

    portfolio_cfg = cfg["portfolio"]
    max_weight = float(portfolio_cfg["max_weight"])
    l2_penalty = float(portfolio_cfg.get("l2_penalty", 0.0))
    alpha = float(portfolio_cfg["alpha_cvar"])
    risk_free = float(portfolio_cfg["rf"])
    seed = int(cfg["evaluation"]["seed"])

    weights: dict[str, np.ndarray] = {}
    for name, (scenarios, probabilities) in models.items():
        mean, covariance = weighted_mean_cov(scenarios, probabilities)
        weights[f"WF_{name}_MinVariance"] = minimum_variance(
            covariance, max_weight, l2_penalty
        )
        weights[f"WF_{name}_MinCVaR"] = minimum_cvar(
            scenarios, probabilities, alpha, max_weight
        )
        weights[f"WF_{name}_MaxSharpe"] = maximum_sharpe(
            mean,
            covariance,
            max_weight=max_weight,
            risk_free_rate=risk_free,
            l2_penalty=l2_penalty,
            seed=seed,
        )
    return weights


def walk_forward_model_schedules(
    full_returns: pd.DataFrame,
    evaluation_start: str | pd.Timestamp,
    cfg: dict[str, Any],
    frequency: str = "Q",
    min_history: int = 252,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[pd.Timestamp, np.ndarray]], pd.DataFrame]:
    """Calendarios de rebalanceo con todos los modelos recalibrados en cada origen.

    Devuelve un calendario por estrategia y una tabla de trazabilidad con cuantas
    filas de entrenamiento y cuantos escenarios sostienen cada decision. Esa
    tabla es la evidencia de que ninguna cartera vio datos posteriores a su
    propia fecha.
    """
    returns = full_returns.copy()
    returns.index = pd.DatetimeIndex(pd.to_datetime(returns.index))
    evaluation = returns.loc[returns.index >= pd.Timestamp(evaluation_start)]
    if evaluation.empty:
        raise ValueError("El periodo de evaluacion esta vacio")

    schedules: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    audit: list[dict[str, object]] = []
    evaluation_index = cast(pd.DatetimeIndex, pd.to_datetime(evaluation.index))
    for date in rebalance_dates(evaluation_index, frequency):
        training = returns.loc[returns.index < date]
        if len(training) < min_history:
            continue
        if progress is not None:
            progress(f"    origen {date.date()}: {len(training)} filas")
        models = _origin_models(training, cfg)
        for name, weights in _origin_portfolios(models, cfg).items():
            schedules.setdefault(name, {})[pd.Timestamp(date)] = weights
        audit.append(
            {
                "rebalance_date": str(date.date()),
                "training_rows": len(training),
                "training_end": str(training.index.max().date()),
                **{
                    f"scenarios_{name}": int(scenarios.shape[0])
                    for name, (scenarios, _) in models.items()
                },
            }
        )

    if not schedules:
        raise RuntimeError(
            "Ninguna fecha de rebalanceo alcanzo el minimo de historia requerido"
        )
    return schedules, pd.DataFrame(audit)
