"""Como se degrada el ajuste de MM-BCD al crecer la dimension.

El objetivo ajusta cuatro momentos marginales por activo y las covarianzas
cruzadas. Los primeros crecen de forma lineal con el numero de activos y las
segundas como su cuadrado, de modo que ampliar el universo no encarece ambos
terminos por igual. Este modulo mide esa asimetria en vez de suponerla.

La medicion separa tres explicaciones posibles de cualquier degradacion:

``capacidad``
    El soporte de escenarios no alcanza para representar la dependencia. Se
    contrasta elevando ``N_scenarios``.

``presupuesto``
    El solver se queda sin iteraciones. Se contrasta elevando ``bcd_max_iter``.

``parada``
    El criterio de convergencia corta antes de tiempo. Se contrasta ajustando
    ``tol``.

Distinguirlas importa porque llevan a remedios opuestos, y porque la primera es
la unica que el argumento de conteo respalda a primera vista: con quinientos
escenarios en veintinueve dimensiones hay del orden de quince mil parametros
libres para quinientos veintidos objetivos, de modo que la representacion no
puede ser el limite.
"""

from __future__ import annotations

import copy
import io
import re
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def moment_fit(
    objective: Any, scenarios: np.ndarray, probabilities: np.ndarray,
    moment_targets: np.ndarray, covariance_target: np.ndarray,
) -> dict[str, float]:
    """Error relativo maximo por tipo de objetivo.

    Se reporta el maximo y no el promedio porque un solo momento mal ajustado
    basta para invalidar la predictiva, y el promedio lo escondería.
    """
    fitted_moments, _, fitted_covariance = objective.compute_moments(
        scenarios, probabilities
    )
    off_diagonal = ~np.eye(moment_targets.shape[1], dtype=bool)

    def relative(fitted: np.ndarray, target: np.ndarray) -> float:
        denominator = np.maximum(np.abs(target), 1e-12)
        return float(np.max(np.abs(fitted - target) / denominator))

    return {
        "mean_error": relative(fitted_moments[0], moment_targets[0]),
        "variance_error": relative(fitted_moments[1], moment_targets[1]),
        "third_moment_error": relative(fitted_moments[2], moment_targets[2]),
        "fourth_moment_error": relative(fitted_moments[3], moment_targets[3]),
        "covariance_error": relative(
            fitted_covariance[off_diagonal], covariance_target[off_diagonal]
        ),
    }


def solve_with_overrides(
    prices: pd.DataFrame, main_cfg: dict[str, Any], **overrides: Any
) -> dict[str, Any]:
    """Calibra MM-BCD sobre un panel y devuelve ajuste, coste y diagnostico.

    ``strict_solver`` se desactiva a proposito: la pregunta es como queda el
    ajuste bajo cada configuracion, y un fallo por estacionariedad es parte de
    la respuesta y no un motivo para interrumpir la medicion.
    """
    from mm_ipsa.config import objective_weights, target_parameters
    from mm_ipsa.data.transform import _rolling_terminal_fast
    from mm_ipsa.mm.bcd import BCDSolver
    from mm_ipsa.mm.objective import MMObjective
    from mm_ipsa.mm.targets import compute_targets

    settings = copy.deepcopy(main_cfg["mm"])
    settings.update(overrides)
    settings["strict_solver"] = False

    daily = prices.pct_change().dropna(how="all")
    terminal = _rolling_terminal_fast(daily, int(main_cfg["data"]["H"]))
    moments, covariance, _ = compute_targets(
        terminal, daily, **target_parameters(settings)
    )
    assets = moments.shape[1]

    objective = MMObjective(
        moments, covariance, objective_weights(settings),
        int(settings["N_scenarios"]),
    )
    log = io.StringIO()
    started = time.time()
    with redirect_stdout(log):
        scenarios, probabilities = BCDSolver(
            objective, settings, n_workers=1
        ).solve()
    elapsed = time.time() - started
    captured = log.getvalue()

    iterations = [int(value) for value in re.findall(r"iters=(\d+)", captured)]
    return {
        "assets": assets,
        "covariance_targets": assets * (assets - 1) // 2,
        "targets": 4 * assets + assets * (assets - 1) // 2,
        "n_scenarios": int(settings["N_scenarios"]),
        "max_iterations": int(settings["bcd_max_iter"]),
        "tolerance": float(settings["tol"]),
        "objective": float(objective.evaluate(scenarios, probabilities)),
        "seconds": float(elapsed),
        "iterations_used": max(iterations) if iterations else 0,
        "stationary_starts": captured.count("stationary")
        - captured.count("nonstationary"),
        **moment_fit(objective, scenarios, probabilities, moments, covariance),
    }


def scalability_sweep(
    panels: dict[str, pd.DataFrame],
    main_cfg: dict[str, Any],
    variants: dict[str, dict[str, Any]] | None = None,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """Recorre paneles y configuraciones del solver, y tabula el ajuste.

    ``variants`` mapea una etiqueta a los parametros que se sobreescriben. Por
    defecto contrasta la configuracion publicada contra mas escenarios y mas
    iteraciones, que son las dos explicaciones que se examinan primero.
    """
    if variants is None:
        variants = {
            "publicada": {},
            "mas_escenarios": {"N_scenarios": 2000},
            "mas_iteraciones": {"bcd_max_iter": 600},
        }

    rows: list[dict[str, Any]] = []
    for panel, prices in panels.items():
        for variant, overrides in variants.items():
            record = solve_with_overrides(prices, main_cfg, **overrides)
            rows.append({"panel": panel, "variant": variant, **record})

    table = pd.DataFrame(rows)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
    return table
