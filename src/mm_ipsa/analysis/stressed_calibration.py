"""Calibracion sobre periodo de estres y el precio que cobra.

El diagnostico de prociclicidad establece el problema: los excesos se concentran
en el regimen tensionado porque los generadores se estiman sobre historia
reciente y esa historia describe la calma. El marco regulatorio responde
exigiendo que la medida de riesgo se calibre ademas sobre un periodo de estres,
de modo que el limite no dependa de que el mercado este tranquilo cuando se
estima.

Este modulo contrasta esa respuesta en vez de asumirla. Calibra cada generador
dos veces sobre la misma especificacion -una con el tramo tensionado de la
muestra y otra con el tramo en calma- y evalua ambas sobre el mismo periodo
posterior.

La comparacion tiene dos lados y ninguno se puede omitir. La calibracion
estresada deberia reducir los excesos cuando llega la tension, que es para lo que
existe. Pero produce un limite mas alto en todo momento, tambien durante los
tramos tranquilos, y ese exceso de conservadurismo es capital inmovilizado. Un
informe que solo muestre la reduccion de excesos presenta media pregunta.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

# ``generate_benchmarks`` nombra los controles por su construccion y los
# artefactos publicados usan la forma corta; se unifican para poder comparar.
CANONICAL_NAMES = {
    "gaussian_terminal": "gaussian",
    "student_t_terminal": "student_t",
    "historical_weighted": "historical",
}


def _canonical(models: dict[str, Any]) -> dict[str, Any]:
    return {CANONICAL_NAMES.get(name, name): value for name, value in models.items()}


def calibrate_on_window(
    daily: pd.DataFrame,
    start: str,
    end: str,
    main_cfg: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calibra los cinco generadores usando solo el tramo indicado.

    Reproduce la secuencia del analisis principal, de modo que la unica
    diferencia frente a la calibracion publicada es el tramo de historia que la
    alimenta.
    """
    from mm_ipsa.analysis.recalibration_ablation import _calibrate_origin
    from mm_ipsa.data.transform import _rolling_terminal_fast

    window = daily.loc[start:end]
    horizon = int(main_cfg["data"]["H"])
    if len(window) < horizon * 20:
        raise ValueError(
            f"El tramo {start}..{end} tiene {len(window)} jornadas, "
            "insuficientes para calibrar"
        )
    terminal = _rolling_terminal_fast(window, horizon)
    return _canonical(_calibrate_origin(window, terminal, main_cfg, seed_offset))


def realised_volatility(daily: pd.DataFrame, start: str, end: str) -> float:
    """Volatilidad anualizada equiponderada del tramo, para describirlo."""
    window = daily.loc[start:end]
    return float(window.mean(axis=1).std(ddof=1) * np.sqrt(252.0))


def compare_calibrations(
    calibrations: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    windows: pd.DataFrame,
    regimes: pd.Series,
    main_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Excesos por regimen y nivel del limite, para cada calibracion y modelo.

    ``windows`` son las ventanas ya disjuntas y ya clasificadas, una fila por
    etiqueta de ``regimes``. Se pide asi, y no la serie solapada, para que la
    correspondencia entre ventana y regimen sea explicita y no dependa de que
    esta funcion repita el submuestreo.

    ``capital_ratio`` compara el ES de cada calibracion contra el de la primera
    declarada, que actua como referencia. Es la medida del conservadurismo que la
    calibracion alternativa impone en todo momento.
    """
    from mm_ipsa.evaluation.procyclicality import CALM, STRESSED
    from mm_ipsa.evaluation.scoring import lower_tail_mean, weighted_quantile

    alpha = float(main_cfg["portfolio"]["alpha_cvar"])
    weights = np.full(windows.shape[1], 1.0 / windows.shape[1])

    labels = regimes.to_numpy()
    if len(windows) != len(labels):
        raise ValueError("windows y regimes deben tener el mismo numero de filas")

    portfolio = windows.to_numpy() @ weights
    calm_mask = labels == CALM
    stressed_mask = labels == STRESSED
    if not calm_mask.any() or not stressed_mask.any():
        raise ValueError("Se requieren ventanas de ambos regimenes")

    reference = next(iter(calibrations))
    baseline: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for calibration, models in calibrations.items():
        for model, (scenarios, probabilities) in models.items():
            projected = scenarios @ weights
            var = weighted_quantile(projected, probabilities, alpha)
            shortfall = lower_tail_mean(projected, probabilities, alpha)
            if calibration == reference:
                baseline[model] = shortfall

            rows.append(
                {
                    "calibration": calibration,
                    "model": model,
                    "var": float(var),
                    "expected_shortfall": float(shortfall),
                    "capital_ratio": float(shortfall / baseline[model])
                    if model in baseline
                    else float("nan"),
                    "calm_breach_rate": float(
                        np.mean(portfolio[calm_mask] < var)
                    ),
                    "stressed_breach_rate": float(
                        np.mean(portfolio[stressed_mask] < var)
                    ),
                    "overall_breach_rate": float(np.mean(portfolio < var)),
                }
            )

    table = pd.DataFrame(rows)
    table["calm_excess"] = table["calm_breach_rate"] - alpha
    table["stressed_excess"] = table["stressed_breach_rate"] - alpha
    return table


def run_stressed_calibration(
    main_cfg: dict[str, Any],
    daily_in_sample: pd.DataFrame,
    daily_out_of_sample: pd.DataFrame,
    observations: pd.DataFrame,
    output: str | Path,
    *,
    stress_window: tuple[str, str] = ("2020-01-01", "2021-12-31"),
    calm_window: tuple[str, str] = ("2022-01-01", "2023-12-31"),
    lookback: int = 63,
    quantile: float = 0.75,
) -> dict[str, pd.DataFrame]:
    """Ejecuta el contraste completo y escribe la tabla resultante."""
    from mm_ipsa.evaluation.procyclicality import (
        classify_regimes,
        trailing_volatility,
    )

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    horizon = int(main_cfg["data"]["H"])

    ends = cast(pd.DatetimeIndex, pd.DatetimeIndex(observations.iloc[::horizon].index))
    regimes = classify_regimes(
        trailing_volatility(daily_out_of_sample, ends, horizon, lookback), quantile
    )
    labelled = regimes.notna().to_numpy()

    calibrations = {
        "calma": calibrate_on_window(
            daily_in_sample, *calm_window, main_cfg, seed_offset=0
        ),
        "estres": calibrate_on_window(
            daily_in_sample, *stress_window, main_cfg, seed_offset=1_000
        ),
    }

    windows = observations.iloc[::horizon].loc[labelled]
    table = compare_calibrations(
        calibrations, windows, regimes.dropna(), main_cfg
    )

    periods = pd.DataFrame(
        [
            {
                "window": "calma",
                "start": calm_window[0],
                "end": calm_window[1],
                "annualised_volatility": realised_volatility(
                    daily_in_sample, *calm_window
                ),
            },
            {
                "window": "estres",
                "start": stress_window[0],
                "end": stress_window[1],
                "annualised_volatility": realised_volatility(
                    daily_in_sample, *stress_window
                ),
            },
        ]
    )

    table.to_csv(target / "stressed_calibration.csv", index=False)
    periods.to_csv(target / "stressed_calibration_periods.csv", index=False)
    return {"comparison": table, "periods": periods}
