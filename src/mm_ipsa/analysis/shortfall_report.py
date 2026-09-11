"""Informe de calibracion de Expected Shortfall para todos los generadores.

Contrasta el ES de cada modelo contra las perdidas de cola realmente observadas
fuera de muestra, activo por activo, y corrige por multiplicidad. Sin esa
correccion el informe es enganoso: con quince activos y un umbral del cinco por
ciento se esperan contrastes significativos por azar aunque todos los modelos
esten bien calibrados.

El horizonte compuesto obliga a un cuidado adicional. Los retornos terminales se
calculan cada dia sobre una ventana de ``H`` jornadas, de modo que observaciones
consecutivas comparten ``H - 1`` dias. El contraste se aplica por eso sobre
submuestras disjuntas y se resume con la mediana entre desplazamientos, que no
privilegia el mas favorable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.comparison import holm_adjust
from mm_ipsa.evaluation.expected_shortfall import disjoint_subsample_backtests

MODEL_ARTIFACTS = {
    "MM": ("mm_scenarios_x", "mm_probabilities_p"),
    "gaussian": ("gaussian_terminal_scenarios", "gaussian_terminal_probabilities"),
    "student_t": ("student_t_terminal_scenarios", "student_t_terminal_probabilities"),
    "historical": ("historical_weighted_scenarios", "historical_weighted_probabilities"),
    "dcc_garch": ("dcc_garch_scenarios", "dcc_garch_probabilities"),
}


def load_model_predictives(
    source: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carga los escenarios y probabilidades de cada generador disponible."""
    base = Path(source)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (scenarios, probabilities) in MODEL_ARTIFACTS.items():
        scenario_path = base / f"{scenarios}.npy"
        probability_path = base / f"{probabilities}.npy"
        if scenario_path.is_file() and probability_path.is_file():
            loaded[name] = (np.load(scenario_path), np.load(probability_path))
    if not loaded:
        raise FileNotFoundError(f"No se encontraron predictivas en {base}")
    return loaded


def shortfall_table(
    observations: pd.DataFrame,
    predictives: dict[str, tuple[np.ndarray, np.ndarray]],
    alpha: float,
    horizon: int,
    *,
    n_simulations: int = 4_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Una fila por modelo y activo, con el valor p ya corregido por Holm.

    La correccion se aplica dentro de cada modelo, sobre sus quince activos,
    porque la pregunta es si ese modelo esta bien calibrado y no cual de los
    setenta y cinco contrastes resulto mas extremo.
    """
    labels = list(observations.columns)
    frames: list[pd.DataFrame] = []

    for model, (scenarios, probabilities) in predictives.items():
        if scenarios.shape[1] != len(labels):
            raise ValueError(
                f"{model} tiene {scenarios.shape[1]} activos y se esperaban "
                f"{len(labels)}"
            )
        rows: list[dict[str, Any]] = []
        for column, label in enumerate(labels):
            result = disjoint_subsample_backtests(
                observations[label].to_numpy(),
                scenarios[:, column],
                probabilities,
                alpha,
                horizon,
                n_simulations=n_simulations,
                seed=seed + column,
            )
            subsamples = result["subsamples"]
            realised = np.array(
                [r["realised_tail_mean"] for r in subsamples], dtype=float
            )
            predicted = float(subsamples[0]["predicted_es"])
            rows.append(
                {
                    "model": model,
                    "asset": label,
                    "predicted_es": predicted,
                    "realised_tail_mean": float(np.nanmean(realised)),
                    "severity_ratio": float(np.nanmean(realised) / predicted),
                    "exceedance_rate": float(
                        np.mean([r["exceedance_rate"] for r in subsamples])
                    ),
                    "pvalue_z1": float(
                        np.nanmedian([r["pvalue_z1"] for r in subsamples])
                    ),
                    "pvalue_z2": float(
                        np.median([r["pvalue_z2"] for r in subsamples])
                    ),
                    "subsamples_agree": bool(result["consistent"]),
                }
            )

        table = pd.DataFrame(rows)
        table["pvalue_z2_holm"] = holm_adjust(table["pvalue_z2"])
        table["underestimates_tail"] = table["pvalue_z2_holm"] < 0.05
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Resumen por modelo, ordenado por calidad de calibracion de cola.

    ``severity_ratio`` es el diagnostico principal: cuanto mayor que uno, mas
    profundas fueron las perdidas realizadas frente al ES predicho. Por debajo de
    uno el modelo es conservador, que no se penaliza.
    """
    grouped = (
        table.groupby("model")
        .agg(
            severity_ratio=("severity_ratio", "mean"),
            predicted_es=("predicted_es", "mean"),
            realised_tail_mean=("realised_tail_mean", "mean"),
            exceedance_rate=("exceedance_rate", "mean"),
            assets_underestimating=("underestimates_tail", "sum"),
            assets=("asset", "count"),
            subsample_disagreements=("subsamples_agree", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    return grouped.sort_values("severity_ratio").reset_index(drop=True)


def portfolio_weights(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    main_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Las tres carteras del protocolo derivadas de una predictiva."""
    from mm_ipsa.portfolio.optimization import (
        maximum_sharpe,
        minimum_cvar,
        minimum_variance,
        weighted_mean_cov,
    )

    cfg = main_cfg["portfolio"]
    max_weight = float(cfg["max_weight"])
    l2_penalty = float(cfg.get("l2_penalty", 0.0))
    alpha = float(cfg["alpha_cvar"])
    mean, covariance = weighted_mean_cov(scenarios, probabilities)
    return {
        "MinVariance": minimum_variance(covariance, max_weight, l2_penalty),
        "MinCVaR": minimum_cvar(scenarios, probabilities, alpha, max_weight),
        "MaxSharpe": maximum_sharpe(
            mean,
            covariance,
            max_weight=max_weight,
            risk_free_rate=float(cfg["rf"]),
            l2_penalty=l2_penalty,
            seed=int(main_cfg["evaluation"]["seed"]),
        ),
    }


def portfolio_shortfall_table(
    observations: pd.DataFrame,
    predictives: dict[str, tuple[np.ndarray, np.ndarray]],
    main_cfg: dict[str, Any],
    *,
    n_simulations: int = 4_000,
    basel_level: float = 0.01,
) -> pd.DataFrame:
    """Calibracion de cola de cada cartera, que es la unidad que mira un regulador.

    Un modelo de riesgo no se valida activo por activo sino sobre la posicion que
    efectivamente se mantiene. La predictiva de la cartera es la proyeccion de los
    escenarios sobre sus pesos, y lo realizado es la misma proyeccion sobre los
    retornos observados.

    Se reporta ademas el semaforo de Basilea al ``basel_level``, con su propia
    declaracion de si la muestra alcanza para clasificar.
    """
    from mm_ipsa.evaluation.regulatory import basel_traffic_light
    from mm_ipsa.evaluation.scoring import weighted_quantile

    alpha = float(main_cfg["portfolio"]["alpha_cvar"])
    horizon = int(main_cfg["data"]["H"])
    realised_matrix = observations.to_numpy()
    rows: list[dict[str, Any]] = []

    for model, (scenarios, probabilities) in predictives.items():
        for strategy, weights in portfolio_weights(
            scenarios, probabilities, main_cfg
        ).items():
            projected_scenarios = scenarios @ weights
            projected_observations = realised_matrix @ weights

            result = disjoint_subsample_backtests(
                projected_observations,
                projected_scenarios,
                probabilities,
                alpha,
                horizon,
                n_simulations=n_simulations,
                seed=int(main_cfg["evaluation"]["seed"]),
            )
            subsamples = result["subsamples"]
            realised = np.array(
                [r["realised_tail_mean"] for r in subsamples], dtype=float
            )
            predicted = float(subsamples[0]["predicted_es"])

            # El semaforo se evalua sobre una submuestra disjunta, porque las
            # excepciones solapadas no son ensayos independientes.
            independent = projected_observations[::horizon]
            basel_var = weighted_quantile(
                projected_scenarios, probabilities, basel_level
            )
            exceptions = int(np.sum(independent < basel_var))
            traffic = basel_traffic_light(
                exceptions, len(independent), basel_level
            )

            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "predicted_es": predicted,
                    "realised_tail_mean": float(np.nanmean(realised)),
                    "severity_ratio": float(np.nanmean(realised) / predicted),
                    "pvalue_z2": float(
                        np.median([r["pvalue_z2"] for r in subsamples])
                    ),
                    "basel_var": float(basel_var),
                    "basel_exceptions": exceptions,
                    "basel_expected": traffic["expected_exceptions"],
                    "basel_zone": traffic["zone"],
                    "basel_sample_adequate": traffic["sample_is_adequate"],
                    "max_weight": float(np.max(weights)),
                }
            )

    return pd.DataFrame(rows)


def run_shortfall_report(
    main_cfg: dict[str, Any],
    observations: pd.DataFrame,
    source: str | Path,
    output: str | Path,
    daily: pd.DataFrame,
    *,
    n_simulations: int = 4_000,
) -> dict[str, pd.DataFrame]:
    """Genera y escribe el informe completo de calibracion de cola."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)

    predictives = load_model_predictives(source)
    table = shortfall_table(
        observations,
        predictives,
        float(main_cfg["portfolio"]["alpha_cvar"]),
        int(main_cfg["data"]["H"]),
        n_simulations=n_simulations,
        seed=int(main_cfg["evaluation"]["seed"]),
    )
    summary = summarise(table)
    portfolios = portfolio_shortfall_table(
        observations, predictives, main_cfg, n_simulations=n_simulations
    )
    portfolios["pvalue_z2_holm"] = holm_adjust(portfolios["pvalue_z2"])
    portfolios = portfolios.sort_values("severity_ratio").reset_index(drop=True)

    table.to_csv(target / "expected_shortfall_by_asset.csv", index=False)
    summary.to_csv(target / "expected_shortfall_summary.csv", index=False)
    attribution = attribution_table(observations, predictives, main_cfg)
    procyclicality = procyclicality_table(daily, observations, predictives, main_cfg)

    portfolios.to_csv(target / "expected_shortfall_by_portfolio.csv", index=False)
    attribution.to_csv(target / "tail_attribution.csv", index=False)
    procyclicality.to_csv(target / "procyclicality.csv", index=False)
    return {
        "by_asset": table,
        "summary": summary,
        "by_portfolio": portfolios,
        "attribution": attribution,
        "procyclicality": procyclicality,
    }


def attribution_table(
    observations: pd.DataFrame,
    predictives: dict[str, tuple[np.ndarray, np.ndarray]],
    main_cfg: dict[str, Any],
    *,
    alpha: float = 0.20,
) -> pd.DataFrame:
    """Acierto de cada modelo sobre la composicion de la cola, no su magnitud.

    Se evalua sobre las submuestras disjuntas del horizonte y se promedia, porque
    una sola eleccion de desplazamiento da una composicion realizada demasiado
    ruidosa para leerla.

    El ``alpha`` por defecto es mas ancho que el del protocolo: con la cola del
    cinco por ciento quedan menos de diez observaciones realizadas y la
    composicion deja de ser estimable.
    """
    from mm_ipsa.evaluation.attribution import (
        compare_attribution,
        component_expected_shortfall,
        realised_component_shortfall,
    )

    horizon = int(main_cfg["data"]["H"])
    matrix = observations.to_numpy()
    rows: list[dict[str, Any]] = []

    for model, (scenarios, probabilities) in predictives.items():
        for strategy, weights in portfolio_weights(
            scenarios, probabilities, main_cfg
        ).items():
            predicted = component_expected_shortfall(
                scenarios, probabilities, weights, alpha
            )
            comparisons = [
                compare_attribution(
                    predicted,
                    realised_component_shortfall(
                        matrix[offset::horizon], weights, alpha
                    ),
                )
                for offset in range(horizon)
            ]
            correlations = [
                c["conditional_rank_correlation"] for c in comparisons
            ]
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "alpha": alpha,
                    "conditional_rank_correlation": float(np.mean(correlations)),
                    "correlation_spread": float(
                        max(correlations) - min(correlations)
                    ),
                    "share_rank_correlation": float(
                        np.mean([c["share_rank_correlation"] for c in comparisons])
                    ),
                    "composition_error": float(
                        np.mean([c["composition_error"] for c in comparisons])
                    ),
                    "reliable": all(c["reliable"] for c in comparisons),
                }
            )

    return pd.DataFrame(rows)


def procyclicality_table(
    daily: pd.DataFrame,
    observations: pd.DataFrame,
    predictives: dict[str, tuple[np.ndarray, np.ndarray]],
    main_cfg: dict[str, Any],
    *,
    lookback: int = 63,
    quantile: float = 0.75,
) -> pd.DataFrame:
    """Concentracion de excesos por regimen de volatilidad, para cada cartera.

    Se evalua sobre ventanas disjuntas porque un exceso solapado no es un ensayo
    nuevo, y el regimen se determina con volatilidad anterior al inicio de cada
    ventana.
    """
    from mm_ipsa.evaluation.procyclicality import (
        STRESSED,
        breach_concentration,
        classify_regimes,
        trailing_volatility,
    )
    from mm_ipsa.evaluation.scoring import weighted_quantile

    horizon = int(main_cfg["data"]["H"])
    alpha = float(main_cfg["portfolio"]["alpha_cvar"])
    disjoint = observations.iloc[::horizon]
    ends = cast(pd.DatetimeIndex, pd.DatetimeIndex(disjoint.index))

    volatility = trailing_volatility(daily, ends, horizon, lookback)
    regimes = classify_regimes(volatility, quantile)
    labelled = regimes.notna().to_numpy()
    classified = regimes.dropna()

    stressed = (classified == STRESSED).to_numpy()
    episodes = int(np.sum(stressed[1:] & ~stressed[:-1]) + int(stressed[0]))

    matrix = disjoint.to_numpy()
    rows: list[dict[str, Any]] = []
    for model, (scenarios, probabilities) in predictives.items():
        for strategy, weights in portfolio_weights(
            scenarios, probabilities, main_cfg
        ).items():
            var = weighted_quantile(scenarios @ weights, probabilities, alpha)
            result = breach_concentration(
                (matrix @ weights)[labelled], classified, var, alpha
            )
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "calm_breach_rate": result["calm"]["breach_rate"],
                    "stressed_breach_rate": result["stressed"]["breach_rate"],
                    "rate_ratio": result["rate_ratio"],
                    "pvalue": result["pvalue"],
                    "total_breaches": result["total_breaches"],
                    "reliable": result["reliable"],
                    "stress_episodes": episodes,
                }
            )

    table = pd.DataFrame(rows)
    table["pvalue_holm"] = holm_adjust(table["pvalue"])
    return table.sort_values("stressed_breach_rate").reset_index(drop=True)
