import unittest
from typing import cast

import numpy as np
import pandas as pd

from mm_ipsa.analysis.shortfall_report import shortfall_table, summarise

ALPHA = 0.05
HORIZON = 5
ASSETS = ["A", "B", "C"]


def _predictive(scale: float, seed: int, size: int = 2_000):
    draws = np.random.default_rng(seed).standard_normal((size, len(ASSETS))) * scale
    return draws, np.full(size, 1.0 / size)


def _overlapping_observations(scale: float, seed: int, rows: int = 300):
    rng = np.random.default_rng(seed)
    daily = rng.standard_normal((rows + HORIZON, len(ASSETS))) * scale / np.sqrt(HORIZON)
    terminal = np.array(
        [daily[i : i + HORIZON].sum(axis=0) for i in range(rows)]
    )
    index = pd.bdate_range("2024-01-01", periods=rows)
    return pd.DataFrame(terminal, index=index, columns=ASSETS)


class TestShortfallTable(unittest.TestCase):
    def setUp(self):
        self.observations = _overlapping_observations(0.02, seed=1)

    def test_produces_one_row_per_model_and_asset(self):
        predictives = {
            "narrow": _predictive(0.014, seed=2),
            "wide": _predictive(0.030, seed=3),
        }
        table = shortfall_table(
            self.observations, predictives, ALPHA, HORIZON, n_simulations=400
        )
        self.assertEqual(len(table), 2 * len(ASSETS))
        self.assertEqual(set(table["model"]), {"narrow", "wide"})

    def test_severity_ratio_separates_narrow_from_conservative(self):
        predictives = {
            "narrow": _predictive(0.014, seed=4),
            "wide": _predictive(0.032, seed=5),
        }
        summary = summarise(
            shortfall_table(
                self.observations, predictives, ALPHA, HORIZON, n_simulations=400
            )
        ).set_index("model")

        self.assertGreater(cast(float, summary.loc["narrow", "severity_ratio"]), 1.0)
        self.assertLess(cast(float, summary.loc["wide", "severity_ratio"]), 1.0)

    def test_holm_adjustment_is_never_smaller_than_the_raw_pvalue(self):
        table = shortfall_table(
            self.observations,
            {"model": _predictive(0.018, seed=6)},
            ALPHA,
            HORIZON,
            n_simulations=400,
        )
        self.assertTrue((table["pvalue_z2_holm"] >= table["pvalue_z2"] - 1e-12).all())

    def test_rejects_a_predictive_with_the_wrong_asset_count(self):
        scenarios, probabilities = _predictive(0.02, seed=7)
        with self.assertRaises(ValueError):
            shortfall_table(
                self.observations,
                {"broken": (scenarios[:, :2], probabilities)},
                ALPHA,
                HORIZON,
                n_simulations=100,
            )


class TestSummary(unittest.TestCase):
    def test_orders_models_from_most_to_least_conservative(self):
        observations = _overlapping_observations(0.02, seed=8)
        predictives = {
            "wide": _predictive(0.030, seed=9),
            "narrow": _predictive(0.014, seed=10),
            "fair": _predictive(0.020, seed=11),
        }
        summary = summarise(
            shortfall_table(
                observations, predictives, ALPHA, HORIZON, n_simulations=400
            )
        )
        self.assertEqual(list(summary["model"])[0], "wide")
        self.assertEqual(list(summary["model"])[-1], "narrow")
        self.assertTrue(summary["severity_ratio"].is_monotonic_increasing)



class TestPortfolioShortfall(unittest.TestCase):
    """La unidad que valida un regulador es la cartera, no el activo aislado."""

    def _config(self):
        return {
            "portfolio": {
                "max_weight": 0.5,
                "l2_penalty": 0.0,
                "alpha_cvar": ALPHA,
                "rf": 0.0,
            },
            "data": {"H": HORIZON},
            "evaluation": {"seed": 0},
        }

    def test_produces_one_row_per_model_and_strategy(self):
        from mm_ipsa.analysis.shortfall_report import portfolio_shortfall_table

        observations = _overlapping_observations(0.02, seed=20)
        table = portfolio_shortfall_table(
            observations,
            {"a": _predictive(0.02, seed=21), "b": _predictive(0.025, seed=22)},
            self._config(),
            n_simulations=300,
        )

        self.assertEqual(len(table), 2 * 3)
        self.assertEqual(
            set(table["strategy"]), {"MinVariance", "MinCVaR", "MaxSharpe"}
        )

    def test_weights_respect_the_configured_cap(self):
        from mm_ipsa.analysis.shortfall_report import portfolio_shortfall_table

        table = portfolio_shortfall_table(
            _overlapping_observations(0.02, seed=23),
            {"a": _predictive(0.02, seed=24)},
            self._config(),
            n_simulations=300,
        )
        self.assertTrue((table["max_weight"] <= 0.5 + 1e-6).all())

    def test_a_narrow_predictive_shows_severity_above_one(self):
        from mm_ipsa.analysis.shortfall_report import portfolio_shortfall_table

        table = portfolio_shortfall_table(
            _overlapping_observations(0.030, seed=25),
            {"narrow": _predictive(0.012, seed=26)},
            self._config(),
            n_simulations=300,
        )
        self.assertTrue((table["severity_ratio"] > 1.0).all())

    def test_reports_basel_sample_adequacy(self):
        from mm_ipsa.analysis.shortfall_report import portfolio_shortfall_table

        table = portfolio_shortfall_table(
            _overlapping_observations(0.02, seed=27),
            {"a": _predictive(0.02, seed=28)},
            self._config(),
            n_simulations=300,
        )
        # 300 filas solapadas dejan 60 independientes: 0.6 excepciones esperadas.
        self.assertFalse(bool(table["basel_sample_adequate"].iloc[0]))

if __name__ == "__main__":
    unittest.main()
