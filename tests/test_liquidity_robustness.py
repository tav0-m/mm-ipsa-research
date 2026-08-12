import unittest

import pandas as pd

from src.analysis.liquidity_robustness import select_liquid_universe


class TestLiquidityRobustness(unittest.TestCase):
    def setUp(self):
        self.labels = ["A", "B", "C", "D"]
        self.quality = pd.DataFrame(
            {
                "asset": self.labels,
                "zero_return_rate_is": [0.01, 0.031, 0.02, 0.015],
                "max_zero_run_is": [2, 2, 6, 1],
                "zero_return_rate_oos": [0.90, 0.00, 0.00, 0.75],
                "max_zero_run_oos": [99, 0, 0, 50],
            }
        )
        self.selection = {
            "max_zero_return_rate_is": 0.03,
            "max_consecutive_zero_returns_is": 5,
            "min_assets": 2,
        }

    def test_oos_metrics_cannot_change_selection(self):
        selected, audit = select_liquid_universe(
            self.quality, self.labels, self.selection
        )
        mutated = self.quality.copy()
        mutated["zero_return_rate_oos"] = [0.0, 1.0, 1.0, 0.0]
        mutated["max_zero_run_oos"] = [0, 999, 999, 0]
        selected_mutated, audit_mutated = select_liquid_universe(
            mutated, self.labels, self.selection
        )

        self.assertEqual(selected, ["A", "D"])
        self.assertEqual(selected_mutated, selected)
        self.assertListEqual(audit["selected"].tolist(), audit_mutated["selected"].tolist())
        self.assertTrue((audit["selection_uses_oos"] == False).all())  # noqa: E712

    def test_minimum_universe_guard(self):
        strict = {**self.selection, "min_assets": 3}
        with self.assertRaisesRegex(RuntimeError, "menos que min_assets"):
            select_liquid_universe(self.quality, self.labels, strict)

    def test_missing_is_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Faltan columnas"):
            select_liquid_universe(
                self.quality.drop(columns="zero_return_rate_is"),
                self.labels,
                self.selection,
            )


if __name__ == "__main__":
    unittest.main()
