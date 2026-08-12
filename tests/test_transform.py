import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.transform import _max_true_run, _rolling_terminal_fast, build_returns, non_overlapping_windows


class TestTransform(unittest.TestCase):
    def test_terminal_return_compounds_exactly(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        returns = pd.DataFrame({"x": [0.01, -0.02, 0.03, 0.00, 0.01, -0.01]}, index=dates)
        result = _rolling_terminal_fast(returns, 3)
        expected = np.array(
            [
                (1.01 * 0.98 * 1.03) - 1,
                (0.98 * 1.03 * 1.00) - 1,
                (1.03 * 1.00 * 1.01) - 1,
                (1.00 * 1.01 * 0.99) - 1,
            ]
        )
        np.testing.assert_allclose(result["x"].to_numpy(), expected)

    def test_non_overlapping_uses_stride_h(self):
        frame = pd.DataFrame({"x": np.arange(12)})
        self.assertEqual(non_overlapping_windows(frame, 5)["x"].tolist(), [0, 5, 10])

    def test_maximum_zero_run(self):
        values = pd.Series([False, True, True, False, True, True, True])
        self.assertEqual(_max_true_run(values), 3)

    def test_imputed_price_endpoints_are_excluded(self):
        dates = pd.date_range("2023-12-27", periods=10, freq="B")
        prices = pd.DataFrame(
            {"A": 100.0 + np.arange(10), "B": 200.0 + 2 * np.arange(10)}, index=dates
        )
        mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
        mask.loc[dates[2], "A"] = False
        with tempfile.TemporaryDirectory() as directory:
            cfg = {
                "asset_labels": ["A", "B"],
                "data": {
                    "H": 2,
                    "end_train": "2023-12-31",
                    "start_oos": "2024-01-01",
                    "exclude_imputed_return_endpoints": True,
                    "max_zero_return_rate": 1.0,
                    "max_consecutive_zero_returns": 10,
                },
                "paths": {"data_raw": directory},
            }
            daily_is, _, daily_oos, _ = build_returns(cfg, prices, observation_mask=mask)
            self.assertNotIn(dates[2], daily_is.index)
            self.assertNotIn(dates[3], daily_oos.index)


if __name__ == "__main__":
    unittest.main()
