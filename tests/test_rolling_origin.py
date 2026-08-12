import unittest

import numpy as np
import pandas as pd

from mm_ipsa.analysis.rolling_origin import (
    build_fold_samples,
    combine_daily_returns,
    load_rolling_origin_config,
)


class TestRollingOrigin(unittest.TestCase):
    def test_protocol_has_ordered_disjoint_folds(self):
        cfg = load_rolling_origin_config("research/rolling_origin.yaml")
        self.assertEqual(len(cfg["folds"]), 4)
        self.assertEqual(cfg["experiment"]["window_type"], "expanding")
        self.assertTrue(cfg["validation"]["refit_all_models_each_fold"])

    def test_fold_samples_never_cross_origin(self):
        index = pd.bdate_range("2020-01-02", periods=900)
        daily = pd.DataFrame(
            np.full((len(index), 2), 0.001),
            index=index,
            columns=["A", "B"],
        )
        fold = {
            "fold_id": "test",
            "train_end": str(index[699].date()),
            "evaluation_start": str(index[700].date()),
            "evaluation_end": str(index[849].date()),
        }
        train_daily, train_terminal, evaluation = build_fold_samples(
            daily,
            fold,
            5,
            minimum_training_daily_rows=500,
            minimum_evaluation_terminal_rows=20,
        )
        self.assertLess(train_daily.index.max(), evaluation.index.min())
        self.assertLessEqual(train_terminal.index.max(), pd.Timestamp(fold["train_end"]))
        self.assertGreaterEqual(evaluation.index.min(), pd.Timestamp(fold["evaluation_start"]))
        self.assertEqual(len(evaluation), 30)

    def test_combined_panel_rejects_duplicate_dates(self):
        index = pd.bdate_range("2024-01-02", periods=3)
        frame = pd.DataFrame({"A": [0.0, 0.1, 0.2]}, index=index)
        with self.assertRaises(ValueError):
            combine_daily_returns(frame, frame.iloc[-1:], ["A"])


if __name__ == "__main__":
    unittest.main()
