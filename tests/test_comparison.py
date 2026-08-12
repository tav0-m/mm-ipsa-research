import unittest

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.comparison import (
    compare_focal_model,
    compare_focal_model_by_group,
    grouped_moving_block_bootstrap_loss_difference,
    holm_adjust,
    moving_block_bootstrap_loss_difference,
)


class TestScoreComparison(unittest.TestCase):
    def test_identical_losses_have_zero_interval_and_no_rejection(self):
        losses = np.linspace(0.1, 0.2, 30)
        result = moving_block_bootstrap_loss_difference(
            losses,
            losses,
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=7,
        )
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)
        self.assertEqual(result["pvalue_raw"], 1.0)

    def test_negative_difference_means_focal_is_better(self):
        benchmark = np.linspace(1.0, 2.0, 40)
        focal = benchmark - 0.2
        result = moving_block_bootstrap_loss_difference(
            focal,
            benchmark,
            block_size=5,
            samples=500,
            confidence_level=0.95,
            seed=11,
        )
        self.assertLess(result["mean_difference"], 0.0)
        self.assertLess(result["ci_high"], 0.0)
        self.assertGreater(result["probability_focal_better"], 0.99)

    def test_interval_and_equal_tail_test_are_directionally_coherent(self):
        rng = np.random.default_rng(101)
        benchmark = rng.normal(1.0, 0.1, size=80)
        focal = benchmark + rng.normal(-0.08, 0.01, size=80)
        result = moving_block_bootstrap_loss_difference(
            focal,
            benchmark,
            block_size=4,
            samples=1_000,
            confidence_level=0.95,
            seed=19,
        )
        self.assertLess(result["ci_high"], 0.0)
        self.assertLess(result["pvalue_raw"], 0.05)

    def test_holm_adjustment_is_valid_and_monotone(self):
        raw = np.array([0.001, 0.02, 0.04, 0.8])
        adjusted = holm_adjust(raw)
        self.assertTrue(np.all(adjusted >= raw))
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-15))

    def test_comparison_builds_nine_paired_contrasts(self):
        observations = [f"t{i:02d}" for i in range(24)]
        rows = []
        offsets = {
            "MM": 0.00,
            "gaussian_terminal": 0.02,
            "student_t_terminal": -0.01,
            "historical_weighted": 0.01,
        }
        for model, offset in offsets.items():
            for index, observation in enumerate(observations):
                base = 0.1 + index / 1_000
                rows.append(
                    {
                        "model": model,
                        "observation": observation,
                        "mean_crps": base + offset,
                        "energy_score": 2 * base + offset,
                        "variogram_score": 3 * base + offset,
                    }
                )
        frame = compare_focal_model(
            pd.DataFrame(rows),
            focal_model="MM",
            benchmark_models=[
                "gaussian_terminal",
                "student_t_terminal",
                "historical_weighted",
            ],
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=5,
        )
        self.assertEqual(len(frame), 9)
        self.assertEqual(int(frame["registered_primary"].sum()), 1)
        self.assertTrue((frame["difference_direction"] == "focal_minus_benchmark").all())
        self.assertTrue((frame["pvalue_holm"] >= frame["pvalue_raw"]).all())

    def test_grouped_bootstrap_preserves_zero_difference(self):
        losses = np.linspace(0.1, 0.3, 24)
        groups = np.repeat(["fold_a", "fold_b"], 12)
        result = grouped_moving_block_bootstrap_loss_difference(
            losses,
            losses,
            groups,
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=31,
        )
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)
        self.assertEqual(result["n_groups"], 2.0)

    def test_grouped_comparison_builds_nine_contrasts(self):
        rows = []
        offsets = {
            "MM": 0.00,
            "gaussian_terminal": 0.02,
            "student_t_terminal": -0.01,
            "historical_weighted": 0.01,
        }
        for fold in ("a", "b"):
            for model, offset in offsets.items():
                for index in range(12):
                    base = 0.1 + index / 1_000
                    rows.append(
                        {
                            "fold_id": fold,
                            "model": model,
                            "observation": f"{fold}-{index:02d}",
                            "mean_crps": base + offset,
                            "energy_score": 2 * base + offset,
                            "variogram_score": 3 * base + offset,
                        }
                    )
        frame = compare_focal_model_by_group(
            pd.DataFrame(rows),
            group_column="fold_id",
            focal_model="MM",
            benchmark_models=[
                "gaussian_terminal",
                "student_t_terminal",
                "historical_weighted",
            ],
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=37,
        )
        self.assertEqual(len(frame), 9)
        self.assertTrue((frame["n_groups"] == 2.0).all())
        self.assertTrue((frame["n_observations"] == 24).all())
        self.assertEqual(int(frame["registered_primary"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
