import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mm_ipsa.analysis.recalibration_ablation import (
    _first_fold_is_identical,
    _windows_per_model,
    run_recalibration_ablation,
)


def _daily(rows: int, assets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(rows) * 0.007
    data = np.column_stack(
        [common + rng.standard_normal(rows) * 0.009 for _ in range(assets)]
    )
    index = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(assets)])


def _main_config(labels: list[str]) -> dict:
    return {
        "asset_labels": labels,
        "data": {"H": 5},
        "mm": {
            "N_scenarios": 30,
            "bcd_max_iter": 10,
            "tol": 1e-4,
            "min_iter": 1,
            "convergence_patience": 2,
            "n_starts": 2,
            "seed": 4,
            "entropy_lambda": 1e-5,
            "update_mode": "joint_lbfgs",
            "p_solver": "mirror_descent",
            "start_strategy": "independent",
            "solution_mode": "ensemble",
            "strict_solver": False,
            "moment_weights": {"k1": 1.0, "k2": 1.0, "k3": 0.5, "k4": 0.25,
                               "cov_weight": 0.05},
            "covariance_shrinkage": "auto",
            "ewma_half_life_weeks": 11.0,
            "observations_per_week": 5.0,
        },
        "benchmarks": {
            "N_scenarios": 300,
            "seed": 9,
            "student_t_df_mode": "fixed",
            "student_t_df": 8.0,
            "include": ["gaussian_terminal"],
        },
        "portfolio": {"alpha_cvar": 0.1},
        "evaluation": {
            "seed": 5,
            "energy_pair_samples": 2_000,
            "score_bootstrap_samples": 300,
            "score_bootstrap_block_size": 2,
            "score_bootstrap_block_size_mode": "auto",
            "score_bootstrap_confidence": 0.95,
            "status": "development_validation",
        },
    }


def _experiment_config() -> dict:
    return {
        "validation": {
            "minimum_training_daily_rows": 200,
            "minimum_evaluation_terminal_rows": 8,
        },
        "folds": [
            {"fold_id": "F1", "train_end": "2021-06-30",
             "evaluation_start": "2021-07-12", "evaluation_end": "2021-12-31"},
            {"fold_id": "F2", "train_end": "2022-06-30",
             "evaluation_start": "2022-07-12", "evaluation_end": "2022-12-30"},
            {"fold_id": "F3", "train_end": "2023-06-30",
             "evaluation_start": "2023-07-12", "evaluation_end": "2023-12-29"},
        ],
    }


class TestRecalibrationAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daily = _daily(1_100, 3, 11)
        cls.labels = list(daily.columns)
        cls.directory = tempfile.TemporaryDirectory()
        cls.result = run_recalibration_ablation(
            _main_config(cls.labels),
            _experiment_config(),
            daily,
            Path(cls.directory.name),
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_first_fold_is_identical_by_construction(self):
        # La variante congelada ES la calibración del primer origen, de modo que
        # una diferencia allí delataría un error de montaje.
        self.assertTrue(self.result["metadata"]["first_fold_variants_identical"])

    def test_both_variants_score_the_same_windows(self):
        scores = self.result["scores"]
        counts = scores.groupby(["variant", "model"]).size().unstack("variant")
        self.assertTrue((counts["frozen"] == counts["refit"]).all())

    def test_comparison_excludes_the_first_fold(self):
        metadata = self.result["metadata"]
        self.assertNotIn("F1", metadata["folds_compared"])
        self.assertLess(metadata["windows_compared"], metadata["windows_total"])

    def test_effect_table_covers_every_model_and_metric(self):
        differences = self.result["differences"]
        self.assertEqual(
            len(differences),
            len(self.result["metadata"]["models"]) * 3,
        )
        self.assertTrue(
            (differences["difference_direction"] == "frozen_minus_refit").all()
        )

    def test_adjusted_pvalues_never_decrease(self):
        differences = self.result["differences"]
        self.assertTrue(
            (differences["pvalue_holm"] + 1e-12 >= differences["pvalue_raw"]).all()
        )

    def test_recalibration_flag_requires_significance_and_sign(self):
        differences = self.result["differences"]
        flagged = differences.loc[differences["recalibration_helps"]]
        self.assertTrue((flagged["mean_difference"] > 0).all())
        self.assertTrue((flagged["pvalue_holm"] < 0.05).all())

    def test_artifacts_are_written(self):
        target = Path(self.directory.name)
        for name in (
            "ablation_scores_by_observation.csv",
            "ablation_scores_pooled.csv",
            "ablation_recalibration_effect.csv",
            "ablation_metadata.json",
        ):
            self.assertTrue((target / name).is_file(), name)

    def test_rejects_a_single_fold_experiment(self):
        config = _experiment_config()
        config["folds"] = config["folds"][:1]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_recalibration_ablation(
                    _main_config(self.labels), config, _daily(600, 3, 2), directory
                )


class TestAblationHelpers(unittest.TestCase):
    def test_windows_per_model_counts_one_variant(self):
        scores = pd.DataFrame(
            {
                "variant": ["refit"] * 4 + ["frozen"] * 4,
                "model": ["a", "a", "b", "b"] * 2,
                "observation": ["t1", "t2"] * 4,
            }
        )
        self.assertEqual(_windows_per_model(scores), 2)

    def test_windows_per_model_handles_an_empty_frame(self):
        empty = pd.DataFrame({"variant": [], "model": []})
        self.assertEqual(_windows_per_model(empty), 0)

    def test_identity_check_detects_a_divergence(self):
        base = pd.DataFrame(
            {
                "variant": ["frozen", "refit"],
                "fold_id": ["F1", "F1"],
                "model": ["a", "a"],
                "observation": ["t1", "t1"],
                "mean_crps": [0.1, 0.1],
            }
        )
        self.assertTrue(_first_fold_is_identical(base, "F1"))
        diverged = base.copy()
        diverged.loc[0, "mean_crps"] = 0.2
        self.assertFalse(_first_fold_is_identical(diverged, "F1"))


if __name__ == "__main__":
    unittest.main()
