import unittest

import numpy as np
import pandas as pd

from mm_ipsa.pipeline import (
    STEP_LINEAGE,
    _adjust_portfolio_multiplicity,
    _execution_sequence,
    _model_confidence_sets,
    _read_returns,
)


class TestExecutionSequence(unittest.TestCase):
    def test_all_expands_to_the_full_ordered_pipeline(self):
        sequence = _execution_sequence("all")
        self.assertEqual(sequence[0], "download")
        self.assertEqual(sequence[-1], "snapshot")
        # Cada etapa debe aparecer una sola vez y en orden de dependencia.
        self.assertEqual(len(sequence), len(set(sequence)))
        self.assertLess(sequence.index("transform"), sequence.index("mm"))
        self.assertLess(sequence.index("mm"), sequence.index("benchmarks"))
        self.assertLess(sequence.index("benchmarks"), sequence.index("evaluate"))
        self.assertLess(sequence.index("portfolio"), sequence.index("backtest"))

    def test_single_step_is_returned_alone(self):
        self.assertEqual(_execution_sequence("evaluate"), ("evaluate",))

    def test_every_pipeline_step_has_a_lineage_stage(self):
        # Una etapa sin manifiesto no puede reanudarse ni verificarse; la
        # excepcion deliberada es snapshot, que consolida a las demas.
        for step in _execution_sequence("all"):
            if step == "snapshot":
                continue
            self.assertIn(step, STEP_LINEAGE, f"{step} no declara etapa de linaje")


class TestReadReturns(unittest.TestCase):
    def _write(self, directory, name, frame):
        path = directory / name
        frame.to_csv(path)
        return path

    def test_reorders_columns_to_the_declared_labels(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            index = pd.date_range("2024-01-01", periods=4, freq="B")
            frame = pd.DataFrame(
                {"b": [0.1, 0.2, 0.3, 0.4], "a": [1.0, 2.0, 3.0, 4.0]}, index=index
            )
            path = self._write(Path(directory), "returns.csv", frame)
            result = _read_returns(path, ["a", "b"])
            self.assertEqual(list(result.columns), ["a", "b"])
            np.testing.assert_allclose(result["a"].to_numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_missing_label_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            index = pd.date_range("2024-01-01", periods=3, freq="B")
            frame = pd.DataFrame({"a": [0.1, 0.2, 0.3]}, index=index)
            path = self._write(Path(directory), "returns.csv", frame)
            with self.assertRaises(Exception):
                _read_returns(path, ["a", "ausente"])


class TestPortfolioMultiplicity(unittest.TestCase):
    def _frame(self, designs, pvalues):
        return pd.DataFrame(
            {
                "portfolio": [f"p{i}" for i in range(len(pvalues))],
                "evaluation_design": designs,
                "pvalue_raw": pvalues,
                "ci_low": [-0.1] * len(pvalues),
                "ci_high": [0.3] * len(pvalues),
            }
        )

    def test_holm_is_applied_within_each_family(self):
        frame = self._frame(
            ["static_single_shot_oos"] * 3 + ["rebalancing_effect"] * 3,
            [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
        )
        adjusted = _adjust_portfolio_multiplicity(frame)
        # Con tres contrastes por familia, el menor p se multiplica por 3 en
        # ambas; si Holm se aplicara al conjunto seria por 6.
        smallest = adjusted.loc[adjusted["pvalue_raw"] == 0.01, "pvalue_holm"]
        self.assertTrue(np.allclose(smallest.to_numpy(), 0.03))

    def test_adjusted_pvalues_never_decrease(self):
        frame = self._frame(["static_single_shot_oos"] * 4, [0.001, 0.02, 0.2, 0.9])
        adjusted = _adjust_portfolio_multiplicity(frame)
        self.assertTrue((adjusted["pvalue_holm"] >= adjusted["pvalue_raw"]).all())

    def test_rejection_and_interval_flags_are_consistent(self):
        frame = self._frame(["static_single_shot_oos"] * 2, [0.001, 0.5])
        frame.loc[0, "ci_low"] = 0.05
        frame.loc[0, "ci_high"] = 0.4
        adjusted = _adjust_portfolio_multiplicity(frame)
        self.assertTrue(bool(adjusted.loc[0, "ci_excludes_zero"]))
        self.assertFalse(bool(adjusted.loc[1, "ci_excludes_zero"]))
        self.assertEqual(
            adjusted["multiple_testing"].unique().tolist(),
            ["holm_within_evaluation_design"],
        )

    def test_empty_frame_is_returned_untouched(self):
        empty = pd.DataFrame()
        self.assertTrue(_adjust_portfolio_multiplicity(empty).empty)


class TestModelConfidenceSetsHelper(unittest.TestCase):
    def _scores(self, models, rows, group_column=None):
        rng = np.random.default_rng(3)
        common = rng.standard_normal(rows) * 0.001
        frames = []
        for name, offset in models.items():
            base = 0.02 + offset + common
            frame = pd.DataFrame(
                {
                    "model": name,
                    "observation": [f"2024-01-{i:03d}" for i in range(rows)],
                    "mean_crps": base,
                    "energy_score": 2 * base,
                    "variogram_score": 3 * base,
                }
            )
            if group_column is not None:
                frame[group_column] = np.repeat(["f1", "f2"], rows // 2)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_returns_one_block_per_metric(self):
        scores = self._scores({"MM": 0.0, "control": 0.002}, 40)
        result = _model_confidence_sets(
            scores, block_sizes={"mean_crps": 2}, alpha=0.05, samples=200, seed=1
        )
        self.assertEqual(
            sorted(result["metric"].unique()),
            ["energy_score", "mean_crps", "variogram_score"],
        )
        self.assertEqual(len(result), 6)

    def test_grouped_variant_respects_fold_column(self):
        scores = self._scores({"MM": 0.0, "control": 0.002}, 40, group_column="fold_id")
        result = _model_confidence_sets(
            scores,
            block_sizes={"mean_crps": 2},
            alpha=0.05,
            samples=200,
            seed=1,
            group_column="fold_id",
        )
        self.assertEqual(len(result), 6)
        self.assertTrue((result["block_size"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
