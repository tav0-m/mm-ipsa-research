import unittest

import numpy as np
import pandas as pd

from mm_ipsa.analysis.objective_ablation import (
    DEFAULT_VARIANTS,
    SCORING_RULES,
    _apply_overrides,
    _relative_error,
    compare_against_reference,
)


def _settings() -> dict:
    return {
        "N_scenarios": 500,
        "cov_weight": 0.05,
        "moment_weights": {"k1": 2.0, "k2": 1.5, "k3": 0.5, "k4": 0.3},
    }


def _losses(scale: float, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {rule: np.abs(rng.standard_normal(rows)) * scale for rule in SCORING_RULES}
    )


class TestApplyOverrides(unittest.TestCase):
    def test_merges_moment_weights_key_by_key(self):
        updated = _apply_overrides(_settings(), {"moment_weights": {"k3": 0.0}})
        self.assertEqual(updated["moment_weights"]["k3"], 0.0)
        self.assertEqual(updated["moment_weights"]["k1"], 2.0)
        self.assertEqual(updated["moment_weights"]["k4"], 0.3)

    def test_replaces_scalar_parameters(self):
        updated = _apply_overrides(_settings(), {"cov_weight": 20.0})
        self.assertEqual(updated["cov_weight"], 20.0)

    def test_does_not_mutate_the_source(self):
        original = _settings()
        _apply_overrides(original, {"cov_weight": 99.0, "moment_weights": {"k3": 9.0}})
        self.assertEqual(original["cov_weight"], 0.05)
        self.assertEqual(original["moment_weights"]["k3"], 0.5)

    def test_an_empty_override_is_the_identity(self):
        self.assertEqual(_apply_overrides(_settings(), {}), _settings())


class TestRelativeError(unittest.TestCase):
    def test_zero_for_an_exact_match(self):
        target = np.array([1.0, -2.0, 3.0])
        self.assertEqual(_relative_error(target, target), 0.0)

    def test_reports_the_worst_component(self):
        target = np.array([1.0, 10.0])
        fitted = np.array([1.05, 10.1])
        self.assertAlmostEqual(_relative_error(fitted, target), 0.05, places=12)

    def test_survives_a_target_of_zero(self):
        value = _relative_error(np.array([1e-9]), np.array([0.0]))
        self.assertTrue(np.isfinite(value))


class TestDefaultVariants(unittest.TestCase):
    def test_includes_the_reference_and_both_open_questions(self):
        self.assertIn("publicada", DEFAULT_VARIANTS)
        self.assertEqual(DEFAULT_VARIANTS["publicada"], {})
        self.assertIn("dependencia_alta", DEFAULT_VARIANTS)
        self.assertIn("sin_momentos_superiores", DEFAULT_VARIANTS)

    def test_the_ablation_actually_zeroes_the_higher_moments(self):
        weights = DEFAULT_VARIANTS["sin_momentos_superiores"]["moment_weights"]
        self.assertEqual(weights["k3"], 0.0)
        self.assertEqual(weights["k4"], 0.0)


class TestComparison(unittest.TestCase):
    def setUp(self):
        self.losses = {
            "publicada": _losses(1.0, 200, seed=1),
            "peor": _losses(1.0, 200, seed=1) + 0.5,
            "mejor": _losses(1.0, 200, seed=1) - 0.3,
        }

    def test_produces_one_row_per_variant_and_rule(self):
        table = compare_against_reference(self.losses, "publicada", samples=300)
        self.assertEqual(len(table), 2 * len(SCORING_RULES))
        self.assertNotIn("publicada", set(table["variant"]))

    def test_flags_a_uniformly_worse_variant(self):
        table = compare_against_reference(self.losses, "publicada", samples=1000)
        worse = table.loc[table["variant"] == "peor"]
        self.assertTrue(worse["worse_than_reference"].all())
        self.assertFalse(worse["better_than_reference"].any())

    def test_flags_a_uniformly_better_variant(self):
        table = compare_against_reference(self.losses, "publicada", samples=1000)
        better = table.loc[table["variant"] == "mejor"]
        self.assertTrue(better["better_than_reference"].all())

    def test_holm_never_reports_below_the_raw_significance(self):
        table = compare_against_reference(self.losses, "publicada", samples=500)
        self.assertTrue((table["pvalue_holm"] <= 1.0).all())
        self.assertTrue((table["pvalue_holm"] >= 0.0).all())

    def test_a_variant_identical_to_the_reference_is_not_flagged(self):
        losses = {
            "publicada": _losses(1.0, 200, seed=2),
            "copia": _losses(1.0, 200, seed=2),
        }
        table = compare_against_reference(losses, "publicada", samples=500)
        self.assertFalse(table["worse_than_reference"].any())
        self.assertFalse(table["better_than_reference"].any())
        self.assertTrue(np.allclose(table["mean_difference"], 0.0))

    def test_rejects_a_reference_that_is_not_present(self):
        with self.assertRaises(ValueError):
            compare_against_reference(self.losses, "inexistente", samples=300)


if __name__ == "__main__":
    unittest.main()
