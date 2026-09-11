import unittest

import numpy as np

from mm_ipsa.evaluation.attribution import (
    compare_attribution,
    component_expected_shortfall,
    realised_component_shortfall,
    tail_weights,
)
from mm_ipsa.evaluation.scoring import lower_tail_mean

ALPHA = 0.05


def _scenarios(rows: int = 2_000, assets: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((rows, assets)) * 0.02
    probabilities = rng.random(rows)
    return draws, probabilities / probabilities.sum()


def _weights(assets: int = 6, seed: int = 1):
    raw = np.random.default_rng(seed).random(assets)
    return raw / raw.sum()


class TestTailWeights(unittest.TestCase):
    def test_allocated_mass_equals_alpha(self):
        _, probabilities = _scenarios()
        losses = np.random.default_rng(2).standard_normal(len(probabilities))
        for alpha in (0.01, 0.05, 0.137, 0.5):
            allocated = tail_weights(losses, probabilities, alpha)
            self.assertAlmostEqual(float(allocated.sum()), alpha, places=12)

    def test_only_the_worst_scenarios_receive_mass(self):
        losses = np.array([0.5, -0.3, 0.1, -0.9, 0.2])
        probabilities = np.full(5, 0.2)
        allocated = tail_weights(losses, probabilities, 0.2)
        self.assertGreater(allocated[3], 0.0)
        self.assertEqual(float(allocated[[0, 2, 4]].sum()), 0.0)

    def test_rejects_invalid_alpha(self):
        losses, probabilities = np.zeros(4), np.full(4, 0.25)
        for alpha in (0.0, 1.5, -0.2):
            with self.assertRaises(ValueError):
                tail_weights(losses, probabilities, alpha)


class TestEulerAdditivity(unittest.TestCase):
    """La descomposicion solo es valida si cierra exactamente."""

    def test_components_sum_to_the_total(self):
        for seed, assets, alpha in ((0, 6, 0.05), (3, 15, 0.01), (7, 3, 0.25)):
            scenarios, probabilities = _scenarios(assets=assets, seed=seed)
            weights = _weights(assets, seed=seed + 50)
            report = component_expected_shortfall(
                scenarios, probabilities, weights, alpha
            )
            self.assertAlmostEqual(
                float(report["components"].sum()), report["total"], places=15
            )

    def test_total_matches_the_projects_expected_shortfall(self):
        scenarios, probabilities = _scenarios(seed=4)
        weights = _weights(seed=5)
        report = component_expected_shortfall(
            scenarios, probabilities, weights, ALPHA
        )
        reference = lower_tail_mean(scenarios @ weights, probabilities, ALPHA)
        self.assertAlmostEqual(report["total"], reference, places=15)

    def test_shares_sum_to_one(self):
        scenarios, probabilities = _scenarios(seed=6)
        report = component_expected_shortfall(
            scenarios, probabilities, _weights(seed=7), ALPHA
        )
        self.assertAlmostEqual(float(report["shares"].sum()), 1.0, places=12)

    def test_scaling_every_weight_scales_the_total_proportionally(self):
        # El ES es homogeneo de grado uno, que es lo que habilita Euler.
        scenarios, probabilities = _scenarios(seed=8)
        weights = _weights(seed=9)
        base = component_expected_shortfall(
            scenarios, probabilities, weights, ALPHA
        )
        doubled = component_expected_shortfall(
            scenarios, probabilities, 2.0 * weights, ALPHA
        )
        self.assertAlmostEqual(doubled["total"], 2.0 * base["total"], places=14)


class TestHedgingPosition(unittest.TestCase):
    def test_an_offsetting_asset_contributes_negatively(self):
        rng = np.random.default_rng(11)
        driver = rng.standard_normal(4_000) * 0.03
        hedge = -driver
        scenarios = np.column_stack([driver, hedge])
        probabilities = np.full(4_000, 1.0 / 4_000)

        report = component_expected_shortfall(
            scenarios, probabilities, np.array([0.8, 0.2]), ALPHA
        )

        self.assertLess(report["components"][0], 0.0)
        self.assertGreater(report["components"][1], 0.0)


class TestConfoundedShareCorrelation(unittest.TestCase):
    """Los pesos son identicos en ambas descomposiciones y no deben puntuar."""

    def test_concentration_inflates_the_share_correlation_only(self):
        scenarios, probabilities = _scenarios(rows=3_000, assets=10, seed=12)
        observations = (
            np.random.default_rng(13).standard_normal((400, 10)) * 0.02
        )

        concentrated = np.zeros(10)
        concentrated[0] = 0.7
        concentrated[1:] = 0.3 / 9
        uniform = np.full(10, 0.1)

        results = {}
        for name, weights in (("concentrada", concentrated), ("uniforme", uniform)):
            results[name] = compare_attribution(
                component_expected_shortfall(
                    scenarios, probabilities, weights, 0.20
                ),
                realised_component_shortfall(observations, weights, 0.20),
            )

        # La cartera concentrada correlaciona alto en shares por construccion.
        self.assertGreater(
            results["concentrada"]["share_rank_correlation"],
            results["uniforme"]["share_rank_correlation"],
        )
        # La medida de acierto del modelo no depende de esa concentracion, y
        # ambas carteras ven la misma predictiva equivocada.
        for name in results:
            self.assertLess(abs(results[name]["conditional_rank_correlation"]), 0.8)

    def test_a_model_that_knows_the_truth_scores_high_on_conditionals(self):
        rng = np.random.default_rng(14)
        scale = np.linspace(0.01, 0.05, 8)
        scenarios = rng.standard_normal((4_000, 8)) * scale
        observations = rng.standard_normal((600, 8)) * scale
        probabilities = np.full(4_000, 1.0 / 4_000)
        weights = np.full(8, 0.125)

        comparison = compare_attribution(
            component_expected_shortfall(scenarios, probabilities, weights, 0.20),
            realised_component_shortfall(observations, weights, 0.20),
        )

        self.assertGreater(comparison["conditional_rank_correlation"], 0.7)


class TestComparison(unittest.TestCase):
    def test_identical_decompositions_have_no_composition_error(self):
        scenarios, probabilities = _scenarios(seed=15)
        weights = _weights(seed=16)
        report = component_expected_shortfall(
            scenarios, probabilities, weights, ALPHA
        )
        comparison = compare_attribution(report, report)
        self.assertAlmostEqual(comparison["composition_error"], 0.0, places=12)
        self.assertTrue(comparison["identifies_main_driver"])

    def test_flags_an_unreliable_realised_tail(self):
        scenarios, probabilities = _scenarios(seed=17)
        weights = _weights(seed=18)
        observations = np.random.default_rng(19).standard_normal((40, 6)) * 0.02

        comparison = compare_attribution(
            component_expected_shortfall(scenarios, probabilities, weights, ALPHA),
            realised_component_shortfall(observations, weights, ALPHA),
        )

        self.assertLess(comparison["realised_tail_observations"], 10)
        self.assertFalse(comparison["reliable"])

    def test_rejects_decompositions_of_different_width(self):
        wide, probabilities = _scenarios(assets=6, seed=20)
        narrow, narrow_probabilities = _scenarios(assets=4, seed=21)
        with self.assertRaises(ValueError):
            compare_attribution(
                component_expected_shortfall(
                    wide, probabilities, _weights(6, seed=22), ALPHA
                ),
                component_expected_shortfall(
                    narrow, narrow_probabilities, _weights(4, seed=23), ALPHA
                ),
            )


class TestValidation(unittest.TestCase):
    def test_rejects_weights_that_do_not_match_the_scenarios(self):
        scenarios, probabilities = _scenarios(assets=6, seed=24)
        with self.assertRaises(ValueError):
            component_expected_shortfall(
                scenarios, probabilities, np.full(5, 0.2), ALPHA
            )

    def test_rejects_one_dimensional_observations(self):
        with self.assertRaises(ValueError):
            realised_component_shortfall(np.zeros(50), np.array([1.0]), ALPHA)

    def test_rejects_negative_probabilities(self):
        scenarios, probabilities = _scenarios(seed=25)
        probabilities = probabilities.copy()
        probabilities[0] = -0.1
        with self.assertRaises(ValueError):
            component_expected_shortfall(
                scenarios, probabilities, _weights(seed=26), ALPHA
            )


if __name__ == "__main__":
    unittest.main()
