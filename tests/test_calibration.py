import unittest

import numpy as np

from mm_ipsa.evaluation.calibration import (
    calibration_report,
    classify_dispersion,
    pit_diagnostics,
    randomized_pit,
    reliability_index,
    resample_support,
)


def _ensemble(scale: float, size: int, assets: int, seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    scenarios = rng.standard_normal((size, assets)) * 0.02 * scale + shift
    return scenarios, np.full(size, 1.0 / size)


def _truth(rows: int, assets: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, assets)) * 0.02


class TestRandomizedPIT(unittest.TestCase):
    def test_pit_is_uniform_under_correct_specification(self):
        truth = _truth(600, 1, 1)
        scenarios, probabilities = _ensemble(1.0, 40_000, 1, 2)
        pit = randomized_pit(scenarios, probabilities, truth, seed=3)
        diagnostics = pit_diagnostics(pit, ["a"])
        self.assertGreater(diagnostics["ks_pvalue"].iloc[0], 0.05)
        self.assertAlmostEqual(diagnostics["pit_mean"].iloc[0], 0.5, delta=0.05)

    def test_narrow_forecast_is_flagged_underdispersed(self):
        truth = _truth(500, 1, 4)
        scenarios, probabilities = _ensemble(0.5, 20_000, 1, 5)
        pit = randomized_pit(scenarios, probabilities, truth, seed=6)
        diagnostics = pit_diagnostics(pit, ["a"])
        self.assertGreater(diagnostics["dispersion_ratio"].iloc[0], 1.1)
        self.assertEqual(diagnostics["dispersion"].iloc[0], "subdispersa")

    def test_wide_forecast_is_flagged_overdispersed(self):
        truth = _truth(500, 1, 7)
        scenarios, probabilities = _ensemble(2.0, 20_000, 1, 8)
        pit = randomized_pit(scenarios, probabilities, truth, seed=9)
        diagnostics = pit_diagnostics(pit, ["a"])
        self.assertLess(diagnostics["dispersion_ratio"].iloc[0], 0.9)
        self.assertEqual(diagnostics["dispersion"].iloc[0], "sobredispersa")

    def test_location_bias_shifts_the_pit_mean(self):
        truth = _truth(500, 1, 10)
        scenarios, probabilities = _ensemble(1.0, 20_000, 1, 11, shift=0.02)
        pit = randomized_pit(scenarios, probabilities, truth, seed=12)
        diagnostics = pit_diagnostics(pit, ["a"])
        # Predictiva desplazada hacia arriba: las observaciones caen bajo, por
        # lo que la masa acumulada por debajo de ellas es pequena.
        self.assertLess(diagnostics["pit_mean"].iloc[0], 0.45)

    def test_values_stay_inside_the_unit_interval(self):
        truth = _truth(200, 3, 13)
        scenarios, probabilities = _ensemble(1.0, 1_000, 3, 14)
        pit = randomized_pit(scenarios, probabilities, truth, seed=15)
        self.assertTrue(np.all((pit >= 0.0) & (pit <= 1.0)))
        self.assertEqual(pit.shape, truth.shape)

    def test_randomization_activates_only_on_ties(self):
        scenarios, probabilities = _ensemble(1.0, 200, 1, 17)
        # Observaciones continuas: no hay empates y el resultado es
        # deterministico pese a cambiar la semilla.
        continuous = _truth(300, 1, 16)
        self.assertTrue(
            np.allclose(
                randomized_pit(scenarios, probabilities, continuous, seed=1),
                randomized_pit(scenarios, probabilities, continuous, seed=2),
            )
        )
        # Observaciones tomadas del propio soporte: cada una cae sobre un atomo
        # y la aleatorizacion reparte la masa del salto.
        tied = scenarios[:50].copy()
        first = randomized_pit(scenarios, probabilities, tied, seed=1)
        second = randomized_pit(scenarios, probabilities, tied, seed=2)
        self.assertFalse(np.allclose(first, second))

    def test_ties_remain_uniform_after_randomisation(self):
        scenarios, probabilities = _ensemble(1.0, 400, 1, 41)
        tied = scenarios.copy()
        pit = randomized_pit(scenarios, probabilities, tied, seed=5)
        # Si cada atomo aporta su propio salto repartido uniformemente, el
        # conjunto de PIT debe cubrir (0,1) sin concentrarse.
        diagnostics = pit_diagnostics(pit, ["a"])
        self.assertGreater(diagnostics["ks_pvalue"].iloc[0], 0.01)

    def test_rejects_inconsistent_shapes(self):
        truth = _truth(50, 3, 18)
        scenarios, probabilities = _ensemble(1.0, 100, 2, 19)
        with self.assertRaises(ValueError):
            randomized_pit(scenarios, probabilities, truth)
        with self.assertRaises(ValueError):
            randomized_pit(scenarios, probabilities[:-1], truth[:, :2])

    def test_rejects_invalid_probabilities(self):
        truth = _truth(50, 2, 20)
        scenarios, _ = _ensemble(1.0, 100, 2, 21)
        with self.assertRaises(ValueError):
            randomized_pit(scenarios, np.full(100, 0.5), truth)


class TestCalibrationHelpers(unittest.TestCase):
    def test_reliability_index_is_zero_for_flat_histogram(self):
        values = (np.arange(1_000) + 0.5) / 1_000
        self.assertAlmostEqual(reliability_index(values, bins=10), 0.0, places=9)

    def test_reliability_index_grows_with_concentration(self):
        flat = (np.arange(1_000) + 0.5) / 1_000
        concentrated = np.full(1_000, 0.05)
        self.assertGreater(
            reliability_index(concentrated), reliability_index(flat)
        )

    def test_classify_dispersion_boundaries(self):
        self.assertEqual(classify_dispersion(1.0), "calibrada")
        self.assertEqual(classify_dispersion(1.5), "subdispersa")
        self.assertEqual(classify_dispersion(0.5), "sobredispersa")

    def test_resample_support_reduces_and_normalises(self):
        scenarios, probabilities = _ensemble(1.0, 5_000, 2, 22)
        reduced, weights = resample_support(scenarios, probabilities, 500, seed=1)
        self.assertEqual(reduced.shape, (500, 2))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_resample_support_is_identity_when_already_small(self):
        scenarios, probabilities = _ensemble(1.0, 100, 2, 23)
        reduced, weights = resample_support(scenarios, probabilities, 500, seed=1)
        np.testing.assert_allclose(reduced, scenarios)
        np.testing.assert_allclose(weights, probabilities)

    def test_resample_support_rejects_tiny_sizes(self):
        scenarios, probabilities = _ensemble(1.0, 100, 2, 24)
        with self.assertRaises(ValueError):
            resample_support(scenarios, probabilities, 1, seed=1)


class TestCalibrationReport(unittest.TestCase):
    def test_equalising_support_removes_the_resolution_advantage(self):
        # Ambos modelos son correctos; solo difieren en cuantos escenarios
        # tienen. Sin igualar soporte el pequeno parece peor calibrado.
        truth = _truth(400, 2, 30)
        models = {
            "grande": _ensemble(1.0, 20_000, 2, 31),
            "pequeno": _ensemble(1.0, 300, 2, 32),
        }
        _, native = calibration_report(models, truth, ["a", "b"], seed=1)
        _, equalised = calibration_report(
            models, truth, ["a", "b"], seed=1, support_size=300
        )
        self.assertEqual(sorted(native["native_support"]), [300, 20_000])
        self.assertEqual(list(equalised["effective_support"]), [300, 300])

    def test_detail_and_summary_cover_every_model_and_asset(self):
        truth = _truth(200, 3, 33)
        models = {
            "a": _ensemble(1.0, 2_000, 3, 34),
            "b": _ensemble(1.6, 2_000, 3, 35),
        }
        detail, summary = calibration_report(models, truth, ["x", "y", "z"], seed=2)
        self.assertEqual(len(detail), 6)
        self.assertEqual(len(summary), 2)
        self.assertTrue((detail["ks_pvalue_holm"] >= detail["ks_pvalue"]).all())

    def test_summary_identifies_the_miscalibrated_model(self):
        truth = _truth(500, 2, 36)
        models = {
            "correcto": _ensemble(1.0, 5_000, 2, 37),
            "estrecho": _ensemble(0.4, 5_000, 2, 38),
        }
        _, summary = calibration_report(models, truth, ["a", "b"], seed=3)
        narrow = summary.loc[summary["model"] == "estrecho"].iloc[0]
        correct = summary.loc[summary["model"] == "correcto"].iloc[0]
        self.assertGreater(narrow["mean_dispersion_ratio"], correct["mean_dispersion_ratio"])
        self.assertEqual(int(narrow["assets_underdispersed"]), 2)


if __name__ == "__main__":
    unittest.main()
