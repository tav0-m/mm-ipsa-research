import unittest

import numpy as np

from mm_ipsa.evaluation.expected_shortfall import (
    _monte_carlo_pvalue,
    _verdict,
    expected_shortfall_backtest,
)

ALPHA = 0.05


def _predictive(size: int = 4_000, scale: float = 0.02, seed: int = 0):
    support = np.random.default_rng(seed).standard_normal(size) * scale
    return support, np.full(size, 1.0 / size)


def _rejection_rate(sampler, repetitions: int, horizon: int, support, weights) -> float:
    rejected = 0
    for index in range(repetitions):
        observations = sampler(horizon, np.random.default_rng(50_000 + index))
        report = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=500, seed=index
        )
        rejected += report["verdict"] == "riesgo subestimado"
    return rejected / repetitions


class TestMonteCarloPValue(unittest.TestCase):
    def test_never_returns_zero(self):
        null = np.zeros(100)
        self.assertGreater(_monte_carlo_pvalue(null, -10.0), 0.0)

    def test_counts_ties_against_the_observed_value(self):
        null = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(_monte_carlo_pvalue(null, 1.0), 3 / 5)


class TestSize(unittest.TestCase):
    def test_correctly_specified_predictive_is_not_rejected(self):
        # La hipotesis nula es que las observaciones vienen de la predictiva. Si
        # se generan desde la ley continua que origino el soporte, el contraste
        # detecta -legitimamente- la diferencia entre esa ley y su aproximacion
        # discreta, y el tamano medido dejaria de ser el del contraste.
        support, weights = _predictive()
        rate = _rejection_rate(
            lambda horizon, rng: rng.choice(support, horizon),
            repetitions=200,
            horizon=250,
            support=support,
            weights=weights,
        )
        self.assertLess(rate, 0.12)


class TestPower(unittest.TestCase):
    def test_detects_a_predictive_that_is_too_narrow(self):
        support, weights = _predictive()
        rate = _rejection_rate(
            lambda horizon, rng: rng.standard_normal(horizon) * 0.024,
            repetitions=120,
            horizon=250,
            support=support,
            weights=weights,
        )
        self.assertGreater(rate, 0.50)

    def test_detects_a_heavier_tail_than_predicted(self):
        support, weights = _predictive()
        rate = _rejection_rate(
            lambda horizon, rng: rng.standard_t(4, horizon) * 0.02 / np.sqrt(2.0),
            repetitions=120,
            horizon=250,
            support=support,
            weights=weights,
        )
        self.assertGreater(rate, 0.35)

    def test_the_two_statistics_cover_different_failure_modes(self):
        # Ninguno domina al otro y por eso el veredicto usa ambos. Una cola t(4)
        # produce excedencias menos frecuentes pero mas profundas, de modo que
        # ambos efectos se compensan en el estadistico incondicional.
        support, weights = _predictive()
        rng = np.random.default_rng(7)

        narrow = [
            expected_shortfall_backtest(
                rng.standard_normal(250) * 0.024, support, weights, ALPHA,
                n_simulations=800, seed=index,
            )
            for index in range(40)
        ]
        heavy = [
            expected_shortfall_backtest(
                rng.standard_t(4, 250) * 0.02 / np.sqrt(2.0), support, weights,
                ALPHA, n_simulations=800, seed=index,
            )
            for index in range(40)
        ]

        narrow_z1 = np.mean([r["pvalue_z1"] < 0.025 for r in narrow])
        narrow_z2 = np.mean([r["pvalue_z2"] < 0.025 for r in narrow])
        heavy_z1 = np.mean([r["pvalue_z1"] < 0.025 for r in heavy])
        heavy_z2 = np.mean([r["pvalue_z2"] < 0.025 for r in heavy])

        self.assertGreater(narrow_z2, narrow_z1)
        self.assertGreater(heavy_z1, heavy_z2)

    def test_an_overly_wide_predictive_is_not_called_underestimated(self):
        support, weights = _predictive()
        rate = _rejection_rate(
            lambda horizon, rng: rng.standard_normal(horizon) * 0.016,
            repetitions=120,
            horizon=250,
            support=support,
            weights=weights,
        )
        self.assertLess(rate, 0.05)


class TestSignConvention(unittest.TestCase):
    def test_realised_losses_worse_than_predicted_give_negative_statistics(self):
        support, weights = _predictive()
        observations = np.random.default_rng(11).standard_normal(400) * 0.030

        report = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=1_000, seed=1
        )

        self.assertLess(report["z1"], 0.0)
        self.assertLess(report["z2"], 0.0)
        self.assertLess(report["realised_tail_mean"], report["predicted_es"])
        self.assertEqual(report["verdict"], "riesgo subestimado")

    def test_a_conservative_predictive_gives_positive_statistics(self):
        support, weights = _predictive()
        observations = np.random.default_rng(12).standard_normal(400) * 0.012

        report = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=1_000, seed=2
        )

        self.assertGreater(report["z2"], 0.0)
        self.assertNotEqual(report["verdict"], "riesgo subestimado")


class TestReportContents(unittest.TestCase):
    def test_reports_exceedance_rate_near_alpha_under_the_null(self):
        support, weights = _predictive()
        observations = np.random.default_rng(13).choice(support, 2_000)

        report = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=500, seed=3
        )

        self.assertAlmostEqual(report["exceedance_rate"], ALPHA, delta=0.02)
        self.assertEqual(report["observations"], 2_000)
        self.assertLess(report["predicted_es"], report["var"])

    def test_is_deterministic_for_a_fixed_seed(self):
        support, weights = _predictive()
        observations = np.random.default_rng(14).standard_normal(200) * 0.02
        first = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=500, seed=99
        )
        second = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=500, seed=99
        )
        self.assertEqual(first, second)

    def test_handles_a_sample_without_exceedances(self):
        support, weights = _predictive()
        observations = np.full(50, 0.05)

        report = expected_shortfall_backtest(
            observations, support, weights, ALPHA, n_simulations=500, seed=4
        )

        self.assertEqual(report["exceedances"], 0)
        self.assertTrue(np.isnan(report["realised_tail_mean"]))
        self.assertTrue(np.isnan(report["pvalue_z1"]))
        self.assertTrue(np.isfinite(report["pvalue_z2"]))
        self.assertEqual(report["verdict"], "riesgo sobreestimado")


class TestVerdict(unittest.TestCase):
    def test_uses_whichever_statistic_rejects(self):
        self.assertEqual(_verdict(float("nan"), 0.001), "riesgo subestimado")
        self.assertEqual(_verdict(0.001, 0.400), "riesgo subestimado")
        self.assertEqual(_verdict(0.400, 0.400), "compatible")

    def test_is_indeterminate_without_any_finite_pvalue(self):
        self.assertEqual(_verdict(float("nan"), float("nan")), "indeterminado")


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.support, self.weights = _predictive(size=500)
        self.observations = np.random.default_rng(15).standard_normal(50) * 0.02

    def test_rejects_alpha_outside_the_unit_interval(self):
        for alpha in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                expected_shortfall_backtest(
                    self.observations, self.support, self.weights, alpha
                )

    def test_rejects_mismatched_probability_length(self):
        with self.assertRaises(ValueError):
            expected_shortfall_backtest(
                self.observations, self.support, self.weights[:-1], ALPHA
            )

    def test_rejects_negative_probabilities(self):
        weights = self.weights.copy()
        weights[0] = -0.5
        with self.assertRaises(ValueError):
            expected_shortfall_backtest(
                self.observations, self.support, weights, ALPHA
            )

    def test_rejects_an_empty_sample(self):
        with self.assertRaises(ValueError):
            expected_shortfall_backtest(
                np.array([]), self.support, self.weights, ALPHA
            )


if __name__ == "__main__":
    unittest.main()
