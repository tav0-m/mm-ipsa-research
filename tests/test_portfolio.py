import unittest

import numpy as np

from mm_ipsa.portfolio.optimization import (
    _feasible_weights,
    equal_weight,
    hierarchical_risk_parity,
    inverse_variance,
    maximum_sharpe,
    minimum_cvar,
    minimum_variance,
    portfolio_diagnostics,
)


class TestFeasibleWeights(unittest.TestCase):
    def test_respects_the_cap_and_the_budget(self):
        weights = _feasible_weights(np.array([0.9, 0.05, 0.05]), 0.4)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertLessEqual(float(weights.max()), 0.4 + 1e-9)

    def test_rejects_infeasible_cap(self):
        # Cuatro activos con tope 0.2 no alcanzan a sumar uno.
        with self.assertRaises(ValueError):
            _feasible_weights(np.full(4, 0.25), 0.2)

    def test_rejects_invalid_cap_values(self):
        for cap in (0.0, -0.1, 1.5):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    _feasible_weights(np.full(3, 1 / 3), cap)

    def test_uniform_input_is_unchanged_when_cap_is_slack(self):
        uniform = np.full(5, 0.2)
        np.testing.assert_allclose(_feasible_weights(uniform, 0.5), uniform)

    def test_optimisers_never_breach_the_cap(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal((400, 6)) * 0.01
        covariance = np.cov(data, rowvar=False)
        cap = 0.25
        for weights in (
            minimum_variance(covariance, cap),
            maximum_sharpe(data.mean(axis=0), covariance, max_weight=cap, seed=1),
        ):
            self.assertLessEqual(float(weights.max()), cap + 1e-9)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)


class TestPortfolioDiagnosticsRiskFreeRate(unittest.TestCase):
    def _inputs(self):
        rng = np.random.default_rng(9)
        scenarios = rng.standard_normal((2_000, 3)) * 0.02 + 0.01
        probabilities = np.full(2_000, 1 / 2_000)
        return scenarios, probabilities

    def test_sharpe_matches_the_optimised_definition(self):
        scenarios, probabilities = self._inputs()
        weights = equal_weight(3)
        rate = 0.004
        result = portfolio_diagnostics(
            weights, scenarios, probabilities, risk_free_rate=rate
        )
        expected = (result["expected_return"] - rate) / result["volatility"]
        self.assertAlmostEqual(result["sharpe"], expected, places=10)
        self.assertEqual(result["risk_free_rate"], rate)

    def test_nonzero_rate_lowers_the_reported_sharpe(self):
        scenarios, probabilities = self._inputs()
        weights = equal_weight(3)
        without = portfolio_diagnostics(weights, scenarios, probabilities)
        with_rate = portfolio_diagnostics(
            weights, scenarios, probabilities, risk_free_rate=0.005
        )
        self.assertLess(with_rate["sharpe"], without["sharpe"])


class TestPortfolio(unittest.TestCase):
    def test_naive_and_robust_weights_are_valid(self):
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
        for weights in (equal_weight(3), inverse_variance(covariance), hierarchical_risk_parity(covariance), minimum_variance(covariance, 0.6, 0.01)):
            self.assertAlmostEqual(weights.sum(), 1.0, places=10)
            self.assertTrue(np.all(weights >= 0))

    def test_minimum_cvar_respects_constraints(self):
        rng = np.random.default_rng(3)
        scenarios = rng.normal(size=(100, 4)) * np.array([0.01, 0.02, 0.03, 0.04])
        weights = minimum_cvar(scenarios, np.full(100, 0.01), max_weight=0.4)
        self.assertAlmostEqual(weights.sum(), 1.0, places=9)
        self.assertLessEqual(weights.max(), 0.4 + 1e-10)


if __name__ == "__main__":
    unittest.main()
