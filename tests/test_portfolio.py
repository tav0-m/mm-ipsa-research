import unittest

import numpy as np

from mm_ipsa.portfolio.optimization import (
    equal_weight,
    hierarchical_risk_parity,
    inverse_variance,
    minimum_cvar,
    minimum_variance,
)


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
