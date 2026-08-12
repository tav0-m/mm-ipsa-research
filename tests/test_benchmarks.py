import unittest

import numpy as np

from src.models.benchmarks import gaussian_terminal, student_t_terminal


class TestBenchmarks(unittest.TestCase):
    def test_gaussian_matches_terminal_parameters(self):
        mean = np.array([0.01, -0.005])
        covariance = np.array([[0.002, 0.0007], [0.0007, 0.001]])
        scenarios, probabilities = gaussian_terminal(mean, covariance, 100_000, 4)
        np.testing.assert_allclose(probabilities.sum(), 1.0)
        np.testing.assert_allclose(scenarios.mean(axis=0), mean, atol=3e-4)
        np.testing.assert_allclose(np.cov(scenarios, rowvar=False, ddof=0), covariance, atol=3e-5)

    def test_student_t_has_heavier_marginal_tails(self):
        mean = np.zeros(2)
        covariance = np.eye(2)
        normal, _ = gaussian_terminal(mean, covariance, 80_000, 1)
        heavy, _ = student_t_terminal(mean, covariance, 80_000, 6.0, 2)
        normal_kurt = np.mean(np.mean((normal - normal.mean(0)) ** 4, 0) / np.var(normal, 0) ** 2)
        heavy_kurt = np.mean(np.mean((heavy - heavy.mean(0)) ** 4, 0) / np.var(heavy, 0) ** 2)
        self.assertGreater(heavy_kurt, normal_kurt + 1.0)


if __name__ == "__main__":
    unittest.main()
