import unittest

import numpy as np

from src.mm.objective import MMObjective


class TestMMObjective(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260809)
        self.N, self.n = 11, 3
        self.x = rng.normal(0.002, 0.025, size=(self.N, self.n))
        self.p = rng.dirichlet(np.ones(self.N) * 2.0)

        mu = self.p @ self.x
        dev = self.x - mu
        M = np.vstack(
            [
                mu + np.array([0.001, -0.002, 0.0005]),
                self.p @ dev**2 * np.array([1.05, 0.95, 1.10]),
                self.p @ dev**3 + np.array([2e-6, -1e-6, 3e-6]),
                self.p @ dev**4 * np.array([0.90, 1.15, 1.05]),
            ]
        )
        Sigma = (self.p[:, None] * dev).T @ dev
        Sigma[0, 1] *= 0.85
        Sigma[1, 0] = Sigma[0, 1]
        Sigma[0, 2] *= 1.20
        Sigma[2, 0] = Sigma[0, 2]

        weights = {
            "k1": 1.0,
            "k2": 1.0,
            "k3": 1.0,
            "k4": 1.0,
            "cov_weight": 1.0,
            "cov_offdiag_only": True,
            "moment_scale_mode": "volatility_power",
        }
        self.obj = MMObjective(M, Sigma, weights, self.N)

    def test_directional_gradient_x_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        direction = rng.normal(size=self.x.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        numeric = (
            self.obj.evaluate(self.x + eps * direction, self.p)
            - self.obj.evaluate(self.x - eps * direction, self.p)
        ) / (2 * eps)
        analytic = float(np.sum(self.obj.grad_x(self.x, self.p) * direction))
        self.assertAlmostEqual(analytic, numeric, delta=1e-6 * max(1.0, abs(numeric)))

    def test_directional_gradient_p_on_simplex_matches_finite_difference(self):
        rng = np.random.default_rng(11)
        direction = rng.normal(size=self.N)
        direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        eps = min(1e-6, 0.2 * self.p.min() / np.max(np.abs(direction)))
        numeric = (
            self.obj.evaluate(self.x, self.p + eps * direction)
            - self.obj.evaluate(self.x, self.p - eps * direction)
        ) / (2 * eps)
        analytic = float(self.obj.grad_p(self.x, self.p) @ direction)
        self.assertAlmostEqual(analytic, numeric, delta=1e-6 * max(1.0, abs(numeric)))

    def test_components_sum_to_objective(self):
        components = self.obj.components(self.x, self.p)
        self.assertAlmostEqual(components["total"], self.obj.evaluate(self.x, self.p), places=13)

    def test_covariance_diagonal_is_not_double_counted(self):
        M = self.obj.M.copy()
        Sigma = self.obj.Sigma_tgt.copy()
        Sigma[np.diag_indices_from(Sigma)] *= 4.0
        weights = {
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "cov_weight": 1.0,
            "cov_offdiag_only": True,
        }
        obj = MMObjective(M, Sigma, weights, self.N)
        self.assertAlmostEqual(obj.evaluate(self.x, self.p), obj.components(self.x, self.p)["dependence"])

    def test_near_zero_target_moments_have_finite_weights(self):
        M = self.obj.M.copy()
        M[0] = 0.0
        M[2] = 0.0
        obj = MMObjective(M, self.obj.Sigma_tgt, {
            "k1": 1.0, "k2": 1.0, "k3": 1.0, "k4": 1.0,
            "cov_weight": 1.0,
        }, self.N)
        self.assertTrue(np.all(np.isfinite(obj.W)))
        self.assertTrue(np.isfinite(obj.evaluate(self.x, self.p)))

    def test_rejects_material_probability_sum_error(self):
        invalid = self.p * 0.99
        with self.assertRaises(ValueError):
            self.obj.evaluate(self.x, invalid)


if __name__ == "__main__":
    unittest.main()
