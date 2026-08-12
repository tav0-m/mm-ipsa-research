import unittest

import numpy as np

from mm_ipsa.mm.bcd import BCDSolver
from mm_ipsa.mm.objective import MMObjective


class TestBCDSolver(unittest.TestCase):
    def test_joint_x_solver_respects_small_objective_scale(self):
        class TinyQuadratic:
            N = 4
            n = 2
            M = np.vstack([np.zeros(2), np.ones(2), np.zeros(2), np.ones(2)])

            @staticmethod
            def evaluate(x, p):
                del p
                return 0.5e-5 * float(np.sum(x**2))

            @staticmethod
            def grad_x(x, p):
                del p
                return 1e-5 * x

            @staticmethod
            def grad_p(x, p):
                del x
                return np.zeros_like(p)

        cfg = {
            "N_scenarios": 4,
            "bcd_max_iter": 2,
            "tol": 1e-8,
            "n_starts": 1,
            "seed": 1,
            "entropy_lambda": 0.0,
            "update_mode": "joint_lbfgs",
            "x_inner_max_iter": 100,
            "x_ftol": 1e-12,
            "x_gtol": 1e-10,
            "p_solver": "mirror_descent",
            "start_strategy": "independent",
        }
        solver = BCDSolver(TinyQuadratic(), cfg, n_workers=1)
        x0 = np.full((4, 2), 0.02)
        p = np.full(4, 0.25)
        x1 = solver._step_x_joint(x0, p)

        self.assertLess(TinyQuadratic.evaluate(x1, p), 1e-4 * TinyQuadratic.evaluate(x0, p))

    def test_regularized_objective_is_monotone(self):
        rng = np.random.default_rng(91)
        observations = rng.multivariate_normal(
            mean=np.array([0.002, -0.001]),
            cov=np.array([[0.0005, 0.00018], [0.00018, 0.0008]]),
            size=80,
        )
        mu = observations.mean(axis=0)
        dev = observations - mu
        M = np.vstack([mu, np.mean(dev**2, axis=0), np.mean(dev**3, axis=0), np.mean(dev**4, axis=0)])
        Sigma = dev.T @ dev / len(observations)
        N = 16
        obj = MMObjective(
            M,
            Sigma,
            {"k1": 1.0, "k2": 1.0, "k3": 0.5, "k4": 0.25, "cov_weight": 1.0},
            N,
        )
        cfg = {
            "N_scenarios": N,
            "bcd_max_iter": 4,
            "tol": 1e-8,
            "min_iter": 1,
            "n_starts": 1,
            "seed": 123,
            "entropy_lambda": 1e-5,
            "update_mode": "sequential",
            "p_solver": "mirror_descent",
            "start_strategy": "independent",
            "convergence_patience": 2,
            "strict_solver": False,
        }
        solver = BCDSolver(obj, cfg, n_workers=1)
        x, p = solver.solve()
        history = np.asarray(solver.all_starts[0]["history_G"])

        self.assertTrue(np.all(np.diff(history) <= 1e-10))
        self.assertAlmostEqual(p.sum(), 1.0, places=10)
        self.assertTrue(np.all(p >= 0.0))
        self.assertEqual(x.shape, (N, 2))
        self.assertFalse(solver.all_starts[0]["warm"])


if __name__ == "__main__":
    unittest.main()
