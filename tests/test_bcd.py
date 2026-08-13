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


def _small_objective(seed: int = 7, assets: int = 2, scenarios: int = 12):
    """Objetivo MM pequeno pero bien condicionado para pruebas de contorno."""
    rng = np.random.default_rng(seed)
    observations = rng.multivariate_normal(
        mean=np.array([0.002, -0.001])[:assets],
        cov=np.array([[0.0005, 0.00018], [0.00018, 0.0008]])[:assets, :assets],
        size=120,
    )
    mu = observations.mean(axis=0)
    dev = observations - mu
    moments = np.vstack(
        [
            mu,
            np.mean(dev**2, axis=0),
            np.mean(dev**3, axis=0),
            np.mean(dev**4, axis=0),
        ]
    )
    covariance = dev.T @ dev / len(observations)
    objective = MMObjective(
        moments,
        covariance,
        {"k1": 1.0, "k2": 1.0, "k3": 0.5, "k4": 0.25, "cov_weight": 1.0},
        scenarios,
    )
    return objective, scenarios


def _base_config(scenarios: int, **overrides) -> dict:
    config = {
        "N_scenarios": scenarios,
        "bcd_max_iter": 3,
        "tol": 1e-8,
        "min_iter": 1,
        "n_starts": 1,
        "seed": 5,
        "entropy_lambda": 1e-5,
        "update_mode": "sequential",
        "p_solver": "mirror_descent",
        "start_strategy": "independent",
        "convergence_patience": 2,
        "strict_solver": False,
    }
    config.update(overrides)
    return config


class TestBCDConfigurationContracts(unittest.TestCase):
    def test_rejects_unknown_modes(self):
        objective, scenarios = _small_objective()
        for key, value in (
            ("update_mode", "newton"),
            ("start_strategy", "hot"),
            ("p_solver", "lbfgs"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    BCDSolver(objective, _base_config(scenarios, **{key: value}))

    def test_rejects_nonpositive_solver_parameters(self):
        objective, scenarios = _small_objective()
        for key, value in (
            ("p_inner_max_iter", 0),
            ("p_step_scale", 0.0),
            ("x_inner_max_iter", 0),
            ("x_ftol", 0.0),
            ("convergence_patience", 0),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    BCDSolver(objective, _base_config(scenarios, **{key: value}))


class TestBCDDegenerateBehaviour(unittest.TestCase):
    def test_probabilities_stay_on_the_simplex(self):
        objective, scenarios = _small_objective()
        solver = BCDSolver(objective, _base_config(scenarios), n_workers=1)
        _, probabilities = solver.solve()
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertGreaterEqual(float(probabilities.min()), 0.0)

    def test_entropy_regularisation_pushes_towards_uniform(self):
        objective, scenarios = _small_objective()
        weak = BCDSolver(
            objective, _base_config(scenarios, entropy_lambda=0.0), n_workers=1
        )
        strong = BCDSolver(
            objective, _base_config(scenarios, entropy_lambda=1.0), n_workers=1
        )
        _, weak_p = weak.solve()
        _, strong_p = strong.solve()
        uniform = 1.0 / scenarios
        # Con lambda grande la KL domina y la solucion se aplana.
        self.assertLess(
            float(np.abs(strong_p - uniform).max()),
            float(np.abs(weak_p - uniform).max()) + 1e-12,
        )

    def test_kl_divergence_is_zero_only_for_uniform(self):
        objective, scenarios = _small_objective()
        solver = BCDSolver(objective, _base_config(scenarios), n_workers=1)
        uniform = np.full(scenarios, 1.0 / scenarios)
        self.assertAlmostEqual(solver._kl_uniform(uniform), 0.0, places=12)
        skewed = np.zeros(scenarios)
        skewed[0] = 1.0
        self.assertGreater(solver._kl_uniform(skewed), 0.0)

    def test_strict_solver_rejects_nonstationary_solutions(self):
        objective, scenarios = _small_objective()
        # Umbrales imposibles: ningun start califica y el modo estricto debe
        # fallar en lugar de publicar una solucion que no cumple el contrato.
        config = _base_config(
            scenarios,
            strict_solver=True,
            x_stationarity_tol=1e-30,
            p_stationarity_tol=1e-30,
        )
        solver = BCDSolver(objective, config, n_workers=1)
        with self.assertRaises(RuntimeError):
            solver.solve()

    def test_non_strict_solver_falls_back_to_lowest_objective(self):
        objective, scenarios = _small_objective()
        config = _base_config(
            scenarios,
            strict_solver=False,
            n_starts=2,
            x_stationarity_tol=1e-30,
            p_stationarity_tol=1e-30,
        )
        solver = BCDSolver(objective, config, n_workers=1)
        x, probabilities = solver.solve()
        self.assertEqual(x.shape, (scenarios, objective.n))
        best = min(record["G_fin"] for record in solver.all_starts)
        self.assertAlmostEqual(solver.best_G, best, places=12)

    def test_summary_table_is_empty_before_solving(self):
        objective, scenarios = _small_objective()
        solver = BCDSolver(objective, _base_config(scenarios), n_workers=1)
        self.assertTrue(solver.summary_table().empty)

    def test_summary_table_marks_exactly_one_best_start(self):
        objective, scenarios = _small_objective()
        solver = BCDSolver(objective, _base_config(scenarios, n_starts=3), n_workers=1)
        solver.solve()
        table = solver.summary_table()
        self.assertEqual(len(table), 3)
        self.assertEqual(int(table["is_best"].sum()), 1)

    def test_stationarity_metrics_are_finite_and_nonnegative(self):
        objective, scenarios = _small_objective()
        solver = BCDSolver(objective, _base_config(scenarios), n_workers=1)
        x, probabilities = solver.solve()
        metrics = solver.stationarity_metrics(x, probabilities)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
