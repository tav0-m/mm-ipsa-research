import unittest

import numpy as np

from mm_ipsa.models.dcc_garch import (
    dcc_garch_terminal,
    dcc_quasi_log_likelihood,
    fit_dcc,
    fit_dcc_garch,
    fit_garch11,
    garch11_variances,
    simulate_dcc_terminal,
)


def _simulate_garch(alpha: float, beta: float, size: int, seed: int, sigma2=1e-4):
    """Serie GARCH(1,1) con parámetros conocidos para validar el estimador."""
    rng = np.random.default_rng(seed)
    omega = sigma2 * (1.0 - alpha - beta)
    variance = np.empty(size)
    residuals = np.empty(size)
    variance[0] = sigma2
    for step in range(size):
        residuals[step] = np.sqrt(variance[step]) * rng.standard_normal()
        if step + 1 < size:
            variance[step + 1] = (
                omega + alpha * residuals[step] ** 2 + beta * variance[step]
            )
    return residuals


def _simulate_dcc(a: float, b: float, size: int, assets: int, seed: int):
    """Residuos estandarizados con correlación dinámica conocida."""
    rng = np.random.default_rng(seed)
    base = np.full((assets, assets), 0.4)
    np.fill_diagonal(base, 1.0)
    q = base.copy()
    output = np.empty((size, assets))
    for step in range(size):
        scale = np.sqrt(np.diag(q))
        correlation = q / np.outer(scale, scale)
        cholesky = np.linalg.cholesky(correlation + 1e-10 * np.eye(assets))
        output[step] = cholesky @ rng.standard_normal(assets)
        q = (
            (1.0 - a - b) * base
            + a * np.outer(output[step], output[step])
            + b * q
        )
    return output


class TestGarchVariances(unittest.TestCase):
    def test_recursion_matches_the_definition(self):
        residuals = np.array([0.01, -0.02, 0.005])
        alpha, beta, unconditional = 0.1, 0.85, 1e-4
        variances = garch11_variances(residuals, alpha, beta, unconditional)
        omega = unconditional * (1.0 - alpha - beta)
        self.assertAlmostEqual(variances[0], unconditional)
        self.assertAlmostEqual(
            variances[1], omega + alpha * residuals[0] ** 2 + beta * unconditional
        )
        # Se devuelve una varianza extra: la de un paso adelante.
        self.assertEqual(len(variances), len(residuals) + 1)

    def test_variance_targeting_reproduces_unconditional_level(self):
        # Con residuos consistentes con la varianza objetivo, el sendero debe
        # mantenerse en torno a ella en vez de derivar.
        residuals = _simulate_garch(0.05, 0.90, 3_000, 5)
        variances = garch11_variances(residuals, 0.05, 0.90, float(np.var(residuals)))
        self.assertAlmostEqual(
            float(np.mean(variances)), float(np.var(residuals)), delta=3e-5
        )

    def test_rejects_nonstationary_parameters(self):
        residuals = np.array([0.01, -0.01, 0.02])
        with self.assertRaises(ValueError):
            garch11_variances(residuals, 0.5, 0.6, 1e-4)
        with self.assertRaises(ValueError):
            garch11_variances(residuals, -0.1, 0.5, 1e-4)
        with self.assertRaises(ValueError):
            garch11_variances(residuals, 0.1, 0.5, 0.0)


class TestGarchEstimation(unittest.TestCase):
    def test_recovers_known_parameters(self):
        for alpha, beta in ((0.08, 0.88), (0.15, 0.80)):
            with self.subTest(alpha=alpha, beta=beta):
                residuals = _simulate_garch(alpha, beta, 5_000, 11)
                fitted = fit_garch11(residuals)
                self.assertAlmostEqual(fitted["alpha"], alpha, delta=0.04)
                self.assertAlmostEqual(fitted["beta"], beta, delta=0.07)

    def test_estimated_process_is_always_stationary(self):
        # La parametrización por persistencia y reparto debe hacer imposible
        # devolver un proceso explosivo, sea cual sea la muestra.
        for seed in range(6):
            residuals = _simulate_garch(0.10, 0.85, 400, seed)
            fitted = fit_garch11(residuals)
            self.assertLess(fitted["persistence"], 1.0)
            self.assertGreaterEqual(fitted["alpha"], 0.0)
            self.assertGreaterEqual(fitted["beta"], 0.0)
            self.assertGreater(fitted["omega"], 0.0)

    def test_optimiser_moves_away_from_its_starting_points(self):
        # Si el gradiente se rompe, L-BFGS-B devuelve el punto de arranque.
        # Ninguna combinación de arranque debe reaparecer como solución.
        residuals = _simulate_garch(0.18, 0.70, 4_000, 3)
        fitted = fit_garch11(residuals)
        starts = ((0.95, 0.05), (0.90, 0.12), (0.97, 0.03), (0.80, 0.25))
        for total, split in starts:
            alpha = 0.999 * total * split
            beta = 0.999 * total * (1.0 - split)
            self.assertFalse(
                abs(fitted["alpha"] - alpha) < 1e-9 and abs(fitted["beta"] - beta) < 1e-9
            )

    def test_rejects_short_or_degenerate_series(self):
        with self.assertRaises(ValueError):
            fit_garch11(np.zeros(10))
        with self.assertRaises(ValueError):
            fit_garch11(np.zeros(200))
        with self.assertRaises(ValueError):
            fit_garch11(np.full(200, np.nan))


class TestDccEstimation(unittest.TestCase):
    def test_recovers_known_correlation_dynamics(self):
        for a, b in ((0.03, 0.95), (0.08, 0.85)):
            with self.subTest(a=a, b=b):
                standardized = _simulate_dcc(a, b, 1_200, 4, 7)
                fitted = fit_dcc(standardized)
                self.assertAlmostEqual(fitted["a"], a, delta=0.03)
                self.assertAlmostEqual(fitted["b"], b, delta=0.07)

    def test_estimated_dynamics_are_always_stationary(self):
        standardized = _simulate_dcc(0.05, 0.90, 400, 3, 2)
        fitted = fit_dcc(standardized)
        self.assertLess(fitted["persistence"], 1.0)
        self.assertGreater(fitted["a"], 0.0)

    def test_quasi_likelihood_rejects_explosive_parameters(self):
        standardized = _simulate_dcc(0.05, 0.90, 200, 3, 1)
        self.assertEqual(dcc_quasi_log_likelihood(standardized, 0.6, 0.6), -np.inf)
        self.assertEqual(dcc_quasi_log_likelihood(standardized, -0.1, 0.5), -np.inf)

    def test_quasi_likelihood_prefers_the_true_parameters(self):
        standardized = _simulate_dcc(0.05, 0.90, 900, 4, 9)
        best = dcc_quasi_log_likelihood(standardized, 0.05, 0.90)
        for a, b in ((0.30, 0.50), (0.001, 0.30), (0.20, 0.70)):
            self.assertGreater(best, dcc_quasi_log_likelihood(standardized, a, b))

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            fit_dcc(np.zeros((10, 3)))
        with self.assertRaises(ValueError):
            fit_dcc(np.zeros((100, 1)))


def _panel(rows: int = 600, assets: int = 3, seed: int = 4) -> np.ndarray:
    """Panel diario con un factor común, para que exista dependencia real."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(rows) * 0.008
    return np.column_stack(
        [common + rng.standard_normal(rows) * 0.010 for _ in range(assets)]
    )


class TestDccGarchSimulation(unittest.TestCase):
    """El ajuste es costoso y no depende del caso, por lo que se hace una vez."""

    @classmethod
    def setUpClass(cls):
        cls.panel = _panel()
        cls.state = fit_dcc_garch(cls.panel)

    def test_full_fit_exposes_the_projection_state(self):
        state = self.state
        self.assertEqual(len(state["marginals"]), 3)
        self.assertEqual(state["next_variance"].shape, (3,))
        self.assertEqual(state["next_q"].shape, (3, 3))
        self.assertLess(state["a"] + state["b"], 1.0)
        self.assertGreater(state["innovation_df"], 2.0)

    def test_simulation_shape_and_weights(self):
        scenarios, probabilities = simulate_dcc_terminal(self.state, 5, 2_000, seed=1)
        self.assertEqual(scenarios.shape, (2_000, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.isfinite(scenarios).all())

    def test_terminal_mean_is_imposed_when_requested(self):
        target = np.array([0.004, -0.002, 0.001])
        scenarios, _ = simulate_dcc_terminal(
            self.state, 5, 4_000, seed=2, terminal_mean=target
        )
        np.testing.assert_allclose(scenarios.mean(axis=0), target, atol=1e-12)

    def test_longer_horizon_widens_the_terminal_distribution(self):
        short, _ = simulate_dcc_terminal(self.state, 1, 4_000, seed=3)
        long, _ = simulate_dcc_terminal(self.state, 10, 4_000, seed=3)
        self.assertGreater(long.std(axis=0).mean(), short.std(axis=0).mean())

    def test_simulation_preserves_cross_sectional_dependence(self):
        # El panel sintético tiene un factor común; la correlación simulada no
        # debe colapsar a cero.
        scenarios, _ = simulate_dcc_terminal(self.state, 5, 5_000, seed=4)
        correlation = np.corrcoef(scenarios, rowvar=False)
        offdiag = correlation[~np.eye(3, dtype=bool)]
        self.assertGreater(float(offdiag.mean()), 0.1)

    def test_entry_point_returns_diagnostics(self):
        scenarios, probabilities, diagnostics = dcc_garch_terminal(
            self.panel, horizon=5, n_scenarios=1_500, seed=5
        )
        self.assertEqual(scenarios.shape, (1_500, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        for key in ("dcc_a", "dcc_b", "dcc_persistence", "innovation_df"):
            self.assertIn(key, diagnostics)
        self.assertLess(diagnostics["max_garch_persistence"], 1.0)

    def test_rejects_invalid_simulation_parameters(self):
        with self.assertRaises(ValueError):
            simulate_dcc_terminal(self.state, 0, 100, seed=1)
        with self.assertRaises(ValueError):
            simulate_dcc_terminal(self.state, 5, 1, seed=1)


if __name__ == "__main__":
    unittest.main()
