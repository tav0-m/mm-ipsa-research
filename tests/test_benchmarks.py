import unittest

import numpy as np

from mm_ipsa.models.benchmarks import (
    estimate_student_t_df,
    gaussian_terminal,
    nearest_psd,
    resolve_student_t_df,
    student_t_terminal,
)


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


class TestStudentTDegreesOfFreedom(unittest.TestCase):
    """El estimador debe recuperar nu cuando el modelo esta bien especificado."""

    def _sample(self, degrees_of_freedom: float, size: int, seed: int):
        mean = np.array([0.004, -0.002, 0.001])
        covariance = np.array(
            [
                [0.0030, 0.0011, 0.0006],
                [0.0011, 0.0025, 0.0009],
                [0.0006, 0.0009, 0.0020],
            ]
        )
        scenarios, _ = student_t_terminal(
            mean, covariance, size, degrees_of_freedom, seed
        )
        return scenarios, mean, covariance

    def test_recovers_known_degrees_of_freedom(self):
        for true_df in (6.0, 12.0):
            with self.subTest(true_df=true_df):
                sample, mean, covariance = self._sample(true_df, 200_000, 11)
                estimate = estimate_student_t_df(sample, mean, covariance)
                # Muestra grande y modelo correcto: el sesgo debe ser pequeno
                # en relacion al propio valor, no en valor absoluto, porque la
                # informacion de Fisher sobre nu decae al crecer nu.
                self.assertLess(
                    abs(estimate["degrees_of_freedom"] - true_df) / true_df, 0.15
                )

    def test_gaussian_data_pushes_estimate_to_upper_bound(self):
        mean = np.zeros(3)
        covariance = np.diag([0.002, 0.003, 0.0015])
        sample, _ = gaussian_terminal(mean, covariance, 120_000, 7)
        estimate = estimate_student_t_df(sample, mean, covariance, bounds=(4.5, 60.0))
        # Sin colas pesadas la verosimilitud es monotona creciente en nu.
        self.assertGreater(estimate["degrees_of_freedom"], 40.0)
        self.assertEqual(estimate["at_bound"], 1.0)

    def test_log_likelihood_peaks_at_the_estimate(self):
        sample, mean, covariance = self._sample(8.0, 60_000, 3)
        estimate = estimate_student_t_df(sample, mean, covariance)
        best = estimate["log_likelihood"]
        for offset in (-2.0, -1.0, 1.0, 2.0):
            candidate = estimate["degrees_of_freedom"] + offset
            if not 4.5 < candidate < 60.0:
                continue
            neighbour = estimate_student_t_df(
                sample, mean, covariance, bounds=(candidate - 1e-6, candidate + 1e-6)
            )
            self.assertLessEqual(neighbour["log_likelihood"], best + 1e-9)

    def test_weights_shift_the_estimate_towards_the_weighted_regime(self):
        heavy, mean, covariance = self._sample(5.0, 30_000, 21)
        light, _ = gaussian_terminal(mean, covariance, 30_000, 22)
        stacked = np.vstack([light, heavy])
        # Peso concentrado en el bloque de cola pesada (el segundo).
        weights = np.concatenate([np.full(30_000, 1e-6), np.full(30_000, 1.0)])
        weighted = estimate_student_t_df(stacked, mean, covariance, weights)
        uniform = estimate_student_t_df(stacked, mean, covariance)
        self.assertLess(weighted["degrees_of_freedom"], uniform["degrees_of_freedom"])

    def test_rejects_degenerate_inputs(self):
        sample, mean, covariance = self._sample(7.0, 500, 5)
        with self.assertRaises(ValueError):
            estimate_student_t_df(sample[:2], mean, covariance)
        with self.assertRaises(ValueError):
            estimate_student_t_df(sample, mean, covariance, bounds=(8.0, 6.0))
        with self.assertRaises(ValueError):
            estimate_student_t_df(sample, mean, covariance, np.zeros(len(sample)))

    def test_resolver_honours_fixed_mode(self):
        sample, mean, covariance = self._sample(7.0, 2_000, 9)
        value, report = resolve_student_t_df(
            {"student_t_df_mode": "fixed", "student_t_df": 6.0},
            sample,
            mean,
            covariance,
        )
        self.assertEqual(value, 6.0)
        self.assertEqual(report["student_t_df_mode"], "fixed")

    def test_resolver_estimates_in_mle_mode(self):
        sample, mean, covariance = self._sample(7.0, 40_000, 13)
        value, report = resolve_student_t_df(
            {"student_t_df_mode": "mle", "student_t_df": 6.0},
            sample,
            mean,
            covariance,
        )
        self.assertEqual(report["student_t_df_mode"], "mle")
        self.assertNotAlmostEqual(value, 6.0, places=3)
        self.assertIn("student_t_log_likelihood", report)

    def test_resolver_rejects_unknown_mode(self):
        sample, mean, covariance = self._sample(7.0, 500, 4)
        with self.assertRaises(ValueError):
            resolve_student_t_df(
                {"student_t_df_mode": "grid"}, sample, mean, covariance
            )


class TestNearestPSD(unittest.TestCase):
    def test_relative_floor_regularises_small_scale_matrices(self):
        # Covarianza casi singular con la magnitud tipica de retornos H=5.
        base = np.array([[1e-4, 1e-4], [1e-4, 1e-4]])
        absolute_only = nearest_psd(base, floor=1e-12, relative_floor=0.0)
        relative = nearest_psd(base, floor=1e-12, relative_floor=1e-6)
        self.assertGreater(
            np.linalg.cond(absolute_only), np.linalg.cond(relative) * 100
        )
        np.testing.assert_allclose(relative, relative.T)
        self.assertGreater(np.linalg.eigvalsh(relative).min(), 0.0)

    def test_preserves_well_conditioned_matrices(self):
        covariance = np.array([[0.003, 0.001], [0.001, 0.002]])
        np.testing.assert_allclose(nearest_psd(covariance), covariance, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
