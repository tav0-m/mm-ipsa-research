import unittest

import numpy as np
import pandas as pd

from mm_ipsa.analysis.scalability import moment_fit, scalability_sweep


class _Objective:
    """Doble de prueba que devuelve momentos y covarianza prefijados."""

    def __init__(self, moments: np.ndarray, covariance: np.ndarray):
        self._moments = moments
        self._covariance = covariance

    def compute_moments(self, scenarios, probabilities):
        return self._moments, self._moments[0], self._covariance


def _targets(assets: int = 4):
    moments = np.vstack(
        [
            np.full(assets, 0.01),
            np.full(assets, 0.0004),
            np.full(assets, 1e-6),
            np.full(assets, 5e-7),
        ]
    )
    covariance = np.full((assets, assets), 1e-4) + np.eye(assets) * 3e-4
    return moments, covariance


class TestMomentFit(unittest.TestCase):
    def test_a_perfect_fit_reports_zero_error(self):
        moments, covariance = _targets()
        report = moment_fit(
            _Objective(moments, covariance), np.zeros((2, 4)), np.zeros(2),
            moments, covariance,
        )
        self.assertTrue(all(value == 0.0 for value in report.values()))

    def test_reports_the_maximum_and_not_the_average(self):
        moments, covariance = _targets()
        perturbed = moments.copy()
        # Un unico momento desviado un diez por ciento sobre cuatro activos.
        perturbed[1, 0] *= 1.10

        report = moment_fit(
            _Objective(perturbed, covariance), np.zeros((2, 4)), np.zeros(2),
            moments, covariance,
        )

        self.assertAlmostEqual(report["variance_error"], 0.10, places=10)
        self.assertEqual(report["mean_error"], 0.0)

    def test_separates_marginal_error_from_dependence_error(self):
        moments, covariance = _targets()
        perturbed = covariance.copy()
        perturbed[0, 1] *= 1.5
        perturbed[1, 0] *= 1.5

        report = moment_fit(
            _Objective(moments, perturbed), np.zeros((2, 4)), np.zeros(2),
            moments, covariance,
        )

        self.assertAlmostEqual(report["covariance_error"], 0.5, places=10)
        self.assertEqual(report["variance_error"], 0.0)

    def test_ignores_the_diagonal_of_the_covariance(self):
        # La varianza ya se contrasta como segundo momento; contarla otra vez
        # en el termino de dependencia duplicaria su peso en el diagnostico.
        moments, covariance = _targets()
        perturbed = covariance.copy()
        np.fill_diagonal(perturbed, np.diag(covariance) * 4.0)

        report = moment_fit(
            _Objective(moments, perturbed), np.zeros((2, 4)), np.zeros(2),
            moments, covariance,
        )

        self.assertEqual(report["covariance_error"], 0.0)


class TestSweepShape(unittest.TestCase):
    def test_produces_one_row_per_panel_and_variant(self):
        calls: list[tuple[str, int]] = []

        def fake_solver(prices, main_cfg, **overrides):
            calls.append((prices.columns[0], overrides.get("N_scenarios", 0)))
            return {
                "assets": prices.shape[1],
                "covariance_targets": 0,
                "targets": 0,
                "n_scenarios": overrides.get("N_scenarios", 500),
                "max_iterations": 150,
                "tolerance": 1e-5,
                "objective": 1e-10,
                "seconds": 1.0,
                "iterations_used": 7,
                "stationary_starts": 5,
                "mean_error": 0.0,
                "variance_error": 0.0,
                "third_moment_error": 0.0,
                "fourth_moment_error": 0.0,
                "covariance_error": 0.0,
            }

        import mm_ipsa.analysis.scalability as module

        original = module.solve_with_overrides
        module.solve_with_overrides = fake_solver
        try:
            panels = {
                "chico": pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]}),
                "grande": pd.DataFrame({"C": [1.0, 2.0], "D": [1.0, 2.0]}),
            }
            table = scalability_sweep(
                panels, {"data": {"H": 5}},
                variants={"a": {}, "b": {"N_scenarios": 2000}},
            )
        finally:
            module.solve_with_overrides = original

        self.assertEqual(len(table), 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(set(table["panel"]), {"chico", "grande"})
        self.assertEqual(set(table["variant"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
