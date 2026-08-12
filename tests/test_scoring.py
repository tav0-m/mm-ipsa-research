import unittest

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.scoring import (
    christoffersen_test,
    crps_ensemble,
    evaluate_scenarios,
    kupiec_test,
    lower_tail_mean,
    score_scenarios_by_observation,
    weighted_quantile,
)


class TestScoring(unittest.TestCase):
    def test_crps_two_point_distribution(self):
        # E|X-y|=1 y 0.5*E|X-X'|=0.5 para X in {-1,1}, y=0.
        self.assertAlmostEqual(crps_ensemble(np.array([-1.0, 1.0]), np.array([0.5, 0.5]), 0.0), 0.5)

    def test_fractional_expected_shortfall(self):
        values = np.array([-10.0, -2.0, 3.0])
        weights = np.array([0.02, 0.08, 0.90])
        # Cola 5%: 2% a -10 y 3% a -2.
        self.assertAlmostEqual(lower_tail_mean(values, weights, 0.05), -5.2)
        self.assertEqual(weighted_quantile(values, weights, 0.05), -2.0)

    def test_coverage_tests_return_valid_probabilities(self):
        hits = np.array([0] * 18 + [1] + [0] * 20 + [1] + [0] * 20, dtype=bool)
        uc = kupiec_test(hits, 0.05)
        ind = christoffersen_test(hits)
        self.assertTrue(0 <= uc["pvalue_uc"] <= 1)
        self.assertTrue(0 <= ind["pvalue_ind"] <= 1)

    def test_observation_scores_aggregate_to_summary(self):
        rng = np.random.default_rng(17)
        scenarios = rng.normal(size=(30, 3))
        probabilities = rng.dirichlet(np.ones(30))
        observations = rng.normal(size=(8, 3))
        ids = pd.date_range("2024-01-01", periods=8, freq="5D")
        by_observation = score_scenarios_by_observation(
            "test",
            scenarios,
            probabilities,
            observations,
            observation_ids=ids.tolist(),
            seed=99,
            energy_pair_samples=2_000,
        )
        _, aggregate = evaluate_scenarios(
            "test",
            scenarios,
            probabilities,
            observations,
            ["A", "B", "C"],
            seed=99,
            energy_pair_samples=2_000,
        )
        for metric in ("mean_crps", "energy_score", "variogram_score"):
            observed_mean = by_observation[metric].to_numpy(dtype=float).mean()
            expected_mean = aggregate[metric].to_numpy(dtype=float)[0]
            self.assertAlmostEqual(
                float(observed_mean), float(expected_mean)
            )


if __name__ == "__main__":
    unittest.main()
