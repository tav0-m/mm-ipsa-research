import unittest

import numpy as np
import pandas as pd

from mm_ipsa.mm.targets import (
    _ewma_weights,
    compute_targets,
    decay_from_half_life,
    effective_sample_size,
    ledoit_wolf_shrinkage,
    resolve_decay_lambda,
    resolve_shrinkage,
)


class TestLedoitWolfShrinkage(unittest.TestCase):
    def _sample(self, rows: int, assets: int, correlation: float, seed: int):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((rows, 1))
        idiosyncratic = rng.standard_normal((rows, assets))
        data = correlation * base + np.sqrt(1.0 - correlation**2) * idiosyncratic
        weights = np.full(rows, 1.0 / rows)
        return data - weights @ data, weights

    def test_pure_noise_shrinks_almost_completely(self):
        # Pocas observaciones y activos independientes: las covarianzas
        # cruzadas son ruido y el estimador debe colapsar a la diagonal.
        deviations, weights = self._sample(25, 12, 0.0, 1)
        result = ledoit_wolf_shrinkage(deviations, weights)
        self.assertGreater(result["intensity"], 0.5)

    def test_strong_signal_barely_shrinks(self):
        # Muchas observaciones y correlacion real fuerte: casi no hay que
        # contraer porque las covarianzas cruzadas estan bien estimadas.
        deviations, weights = self._sample(5_000, 4, 0.8, 2)
        result = ledoit_wolf_shrinkage(deviations, weights)
        self.assertLess(result["intensity"], 0.05)

    def test_intensity_decreases_with_sample_size(self):
        small = ledoit_wolf_shrinkage(*self._sample(40, 8, 0.4, 3))["intensity"]
        large = ledoit_wolf_shrinkage(*self._sample(4_000, 8, 0.4, 3))["intensity"]
        self.assertGreater(small, large)

    def test_intensity_stays_inside_unit_interval(self):
        for rows in (20, 100, 1_000):
            result = ledoit_wolf_shrinkage(*self._sample(rows, 6, 0.3, rows))
            self.assertGreaterEqual(result["intensity"], 0.0)
            self.assertLessEqual(result["intensity"], 1.0)

    def test_rejects_malformed_inputs(self):
        deviations, weights = self._sample(30, 5, 0.2, 4)
        with self.assertRaises(ValueError):
            ledoit_wolf_shrinkage(deviations[:, :1], weights)
        with self.assertRaises(ValueError):
            ledoit_wolf_shrinkage(deviations, weights[:-1])
        with self.assertRaises(ValueError):
            ledoit_wolf_shrinkage(deviations, np.full(30, 1.0))

    def test_resolver_reports_provenance(self):
        deviations, weights = self._sample(200, 5, 0.3, 5)
        fixed_value, fixed_report = resolve_shrinkage(0.25, deviations, weights)
        self.assertEqual(fixed_value, 0.25)
        self.assertEqual(fixed_report["covariance_shrinkage_mode"], "fixed")

        auto_value, auto_report = resolve_shrinkage("auto", deviations, weights)
        self.assertEqual(auto_report["covariance_shrinkage_mode"], "ledoit_wolf")
        self.assertIn("offdiagonal_noise", auto_report)
        self.assertGreaterEqual(auto_value, 0.0)

    def test_resolver_rejects_invalid_settings(self):
        deviations, weights = self._sample(30, 4, 0.1, 6)
        with self.assertRaises(ValueError):
            resolve_shrinkage("ledoit", deviations, weights)
        with self.assertRaises(ValueError):
            resolve_shrinkage(1.5, deviations, weights)


class TestTargets(unittest.TestCase):
    def test_eleven_week_half_life_is_fifty_five_daily_rows(self):
        decay = decay_from_half_life(11.0, 5.0)
        self.assertAlmostEqual(decay**55, 0.5, places=12)
        self.assertGreater(decay, 0.98)

    def test_point_94_has_an_eleven_row_half_life(self):
        half_life_rows = np.log(0.5) / np.log(0.94)
        self.assertAlmostEqual(half_life_rows, 11.20, places=2)

    def test_effective_sample_size(self):
        weights = _ewma_weights(1000, 0.94)
        self.assertAlmostEqual(effective_sample_size(weights), (1 + 0.94) / (1 - 0.94), places=8)

    def test_half_life_config_has_precedence(self):
        decay = resolve_decay_lambda({"ewma_half_life_weeks": 11, "decay_lambda": 0.5})
        self.assertAlmostEqual(decay**55, 0.5, places=12)

    def test_shrinkage_preserves_variances_and_reduces_covariance(self):
        rng = np.random.default_rng(10)
        values = rng.normal(size=(50, 3)) * 0.01
        terminal = pd.DataFrame(values, columns=list("abc"))
        daily = terminal.copy()
        _, raw, _ = compute_targets(terminal, daily, covariance_shrinkage=0.0)
        _, shrunk, _ = compute_targets(terminal, daily, covariance_shrinkage=0.25)
        np.testing.assert_allclose(np.diag(raw), np.diag(shrunk))
        mask = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(shrunk[mask], 0.75 * raw[mask])


if __name__ == "__main__":
    unittest.main()
