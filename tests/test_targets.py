import unittest

import numpy as np
import pandas as pd

from src.mm.targets import (
    _ewma_weights,
    compute_targets,
    decay_from_half_life,
    effective_sample_size,
    resolve_decay_lambda,
)


class TestTargets(unittest.TestCase):
    def test_eleven_week_half_life_is_fifty_five_daily_rows(self):
        decay = decay_from_half_life(11.0, 5.0)
        self.assertAlmostEqual(decay**55, 0.5, places=12)
        self.assertGreater(decay, 0.98)

    def test_legacy_point_94_is_about_eleven_rows_not_weeks(self):
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
