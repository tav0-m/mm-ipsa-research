import unittest
from typing import cast

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.procyclicality import (
    CALM,
    STRESSED,
    breach_concentration,
    classify_regimes,
    trailing_volatility,
)

HORIZON = 5
LOOKBACK = 20


def _daily(rows: int = 300, assets: int = 3, seed: int = 0, scale: float = 0.01):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, assets)) * scale
    return pd.DataFrame(
        values, index=pd.bdate_range("2024-01-01", periods=rows),
        columns=[f"A{i}" for i in range(assets)],
    )


def _window_ends(daily: pd.DataFrame) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, pd.DatetimeIndex(daily.index[HORIZON - 1 :: HORIZON]))


class TestNoLookahead(unittest.TestCase):
    """La clasificacion no puede usar el movimiento que pretende anticipar."""

    def test_a_shock_inside_the_window_does_not_change_its_own_label(self):
        daily = _daily(seed=1)
        ends = _window_ends(daily)
        original = trailing_volatility(daily, ends, HORIZON, LOOKBACK)

        position = cast(int, daily.index.get_loc(ends[20]))
        contaminated = daily.copy()
        contaminated.iloc[position - HORIZON + 1 : position + 1] *= 50.0

        altered = trailing_volatility(contaminated, ends, HORIZON, LOOKBACK)

        self.assertAlmostEqual(
            float(original.iloc[20]), float(altered.iloc[20]), places=12
        )

    def test_a_shock_before_the_window_does_change_its_label(self):
        daily = _daily(seed=2)
        ends = _window_ends(daily)
        original = trailing_volatility(daily, ends, HORIZON, LOOKBACK)

        position = cast(int, daily.index.get_loc(ends[20]))
        contaminated = daily.copy()
        contaminated.iloc[position - HORIZON - 5 : position - HORIZON] *= 50.0

        altered = trailing_volatility(contaminated, ends, HORIZON, LOOKBACK)

        self.assertGreater(float(altered.iloc[20]), 2.0 * float(original.iloc[20]))

    def test_windows_without_enough_history_are_not_classified(self):
        daily = _daily(seed=3)
        volatility = trailing_volatility(daily, _window_ends(daily), HORIZON, LOOKBACK)
        self.assertTrue(np.isnan(volatility.iloc[0]))
        self.assertTrue(np.isfinite(volatility.iloc[-1]))


class TestTrailingVolatility(unittest.TestCase):
    def test_tracks_a_change_in_scale(self):
        daily = _daily(rows=400, seed=4, scale=0.005)
        daily.iloc[200:] *= 6.0
        volatility = trailing_volatility(daily, _window_ends(daily), HORIZON, LOOKBACK)
        early = float(volatility.dropna().iloc[5])
        late = float(volatility.dropna().iloc[-5])
        self.assertGreater(late, 3.0 * early)

    def test_rejects_invalid_parameters(self):
        daily = _daily(seed=5)
        ends = _window_ends(daily)
        with self.assertRaises(ValueError):
            trailing_volatility(daily, ends, 0, LOOKBACK)
        with self.assertRaises(ValueError):
            trailing_volatility(daily, ends, HORIZON, 1)


class TestClassifyRegimes(unittest.TestCase):
    def test_splits_at_the_requested_quantile(self):
        volatility = pd.Series(np.arange(100, dtype=float))
        labels = classify_regimes(volatility, 0.75)
        self.assertEqual(int((labels == STRESSED).sum()), 25)
        self.assertEqual(int((labels == CALM).sum()), 75)

    def test_missing_volatility_stays_unlabelled(self):
        volatility = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
        labels = classify_regimes(volatility, 0.5)
        self.assertTrue(labels.iloc[:2].isna().all())
        self.assertFalse(labels.iloc[2:].isna().any())

    def test_rejects_an_invalid_quantile(self):
        volatility = pd.Series([1.0, 2.0, 3.0])
        for quantile in (0.0, 1.0, -0.5):
            with self.assertRaises(ValueError):
                classify_regimes(volatility, quantile)

    def test_rejects_a_series_without_any_volatility(self):
        with self.assertRaises(ValueError):
            classify_regimes(pd.Series([np.nan, np.nan]), 0.5)


class TestBreachConcentration(unittest.TestCase):
    def _regimes(self, calm: int, stressed: int) -> pd.Series:
        return pd.Series([CALM] * calm + [STRESSED] * stressed)

    def test_detects_breaches_concentrated_in_stress(self):
        losses = np.concatenate([np.full(80, 0.01), np.full(20, -0.10)])
        result = breach_concentration(losses, self._regimes(80, 20), -0.05, 0.05)

        self.assertEqual(result["stressed"]["breaches"], 20)
        self.assertEqual(result["calm"]["breaches"], 0)
        self.assertTrue(result["concentrates_in_stress"])
        self.assertLess(result["pvalue"], 0.001)

    def test_uniform_breaches_are_not_flagged(self):
        rng = np.random.default_rng(6)
        losses = rng.standard_normal(200) * 0.02
        regimes = self._regimes(150, 50)
        result = breach_concentration(losses, regimes, float(np.quantile(losses, 0.05)), 0.05)
        self.assertGreater(result["pvalue"], 0.05)
        self.assertFalse(result["concentrates_in_stress"])

    def test_reports_mean_depth_of_the_breaches(self):
        losses = np.concatenate([np.full(90, 0.01), np.full(10, -0.09)])
        result = breach_concentration(losses, self._regimes(90, 10), -0.05, 0.05)
        self.assertAlmostEqual(result["stressed"]["mean_depth"], 0.04, places=10)
        self.assertTrue(np.isnan(result["calm"]["mean_depth"]))

    def test_flags_too_few_breaches_as_unreliable(self):
        losses = np.concatenate([np.full(95, 0.01), np.full(5, -0.06)])
        result = breach_concentration(losses, self._regimes(95, 5), -0.05, 0.05)
        self.assertLess(result["total_breaches"], 8)
        self.assertFalse(result["reliable"])

    def test_requires_both_regimes(self):
        losses = np.zeros(50)
        with self.assertRaises(ValueError):
            breach_concentration(losses, pd.Series([CALM] * 50), -0.05, 0.05)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            breach_concentration(np.zeros(50), self._regimes(30, 10), -0.05, 0.05)


if __name__ == "__main__":
    unittest.main()
