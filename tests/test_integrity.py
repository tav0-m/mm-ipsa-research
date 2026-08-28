import unittest

import numpy as np
import pandas as pd

from mm_ipsa.data.integrity import (
    audit_prices,
    calendar_integrity,
    corporate_action_candidates,
    extreme_returns,
    robust_scale,
    stale_price_report,
)


def _random_walk(rows: int = 400, assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((rows, assets)) * 0.015
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    index = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame(prices, index=index, columns=[f"A{i}" for i in range(assets)])


class TestRobustScale(unittest.TestCase):
    def test_matches_sigma_for_gaussian_sample(self):
        sample = np.random.default_rng(3).standard_normal(20_000)
        self.assertAlmostEqual(robust_scale(sample), 1.0, delta=0.03)

    def test_ignores_contamination_that_inflates_the_sample_deviation(self):
        sample = np.random.default_rng(4).standard_normal(2_000)
        contaminated = np.concatenate([sample, np.full(20, 50.0)])
        self.assertGreater(contaminated.std(), 2.0 * sample.std())
        self.assertAlmostEqual(robust_scale(contaminated), robust_scale(sample), delta=0.05)


class TestCorporateActionDetection(unittest.TestCase):
    def test_detects_an_unadjusted_two_for_one_split(self):
        prices = _random_walk(seed=1)
        prices.iloc[200:, 0] /= 2.0

        found = corporate_action_candidates(prices)
        suspected = found.loc[found["suspected"]]

        self.assertEqual(len(suspected), 1)
        row = suspected.iloc[0]
        self.assertEqual(row["asset"], "A0")
        self.assertEqual(row["date"], str(prices.index[200].date()))
        self.assertAlmostEqual(row["nearest_split_ratio"], 0.5, places=10)
        self.assertGreater(row["level_persistence"], 0.9)

    def test_detects_a_reverse_split(self):
        prices = _random_walk(seed=2)
        prices.iloc[150:, 1] *= 3.0

        found = corporate_action_candidates(prices)
        suspected = found.loc[found["suspected"]]

        self.assertEqual(len(suspected), 1)
        self.assertAlmostEqual(suspected.iloc[0]["nearest_split_ratio"], 3.0, places=10)

    def test_clean_series_produces_no_suspects(self):
        found = corporate_action_candidates(_random_walk(seed=5))
        self.assertEqual(int(found["suspected"].sum()) if len(found) else 0, 0)

    def test_transient_crash_at_a_split_ratio_is_not_flagged(self):
        # Un desplome del 50% que se revierte al dia siguiente cae exactamente
        # sobre una razon de split. Solo la persistencia lo distingue.
        prices = _random_walk(seed=6)
        original = float(prices.iloc[300, 2])
        prices.iloc[300, 2] = original * 0.5

        found = corporate_action_candidates(prices)
        matched = found.loc[found["date"] == str(prices.index[300].date())]

        self.assertEqual(len(matched), 1)
        self.assertFalse(bool(matched.iloc[0]["suspected"]))
        self.assertLess(matched.iloc[0]["level_persistence"], 0.8)

    def test_detects_a_split_contaminated_by_the_move_of_the_day(self):
        # Un split real no produce la razon exacta: el precio observado incluye
        # tambien el movimiento de mercado de esa jornada. Con una tolerancia
        # fija del dos por ciento este caso -desviado un 2.2%- se perdia.
        prices = _random_walk(seed=2)
        prices.iloc[150:, 1] *= 3.0
        observed = float(prices.iloc[150, 1] / prices.iloc[149, 1])
        self.assertGreater(abs(observed / 3.0 - 1.0), 0.02)

        found = corporate_action_candidates(prices)

        self.assertEqual(len(found.loc[found["suspected"]]), 1)

    def test_detection_survives_a_wide_range_of_market_moves(self):
        detected = 0
        for seed in range(30):
            prices = _random_walk(seed=100 + seed)
            prices.iloc[200:, 0] /= 2.0
            found = corporate_action_candidates(prices)
            detected += int(found["suspected"].sum())
        self.assertEqual(detected, 30)

    def test_rejects_non_positive_prices(self):
        prices = _random_walk(seed=7)
        prices.iloc[10, 0] = 0.0
        with self.assertRaises(ValueError):
            corporate_action_candidates(prices)

    def test_persistence_window_must_allow_a_median(self):
        with self.assertRaises(ValueError):
            corporate_action_candidates(_random_walk(), persistence_window=1)


class TestExtremeReturns(unittest.TestCase):
    def test_flags_an_injected_outlier(self):
        prices = _random_walk(seed=8)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        returns.iloc[100, 0] = 0.40

        found = extreme_returns(returns)

        self.assertIn("A0", set(found["asset"]))
        self.assertEqual(
            found.loc[found["asset"] == "A0", "date"].iloc[0],
            str(returns.index[100].date()),
        )

    def test_threshold_is_monotone_in_the_number_of_hits(self):
        returns = np.log(_random_walk(seed=9)).diff().dropna(how="all")
        self.assertGreaterEqual(
            len(extreme_returns(returns, threshold=3.0)),
            len(extreme_returns(returns, threshold=8.0)),
        )


class TestStalePrices(unittest.TestCase):
    def test_counts_a_frozen_stretch(self):
        prices = _random_walk(seed=10)
        prices.iloc[50:56, 0] = float(prices.iloc[50, 0])

        report = stale_price_report(prices).set_index("asset")

        self.assertEqual(report.loc["A0", "longest_unchanged_run"], 5)
        self.assertEqual(report.loc["A0", "unchanged_days"], 5)

    def test_continuous_series_has_no_unchanged_days(self):
        report = stale_price_report(_random_walk(seed=11))
        self.assertEqual(int(report["unchanged_days"].sum()), 0)


class TestCalendarIntegrity(unittest.TestCase):
    def test_clean_index_passes(self):
        summary = calendar_integrity(_random_walk(seed=12))
        self.assertEqual(summary["duplicated_dates"], 0)
        self.assertEqual(summary["weekend_rows"], 0)
        self.assertTrue(summary["monotonic_increasing"])

    def test_detects_duplicates_and_weekends(self):
        prices = _random_walk(rows=10, seed=13)
        prices.index = pd.DatetimeIndex(
            ["2021-01-04"] * 2 + ["2021-01-09"] + list(prices.index[3:].astype(str))
        )
        summary = calendar_integrity(prices)
        self.assertEqual(summary["duplicated_dates"], 1)
        self.assertEqual(summary["weekend_rows"], 1)


class TestAuditPrices(unittest.TestCase):
    def test_clean_series_passes_without_blocking_findings(self):
        report = audit_prices(_random_walk(seed=14))
        self.assertTrue(report["passed"])
        self.assertEqual(report["blocking"], [])

    def test_unadjusted_split_blocks_the_audit(self):
        prices = _random_walk(seed=15)
        prices.iloc[180:, 0] /= 2.0

        report = audit_prices(prices)

        self.assertFalse(report["passed"])
        self.assertEqual(len(report["blocking"]), 1)
        self.assertIn("split no ajustado", report["blocking"][0])

    def test_market_outliers_are_reported_without_blocking(self):
        prices = _random_walk(seed=16)
        prices.iloc[220, 1] *= 0.80

        report = audit_prices(prices)

        self.assertTrue(report["passed"])
        self.assertGreater(len(report["extreme_returns"]), 0)


if __name__ == "__main__":
    unittest.main()
