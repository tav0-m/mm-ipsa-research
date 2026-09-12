import unittest
from typing import cast

import numpy as np
import pandas as pd

from mm_ipsa.analysis.stressed_calibration import (
    CANONICAL_NAMES,
    _canonical,
    calibrate_on_window,
    compare_calibrations,
    realised_volatility,
)
from mm_ipsa.evaluation.procyclicality import CALM, STRESSED

ALPHA = 0.05
ASSETS = ["A", "B", "C"]


def _config() -> dict:
    return {
        "portfolio": {"alpha_cvar": ALPHA},
        "data": {"H": 5},
        "evaluation": {"seed": 0},
    }


def _predictive(scale: float, seed: int, size: int = 3_000):
    draws = np.random.default_rng(seed).standard_normal((size, len(ASSETS))) * scale
    return draws, np.full(size, 1.0 / size)


def _windows(calm: int = 60, stressed: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    quiet = rng.standard_normal((calm, len(ASSETS))) * 0.01
    rough = rng.standard_normal((stressed, len(ASSETS))) * 0.05
    frame = pd.DataFrame(np.vstack([quiet, rough]), columns=ASSETS)
    regimes = pd.Series([CALM] * calm + [STRESSED] * stressed)
    return frame, regimes


class TestCanonicalNames(unittest.TestCase):
    def test_maps_the_constructor_names_to_the_published_ones(self):
        renamed = _canonical(
            {
                "MM": 1,
                "gaussian_terminal": 2,
                "student_t_terminal": 3,
                "historical_weighted": 4,
                "dcc_garch": 5,
            }
        )
        self.assertEqual(
            set(renamed), {"MM", "gaussian", "student_t", "historical", "dcc_garch"}
        )

    def test_leaves_already_canonical_names_untouched(self):
        self.assertEqual(_canonical({"MM": 1, "dcc_garch": 2}), {"MM": 1, "dcc_garch": 2})

    def test_every_mapped_name_changes(self):
        for source, target in CANONICAL_NAMES.items():
            self.assertNotEqual(source, target)


class TestCalibrationWindow(unittest.TestCase):
    def test_rejects_a_window_too_short_to_calibrate(self):
        daily = pd.DataFrame(
            np.zeros((30, len(ASSETS))),
            index=pd.bdate_range("2020-01-01", periods=30),
            columns=ASSETS,
        )
        with self.assertRaises(ValueError):
            calibrate_on_window(daily, "2020-01-01", "2020-02-10", _config())


class TestRealisedVolatility(unittest.TestCase):
    def test_separates_a_quiet_window_from_a_rough_one(self):
        rng = np.random.default_rng(1)
        values = np.vstack(
            [
                rng.standard_normal((250, len(ASSETS))) * 0.02,
                rng.standard_normal((250, len(ASSETS))) * 0.005,
            ]
        )
        daily = pd.DataFrame(
            values, index=pd.bdate_range("2020-01-01", periods=500), columns=ASSETS
        )
        rough = realised_volatility(daily, "2020-01-01", "2020-12-15")
        quiet = realised_volatility(daily, "2021-01-01", "2021-12-01")
        self.assertGreater(rough, 2.0 * quiet)


class TestCompareCalibrations(unittest.TestCase):
    def setUp(self):
        self.windows, self.regimes = _windows()
        self.calibrations = {
            "calma": {"m": _predictive(0.010, seed=2)},
            "estres": {"m": _predictive(0.035, seed=3)},
        }

    def test_the_reference_calibration_has_unit_capital_ratio(self):
        table = compare_calibrations(
            self.calibrations, self.windows, self.regimes, _config()
        )
        reference = table.loc[table["calibration"] == "calma", "capital_ratio"]
        self.assertTrue(np.allclose(reference, 1.0))

    def test_a_wider_calibration_costs_more_capital(self):
        table = compare_calibrations(
            self.calibrations, self.windows, self.regimes, _config()
        ).set_index("calibration")
        self.assertGreater(cast(float, table.loc["estres", "capital_ratio"]), 2.0)

    def test_a_wider_calibration_breaches_less_often(self):
        table = compare_calibrations(
            self.calibrations, self.windows, self.regimes, _config()
        ).set_index("calibration")
        self.assertLess(
            cast(float, table.loc["estres", "stressed_breach_rate"]),
            cast(float, table.loc["calma", "stressed_breach_rate"]),
        )

    def test_a_narrow_calibration_concentrates_breaches_in_stress(self):
        table = compare_calibrations(
            self.calibrations, self.windows, self.regimes, _config()
        ).set_index("calibration")
        self.assertGreater(
            cast(float, table.loc["calma", "stressed_breach_rate"]),
            cast(float, table.loc["calma", "calm_breach_rate"]),
        )

    def test_excess_columns_are_measured_against_the_nominal_level(self):
        table = compare_calibrations(
            self.calibrations, self.windows, self.regimes, _config()
        )
        self.assertTrue(
            np.allclose(
                table["calm_excess"], table["calm_breach_rate"] - ALPHA
            )
        )
        self.assertTrue(
            np.allclose(
                table["stressed_excess"], table["stressed_breach_rate"] - ALPHA
            )
        )

    def test_rejects_regimes_that_do_not_align_with_the_windows(self):
        with self.assertRaises(ValueError):
            compare_calibrations(
                self.calibrations, self.windows, self.regimes.iloc[:-5], _config()
            )

    def test_requires_both_regimes_to_be_present(self):
        calm_only = pd.Series([CALM] * len(self.windows))
        with self.assertRaises(ValueError):
            compare_calibrations(
                self.calibrations, self.windows, calm_only, _config()
            )


if __name__ == "__main__":
    unittest.main()
