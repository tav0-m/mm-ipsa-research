import unittest

import numpy as np
import pandas as pd

from mm_ipsa.backtest.model_schedules import (
    _origin_portfolios,
    walk_forward_model_schedules,
)


def _panel(rows: int, assets: int, seed: int) -> pd.DataFrame:
    """Retornos diarios con factor común, indexados en días hábiles."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(rows) * 0.006
    data = np.column_stack(
        [common + rng.standard_normal(rows) * 0.009 for _ in range(assets)]
    )
    index = pd.bdate_range("2019-01-01", periods=rows)
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(assets)])


def _config(assets: int) -> dict:
    return {
        "data": {"H": 5},
        "mm": {
            "N_scenarios": 40,
            "bcd_max_iter": 12,
            "tol": 1e-4,
            "min_iter": 1,
            "convergence_patience": 2,
            "n_starts": 2,
            "seed": 3,
            "entropy_lambda": 1e-5,
            "update_mode": "joint_lbfgs",
            "p_solver": "mirror_descent",
            "start_strategy": "independent",
            "solution_mode": "ensemble",
            "strict_solver": False,
            "moment_weights": {"k1": 1.0, "k2": 1.0, "k3": 0.5, "k4": 0.25,
                               "cov_weight": 0.05},
            "covariance_shrinkage": "auto",
            "ewma_half_life_weeks": 11.0,
            "observations_per_week": 5.0,
        },
        "benchmarks": {
            "N_scenarios": 400,
            "seed": 7,
            "student_t_df_mode": "fixed",
            "student_t_df": 8.0,
            "include": ["gaussian_terminal", "historical_weighted"],
        },
        "portfolio": {
            "max_weight": 0.6,
            "alpha_cvar": 0.1,
            "rf": 0.0,
            "l2_penalty": 0.01,
        },
        "evaluation": {"seed": 11},
    }


class TestOriginPortfolios(unittest.TestCase):
    def test_builds_three_portfolios_per_model(self):
        rng = np.random.default_rng(1)
        models = {
            "MM": (rng.standard_normal((60, 3)) * 0.02, np.full(60, 1 / 60)),
            "control": (rng.standard_normal((60, 3)) * 0.02, np.full(60, 1 / 60)),
        }
        weights = _origin_portfolios(models, _config(3))
        self.assertEqual(len(weights), 6)
        for name, vector in weights.items():
            with self.subTest(name=name):
                self.assertTrue(name.startswith("WF_"))
                self.assertAlmostEqual(float(vector.sum()), 1.0, places=9)
                self.assertLessEqual(float(vector.max()), 0.6 + 1e-9)


class TestWalkForwardSchedules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = _panel(700, 3, 5)
        cls.start = cls.returns.index[400]
        cls.schedules, cls.audit = walk_forward_model_schedules(
            cls.returns, cls.start, _config(3), frequency="Q", min_history=300
        )

    def test_every_strategy_has_the_same_rebalance_dates(self):
        dates = {frozenset(schedule) for schedule in self.schedules.values()}
        self.assertEqual(len(dates), 1)
        self.assertGreaterEqual(len(next(iter(dates))), 2)

    def test_weights_are_valid_portfolios_at_every_date(self):
        for name, schedule in self.schedules.items():
            for date, weights in schedule.items():
                with self.subTest(name=name, date=date):
                    self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
                    self.assertGreaterEqual(float(weights.min()), -1e-12)

    def test_training_never_reaches_the_rebalance_date(self):
        # La garantía central: cada decisión usa solo observaciones anteriores.
        for _, row in self.audit.iterrows():
            self.assertLess(
                pd.Timestamp(row["training_end"]), pd.Timestamp(row["rebalance_date"])
            )

    def test_training_window_expands_monotonically(self):
        rows = self.audit["training_rows"].tolist()
        self.assertEqual(rows, sorted(rows))
        self.assertGreater(rows[-1], rows[0])

    def test_audit_records_the_support_of_every_model(self):
        for column in ("scenarios_MM", "scenarios_gaussian_terminal"):
            self.assertIn(column, self.audit.columns)
            self.assertTrue((self.audit[column] > 0).all())

    def test_schedules_cover_mm_and_every_declared_control(self):
        prefixes = {name.split("_Min")[0].split("_Max")[0] for name in self.schedules}
        self.assertIn("WF_MM", prefixes)
        self.assertIn("WF_gaussian_terminal", prefixes)
        self.assertIn("WF_historical_weighted", prefixes)

    def test_rejects_an_empty_evaluation_period(self):
        with self.assertRaises(ValueError):
            walk_forward_model_schedules(
                self.returns, self.returns.index[-1] + pd.Timedelta(days=400),
                _config(3), frequency="Q", min_history=300,
            )

    def test_fails_when_no_origin_has_enough_history(self):
        with self.assertRaises(RuntimeError):
            walk_forward_model_schedules(
                self.returns, self.start, _config(3),
                frequency="Q", min_history=10_000,
            )


if __name__ == "__main__":
    unittest.main()
