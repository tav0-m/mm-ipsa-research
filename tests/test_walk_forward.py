import unittest

import numpy as np
import pandas as pd

from mm_ipsa.backtest.walk_forward import (
    moving_block_bootstrap_sharpe_difference,
    simulate_strategy,
    walk_forward_weights,
)


class TestRebalanceScheduleValidation(unittest.TestCase):
    def test_missing_rebalance_date_raises(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        returns = pd.DataFrame(np.zeros((20, 2)), index=dates, columns=["a", "b"])
        # Fecha fuera del indice: antes se omitia en silencio y la cartera
        # conservaba la asignacion previa sin dejar rastro.
        schedule = {pd.Timestamp("2024-01-06"): np.array([0.5, 0.5])}
        with self.assertRaises(ValueError) as context:
            simulate_strategy(returns, schedule)
        self.assertIn("2024-01-06", str(context.exception))

    def test_valid_schedule_is_accepted(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        returns = pd.DataFrame(np.zeros((20, 2)), index=dates, columns=["a", "b"])
        wealth, _, _ = simulate_strategy(
            returns, {dates[0]: np.array([0.5, 0.5])}
        )
        self.assertEqual(len(wealth), 20)


class TestSharpeDifferenceInference(unittest.TestCase):
    def test_identical_series_give_zero_difference_and_high_pvalue(self):
        rng = np.random.default_rng(0)
        returns = rng.standard_normal(400) * 0.01
        result = moving_block_bootstrap_sharpe_difference(
            returns, returns.copy(), block_size=8, samples=500, seed=1
        )
        self.assertAlmostEqual(result["sharpe_difference"], 0.0, places=12)
        self.assertGreater(result["pvalue_raw"], 0.5)

    def test_clear_advantage_is_detected(self):
        rng = np.random.default_rng(2)
        common = rng.standard_normal(600) * 0.005
        strategy = 0.002 + common
        benchmark = common
        result = moving_block_bootstrap_sharpe_difference(
            strategy, benchmark, block_size=8, samples=1_000, seed=3
        )
        self.assertGreater(result["sharpe_difference"], 0.0)
        self.assertLess(result["pvalue_raw"], 0.05)
        self.assertGreater(result["ci_low"], 0.0)

    def test_confidence_level_widens_the_interval(self):
        rng = np.random.default_rng(4)
        strategy = rng.standard_normal(300) * 0.01
        benchmark = rng.standard_normal(300) * 0.01
        narrow = moving_block_bootstrap_sharpe_difference(
            strategy, benchmark, block_size=5, samples=800, seed=5, confidence_level=0.80
        )
        wide = moving_block_bootstrap_sharpe_difference(
            strategy, benchmark, block_size=5, samples=800, seed=5, confidence_level=0.99
        )
        self.assertGreater(
            wide["ci_high"] - wide["ci_low"], narrow["ci_high"] - narrow["ci_low"]
        )

    def test_rejects_invalid_parameters(self):
        series = np.zeros(50)
        with self.assertRaises(ValueError):
            moving_block_bootstrap_sharpe_difference(series, series, samples=10)
        with self.assertRaises(ValueError):
            moving_block_bootstrap_sharpe_difference(series, series[:-1])
        with self.assertRaises(ValueError):
            moving_block_bootstrap_sharpe_difference(
                series, series, confidence_level=1.5
            )


class TestWalkForward(unittest.TestCase):
    def test_estimator_never_sees_rebalance_day(self):
        dates = pd.date_range("2020-01-01", periods=900, freq="B")
        returns = pd.DataFrame(np.zeros((900, 2)), index=dates, columns=["a", "b"])
        seen_maxima = []

        def estimator(training):
            seen_maxima.append(training.index.max())
            return np.array([0.5, 0.5])

        schedule = walk_forward_weights(returns, dates[600], estimator, min_history=252)
        for trained_to, rebalance in zip(seen_maxima, schedule):
            self.assertLess(trained_to, rebalance)

    def test_transaction_cost_reduces_wealth(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        returns = pd.DataFrame(0.0, index=dates, columns=["a", "b"])
        schedule = {dates[0]: np.array([1.0, 0.0]), dates[1]: np.array([0.0, 1.0])}
        wealth, execution, info = simulate_strategy(returns, schedule, transaction_cost_bps=100)
        self.assertLess(wealth.iloc[-1], 1.0)
        self.assertGreater(info["total_turnover"], 0.0)
        self.assertGreater(execution["cost_fraction"].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
