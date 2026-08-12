import unittest

import numpy as np
import pandas as pd

from mm_ipsa.backtest.walk_forward import simulate_strategy, walk_forward_weights


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
