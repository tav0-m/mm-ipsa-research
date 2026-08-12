import unittest

import numpy as np
import pandas as pd

from src.data.download import _extract_close, _max_missing_run


class TestDownload(unittest.TestCase):
    def test_maximum_missing_run(self):
        series = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan])
        self.assertEqual(_max_missing_run(series), 2)

    def test_extract_close_preserves_requested_ticker_order(self):
        columns = pd.MultiIndex.from_product([["Close", "Open"], ["B.SN", "A.SN"]])
        raw = pd.DataFrame([[2.0, 1.0, 2.1, 1.1]], columns=columns)
        result = _extract_close(raw, ["A.SN", "B.SN"])
        self.assertEqual(result.columns.tolist(), ["A.SN", "B.SN"])
        self.assertEqual(result.iloc[0].tolist(), [1.0, 2.0])

    def test_extract_close_rejects_missing_ticker(self):
        columns = pd.MultiIndex.from_product([["Close"], ["A.SN"]])
        raw = pd.DataFrame([[1.0]], columns=columns)
        with self.assertRaises(RuntimeError):
            _extract_close(raw, ["A.SN", "B.SN"])


if __name__ == "__main__":
    unittest.main()
