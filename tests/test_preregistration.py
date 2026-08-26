import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from mm_ipsa.analysis.preregistration import (
    audit_preregistration,
    confirmatory_readiness,
    freeze,
    load_preregistration,
    summarise,
    write_preregistration,
)


def _project(root: Path) -> None:
    """Arbol minimo con los archivos que el sello cubre."""
    (root / "research").mkdir(parents=True, exist_ok=True)
    (root / "src/mm_ipsa/mm").mkdir(parents=True, exist_ok=True)
    (root / "src/mm_ipsa/models").mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("mm:\n  N_scenarios: 500\n", encoding="utf-8")
    (root / "research/PROTOCOL.md").write_text("# Protocolo\n", encoding="utf-8")
    (root / "research/rolling_origin.yaml").write_text("folds: []\n", encoding="utf-8")
    (root / "research/liquidity_robustness.yaml").write_text("id: x\n", encoding="utf-8")
    (root / "src/mm_ipsa/mm/bcd.py").write_text("VALOR = 1\n", encoding="utf-8")
    (root / "src/mm_ipsa/models/benchmarks.py").write_text("VALOR = 2\n", encoding="utf-8")


def _future(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


class TestFreeze(unittest.TestCase):
    def test_records_specification_and_implementation_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project(root)
            document = freeze(root, _future(7), version="1.2.3")
            self.assertEqual(len(document["specification"]), 4)
            self.assertIn("config.yaml", document["specification"])
            self.assertIn("src/mm_ipsa/mm/bcd.py", document["implementation"])
            self.assertNotIn("config.yaml", document["implementation"])
            self.assertEqual(document["frozen_version"], "1.2.3")

    def test_rejects_a_start_that_already_passed(self):
        # Un periodo transcurrido no puede sellarse: nada garantiza que no haya
        # sido observado antes de declararlo confirmatorio.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project(root)
            for offset in (0, -1, -30):
                with self.subTest(offset=offset):
                    with self.assertRaises(ValueError):
                        freeze(root, _future(offset), version="1.0.0")

    def test_rejects_a_trivial_minimum_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project(root)
            with self.assertRaises(ValueError):
                freeze(root, _future(7), version="1.0.0", minimum_windows=1)

    def test_declares_what_is_prohibited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project(root)
            document = freeze(root, _future(7), version="1.0.0")
            self.assertGreaterEqual(len(document["prohibited_after_freeze"]), 3)
            self.assertIn("metric", document["primary_contrast"])


class TestAudit(unittest.TestCase):
    def _sealed(self, root: Path) -> dict:
        _project(root)
        return freeze(root, _future(10), version="1.0.0")

    def test_intact_project_reports_no_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._sealed(root)
            audit = audit_preregistration(document, root)
            self.assertTrue(audit["specification_intact"])
            self.assertEqual(audit["implementation_drift"], [])

    def test_specification_change_breaks_the_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._sealed(root)
            (root / "config.yaml").write_text(
                "mm:\n  N_scenarios: 600\n", encoding="utf-8"
            )
            audit = audit_preregistration(document, root)
            self.assertFalse(audit["specification_intact"])
            self.assertEqual(audit["specification_drift"], ["config.yaml"])

    def test_implementation_change_is_recorded_but_not_fatal(self):
        # Una reescritura numericamente equivalente no altera el experimento;
        # debe quedar registrada sin invalidar el sello.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._sealed(root)
            (root / "src/mm_ipsa/mm/bcd.py").write_text("VALOR = 1  # rapido\n", encoding="utf-8")
            audit = audit_preregistration(document, root)
            self.assertTrue(audit["specification_intact"])
            self.assertEqual(audit["implementation_drift"], ["src/mm_ipsa/mm/bcd.py"])

    def test_a_deleted_sealed_file_counts_as_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._sealed(root)
            (root / "research/PROTOCOL.md").unlink()
            audit = audit_preregistration(document, root)
            self.assertIn("research/PROTOCOL.md", audit["specification_drift"])

    def test_a_new_implementation_file_counts_as_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._sealed(root)
            (root / "src/mm_ipsa/mm/extra.py").write_text("X = 0\n", encoding="utf-8")
            audit = audit_preregistration(document, root)
            self.assertIn("src/mm_ipsa/mm/extra.py", audit["implementation_drift"])


class TestReadiness(unittest.TestCase):
    def _document(self, start: str, minimum: int = 4) -> dict:
        return {
            "frozen_at": "2026-01-01",
            "confirmatory_start": start,
            "minimum_evaluation_windows": minimum,
            "specification": {},
            "implementation": {},
            "primary_contrast": {"metric": "mean_crps"},
        }

    def test_counts_only_observations_after_the_declared_start(self):
        index = pd.DatetimeIndex(pd.bdate_range("2026-01-01", periods=60))
        readiness = confirmatory_readiness(
            self._document("2026-02-01"), index, horizon=5,
            today=date(2026, 4, 1),
        )
        # Solo las fechas desde febrero cuentan, agrupadas de a cinco.
        eligible = index[index >= pd.Timestamp("2026-02-01")]
        self.assertEqual(readiness["windows_available"], len(eligible) // 5)

    def test_is_not_ready_below_the_declared_minimum(self):
        index = pd.DatetimeIndex(pd.bdate_range("2026-02-01", periods=10))
        readiness = confirmatory_readiness(
            self._document("2026-02-01", minimum=40), index, horizon=5,
            today=date(2026, 3, 1),
        )
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["status"], "awaiting_confirmatory_sample")

    def test_becomes_ready_once_the_sample_arrives(self):
        index = pd.DatetimeIndex(pd.bdate_range("2026-02-02", periods=30))
        readiness = confirmatory_readiness(
            self._document("2026-02-01", minimum=4), index, horizon=5,
            today=date(2026, 4, 1),
        )
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["status"], "confirmatory_ready")

    def test_handles_an_empty_observation_index(self):
        readiness = confirmatory_readiness(
            self._document("2026-02-01"), None, horizon=5, today=date(2026, 3, 1)
        )
        self.assertEqual(readiness["windows_available"], 0)
        self.assertIsNone(readiness["latest_observation"])
        self.assertFalse(readiness["ready"])


class TestRoundTrip(unittest.TestCase):
    def test_written_seal_reloads_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _project(root)
            document = freeze(root, _future(15), version="2.0.0")
            path = write_preregistration(document, root / "research/preregistration.yaml")
            reloaded = load_preregistration(path)
            self.assertEqual(reloaded["specification"], document["specification"])
            self.assertEqual(reloaded["confirmatory_start"], document["confirmatory_start"])

    def test_rejects_a_seal_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incompleto.yaml"
            path.write_text("frozen_at: '2026-01-01'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_preregistration(path)

    def test_rejects_a_seal_whose_start_precedes_the_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invertido.yaml"
            path.write_text(
                "frozen_at: '2026-06-01'\n"
                "confirmatory_start: '2026-01-01'\n"
                "specification: {}\n"
                "implementation: {}\n"
                "primary_contrast: {metric: mean_crps}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_preregistration(path)

    def test_summary_names_the_state(self):
        audit = {
            "specification_intact": False,
            "specification_drift": ["config.yaml"],
            "implementation_drift": [],
        }
        readiness = {"windows_available": 0, "windows_required": 40}
        text = summarise(audit, readiness)
        self.assertIn("ALTERADA", text)
        self.assertIn("0/40", text)


if __name__ == "__main__":
    unittest.main()
