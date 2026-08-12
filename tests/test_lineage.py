import tempfile
import unittest
from pathlib import Path

from mm_ipsa.lineage import validate_lineage, write_lineage


class TestLineage(unittest.TestCase):
    def test_detects_stale_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            result = root / "result.txt"
            manifest = root / "lineage" / "stage.json"
            source.write_text("input-v1", encoding="utf-8")
            result.write_text("output-v1", encoding="utf-8")
            write_lineage(manifest, "test", [source], [result], root=root)

            self.assertTrue(validate_lineage(manifest, root=root)["valid"])
            source.write_text("input-v2", encoding="utf-8")
            stale_input = validate_lineage(manifest, root=root)
            self.assertFalse(stale_input["valid"])
            self.assertTrue(any("source.txt" in error for error in stale_input["errors"]))

            source.write_text("input-v1", encoding="utf-8")
            result.write_text("output-v2", encoding="utf-8")
            stale_output = validate_lineage(manifest, root=root)
            self.assertFalse(stale_output["valid"])
            self.assertTrue(any("result.txt" in error for error in stale_output["errors"]))


if __name__ == "__main__":
    unittest.main()
