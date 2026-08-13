import hashlib
import tempfile
import unittest
from pathlib import Path

from mm_ipsa.verification import Verifier, sha256


class TestSha256(unittest.TestCase):
    def test_matches_hashlib_for_small_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            content = b"contenido reproducible\n"
            path.write_bytes(content)
            self.assertEqual(sha256(path), hashlib.sha256(content).hexdigest())

    def test_streaming_matches_single_pass_for_multichunk_file(self):
        # El lector avanza en bloques de 1 MiB; un archivo mayor ejercita la
        # acumulacion incremental, donde un digest mal encadenado se notaria.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grande.bin"
            content = bytes(range(256)) * 12_000
            path.write_bytes(content)
            self.assertEqual(sha256(path), hashlib.sha256(content).hexdigest())

    def test_distinct_content_yields_distinct_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.txt"
            second = Path(directory) / "b.txt"
            first.write_bytes(b"alfa")
            second.write_bytes(b"beta")
            self.assertNotEqual(sha256(first), sha256(second))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                sha256(Path(directory) / "ausente.bin")


class TestVerifier(unittest.TestCase):
    def test_starts_without_failures(self):
        self.assertEqual(Verifier().failures, [])

    def test_passing_check_records_nothing(self):
        verifier = Verifier()
        verifier.check(True, "contrato")
        self.assertEqual(verifier.failures, [])

    def test_failing_check_is_recorded_with_detail(self):
        verifier = Verifier()
        verifier.check(False, "cobertura", "0.80 < 0.95")
        self.assertEqual(verifier.failures, ["cobertura: 0.80 < 0.95"])

    def test_failures_accumulate_in_order(self):
        verifier = Verifier()
        verifier.check(False, "primero")
        verifier.check(True, "segundo")
        verifier.check(False, "tercero", "detalle")
        self.assertEqual(verifier.failures, ["primero", "tercero: detalle"])

    def test_truthy_values_are_accepted_as_conditions(self):
        verifier = Verifier()
        verifier.check(bool([1]), "lista no vacia")
        verifier.check(bool([]), "lista vacia")
        self.assertEqual(verifier.failures, ["lista vacia"])


if __name__ == "__main__":
    unittest.main()
