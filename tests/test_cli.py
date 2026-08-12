import contextlib
import io
import unittest

from mm_ipsa import __version__
from mm_ipsa.cli import main as cli_main
from mm_ipsa.pipeline import _execution_sequence


class TestCLI(unittest.TestCase):
    def test_version_is_the_public_release(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = cli_main(["--version"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stream.getvalue().strip(), f"mm-ipsa {__version__}")
        self.assertEqual(__version__, "0.5.0")

    def test_unknown_command_returns_usage_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = cli_main(["unknown"])
        self.assertEqual(exit_code, 2)
        self.assertIn("Comando desconocido", stderr.getvalue())

    def test_full_sequence_preserves_dependencies(self):
        sequence = _execution_sequence("all")
        self.assertEqual(sequence[0], "download")
        self.assertEqual(sequence[-1], "snapshot")
        self.assertLess(sequence.index("transform"), sequence.index("rolling"))
        self.assertLess(sequence.index("rolling"), sequence.index("snapshot"))

    def test_single_step_sequence_is_exact(self):
        self.assertEqual(_execution_sequence("evaluate"), ("evaluate",))


if __name__ == "__main__":
    unittest.main()
