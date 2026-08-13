"""Punto de entrada único para el pipeline, verificación y activos públicos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mm_ipsa import __version__


def _print_help() -> None:
    print(
        "MM-IPSA Research\n\n"
        "Uso:\n"
        "  mm-ipsa run --step STEP [--resume | --plan]\n"
        "  mm-ipsa verify [--scope core|full]\n"
        "  mm-ipsa assets [--rolling-dir PATH] [--destination PATH]\n"
        "  mm-ipsa --version\n"
    )


def _assets(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Regenera gráficos y tablas de la release")
    parser.add_argument(
        "--rolling-dir",
        default="outputs/robustness/rolling_origin",
    )
    parser.add_argument("--destination", default="docs/assets")
    args = parser.parse_args(argv)
    from mm_ipsa.analysis.release_assets import generate_release_assets

    for artifact in generate_release_assets(args.rolling_dir, args.destination):
        print(artifact)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Despacha los subcomandos run y verify de la interfaz mm-ipsa."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(f"mm-ipsa {__version__}")
        return 0

    command, command_arguments = arguments[0], arguments[1:]
    if command == "run":
        from mm_ipsa.pipeline import main as pipeline_main

        return pipeline_main(command_arguments)
    if command == "verify":
        from mm_ipsa.verification import main as verification_main

        return verification_main(command_arguments)
    if command == "assets":
        return _assets(command_arguments)
    print(f"Comando desconocido: {command}", file=sys.stderr)
    _print_help()
    return 2
