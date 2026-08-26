"""Punto de entrada único para el pipeline, verificación y activos públicos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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



def _freeze(argv: Sequence[str]) -> int:
    """Sella la especificacion vigente y fija el inicio del test confirmatorio."""
    parser = argparse.ArgumentParser(
        description="Congela la especificacion para una evaluacion confirmatoria"
    )
    parser.add_argument("--start", required=True, help="Inicio confirmatorio YYYY-MM-DD")
    parser.add_argument("--min-windows", type=int, default=40)
    parser.add_argument("--output", default="research/preregistration.yaml")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescribe un sello existente; invalida el compromiso anterior",
    )
    args = parser.parse_args(argv)

    from mm_ipsa.analysis.preregistration import freeze, write_preregistration

    target = Path(args.output)
    if target.exists() and not args.force:
        print(
            f"Ya existe un sello en {target}. Sobrescribirlo anula el compromiso "
            "previo; use --force solo si esa es la intencion.",
            file=sys.stderr,
        )
        return 1

    document = freeze(
        Path.cwd(),
        args.start,
        version=__version__,
        minimum_windows=args.min_windows,
        notes=args.notes,
    )
    write_preregistration(document, target)
    print(f"Sello escrito en {target}")
    print(f"  congelado          {document['frozen_at']} (v{document['frozen_version']})")
    print(f"  confirmatorio      desde {document['confirmatory_start']}")
    print(f"  ventanas minimas   {document['minimum_evaluation_windows']}")
    print(f"  archivos sellados  {len(document['specification'])} de especificacion, "
          f"{len(document['implementation'])} de implementacion")
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
    if command == "freeze":
        return _freeze(command_arguments)
    print(f"Comando desconocido: {command}", file=sys.stderr)
    _print_help()
    return 2
