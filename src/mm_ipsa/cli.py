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
        "  mm-ipsa audit-data [--prices PATH] [--output DIR]\n"
        "  mm-ipsa shortfall [--source DIR] [--output DIR]\n"
        "  mm-ipsa freeze --start YYYY-MM-DD [--min-windows N]\n"
        "  mm-ipsa --version\n"
    )


def _assets(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Regenera gráficos y tablas de la release")
    parser.add_argument(
        "--rolling-dir",
        default="outputs/robustness/rolling_origin",
    )
    parser.add_argument("--destination", default="research/assets")
    args = parser.parse_args(argv)
    from mm_ipsa.analysis.release_assets import generate_release_assets

    for artifact in generate_release_assets(args.rolling_dir, args.destination):
        print(artifact)
    return 0



def _audit_data(argv: Sequence[str]) -> int:
    """Audita la integridad estructural de la serie de precios."""
    parser = argparse.ArgumentParser(
        description="Comprueba calendario, eventos corporativos y precios estancados"
    )
    parser.add_argument("--prices", default="outputs/adj_close_prices.csv")
    parser.add_argument("--extreme-threshold", type=float, default=8.0)
    parser.add_argument("--scale-tolerance", type=float, default=3.0)
    parser.add_argument("--output", default=None, help="directorio para los CSV")
    args = parser.parse_args(argv)

    import pandas as pd

    from mm_ipsa.data.integrity import audit_prices

    source = Path(args.prices)
    if not source.is_file():
        print(f"No existe el archivo de precios {source}", file=sys.stderr)
        return 1

    prices = pd.read_csv(source, index_col=0, parse_dates=True)
    report = audit_prices(
        prices,
        extreme_threshold=args.extreme_threshold,
        split_tolerance=args.scale_tolerance,
    )

    calendar = report["calendar"]
    print(f"Auditoria de integridad sobre {source}")
    print(f"  filas              {calendar['rows']}")
    print(f"  fechas duplicadas  {calendar['duplicated_dates']}")
    print(f"  filas en fin de semana {calendar['weekend_rows']}")
    print(f"  hueco maximo       {calendar['largest_gap_days']} dias")

    candidates = report["corporate_action_candidates"]
    suspected = candidates.loc[candidates["suspected"]]
    print(
        f"  eventos corporativos {len(candidates)} candidatos, "
        f"{len(suspected)} sospechosos"
    )
    for row in suspected.itertuples():
        print(
            f"    {row.asset} {row.date} ratio={row.ratio:.4f} "
            f"~{row.nearest_split_ratio:.4f} persistencia={row.level_persistence:.2f}"
        )

    print(f"  retornos extremos  {len(report['extreme_returns'])}")

    stale = report["stale_prices"].sort_values("unchanged_share", ascending=False)
    print("  precios sin variacion, tres activos mas afectados:")
    for row in stale.head(3).itertuples():
        print(
            f"    {row.asset:12s} {row.unchanged_share:6.2%} "
            f"racha maxima {row.longest_unchanged_run}"
        )

    if args.output:
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(destination / "integrity_corporate_actions.csv", index=False)
        report["extreme_returns"].to_csv(
            destination / "integrity_extreme_returns.csv", index=False
        )
        report["stale_prices"].to_csv(
            destination / "integrity_stale_prices.csv", index=False
        )
        print(f"  informes escritos en {destination}")

    if report["passed"]:
        print("Integridad estructural sin hallazgos bloqueantes")
        return 0
    for finding in report["blocking"]:
        print(f"BLOQUEANTE: {finding}", file=sys.stderr)
    return 1


def _shortfall(argv: Sequence[str]) -> int:
    """Contrasta la calibracion de Expected Shortfall de cada generador."""
    parser = argparse.ArgumentParser(
        description="Backtest de Expected Shortfall segun Acerbi y Szekely"
    )
    parser.add_argument("--source", default="outputs")
    parser.add_argument("--observations", default="outputs/terminal_returns_H5_OOS.csv")
    parser.add_argument("--daily", default="outputs/daily_returns_OOS.csv")
    parser.add_argument("--output", default="outputs/risk")
    parser.add_argument("--simulations", type=int, default=4_000)
    args = parser.parse_args(argv)

    import pandas as pd

    from mm_ipsa.analysis.shortfall_report import run_shortfall_report
    from mm_ipsa.config import load_config

    observations_path = Path(args.observations)
    if not observations_path.is_file():
        print(f"No existe {observations_path}", file=sys.stderr)
        return 1

    observations = pd.read_csv(observations_path, index_col=0, parse_dates=True)
    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    report = run_shortfall_report(
        load_config(),
        observations,
        args.source,
        args.output,
        daily,
        n_simulations=args.simulations,
    )

    summary = report["summary"]
    print(f"Calibracion de cola sobre {len(observations)} observaciones solapadas")
    print(
        f"  {'modelo':12s} {'ES predicho':>12s} {'ES realizado':>13s} "
        f"{'severidad':>10s} {'Holm<0.05':>10s}"
    )
    for record in summary.to_dict("records"):
        print(
            f"  {str(record['model']):12s} {float(record['predicted_es']):12.4f} "
            f"{float(record['realised_tail_mean']):13.4f} "
            f"{float(record['severity_ratio']):10.3f} "
            f"{int(record['assets_underestimating']):6d}/"
            f"{int(record['assets'])}"
        )
    portfolios = report["by_portfolio"]
    print()
    print("Por cartera, que es la unidad que valida un regulador:")
    print(
        f"  {'modelo':11s} {'cartera':12s} {'severidad':>10s} {'Holm':>8s} "
        f"{'exc':>4s} {'zona':>9s}"
    )
    for record in portfolios.to_dict("records"):
        print(
            f"  {str(record['model']):11s} {str(record['strategy']):12s} "
            f"{float(record['severity_ratio']):10.3f} "
            f"{float(record['pvalue_z2_holm']):8.4f} "
            f"{int(record['basel_exceptions']):4d} "
            f"{str(record['basel_zone']):>9s}"
        )
    if not bool(portfolios["basel_sample_adequate"].iloc[0]):
        expected = float(portfolios["basel_expected"].iloc[0])
        print(
            f"  Semaforo indicativo: solo {expected:.2f} excepciones esperadas "
            "por cartera."
        )

    attribution = report["attribution"].sort_values(
        "conditional_rank_correlation", ascending=False
    )
    print()
    print("Acierto sobre la composicion de la cola, sin el peso de la cartera:")
    print(f"  {'modelo':11s} {'cartera':12s} {'rho':>7s} {'rango':>7s}")
    for record in attribution.to_dict("records"):
        print(
            f"  {str(record['model']):11s} {str(record['strategy']):12s} "
            f"{float(record['conditional_rank_correlation']):+7.3f} "
            f"{float(record['correlation_spread']):7.2f}"
        )
    print("  rho cerca de cero: el modelo no informa que posiciones mueven la cola.")

    cycle = report["procyclicality"]
    episodes = int(cycle["stress_episodes"].iloc[0])
    print()
    print("Concentracion de excesos por regimen de volatilidad:")
    print(f"  {'modelo':11s} {'cartera':12s} {'calma':>7s} {'tension':>8s} {'Holm':>8s}")
    for record in cycle.to_dict("records"):
        print(
            f"  {str(record['model']):11s} {str(record['strategy']):12s} "
            f"{float(record['calm_breach_rate']):7.1%} "
            f"{float(record['stressed_breach_rate']):8.1%} "
            f"{float(record['pvalue_holm']):8.4f}"
        )
    print(
        f"  El regimen tensionado son {episodes} episodio(s): la evidencia "
        "descansa en un solo cambio de regimen."
    )

    print(f"\n  informes en {Path(args.output)}")
    print("Severidad por encima de uno indica perdidas de cola peores que el ES.")
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
    if command == "audit-data":
        return _audit_data(command_arguments)
    if command == "shortfall":
        return _shortfall(command_arguments)
    print(f"Comando desconocido: {command}", file=sys.stderr)
    _print_help()
    return 2
