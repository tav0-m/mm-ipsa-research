"""Congelamiento verificable para el test confirmatorio.

El protocolo declara desde la version 0.5.0 que ningun resultado puede llamarse
confirmatorio porque el periodo de evaluacion ya fue observado, y que un test
sellado requiere congelar antes version, universo, hipotesis, score primario e
hiperparametros. Esa declaracion nunca se hizo efectiva.

Este modulo la vuelve exigible. Cualquiera puede prometer que no reajustara nada
al llegar los datos nuevos; lo que distingue un compromiso de una promesa es que
el codigo detecte el incumplimiento.

El sello separa dos clases de contenido:

``specification``
    Configuracion, protocolo y cortes temporales. Son las decisiones cientificas
    del experimento. Si cambian, el sello queda invalidado y el test deja de ser
    confirmatorio.

``implementation``
    El codigo fuente. Puede cambiar -una reescritura numericamente equivalente no
    altera el experimento- pero cada modificacion queda registrada, de modo que
    la diferencia entre optimizar y reespecificar sea auditable.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SPECIFICATION_FILES = (
    "config.yaml",
    "research/PROTOCOL.md",
    "research/rolling_origin.yaml",
    "research/liquidity_robustness.yaml",
)

# Modulos que determinan como se generan y puntuan los escenarios. Un cambio
# aqui no invalida el sello, pero debe quedar declarado.
IMPLEMENTATION_GLOBS = (
    "src/mm_ipsa/mm/*.py",
    "src/mm_ipsa/models/*.py",
    "src/mm_ipsa/evaluation/*.py",
    "src/mm_ipsa/portfolio/*.py",
)


def _digest(path: Path) -> str:
    """SHA-256 de un archivo, leido por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    """Hashes de los archivos que coinciden, en orden estable."""
    collected: dict[str, str] = {}
    for pattern in patterns:
        if "*" in pattern:
            matches = sorted(root.glob(pattern))
        else:
            candidate = root / pattern
            matches = [candidate] if candidate.is_file() else []
        for path in matches:
            collected[path.relative_to(root).as_posix()] = _digest(path)
    return dict(sorted(collected.items()))


def freeze(
    root: str | Path,
    confirmatory_start: str,
    *,
    version: str,
    minimum_windows: int = 40,
    primary_metric: str = "mean_crps",
    primary_benchmark: str = "dcc_garch",
    notes: str = "",
) -> dict[str, Any]:
    """Genera el sello con la especificacion vigente y la fecha confirmatoria.

    ``confirmatory_start`` debe ser posterior a la fecha de congelamiento: un
    periodo que ya transcurrio no puede sellarse, porque nada garantiza que no
    haya sido observado.
    """
    base = Path(root)
    frozen_at = datetime.now(timezone.utc).date()
    start = date.fromisoformat(confirmatory_start)
    if start <= frozen_at:
        raise ValueError(
            "confirmatory_start debe ser posterior al congelamiento; "
            f"recibido {start} con sello {frozen_at}"
        )
    if minimum_windows < 2:
        raise ValueError("minimum_windows debe ser al menos dos")

    return {
        "preregistration_id": "confirmatory_v1",
        "frozen_at": frozen_at.isoformat(),
        "frozen_version": version,
        "confirmatory_start": start.isoformat(),
        "minimum_evaluation_windows": int(minimum_windows),
        "primary_contrast": {
            "metric": primary_metric,
            "focal_model": "MM",
            "benchmark_model": primary_benchmark,
            "alternative": "two_sided",
        },
        "decision_rule": {
            "evidence_requires": "pvalue_holm < 0.05 y el IC95 excluye cero",
            "multiple_testing": "holm sobre todos los contrastes declarados",
            "block_length": "politis_white automatico dentro de folds",
            "independent_route": "diebold_mariano con varianza HAC debe coincidir",
        },
        "prohibited_after_freeze": [
            "reestimar o ajustar cualquier hiperparametro con datos confirmatorios",
            "agregar, quitar o sustituir modelos",
            "cambiar el score primario o el contraste registrado",
            "modificar el universo de activos",
            "extender el periodo confirmatorio despues de mirar los resultados",
        ],
        "specification": _collect(base, SPECIFICATION_FILES),
        "implementation": _collect(base, IMPLEMENTATION_GLOBS),
        "notes": notes,
    }


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Carga el sello y valida que contenga los campos exigidos."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("El preregistro debe ser un mapeo")
    required = {
        "frozen_at",
        "confirmatory_start",
        "specification",
        "implementation",
        "primary_contrast",
    }
    missing = sorted(required.difference(document))
    if missing:
        raise ValueError(f"El preregistro no declara: {missing}")
    if date.fromisoformat(document["confirmatory_start"]) <= date.fromisoformat(
        document["frozen_at"]
    ):
        raise ValueError("confirmatory_start debe ser posterior a frozen_at")
    return document


def audit_preregistration(
    preregistration: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    """Compara el estado actual contra el sello y reporta cualquier desvio.

    Un cambio en la especificacion invalida el caracter confirmatorio del test.
    Un cambio en la implementacion no lo invalida, pero se enumera para que la
    distincion entre optimizar y reespecificar quede a la vista.
    """
    base = Path(root)
    current_specification = _collect(base, SPECIFICATION_FILES)
    current_implementation = _collect(base, IMPLEMENTATION_GLOBS)

    def differences(frozen: dict[str, str], current: dict[str, str]) -> list[str]:
        names = sorted(set(frozen) | set(current))
        return [
            name for name in names if frozen.get(name) != current.get(name)
        ]

    specification_drift = differences(
        preregistration["specification"], current_specification
    )
    implementation_drift = differences(
        preregistration["implementation"], current_implementation
    )
    return {
        "specification_intact": not specification_drift,
        "specification_drift": specification_drift,
        "implementation_drift": implementation_drift,
        "frozen_at": preregistration["frozen_at"],
        "confirmatory_start": preregistration["confirmatory_start"],
    }


def confirmatory_readiness(
    preregistration: dict[str, Any],
    observations: pd.DatetimeIndex | None,
    horizon: int,
    today: date | None = None,
) -> dict[str, Any]:
    """Cuenta cuantas ventanas confirmatorias existen y si alcanzan el minimo.

    Solo cuentan las observaciones posteriores al inicio declarado. Mientras el
    conteo no alcance el minimo preregistrado, cualquier lectura del periodo es
    exploratoria y debe presentarse como tal.
    """
    start = date.fromisoformat(preregistration["confirmatory_start"])
    current = today or datetime.now(timezone.utc).date()
    minimum = int(preregistration["minimum_evaluation_windows"])

    if observations is None or len(observations) == 0:
        available = 0
        latest = None
    else:
        index = pd.DatetimeIndex(observations)
        eligible = index[index >= pd.Timestamp(start)]
        available = len(eligible) // max(horizon, 1)
        latest = str(eligible.max().date()) if len(eligible) else None

    return {
        "confirmatory_start": start.isoformat(),
        "days_since_start": max((current - start).days, 0),
        "windows_available": int(available),
        "windows_required": minimum,
        "latest_observation": latest,
        "ready": bool(available >= minimum),
        "status": (
            "confirmatory_ready"
            if available >= minimum
            else "awaiting_confirmatory_sample"
        ),
    }


def write_preregistration(document: dict[str, Any], path: str | Path) -> Path:
    """Escribe el sello en YAML legible y estable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def summarise(audit: dict[str, Any], readiness: dict[str, Any]) -> str:
    """Resumen de una linea para la salida de verificacion."""
    state = "intacta" if audit["specification_intact"] else "ALTERADA"
    return (
        f"especificacion {state}; implementacion con "
        f"{len(audit['implementation_drift'])} archivo(s) modificado(s); "
        f"{readiness['windows_available']}/{readiness['windows_required']} "
        f"ventanas confirmatorias"
    )


def preregistration_report(root: str | Path, path: str | Path) -> dict[str, Any]:
    """Carga, audita y evalua disponibilidad en una sola llamada."""
    document = load_preregistration(path)
    audit = audit_preregistration(document, root)
    return {"preregistration": document, "audit": audit}
