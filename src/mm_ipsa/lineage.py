"""Linaje verificable de artefactos mediante SHA-256.

Cada etapa registra las entradas exactas —incluido el código relevante— y sus
salidas. Una etapa posterior puede negarse a consumir un manifiesto obsoleto.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypedDict

SCHEMA_VERSION = 1


class LineageValidation(TypedDict):
    valid: bool
    stage: str | None
    errors: list[str]


def sha256_file(path: str | Path) -> str:
    """Digest SHA-256 leido por bloques para artefactos de cualquier tamano."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: str | Path, root: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artefacto de linaje inexistente: {resolved}")
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"El artefacto queda fuera del proyecto: {resolved}") from exc
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _unique_paths(paths: Iterable[str | Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = Path(path).resolve()
        unique[str(resolved).casefold()] = resolved
    return [unique[key] for key in sorted(unique)]


def write_lineage(
    manifest_path: str | Path,
    stage: str,
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    *,
    root: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Escribe atómicamente el manifiesto de una etapa ya completada."""
    root_path = Path(root).resolve()
    manifest = Path(manifest_path)
    input_records = [_record(path, root_path) for path in _unique_paths(inputs)]
    output_records = [_record(path, root_path) for path in _unique_paths(outputs)]
    if not input_records or not output_records:
        raise ValueError("El linaje requiere al menos una entrada y una salida")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_records,
        "outputs": output_records,
        "metadata": metadata or {},
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def validate_lineage(manifest_path: str | Path, *, root: str | Path) -> LineageValidation:
    """Valida esquema, existencia, tamaño y hash de entradas y salidas."""
    manifest = Path(manifest_path)
    errors: list[str] = []
    if not manifest.is_file():
        return {"valid": False, "stage": None, "errors": [f"missing_manifest:{manifest}"]}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "stage": None, "errors": [f"invalid_json:{exc}"]}
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema")
    root_path = Path(root).resolve()
    for group in ("inputs", "outputs"):
        records = payload.get(group)
        if not isinstance(records, list) or not records:
            errors.append(f"missing_{group}")
            continue
        for record in records:
            relative = record.get("path") if isinstance(record, dict) else None
            if not isinstance(relative, str):
                errors.append(f"invalid_record:{group}")
                continue
            path = (root_path / relative).resolve()
            try:
                path.relative_to(root_path)
            except ValueError:
                errors.append(f"outside_root:{relative}")
                continue
            if not path.is_file():
                errors.append(f"missing:{relative}")
                continue
            if path.stat().st_size != record.get("bytes"):
                errors.append(f"size_mismatch:{relative}")
                continue
            if sha256_file(path) != record.get("sha256"):
                errors.append(f"hash_mismatch:{relative}")
    stage = payload.get("stage")
    return {
        "valid": not errors,
        "stage": stage if isinstance(stage, str) else None,
        "errors": errors,
    }


def assert_lineage_current(manifest_path: str | Path, *, root: str | Path) -> None:
    """Falla si el manifiesto no refleja el estado actual de entradas y salidas."""
    result = validate_lineage(manifest_path, root=root)
    if not result["valid"]:
        details = ", ".join(result["errors"])
        raise RuntimeError(f"Linaje obsoleto o inválido en {manifest_path}: {details}")
