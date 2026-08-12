"""Ingestion de precios con evidencia de cobertura previa a imputacion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from mm_ipsa.config import load_config
from mm_ipsa.lineage import sha256_file


def _atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _max_missing_run(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    best = current = 0
    for value in missing:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("La respuesta del proveedor no contiene el campo Close")
        close = raw["Close"]
        if not isinstance(close, pd.DataFrame):
            raise RuntimeError("El campo Close no contiene un panel por ticker")
        prices = close.copy()
    else:
        prices = raw.copy()
    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_tickers:
        raise RuntimeError(f"Tickers ausentes en la descarga: {missing_tickers}")
    result = prices.loc[:, tickers]
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("La seleccion de precios no produjo un panel tabular")
    return result


def download_prices(cfg: dict) -> pd.DataFrame:
    """Descarga, audita e imputa de forma acotada los precios ajustados."""
    tickers = list(cfg["assets"])
    labels = list(cfg["asset_labels"])
    data_cfg = cfg["data"]
    out_path = Path(cfg["paths"]["data_raw"])
    out_path.mkdir(parents=True, exist_ok=True)
    cache_path = out_path / ".yfinance_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_path))

    print("\n[download] precios Yahoo Finance")
    print(f"  periodo={data_cfg['start_train']} -> {data_cfg['end_oos']} (fin exclusivo)")
    raw = yf.download(
        tickers=tickers,
        start=data_cfg["start_train"],
        end=data_cfg["end_oos"],
        auto_adjust=True,
        actions=False,
        repair=False,
        keepna=True,
        group_by="column",
        multi_level_index=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("Yahoo Finance no devolvio precios; no se reemplaza el panel canonico")

    downloaded_rows = len(raw)
    prices_raw = _extract_close(raw, tickers)
    prices_raw.columns = labels
    raw_dates = pd.DatetimeIndex(pd.to_datetime(prices_raw.index))
    prices_raw.index = raw_dates.tz_localize(None)
    duplicate_dates = int(prices_raw.index.duplicated(keep="last").sum())
    prices_raw = prices_raw[~prices_raw.index.duplicated(keep="last")].sort_index()
    prices_raw = prices_raw.replace([np.inf, -np.inf], np.nan)
    if (prices_raw <= 0).any().any():
        raise RuntimeError("Se detectaron precios no positivos")

    coverage_raw = prices_raw.notna().mean()
    min_coverage = float(data_cfg["min_coverage"])
    quality = pd.DataFrame(
        {
            "ticker": tickers,
            "asset": labels,
            "observations": prices_raw.notna().sum().to_numpy(),
            "missing_raw": prices_raw.isna().sum().to_numpy(),
            "coverage_raw": coverage_raw.to_numpy(),
            "max_missing_run": [_max_missing_run(prices_raw[c]) for c in labels],
            "first_observed": [
                str(prices_raw[c].first_valid_index().date()) if prices_raw[c].first_valid_index() is not None else None
                for c in labels
            ],
            "last_observed": [
                str(prices_raw[c].last_valid_index().date()) if prices_raw[c].last_valid_index() is not None else None
                for c in labels
            ],
        }
    )

    low = coverage_raw[coverage_raw < min_coverage]
    if not low.empty:
        detail = ", ".join(f"{asset}={value:.1%}" for asset, value in low.items())
        raise RuntimeError(
            f"Cobertura raw inferior a {min_coverage:.1%}: {detail}. "
            "Revise universo/periodo antes de imputar."
        )

    limit = int(data_cfg.get("ffill_limit_days", 0))
    prices_filled = prices_raw.ffill(limit=limit) if limit > 0 else prices_raw.copy()
    imputed_mask = prices_raw.isna() & prices_filled.notna()
    quality["imputed"] = imputed_mask.sum().to_numpy()
    quality["unresolved_after_fill"] = prices_filled.isna().sum().to_numpy()
    complete = prices_filled.dropna(how="any")
    if complete.empty:
        raise RuntimeError("No quedan filas completas despues de la imputacion acotada")

    raw_path = out_path / "adj_close_prices_raw.csv"
    mask_path = out_path / "price_observation_mask.csv"
    quality_path = out_path / "data_quality_prices.csv"
    canonical_path = out_path / "adj_close_prices.csv"
    # Escritura atómica por artefacto. El manifiesto de linaje se escribe solo
    # después de completar todo el paso, por lo que una interrupción se detecta.
    _atomic_csv(prices_raw, raw_path)
    _atomic_csv(prices_raw.notna().astype("int8"), mask_path)
    _atomic_csv(quality, quality_path, index=False)
    _atomic_csv(complete, canonical_path)
    try:
        yf_version = version("yfinance")
    except PackageNotFoundError:
        yf_version = "unknown"
    metadata = {
        "provider": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yf_version,
        "start_inclusive": data_cfg["start_train"],
        "end_exclusive": data_cfg["end_oos"],
        "auto_adjust": True,
        "actions": False,
        "repair": False,
        "keepna": True,
        "threads": False,
        "timeout_seconds": 30,
        "ffill_limit_days": limit,
        "requested_tickers": tickers,
        "asset_labels": labels,
        "downloaded_rows": int(downloaded_rows),
        "duplicate_dates_removed": duplicate_dates,
        "first_index": str(prices_raw.index.min().date()),
        "last_index": str(prices_raw.index.max().date()),
        "raw_rows": int(len(prices_raw)),
        "complete_rows": int(len(complete)),
        "dropped_incomplete_rows": int(len(prices_raw) - len(complete)),
        "raw_coverage": {key: float(value) for key, value in coverage_raw.items()},
        "sha256": {
            "adj_close_prices_raw.csv": sha256_file(raw_path),
            "price_observation_mask.csv": sha256_file(mask_path),
            "data_quality_prices.csv": sha256_file(quality_path),
            "adj_close_prices.csv": sha256_file(canonical_path),
        },
        "limitations": [
            "Yahoo Finance es una fuente exploratoria y no institucional.",
            "auto_adjust=True incorpora ajustes del proveedor sin un ledger corporativo local.",
        ],
    }
    _atomic_json(metadata, out_path / "data_download_metadata.json")
    print(
        f"  [ok] raw={len(prices_raw)}, panel_completo={len(complete)}, "
        f"cobertura_min_raw={coverage_raw.min():.1%}, imputados={int(imputed_mask.sum().sum())}"
    )
    return complete


if __name__ == "__main__":
    download_prices(load_config())
