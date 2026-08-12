"""Transformacion de precios a retornos con separacion temporal estricta."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_prices(prices: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    missing = [label for label in labels if label not in prices.columns]
    if missing:
        raise ValueError(f"Faltan columnas de precios: {missing}")
    panel = prices[labels].copy()
    panel_dates = pd.DatetimeIndex(pd.to_datetime(panel.index))
    panel.index = panel_dates.tz_localize(None)
    panel = panel.sort_index()
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("El indice de precios debe ser unico y creciente")
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("El panel canonico de precios debe ser completo y finito")
    if (panel <= 0).any().any():
        raise ValueError("Los precios deben ser estrictamente positivos")
    return panel


def _rolling_terminal_fast(df: pd.DataFrame, H: int) -> pd.DataFrame:
    """Retorno simple compuesto para todas las ventanas consecutivas H."""
    if H <= 0:
        raise ValueError("H debe ser positivo")
    if len(df) < H:
        return pd.DataFrame(columns=df.columns, dtype=float)
    values = df.to_numpy(dtype=float)
    if np.any(values <= -1):
        raise ValueError("Un retorno <= -100% impide la composicion logaritmica")
    T, n = values.shape
    log_returns = np.log1p(values)
    cumulative = np.vstack([np.zeros((1, n)), np.cumsum(log_returns, axis=0)])
    terminal = np.expm1(cumulative[H:] - cumulative[: T - H + 1])
    return pd.DataFrame(terminal, index=df.index[H - 1 :], columns=df.columns)


def non_overlapping_windows(terminal: pd.DataFrame, H: int) -> pd.DataFrame:
    """Selecciona ventanas H que no comparten retornos diarios."""
    if H <= 0:
        raise ValueError("H debe ser positivo")
    return terminal.iloc[::H].copy()


def _max_true_run(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool).to_numpy():
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _segment_metrics(returns: pd.DataFrame, suffix: str) -> dict[str, np.ndarray]:
    zero = returns.eq(0.0)
    return {
        f"zero_return_rate_{suffix}": zero.mean().to_numpy(),
        f"max_zero_run_{suffix}": np.asarray([_max_true_run(zero[column]) for column in returns.columns]),
        f"volatility_{suffix}": returns.std(ddof=0).to_numpy(),
    }


def build_returns(
    cfg: dict,
    prices: pd.DataFrame,
    observation_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Construye retornos IS/OOS y conserva muestras rolling/no solapadas."""
    labels = list(cfg["asset_labels"])
    data_cfg = cfg["data"]
    H = int(data_cfg["H"])
    out_path = Path(cfg["paths"]["data_raw"])
    out_path.mkdir(parents=True, exist_ok=True)
    panel = _validate_prices(prices, labels)

    daily_unfiltered = (panel / panel.shift(1) - 1.0).iloc[1:]
    invalid_endpoints = pd.DataFrame(False, index=daily_unfiltered.index, columns=labels)
    imputed_price_count = np.full(len(labels), np.nan)
    mask_status = "unavailable"
    if observation_mask is not None:
        observed = observation_mask.copy()
        observed_dates = pd.DatetimeIndex(pd.to_datetime(observed.index))
        observed.index = observed_dates.tz_localize(None)
        observed.rename(columns={column: column.replace(".SN", "") for column in observed.columns}, inplace=True)
        missing_mask = [label for label in labels if label not in observed.columns]
        if missing_mask:
            raise ValueError(f"La mascara raw no contiene activos: {missing_mask}")
        observed = observed.reindex(index=panel.index, columns=labels)
        if observed.isna().any().any():
            raise ValueError("La mascara raw no cubre exactamente el panel canonico")
        observed = observed.astype(bool)
        imputed_price_count = (~observed).sum().to_numpy()
        valid_endpoints = observed & observed.shift(1, fill_value=False)
        invalid_endpoints = ~valid_endpoints.loc[daily_unfiltered.index]
        mask_status = "available"

    exclude_imputed = bool(data_cfg.get("exclude_imputed_return_endpoints", True))
    daily = daily_unfiltered.copy()
    dropped_endpoint_rows = 0
    if exclude_imputed and observation_mask is not None:
        daily = daily.mask(invalid_endpoints)
        before = len(daily)
        daily = daily.dropna(how="any")
        dropped_endpoint_rows = before - len(daily)
    if daily.isna().any().any() or not np.isfinite(daily.to_numpy()).all():
        raise RuntimeError("La transformacion produjo retornos faltantes o infinitos")

    end_train = pd.Timestamp(data_cfg["end_train"])
    start_oos = pd.Timestamp(data_cfg["start_oos"])
    if end_train >= start_oos:
        raise ValueError("end_train debe ser anterior a start_oos")
    daily_train = daily.loc[daily.index <= end_train].copy()
    daily_oos = daily.loc[daily.index >= start_oos].copy()
    if daily_train.empty or daily_oos.empty:
        raise RuntimeError("El split temporal genero una muestra IS u OOS vacia")
    if daily_train.index.max() >= daily_oos.index.min():
        raise RuntimeError("Se detecto solapamiento temporal IS/OOS")

    terminal_train = _rolling_terminal_fast(daily_train, H)
    terminal_oos = _rolling_terminal_fast(daily_oos, H)
    terminal_train_nonoverlap = non_overlapping_windows(terminal_train, H)
    terminal_oos_nonoverlap = non_overlapping_windows(terminal_oos, H)

    zero_rates = (daily == 0.0).mean()
    threshold = float(data_cfg.get("max_zero_return_rate", 1.0))
    quality_values: dict[str, object] = {
        "asset": labels,
        "zero_return_rate": zero_rates.to_numpy(),
        "daily_min": daily.min().to_numpy(),
        "daily_max": daily.max().to_numpy(),
        "daily_volatility": daily.std(ddof=0).to_numpy(),
        "unique_price_ratio": panel.nunique().to_numpy() / len(panel),
        "imputed_price_count": imputed_price_count,
        "invalid_return_endpoint_count": invalid_endpoints.sum().to_numpy() if observation_mask is not None else np.nan,
    }
    quality_values.update(_segment_metrics(daily, "full"))
    quality_values.update(_segment_metrics(daily_train, "is"))
    quality_values.update(_segment_metrics(daily_oos, "oos"))
    quality = pd.DataFrame(quality_values)
    max_zero_run_limit = int(data_cfg.get("max_consecutive_zero_returns", len(daily)))
    quality["zero_rate_flag"] = quality["zero_return_rate_full"] > threshold
    quality["zero_run_flag"] = quality["max_zero_run_full"] > max_zero_run_limit
    quality.to_csv(out_path / "data_quality_returns.csv", index=False)
    high_zero = zero_rates[zero_rates > threshold]
    if not high_zero.empty:
        detail = ", ".join(f"{key}={value:.1%}" for key, value in high_zero.items())
        print(f"  [warn] alta tasa de retornos cero (iliquidez o precios stale): {detail}")
    long_zero = quality.loc[quality["zero_run_flag"], ["asset", "max_zero_run_full"]]
    if not long_zero.empty:
        detail = ", ".join(
            f"{asset}={int(value)}"
            for asset, value in zip(long_zero["asset"], long_zero["max_zero_run_full"])
        )
        print(f"  [warn] rachas largas de retornos cero: {detail}")

    artifacts = {
        "daily_returns.csv": daily_train,
        f"terminal_returns_H{H}.csv": terminal_train,
        f"terminal_returns_H{H}_nonoverlap.csv": terminal_train_nonoverlap,
        "daily_returns_OOS.csv": daily_oos,
        f"terminal_returns_H{H}_OOS.csv": terminal_oos,
        f"terminal_returns_H{H}_OOS_nonoverlap.csv": terminal_oos_nonoverlap,
        # Alias historico consumido por los scripts existentes.
        f"hist_terminal_returns_H{H}.csv": terminal_train,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(out_path / filename)

    metadata = {
        "H": H,
        "return_definition": "simple_compounded",
        "pct_change_fill_method": None,
        "observation_mask_status": mask_status,
        "exclude_imputed_return_endpoints": exclude_imputed,
        "rows_dropped_for_imputed_endpoints": dropped_endpoint_rows,
        "staleness_policy": "warn_only_no_automatic_asset_exclusion",
        "end_train_inclusive": str(end_train.date()),
        "start_oos_inclusive": str(start_oos.date()),
        "daily_train_rows": len(daily_train),
        "daily_oos_rows": len(daily_oos),
        "terminal_train_rolling_rows": len(terminal_train),
        "terminal_oos_rolling_rows": len(terminal_oos),
        "terminal_train_nonoverlap_rows": len(terminal_train_nonoverlap),
        "terminal_oos_nonoverlap_rows": len(terminal_oos_nonoverlap),
    }
    (out_path / "returns_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\n[transform] retornos")
    print(
        f"  IS diario={len(daily_train)}, OOS diario={len(daily_oos)}, "
        f"IS H{H} rolling={len(terminal_train)}, OOS no-solapado={len(terminal_oos_nonoverlap)}"
    )
    if observation_mask is not None:
        print(f"  endpoints_imputados: filas_panel_eliminadas={dropped_endpoint_rows}")
    return daily_train, terminal_train, daily_oos, terminal_oos


if __name__ == "__main__":
    from mm_ipsa.config import load_config

    config = load_config()
    path = Path(config["paths"]["data_raw"]) / "adj_close_prices.csv"
    build_returns(config, pd.read_csv(path, index_col=0, parse_dates=True))
