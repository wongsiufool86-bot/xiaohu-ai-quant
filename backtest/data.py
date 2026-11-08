"""Data loading and validation.

No data ships with this repo. Use your own, either way:

    DATA_DIR=/path/to/csv_folder            # plain CSV files, see below
    PARQUET_EXPORT_DIR=/path/to/export      # a parquet export bundle with a symbol catalog

If both are set, DATA_DIR wins.

CSV layout:
    <DATA_DIR>/SPY.csv    date,open,high,low,close,volume[,unadjusted_close]
    <DATA_DIR>/rate.csv   date,rate_pct      optional; 4.17 means 4.17% per year

Prices must be adjusted for splits and dividends. Without rate.csv, cash earns 0%.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def _csv_dir() -> Path | None:
    root = os.environ.get("DATA_DIR")
    return Path(root) if root else None


def _root() -> Path:
    root = os.environ.get("PARQUET_EXPORT_DIR")
    if not root:
        raise RuntimeError("Set DATA_DIR to a folder of CSV files (see backtest/data.py), "
                           "or PARQUET_EXPORT_DIR to a parquet export.")
    return Path(root)


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip").set_index("date").sort_index()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return df


def _catalog(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "python" / "metadata" / "symbol_catalog.parquet")


def _path(symbol: str, database: str) -> Path:
    root = _root()
    cat = _catalog(root)
    row = cat[(cat.symbol == symbol) & (cat.database == database)]
    if row.empty:
        raise KeyError(f"{symbol} not found in {database}")
    return root / row.iloc[0]["relative_parquet_path"]


def load_prices(symbol: str, database: str = "US Equities") -> pd.DataFrame:
    """Daily bars. open/high/low/close are total-return adjusted (dividends reinvested)."""
    csv_dir = _csv_dir()
    if csv_dir:
        df = _read_csv(csv_dir / f"{symbol}.csv", PRICE_COLS)
    else:
        df = pd.read_parquet(_path(symbol, database))
        df = df.set_index("date").sort_index()
    df.index.name = "date"
    keep = PRICE_COLS + [c for c in ["unadjusted_close"] if c in df.columns]
    return df[keep].astype("float64")


def load_rate(symbol: str = "%3MTCM") -> pd.Series:
    """Annualised yield in percent, e.g. 4.17 means 4.17% per year."""
    csv_dir = _csv_dir()
    if csv_dir:
        path = csv_dir / "rate.csv"
        if not path.exists():
            warnings.warn("rate.csv not found: cash earns 0%, so results will differ.")
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([]), name="rate_pct")
        return _read_csv(path, ["rate_pct"])["rate_pct"].astype("float64")
    df = pd.read_parquet(_path(symbol, "Economic")).set_index("date").sort_index()
    return df["close"].astype("float64").rename("rate_pct")


def daily_rate(rate_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily simple rate aligned to trading days (last known value, no look-ahead)."""
    r = rate_pct.reindex(rate_pct.index.union(index)).ffill().reindex(index)
    return (r / 100.0 / 252.0).fillna(0.0).rename("rf_daily")


def validate(df: pd.DataFrame, big_move: float = 0.10) -> dict:
    """Data quality checks. Returns counts, plus a list of the dates behind each count."""
    out: dict = {}
    idx = df.index
    out["rows"] = int(len(df))
    out["first_date"] = str(idx.min().date())
    out["last_date"] = str(idx.max().date())
    out["duplicate_dates"] = int(idx.duplicated().sum())
    out["unsorted"] = bool(not idx.is_monotonic_increasing)
    out["missing_values"] = int(df[PRICE_COLS].isna().any(axis=1).sum())
    lo_bad = df["low"] > df[["open", "close"]].min(axis=1) * (1 + 1e-6)
    hi_bad = df["high"] < df[["open", "close"]].max(axis=1) * (1 - 1e-6)
    out["ohlc_inconsistent"] = int((lo_bad | hi_bad).sum())
    out["zero_volume"] = int((df["volume"] <= 0).sum())
    ret = df["close"].pct_change()
    big = ret[ret.abs() > big_move]
    out["big_moves"] = {str(d.date()): round(float(v), 4) for d, v in big.items()}
    gaps = idx.to_series().diff().dt.days
    out["gaps_over_4_calendar_days"] = {str(d.date()): int(g) for d, g in gaps[gaps > 4].items()}
    return out
