"""Episode 1: moving-average crossover (golden cross / death cross). Teaching example only.

Rules (see specs/ep01_ma_cross.md):
  golden cross: yesterday fast <= slow and today fast > slow -> target 1 (buy next open)
  death cross : yesterday fast >= slow and today fast < slow -> target 0 (sell next open)
  no signal until both averages exist; if fast starts above slow, wait for the next cross.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def target(prices: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    c = prices["close"]
    f = c.rolling(fast).mean()
    s = c.rolling(slow).mean()
    up = (f.shift(1) <= s.shift(1)) & (f > s)
    down = (f.shift(1) >= s.shift(1)) & (f < s)
    sig = pd.Series(np.nan, index=c.index)
    sig[up] = 1.0
    sig[down] = 0.0
    return sig.ffill().fillna(0.0).rename("target")


def first_signal_date(prices: pd.DataFrame, slow: int = 60) -> pd.Timestamp:
    """First day both averages exist (warm-up done). Benchmarks start here too."""
    return prices.index[slow - 1]
