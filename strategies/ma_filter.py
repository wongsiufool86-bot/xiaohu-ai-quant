"""Episode 3: 200-day moving-average filter. Teaching example only.

Rules (see specs/ep03_ma200_filter.md):
  close above the average -> target 1; close below -> target 0; exactly equal -> keep the state.
  band: only act once the close is more than `band` away from the average.
  monthly: only look at the signal on the last trading day of each month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def target(prices: pd.DataFrame, window: int = 200, band: float = 0.0, monthly: bool = False) -> pd.Series:
    c = prices["close"]
    ma = c.rolling(window).mean()
    sig = pd.Series(np.nan, index=c.index)
    sig[c > ma * (1 + band)] = 1.0
    sig[c < ma * (1 - band)] = 0.0
    if monthly:
        idx = c.index.to_series()
        month_end = idx.dt.to_period("M") != idx.dt.to_period("M").shift(-1)
        sig[~month_end.values] = np.nan
    return sig.ffill().fillna(0.0).rename("target")


def first_signal_date(prices: pd.DataFrame, window: int = 200) -> pd.Timestamp:
    return prices.index[window - 1]
