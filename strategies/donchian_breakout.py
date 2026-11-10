"""Episode 5: simplified "Turtle" channel breakout. Teaching example only.

Rules (see specs/ep05_donchian_breakout.md):
  flat and close above the highest close of the previous `entry` days -> target 1
  long and close below the lowest close of the previous `exit` days  -> target 0
  ties do nothing; the channels exclude today.
  use_hl=True builds the channels from highs and lows instead of closes (still judged on the close).

The original Turtle rules also used volatility sizing, pyramiding, leverage, shorting and
many futures markets. None of that is here, so this says nothing about the original system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def channels(prices: pd.DataFrame, entry: int = 20, exit: int = 10, use_hl: bool = False) -> pd.DataFrame:
    hi = prices["high"] if use_hl else prices["close"]
    lo = prices["low"] if use_hl else prices["close"]
    return pd.DataFrame({"upper": hi.shift(1).rolling(entry).max(),
                         "lower": lo.shift(1).rolling(exit).min()}, index=prices.index)


def target(prices: pd.DataFrame, entry: int = 20, exit: int = 10, use_hl: bool = False) -> pd.Series:
    ch = channels(prices, entry, exit, use_hl)
    c, up, lo = prices["close"].values, ch["upper"].values, ch["lower"].values
    out = np.zeros(len(c))
    state = 0.0
    for i in range(len(c)):
        if state == 0.0:
            if not np.isnan(up[i]) and c[i] > up[i]:
                state = 1.0
        elif not np.isnan(lo[i]) and c[i] < lo[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=prices.index, name="target")


def first_signal_date(prices: pd.DataFrame, entry: int = 20) -> pd.Timestamp:
    return prices.index[entry]
