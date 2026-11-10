"""Episode 6: a trailing stop on an index ETF. Teaching example only.

Rules (see specs/ep06_stop_loss.md):
  start fully invested; sell when the close falls `stop` below the highest close of this holding period;
  once out, buy back when the close is above the highest close of the previous `reentry` days.

Variants, all selected by keyword:
  basis="cost"        measure the stop from the entry price instead of the peak
  wait=N              ignore the re-entry rule: come back N trading days after selling
  never_back=True     the first stop is final; the money stays in cash
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def target(prices: pd.DataFrame, stop: float = 0.10, reentry: int = 20, basis: str = "peak",
           wait: int | None = None, never_back: bool = False) -> pd.Series:
    """Target weight decided after each close, traded at the next open."""
    c, o = prices["close"].to_numpy(float), prices["open"].to_numpy(float)
    prior_high = pd.Series(c).rolling(reentry).max().shift(1).to_numpy()
    out = np.zeros(len(c))
    out[0] = 1.0
    ref = stop_signal = None
    for t in range(1, len(c)):
        if out[t - 1] == 1.0:                                  # held through today
            just_entered = t == 1 or out[t - 2] == 0.0
            if basis == "cost":
                ref = o[t] if just_entered else ref
            else:
                ref = c[t] if just_entered else max(ref, c[t])
            if c[t] <= ref * (1.0 - stop):
                out[t], stop_signal = 0.0, t
            else:
                out[t] = 1.0
        elif never_back:
            out[t] = 0.0
        elif wait is not None:
            out[t] = 1.0 if t - stop_signal >= wait else 0.0
        else:
            out[t] = 1.0 if c[t] > prior_high[t] else 0.0     # NaN compares False: no signal yet
    return pd.Series(out, index=prices.index, name="target")


def ledger(prices: pd.DataFrame, weight: pd.Series, whipsaw_floor: float = -0.05) -> pd.DataFrame:
    """One row per stop: where we sold, where we came back, how far it fell in between."""
    w = weight.fillna(0.0)
    held = w.to_numpy()
    idx = w.index
    o, c = prices["open"].reindex(idx), prices["close"].reindex(idx)
    rows = []
    for i in range(1, len(idx)):
        if held[i] == 0.0 and held[i - 1] > 0.0:
            n = next((k for k in range(i + 1, len(idx)) if held[k] > 0.0), None)
            end = n if n is not None else len(idx)
            low = float(c.iloc[i:end].min()) / float(o.iloc[i]) - 1.0
            sold = float(o.iloc[i])
            row = {"signal": idx[i - 1], "exit": idx[i], "exit_open": sold,
                   "reentry": idx[n] if n is not None else None,
                   "reentry_open": float(o.iloc[n]) if n is not None else None,
                   "days_out": end - i, "price_change": float(o.iloc[n]) / sold - 1.0 if n is not None else None,
                   "low_vs_exit": low}
            ch = row["price_change"]
            row["higher"] = bool(ch > 0) if ch is not None else None
            row["whipsaw"] = bool(ch > 0 and low > whipsaw_floor) if ch is not None else None
            row["whipsaw_strict"] = bool(ch > 0 and low >= 0.0) if ch is not None else None
            rows.append(row)
    return pd.DataFrame(rows)
