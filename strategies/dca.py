"""Episode 2: lump sum vs dollar-cost averaging (DCA). Teaching example only.

Lump sum: invest everything at the open of the first trading day of the start month.
DCA(N):   invest 1/N of the starting cash at the open of the first trading day of each of
          the first N months; idle cash earns the daily risk-free rate; the last instalment
          invests whatever cash is left (including interest).
Both pay cost_bps on every purchase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).min().values)


def _buy_days(index, start, n):
    ms = month_starts(index)
    k = ms.get_loc(start)
    days = ms[k:k + n]
    if len(days) < n:
        raise ValueError("not enough months after start")
    return days


def lump_sum_path(px: pd.DataFrame, start, end, cost_bps: float = 0.0) -> pd.Series:
    """Daily account value from start to end; starts with 1.0 cash."""
    o, c = px["open"], px["close"]
    shares = (1.0 - cost_bps / 1e4) / o.loc[start]
    return (shares * c.loc[start:end]).rename("lump_sum")


def dca_path(px: pd.DataFrame, rf_daily: pd.Series, start, end, n: int,
             cost_bps: float = 0.0) -> pd.Series:
    """Day-by-day bookkeeping. Interest is credited at each close on the cash held that day."""
    o, c = px["open"], px["close"]
    days = set(_buy_days(px.index, start, n))
    last = max(days)
    idx = px.index[(px.index >= start) & (px.index <= end)]
    cash, shares, inst = 1.0, 0.0, 1.0 / n
    out = np.empty(len(idx))
    for i, d in enumerate(idx):
        if d in days:
            amount = cash if d == last else min(inst, cash)
            shares += amount * (1 - cost_bps / 1e4) / o.loc[d]
            cash -= amount
        cash *= 1.0 + rf_daily.loc[d]
        out[i] = cash + shares * c.loc[d]
    return pd.Series(out, index=idx, name="dca")


def dca_final_formula(px: pd.DataFrame, rf_daily: pd.Series, start, end, n: int,
                      cost_bps: float = 0.0) -> float:
    """Same thing in closed form, written independently, for the reconcile check."""
    o, c = px["open"], px["close"]
    days = list(_buy_days(px.index, start, n))
    idx = px.index[(px.index >= start) & (px.index <= end)]
    g = (1.0 + rf_daily.reindex(idx).fillna(0.0)).cumprod()        # growth of cash, credited at each close
    g_before = g.shift(1, fill_value=1.0)                           # cash growth up to (not incl.) a day
    keep = 1 - cost_bps / 1e4
    wealth_in_stock = 0.0
    # cash balance just before buying on day k, in "value at that moment":
    cash = 1.0
    prev_gb = 1.0
    for k, d in enumerate(days):
        cash = cash * g_before.loc[d] / prev_gb
        amount = cash if k == len(days) - 1 else min(1.0 / n, cash)
        wealth_in_stock += amount * keep * c.loc[end] / o.loc[d]
        cash -= amount
        prev_gb = g_before.loc[d]
    cash_end = cash * g.loc[end] / prev_gb
    return float(wealth_in_stock + cash_end)
