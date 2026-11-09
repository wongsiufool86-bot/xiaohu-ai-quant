"""Episode 4: buy when RSI is oversold, sell when it is overbought. Teaching example only.

Rules (see specs/ep04_rsi_dip.md):
  flat and RSI < buy  -> target 1
  held and RSI > sell -> target 0
  otherwise keep the state. No signal until RSI exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Seed is the simple average of the first `period` changes, so the first value is at bar `period`."""
    c = close.to_numpy(dtype="float64")
    out = np.full(len(c), np.nan)
    if len(c) <= period:
        return pd.Series(out, index=close.index, name="rsi")
    d = np.diff(c)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()

    def value(g, l):
        if l == 0:
            return 50.0 if g == 0 else 100.0
        return 100.0 - 100.0 / (1.0 + g / l)

    out[period] = value(ag, al)
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
        out[i + 1] = value(ag, al)
    return pd.Series(out, index=close.index, name="rsi")


def _state_machine(buy_signal: np.ndarray, sell_signal: np.ndarray, index) -> pd.Series:
    held, out = 0.0, np.zeros(len(index))
    for i in range(len(index)):
        if held == 0.0 and buy_signal[i]:
            held = 1.0
        elif held == 1.0 and sell_signal[i]:
            held = 0.0
        out[i] = held
    return pd.Series(out, index=index, name="target")


def target(prices: pd.DataFrame, period: int = 14, buy: float = 30.0, sell: float = 70.0) -> pd.Series:
    r = rsi(prices["close"], period)
    # comparisons with NaN are False, so no signal before RSI exists
    return _state_machine((r < buy).to_numpy(), (r > sell).to_numpy(), prices.index)


def target_ma_exit(prices: pd.DataFrame, period: int = 2, buy: float = 10.0, ma: int = 5) -> pd.Series:
    """Short version: buy when RSI(period) < buy, sell when the close is above its ma-day average."""
    c = prices["close"]
    r, avg = rsi(c, period), c.rolling(ma).mean()
    return _state_machine((r < buy).to_numpy(), (c > avg).to_numpy(), prices.index)


def first_signal_date(prices: pd.DataFrame, period: int = 14) -> pd.Timestamp:
    return prices.index[period]


def trade_log(prices: pd.DataFrame, weight: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """One row per round trip. Prices stay out of the output; everything is relative to the entry open."""
    w = weight.fillna(0.0)
    o, c, lo = prices["open"], prices["close"], prices["low"]
    peak = c.cummax()
    rs = rsi(c)
    entries = w.index[(w > 0) & (w.shift(1, fill_value=0.0) == 0)]
    exits = w.index[(w == 0) & (w.shift(1, fill_value=0.0) > 0)]
    all_days = w.index
    rows = []
    for e in entries:
        later = exits[exits > e]
        x = later[0] if len(later) else None
        held = w.loc[e:x].iloc[:-1] if x is not None else w.loc[e:]
        seg = ret.loc[e:x] if x is not None else ret.loc[e:]
        entry_open = o.loc[e]
        closes, lows = c.loc[held.index], lo.loc[held.index]
        sig_day = all_days[all_days.get_loc(e) - 1]
        low_i = closes.values.argmin()
        rows.append({
            "entry": e, "exit": x, "open": x is None,
            "return": float((1 + seg).prod() - 1),
            "days_held": int(len(held)),
            "rsi_at_signal": float(rs.loc[sig_day]),
            "below_peak_at_signal": float(c.loc[sig_day] / peak.loc[sig_day] - 1),
            "mae_close": min(0.0, float(closes.min() / entry_open - 1)),
            "mae_low": min(0.0, float(lows.min() / entry_open - 1)),
            "days_to_worst": int(low_i),
            "worst_date": closes.index[low_i],
        })
    return pd.DataFrame(rows)
