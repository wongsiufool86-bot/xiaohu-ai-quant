"""Episode 8: a fixed-weight stock/bond portfolio, rebalanced on the calendar. Teaching example only.

Rules (see specs/ep08_rebalance_6040.md):
  decide after the first close, buy the target weights at the next open, then at the OPEN of the first trading
  day of each period trade back to the target weights. freq="none" never trades again (the weights drift).

Why this file has its own engines: backtest/engine.py holds one asset plus cash; this portfolio holds two assets.
Timing and costs copy that engine:
  * the first bar only decides, so it sits in cash and earns that day's risk-free rate;
  * cost = sum over assets of |target weight - drifted weight at the open| x equity at the open x bps / 10,000;
  * the new weights are applied to what is left after the cost; there is no cash after the first trade.

  simulate()    one pass per holding segment, numpy inside each segment. Used for research.
  event_loop()  day-by-day shares and cash. Used only to reconcile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FREQS = ("none", "annual", "quarterly", "monthly")


def rebalance_flags(index: pd.DatetimeIndex, freq: str, month: int = 1) -> np.ndarray:
    """True on the first trading day of each period. Uses dates only, never prices. Bar 0 is never flagged.
    `month` is the calendar month of the annual rebalance."""
    if freq not in FREQS:
        raise ValueError(f"unknown freq {freq!r}")
    new_month = np.zeros(len(index), dtype=bool)
    new_month[1:] = (index.year[1:] != index.year[:-1]) | (index.month[1:] != index.month[:-1])
    if freq == "none":
        return np.zeros(len(index), dtype=bool)
    if freq == "monthly":
        return new_month
    if freq == "quarterly":
        return new_month & np.isin(np.asarray(index.month), [1, 4, 7, 10])
    return new_month & (np.asarray(index.month) == month)


def _weights(w: dict, opens: pd.DataFrame) -> tuple[list, np.ndarray]:
    names = list(w)
    arr = np.array([w[k] for k in names], float)
    if abs(arr.sum() - 1.0) > 1e-12 or (arr < 0).any():
        raise ValueError("weights must be non-negative and sum to 1")
    missing = [k for k in names if k not in opens.columns]
    if missing:
        raise KeyError(missing)
    return names, arr


def _trade_mask(index: pd.DatetimeIndex, freq: str, month: int) -> np.ndarray:
    mask = rebalance_flags(index, freq, month)
    if len(mask) > 1:
        mask[1] = True                       # the first purchase
    return mask


def simulate(opens: pd.DataFrame, closes: pd.DataFrame, w: dict, freq: str = "annual", cost_bps: float = 5.0,
             rf_daily: pd.Series | None = None, month: int = 1) -> pd.DataFrame:
    """Segment by segment. Inside a segment each asset's value is (value at the trade) x close / trade open."""
    names, tgt = _weights(w, opens)
    idx = opens.index
    o, c = opens[names].to_numpy(float), closes[names].to_numpy(float)
    rf0 = 0.0 if rf_daily is None else float(rf_daily.reindex(idx).fillna(0.0).iloc[0])
    n, k = len(idx), len(names)
    trades = np.flatnonzero(_trade_mask(idx, freq, month))
    values = np.zeros((n, k))
    equity, turnover, cost = np.zeros(n), np.zeros(n), np.zeros(n)
    equity[0] = 1.0 + rf0
    held = np.zeros(k)
    prev_t = None
    for a, t in enumerate(trades):
        if prev_t is None:
            value_open, drift = equity[0], np.zeros(k)
        else:
            per_asset = held * o[t] / o[prev_t]
            value_open = per_asset.sum()
            drift = per_asset / value_open
        turnover[t] = np.abs(tgt - drift).sum()
        cost[t] = turnover[t] * cost_bps / 1e4
        held = tgt * value_open * (1.0 - cost[t])
        end = trades[a + 1] if a + 1 < len(trades) else n
        values[t:end] = held * c[t:end] / o[t]
        prev_t = t
    if len(trades):
        equity[trades[0]:] = values[trades[0]:].sum(axis=1)
    ret = np.zeros(n)
    ret[0] = equity[0] - 1.0
    ret[1:] = equity[1:] / equity[:-1] - 1.0
    traded = np.zeros(n, dtype=bool)
    traded[trades] = True
    out = pd.DataFrame({"ret": ret, "equity": equity, "turnover": turnover, "cost": cost, "traded": traded}, index=idx)
    total = values.sum(axis=1)
    for j, name in enumerate(names):
        out[f"w_{name}"] = np.divide(values[:, j], total, out=np.zeros(n), where=total > 0)
    return out


def event_loop(opens: pd.DataFrame, closes: pd.DataFrame, w: dict, freq: str = "annual", cost_bps: float = 5.0,
               rf_daily: pd.Series | None = None, month: int = 1) -> pd.Series:
    """Day-by-day cash and fractional shares. Returns equity."""
    names, tgt = _weights(w, opens)
    idx = opens.index
    o, c = opens[names].to_numpy(float), closes[names].to_numpy(float)
    rf = np.zeros(len(idx)) if rf_daily is None else rf_daily.reindex(idx).fillna(0.0).to_numpy(float)
    trade = _trade_mask(idx, freq, month)
    cash, shares = 1.0, np.zeros(len(names))
    out = np.empty(len(idx))
    for t in range(len(idx)):
        if trade[t]:
            value_open = cash + float(shares @ o[t])
            now = shares * o[t] / value_open
            fee = np.abs(tgt - now).sum() * value_open * cost_bps / 1e4
            shares = tgt * (value_open - fee) / o[t]
            cash = 0.0
        elif t == 0:
            cash *= 1.0 + rf[0]
        out[t] = cash + float(shares @ c[t])
    return pd.Series(out, index=idx, name="equity")


def ledger(opens: pd.DataFrame, closes: pd.DataFrame, w: dict, bt: pd.DataFrame, stock: str = "SPY") -> pd.DataFrame:
    """One row per rebalance (not the first purchase): where the weights had drifted, what the trade cost,
    and whether the trade helped by the next rebalance.

    The counterfactual is one step only: from this open to the next rebalance (or the last close),
    compare the portfolio after the trade (target weights, net of cost) with the same money left at the drifted
    weights. Effect = after / left alone - 1."""
    names, tgt = _weights(w, opens)
    idx = opens.index
    trades = np.flatnonzero(bt["traded"].to_numpy())
    o, c = opens[names].to_numpy(float), closes[names].to_numpy(float)
    rows = []
    j = names.index(stock)
    for a, t in enumerate(trades):
        if a == 0:
            continue
        prev = trades[a - 1]
        per_asset = tgt * o[t] / o[prev]
        drift = per_asset / per_asset.sum()
        last = (o[trades[a + 1]] / o[t]) if a + 1 < len(trades) else (c[-1] / o[t])
        cost = float(bt["cost"].iat[t])
        traded = (1.0 - cost) * float(tgt @ last)
        left = float(drift @ last)
        rows.append({"date": idx[t], "stock_weight_before": float(drift[j]), "stock_weight_after": float(tgt[j]),
                     "action": "sold_stocks" if drift[j] > tgt[j] else "bought_stocks", "turnover": float(bt["turnover"].iat[t]),
                     "cost": cost, "next_end": idx[trades[a + 1]] if a + 1 < len(trades) else idx[-1],
                     "traded_growth": traded, "left_alone_growth": left, "effect": traded / left - 1.0})
    return pd.DataFrame(rows)
