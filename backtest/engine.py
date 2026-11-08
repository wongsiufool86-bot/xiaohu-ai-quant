"""Backtest engines.

Timing convention, used everywhere in this repo:
  * target[t] is decided after the close of day t, using data up to and including day t.
  * It is executed at the OPEN of day t+1.
  * The overnight move (close t -> open t+1) belongs to the position held before the trade.
  * Cash earns the risk-free rate on the part of equity not invested during the day.
  * Cost = |change in weight| x equity at the open x cost_bps / 10,000.

Two independent implementations:
  * vectorized(): pandas, fast. Used for research.
  * event_loop(): day-by-day cash and shares bookkeeping. Used only to reconcile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _inputs(prices: pd.DataFrame, target: pd.Series, rf_daily: pd.Series | None):
    target = target.reindex(prices.index).astype("float64")
    if rf_daily is None:
        rf_daily = pd.Series(0.0, index=prices.index)
    return prices["open"], prices["close"], target, rf_daily.reindex(prices.index).fillna(0.0)


def vectorized(prices: pd.DataFrame, target: pd.Series, cost_bps: float = 0.0,
               rf_daily: pd.Series | None = None) -> pd.DataFrame:
    o, c, target, rf = _inputs(prices, target, rf_daily)
    held = target.shift(1).fillna(0.0)          # weight after today's open trade
    before = held.shift(1).fillna(0.0)          # weight carried in from yesterday's close
    r_on = (o / c.shift(1) - 1.0).fillna(0.0)   # overnight return
    r_id = c / o - 1.0                          # open-to-close return
    open_factor = 1.0 + before * r_on
    w_drift = np.where(open_factor > 0, before * (1.0 + r_on) / open_factor, 0.0)
    turnover = (held - w_drift).abs()
    cost = turnover * cost_bps / 1e4
    after_cost = open_factor * (1.0 - cost)
    close_factor = after_cost * (1.0 + held * r_id + (1.0 - held) * rf)
    equity = close_factor.cumprod()
    return pd.DataFrame({
        "weight": held,
        "turnover": turnover,
        "cost": cost,
        "ret": close_factor - 1.0,
        "equity": equity,
    }, index=prices.index)


def event_loop(prices: pd.DataFrame, target: pd.Series, cost_bps: float = 0.0,
               rf_daily: pd.Series | None = None, start_cash: float = 1.0) -> pd.Series:
    """Day-by-day bookkeeping with cash and (fractional) shares. Returns equity."""
    o, c, target, rf = _inputs(prices, target, rf_daily)
    cash, shares = start_cash, 0.0
    pending = 0.0                                # weight decided at yesterday's close
    out = np.empty(len(prices))
    for i in range(len(prices)):
        op, cl = o.iat[i], c.iat[i]
        value_open = cash + shares * op
        w_now = shares * op / value_open if value_open > 0 else 0.0
        cost = abs(pending - w_now) * value_open * cost_bps / 1e4
        cash -= cost
        value_after_cost = value_open - cost
        want_shares = pending * value_after_cost / op
        cash -= (want_shares - shares) * op
        shares = want_shares
        # interest on the part of equity that sat out of the market today
        cash += value_after_cost * (1.0 - pending) * rf.iat[i]
        out[i] = cash + shares * cl
        t = target.iat[i]
        pending = 0.0 if np.isnan(t) else float(t)
    return pd.Series(out / start_cash, index=prices.index, name="equity")


def same_bar_cheat(prices: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """THE BUG, on purpose: use today's close-based signal to earn today's close-to-close
    return. This is what you get if you forget the one-day shift. For demonstration only."""
    c = prices["close"]
    held = target.reindex(prices.index).fillna(0.0)
    ret = held * c.pct_change().fillna(0.0)
    return pd.DataFrame({"weight": held, "ret": ret, "equity": (1 + ret).cumprod()})
