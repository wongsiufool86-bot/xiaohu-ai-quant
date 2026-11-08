"""Performance metrics. Every function takes daily returns or an equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def ann_vol(ret: pd.Series) -> float:
    return float(ret.std(ddof=1) * np.sqrt(DAYS))


def sharpe(ret: pd.Series, rf_daily: pd.Series | None = None) -> float:
    ex = ret if rf_daily is None else ret - rf_daily.reindex(ret.index).fillna(0.0)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(DAYS)) if sd > 0 else float("nan")


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def longest_underwater(equity: pd.Series) -> dict:
    """Longest stretch from a peak until equity first gets back above that peak."""
    peak = equity.cummax()
    under = equity < peak
    best = {"days": 0, "start": None, "end": None, "recovered": True}
    start = None
    for d, u in under.items():
        if u and start is None:
            start = prev
        if not u and start is not None:
            days = (d - start).days
            if days > best["days"]:
                best = {"days": days, "start": str(start.date()), "end": str(d.date()), "recovered": True}
            start = None
        prev = d
    if start is not None:
        days = (equity.index[-1] - start).days
        if days > best["days"]:
            best = {"days": days, "start": str(start.date()), "end": None, "recovered": False}
    return best


def trades(weight: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """Round trips for a 0/1 weight series. A trade's return compounds its daily returns."""
    w = weight.fillna(0.0)
    entries = w.index[(w > 0) & (w.shift(1, fill_value=0.0) == 0)]
    exits = w.index[(w == 0) & (w.shift(1, fill_value=0.0) > 0)]
    rows = []
    for e in entries:
        later = exits[exits > e]
        x = later[0] if len(later) else None
        seg = ret.loc[e:x] if x is not None else ret.loc[e:]
        rows.append({"entry": e, "exit": x, "return": float((1 + seg).prod() - 1),
                     "open": x is None})
    return pd.DataFrame(rows)


def summary(bt: pd.DataFrame, rf_daily: pd.Series | None = None) -> dict:
    eq, ret = bt["equity"], bt["ret"]
    out = {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "final_multiple": float(eq.iloc[-1] / eq.iloc[0]),
        "cagr": cagr(eq),
        "ann_vol": ann_vol(ret),
        "sharpe": sharpe(ret, rf_daily),
        "max_drawdown": max_drawdown(eq),
        "longest_underwater": longest_underwater(eq),
        "exposure": float(bt["weight"].mean()) if "weight" in bt else float("nan"),
    }
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else float("nan")
    if "weight" in bt:
        tr = trades(bt["weight"], ret)
        closed = tr[~tr["open"]] if len(tr) else tr
        out["n_trades"] = int(len(tr))
        out["win_rate"] = float((closed["return"] > 0).mean()) if len(closed) else float("nan")
        out["avg_win"] = float(closed.loc[closed["return"] > 0, "return"].mean()) if len(closed) else float("nan")
        out["avg_loss"] = float(closed.loc[closed["return"] <= 0, "return"].mean()) if len(closed) else float("nan")
    return out
