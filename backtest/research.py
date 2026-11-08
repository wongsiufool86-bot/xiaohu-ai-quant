"""The fixed three-part health check: costs, parameters, before/after split."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import engine, metrics


def window(bt: pd.DataFrame, start, end=None) -> pd.DataFrame:
    """Slice a backtest and restart equity at 1 on the first day."""
    part = bt.loc[start:end].copy()
    part["equity"] = (1 + part["ret"]).cumprod()
    return part


def cost_sweep(prices, target, bench_target, rf, start, bps_list) -> pd.DataFrame:
    rows = []
    for b in bps_list:
        s = window(engine.vectorized(prices, target, b, rf), start)
        h = window(engine.vectorized(prices, bench_target, b, rf), start)
        rows.append({"cost_bps": b, "strategy_cagr": metrics.cagr(s["equity"]),
                     "buyhold_cagr": metrics.cagr(h["equity"])})
    return pd.DataFrame(rows)


def breakeven_bps(sweep: pd.DataFrame, col_a: str, col_b: str | None = None) -> float | None:
    """Smallest cost at which col_a falls below col_b (or below zero), by linear interpolation."""
    a = sweep[col_a].values
    b = sweep[col_b].values if col_b else np.zeros(len(sweep))
    d = a - b
    x = sweep["cost_bps"].values
    if d[0] <= 0:
        return 0.0
    for i in range(1, len(d)):
        if d[i] <= 0:
            return float(x[i - 1] + (x[i] - x[i - 1]) * d[i - 1] / (d[i - 1] - d[i]))
    return None


def param_grid(prices, target_fn: Callable, fasts, slows, cost_bps, rf, start, end=None) -> pd.DataFrame:
    rows = []
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            bt = window(engine.vectorized(prices, target_fn(prices, f, s), cost_bps, rf), start, end)
            rows.append({"fast": f, "slow": s,
                         "sharpe": metrics.sharpe(bt["ret"], rf.loc[bt.index]),
                         "cagr": metrics.cagr(bt["equity"]),
                         "max_drawdown": metrics.max_drawdown(bt["equity"])})
    return pd.DataFrame(rows)


def neighbourhood(grid: pd.DataFrame, fast: int, slow: int, col: str = "sharpe") -> float:
    """Average of `col` over the grid points immediately around (fast, slow)."""
    fs = sorted(grid.fast.unique())
    ss = sorted(grid.slow.unique())
    i, j = fs.index(fast), ss.index(slow)
    nf = fs[max(0, i - 1): i + 2]
    ns = ss[max(0, j - 1): j + 2]
    g = grid[grid.fast.isin(nf) & grid.slow.isin(ns)]
    return float(g[col].mean())
