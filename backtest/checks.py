"""Mandatory checks that every strategy must pass before its results are shown."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import engine


def lookahead_check(target_fn: Callable[[pd.DataFrame], pd.Series], prices: pd.DataFrame,
                    n_cuts: int = 20, seed: int = 0) -> dict:
    """Cut the data at random dates. Signals up to each cut must not change.
    If they do, the strategy is peeking at data after the cut."""
    full = target_fn(prices)
    rng = np.random.default_rng(seed)
    cuts = sorted(rng.choice(np.arange(len(prices) // 10, len(prices) - 1), n_cuts, replace=False))
    bad = []
    for k in cuts:
        part = target_fn(prices.iloc[: k + 1])
        a, b = part.values, full.iloc[: k + 1].values
        if not np.allclose(a, b, equal_nan=True):
            bad.append(str(prices.index[k].date()))
    return {"cuts_tested": len(cuts), "failed_cuts": bad, "passed": not bad}


def reconcile(prices: pd.DataFrame, target: pd.Series, cost_bps: float,
              rf_daily: pd.Series | None, tol: float = 1e-6) -> dict:
    """Vectorized engine vs day-by-day bookkeeping. Max relative gap must be below tol."""
    v = engine.vectorized(prices, target, cost_bps, rf_daily)["equity"]
    e = engine.event_loop(prices, target, cost_bps, rf_daily)
    gap = (v / e - 1.0).abs()
    return {"max_relative_gap": float(gap.max()), "tolerance": tol, "passed": bool(gap.max() < tol)}
