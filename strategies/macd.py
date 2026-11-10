"""Episode 7: MACD crossover on an index ETF. Teaching example only.

Rules (see specs/ep07_macd.md):
  DIF = EMA(close, fast) - EMA(close, slow); DEA = EMA(DIF, signal).
  golden cross: yesterday DIF <= DEA and today DIF > DEA -> target 1 (buy next open)
  death cross : yesterday DIF >= DEA and today DIF < DEA -> target 0 (sell next open)
  no signal during warm-up; if DIF starts above DEA, wait for the next golden cross.

Modes:
  mode="cross"       the main rule
  mode="zero_above"  a golden cross counts only if DIF and DEA are both above 0; death crosses always sell
  mode="dif_zero"    DIF crossing 0, which is the same thing as EMA(fast) crossing EMA(slow)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def ema(x: pd.Series, n: int, seed: str = "first") -> pd.Series:
    """Recursive EMA with alpha = 2 / (n + 1).
    seed="first": the first value is the first input.
    seed="sma":   NaN for n - 1 bars, then starts from the simple average of the first n inputs."""
    if seed == "first":
        return x.ewm(span=n, adjust=False).mean()
    v = x.to_numpy(float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    start = valid[0] + n - 1
    out[start] = v[valid[0]: start + 1].mean()
    a = 2.0 / (n + 1.0)
    for i in range(start + 1, len(v)):
        out[i] = a * v[i] + (1.0 - a) * out[i - 1]
    return pd.Series(out, index=x.index)


def lines(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, seed: str = "first") -> pd.DataFrame:
    c = prices["close"]
    dif = ema(c, fast, seed) - ema(c, slow, seed)
    dea = ema(dif, signal, seed)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": dif - dea}, index=prices.index)


def default_warmup(slow: int, signal: int) -> int:
    return slow + signal - 1


def crosses(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, warmup: int | None = None,
            seed: str = "first") -> pd.DataFrame:
    """Boolean golden / death crosses, with DIF and DEA blanked out during the warm-up."""
    warmup = default_warmup(slow, signal) if warmup is None else warmup
    ln = lines(prices, fast, slow, signal, seed)
    ln.iloc[: warmup - 1] = np.nan
    dif, dea = ln["dif"], ln["dea"]
    up = (dif.shift(1) <= dea.shift(1)) & (dif > dea)
    down = (dif.shift(1) >= dea.shift(1)) & (dif < dea)
    above = (dif > 0) & (dea > 0)
    return pd.DataFrame({"dif": dif, "dea": dea, "up": up, "down": down, "above_zero": above}, index=prices.index)


def target(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, mode: str = "cross",
           warmup: int | None = None, seed: str = "first") -> pd.Series:
    """Target weight decided after each close, traded at the next open."""
    x = crosses(prices, fast, slow, signal, warmup, seed)
    if mode == "dif_zero":
        up = (x["dif"].shift(1) <= 0) & (x["dif"] > 0)
        down = (x["dif"].shift(1) >= 0) & (x["dif"] < 0)
    elif mode == "zero_above":
        up, down = x["up"] & x["above_zero"], x["down"]
    elif mode == "cross":
        up, down = x["up"], x["down"]
    else:
        raise ValueError(mode)
    sig = pd.Series(np.nan, index=prices.index)
    sig[up] = 1.0
    sig[down] = 0.0
    return sig.ffill().fillna(0.0).rename("target")


def future_change_check(target_fn: Callable[[pd.DataFrame], pd.Series], prices: pd.DataFrame,
                        n_cuts: int = 20, seed: int = 0) -> dict:
    """Replace every price after a random cut with random values. Signals up to the cut must not change."""
    full = target_fn(prices)
    rng = np.random.default_rng(seed)
    cuts = sorted(rng.choice(np.arange(len(prices) // 10, len(prices) - 1), n_cuts, replace=False))
    cols = ["open", "high", "low", "close"]
    bad = []
    for k in cuts:
        changed = prices.copy()
        n = len(prices) - k - 1
        factor = np.exp(rng.normal(0.0, 0.5, n))
        for col in cols:
            changed.iloc[k + 1:, changed.columns.get_loc(col)] = prices[col].to_numpy()[k + 1:] * factor
        a = target_fn(changed).iloc[: k + 1].to_numpy()
        if not np.allclose(a, full.iloc[: k + 1].to_numpy(), equal_nan=True):
            bad.append(str(prices.index[k].date()))
    return {"cuts_tested": len(cuts), "failed_cuts": bad, "passed": not bad}


def trade_table(prices: pd.DataFrame, weight: pd.Series, ret: pd.Series, xs: pd.DataFrame | None = None,
                short_days: int = 10) -> pd.DataFrame:
    """One row per round trip: dates, days held, return, and whether the entry signal was above the zero line."""
    w = weight.fillna(0.0)
    held = w.to_numpy()
    idx = w.index
    r = ret.reindex(idx).to_numpy()
    above = None if xs is None else xs["above_zero"].reindex(idx).to_numpy()
    rows, i = [], 1
    while i < len(idx):
        if held[i] > 0.0 and held[i - 1] == 0.0:
            x = next((k for k in range(i + 1, len(idx)) if held[k] == 0.0), None)
            end = x if x is not None else len(idx) - 1
            rows.append({"signal": idx[i - 1], "entry": idx[i], "exit": idx[x] if x is not None else None,
                         "days_held": end - i, "return": float(np.prod(1.0 + r[i: end + 1]) - 1.0),
                         "open": x is None, "short": x is not None and end - i <= short_days,
                         "above_zero": bool(above[i - 1]) if above is not None else None})
            i = end + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def trade_stats(tr: pd.DataFrame, years: float) -> dict:
    closed = tr[~tr["open"]]
    wins, losses = closed.loc[closed["return"] > 0, "return"], closed.loc[closed["return"] <= 0, "return"]
    return {"n": int(len(tr)), "closed": int(len(closed)), "per_year": float(len(tr) / years),
            "median_days_held": float(closed["days_held"].median()),
            "n_short": int(closed["short"].sum()), "share_short": float(closed["short"].mean()),
            "n_wins": int(len(wins)), "win_rate": float(len(wins) / len(closed)),
            "avg_win": float(wins.mean()), "avg_loss": float(losses.mean()),
            "avg_return": float(closed["return"].mean())}
