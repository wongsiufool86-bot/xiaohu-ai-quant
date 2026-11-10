"""Interstitial x01: next-day signals from an open-source K-line foundation model. Teaching example only.

Rules (see specs/x01_kronos.md):
  after the close of day t, feed the last CONTEXT daily bars (open, high, low, close, volume),
  ending at t, to the model; draw N one-bar-ahead paths for day t+1;
  P(up) = share of paths whose close is above close_t;
  hold the ETF from the open of t+1 if P(up) > threshold, otherwise hold cash.

This file has no model code and no torch import. run_x01.py runs the model and caches the paths;
everything here works on the cached numbers, so tests can feed it synthetic paths.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

CONTEXT = 400
COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------- windows and dates
def asof_window(px: pd.DataFrame, asof, n: int = CONTEXT) -> pd.DataFrame:
    """The n bars ending at `asof`, inclusive. Nothing after `asof` is ever returned."""
    i = px.index.get_loc(pd.Timestamp(asof))
    if i + 1 < n:
        raise ValueError(f"need {n} bars up to {asof}, have {i + 1}")
    out = px.iloc[i + 1 - n: i + 1][COLS]
    assert out.index[-1] == pd.Timestamp(asof)
    return out


def segment_days(index: pd.DatetimeIndex, test_start: str, test_end: str, control_end: str,
                 context: int = CONTEXT) -> dict:
    """(asof, target) pairs. 'test' = targets in [test_start, test_end]; 'control' = the same number
    of targets ending on or before control_end. asof is the trading day before target."""
    idx = pd.DatetimeIndex(index)
    pos = np.arange(len(idx))
    t_mask = (idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(test_end))
    test_pos = pos[t_mask]
    n = len(test_pos)
    last_c = int(pos[idx <= pd.Timestamp(control_end)][-1])
    ctrl_pos = np.arange(last_c - n + 1, last_c + 1)
    if ctrl_pos[0] - 1 < context - 1:
        raise ValueError("not enough history for the control segment")
    pairs = lambda p: [(idx[k - 1], idx[k]) for k in p]
    return {"test": pairs(test_pos), "control": pairs(ctrl_pos)}


def day_seed(seed: int, symbol: str, asof) -> int:
    """A fixed seed per (run seed, symbol, day): any day can be rerun alone and give the same paths."""
    return zlib.crc32(f"{seed}|{symbol}|{pd.Timestamp(asof).date()}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------- per-day summaries
def summarize(paths, close_t: float) -> dict:
    p = np.asarray(paths, dtype=float)
    return {"pred_mean": float(p.mean()), "pred_ret": float(p.mean() / close_t - 1.0),
            "p_up": float((p > close_t).mean()),
            "q10": float(np.quantile(p, 0.10)), "q50": float(np.quantile(p, 0.50)),
            "q90": float(np.quantile(p, 0.90))}


def window_z(px: pd.DataFrame, asof, n: int = CONTEXT) -> float:
    """(close_t - mean of the window's closes) / std of the window's closes. Same window the model sees."""
    c = asof_window(px, asof, n)["close"].to_numpy(float)
    return float((c[-1] - c.mean()) / c.std())


def daily_frame(pred: pd.DataFrame, px: pd.DataFrame, n_paths: int | None = None,
                context: int = CONTEXT) -> pd.DataFrame:
    """One row per asof day: model summary + what actually happened. `pred` has columns
    asof, target, close_t, paths (array). Uses only the first n_paths paths if given."""
    rows = []
    close = px["close"]
    for r in pred.itertuples(index=False):
        paths = np.asarray(r.paths, dtype=float)
        if n_paths is not None:
            paths = paths[:n_paths]
        s = summarize(paths, r.close_t)
        prev = close.iloc[close.index.get_loc(r.asof) - 1]
        s.update({"asof": r.asof, "target": r.target, "close_t": r.close_t,
                  "actual_close": float(close.loc[r.target]),
                  "actual_ret": float(close.loc[r.target] / r.close_t - 1.0),
                  "last_ret": float(r.close_t / prev - 1.0),
                  "z": window_z(px, r.asof, context)})
        rows.append(s)
    return pd.DataFrame(rows).set_index("asof").sort_index()


# ---------------------------------------------------------------- statistics, no scipy
def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (sum of outcomes no more likely than k)."""
    logp = lambda i: (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log(1 - p))
    lk = logp(k)
    tot = sum(math.exp(logp(i)) for i in range(n + 1) if logp(i) <= lk + 1e-9)
    return float(min(1.0, tot))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def mcnemar(a_right: np.ndarray, b_right: np.ndarray) -> dict:
    """Paired comparison of two guessers on the same days: exact test on the days they disagree."""
    a, b = np.asarray(a_right, bool), np.asarray(b_right, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return {"only_model_right": only_a, "only_baseline_right": only_b,
            "p_value": binom_two_sided(only_a, n) if n else 1.0}


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(float)


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x, y) -> float:
    return corr(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))


def corr_t(r: float, n: int) -> float:
    return float(r * math.sqrt((n - 2) / max(1e-12, 1 - r * r)))


def direction_table(df: pd.DataFrame) -> dict:
    """Hit rates of the model and three baselines on the same days. Flat days (actual_ret == 0) dropped."""
    d = df[df["actual_ret"] != 0]
    up = d["actual_ret"] > 0
    model = (d["pred_ret"] > 0) == up
    always_up = up
    momentum = (d["last_ret"] > 0) == up
    n, k = len(d), int(model.sum())
    lo, hi = wilson(k, n)
    return {"days": n, "model_hits": k, "model_hit_rate": k / n, "model_ci95": [lo, hi],
            "p_vs_50": binom_two_sided(k, n), "coin": 0.5,
            "always_up_hit_rate": float(always_up.mean()), "momentum_hit_rate": float(momentum.mean()),
            "model_says_up_share": float((d["pred_ret"] > 0).mean()),
            "p_up_says_up_share": float((d["p_up"] > 0.5).mean()),
            "vs_always_up": mcnemar(model.to_numpy(), always_up.to_numpy()),
            "vs_momentum": mcnemar(model.to_numpy(), momentum.to_numpy())}


def ic_table(df: pd.DataFrame) -> dict:
    n = len(df)
    ic, ric = corr(df["pred_ret"], df["actual_ret"]), rank_corr(df["pred_ret"], df["actual_ret"])
    ric_p = rank_corr(df["p_up"], df["actual_ret"])
    return {"days": n, "ic": ic, "ic_t": corr_t(ic, n), "rank_ic": ric, "rank_ic_t": corr_t(ric, n),
            "rank_ic_p_up": ric_p}


def cross_sectional_rank_ic(frames: dict) -> dict:
    """Each day, rank correlation across symbols of predicted vs actual return; then average."""
    pr = pd.DataFrame({s: f["pred_ret"] for s, f in frames.items()}).dropna()
    ar = pd.DataFrame({s: f["actual_ret"] for s, f in frames.items()}).loc[pr.index]
    daily = pd.Series([rank_corr(pr.loc[d].to_numpy(), ar.loc[d].to_numpy()) for d in pr.index],
                      index=pr.index).dropna()
    m, s = float(daily.mean()), float(daily.std(ddof=1))
    return {"days": int(len(daily)), "symbols": list(frames), "mean": m, "std": s,
            "icir": m / s if s > 0 else float("nan"), "t": m / s * math.sqrt(len(daily)) if s > 0 else float("nan"),
            "share_positive": float((daily > 0).mean())}


CAL_EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]


def calibration(df: pd.DataFrame, edges=CAL_EDGES) -> list[dict]:
    """Bucket the days by P(up); in each bucket, how often did the close actually rise?"""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = df[(df["p_up"] >= lo) & (df["p_up"] < hi)]
        out.append({"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "days": int(len(b)),
                    "mean_p_up": float(b["p_up"].mean()) if len(b) else None,
                    "actual_up": float((b["actual_ret"] > 0).mean()) if len(b) else None})
    return out


def coverage(df: pd.DataFrame) -> dict:
    a = df["actual_close"]
    inside = (a >= df["q10"]) & (a <= df["q90"])
    width = (df["q90"] - df["q10"]) / df["close_t"]
    return {"inside_10_90": float(inside.mean()), "below_q10": float((a < df["q10"]).mean()),
            "above_q90": float((a > df["q90"]).mean()), "median_band_width_pct": float(width.median()),
            "actual_abs_move_median_pct": float(df["actual_ret"].abs().median())}


def mean_pull(df: pd.DataFrame) -> dict:
    """Does the forecast lean back towards the window average? corr(pred_ret, z) strongly negative
    while corr(actual_ret, z) is ~0 means the pull comes from the model/normalisation, not the market."""
    hi, lo = df[df["z"] > 1], df[df["z"] < -1]
    return {"corr_pred_z": corr(df["pred_ret"], df["z"]), "rank_corr_pred_z": rank_corr(df["pred_ret"], df["z"]),
            "corr_actual_z": corr(df["actual_ret"], df["z"]),
            "days_z_above_1": int(len(hi)), "pred_ret_z_above_1": float(hi["pred_ret"].mean()) if len(hi) else None,
            "actual_ret_z_above_1": float(hi["actual_ret"].mean()) if len(hi) else None,
            "p_up_z_above_1": float(hi["p_up"].mean()) if len(hi) else None,
            "days_z_below_m1": int(len(lo)), "pred_ret_z_below_m1": float(lo["pred_ret"].mean()) if len(lo) else None,
            "actual_ret_z_below_m1": float(lo["actual_ret"].mean()) if len(lo) else None,
            "p_up_z_below_m1": float(lo["p_up"].mean()) if len(lo) else None,
            "z_mean": float(df["z"].mean()), "share_z_positive": float((df["z"] > 0).mean())}


# ---------------------------------------------------------------- trading
def target_weights(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """1 = hold from the next open, 0 = cash. Indexed by asof (decision day)."""
    return (df["p_up"] > threshold).astype(float).rename("target")


def close_to_close(df: pd.DataFrame, weights: pd.Series, cost_bps: float, rf_daily: pd.Series) -> pd.DataFrame:
    """THE UPPER BOUND, not tradable: buy or sell exactly at close_t with a signal that needs close_t,
    and earn close_t -> close_{t+1}. Cost on each weight change. Indexed by target day."""
    w = weights.reindex(df.index).fillna(0.0).to_numpy()
    prev = np.concatenate([[0.0], w[:-1]])
    cost = np.abs(w - prev) * cost_bps / 1e4
    rf = rf_daily.reindex(pd.DatetimeIndex(df["target"])).fillna(0.0).to_numpy()
    ret = (1 - cost) * (1 + w * df["actual_ret"].to_numpy() + (1 - w) * rf) - 1
    out = pd.DataFrame({"weight": w, "ret": ret}, index=pd.DatetimeIndex(df["target"]))
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


# ---------------------------------------------------------------- multi-step fans
def fan_stats(paths: np.ndarray, close_t: float, actual: np.ndarray) -> dict:
    """paths: (n_paths, horizon) closes; actual: (horizon,) realised closes."""
    p = np.asarray(paths, float)
    a = np.asarray(actual, float)
    lo, hi = np.quantile(p, 0.10, axis=0), np.quantile(p, 0.90, axis=0)
    mean_end = float(p[:, -1].mean())
    return {"end_pred_ret": mean_end / close_t - 1, "end_actual_ret": float(a[-1] / close_t - 1),
            "end_p_up": float((p[:, -1] > close_t).mean()),
            "direction_right": bool((mean_end > close_t) == (a[-1] > close_t)),
            "end_error_pct": float(mean_end / a[-1] - 1),
            "share_days_inside_10_90": float(((a >= lo) & (a <= hi)).mean()),
            "end_inside_10_90": bool(lo[-1] <= a[-1] <= hi[-1]),
            "end_band_width_pct": float((hi[-1] - lo[-1]) / close_t)}
