"""Interstitial x01: can an open-source K-line foundation model call tomorrow? Writes every number
to results/x01/results.json.

    PARQUET_EXPORT_DIR=... or DATA_DIR=... python run_x01.py predict   # slow: runs the model, caches paths
    PARQUET_EXPORT_DIR=... or DATA_DIR=... python run_x01.py           # fast: statistics from the cache

The model is Kronos (arXiv 2508.02739, github.com/shiyu-coder/Kronos, MIT). Its code is NOT in this
repo: put a checkout next to this folder at ../third_party/Kronos, or point KRONOS_DIR at one.
The weights load through Hugging Face (NeoQuasar/Kronos-small, NeoQuasar/Kronos-Tokenizer-base);
set HF_HUB_OFFLINE=1 once they are cached. CPU only.

Every prediction is cached in results/x01/predictions.parquet (one row per symbol, seed, day), so an
interrupted run continues where it stopped. Each day has its own fixed seed (strategies/kronos_signal.day_seed),
so any single day can be rerun and must give the same paths.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import checks, data, engine, metrics, research
from strategies import kronos_signal as ks

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "x01"
PRED_FILE = OUT / "predictions.parquet"
FAN_FILE = OUT / "fans.parquet"
LOOK_FILE = OUT / "lookahead.parquet"
CACHE = OUT / "cache"

SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
MAIN = "SPY"
TEST_START, TEST_END, CONTROL_END = "2024-07-01", "2026-10-02", "2024-06-28"
DATA_END = TEST_END
N_PATHS, MAIN_PATHS_RUN = 32, 64          # SPY test segment, seed 0 is drawn with 64 paths for the 8/32/64 check
SEEDS = [0, 1, 2]
T, TOP_P, TOP_K = 1.0, 0.9, 0
FAN_LEN, FAN_STEP = 20, 20
MODEL_NAME, TOKENIZER_NAME = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"
LOOKAHEAD_DAYS = 3

# Facts from the paper (arXiv 2508.02739v1) and the official README, checked by hand on 2026-11-10.
PAPER = {
    "arxiv": "2508.02739", "submitted": "2025-08-02", "authors": 7, "affiliation": "Tsinghua University",
    "exchanges": 45, "klines_billion": 12, "pretrain_data_ends": "2024-06", "test_starts": "2024-07",
    "params_m": {"mini": 4.1, "small": 24.7, "base": 102.3, "large": 499.2}, "large_open_source": False,
    "price_forecast": {
        "where": "Table 14 (Kronos and full-shot models), row 'Average', over 9 stock exchanges + crypto + forex",
        "how": "per sample: IC/RankIC between the predicted and true price series over the forecast horizon, "
               "for each of open/high/low/close, then averaged (Appendix D)",
        "rank_ic": {"small": 0.0254, "base": 0.0258, "large": 0.0267},
        "ic": {"small": 0.0431, "base": 0.0435, "large": 0.0440},
        "best_full_shot_baseline_rank_ic": 0.0143,      # TimesNet, Table 14
        "best_zero_shot_tsfm_rank_ic": 0.0138,          # Time-MOE base, Table 15
        "abstract_claim_pct_over_tsfm": 93, "abstract_claim_pct_over_baseline": 87},
    "return_forecast_rank_ic": {"small": 0.0622, "base": 0.0634, "large": 0.0675},   # Table 16, 'Average'
}


def load(symbol: str) -> pd.DataFrame:
    """Prices up to DATA_END only. The study is fixed to data ending 2026-10-02; bars added later
    to the data folder are ignored, so the 'last 20 bars' are always the same 20 bars."""
    return data.load_prices(symbol).loc[:DATA_END]


# ================================================================ model (only for `predict`)
def load_predictor():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    kdir = os.environ.get("KRONOS_DIR", str(ROOT.parent / "third_party" / "Kronos"))
    sys.path.insert(0, kdir)
    import torch
    torch.set_num_threads(int(os.environ.get("X01_THREADS", "6")))
    from model import Kronos, KronosPredictor, KronosTokenizer
    tok = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    m = Kronos.from_pretrained(MODEL_NAME)
    tok.eval(); m.eval()
    return KronosPredictor(m, tok, device="cpu", max_context=512), torch


def draw_paths(pred, torch, px, asof, future_dates, n_paths, seed, symbol) -> np.ndarray:
    """n_paths independent paths. The official sample_count AVERAGES its samples into one path, so
    the same window goes in n_paths times with sample_count=1. Returns (n_paths, horizon) closes."""
    x = ks.asof_window(px, asof)
    ts = pd.Series(x.index)
    ys = pd.Series(pd.DatetimeIndex(future_dates))
    torch.manual_seed(ks.day_seed(seed, symbol, asof))
    out = pred.predict_batch([x] * n_paths, [ts] * n_paths, [ys] * n_paths, pred_len=len(ys),
                             T=T, top_k=TOP_K, top_p=TOP_P, sample_count=1, verbose=False)
    return np.stack([o["close"].to_numpy(float) for o in out])


def _save(rows: list, path: Path):
    tmp = path.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)


def _load_rows(path: Path) -> list:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    for r in rows:
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                r[k] = v.tolist()
    return rows


def jobs():
    """One job = one symbol, segment and seed. Ordered so the main result is ready first."""
    out = []
    for seg in ("test", "control"):
        for s in SYMBOLS:
            n = MAIN_PATHS_RUN if (s == MAIN and seg == "test") else N_PATHS
            out.append((s, seg, 0, n))
    for seed in SEEDS[1:]:
        out.append((MAIN, "test", seed, N_PATHS))
    return out


def fan_asofs(px) -> list:
    """Every FAN_STEP trading days through the test segment, plus the post's 'hide the last 20 bars'."""
    days = ks.segment_days(px.index, TEST_START, TEST_END, CONTROL_END)["test"]
    pos = [px.index.get_loc(a) for a, _ in days]
    grid = [px.index[p] for p in pos[::FAN_STEP] if p + FAN_LEN < len(px)]
    last = px.index[len(px) - 1 - FAN_LEN]
    return grid, last


def shard(symbol, seg, seed) -> Path:
    return CACHE / f"{symbol}_{seg}_seed{seed}.parquet"


def merge_cache() -> pd.DataFrame:
    """All finished shards -> results/x01/predictions.parquet (what the statistics read)."""
    parts = [pd.read_parquet(f) for f in sorted(CACHE.glob("*_seed*.parquet"))]
    df = pd.concat(parts, ignore_index=True).sort_values(["symbol", "seed", "asof"])
    # a job can be finished by two processes; fixed per-day seeds make the rows identical, keep one
    df = df.drop_duplicates(subset=["symbol", "segment", "seed", "asof"]).reset_index(drop=True)
    df.to_parquet(PRED_FILE, index=False)
    return df


def to_current_units(pred: pd.DataFrame, prices: dict) -> tuple[pd.DataFrame, dict]:
    """The data folder can be re-adjusted for a new dividend while the model runs: the whole adjusted
    history is then multiplied by one constant. The model z-scores each window, so its paths scale by
    the same constant. Bring every cached row to today's units (paths x f, close_t = today's close_t)
    and refuse if the factor is not one constant per symbol. `scalecheck` reruns the model to confirm."""
    out, factors = pred.copy(), {}
    for sym, g in pred.groupby("symbol"):
        now = prices[sym]["close"].reindex(pd.DatetimeIndex(g["asof"])).to_numpy()
        f = now / g["close_t"].to_numpy()
        if np.nanmax(f) / np.nanmin(f) - 1 > 1e-6:
            raise SystemExit(f"{sym}: cache and data differ by more than one constant factor; rerun predict")
        factors[sym] = {"min": float(f.min()), "max": float(f.max()), "rescaled": bool(abs(f.mean() - 1) > 1e-9)}
        out.loc[g.index, "paths"] = pd.Series([np.asarray(pp) * ff for pp, ff in zip(g["paths"], f)], index=g.index)
        out.loc[g.index, "close_t"] = now
    return out, factors


def predict(part: int = 0, parts: int = 1):
    """Run with `predict i/k` in k processes to share the CPU; each process takes every k-th job and
    keeps its own shard files, so nothing is written twice. The last process also draws the fans."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pred, torch = load_predictor()
    prices = {s: load(s) for s in SYMBOLS}
    t0, made = time.time(), 0
    only = os.environ.get("X01_ONLY")          # e.g. "GLD,control,0": run just this job (to use an idle core)
    for j, (symbol, seg, seed, n) in enumerate(jobs()):
        if only:
            if f"{symbol},{seg},{seed}" != only:
                continue
        elif j % parts != part:
            continue
        px = prices[symbol]
        path = shard(symbol, seg, seed)
        rows = _load_rows(path)
        done = {str(pd.Timestamp(r["asof"]).date()) for r in rows}
        days = ks.segment_days(px.index, TEST_START, TEST_END, CONTROL_END)[seg]
        todo = [(a, b) for a, b in days if str(a.date()) not in done]
        print(f"{symbol} {seg} seed{seed} paths{n}: {len(days) - len(todo)} cached, {len(todo)} to go", flush=True)
        for k, (asof, target) in enumerate(todo):
            p = draw_paths(pred, torch, px, asof, [target], n, seed, symbol)[:, 0]
            rows.append({"symbol": symbol, "segment": seg, "seed": seed, "asof": asof, "target": target,
                         "close_t": float(px["close"].loc[asof]), "n_paths": n, "paths": p.tolist()})
            made += 1
            if made % 25 == 0 or k == len(todo) - 1:
                _save(rows, path)
                el = time.time() - t0
                print(f"  {symbol} {seg} seed{seed} {k + 1}/{len(todo)}  {el / made:.2f}s/day  {el / 60:.1f} min", flush=True)
    if (only and only != "fans") or (not only and part != parts - 1):   # X01_ONLY=fans: just the stage below
        print("predict part done", flush=True)
        return

    # look-ahead check: same day, everything after asof scrambled -> paths must be identical
    look = _load_rows(LOOK_FILE)
    if not look:
        px = prices[MAIN]
        days = ks.segment_days(px.index, TEST_START, TEST_END, CONTROL_END)["test"]
        picks = [days[0], days[len(days) // 2], days[-1]][:LOOKAHEAD_DAYS]
        rng = np.random.default_rng(1)
        for asof, target in picks:
            bad = px.copy()
            i = bad.index.get_loc(asof)
            m = rng.uniform(0.5, 1.5, len(bad) - i - 1)
            for c in ("open", "high", "low", "close", "volume"):
                bad.iloc[i + 1:, bad.columns.get_loc(c)] = bad[c].to_numpy()[i + 1:] * m
            # same path count and seed as the cached main run, so `clean` must also equal the cache
            a = draw_paths(pred, torch, px, asof, [target], MAIN_PATHS_RUN, 0, MAIN)[:, 0]
            b = draw_paths(pred, torch, bad, asof, [target], MAIN_PATHS_RUN, 0, MAIN)[:, 0]
            look.append({"asof": asof, "target": target, "clean": a.tolist(), "scrambled": b.tolist()})
        _save(look, LOOK_FILE)
        print("look-ahead runs saved", flush=True)

    # multi-step fans: 20 bars hidden, 20 bars drawn
    fans = _load_rows(FAN_FILE)
    fdone = {str(pd.Timestamp(r["asof"]).date()) for r in fans}
    px = prices[MAIN]
    grid, last = fan_asofs(px)
    for kind, asof in [("last20", last)] + [("grid", a) for a in grid]:
        if str(asof.date()) in fdone:
            continue
        i = px.index.get_loc(asof)
        fut = px.index[i + 1: i + 1 + FAN_LEN]
        p = draw_paths(pred, torch, px, asof, fut, N_PATHS, 0, MAIN)
        fans.append({"kind": kind, "asof": asof, "close_t": float(px["close"].loc[asof]),
                     "dates": [str(d.date()) for d in fut], "paths": p.ravel().tolist(),
                     "n_paths": N_PATHS, "horizon": FAN_LEN})
        fdone.add(str(asof.date()))
        _save(fans, FAN_FILE)
        print(f"fan {kind} {asof.date()} done  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("predict: all done", flush=True)

# ================================================================ statistics (from the cache only)
COST_BPS = 5.0
COSTS = [0, 2, 5, 10, 20]
THRESHOLDS = [0.5, 0.55, 0.6]
PATH_COUNTS = [8, 32, 64]
LOOKS_RIGHT = 0.8     # a fan "looks right" if the real closes stay inside its 10-90 band on >= 80% of the 20 days


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def frames_from(pred: pd.DataFrame, prices: dict, n_paths=N_PATHS, seed=0) -> dict:
    """{(symbol, segment): daily frame}. SPY's test run has 64 paths; the main result uses its first 32."""
    out = {}
    for (sym, seg, sd), g in pred.groupby(["symbol", "segment", "seed"]):
        if sd != seed:
            continue
        out[(sym, seg)] = ks.daily_frame(g.sort_values("asof"), prices[sym], n_paths)
    return out


def bars_for(px, df):
    return px.loc[df.index[0]: df["target"].iloc[-1]]


def perf(bt, rf) -> dict:
    w = bt["weight"]
    orders = int((w.diff().fillna(w.iloc[0]).abs() > 1e-12).sum())
    rt = metrics.trades(w, bt["ret"])
    return {"start": str(bt.index[0].date()), "end": str(bt.index[-1].date()),
            "final_multiple": float(bt["equity"].iloc[-1] / bt["equity"].iloc[0]), "cagr": metrics.cagr(bt["equity"]),
            "ann_vol": metrics.ann_vol(bt["ret"]), "sharpe": metrics.sharpe(bt["ret"], None if rf is None else rf.reindex(bt.index)),
            "max_drawdown": metrics.max_drawdown(bt["equity"]), "exposure": float(w.mean()),
            "orders": orders, "round_trips": int(len(rt)),
            "win_rate": float((rt["return"] > 0).mean()) if len(rt) else None,
            "total_cost": float(bt["cost"].sum()) if "cost" in bt else None}


def rule_bt(px, df, rf, threshold=0.5, cost=COST_BPS):
    bars = bars_for(px, df)
    tgt = ks.target_weights(df, threshold).reindex(bars.index).fillna(0.0)
    return engine.vectorized(bars, tgt, cost, rf), tgt, bars


def hold_bt(px, df, rf, cost=COST_BPS):
    bars = bars_for(px, df)
    return engine.vectorized(bars, pd.Series(1.0, index=bars.index), cost, rf)


def with_start_row(bt, df):
    """close_to_close is indexed by target day; add the first decision day (equity 1) so that
    final_multiple and CAGR cover exactly the same span as the next-open backtest."""
    first = pd.DataFrame({"weight": [0.0], "ret": [0.0], "equity": [1.0]}, index=[df.index[0]])
    out = pd.concat([first, bt])
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


def ideal_bt(df, rf, threshold=0.5, cost=COST_BPS):
    return with_start_row(ks.close_to_close(df, ks.target_weights(df, threshold), cost, rf), df)


def trading(px, df, rf, threshold=0.5, cost=COST_BPS) -> dict:
    bt, tgt, bars = rule_bt(px, df, rf, threshold, cost)
    h = hold_bt(px, df, rf, cost)
    i = ideal_bt(df, rf, threshold, cost)
    ih = with_start_row(ks.close_to_close(df, pd.Series(1.0, index=df.index), cost, rf), df)
    return {"strategy": perf(bt, rf), "buyhold": perf(h, rf),
            "ideal_close": {**perf(i.assign(cost=0.0), rf), "note": "trades at the same close the signal needs: not possible, upper bound only"},
            "ideal_close_buyhold": perf(ih.assign(cost=0.0), rf)}


def independent_ledger(px, df, rf, threshold=0.5, cost_bps=COST_BPS) -> dict:
    """A separate account: plain lists, cash and shares, no engine.py and no kronos_signal call.
    Signal from the raw P(up) numbers; order at the next open; interest on idle cash."""
    bars = px.loc[df.index[0]: df["target"].iloc[-1]]
    dates = list(bars.index)
    o, c = bars["open"].tolist(), bars["close"].tolist()
    r = rf.reindex(bars.index).fillna(0.0).tolist()
    pu = {d: v for d, v in zip(df.index, df["p_up"].tolist())}
    cash, shares, want, orders = 1.0, 0.0, 0.0, 0
    for i, d in enumerate(dates):
        value = cash + shares * o[i]
        have = 1.0 if shares > 0 else 0.0
        if want != have:
            fee = value * cost_bps / 1e4
            value -= fee
            if want == 1.0:
                shares, cash = value / o[i], 0.0
            else:
                shares, cash = 0.0, value
            orders += 1
        if i > 0 and shares == 0:          # equity is measured from the close of the first decision day
            cash *= 1 + r[i]
        want = 1.0 if pu.get(d, 0.0) > threshold else 0.0
    return {"final_multiple": cash + shares * c[-1], "orders": orders}


def independent_hits(pred_rows: pd.DataFrame, px, n_paths) -> dict:
    """Recount the direction hits from the raw cached paths with plain Python."""
    close = dict(zip(px.index, px["close"].tolist()))
    k = n = 0
    for r in pred_rows.itertuples(index=False):
        paths = list(r.paths)[:n_paths]
        m = sum(paths) / len(paths)
        a = close[r.target]
        if a == r.close_t:
            continue
        n += 1
        k += (m > r.close_t) == (a > r.close_t)
    return {"days": n, "hits": k, "hit_rate": k / n}


def symbol_block(df) -> dict:
    return {"direction": ks.direction_table(df), "ic": ks.ic_table(df), "calibration": ks.calibration(df),
            "coverage": ks.coverage(df), "mean_pull": ks.mean_pull(df),
            "days": int(len(df)), "first_target": str(df["target"].iloc[0].date()), "last_target": str(df["target"].iloc[-1].date())}


def annual_table(bt, h) -> pd.DataFrame:
    y = lambda b: (1 + b["ret"]).groupby(b.index.year).prod() - 1
    out = pd.DataFrame({"strategy": y(bt), "buyhold": y(h)})
    out["excess"] = out["strategy"] - out["buyhold"]
    out.index.name = "year"
    return out


def fan_block(px) -> dict:
    fans = pd.read_parquet(FAN_FILE)
    rows = []
    for r in fans.itertuples(index=False):
        paths = np.asarray(r.paths, float).reshape(r.n_paths, r.horizon)
        actual = px["close"].loc[pd.DatetimeIndex(r.dates)].to_numpy(float)
        st = ks.fan_stats(paths, r.close_t, actual)
        st.update({"kind": r.kind, "asof": str(pd.Timestamp(r.asof).date()), "last_date": r.dates[-1],
                   "looks_right": st["share_days_inside_10_90"] >= LOOKS_RIGHT, "z": ks.window_z(px, r.asof)})
        rows.append(st)
    t = pd.DataFrame(rows)
    g = t[t.kind == "grid"]
    last = t[t.kind == "last20"].iloc[0].to_dict()
    return {"last20": last,
            "grid": {"n": int(len(g)), "looks_right": int(g["looks_right"].sum()),
                     "direction_right": int(g["direction_right"].sum()),
                     "end_inside_10_90": int(g["end_inside_10_90"].sum()),
                     "mean_share_days_inside": float(g["share_days_inside_10_90"].mean()),
                     "median_abs_end_error_pct": float(g["end_error_pct"].abs().median()),
                     "mean_abs_end_error_pct": float(g["end_error_pct"].abs().mean()),
                     "looks_right_and_direction_wrong": int((g["looks_right"] & ~g["direction_right"]).sum()),
                     "always_up_direction_right": int((g["end_actual_ret"] > 0).sum()),
                     "median_end_band_width_pct": float(g["end_band_width_pct"].median()),
                     "median_abs_end_actual_ret": float(g["end_actual_ret"].abs().median()),
                     # descriptive, added after seeing the fans: do the 20-day fans lean the same way as the 1-day ones?
                     "mean_end_pred_ret": float(g["end_pred_ret"].mean()), "mean_end_actual_ret": float(g["end_actual_ret"].mean()),
                     "share_end_pred_down": float((g["end_pred_ret"] < 0).mean()),
                     "corr_end_pred_z": ks.corr(g["end_pred_ret"], g["z"]), "corr_end_actual_z": ks.corr(g["end_actual_ret"], g["z"])},
            "rows": rows}


def fan_cards(px):
    """Local-only file for the video team: every fan relative to close_t = 1 (quantiles + real)."""
    fans = pd.read_parquet(FAN_FILE)
    out = []
    for r in fans.itertuples(index=False):
        paths = np.asarray(r.paths, float).reshape(r.n_paths, r.horizon) / r.close_t
        actual = px["close"].loc[pd.DatetimeIndex(r.dates)].to_numpy(float) / r.close_t
        for h in range(r.horizon):
            out.append({"kind": r.kind, "asof": str(pd.Timestamp(r.asof).date()), "step": h + 1, "date": r.dates[h],
                        "q10": np.quantile(paths[:, h], 0.1), "q50": np.quantile(paths[:, h], 0.5),
                        "q90": np.quantile(paths[:, h], 0.9), "mean": paths[:, h].mean(), "actual": actual[h],
                        **{f"path{j:02d}": paths[j, h] for j in range(min(8, r.n_paths))}})
    pd.DataFrame(out).to_csv(OUT / "fan_cards.csv", index=False, float_format="%.6f")


def main():
    prices = {s: load(s) for s in SYMBOLS}
    pred, factors = to_current_units(merge_cache(), prices)
    rf = data.daily_rate(data.load_rate("%3MTCM"), prices[MAIN].index)
    px = prices[MAIN]
    F = frames_from(pred, prices)
    res = {"rules": {"model": MODEL_NAME, "tokenizer": TOKENIZER_NAME, "context_bars": ks.CONTEXT, "paths": N_PATHS,
                     "temperature": T, "top_p": TOP_P, "top_k": TOP_K, "seeds": SEEDS, "symbols": SYMBOLS, "main": MAIN,
                     "test_targets": [TEST_START, TEST_END], "control_end": CONTROL_END, "cost_bps": COST_BPS,
                     "threshold": 0.5, "fan_len": FAN_LEN, "fan_step": FAN_STEP, "looks_right_rule": LOOKS_RIGHT,
                     "training_data_ends": "2024-06", "device": "cpu"}}

    # ---- forecasting quality, per symbol and segment
    res["forecast"] = {seg: {s: symbol_block(F[(s, seg)]) for s in SYMBOLS} for seg in ("test", "control")}
    res["cross_section_rank_ic"] = {seg: ks.cross_sectional_rank_ic({s: F[(s, seg)] for s in SYMBOLS}) for seg in ("test", "control")}
    pooled = {seg: pd.concat([F[(s, seg)] for s in SYMBOLS]) for seg in ("test", "control")}
    res["pooled"] = {seg: {"calibration": ks.calibration(pooled[seg]), "coverage": ks.coverage(pooled[seg]),
                           "direction": ks.direction_table(pooled[seg]), "mean_pull": ks.mean_pull(pooled[seg])} for seg in pooled}

    # ---- trading, SPY
    main_df = F[(MAIN, "test")]
    res["trading"] = {seg: trading(px, F[(MAIN, seg)], rf) for seg in ("test", "control")}
    bt, tgt, bars = rule_bt(px, main_df, rf)
    h = hold_bt(px, main_df, rf)
    annual_table(bt, h).to_csv(OUT / "annual.csv", float_format="%.6f")
    # local-only files for the video team (not whitelisted in .gitignore): no price levels, only
    # equity curves that start at 1 and daily returns / model outputs
    pd.DataFrame({"strategy": bt["equity"] / bt["equity"].iloc[0], "buyhold": h["equity"] / h["equity"].iloc[0],
                  "weight": bt["weight"]}).to_csv(OUT / "curves.csv", float_format="%.6f")
    main_df[["target", "p_up", "pred_ret", "actual_ret", "z"]].assign(said_up=lambda d: d["p_up"] > 0.5,
                                                                      went_up=lambda d: d["actual_ret"] > 0).to_csv(
        OUT / "daily_signals.csv", float_format="%.6f")
    tr = metrics.trades(bt["weight"], bt["ret"])
    tr.to_csv(OUT / "trades.csv", index=False, float_format="%.6f")
    res["trades_summary"] = {"round_trips": int(len(tr)), "median_days": float(tr.apply(lambda r: len(bt.loc[r["entry"]:r["exit"]]) - (0 if r["open"] else 1), axis=1).median()) if len(tr) else None,
                             "win_rate": float((tr["return"] > 0).mean()) if len(tr) else None}

    # ---- checks
    rec = checks.reconcile(bars, tgt, COST_BPS, rf)
    rec_h = checks.reconcile(bars, pd.Series(1.0, index=bars.index), COST_BPS, rf)
    ind = independent_ledger(px, main_df, rf)
    s_perf = res["trading"]["test"]["strategy"]
    ihits = independent_hits(pred[(pred.symbol == MAIN) & (pred.segment == "test") & (pred.seed == 0)].sort_values("asof"), px, N_PATHS)
    look = pd.read_parquet(LOOK_FILE)
    cached = pred[(pred.symbol == MAIN) & (pred.seed == 0)].set_index("asof")
    lrows = []
    for r in look.itertuples(index=False):
        a, b = np.asarray(r.clean), np.asarray(r.scrambled)
        cpaths = np.asarray(cached.loc[pd.Timestamp(r.asof), "paths"])
        close_t = float(cached.loc[pd.Timestamp(r.asof), "close_t"])
        lrows.append({"asof": str(pd.Timestamp(r.asof).date()), "same_after_scramble": bool(np.array_equal(a, b)),
                      # a rerun with a different thread count differs only in float32 rounding of the decoder
                      "same_as_cache": bool(np.allclose(a, cpaths, rtol=1e-6, atol=0)),
                      "max_rel_diff_vs_cache": float(np.max(np.abs(a / cpaths - 1))),
                      "same_p_up_as_cache": bool((a > close_t).mean() == (cpaths > close_t).mean())})
    win_ok = all(ks.asof_window(px, d).index[-1] == d for d in main_df.index[:: max(1, len(main_df) // 20)])
    res["checks"] = {
        "reconcile_strategy": rec, "reconcile_buyhold": rec_h,
        "independent_ledger": {**ind, "engine_final_multiple": s_perf["final_multiple"], "engine_orders": s_perf["orders"],
                               "relative_gap": abs(ind["final_multiple"] / s_perf["final_multiple"] - 1),
                               "passed": abs(ind["final_multiple"] / s_perf["final_multiple"] - 1) < 1e-9 and ind["orders"] == s_perf["orders"]},
        "independent_hits": {**ihits, "main_hits": res["forecast"]["test"][MAIN]["direction"]["model_hits"],
                             "passed": ihits["hits"] == res["forecast"]["test"][MAIN]["direction"]["model_hits"]},
        "lookahead": {"days": lrows, "passed": all(x["same_after_scramble"] and x["same_as_cache"] and x["same_p_up_as_cache"] for x in lrows),
                      "how": "prices and volume after the decision day multiplied by random 0.5-1.5; the real model rerun twice in one "
                             "process (clean vs scrambled must be bit-identical) and compared with the cache (float32 precision)"},
        "window_ends_at_asof": win_ok,
        "cache_vs_data": {"rows": int(len(pred)), "factor_per_symbol": factors,
                          "note": "the data folder was refreshed during the run: bars after 2026-10-02 were added (ignored here) "
                                  "and TLT's adjusted history was rescaled by one constant for a new dividend; TLT rows are "
                                  "brought to today's units, and `scalecheck` reruns the model on today's data to confirm"},
        "scalecheck": scalecheck_block(pred),
        "split_2010": "not applicable: the split that matters here is the model's training cut-off (June 2024), so test vs control replaces before/after 2010",
    }

    # ---- seeds and path counts (SPY, test)
    seeds = {}
    for sd in SEEDS:
        g = pred[(pred.symbol == MAIN) & (pred.segment == "test") & (pred.seed == sd)].sort_values("asof")
        if g.empty:
            continue
        d = ks.daily_frame(g, px, N_PATHS)
        t = trading(px, d, rf)["strategy"]
        seeds[str(sd)] = {"hit_rate": ks.direction_table(d)["model_hit_rate"], "rank_ic": ks.ic_table(d)["rank_ic"],
                          "coverage": ks.coverage(d)["inside_10_90"], "corr_pred_z": ks.mean_pull(d)["corr_pred_z"],
                          "cagr": t["cagr"], "max_drawdown": t["max_drawdown"], "orders": t["orders"],
                          "exposure": t["exposure"], "signal": ks.target_weights(d).rename(sd)}
    sig = pd.concat([v.pop("signal") for v in seeds.values()], axis=1)
    res["seeds"] = {"runs": seeds, "days_all_seeds_agree": float((sig.nunique(axis=1) == 1).mean())}
    paths = {}
    g0 = pred[(pred.symbol == MAIN) & (pred.segment == "test") & (pred.seed == 0)].sort_values("asof")
    for n in PATH_COUNTS:
        d = ks.daily_frame(g0, px, n)
        t = trading(px, d, rf)["strategy"]
        paths[str(n)] = {"hit_rate": ks.direction_table(d)["model_hit_rate"], "rank_ic": ks.ic_table(d)["rank_ic"],
                         "coverage": ks.coverage(d)["inside_10_90"], "cagr": t["cagr"], "max_drawdown": t["max_drawdown"],
                         "orders": t["orders"], "exposure": t["exposure"]}
    res["path_counts"] = paths

    # ---- health check: costs, thresholds (after seeing results), halves
    sweep = []
    for c in COSTS:
        tt = trading(px, main_df, rf, cost=c)
        sweep.append({"cost_bps": c, "strategy_cagr": tt["strategy"]["cagr"], "buyhold_cagr": tt["buyhold"]["cagr"],
                      "ideal_close_cagr": tt["ideal_close"]["cagr"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "cost_sweep.csv", index=False, float_format="%.6f")
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")
    grid = []
    for n in PATH_COUNTS:
        d = ks.daily_frame(g0, px, n)
        for th in THRESHOLDS:
            t = trading(px, d, rf, threshold=th)
            grid.append({"paths": n, "threshold": th, "cagr": t["strategy"]["cagr"], "sharpe": t["strategy"]["sharpe"],
                         "max_drawdown": t["strategy"]["max_drawdown"], "orders": t["strategy"]["orders"],
                         "exposure": t["strategy"]["exposure"], "buyhold_cagr": t["buyhold"]["cagr"]})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "param_grid.csv", index=False, float_format="%.6f")
    res["param_grid"] = {"note": "tried after seeing the main result; not a conclusion", "rows": grid.to_dict("records"),
                         "n": int(len(grid)), "n_beat_buyhold_cagr": int((grid["cagr"] > grid["buyhold_cagr"]).sum())}
    half = len(main_df) // 2
    halves = {}
    for name, d in (("first", main_df.iloc[:half]), ("second", main_df.iloc[half:])):
        t = trading(px, d, rf)
        halves[name] = {"first_target": str(d["target"].iloc[0].date()), "last_target": str(d["target"].iloc[-1].date()),
                        "hit_rate": ks.direction_table(d)["model_hit_rate"], "always_up": ks.direction_table(d)["always_up_hit_rate"],
                        "rank_ic": ks.ic_table(d)["rank_ic"], "strategy_cagr": t["strategy"]["cagr"], "buyhold_cagr": t["buyhold"]["cagr"],
                        "strategy_max_drawdown": t["strategy"]["max_drawdown"], "buyhold_max_drawdown": t["buyhold"]["max_drawdown"]}
    res["halves"] = halves
    res["idle_cash_zero_rate"] = perf(engine.vectorized(bars, tgt, COST_BPS, None), None)

    # ---- other symbols, same rule (test segment)
    res["other_assets"] = {s: {seg: trading(prices[s], F[(s, seg)], rf) for seg in ("test",)} for s in SYMBOLS if s != MAIN}

    # ---- fans
    if FAN_FILE.exists():
        res["fans"] = fan_block(px)
        fan_cards(px)

    # ---- diagnostic added after seeing the mean pull (python run_x01.py roundtrip)
    if (OUT / "roundtrip.parquet").exists():
        rt = pd.read_parquet(OUT / "roundtrip.parquet").set_index("asof").reindex(main_df.index)
        d = main_df.copy()
        d["rt_err"] = rt["close_t_roundtrip"] / d["close_t"] - 1
        g = pred[(pred.symbol == MAIN) & (pred.segment == "test") & (pred.seed == 0)].set_index("asof").reindex(d.index)
        p32 = np.stack([np.asarray(x)[:N_PATHS] for x in g["paths"]])
        d["p_up_vs_rt"] = (p32 > rt["close_t_roundtrip"].to_numpy()[:, None]).mean(axis=1)
        d["pred_vs_rt"] = p32.mean(axis=1) / rt["close_t_roundtrip"].to_numpy() - 1
        hi = d[d["z"] > 1]
        res["roundtrip"] = {"note": "added after seeing the mean-pull result; not a conclusion of the rule",
                            "days": int(len(d)), "corr_err_z": ks.corr(d["rt_err"], d["z"]),
                            "mean_err": float(d["rt_err"].mean()), "median_abs_err": float(d["rt_err"].abs().median()),
                            "mean_err_z_above_1": float(hi["rt_err"].mean()), "days_z_above_1": int(len(hi)),
                            "mean_pred_ret": float(d["pred_ret"].mean()),
                            "corr_pred_vs_rt_z": ks.corr(d["pred_vs_rt"], d["z"]),
                            "mean_pred_vs_rt": float(d["pred_vs_rt"].mean()),
                            "share_up_vs_real_close": float((d["p_up"] > 0.5).mean()),
                            "share_up_vs_roundtrip": float((d["p_up_vs_rt"] > 0.5).mean()),
                            "hit_rate_vs_roundtrip": float((((d["pred_vs_rt"] > 0) == (d["actual_ret"] > 0))[d["actual_ret"] != 0]).mean())}

    res["paper"] = PAPER
    res["counts"] = {"forecast_days": int(len(pred)), "paths_drawn": int(pred["n_paths"].sum()),
                     "fan_runs": int(len(pd.read_parquet(FAN_FILE))) if FAN_FILE.exists() else 0}
    res["data"] = {s: {"first": str(prices[s].index[0].date()), "last": str(prices[s].index[-1].date())} for s in SYMBOLS}
    (OUT / "results.json").write_text(json.dumps(clean(res), ensure_ascii=False, indent=1))
    print(json.dumps(clean({"hit": res["forecast"]["test"][MAIN]["direction"], "trading": res["trading"]["test"]["strategy"],
                            "bh": res["trading"]["test"]["buyhold"]}), indent=1))



# ================================================================ diagnostic added after seeing the mean-pull result
ROUND_FILE = OUT / "roundtrip.parquet"


def roundtrip():
    """No sampling: normalise each SPY test window exactly as KronosPredictor does, encode it into
    tokens and decode it straight back. If even TODAY's reconstructed close leans towards the
    window average, part of the pull happens before any forecasting. Cached like the predictions."""
    pred, torch = load_predictor()
    px = load(MAIN)
    days = ks.segment_days(px.index, TEST_START, TEST_END, CONTROL_END)["test"]
    rows = []
    for k in range(0, len(days), 32):
        chunk = days[k:k + 32]
        xs, stats = [], []
        for asof, _ in chunk:
            w = ks.asof_window(px, asof).copy()
            w["amount"] = w["volume"] * w[["open", "high", "low", "close"]].mean(axis=1)
            x = w[["open", "high", "low", "close", "volume", "amount"]].to_numpy(np.float32)
            m, sd = x.mean(axis=0), x.std(axis=0)
            xs.append(np.clip((x - m) / (sd + 1e-5), -pred.clip, pred.clip))
            stats.append((m, sd))
        with torch.no_grad():
            t = torch.from_numpy(np.stack(xs))
            z = pred.tokenizer.decode(pred.tokenizer.encode(t, half=True), half=True).numpy()
        for (asof, _), zz, (m, sd) in zip(chunk, z, stats):
            back = zz[-1] * (sd + 1e-5) + m
            rows.append({"asof": asof, "close_t": float(px["close"].loc[asof]), "close_t_roundtrip": float(back[3])})
    pd.DataFrame(rows).to_parquet(ROUND_FILE, index=False)
    print("roundtrip saved", len(rows), flush=True)


# ================================================================ check added after the TLT data was re-adjusted mid-run
SCALE_FILE = OUT / "scalecheck.parquet"


def scalecheck():
    """Rerun the model on TODAY's data for 3 days of every symbol whose cache needed rescaling."""
    pred, torch = load_predictor()
    prices = {s: load(s) for s in SYMBOLS}
    cache = merge_cache()
    rows = []
    for sym in SYMBOLS:
        g = cache[(cache.symbol == sym) & (cache.seed == 0)].sort_values("asof")
        f = prices[sym]["close"].reindex(pd.DatetimeIndex(g["asof"])).to_numpy() / g["close_t"].to_numpy()
        if abs(f.mean() - 1) < 1e-9:
            continue
        for k in (0, len(g) // 2, len(g) - 1):
            r = g.iloc[k]
            p = draw_paths(pred, torch, prices[sym], r["asof"], [r["target"]], int(r["n_paths"]), 0, sym)[:, 0]
            rows.append({"symbol": sym, "segment": r["segment"], "asof": r["asof"], "rerun": p.tolist()})
    pd.DataFrame(rows).to_parquet(SCALE_FILE, index=False)
    print("scalecheck saved", len(rows), flush=True)


def scalecheck_block(pred) -> dict:
    if not SCALE_FILE.exists():
        return {"days": [], "passed": None}
    sc = pd.read_parquet(SCALE_FILE)
    out = []
    for r in sc.itertuples(index=False):
        day = pd.DatetimeIndex(pred["asof"]).normalize() == pd.Timestamp(r.asof).normalize()
        g = pred[(pred.symbol == r.symbol) & (pred.seed == 0) & day].iloc[0]
        a, b = np.asarray(r.rerun), np.asarray(g["paths"])
        out.append({"symbol": r.symbol, "asof": str(pd.Timestamp(r.asof).date()),
                    "max_rel_diff": float(np.max(np.abs(a / b - 1))),
                    "same_p_up": bool((a > g["close_t"]).mean() == (b > g["close_t"]).mean())})
    return {"days": out, "passed": all(x["max_rel_diff"] < 1e-5 and x["same_p_up"] for x in out)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        i, k = (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/")
        predict(int(i), int(k))
    elif len(sys.argv) > 1 and sys.argv[1] == "roundtrip":
        roundtrip()
    elif len(sys.argv) > 1 and sys.argv[1] == "scalecheck":
        scalecheck()
    else:
        main()
