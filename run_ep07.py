"""Episode 7: MACD(12,26,9) on SPY, against the moving-average cross of episode 1.
Writes every number to results/ep07/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep07.py   # data format: see backtest/data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import checks, data, engine, metrics, research
from run_ep03 import episodes
from strategies import ma_cross, macd

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ep07"
SYMBOL, FAST, SLOW, SIGNAL, COST_BPS, WARMUP = "SPY", 12, 26, 9, 5.0, 60
MA_FAST, MA_SLOW = 20, 60
SPLIT = "2010-01-01"
OTHERS = ["QQQ", "IWM", "TLT", "GLD"]
MIN_DD = -0.15
GRID_WARMUP = 300      # the grid window starts where the parameter grid of episode 1 starts; positions carry in, as they do there
FASTS, SLOWS, SIGNALS = [5, 8, 12, 16, 20], [20, 26, 35, 50, 100], [5, 9, 14]
MA_FASTS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
MA_SLOWS = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300]


def stats(bt, rf):
    return {"sharpe": metrics.sharpe(bt["ret"], rf.loc[bt.index]), "cagr": metrics.cagr(bt["equity"]),
            "max_drawdown": metrics.max_drawdown(bt["equity"]), "ann_vol": metrics.ann_vol(bt["ret"]),
            "final_multiple": float(bt["equity"].iloc[-1] / bt["equity"].iloc[0]), "exposure": float(bt["weight"].mean()),
            "trades": int(len(metrics.trades(bt["weight"], bt["ret"])))}


def run(px, tgt, rf, start, cost=COST_BPS):
    return research.window(engine.vectorized(px, tgt, cost, rf), start)


def independent_replay(px: pd.DataFrame, rf: pd.Series, fast: int, slow: int, signal: int, warmup: int,
                       cost_bps: float) -> dict:
    """A separate ledger: no call into engine.py or macd.py. Plain lists, one pass, cash and shares.
    Accounts start at the close of day warmup - 1, so compare with equity[-1] / equity[0]."""
    o, c, r = px["open"].tolist(), px["close"].tolist(), rf.reindex(px.index).fillna(0.0).tolist()
    fee = cost_bps / 1e4
    af, asl, asi = 2 / (fast + 1), 2 / (slow + 1), 2 / (signal + 1)
    ef, es = c[0], c[0]
    dea = 0.0
    prev_dif, prev_dea = None, None
    cash, shares, order, entries = 1.0, 0.0, None, 0
    first_valid = warmup - 1
    for i in range(len(c)):
        if i > first_valid:
            if order == "buy":
                shares, cash, entries = cash * (1 - fee) / o[i], 0.0, entries + 1
            elif order == "sell":
                cash, shares = shares * o[i] * (1 - fee), 0.0
            order = None
            if shares == 0.0:
                cash *= 1 + r[i]
        if i > 0:
            ef, es = af * c[i] + (1 - af) * ef, asl * c[i] + (1 - asl) * es
        dif = ef - es
        dea = dif if i == 0 else asi * dif + (1 - asi) * dea
        if i > first_valid:
            if prev_dif <= prev_dea and dif > dea and shares == 0.0:
                order = "buy"
            elif prev_dif >= prev_dea and dif < dea and shares > 0.0:
                order = "sell"
        if i >= first_valid:
            prev_dif, prev_dea = dif, dea
    return {"final_multiple": float(cash + shares * c[-1]), "entries": entries}


def first_exit_after(weight: pd.Series, peak) -> tuple[pd.Timestamp | None, bool]:
    """First real exit after the peak, and whether the strategy was already in cash on the peak day."""
    w = weight.loc[peak:]
    flat = float(w.iloc[0]) == 0.0
    out = w.index[(w == 0) & (w.shift(1) > 0)]
    return (out[0] if len(out) else None), flat


def episode_table(h, m, a):
    rows = []
    pos = {d: i for i, d in enumerate(h.index)}
    for e in episodes(h["equity"], MIN_DD):
        end = e["recovered"] if e["recovered"] is not None else h.index[-1]
        def depth(bt):
            d = bt["equity"].loc[e["peak"]:e["trough"]]
            return float(d.iloc[-1] / d.iloc[0] - 1)
        def trips(bt):
            w = bt["weight"].loc[e["peak"]:end]
            return int(((w == 0) & (w.shift(1) > 0)).sum())
        (fm, fm_flat), (fa, fa_flat) = first_exit_after(m["weight"], e["peak"]), first_exit_after(a["weight"], e["peak"])
        both_in = not fm_flat and not fa_flat and fm is not None and fa is not None
        rows.append({"peak": str(e["peak"].date()), "trough": str(e["trough"].date()),
                     "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
                     "buyhold": e["depth"], "macd": depth(m), "ma": depth(a),
                     "macd_first_exit": str(fm.date()) if fm is not None else None,
                     "ma_first_exit": str(fa.date()) if fa is not None else None,
                     "macd_flat_at_peak": fm_flat, "ma_flat_at_peak": fa_flat,
                     "macd_lead_days": pos[fa] - pos[fm] if both_in else None,
                     "macd_exits": trips(m), "ma_exits": trips(a),
                     "macd_cycle": float(m["equity"].loc[end] / m["equity"].loc[e["peak"]] - 1),
                     "ma_cycle": float(a["equity"].loc[end] / a["equity"].loc[e["peak"]] - 1),
                     "buyhold_cycle": float(h["equity"].loc[end] / h["equity"].loc[e["peak"]] - 1)})
    return rows


def macd_grid(px, rf, start, end=None):
    rows = []
    for f in FASTS:
        for sl in SLOWS:
            if f >= sl:
                continue
            for si in SIGNALS:
                full = engine.vectorized(px, macd.target(px, f, sl, si), COST_BPS, rf)
                rows.append({"fast": f, "slow": sl, "signal": si, **stats(research.window(full, start, end), rf)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    start = px.index[WARMUP - 1]
    e1 = json.loads((ROOT / "results" / "ep01" / "results.json").read_text())
    res: dict = {"symbol": SYMBOL, "rules": {"fast": FAST, "slow": SLOW, "signal": SIGNAL, "warmup": WARMUP, "cost_bps_per_side": COST_BPS,
                                             "fill": "next open", "cash": "3-month Treasury (constant maturity)"}}
    res["data_quality"] = data.validate(px)

    tgt = macd.target(px, FAST, SLOW, SIGNAL, warmup=WARMUP)
    ma_tgt = ma_cross.target(px, MA_FAST, MA_SLOW)
    bench = pd.Series(0.0, index=px.index)
    bench.loc[start:] = 1.0
    fn = lambda mode: (lambda p: macd.target(p, FAST, SLOW, SIGNAL, mode=mode, warmup=WARMUP))

    res["checks"] = {"lookahead": checks.lookahead_check(fn("cross"), px),
                     "future_change": macd.future_change_check(fn("cross"), px),
                     "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
                     "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf),
                     "lookahead_variants": {m: checks.lookahead_check(fn(m), px)["passed"] for m in ("zero_above", "dif_zero")},
                     "future_change_variants": {m: macd.future_change_check(fn(m), px)["passed"] for m in ("zero_above", "dif_zero")}}

    s, h, a = run(px, tgt, rf, start), run(px, bench, rf, start), run(px, ma_tgt, rf, start)
    years = (s.index[-1] - s.index[0]).days / 365.25
    res["strategy"], res["buyhold"], res["ma_cross"] = metrics.summary(s, rf), metrics.summary(h, rf), metrics.summary(a, rf)
    res["years"] = years
    tr_raw = metrics.trades(s["weight"], s["ret"])
    tr_raw.to_csv(OUT / "trades.csv", index=False)
    res["trade_count"] = int(len(tr_raw))

    ind = independent_replay(px, rf, FAST, SLOW, SIGNAL, WARMUP, COST_BPS)
    engine_mult = float(s["equity"].iloc[-1] / s["equity"].iloc[0])
    res["independent"] = {**ind, "engine_final_multiple": engine_mult, "engine_entries": res["trade_count"],
                          "relative_gap": float(abs(ind["final_multiple"] / engine_mult - 1))}

    e1s = e1["strategy"]
    res["ep01_reproduced"] = {"start_matches": str(start.date()) == e1s["start"], "final_multiple_here": res["ma_cross"]["final_multiple"],
                              "final_multiple_ep01": e1s["final_multiple"],
                              "relative_gap": abs(res["ma_cross"]["final_multiple"] / e1s["final_multiple"] - 1),
                              "buyhold_gap": abs(res["buyhold"]["final_multiple"] / e1["buyhold"]["final_multiple"] - 1)}

    yr = pd.DataFrame({"strategy": (1 + s["ret"]).groupby(s.index.year).prod() - 1,
                       "buyhold": (1 + h["ret"]).groupby(h.index.year).prod() - 1,
                       "ma_cross": (1 + a["ret"]).groupby(a.index.year).prod() - 1})
    yr = yr.loc[yr.index > s.index[0].year]
    yr["excess"] = yr.strategy - yr.buyhold
    yr["vs_ma"] = yr.strategy - yr.ma_cross
    yr.to_csv(OUT / "annual.csv")
    res["annual"] = {"years": int(len(yr)), "years_strategy_won": int((yr.excess > 0).sum()),
                     "years_beat_ma": int((yr.vs_ma > 0).sum()), "years_lost_to_ma": int((yr.vs_ma < 0).sum()),
                     "years_ma_beat_buyhold": int((yr.ma_cross > yr.buyhold).sum()),
                     "down_years_buyhold": int((yr.buyhold < 0).sum()),
                     "strategy_won_in_down_years": int(((yr.excess > 0) & (yr.buyhold < 0)).sum()),
                     "best_vs_ma": {int(k): float(v) for k, v in yr.vs_ma.nlargest(3).items()},
                     "worst_vs_ma": {int(k): float(v) for k, v in yr.vs_ma.nsmallest(3).items()},
                     "partial_last_year": int(yr.index[-1])}

    # trade health
    xs = macd.crosses(px, FAST, SLOW, SIGNAL, warmup=WARMUP)
    tm = macd.trade_table(px, s["weight"], s["ret"], xs)
    ta = macd.trade_table(px, a["weight"], a["ret"])
    tm.to_csv(OUT / "trades_detail.csv", index=False)
    ta.to_csv(OUT / "trades_ma.csv", index=False)
    res["trade_health"] = {"macd": macd.trade_stats(tm, years), "ma": macd.trade_stats(ta, years)}

    # above / below the zero line
    closed = tm[~tm["open"]]
    res["zero_line"] = {}
    for name, part in {"above": closed[closed["above_zero"]], "other": closed[~closed["above_zero"].astype(bool)]}.items():
        res["zero_line"][name] = {"n": int(len(part)), "win_rate": float((part["return"] > 0).mean()),
                                  "avg_return": float(part["return"].mean()), "median_return": float(part["return"].median()),
                                  "total_multiple": float((1 + part["return"]).prod())}

    # variants, decided before running
    v = {"base": stats(s, rf)}
    for m in ("zero_above", "dif_zero"):
        bt = run(px, macd.target(px, FAST, SLOW, SIGNAL, mode=m, warmup=WARMUP), rf, start)
        v[m] = stats(bt, rf)
    res["variants"] = v
    res["variants_n"] = 2

    # EMA seed sensitivity
    sma_t = macd.target(px, FAST, SLOW, SIGNAL, warmup=WARMUP, seed="sma")
    sb = run(px, sma_t, rf, start)
    res["seed_sensitivity"] = {"sma_seed": stats(sb, rf), "days_target_differs": int((sma_t != tgt).sum()),
                               "final_multiple_gap": abs(stats(sb, rf)["final_multiple"] / engine_mult - 1)}

    # episodes
    eps = episode_table(h, s, a)
    res["episodes"] = eps
    pd.DataFrame(eps).to_csv(OUT / "episodes.csv", index=False)
    leads = [e["macd_lead_days"] for e in eps if e["macd_lead_days"] is not None]
    # measured on the first version of first_exit_after, before the fix; see specs/ep07_build_log.md
    res["episode_lead_before_fix"] = {"peak": "2000-03-24", "lead_days": -8}
    res["episode_summary"] = {"n": len(eps), "macd_shallower_than_ma": int(sum(e["macd"] > e["ma"] for e in eps)),
                              "macd_shallower_than_buyhold": int(sum(e["macd"] > e["buyhold"] for e in eps)),
                              "macd_cycle_beat_ma": int(sum(e["macd_cycle"] > e["ma_cycle"] for e in eps)),
                              "n_with_lead": len(leads), "n_macd_earlier": int(sum(x > 0 for x in leads)),
                              "median_lead_days": float(np.median(leads)), "min_lead_days": int(min(leads)), "max_lead_days": int(max(leads)),
                              "macd_exits_total": int(sum(e["macd_exits"] for e in eps)), "ma_exits_total": int(sum(e["ma_exits"] for e in eps))}

    # health 1: costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sw_ma = research.cost_sweep(px, ma_tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sweep["ma_cagr"] = sw_ma["strategy_cagr"]
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")
    res["breakeven_vs_ma_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "ma_cagr")
    res["breakeven_vs_zero_bps"] = research.breakeven_bps(sweep, "strategy_cagr")
    res["cost_factor"] = {"macd": float((1.0 - s["cost"]).prod()), "ma": float((1.0 - a["cost"]).prod()),
                          "macd_sides": int(s["turnover"].gt(0).sum()), "ma_sides": int(a["turnover"].gt(0).sum())}
    res["cost_drag"] = {"macd": float(sweep.loc[sweep.cost_bps == 0, "strategy_cagr"].iloc[0] - sweep.loc[sweep.cost_bps == 5, "strategy_cagr"].iloc[0]),
                        "ma": float(sweep.loc[sweep.cost_bps == 0, "ma_cagr"].iloc[0] - sweep.loc[sweep.cost_bps == 5, "ma_cagr"].iloc[0])}

    # idle cash earns nothing
    zero = run(px, tgt, rf * 0, start)
    res["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero["equity"]), "final_multiple": float(zero["equity"].iloc[-1] / zero["equity"].iloc[0])}

    # health 2: parameters, on the same window as the episode 1 grid
    grid_start = px.index[GRID_WARMUP - 1]
    grid = macd_grid(px, rf, grid_start)
    grid.to_csv(OUT / "param_grid.csv", index=False)
    hg = stats(research.window(engine.vectorized(px, bench.where(px.index >= grid_start, 0.0), COST_BPS, rf), grid_start), rf)
    g = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    rank = int(g.index[(g.fast == FAST) & (g.slow == SLOW) & (g.signal == SIGNAL)][0]) + 1
    near = grid[grid.fast.isin([8, 12, 16]) & grid.slow.isin([20, 26, 35]) & grid.signal.isin([5, 9, 14])]
    ma_grid = research.param_grid(px, ma_cross.target, MA_FASTS, MA_SLOWS, COST_BPS, rf, grid_start)
    ma_grid.to_csv(OUT / "param_grid_ma.csv", index=False)
    base = grid[(grid.fast == FAST) & (grid.slow == SLOW) & (grid.signal == SIGNAL)].iloc[0]
    res["param_grid"] = {"n": int(len(grid)), "window_start": str(grid_start.date()), "buyhold": hg,
                         "rank_of_default_by_sharpe": rank, "default": base.to_dict(),
                         "best": g.iloc[0].to_dict(), "worst": g.iloc[-1].to_dict(),
                         "median_sharpe": float(grid.sharpe.median()), "max_sharpe": float(grid.sharpe.max()),
                         "neighbourhood_sharpe": float(near.sharpe.mean()), "neighbourhood_n": int(len(near)),
                         "n_beat_buyhold_cagr": int((grid.cagr > hg["cagr"]).sum()),
                         "n_beat_buyhold_sharpe": int((grid.sharpe > hg["sharpe"]).sum()),
                         "ma_n": int(len(ma_grid)), "ma_median_sharpe": float(ma_grid.sharpe.median()), "ma_max_sharpe": float(ma_grid.sharpe.max()),
                         "ma_n_beat_buyhold_cagr": int((ma_grid.cagr > hg["cagr"]).sum()),
                         "ma_n_beat_buyhold_sharpe": int((ma_grid.sharpe > hg["sharpe"]).sum()),
                         "ep01_median_sharpe": e1["param_grid"]["median_sharpe"],
                         "by_slow": {int(k): float(x) for k, x in grid.groupby("slow").sharpe.median().items()},
                         "by_fast": {int(k): float(x) for k, x in grid.groupby("fast").sharpe.median().items()},
                         "by_signal": {int(k): float(x) for k, x in grid.groupby("signal").sharpe.median().items()}}

    # health 3: choose before 2010, look once afterwards
    is_end = "2009-12-31"
    is_grid = macd_grid(px, rf, grid_start, is_end)
    bi = is_grid.sort_values("sharpe", ascending=False).iloc[0]
    res["split"] = {"in_sample": f"{grid_start.date()} to {is_end}", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}"}
    def two(t):
        f = engine.vectorized(px, t, COST_BPS, rf)
        return {"in_sample": stats(research.window(f, grid_start, is_end), rf), "out_of_sample": stats(research.window(f, SPLIT), rf)}
    best = (int(bi.fast), int(bi.slow), int(bi.signal))
    res["split"]["best_in_sample"] = {"fast": best[0], "slow": best[1], "signal": best[2], **two(macd.target(px, *best))}
    res["split"]["default"] = two(macd.target(px, FAST, SLOW, SIGNAL))
    res["split"]["zero_above"] = two(macd.target(px, FAST, SLOW, SIGNAL, mode="zero_above"))
    res["split"]["ma_cross"] = two(ma_cross.target(px, MA_FAST, MA_SLOW))
    res["split"]["buyhold"] = two(bench)

    # other assets
    res["other_assets"] = {}
    for sym in OTHERS:
        p = data.load_prices(sym)
        r = data.daily_rate(data.load_rate("%3MTCM"), p.index)
        st = p.index[WARMUP - 1]
        b = pd.Series(0.0, index=p.index)
        b.loc[st:] = 1.0
        res["other_assets"][sym] = {"start": str(st.date()),
                                    "strategy": stats(run(p, macd.target(p, FAST, SLOW, SIGNAL, warmup=WARMUP), r, st), r),
                                    "ma_cross": stats(run(p, ma_cross.target(p, MA_FAST, MA_SLOW), r, st), r),
                                    "buyhold": stats(run(p, b, r, st), r)}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    curves = pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"], "ma_cross": a["equity"],
                           "strategy_dd": metrics.drawdown(s["equity"]), "buyhold_dd": metrics.drawdown(h["equity"]), "ma_dd": metrics.drawdown(a["equity"]),
                           "weight": s["weight"], "ma_weight": a["weight"], "close": px["close"].loc[s.index],
                           "dif": xs["dif"].loc[s.index], "dea": xs["dea"].loc[s.index], "hist": (xs["dif"] - xs["dea"]).loc[s.index]})
    curves.to_csv(OUT / "curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], color="grey", label="Buy & hold"); ax.plot(a["equity"], color="tab:blue", label="MA 20/60 cross")
    ax.plot(s["equity"], color="black", label="MACD 12/26/9")
    ax.set_yscale("log"); ax.legend(); fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)
    show = ["episode_summary", "checks", "independent", "ep01_reproduced", "strategy", "buyhold", "ma_cross", "trade_count", "annual", "trade_health", "zero_line",
            "variants", "seed_sensitivity", "breakeven_vs_buyhold_bps", "breakeven_vs_ma_bps", "breakeven_vs_zero_bps", "cost_drag",
            "idle_cash_zero_rate", "split", "other_assets"]
    print(json.dumps({k: res[k] for k in show}, indent=1, default=str))
    print(json.dumps(res["episodes"], indent=0, default=str))
    print(json.dumps(res["param_grid"], indent=0, default=str))


if __name__ == "__main__":
    main()
