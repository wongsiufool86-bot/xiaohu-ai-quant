"""Episode 1: moving-average crossover on SPY. Writes every number to results/ep01/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep01.py   # data format: see backtest/data.py
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
from strategies import ma_cross

import os
DATA_START = os.environ.get("DATA_START")          # e.g. 2000-01-03 to mimic a source that starts later
OUT = Path(__file__).parent / "results" / ("ep01" if not DATA_START else f"ep01_from{DATA_START[:4]}")
SYMBOL, FAST, SLOW, COST_BPS = "SPY", 20, 60, 5.0
SPLIT = "2010-01-01"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    if DATA_START:
        px = px.loc[DATA_START:]
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    res: dict = {"symbol": SYMBOL, "rules": {"fast": FAST, "slow": SLOW, "cost_bps_per_side": COST_BPS,
                 "fill": "next open", "cash": "3-month Treasury (constant maturity)"}}
    res["data_quality"] = data.validate(px)

    start = ma_cross.first_signal_date(px, SLOW)
    tgt = ma_cross.target(px, FAST, SLOW)
    bench = pd.Series(0.0, index=px.index)
    bench.loc[start:] = 1.0

    # mandatory checks
    res["checks"] = {
        "lookahead": checks.lookahead_check(lambda p: ma_cross.target(p, FAST, SLOW), px),
        "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
        "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf),
    }

    s = research.window(engine.vectorized(px, tgt, COST_BPS, rf), start)
    h = research.window(engine.vectorized(px, bench, COST_BPS, rf), start)
    res["strategy"] = metrics.summary(s, rf)
    res["buyhold"] = metrics.summary(h, rf)
    tr = metrics.trades(s["weight"], s["ret"])
    tr.to_csv(OUT / "trades.csv", index=False)

    # year by year (calendar years fully inside the window)
    yr = pd.DataFrame({"strategy": (1 + s["ret"]).groupby(s.index.year).prod() - 1,
                       "buyhold": (1 + h["ret"]).groupby(h.index.year).prod() - 1})
    yr = yr.loc[yr.index > s.index[0].year]            # drop the partial first year
    yr["excess"] = yr.strategy - yr.buyhold
    yr.to_csv(OUT / "annual.csv")
    res["annual"] = {
        "years": int(len(yr)), "years_strategy_won": int((yr.excess > 0).sum()),
        "best_years_for_strategy": {int(k): round(float(v), 4) for k, v in yr.excess.nlargest(3).items()},
        "worst_years_for_strategy": {int(k): round(float(v), 4) for k, v in yr.excess.nsmallest(3).items()},
        "partial_last_year": int(yr.index[-1]),
    }
    # 2008: when did it get out, when back in, and what did each side lose in the year
    w = s["weight"]
    exits = w.index[(w == 0) & (w.shift(1) > 0)]
    entries = w.index[(w > 0) & (w.shift(1) == 0)]
    crash_low = h["equity"].loc["2007":"2009"].idxmin()
    out_08 = exits[exits < crash_low][-1]
    in_08 = entries[entries > crash_low][0]
    res["event_2008"] = {
        "exit_before_low": str(out_08.date()), "reentry_after_low": str(in_08.date()),
        "buyhold_low_date": str(crash_low.date()),
        "buyhold_2008": float(yr.loc[2008, "buyhold"]), "strategy_2008": float(yr.loc[2008, "strategy"]),
        "spy_change_exit_to_reentry": float(px["close"].loc[in_08] / px["close"].loc[out_08] - 1),
    }

    # price-only (no dividends) buy & hold, for the "dividends matter" footnote
    has_price_only = "unadjusted_close" in px.columns
    po = px["unadjusted_close"].loc[start:] if has_price_only else None
    res["buyhold_price_only_multiple"] = float(po.iloc[-1] / po.iloc[0]) if has_price_only else None

    # the one-line bug: no shift
    cheat = research.window(engine.same_bar_cheat(px, tgt), start)
    res["cheat_no_shift"] = {"final_multiple": float(cheat["equity"].iloc[-1]),
                             "cagr": metrics.cagr(cheat["equity"]),
                             "max_drawdown": metrics.max_drawdown(cheat["equity"])}

    # health check 1: costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")
    res["breakeven_vs_zero_bps"] = research.breakeven_bps(sweep, "strategy_cagr")

    # health check 2: parameters (full period)
    fasts = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    slows = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300]
    grid = research.param_grid(px, ma_cross.target, fasts, slows, COST_BPS, rf,
                               ma_cross.first_signal_date(px, max(slows)))
    grid.to_csv(OUT / "param_grid.csv", index=False)
    g = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    rank = int(g.index[(g.fast == FAST) & (g.slow == SLOW)][0]) + 1
    h_grid = research.window(engine.vectorized(px, bench, COST_BPS, rf), ma_cross.first_signal_date(px, max(slows)))
    res["param_grid"] = {
        "n_combos": int(len(grid)),
        "rank_of_20_60_by_sharpe": rank,
        "best": g.iloc[0].to_dict(),
        "worst": g.iloc[-1].to_dict(),
        "median_sharpe": float(grid.sharpe.median()),
        "share_beating_buyhold_sharpe": float((grid.sharpe > metrics.sharpe(h_grid["ret"], rf)).mean()),
        "buyhold_sharpe_same_window": metrics.sharpe(h_grid["ret"], rf),
        "neighbourhood_sharpe_20_60": research.neighbourhood(grid, FAST, SLOW),
        "window_start": str(ma_cross.first_signal_date(px, max(slows)).date()),
    }

    # health check 3: choose on the first part, look once at the second part
    is_start = ma_cross.first_signal_date(px, max(slows))
    is_grid = research.param_grid(px, ma_cross.target, fasts, slows, COST_BPS, rf, is_start, "2009-12-31")
    best_is = is_grid.sort_values("sharpe", ascending=False).iloc[0]
    oos = {}
    for name, (f, sl) in {"best_in_sample": (int(best_is.fast), int(best_is.slow)), "20_60": (FAST, SLOW)}.items():
        bt_full = engine.vectorized(px, ma_cross.target(px, f, sl), COST_BPS, rf)
        a = research.window(bt_full, is_start, "2009-12-31")
        b = research.window(bt_full, SPLIT)
        oos[name] = {"fast": f, "slow": sl,
                     "in_sample": {"sharpe": metrics.sharpe(a["ret"], rf), "cagr": metrics.cagr(a["equity"]),
                                   "max_drawdown": metrics.max_drawdown(a["equity"])},
                     "out_of_sample": {"sharpe": metrics.sharpe(b["ret"], rf), "cagr": metrics.cagr(b["equity"]),
                                       "max_drawdown": metrics.max_drawdown(b["equity"])}}
    bh_full = engine.vectorized(px, bench, COST_BPS, rf)
    for name, (a0, a1) in {"in_sample": (is_start, "2009-12-31"), "out_of_sample": (SPLIT, None)}.items():
        w = research.window(bh_full, a0, a1)
        oos.setdefault("buyhold", {})[name] = {"sharpe": metrics.sharpe(w["ret"], rf), "cagr": metrics.cagr(w["equity"]),
                                               "max_drawdown": metrics.max_drawdown(w["equity"])}
    res["split"] = {"in_sample": f"{is_start.date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}", **oos}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"],
                  "strategy_dd": metrics.drawdown(s["equity"]), "buyhold_dd": metrics.drawdown(h["equity"]),
                  "cheat_no_shift": cheat["equity"],
                  "correct_zero_cost": research.window(engine.vectorized(px, tgt, 0.0, rf), start)["equity"],
                  "buyhold_price_only": po / po.iloc[0] if has_price_only else float("nan"),
                  "strategy_weight": s["weight"]}).to_csv(OUT / "curves.csv")
    plots(s, h, cheat, sweep, grid)
    print(json.dumps({k: res[k] for k in ["checks", "strategy", "buyhold"]}, indent=2, default=str))


def plots(s, h, cheat, sweep, grid):
    note = "SPY daily, historical backtest, not a forecast. Teaching example, not investment advice."
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], label="Buy & hold", color="grey")
    ax.plot(s["equity"], label="MA 20/60 cross", color="black")
    ax.set_yscale("log"); ax.legend(); ax.set_title("Equity (log scale)"); fig.text(0.01, 0.01, note, fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(h.index, metrics.drawdown(h["equity"]), color="grey", alpha=0.4, label="Buy & hold")
    ax.plot(metrics.drawdown(s["equity"]), color="black", label="MA 20/60 cross")
    ax.legend(); ax.set_title("Drawdown"); fig.text(0.01, 0.01, note, fontsize=7)
    fig.savefig(OUT / "drawdown.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cheat["equity"], color="red", label="Bug: no one-day shift")
    ax.plot(s["equity"], color="black", label="Correct")
    ax.set_yscale("log"); ax.legend(); ax.set_title("One line of code"); fig.text(0.01, 0.01, note, fontsize=7)
    fig.savefig(OUT / "cheat.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep.cost_bps, sweep.strategy_cagr * 100, marker="o", color="black", label="MA 20/60 cross")
    ax.plot(sweep.cost_bps, sweep.buyhold_cagr * 100, marker="o", color="grey", label="Buy & hold")
    ax.set_xlabel("Cost per side (bp)"); ax.set_ylabel("CAGR %"); ax.legend(); fig.text(0.01, 0.01, note, fontsize=7)
    fig.savefig(OUT / "cost_sweep.png", dpi=150); plt.close(fig)

    piv = grid.pivot(index="fast", columns="slow", values="sharpe")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, cmap="Greys", origin="lower", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("slow"); ax.set_ylabel("fast"); fig.colorbar(im, label="Sharpe"); ax.set_title("Sharpe by parameters")
    fig.text(0.01, 0.01, note, fontsize=7)
    fig.savefig(OUT / "param_grid.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
