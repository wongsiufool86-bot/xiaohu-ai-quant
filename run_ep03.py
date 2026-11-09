"""Episode 3: 200-day moving-average filter on SPY. Writes every number to results/ep03/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep03.py   # data format: see backtest/data.py
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
from strategies import ma_filter

OUT = Path(__file__).parent / "results" / "ep03"
SYMBOL, WINDOW, COST_BPS = "SPY", 200, 5.0
SPLIT = "2010-01-01"
WINDOWS = [50, 75, 100, 120, 150, 180, 200, 220, 250, 300]
MIN_DD = -0.15


def episodes(equity: pd.Series, min_dd: float) -> list[dict]:
    """Peak-to-trough episodes deeper than min_dd. An episode ends when equity makes a new high."""
    out, peak_date, peak = [], equity.index[0], equity.iloc[0]
    trough_date, trough = peak_date, peak
    for d, v in equity.items():
        if v >= peak:
            if trough / peak - 1 <= min_dd:
                out.append({"peak": peak_date, "trough": trough_date, "recovered": d, "depth": trough / peak - 1})
            peak_date, peak, trough_date, trough = d, v, d, v
        elif v < trough:
            trough_date, trough = d, v
    if trough / peak - 1 <= min_dd:
        out.append({"peak": peak_date, "trough": trough_date, "recovered": None, "depth": trough / peak - 1})
    return out


def out_periods(weight: pd.Series, opens: pd.Series) -> pd.DataFrame:
    w = weight.fillna(0.0)
    exits = w.index[(w == 0) & (w.shift(1, fill_value=0.0) > 0)]
    rows = []
    for x in exits:
        later = w.index[(w.index > x) & (w > 0)]
        if not len(later):
            rows.append({"exit": x, "reentry": None, "days_out": int((w.index >= x).sum()), "price_change": None})
            continue
        n = later[0]
        rows.append({"exit": x, "reentry": n, "days_out": int(((w.index >= x) & (w.index < n)).sum()),
                     "price_change": float(opens.loc[n] / opens.loc[x] - 1)})
    return pd.DataFrame(rows)


def stats(bt, rf):
    return {"sharpe": metrics.sharpe(bt["ret"], rf.loc[bt.index]), "cagr": metrics.cagr(bt["equity"]),
            "max_drawdown": metrics.max_drawdown(bt["equity"]), "exposure": float(bt["weight"].mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    res: dict = {"symbol": SYMBOL, "rules": {"window": WINDOW, "cost_bps_per_side": COST_BPS, "fill": "next open",
                                             "cash": "3-month Treasury (constant maturity)"}}
    start = ma_filter.first_signal_date(px, WINDOW)
    tgt = ma_filter.target(px, WINDOW)
    bench = pd.Series(0.0, index=px.index)
    bench.loc[start:] = 1.0

    res["checks"] = {"lookahead": checks.lookahead_check(lambda p: ma_filter.target(p, WINDOW), px),
                     "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
                     "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf)}

    s = research.window(engine.vectorized(px, tgt, COST_BPS, rf), start)
    h = research.window(engine.vectorized(px, bench, COST_BPS, rf), start)
    res["strategy"] = metrics.summary(s, rf)
    res["buyhold"] = metrics.summary(h, rf)
    res["trade_count"] = int(len(metrics.trades(s["weight"], s["ret"])))
    metrics.trades(s["weight"], s["ret"]).to_csv(OUT / "trades.csv", index=False)

    yr = pd.DataFrame({"strategy": (1 + s["ret"]).groupby(s.index.year).prod() - 1,
                       "buyhold": (1 + h["ret"]).groupby(h.index.year).prod() - 1})
    yr = yr.loc[yr.index > s.index[0].year]
    yr["excess"] = yr.strategy - yr.buyhold
    yr.to_csv(OUT / "annual.csv")
    res["annual"] = {"years": int(len(yr)), "years_strategy_won": int((yr.excess > 0).sum()),
                     "up_years_buyhold": int((yr.buyhold > 0).sum()),
                     "strategy_won_in_up_years": int(((yr.excess > 0) & (yr.buyhold > 0)).sum()),
                     "down_years_buyhold": int((yr.buyhold < 0).sum()),
                     "strategy_won_in_down_years": int(((yr.excess > 0) & (yr.buyhold < 0)).sum()),
                     "best": {int(k): float(v) for k, v in yr.excess.nlargest(3).items()},
                     "worst": {int(k): float(v) for k, v in yr.excess.nsmallest(3).items()},
                     "partial_last_year": int(yr.index[-1])}

    # big drawdowns of buy & hold, found automatically
    rows = []
    for e in episodes(h["equity"], MIN_DD):
        se = s["equity"].loc[e["peak"]:e["trough"]]
        out_days = s["weight"].loc[e["peak"]:e["trough"]]
        first_out = out_days.index[out_days == 0]
        rows.append({"peak": str(e["peak"].date()), "trough": str(e["trough"].date()),
                     "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
                     "buyhold": e["depth"], "strategy": float(se.iloc[-1] / se.iloc[0] - 1),
                     "first_exit": str(first_out[0].date()) if len(first_out) else None,
                     "days_peak_to_trough": int(len(se)),
                     "saved": float((se.iloc[-1] / se.iloc[0] - 1) - e["depth"])})
    res["episodes"] = rows
    pd.DataFrame(rows).to_csv(OUT / "episodes.csv", index=False)

    # every exit and the price at which it came back
    op = out_periods(s["weight"], px["open"])
    op.to_csv(OUT / "out_periods.csv", index=False)
    done = op.dropna(subset=["price_change"])
    res["out_periods"] = {"n": int(len(op)), "completed": int(len(done)),
                          "share_bought_back_higher": float((done.price_change > 0).mean()),
                          "median_days_out": float(done.days_out.median()),
                          "share_out_10_days_or_less": float((done.days_out <= 10).mean()),
                          "median_price_change": float(done.price_change.median()),
                          "best_saved": float(done.price_change.min()),
                          "best_saved_exit": str(done.loc[done.price_change.idxmin(), "exit"].date()),
                          "worst_regret": float(done.price_change.max()),
                          "worst_regret_exit": str(done.loc[done.price_change.idxmax(), "exit"].date())}

    by_year = done.groupby(pd.to_datetime(done.exit).dt.year).size()
    res["out_periods"]["busiest_year"] = int(by_year.idxmax())
    res["out_periods"]["busiest_year_exits"] = int(by_year.max())

    # independent check: count days directly from close vs average, not through the engine
    ma_ind = px["close"].rolling(WINDOW).mean()
    above = (px["close"] > ma_ind).loc[start:]
    res["independent"] = {"share_days_above": float(above.mean()), "exposure": float(s["weight"].mean()),
                          "up_crossings": int((above & ~above.shift(1, fill_value=False)).sum())}

    # extreme days
    cc = px["close"].pct_change().loc[start:]
    w = s["weight"].reindex(cc.index)
    top, bot = cc.nlargest(10), cc.nsmallest(10)
    res["extreme_days"] = {"best10_out": int((w.loc[top.index] == 0).sum()), "worst10_out": int((w.loc[bot.index] == 0).sum()),
                           "best10_dates": [str(d.date()) for d in top.index], "worst10_dates": [str(d.date()) for d in bot.index],
                           "best_day": float(top.iloc[0]), "worst_day": float(bot.iloc[0])}

    # idle cash earns nothing
    zero = research.window(engine.vectorized(px, tgt, COST_BPS, rf * 0), start)
    res["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero["equity"]), "final_multiple": float(zero["equity"].iloc[-1])}

    # health 1: costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")

    # health 2: parameters, on a common window so all are comparable
    grid_start = ma_filter.first_signal_date(px, max(WINDOWS))
    rows = []
    for win in WINDOWS:
        bt = research.window(engine.vectorized(px, ma_filter.target(px, win), COST_BPS, rf), grid_start)
        rows.append({"window": win, **stats(bt, rf)})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "param_grid.csv", index=False)
    bh_grid = research.window(engine.vectorized(px, bench, COST_BPS, rf), grid_start)
    g = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    res["param_grid"] = {"window_start": str(grid_start.date()), "n": int(len(grid)),
                         "rank_of_200_by_sharpe": int(g.index[g.window == WINDOW][0]) + 1,
                         "best": g.iloc[0].to_dict(), "worst": g.iloc[-1].to_dict(),
                         "buyhold": stats(bh_grid, rf),
                         "n_beat_buyhold_cagr": int((grid.cagr > metrics.cagr(bh_grid["equity"])).sum()),
                         "n_beat_buyhold_sharpe": int((grid.sharpe > metrics.sharpe(bh_grid["ret"], rf.loc[bh_grid.index])).sum()),
                         "rows": grid.to_dict("records")}

    variants = {}
    for name, kw in {"base": {}, "band_1pct": {"band": 0.01}, "band_2pct": {"band": 0.02}, "band_3pct": {"band": 0.03},
                     "monthly": {"monthly": True}}.items():
        t = ma_filter.target(px, WINDOW, **kw)
        bt = research.window(engine.vectorized(px, t, COST_BPS, rf), start)
        variants[name] = {**stats(bt, rf), "trades": int(len(metrics.trades(bt["weight"], bt["ret"])))}
    res["variants"] = variants

    # health 3: choose the window before 2010, look once afterwards
    rows = []
    for win in WINDOWS:
        bt = research.window(engine.vectorized(px, ma_filter.target(px, win), COST_BPS, rf), grid_start, "2009-12-31")
        rows.append({"window": win, **stats(bt, rf)})
    best_is = pd.DataFrame(rows).sort_values("sharpe", ascending=False).iloc[0]
    res["split"] = {"in_sample": f"{grid_start.date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}"}
    for name, win in {"best_in_sample": int(best_is.window), "w200": WINDOW}.items():
        full = engine.vectorized(px, ma_filter.target(px, win), COST_BPS, rf)
        res["split"][name] = {"window": win, "in_sample": stats(research.window(full, grid_start, "2009-12-31"), rf),
                              "out_of_sample": stats(research.window(full, SPLIT), rf)}
    bfull = engine.vectorized(px, bench, COST_BPS, rf)
    res["split"]["buyhold"] = {"in_sample": stats(research.window(bfull, grid_start, "2009-12-31"), rf),
                               "out_of_sample": stats(research.window(bfull, SPLIT), rf)}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"], "strategy_dd": metrics.drawdown(s["equity"]),
                  "buyhold_dd": metrics.drawdown(h["equity"]), "weight": s["weight"],
                  "close": px["close"].loc[start:], "ma200": px["close"].rolling(WINDOW).mean().loc[start:]}).to_csv(OUT / "curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], color="grey", label="Buy & hold"); ax.plot(s["equity"], color="black", label="200-day filter")
    ax.set_yscale("log"); ax.legend(); fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)
    print(json.dumps({k: res[k] for k in ["checks", "strategy", "buyhold", "trade_count", "out_periods", "extreme_days", "idle_cash_zero_rate", "breakeven_vs_buyhold_bps", "variants", "annual"]}, indent=1, default=str))
    print(json.dumps(res["episodes"], indent=0)); print({k: v for k, v in res["param_grid"].items() if k != "rows"}); print(res["split"])


if __name__ == "__main__":
    main()
