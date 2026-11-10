"""Episode 6: a 10% trailing stop on SPY. Writes every number to results/ep06/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep06.py   # data format: see backtest/data.py

Limits of what this can show: SPY is a basket of about 500 companies and has always recovered so far.
That is not true of single stocks, and single stocks are NOT tested here (survivorship bias would apply).
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
from strategies import stop_loss

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ep06"
SYMBOL, COST_BPS, STOP, REENTRY = "SPY", 5.0, 0.10, 20
SPLIT = "2010-01-01"
STOPS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
OTHERS = ["QQQ", "IWM", "TLT", "GLD"]
MIN_DD = -0.15
EP03_START = "1993-11-11"


def stats(bt, rf):
    return {"sharpe": metrics.sharpe(bt["ret"], rf.loc[bt.index]), "cagr": metrics.cagr(bt["equity"]),
            "max_drawdown": metrics.max_drawdown(bt["equity"]), "ann_vol": metrics.ann_vol(bt["ret"]),
            "final_multiple": float(bt["equity"].iloc[-1] / bt["equity"].iloc[0]), "exposure": float(bt["weight"].mean()),
            "trades": int(len(metrics.trades(bt["weight"], bt["ret"])))}


def run(px, tgt, rf, start, cost=COST_BPS):
    return research.window(engine.vectorized(px, tgt, cost, rf), start)


def independent_replay(px: pd.DataFrame, rf: pd.Series, stop: float, reentry: int, cost_bps: float) -> dict:
    """A separate ledger: no call into engine.py or stop_loss.py. One pass, plain lists, cash and shares.
    Day 0 only decides to buy, so it earns nothing here; compare with equity[-1] / equity[0]."""
    o, c, r = px["open"].tolist(), px["close"].tolist(), rf.reindex(px.index).fillna(0.0).tolist()
    fee = cost_bps / 1e4
    cash, shares, order, peak, new_entry, stops = 1.0, 0.0, "buy", 0.0, False, 0
    for i in range(1, len(c)):
        if order == "buy":
            shares, cash, new_entry = cash * (1 - fee) / o[i], 0.0, True
        elif order == "sell":
            cash, shares, stops = shares * o[i] * (1 - fee), 0.0, stops + 1
        order = None
        if shares == 0.0:
            cash *= 1 + r[i]
            if i >= reentry and c[i] > max(c[i - reentry:i]):
                order = "buy"
        else:
            peak = c[i] if new_entry else max(peak, c[i])
            new_entry = False
            if c[i] <= peak * (1 - stop):
                order = "sell"
    return {"final_multiple": float(cash + shares * c[-1]), "stops": stops}


def episode_table(s, h, weight):
    rows = []
    for e in episodes(h["equity"], MIN_DD):
        end = e["recovered"] if e["recovered"] is not None else h.index[-1]
        down_s = s["equity"].loc[e["peak"]:e["trough"]]
        down_w = weight.loc[e["peak"]:e["trough"]]
        out_days = down_w.index[down_w == 0]
        cycle_s = float(s["equity"].loc[end] / s["equity"].loc[e["peak"]])
        cycle_h = float(h["equity"].loc[end] / h["equity"].loc[e["peak"]])
        w_cycle = weight.loc[e["peak"]:end]
        rows.append({"peak": str(e["peak"].date()), "trough": str(e["trough"].date()),
                     "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
                     "buyhold": e["depth"], "strategy": float(down_s.iloc[-1] / down_s.iloc[0] - 1),
                     "saved": float((down_s.iloc[-1] / down_s.iloc[0] - 1) - e["depth"]),
                     "first_exit": str(out_days[0].date()) if len(out_days) else None,
                     "stops_in_cycle": int(((w_cycle == 0) & (w_cycle.shift(1) > 0)).sum()),
                     "days_peak_to_trough": int(len(down_s)),
                     "cycle_strategy": cycle_s - 1, "cycle_buyhold": cycle_h - 1,
                     "rel_multiple": cycle_s / cycle_h})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    start = px.index[0]
    res: dict = {"symbol": SYMBOL, "rules": {"stop": STOP, "reentry_window": REENTRY, "cost_bps_per_side": COST_BPS,
                                             "fill": "next open", "cash": "3-month Treasury (constant maturity)"},
                 "scope_note": "SPY is a basket of about 500 companies. Single stocks are not tested; survivorship bias would apply."}
    tgt = stop_loss.target(px, STOP, REENTRY)
    bench = pd.Series(1.0, index=px.index)

    res["checks"] = {"lookahead": checks.lookahead_check(lambda p: stop_loss.target(p, STOP, REENTRY), px),
                     "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
                     "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf),
                     "lookahead_variants": {name: checks.lookahead_check(lambda p, k=kw: stop_loss.target(p, STOP, REENTRY, **k), px)["passed"]
                                            for name, kw in {"cost_basis": {"basis": "cost"}, "wait20": {"wait": 20}, "never_back": {"never_back": True}}.items()}}

    s, h = run(px, tgt, rf, start), run(px, bench, rf, start)
    res["strategy"] = metrics.summary(s, rf)
    res["buyhold"] = metrics.summary(h, rf)
    tr = metrics.trades(s["weight"], s["ret"])
    tr.to_csv(OUT / "trades.csv", index=False)
    res["trade_count"] = int(len(tr))

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

    # every stop: where we sold and where we came back
    lg = stop_loss.ledger(px, s["weight"])
    lg.to_csv(OUT / "stops.csv", index=False)
    done = lg.dropna(subset=["price_change"]).copy()
    done["higher"] = done["higher"].astype(bool)
    done["whipsaw"] = done["whipsaw"].astype(bool)
    done["whipsaw_strict"] = done["whipsaw_strict"].astype(bool)
    ratio = (done.exit_open / done.reentry_open)
    deep = done[done.low_vs_exit <= -0.10]
    res["stops"] = {"n": int(len(lg)), "completed": int(len(done)), "n_higher": int(done.higher.sum()),
                    "share_higher": float(done.higher.mean()),
                    "n_whipsaw": int(done.whipsaw.sum()), "share_whipsaw": float(done.whipsaw.mean()),
                    "n_whipsaw_strict": int(done.whipsaw_strict.sum()),
                    "n_deep": int(len(deep)), "share_deep": float(len(deep) / len(done)),
                    "median_days_out": float(done.days_out.median()),
                    "median_price_change": float(done.price_change.median()),
                    "median_low_vs_exit": float(done.low_vs_exit.median()),
                    "deepest_low": float(done.low_vs_exit.min()),
                    "deepest_low_exit": str(done.loc[done.low_vs_exit.idxmin(), "exit"].date()),
                    "best_saved": float(done.price_change.min()),
                    "best_saved_exit": str(done.loc[done.price_change.idxmin(), "exit"].date()),
                    "worst_regret": float(done.price_change.max()),
                    "worst_regret_exit": str(done.loc[done.price_change.idxmax(), "exit"].date()),
                    "shares_multiplier": float(ratio.prod()),
                    "shares_left_per_100": float(100 * ratio.prod()),
                    "cash_interest_multiplier": float((1 + rf.loc[s.index[1:]][s["weight"].iloc[1:] == 0]).prod()),
                    "cost_multiplier": float((1 - COST_BPS / 1e4) ** (int(s["turnover"].gt(0).sum()) - 1)),
                    "open_position_at_end": bool(s["weight"].iloc[-1] == 1.0),
                    "last_exit": str(lg.exit.iloc[-1].date())}
    by_year = done.groupby(pd.to_datetime(done.exit).dt.year).size()
    res["stops"]["busiest_year"] = int(by_year.idxmax())
    res["stops"]["busiest_year_stops"] = int(by_year.max())

    # bridge from shares to wealth: strategy / buy & hold = shares effect x interest x extra costs
    st = res["stops"]
    res["wealth_bridge"] = {"share_effect": st["shares_multiplier"], "interest_effect": st["cash_interest_multiplier"],
                            "extra_cost_effect": st["cost_multiplier"],
                            "product": st["shares_multiplier"] * st["cash_interest_multiplier"] * st["cost_multiplier"],
                            "total_ratio": float(s["equity"].iloc[-1] / h["equity"].iloc[-1])}

    # independent replay: no engine, no strategy function
    ind = independent_replay(px, rf, STOP, REENTRY, COST_BPS)
    res["independent"] = {**ind, "engine_final_multiple": float(s["equity"].iloc[-1] / s["equity"].iloc[0]),
                          "engine_stops": int(len(lg)),
                          "relative_gap": float(abs(ind["final_multiple"] / (s["equity"].iloc[-1] / s["equity"].iloc[0]) - 1)),
                          # measured on the first version of this script, before stats() was fixed; see specs/ep06_build_log.md
                          "gap_before_fix": 0.00011744652366429964}

    # big drawdowns of buy & hold, found automatically
    eps = episode_table(s, h, s["weight"])
    res["episodes"] = eps
    pd.DataFrame(eps).to_csv(OUT / "episodes.csv", index=False)
    total = float(s["equity"].iloc[-1] / h["equity"].iloc[-1])
    in_eps = float(np.prod([e["rel_multiple"] for e in eps]))
    res["wealth_split"] = {"total_ratio": total, "episodes_product": in_eps, "rest_of_time": total / in_eps,
                           "steps": [{"peak": e["peak"], "rel_multiple": e["rel_multiple"]} for e in eps]}

    # stop-size sweep on one common window
    rows = []
    for st in STOPS:
        bt = run(px, stop_loss.target(px, st, REENTRY), rf, start)
        rows.append({"stop": st, **stats(bt, rf), "stops_hit": int(len(stop_loss.ledger(px, bt["weight"])))})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "param_grid.csv", index=False)
    bh_grid = stats(h, rf)
    g = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    res["param_grid"] = {"n": int(len(grid)), "buyhold": bh_grid,
                         "rank_of_10pct_by_sharpe": int(g.index[g["stop"] == STOP][0]) + 1,
                         "best": g.iloc[0].to_dict(), "worst": g.iloc[-1].to_dict(),
                         "n_beat_buyhold_cagr": int((grid.cagr > bh_grid["cagr"]).sum()),
                         "n_beat_buyhold_sharpe": int((grid.sharpe > bh_grid["sharpe"]).sum()),
                         "n_shallower_drawdown": int((grid.max_drawdown > bh_grid["max_drawdown"]).sum()),
                         "rows": grid.to_dict("records")}

    # variants, all at the 10% stop
    v = {"base": stats(s, rf)}
    for name, kw in {"cost_basis": {"basis": "cost"}, "wait20": {"wait": 20}, "never_back": {"never_back": True},
                     "reentry10": {"reentry": 10}, "reentry50": {"reentry": 50}}.items():
        args = {"reentry": REENTRY, **kw}
        bt = run(px, stop_loss.target(px, STOP, **args), rf, start)
        v[name] = {**stats(bt, rf), "stops_hit": int(len(stop_loss.ledger(px, bt["weight"])))}
    v["base"]["stops_hit"] = int(len(lg))
    res["variants"] = v
    res["variants_n"] = 5
    nb_w = run(px, stop_loss.target(px, STOP, REENTRY, never_back=True), rf, start)["weight"]
    nb_exit = nb_w.index[(nb_w == 0) & (nb_w.shift(1) > 0)]
    res["never_back_first_exit"] = str(nb_exit[0].date()) if len(nb_exit) else None
    cb_w = run(px, stop_loss.target(px, STOP, REENTRY, basis="cost"), rf, start)["weight"]
    res["cost_basis_ever_stopped"] = bool(((cb_w == 0) & (cb_w.shift(1) > 0)).any())

    # added after seeing wait20 beat buy & hold: is 20 days special, or are its neighbours similar?
    wait_rows = []
    for n in [10, 15, 20, 25, 30, 40, 60]:
        bt = run(px, stop_loss.target(px, STOP, wait=n), rf, start)
        wait_rows.append({"wait": n, **stats(bt, rf)})
    wdf = pd.DataFrame(wait_rows)
    wf = engine.vectorized(px, stop_loss.target(px, STOP, wait=20), COST_BPS, rf)
    w20 = run(px, stop_loss.target(px, STOP, wait=20), rf, start)
    w20l = stop_loss.ledger(px, w20["weight"]).dropna(subset=["price_change"])
    wait_other = {}
    for sym in OTHERS:
        p = data.load_prices(sym)
        r = data.daily_rate(data.load_rate("%3MTCM"), p.index)
        wb = run(p, stop_loss.target(p, STOP, wait=20), r, p.index[0])
        bb = run(p, pd.Series(1.0, index=p.index), r, p.index[0])
        wait_other[sym] = {"cagr": metrics.cagr(wb["equity"]), "buyhold_cagr": metrics.cagr(bb["equity"]),
                           "sharpe": metrics.sharpe(wb["ret"], r.loc[wb.index]), "buyhold_sharpe": metrics.sharpe(bb["ret"], r.loc[bb.index])}
    res["wait_check"] = {"n": int(len(wdf)), "n_beat_buyhold_cagr": int((wdf.cagr > bh_grid["cagr"]).sum()),
                         "rows": wdf.to_dict("records"),
                         "wait20_stops": int(len(w20l)), "wait20_share_higher": float(w20l.higher.astype(bool).mean()),
                         "wait20_shares_left_per_100": float(100 * (w20l.exit_open / w20l.reentry_open).prod()),
                         "other_assets": wait_other, "other_assets_beat_cagr": int(sum(v["cagr"] > v["buyhold_cagr"] for v in wait_other.values())),
                         "wait20_in_sample": stats(research.window(wf, start, "2009-12-31"), rf),
                         "wait20_out_of_sample": stats(research.window(wf, SPLIT), rf)}

    # health 1: costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")

    # health 2: idle cash earns nothing
    zero = run(px, tgt, rf * 0, start)
    res["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero["equity"]), "final_multiple": float(zero["equity"].iloc[-1])}

    # health 3: choose the stop before 2010, look once afterwards
    rows = []
    for st in STOPS:
        bt = research.window(engine.vectorized(px, stop_loss.target(px, st, REENTRY), COST_BPS, rf), start, "2009-12-31")
        rows.append({"stop": st, **stats(bt, rf)})
    is_df = pd.DataFrame(rows)
    best_is = is_df.sort_values("sharpe", ascending=False).iloc[0]
    res["split"] = {"in_sample": f"{start.date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}",
                    "in_sample_rows": is_df.to_dict("records")}
    for name, st in {"best_in_sample": float(best_is["stop"]), "stop10": STOP}.items():
        full = engine.vectorized(px, stop_loss.target(px, st, REENTRY), COST_BPS, rf)
        res["split"][name] = {"stop": st, "in_sample": stats(research.window(full, start, "2009-12-31"), rf),
                              "out_of_sample": stats(research.window(full, SPLIT), rf)}
    bfull = engine.vectorized(px, bench, COST_BPS, rf)
    res["split"]["buyhold"] = {"in_sample": stats(research.window(bfull, start, "2009-12-31"), rf),
                               "out_of_sample": stats(research.window(bfull, SPLIT), rf)}

    # other assets, same rules. Single stocks are deliberately not here.
    res["other_assets"] = {}
    for sym in OTHERS:
        p = data.load_prices(sym)
        r = data.daily_rate(data.load_rate("%3MTCM"), p.index)
        t = stop_loss.target(p, STOP, REENTRY)
        a, b = run(p, t, r, p.index[0]), run(p, pd.Series(1.0, index=p.index), r, p.index[0])
        l = stop_loss.ledger(p, a["weight"]).dropna(subset=["price_change"])
        res["other_assets"][sym] = {"start": str(p.index[0].date()), "strategy": stats(a, r), "buyhold": stats(b, r),
                                    "stops_hit": int(len(l)), "share_higher": float(l.higher.astype(bool).mean()) if len(l) else None}

    # next to the 200-day filter of episode 3, on its window
    e3 = json.loads((ROOT / "results" / "ep03" / "results.json").read_text())
    full = engine.vectorized(px, tgt, COST_BPS, rf)
    res["vs_ep03"] = {"window_start": EP03_START, "ep03": {**{k: e3["strategy"][k] for k in ["cagr", "sharpe", "max_drawdown", "final_multiple", "exposure"]},
                                                           "trades": e3["trade_count"]},
                      "ep03_annual": {k: e3["annual"][k] for k in ["years", "years_strategy_won", "down_years_buyhold", "strategy_won_in_down_years"]},
                      "ep03_buyhold": {k: e3["buyhold"][k] for k in ["cagr", "sharpe", "max_drawdown", "final_multiple"]},
                      "stop10_same_window": stats(research.window(full, EP03_START), rf),
                      "buyhold_same_window": stats(research.window(bfull, EP03_START), rf)}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    curves = pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"], "strategy_dd": metrics.drawdown(s["equity"]),
                           "buyhold_dd": metrics.drawdown(h["equity"]), "weight": s["weight"], "close": px["close"]})
    for st in STOPS:
        curves[f"stop_{int(st * 100)}"] = run(px, stop_loss.target(px, st, REENTRY), rf, start)["equity"]
    curves.to_csv(OUT / "curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], color="grey", label="Buy & hold"); ax.plot(s["equity"], color="black", label="10% trailing stop")
    ax.set_yscale("log"); ax.legend(); fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)
    print(json.dumps({k: res[k] for k in ["checks", "strategy", "buyhold", "trade_count", "stops", "independent", "wealth_bridge", "wealth_split", "wait_check", "idle_cash_zero_rate", "breakeven_vs_buyhold_bps", "variants", "annual", "vs_ep03", "never_back_first_exit", "cost_basis_ever_stopped"]}, indent=1, default=str))
    print(json.dumps(res["episodes"], indent=0)); print(json.dumps({k: v for k, v in res["param_grid"].items()}, indent=0, default=str)); print(json.dumps(res["other_assets"], indent=0)); print({k: v for k, v in res["split"].items() if k != "in_sample_rows"})


if __name__ == "__main__":
    main()
