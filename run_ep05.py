"""Episode 5: simplified Turtle channel breakout on SPY and on a five-ETF book.
Writes every number to results/ep05/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep05.py   # data format: see backtest/data.py

How the five-ETF book is built (no changes to backtest/):
  * Each ETF gets 1/5 of the money and runs the rule on its own, with its own cash.
    Book equity = the plain average of the five sleeve equity curves. Sleeves are never topped up
    or trimmed against each other.
  * Cash interest: a sleeve that is flat earns the 3-month Treasury rate on that sleeve's own cash,
    through the normal single-asset engine. Nothing is added or moved at book level.
  * Benchmark: same basket, equal weight, always invested, rebalanced to 1/5 at the open of the first
    trading day of each month, 5 bp per side. It holds no cash after the first trade.
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
from strategies import donchian_breakout as db

OUT = Path(__file__).parent / "results" / "ep05"
SYMBOL, ENTRY, EXIT, COST_BPS = "SPY", 20, 10, 5.0
SPLIT = "2010-01-01"
ETFS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
COMMON_START = "2004-11-18"
ENTRIES = [10, 20, 30, 40, 55, 75, 100]
EXITS = [5, 10, 15, 20, 30, 40, 50]
COST_GRID = [0, 2, 5, 10, 15, 20, 30, 40, 50]


# ---- trade ledger and losing streaks ---------------------------------------------------------

def ledger(weight: pd.Series, opens: pd.Series, cost_bps: float, closes: pd.Series | None = None) -> pd.DataFrame:
    """Round trips from a 0/1 held-weight series, priced from opens, not from engine returns.
    Closed trade return = exit open / entry open x (1-c)^2 - 1, no cash interest.
    The final open trade is marked to the last close (or last open) with one side of cost; flagged open."""
    w, o, c = weight.fillna(0.0).values, opens.values, cost_bps / 1e4
    idx, rows, i = weight.index, [], 0
    while i < len(w):
        if w[i] > 0 and (i == 0 or w[i - 1] == 0):
            j = i + 1
            while j < len(w) and w[j] > 0:
                j += 1
            if j < len(w):
                rows.append({"entry": idx[i], "exit": idx[j], "ret": o[j] / o[i] * (1 - c) ** 2 - 1,
                             "hold_days": j - i, "open": False, "entry_i": i, "exit_i": j})
            else:
                last = float(closes.iloc[-1]) if closes is not None else float(o[-1])
                rows.append({"entry": idx[i], "exit": None, "ret": last / o[i] * (1 - c) - 1,
                             "hold_days": len(w) - i, "open": True, "entry_i": i, "exit_i": len(w) - 1})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["entry", "exit", "ret", "hold_days", "open", "entry_i", "exit_i"])


def trade_stats(led: pd.DataFrame) -> dict:
    cl = led[~led["open"].astype(bool)]
    win, loss = cl.loc[cl.ret > 0, "ret"], cl.loc[cl.ret <= 0, "ret"]
    log = np.log1p(cl.ret)
    pos = log[log > 0].sort_values(ascending=False)
    return {"n_total": int(len(led)), "n_closed": int(len(cl)), "n_open": int(led["open"].astype(bool).sum()),
            "win_rate": float((cl.ret > 0).mean()), "n_wins": int(len(win)), "n_losses": int(len(loss)),
            "avg_win": float(win.mean()), "avg_loss": float(loss.mean()), "payoff_ratio": float(win.mean() / abs(loss.mean())),
            "median_win": float(win.median()), "median_loss": float(loss.median()),
            "expectancy": float(cl.ret.mean()), "best": float(cl.ret.max()), "worst": float(cl.ret.min()),
            "avg_hold_days": float(cl.hold_days.mean()), "median_hold_days": float(cl.hold_days.median()),
            "top5_share_of_gross_gain": float(pos.head(5).sum() / pos.sum()) if len(pos) else float("nan")}


def streaks(led: pd.DataFrame) -> dict:
    """Runs of consecutive closed losing trades, in order."""
    cl = led[~led["open"].astype(bool)].reset_index(drop=True)
    runs, cur = [], []
    for _, row in cl.iterrows():
        if row.ret <= 0:
            cur.append(row)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    def info(r):
        ei, xi = r[0]["entry_i"], r[-1]["exit_i"]
        return {"n": len(r), "start": str(pd.Timestamp(r[0]["entry"]).date()), "end": str(pd.Timestamp(r[-1]["exit"]).date()),
                "calendar_days": int((pd.Timestamp(r[-1]["exit"]) - pd.Timestamp(r[0]["entry"])).days),
                "trading_days": int(xi - ei), "compound_return": float(np.prod([1 + float(x["ret"]) for x in r]) - 1)}

    if not runs:
        return {"max_len": 0, "n_runs": 0, "longest": None, "worst_by_return": None, "n_runs_3plus": 0, "n_runs_5plus": 0, "length_counts": {}}
    infos = [info(r) for r in runs]
    return {"max_len": max(i["n"] for i in infos), "n_runs": len(infos), "longest": max(infos, key=lambda i: (i["n"], i["calendar_days"])),
            "worst_by_return": min(infos, key=lambda i: i["compound_return"]),
            "n_runs_3plus": int(sum(i["n"] >= 3 for i in infos)), "n_runs_5plus": int(sum(i["n"] >= 5 for i in infos)),
            "length_counts": {str(k): int(v) for k, v in pd.Series([i["n"] for i in infos]).value_counts().sort_index().items()}}


# ---- five-ETF book ---------------------------------------------------------------------------

def combine(sleeves: dict) -> pd.Series:
    return pd.concat(list(sleeves.values()), axis=1).mean(axis=1)


def book_frame(sleeves: dict) -> pd.DataFrame:
    """Book-level equity, return and average exposure, from each sleeve's windowed frame."""
    eq = combine({k: v["equity"] for k, v in sleeves.items()})
    ret = eq.pct_change().fillna(eq.iloc[0] - 1)
    weights = pd.concat([v["weight"] * v["equity"] for v in sleeves.values()], axis=1).sum(axis=1)
    return pd.DataFrame({"equity": eq, "ret": ret, "exposure": weights / eq / len(sleeves)})


def month_end(idx: pd.DatetimeIndex) -> np.ndarray:
    s = idx.to_series()
    return (s.dt.to_period("M") != s.dt.to_period("M").shift(-1)).values


def basket_monthly(o: pd.DataFrame, c: pd.DataFrame, rf: pd.Series, cost_bps: float) -> pd.Series:
    """Share-and-cash bookkeeping. Starts flat on the first row (earning cash interest that day),
    buys equal weights at the next open, then rebalances to equal weight at the open after each month end."""
    n, names = o.shape[1], list(o.columns)
    target, c_rate = np.full(n, 1.0 / n), cost_bps / 1e4
    o_, c_, rf_ = o[names].values, c[names].values, rf.reindex(o.index).fillna(0.0).values
    shares, cash = np.zeros(n), 1.0
    me = month_end(o.index)
    out = np.empty(len(o))
    for t in range(len(o)):
        trade = t == 1 or (t > 1 and me[t - 1])
        if trade:
            value = cash + float(shares @ o_[t])
            w = shares * o_[t] / value
            cost = float(np.abs(target - w).sum()) * value * c_rate
            after = value - cost
            shares = target * after / o_[t]
            cash = 0.0
        else:
            cash *= 1 + rf_[t]
        out[t] = cash + float(shares @ c_[t])
    return pd.Series(out, index=o.index, name="equity")


def basket_monthly_recursive(o: pd.DataFrame, c: pd.DataFrame, rf: pd.Series, cost_bps: float) -> pd.Series:
    """Second implementation: weight recursion instead of shares. Used only to cross-check."""
    names = list(o.columns)
    o_, c_, n = o[names].values, c[names].values, len(names)
    rf_ = rf.reindex(o.index).fillna(0.0).values
    target, rate = np.full(n, 1.0 / n), cost_bps / 1e4
    w, eq, me = np.zeros(n), 1.0, month_end(o.index)
    out = np.empty(len(o))
    out[0] = eq = eq * (1 + rf_[0])
    for t in range(1, len(o)):
        r_on = o_[t] / c_[t - 1] - 1
        grow = 1 + float(w @ r_on)
        if w.sum() > 0:
            w = w * (1 + r_on) / grow
        eq *= grow
        if t == 1 or me[t - 1]:
            eq *= 1 - float(np.abs(target - w).sum()) * rate
            w = target.copy()
        r_id = c_[t] / o_[t] - 1
        grow = 1 + float(w @ r_id)
        w = w * (1 + r_id) / grow
        eq *= grow
        out[t] = eq
    return pd.Series(out, index=o.index, name="equity")


def frame_from_equity(eq: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"equity": eq, "ret": eq.pct_change().fillna(eq.iloc[0] - 1)})


def stats(bt, rf):
    out = {"sharpe": metrics.sharpe(bt["ret"], rf.reindex(bt.index)), "cagr": metrics.cagr(bt["equity"]),
           "max_drawdown": metrics.max_drawdown(bt["equity"])}
    if "weight" in bt:
        out["exposure"] = float(bt["weight"].mean())
    return out


def book_summary(fr: pd.DataFrame, rf: pd.Series) -> dict:
    out = {k: v for k, v in metrics.summary(fr[["equity", "ret"]], rf).items() if k != "exposure"}
    if "exposure" in fr:
        out["exposure"] = float(fr["exposure"].mean())
    return out


# ---- run --------------------------------------------------------------------------------------

def annual_table(s_ret, h_ret, s_idx0):
    yr = pd.DataFrame({"strategy": (1 + s_ret).groupby(s_ret.index.year).prod() - 1,
                       "buyhold": (1 + h_ret).groupby(h_ret.index.year).prod() - 1})
    yr = yr.loc[yr.index > s_idx0.year]
    yr["excess"] = yr.strategy - yr.buyhold
    return yr


def annual_summary(yr: pd.DataFrame) -> dict:
    return {"years": int(len(yr)), "years_strategy_won": int((yr.excess > 0).sum()),
            "up_years_buyhold": int((yr.buyhold > 0).sum()),
            "strategy_won_in_up_years": int(((yr.excess > 0) & (yr.buyhold > 0)).sum()),
            "down_years_buyhold": int((yr.buyhold < 0).sum()),
            "strategy_won_in_down_years": int(((yr.excess > 0) & (yr.buyhold < 0)).sum()),
            "best": {int(k): float(v) for k, v in yr.excess.nlargest(3).items()},
            "worst": {int(k): float(v) for k, v in yr.excess.nsmallest(3).items()},
            "partial_last_year": int(yr.index[-1]),
            "by_year": {int(y): {"strategy": float(r.strategy), "buyhold": float(r.buyhold)} for y, r in yr.iterrows()}}


def independent_spy(px: pd.DataFrame, rf: pd.Series, start: pd.Timestamp, engine_equity: float) -> dict:
    """No engine and no strategy module: walk the raw closes with explicit windows, then compound by hand."""
    c, o, rf_, idx = px["close"].values, px["open"].values, rf.reindex(px.index).fillna(0.0).values, px.index
    n, flat, trades, days_new_high = len(c), True, [], 0
    held = np.zeros(n)
    entry_day = None
    for t in range(n):
        up = max(c[t - ENTRY:t]) if t >= ENTRY else None
        lo = min(c[t - EXIT:t]) if t >= EXIT else None
        if up is not None and c[t] > up:
            days_new_high += 1
        if flat:
            if up is not None and c[t] > up:
                flat, entry_day = False, t + 1
        elif lo is not None and c[t] < lo:
            trades.append((entry_day, t + 1))
            flat, entry_day = True, None
        if t + 1 < n:
            held[t + 1] = 0.0 if flat else 1.0
    if entry_day is not None and entry_day < n:
        trades.append((entry_day, None))
    rate = COST_BPS / 1e4
    s_pos = int(px.index.get_loc(start))
    mult, n_closed = 1.0, 0
    for e, x in trades:
        if x is not None and x < n:
            mult *= o[x] / o[e] * (1 - rate) ** 2
            n_closed += 1
        elif x is None:
            mult *= c[-1] / o[e] * (1 - rate)
        else:   # exit signal on the last bar: still long at the close of the final day
            mult *= c[-1] / o[e] * (1 - rate)
    # interest on flat days (held = 0) inside the reported window
    for t in range(s_pos + 1, n):     # the reported multiple is equity[-1] / equity[0]
        if held[t] == 0.0:
            mult *= 1 + rf_[t]
    return {"new_high_days": int(days_new_high), "share_days_new_high": float(days_new_high / (n - ENTRY)),
            "trades": int(len(trades)), "closed": int(n_closed),
            "exposure": float(held[s_pos:].mean()),
            "final_multiple_hand": float(mult), "final_multiple_engine": float(engine_equity),
            "relative_gap": float(abs(mult / engine_equity - 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    res: dict = {"symbol": SYMBOL, "rules": {"entry": ENTRY, "exit": EXIT, "cost_bps_per_side": COST_BPS, "fill": "next open",
                                             "cash": "3-month Treasury (constant maturity)"}}
    start = db.first_signal_date(px, ENTRY)
    tgt = db.target(px, ENTRY, EXIT)
    bench = pd.Series(0.0, index=px.index)
    bench.loc[start:] = 1.0

    res["checks"] = {"lookahead": checks.lookahead_check(lambda p: db.target(p, ENTRY, EXIT), px),
                     "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
                     "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf)}

    s = research.window(engine.vectorized(px, tgt, COST_BPS, rf), start)
    h = research.window(engine.vectorized(px, bench, COST_BPS, rf), start)
    res["strategy"] = metrics.summary(s, rf)
    res["buyhold"] = metrics.summary(h, rf)
    led = ledger(s["weight"], px["open"].loc[start:], COST_BPS, px["close"].loc[start:])
    led_pub = led[["entry", "exit", "ret", "hold_days", "open"]].copy()
    led_pub.to_csv(OUT / "trades.csv", index=False)
    res["trades"] = trade_stats(led)
    res["trade_count"] = int(len(led))
    res["streaks"] = streaks(led)

    # the ledger must agree with the engine's own trade list (metrics.trades adds one day of cash
    # interest on the exit day, so trade returns differ by that amount; dates must match exactly)
    mt = metrics.trades(s["weight"], s["ret"])
    same_dates = bool(pd.to_datetime(mt["entry"]).tolist() == pd.to_datetime(led["entry"]).tolist())
    closed_mt = mt[~mt["open"]]; closed_led = led[~led["open"]]
    res["checks"]["ledger_vs_engine_trades"] = {"same_entry_dates": same_dates, "count_engine": int(len(mt)), "count_ledger": int(len(led)),
                                                "max_abs_return_gap_closed": float((closed_mt["return"].values - closed_led["ret"].values).__abs__().max()) if len(closed_led) else 0.0}

    yr = annual_table(s["ret"], h["ret"], s.index[0])
    yr.to_csv(OUT / "annual.csv")
    res["annual"] = annual_summary(yr)

    res["independent"] = independent_spy(px, rf, start, s["equity"].iloc[-1] / s["equity"].iloc[0])
    res["independent"]["exposure_engine"] = float(s["weight"].mean())

    # deliberate bug, for the video: forget the one-day shift
    cheat = research.window(engine.same_bar_cheat(px, tgt), start)
    res["same_bar_cheat"] = {"cagr": metrics.cagr(cheat["equity"]), "final_multiple": float(cheat["equity"].iloc[-1] / cheat["equity"].iloc[0])}

    zero = research.window(engine.vectorized(px, tgt, COST_BPS, rf * 0), start)
    res["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero["equity"]), "final_multiple": float(zero["equity"].iloc[-1])}

    # health 1: costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, COST_GRID)
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")

    # health 2: parameters, all on a common window starting at the longest entry period
    grid_start = db.first_signal_date(px, max(ENTRIES))
    cache = {(a, b): db.target(px, a, b) for a in ENTRIES for b in EXITS}
    rows = []
    for (a, b), t in cache.items():
        bt = research.window(engine.vectorized(px, t, COST_BPS, rf), grid_start)
        rows.append({"entry": a, "exit": b, **stats(bt, rf), "trades": int(len(metrics.trades(bt["weight"], bt["ret"])))})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "param_grid.csv", index=False)
    bh_grid = research.window(engine.vectorized(px, bench, COST_BPS, rf), grid_start)
    bh_grid_stats = stats(bh_grid, rf)
    g = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    rank = lambda a, b: int(g.index[(g.entry == a) & (g.exit == b)][0]) + 1
    res["param_grid"] = {"window_start": str(grid_start.date()), "n": int(len(grid)), "entries": ENTRIES, "exits": EXITS,
                         "rank_20_10_by_sharpe": rank(20, 10), "rank_55_20_by_sharpe": rank(55, 20),
                         "best": g.iloc[0].to_dict(), "worst": g.iloc[-1].to_dict(), "buyhold": bh_grid_stats,
                         "n_beat_buyhold_cagr": int((grid.cagr > bh_grid_stats["cagr"]).sum()),
                         "n_beat_buyhold_sharpe": int((grid.sharpe > bh_grid_stats["sharpe"]).sum()),
                         "n_shallower_drawdown_than_buyhold": int((grid.max_drawdown > bh_grid_stats["max_drawdown"]).sum()),
                         "rows": grid.to_dict("records")}

    # named variants, common window starting at the longer entry period so they compare fairly
    v_start = db.first_signal_date(px, 55)
    bh_v = research.window(engine.vectorized(px, bench, COST_BPS, rf), v_start)
    variants = {"window_start": str(v_start.date()), "buyhold": stats(bh_v, rf), "n_variants_besides_main": 2}
    for name, (a, b, hl) in {"main_20_10": (20, 10, False), "turtle_55_20": (55, 20, False), "high_low_20_10": (20, 10, True)}.items():
        t = db.target(px, a, b, hl)
        bt = research.window(engine.vectorized(px, t, COST_BPS, rf), v_start)
        lv = ledger(bt["weight"], px["open"].loc[v_start:], COST_BPS, px["close"].loc[v_start:])
        variants[name] = {"entry": a, "exit": b, **stats(bt, rf), "trades": int(len(lv)), "win_rate": trade_stats(lv)["win_rate"],
                          "payoff_ratio": trade_stats(lv)["payoff_ratio"], "max_loss_streak": streaks(lv)["max_len"]}
    res["variants"] = variants

    # health 3: choose on the first part, look once at the second
    rows = []
    for (a, b), t in cache.items():
        bt = research.window(engine.vectorized(px, t, COST_BPS, rf), grid_start, "2009-12-31")
        rows.append({"entry": a, "exit": b, **stats(bt, rf)})
    best_is = pd.DataFrame(rows).sort_values("sharpe", ascending=False).iloc[0]
    res["split"] = {"in_sample": f"{grid_start.date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}"}
    for name, (a, b) in {"best_in_sample": (int(best_is.entry), int(best_is.exit)), "main_20_10": (ENTRY, EXIT)}.items():
        full = engine.vectorized(px, cache[(a, b)], COST_BPS, rf)
        res["split"][name] = {"entry": a, "exit": b, "in_sample": stats(research.window(full, grid_start, "2009-12-31"), rf),
                              "out_of_sample": stats(research.window(full, SPLIT), rf)}
    bfull = engine.vectorized(px, bench, COST_BPS, rf)
    res["split"]["buyhold"] = {"in_sample": stats(research.window(bfull, grid_start, "2009-12-31"), rf),
                               "out_of_sample": stats(research.window(bfull, SPLIT), rf)}

    # ---- five ETFs ------------------------------------------------------------------------------
    pxs = {k: data.load_prices(k).loc[COMMON_START:] for k in ETFS}
    idx0 = pxs["GLD"].index
    for k, p in pxs.items():
        if not p.index.equals(idx0):
            raise SystemExit(f"{k} calendar differs from GLD after {COMMON_START}")
    rf_p = data.daily_rate(data.load_rate("%3MTCM"), idx0)
    p_start = db.first_signal_date(pxs["GLD"], ENTRY)
    targets = {k: db.target(p, ENTRY, EXIT) for k, p in pxs.items()}
    bench_p = pd.Series(0.0, index=idx0)
    bench_p.loc[p_start:] = 1.0

    res["etf_checks"] = {}
    res["etfs"] = {}
    sleeves, sleeves_bh = {}, {}
    all_led = []
    for k, p in pxs.items():
        res["etf_checks"][k] = {"lookahead": checks.lookahead_check(lambda q: db.target(q, ENTRY, EXIT), p),
                                "reconcile_strategy": checks.reconcile(p, targets[k], COST_BPS, rf_p),
                                "reconcile_buyhold": checks.reconcile(p, bench_p, COST_BPS, rf_p)}
        full = engine.vectorized(p, targets[k], COST_BPS, rf_p)
        bh = engine.vectorized(p, bench_p, COST_BPS, rf_p)
        sleeves[k] = research.window(full, p_start)
        sleeves_bh[k] = research.window(bh, p_start)
        lk = ledger(sleeves[k]["weight"], p["open"].loc[p_start:], COST_BPS, p["close"].loc[p_start:])
        all_led.append(lk.assign(symbol=k))
        res["etfs"][k] = {"strategy": metrics.summary(sleeves[k], rf_p), "buyhold": metrics.summary(sleeves_bh[k], rf_p),
                          "trades": trade_stats(lk), "streaks": streaks(lk)}
        res["etfs"][k]["beat_buyhold_cagr"] = bool(res["etfs"][k]["strategy"]["cagr"] > res["etfs"][k]["buyhold"]["cagr"])
        res["etfs"][k]["beat_buyhold_sharpe"] = bool(res["etfs"][k]["strategy"]["sharpe"] > res["etfs"][k]["buyhold"]["sharpe"])
    pd.concat(all_led).to_csv(OUT / "portfolio_trades.csv", index=False)

    book = book_frame(sleeves)
    bo, bc = pd.DataFrame({k: p["open"] for k, p in pxs.items()}), pd.DataFrame({k: p["close"] for k, p in pxs.items()})
    bo, bc = bo.loc[p_start:], bc.loc[p_start:]
    basket = basket_monthly(bo, bc, rf_p, COST_BPS)
    basket2 = basket_monthly_recursive(bo, bc, rf_p, COST_BPS)
    res["basket_cross_check"] = {"max_relative_gap": float((basket / basket2 - 1).abs().max())}
    basket_fr = frame_from_equity(basket)
    plain = book_frame(sleeves_bh)  # 1/5 each, never rebalanced

    # independent check of the book: rebuild every sleeve with the event-loop engine
    ev = {k: engine.event_loop(pxs[k], targets[k], COST_BPS, rf_p) for k in ETFS}
    ev_eq = combine({k: research.window(pd.DataFrame({"ret": v.pct_change().fillna(v.iloc[0] - 1), "equity": v}), p_start)["equity"] for k, v in ev.items()})
    res["basket_cross_check"]["book_vs_event_loop_max_gap"] = float((book["equity"] / ev_eq - 1).abs().max())

    res["portfolio"] = {"start": str(p_start.date()), "end": str(idx0[-1].date()),
                        "strategy": book_summary(book, rf_p), "basket_monthly_rebalance": book_summary(basket_fr, rf_p),
                        "basket_no_rebalance": book_summary(plain, rf_p),
                        "sleeve_method": "average of five independent 1/5 sleeves; each flat sleeve earns the 3-month Treasury rate on its own cash",
                        "total_trades": int(sum(res["etfs"][k]["trades"]["n_total"] for k in ETFS)),
                        "total_closed": int(sum(res["etfs"][k]["trades"]["n_closed"] for k in ETFS))}

    pyr = annual_table(book["ret"], basket_fr["ret"], book.index[0])
    res["portfolio"]["annual"] = annual_summary(pyr)
    pyr.to_csv(OUT / "portfolio_annual.csv")

    def run_book(a, b, cost, rf_series, start, end=None, cache_t=None):
        legs = {}
        for k, p in pxs.items():
            t = (cache_t or {}).get((k, a, b))
            if t is None:
                t = db.target(p, a, b)
            legs[k] = research.window(engine.vectorized(p, t, cost, rf_series), start, end)
        return book_frame(legs)

    def run_basket(cost, rf_series, start, end=None):
        b = basket_monthly(bo, bc, rf_series, cost)
        return research.window(frame_from_equity(b), start, end)

    # portfolio with cash earning nothing
    zero_book = run_book(ENTRY, EXIT, COST_BPS, rf_p * 0, p_start)
    res["portfolio"]["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero_book["equity"]), "final_multiple": float(zero_book["equity"].iloc[-1])}

    # portfolio costs
    rows = []
    for c in COST_GRID:
        rows.append({"cost_bps": c, "strategy_cagr": metrics.cagr(run_book(ENTRY, EXIT, c, rf_p, p_start)["equity"]),
                     "buyhold_cagr": metrics.cagr(run_basket(c, rf_p, p_start)["equity"])})
    psweep = pd.DataFrame(rows)
    psweep.to_csv(OUT / "portfolio_cost_sweep.csv", index=False)
    res["portfolio"]["cost_sweep"] = psweep.to_dict("records")
    res["portfolio"]["breakeven_vs_basket_bps"] = research.breakeven_bps(psweep, "strategy_cagr", "buyhold_cagr")

    # portfolio parameters and split, common window starting at the longest entry period
    pg_start = db.first_signal_date(pxs["GLD"], max(ENTRIES))
    tc = {(k, a, b): db.target(p, a, b) for k, p in pxs.items() for a in ENTRIES for b in EXITS}
    rows, rows_is = [], []
    for a in ENTRIES:
        for b in EXITS:
            bk = run_book(a, b, COST_BPS, rf_p, pg_start, cache_t=tc)
            rows.append({"entry": a, "exit": b, "cagr": metrics.cagr(bk["equity"]), "max_drawdown": metrics.max_drawdown(bk["equity"]),
                         "sharpe": metrics.sharpe(bk["ret"], rf_p.reindex(bk.index))})
            bki = run_book(a, b, COST_BPS, rf_p, pg_start, "2009-12-31", cache_t=tc)
            rows_is.append({"entry": a, "exit": b, "cagr": metrics.cagr(bki["equity"]), "max_drawdown": metrics.max_drawdown(bki["equity"]),
                            "sharpe": metrics.sharpe(bki["ret"], rf_p.reindex(bki.index))})
    pgrid = pd.DataFrame(rows)
    pgrid.to_csv(OUT / "portfolio_param_grid.csv", index=False)
    bk_grid = run_basket(COST_BPS, rf_p, pg_start)
    bk_stats = stats(bk_grid, rf_p)
    pg = pgrid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    prank = lambda a, b: int(pg.index[(pg.entry == a) & (pg.exit == b)][0]) + 1
    res["portfolio"]["param_grid"] = {"window_start": str(pg_start.date()), "n": int(len(pgrid)),
                                      "rank_20_10_by_sharpe": prank(20, 10), "rank_55_20_by_sharpe": prank(55, 20),
                                      "best": pg.iloc[0].to_dict(), "worst": pg.iloc[-1].to_dict(), "basket": bk_stats,
                                      "n_beat_basket_cagr": int((pgrid.cagr > bk_stats["cagr"]).sum()),
                                      "n_beat_basket_sharpe": int((pgrid.sharpe > bk_stats["sharpe"]).sum()),
                                      "n_shallower_drawdown_than_basket": int((pgrid.max_drawdown > bk_stats["max_drawdown"]).sum()),
                                      "rows": pgrid.to_dict("records")}
    best_pis = pd.DataFrame(rows_is).sort_values("sharpe", ascending=False).iloc[0]
    psplit = {"in_sample": f"{pg_start.date()} to 2009-12-31 (only about 5 years)", "out_of_sample": f"{SPLIT} to {idx0[-1].date()}"}
    for name, (a, b) in {"best_in_sample": (int(best_pis.entry), int(best_pis.exit)), "main_20_10": (ENTRY, EXIT)}.items():
        psplit[name] = {"entry": a, "exit": b, "in_sample": stats(run_book(a, b, COST_BPS, rf_p, pg_start, "2009-12-31", cache_t=tc), rf_p),
                        "out_of_sample": stats(run_book(a, b, COST_BPS, rf_p, SPLIT, cache_t=tc), rf_p)}
    psplit["basket"] = {"in_sample": stats(run_basket(COST_BPS, rf_p, pg_start, "2009-12-31"), rf_p),
                        "out_of_sample": stats(run_basket(COST_BPS, rf_p, SPLIT), rf_p)}
    res["portfolio"]["split"] = psplit

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))

    ch = db.channels(px, ENTRY, EXIT)
    pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"], "strategy_dd": metrics.drawdown(s["equity"]),
                  "buyhold_dd": metrics.drawdown(h["equity"]), "weight": s["weight"], "close": px["close"].loc[start:],
                  "upper": ch["upper"].loc[start:], "lower": ch["lower"].loc[start:]}).to_csv(OUT / "curves.csv")
    pc = pd.DataFrame({"book": book["equity"], "basket": basket_fr["equity"], "basket_no_rebalance": plain["equity"],
                       "book_dd": metrics.drawdown(book["equity"]), "basket_dd": metrics.drawdown(basket_fr["equity"])})
    for k in ETFS:
        pc[k + "_strategy"] = sleeves[k]["equity"]
        pc[k + "_buyhold"] = sleeves_bh[k]["equity"]
    pc.to_csv(OUT / "portfolio_curves.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], color="grey", label="Buy & hold"); ax.plot(s["equity"], color="black", label="20/10 breakout")
    ax.set_yscale("log"); ax.legend(); fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)

    print(json.dumps({k: res[k] for k in ["checks", "strategy", "buyhold", "trade_count", "trades", "streaks", "independent", "same_bar_cheat", "idle_cash_zero_rate", "breakeven_vs_buyhold_bps", "variants", "annual", "basket_cross_check"]}, indent=1, default=str))
    print({k: v for k, v in res["param_grid"].items() if k != "rows"}); print(res["split"])
    print(json.dumps({k: v for k, v in res["portfolio"].items() if k not in ("param_grid",)}, indent=1, default=str))
    print({k: v for k, v in res["portfolio"]["param_grid"].items() if k != "rows"})
    print(json.dumps({k: {"strategy": {m: v["strategy"][m] for m in ("cagr", "sharpe", "max_drawdown", "n_trades")},
                          "buyhold": {m: v["buyhold"][m] for m in ("cagr", "sharpe", "max_drawdown")}, "win": v["trades"]["win_rate"], "payoff": v["trades"]["payoff_ratio"],
                          "maxstreak": v["streaks"]["max_len"], "beat": v["beat_buyhold_cagr"], "beat_sh": v["beat_buyhold_sharpe"]} for k, v in res["etfs"].items()}, indent=1))
    print(res["etf_checks"])


if __name__ == "__main__":
    main()
