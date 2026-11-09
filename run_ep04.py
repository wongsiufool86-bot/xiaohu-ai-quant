"""Episode 4: RSI dip buying on SPY. Writes every number to results/ep04/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep04.py   # data format: see backtest/data.py
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
from strategies import rsi_dip

OUT = Path(__file__).parent / "results" / "ep04"
SYMBOL, PERIOD, BUY, SELL, COST_BPS = "SPY", 14, 30.0, 70.0, 5.0
OTHERS = ["QQQ", "IWM", "TLT", "GLD"]
SPLIT = "2010-01-01"
PERIODS = [2, 5, 7, 10, 14, 21, 28]
BUYS = [10, 15, 20, 25, 30, 35, 40]
MIN_DD = -0.15
N_RANDOM = 2000
DAYS = 252


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


def independent_multiple(px: pd.DataFrame, rf: pd.Series, period: int, buy: float, sell: float, cost_bps: float) -> float:
    """Final multiple without the backtest engine: plain lists, its own RSI, one factor per trade."""
    closes, opens, rates = list(map(float, px["close"])), list(map(float, px["open"])), list(map(float, rf.reindex(px.index).fillna(0.0)))
    n, keep = len(closes), 1.0 - cost_bps / 1e4
    want = [0] * n
    up = down = 0.0
    state = 0
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        g, l = max(change, 0.0), max(-change, 0.0)
        if i <= period:
            up += g / period
            down += l / period
        else:
            up = (up * (period - 1) + g) / period
            down = (down * (period - 1) + l) / period
        if i >= period:
            if down == 0.0:
                value = 50.0 if up == 0.0 else 100.0
            else:
                value = 100.0 - 100.0 / (1.0 + up / down)
            if state == 0 and value < buy:
                state = 1
            elif state == 1 and value > sell:
                state = 0
        want[i] = state
    multiple, held, entry = 1.0, 0, 0.0
    for i in range(period, n):
        if want[i - 1] == 1 and held == 0:
            held, entry = 1, opens[i]
            multiple *= keep
        elif want[i - 1] == 0 and held == 1:
            multiple *= opens[i] / entry * keep
            held = 0
        if held == 0:
            multiple *= 1.0 + rates[i]
    if held == 1:
        multiple *= closes[-1] / entry
    return multiple


def stats(bt, rf):
    # A parameter pair that never trades only earns cash: its excess return is floating-point noise, so no Sharpe.
    sharpe = metrics.sharpe(bt["ret"], rf.loc[bt.index]) if bt["weight"].sum() > 0 else float("nan")
    return {"sharpe": sharpe, "cagr": metrics.cagr(bt["equity"]),
            "max_drawdown": metrics.max_drawdown(bt["equity"]), "exposure": float(bt["weight"].mean())}


def annualize(growth: float, days: int) -> float:
    return float(growth ** (DAYS / days) - 1) if days > 0 else float("nan")


def fair(px, rf, s, h, cost_bps) -> dict:
    """In-market days vs the rest, plus buy & hold scaled down to the strategy's average exposure."""
    o, c = px["open"], px["close"]
    r_on = (o / c.shift(1) - 1).reindex(s.index).fillna(0.0)
    r_id = (c / o - 1).reindex(s.index)
    held, before = s["weight"], s["weight"].shift(1).fillna(0.0)
    b_held, b_before = h["weight"], h["weight"].shift(1).fillna(0.0)
    inside = (1 + before * r_on) * (1 + held * r_id)
    outside = (1 + (b_before - before) * r_on) * (1 + (b_held - held) * r_id)
    total = (1 + b_before * r_on) * (1 + b_held * r_id)
    n_in, n_out = int((held > 0).sum()), int((b_held > 0).sum() - (held > 0).sum())
    keep = 1 - cost_bps / 1e4
    sides = int(round(s["turnover"].sum()))
    g_in, g_out, g_all = float(inside.prod()), float(outside.prod()), float(total.prod())
    full = research.window(engine.vectorized(px, pd.Series(0.0, index=px.index).where(px.index < s.index[0], float(s["weight"].mean())), cost_bps, rf), s.index[0])
    return {"days_in": n_in, "days_out": n_out, "days_total": n_in + n_out, "share_in": n_in / (n_in + n_out),
            "market_growth_in": g_in, "market_growth_out": g_out, "market_growth_all": g_all,
            "decomposition_gap": abs(g_in * g_out / g_all - 1),
            "ann_market_in": annualize(g_in, n_in), "ann_market_out": annualize(g_out, n_out), "ann_market_all": annualize(g_all, n_in + n_out),
            "ann_strategy_in_net": annualize(g_in * keep ** sides, n_in), "sides": sides,
            "scaled_buyhold": {"weight": float(s["weight"].mean()), **stats(full, rf), "final_multiple": float(full["equity"].iloc[-1])}}


def random_entry_test(px: pd.DataFrame, s: pd.DataFrame, log: pd.DataFrame, n: int = N_RANDOM, seed: int = 0) -> dict:
    """Keep the holding lengths, scatter the flat time at random, compare cumulative open-to-open log return."""
    o = px["open"].reindex(s.index).to_numpy()
    last_close = float(px["close"].reindex(s.index).iloc[-1])
    u = np.log(np.append(o[1:], last_close) / o)[1:]         # unit j: open of day j+1 to the next open
    lengths = log["days_held"].to_numpy()
    held_days = s["weight"].to_numpy()[1:] > 0
    actual = float(u[held_days].sum())
    cum = np.concatenate([[0.0], np.cumsum(u)])
    T, L = len(u), int(lengths.sum())
    k, free, rng, out = len(lengths), T - L, np.random.default_rng(seed), np.empty(n)
    for i in range(n):
        order = rng.permutation(lengths)
        slots = np.sort(rng.choice(free + k, size=k, replace=False))
        starts = slots - np.arange(k) + np.concatenate([[0], np.cumsum(order)[:-1]])
        out[i] = (cum[starts + order] - cum[starts]).sum()
    return {"n": n, "seed": seed, "actual_log_return": actual, "median_random": float(np.median(out)),
            "p05": float(np.quantile(out, 0.05)), "p95": float(np.quantile(out, 0.95)),
            "percentile_of_actual": float((out < actual).mean()), "units_total": T, "units_held": L, "trades": k,
            "histogram": {"edges": np.histogram(out, bins=40)[1].round(4).tolist(), "counts": np.histogram(out, bins=40)[0].tolist()}}


def trade_summary(log: pd.DataFrame) -> dict:
    closed = log[~log["open"]]
    r = closed["return"]
    return {"n_all": int(len(log)), "n_closed": int(len(closed)), "open_at_end": bool(log["open"].any()),
            "n_won": int((r > 0).sum()), "n_lost": int((r <= 0).sum()),
            "win_rate": float((r > 0).mean()), "avg_return": float(r.mean()), "median_return": float(r.median()),
            "avg_win": float(r[r > 0].mean()), "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else None,
            "best": float(r.max()), "worst": float(r.min()),
            "avg_days_held": float(closed["days_held"].mean()), "median_days_held": float(closed["days_held"].median()),
            "max_days_held": int(closed["days_held"].max()), "min_days_held": int(closed["days_held"].min()),
            "avg_mae": float(log["mae_close"].mean()), "median_mae": float(log["mae_close"].median()),
            "worst_mae": float(log["mae_close"].min()), "worst_mae_low": float(log["mae_low"].min()),
            "worst_mae_entry": str(log.loc[log["mae_close"].idxmin(), "entry"].date()),
            "n_mae_over_10": int((log["mae_close"] <= -0.10).sum()), "n_mae_over_20": int((log["mae_close"] <= -0.20).sum()),
            "n_never_under_water": int((log["mae_close"] >= 0).sum()),
            "avg_below_peak_at_signal": float(log["below_peak_at_signal"].mean()),
            "median_below_peak_at_signal": float(log["below_peak_at_signal"].median())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    res: dict = {"symbol": SYMBOL, "rules": {"rsi_period": PERIOD, "buy_below": BUY, "sell_above": SELL, "smoothing": "Wilder",
                                             "cost_bps_per_side": COST_BPS, "fill": "next open", "cash": "3-month Treasury (constant maturity)"},
                 "data_validation": {k: v for k, v in data.validate(px).items() if k not in ("big_moves", "gaps_over_4_calendar_days")}}
    start = rsi_dip.first_signal_date(px, PERIOD)
    tgt = rsi_dip.target(px, PERIOD, BUY, SELL)
    bench = pd.Series(0.0, index=px.index)
    bench.loc[start:] = 1.0

    res["checks"] = {"lookahead": checks.lookahead_check(lambda p: rsi_dip.target(p, PERIOD, BUY, SELL), px),
                     "reconcile_strategy": checks.reconcile(px, tgt, COST_BPS, rf),
                     "reconcile_buyhold": checks.reconcile(px, bench, COST_BPS, rf)}

    s = research.window(engine.vectorized(px, tgt, COST_BPS, rf), start)
    h = research.window(engine.vectorized(px, bench, COST_BPS, rf), start)
    res["strategy"] = metrics.summary(s, rf)
    res["buyhold"] = metrics.summary(h, rf)

    log = rsi_dip.trade_log(px, s["weight"], s["ret"])
    res["trade_count"] = int(len(log))
    res["trades"] = trade_summary(log)
    log.to_csv(OUT / "trades.csv", index=False)

    # bypass the engine
    r_plain = rsi_dip.rsi(px["close"], PERIOD).loc[start:]
    mult = independent_multiple(px, rf, PERIOD, BUY, SELL, COST_BPS)
    res["independent"] = {"final_multiple_no_engine": mult, "final_multiple_engine": float(s["equity"].iloc[-1]),
                          "relative_gap": abs(mult / float(s["equity"].iloc[-1]) - 1),
                          "days_rsi_below_buy": int((r_plain < BUY).sum()), "days_rsi_above_sell": int((r_plain > SELL).sum()),
                          "share_days_below_buy": float((r_plain < BUY).mean()), "rsi_days": int(r_plain.notna().sum())}

    yr = pd.DataFrame({"strategy": (1 + s["ret"]).groupby(s.index.year).prod() - 1,
                       "buyhold": (1 + h["ret"]).groupby(h.index.year).prod() - 1})
    yr = yr.loc[yr.index > s.index[0].year]
    yr["excess"] = yr.strategy - yr.buyhold
    yr["buys"] = pd.Series(log.groupby(pd.to_datetime(log.entry).dt.year).size()).reindex(yr.index).fillna(0).astype(int)
    yr["exposure"] = s["weight"].groupby(s.index.year).mean().reindex(yr.index)
    yr.to_csv(OUT / "annual.csv")
    res["annual"] = {"years": int(len(yr)), "years_strategy_won": int((yr.excess > 0).sum()),
                     "up_years_buyhold": int((yr.buyhold > 0).sum()), "down_years_buyhold": int((yr.buyhold < 0).sum()),
                     "strategy_won_in_up_years": int(((yr.excess > 0) & (yr.buyhold > 0)).sum()),
                     "strategy_won_in_down_years": int(((yr.excess > 0) & (yr.buyhold < 0)).sum()),
                     "years_with_no_buy": int((yr.buys == 0).sum()), "partial_last_year": int(yr.index[-1]),
                     "buys_by_year": {int(k): int(v) for k, v in yr.buys.items() if v > 0},
                     "busiest_year": int(yr.buys.idxmax()), "busiest_year_buys": int(yr.buys.max()),
                     "best": {int(k): float(v) for k, v in yr.excess.nlargest(3).items()},
                     "worst": {int(k): float(v) for k, v in yr.excess.nsmallest(3).items()}}

    # the big drawdowns of buy & hold, found automatically
    rs = rsi_dip.rsi(px["close"], PERIOD)
    rows = []
    for e in episodes(h["equity"], MIN_DD):
        p, t = e["peak"], e["trough"]
        se = s["equity"].loc[p:t]
        in_ep = log[(pd.to_datetime(log.entry) >= p) & (pd.to_datetime(log.entry) <= t)]
        o_days = rs.loc[p:t] < BUY
        already = o_days & (s["weight"].loc[p:t] > 0)
        first = in_ep.iloc[0] if len(in_ep) else None
        before_peak = log[(pd.to_datetime(log.entry) < p) & (pd.to_datetime(log.exit) > p)]
        held = before_peak.iloc[0] if len(before_peak) else None
        rows.append({"peak": str(p.date()), "trough": str(t.date()),
                     "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
                     "buyhold": e["depth"], "strategy": float(se.iloc[-1] / se.iloc[0] - 1),
                     "days_peak_to_trough": int(len(se)),
                     "buys": int(len(in_ep)), "oversold_days": int(o_days.sum()), "oversold_days_already_held": int(already.sum()),
                     "held_at_trough": bool(s["weight"].loc[t] > 0),
                     "first_buy": str(pd.Timestamp(first["entry"]).date()) if first is not None else None,
                     "first_buy_below_peak": float(first["below_peak_at_signal"]) if first is not None else None,
                     "fell_after_first_buy": float(px["close"].loc[t] / px["open"].loc[first["entry"]] - 1) if first is not None else None,
                     "held_from_before_peak": str(pd.Timestamp(held["entry"]).date()) if held is not None else None})
    res["episodes"] = rows
    pd.DataFrame(rows).to_csv(OUT / "episodes.csv", index=False)

    # the two trades that sat through a bear market, and the calendar year 2008
    worst = log.sort_values("mae_close").head(3)
    res["worst_trades"] = []
    for i, r in worst.iterrows():
        nxt = log.iloc[i + 1]["entry"] if i + 1 < len(log) else None
        res["worst_trades"].append({"entry": str(pd.Timestamp(r.entry).date()), "exit": str(pd.Timestamp(r.exit).date()), "return": float(r["return"]),
                                    "days_held": int(r.days_held), "mae_close": float(r.mae_close), "below_peak_at_signal": float(r.below_peak_at_signal),
                                    "days_to_worst": int(r.days_to_worst), "worst_date": str(pd.Timestamp(r.worst_date).date()),
                                    "next_entry": str(pd.Timestamp(nxt).date()) if nxt is not None else None,
                                    "market_while_flat": float(px["open"].loc[nxt] / px["open"].loc[r.exit] - 1) if nxt is not None else None})
    two = log.sort_values("mae_close").head(2)
    res["trades"]["two_worst_days_share"] = float(two["days_held"].sum() / log["days_held"].sum())
    res["trades"]["two_worst_days"] = int(two["days_held"].sum())
    r08 = rs.loc["2008-01-01":"2008-12-31"]
    w08 = s["weight"].loc["2008-01-01":"2008-12-31"]
    res["year2008"] = {"buys": int(res["annual"]["buys_by_year"].get(2008, 0)), "oversold_days": int((r08 < BUY).sum()),
                       "oversold_days_already_held": int(((r08 < BUY) & (w08 > 0)).sum()), "held_all_year": bool((w08 > 0).all()),
                       "max_rsi": float(r08.max())}

    res["fair"] = fair(px, rf, s, h, COST_BPS)
    res["random_entry"] = random_entry_test(px, s, log)

    # extreme days: is it there when the market moves most?
    cc = px["close"].pct_change().loc[start:]
    w = s["weight"].reindex(cc.index)
    top, bot = cc.nlargest(10), cc.nsmallest(10)
    res["extreme_days"] = {"best10_in": int((w.loc[top.index] > 0).sum()), "worst10_in": int((w.loc[bot.index] > 0).sum())}

    # idle cash earns nothing
    zero = research.window(engine.vectorized(px, tgt, COST_BPS, rf * 0), start)
    res["idle_cash_zero_rate"] = {"cagr": metrics.cagr(zero["equity"]), "final_multiple": float(zero["equity"].iloc[-1])}

    # costs
    sweep = research.cost_sweep(px, tgt, bench, rf, start, [0, 2, 5, 10, 15, 20, 30, 40, 50])
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven_vs_buyhold_bps"] = research.breakeven_bps(sweep, "strategy_cagr", "buyhold_cagr")

    # parameters, common window
    grid_start = rsi_dip.first_signal_date(px, max(PERIODS))
    rows = []
    for per in PERIODS:
        for bu in BUYS:
            bt = research.window(engine.vectorized(px, rsi_dip.target(px, per, bu, SELL), COST_BPS, rf), grid_start)
            rows.append({"period": per, "buy": bu, **stats(bt, rf), "trades": int(len(metrics.trades(bt["weight"], bt["ret"])))})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "param_grid.csv", index=False)
    bh_grid = research.window(engine.vectorized(px, bench, COST_BPS, rf), grid_start)
    bh_stats = stats(bh_grid, rf)
    valid = grid[grid.trades > 0].dropna(subset=["sharpe"])
    g = valid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    res["param_grid"] = {"window_start": str(grid_start.date()), "n": int(len(grid)), "n_no_trades": int((grid.trades == 0).sum()),
                         "rank_of_base_by_sharpe": int(g.index[(g.period == PERIOD) & (g.buy == BUY)][0]) + 1,
                         "n_valid": int(len(g)), "best": g.iloc[0].to_dict(), "worst": g.iloc[-1].to_dict(), "base": grid[(grid.period == PERIOD) & (grid.buy == BUY)].iloc[0].to_dict(),
                         "best_cagr": valid.sort_values("cagr", ascending=False).iloc[0].to_dict(),
                         "buyhold": bh_stats, "n_beat_buyhold_cagr": int((valid.cagr > bh_stats["cagr"]).sum()),
                         "n_beat_buyhold_sharpe": int((valid.sharpe > bh_stats["sharpe"]).sum()),
                         "n_shallower_mdd_than_buyhold": int((valid.max_drawdown > bh_stats["max_drawdown"]).sum()),
                         "rows": grid.astype(object).where(grid.notna(), None).to_dict("records")}

    # variants (the spec allows exactly these two)
    variants = {}
    defs = {"base": (lambda p: rsi_dip.target(p, PERIOD, BUY, SELL), PERIOD),
            "exit_50": (lambda p: rsi_dip.target(p, PERIOD, BUY, 50.0), PERIOD),
            "rsi2_ma5": (lambda p: rsi_dip.target_ma_exit(p, 2, 10.0, 5), 5)}
    for name, (fn, per) in defs.items():
        t = fn(px)
        v_start = max(start, rsi_dip.first_signal_date(px, per))
        bt = research.window(engine.vectorized(px, t, COST_BPS, rf), v_start)
        hb = research.window(engine.vectorized(px, bench, COST_BPS, rf), v_start)
        lg = rsi_dip.trade_log(px, bt["weight"], bt["ret"])
        cost10 = research.window(engine.vectorized(px, t, 10.0, rf), v_start)
        cost20 = research.window(engine.vectorized(px, t, 20.0, rf), v_start)
        f = fair(px, rf, bt, hb, COST_BPS)
        variants[name] = {**stats(bt, rf), "trades": int(len(lg)), "final_multiple": float(bt["equity"].iloc[-1]), "win_rate": trade_summary(lg)["win_rate"],
                          "avg_days_held": trade_summary(lg)["avg_days_held"], "worst_mae": trade_summary(lg)["worst_mae"],
                          "ann_strategy_in_net": f["ann_strategy_in_net"], "ann_market_all": f["ann_market_all"], "scaled_buyhold_cagr": f["scaled_buyhold"]["cagr"], "scaled_buyhold_sharpe": f["scaled_buyhold"]["sharpe"],
                          "cagr_cost_10bp": metrics.cagr(cost10["equity"]), "cagr_cost_20bp": metrics.cagr(cost20["equity"]),
                          "checks": {"lookahead": checks.lookahead_check(fn, px)["passed"], "reconcile": checks.reconcile(px, t, COST_BPS, rf)["passed"]}}
    res["variants"] = variants

    # before / after 2010
    rows = []
    for per in PERIODS:
        for bu in BUYS:
            bt = research.window(engine.vectorized(px, rsi_dip.target(px, per, bu, SELL), COST_BPS, rf), grid_start, "2009-12-31")
            rows.append({"period": per, "buy": bu, **stats(bt, rf)})
    best_is = pd.DataFrame(rows).dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).iloc[0]
    res["split"] = {"in_sample": f"{grid_start.date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {px.index[-1].date()}"}
    for name, (per, bu) in {"best_in_sample": (int(best_is.period), float(best_is.buy)), "base": (PERIOD, BUY)}.items():
        full = engine.vectorized(px, rsi_dip.target(px, per, bu, SELL), COST_BPS, rf)
        res["split"][name] = {"period": per, "buy": bu, "in_sample": stats(research.window(full, grid_start, "2009-12-31"), rf),
                              "out_of_sample": stats(research.window(full, SPLIT), rf)}
    bfull = engine.vectorized(px, bench, COST_BPS, rf)
    res["split"]["buyhold"] = {"in_sample": stats(research.window(bfull, grid_start, "2009-12-31"), rf),
                               "out_of_sample": stats(research.window(bfull, SPLIT), rf)}

    # other markets, same rule
    others = {}
    for sym in OTHERS:
        p2 = data.load_prices(sym)
        rf2 = data.daily_rate(data.load_rate("%3MTCM"), p2.index)
        st2 = rsi_dip.first_signal_date(p2, PERIOD)
        t2 = rsi_dip.target(p2, PERIOD, BUY, SELL)
        b2 = pd.Series(0.0, index=p2.index); b2.loc[st2:] = 1.0
        s2 = research.window(engine.vectorized(p2, t2, COST_BPS, rf2), st2)
        h2 = research.window(engine.vectorized(p2, b2, COST_BPS, rf2), st2)
        lg2 = rsi_dip.trade_log(p2, s2["weight"], s2["ret"])
        f2 = fair(p2, rf2, s2, h2, COST_BPS)
        ts2 = trade_summary(lg2)
        others[sym] = {"start": str(st2.date()), "strategy": stats(s2, rf2), "buyhold": stats(h2, rf2), "final_multiple": float(s2["equity"].iloc[-1]), "buyhold_final_multiple": float(h2["equity"].iloc[-1]),
                       "trades": int(len(lg2)), "win_rate": ts2["win_rate"], "worst_mae": ts2["worst_mae"], "avg_days_held": ts2["avg_days_held"],
                       "ann_market_all": f2["ann_market_all"], "ann_market_in": f2["ann_market_in"], "ann_market_out": f2["ann_market_out"], "ann_strategy_in_net": f2["ann_strategy_in_net"],
                       "scaled_buyhold_cagr": f2["scaled_buyhold"]["cagr"], "scaled_buyhold_sharpe": f2["scaled_buyhold"]["sharpe"], "scaled_buyhold_mdd": f2["scaled_buyhold"]["max_drawdown"],
                       "reconcile": checks.reconcile(p2, t2, COST_BPS, rf2)["passed"]}
    res["others"] = others

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame({"strategy": s["equity"], "buyhold": h["equity"], "strategy_dd": metrics.drawdown(s["equity"]),
                  "buyhold_dd": metrics.drawdown(h["equity"]), "weight": s["weight"],
                  "close": px["close"].loc[start:], "rsi": rs.loc[start:]}).to_csv(OUT / "curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["equity"], color="grey", label="Buy & hold"); ax.plot(s["equity"], color="black", label="RSI dip")
    ax.set_yscale("log"); ax.legend(); fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)
    for key in ["checks", "strategy", "buyhold", "trades", "independent", "fair", "random_entry", "extreme_days", "idle_cash_zero_rate", "breakeven_vs_buyhold_bps", "variants", "annual", "others", "split"]:
        print(key, json.dumps(res[key], indent=None, default=str))
    print(json.dumps(res["episodes"], indent=0)); print({k: v for k, v in res["param_grid"].items() if k != "rows"})


if __name__ == "__main__":
    main()
