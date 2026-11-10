"""Episode 8: 60% SPY + 40% long Treasuries, rebalanced once a year. Writes every number to results/ep08/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep08.py   # data format: see backtest/data.py

What is compared (all on one common window, 2002-07-26 to the last date, set by the bond ETFs' first day):
  SPY buy & hold; 60/40 with no rebalancing; 60/40 rebalanced yearly, quarterly, monthly. TLT is the main bond sleeve,
  IEF is the control. Trades happen at the open of the first trading day of the period; costs are 5 bp per side on the
  rebalancing turnover.

Limits of what this can show: one 24-year window that starts right after a large equity fall, and in which bond yields fell
for most of the time. ETFs are data samples, not recommendations.
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
from strategies import rebalance_6040 as rb

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ep08"
STOCK, BOND, CONTROL, COST_BPS = "SPY", "TLT", "IEF", 5.0
W_STOCK, W_BOND = 0.60, 0.40
SPLIT = "2010-01-01"
FREQS = ["none", "annual", "quarterly", "monthly"]
STOCK_WEIGHTS = [0.4, 0.5, 0.6, 0.7, 0.8]
COST_GRID = [0, 2, 5, 10, 15, 20, 30, 40, 50]
MIN_DD = -0.15
CRISIS_YEAR = 2022


def stats(bt, rf, whole=True):
    eq, ret = bt["equity"], bt["ret"]
    roll = metrics.drawdown(eq)
    trough = roll.idxmin()
    peak = eq.loc[:trough].idxmax()
    out = {"sharpe": metrics.sharpe(ret, rf.loc[bt.index]), "cagr": metrics.cagr(eq), "ann_vol": metrics.ann_vol(ret),
           "max_drawdown": metrics.max_drawdown(eq), "max_dd_peak": str(peak.date()), "max_dd_trough": str(trough.date()),
           "final_multiple": float(eq.iloc[-1] / eq.iloc[0]), "longest_underwater": metrics.longest_underwater(eq)}
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"])
    if "traded" in bt:
        # `whole` = the slice starts at the signal bar, so its second bar is the first purchase, which is not a rebalance
        first_buy = 1 if whole else 0
        out["trades"] = int(bt["traded"].sum())
        out["rebalances"] = int(bt["traded"].sum()) - first_buy
        reb_turnover = bt["turnover"].iloc[2:] if whole else bt["turnover"]
        out["turnover_per_year"] = float(reb_turnover.sum() / ((bt.index[-1] - bt.index[0]).days / 365.25))
        out["total_cost_multiplier"] = float((1 - bt["cost"]).prod())
    return out


def independent_replay(opens: dict, closes: dict, dates: list, w: dict, freq: str, rf0: float, cost_bps: float) -> dict:
    """A separate ledger: no call into engine.py or rebalance_6040.py. Plain lists, dates checked by hand.
    Day 0 only decides, so it earns cash interest; compare with equity[-1] / equity[0]."""
    names = list(w)
    fee = cost_bps / 1e4
    shares = {k: 0.0 for k in names}
    equity, n_reb, cost_paid, by_year = [1.0 + rf0], 0, 0.0, {}
    cash = equity[0]
    month_of = lambda d: (d.year, d.month)
    for i in range(1, len(dates)):
        first_of_period = False
        if month_of(dates[i]) != month_of(dates[i - 1]):
            if freq == "monthly":
                first_of_period = True
            elif freq == "quarterly":
                first_of_period = dates[i].month in (1, 4, 7, 10)
            elif freq == "annual":
                first_of_period = dates[i].month == 1
        if i == 1 or first_of_period:
            open_val = {k: shares[k] * opens[k][i] for k in names}
            total = cash + sum(open_val.values())
            traded_notional = sum(abs(w[k] * total - open_val[k]) for k in names)
            paid = traded_notional * fee
            cost_paid += paid
            for k in names:
                shares[k] = w[k] * (total - paid) / opens[k][i]
            cash = 0.0
            n_reb += 0 if i == 1 else 1
        equity.append(cash + sum(shares[k] * closes[k][i] for k in names))
    peak, mdd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    years = (dates[-1] - dates[0]).days / 365.25
    last_by_year = {}
    for d, v in zip(dates, equity):
        last_by_year[d.year] = v
    return {"final_multiple": float(equity[-1] / equity[0]), "cagr": float((equity[-1] / equity[0]) ** (1 / years) - 1),
            "max_drawdown": float(mdd), "rebalances": n_reb, "last_by_year": last_by_year}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = {s: data.load_prices(s) for s in [STOCK, BOND, CONTROL]}
    res: dict = {"rules": {"stock": STOCK, "bond": BOND, "control": CONTROL, "stock_weight": W_STOCK, "bond_weight": W_BOND,
                           "rebalance_day": "open of the first trading day of the calendar period",
                           "annual_month": "January", "cost_bps_per_side": COST_BPS,
                           "cost_on": "sum of |target weight - drifted weight| at the open, x equity, x bps",
                           "cash": "fully invested after the first trade; only the signal bar earns the 3-month rate, like the other episodes"},
                 "data_quality": {s: {k: v for k, v in data.validate(p).items() if k != "big_moves"} | {"big_moves_n": len(data.validate(p)["big_moves"])}
                                  for s, p in full.items()},
                 "scope_note": "One window: 2002-07-26 onward, set by the first day of the bond ETFs. ETFs are data samples, not recommendations."}
    start = max(p.index[0] for p in full.values())
    frames = {s: p.loc[start:] for s, p in full.items()}
    idx = frames[STOCK].index
    same_days = all(frames[s].index.equals(idx) for s in frames)
    if not same_days:
        raise SystemExit("trading days differ between ETFs")
    opens = pd.DataFrame({s: p["open"] for s, p in frames.items()})
    closes = pd.DataFrame({s: p["close"] for s, p in frames.items()})
    rf = data.daily_rate(data.load_rate("%3MTCM"), idx)
    res["window"] = {"start": str(idx[0].date()), "first_trade": str(idx[1].date()), "end": str(idx[-1].date()), "days": int(len(idx)),
                     "years": float((idx[-1] - idx[0]).days / 365.25), "same_trading_days": same_days,
                     "spy_first_available": str(full[STOCK].index[0].date()), "tlt_first_available": str(full[BOND].index[0].date()),
                     "ief_first_available": str(full[CONTROL].index[0].date()),
                     "spy_years_lost_by_common_start": float((idx[0] - full[STOCK].index[0]).days / 365.25)}

    w_main = {STOCK: W_STOCK, BOND: W_BOND}
    w_ctrl = {STOCK: W_STOCK, CONTROL: W_BOND}
    sim = lambda w, f="annual", cost=COST_BPS, month=1, o=opens, c=closes, r=rf: rb.simulate(o, c, w, f, cost, r, month)
    runs = {f: sim(w_main, f) for f in FREQS}
    runs_ief = {f: sim(w_ctrl, f) for f in FREQS}
    solo = {s: sim({s: 1.0}, "none") for s in [STOCK, BOND, CONTROL]}
    s_all = runs["annual"]

    # --- portfolio engine against the repo's single-asset engine ------------------------------
    spy_engine = engine.vectorized(frames[STOCK], pd.Series(1.0, index=idx), COST_BPS, rf)
    tie = float((solo[STOCK]["equity"] / spy_engine["equity"] - 1).abs().max())

    # --- mandatory checks ---------------------------------------------------------------------
    joint = pd.concat({"o": opens, "c": closes}, axis=1)
    run_joint = lambda p, w=w_main, f="annual": rb.simulate(p["o"], p["c"], w, f, COST_BPS, rf.reindex(p.index))
    lookahead = {f: checks.lookahead_check(lambda p: run_joint(p, f=f)["equity"], joint) for f in FREQS}
    lookahead_w = {f: checks.lookahead_check(lambda p: run_joint(p, f=f)[f"w_{STOCK}"], joint) for f in FREQS}

    def tamper_check(f, w, n_cuts=20, seed=8):
        """Replace everything after a random cut with noise. Trades and equity up to the cut must not move."""
        gen = np.random.default_rng(seed)
        base = run_joint(joint, w, f)
        bad = []
        cuts = sorted(gen.choice(np.arange(len(idx) // 10, len(idx) - 2), n_cuts, replace=False))
        for k in cuts:
            j = joint.copy()
            noise = gen.lognormal(0.0, 0.3, size=(len(idx) - k - 1, j.shape[1]))
            j.iloc[k + 1:] = j.iloc[k + 1:].to_numpy() * noise
            alt = run_joint(j, w, f)
            same_eq = np.allclose(alt["equity"].iloc[: k + 1], base["equity"].iloc[: k + 1])
            same_flag = (alt["traded"].iloc[: k + 1].values == base["traded"].iloc[: k + 1].values).all()
            if not (same_eq and same_flag):
                bad.append(str(idx[k].date()))
        return {"cuts_tested": n_cuts, "failed_cuts": bad, "passed": not bad}

    tamper = {f: tamper_check(f, w_main) for f in FREQS}

    def reconcile_port(w, f, cost=COST_BPS):
        v = rb.simulate(opens, closes, w, f, cost, rf)["equity"]
        e = rb.event_loop(opens, closes, w, f, cost, rf)
        gap = float((v / e - 1).abs().max())
        return {"max_relative_gap": gap, "tolerance": 1e-9, "passed": gap < 1e-9}

    res["checks"] = {"lookahead_equity": lookahead, "lookahead_weights": lookahead_w, "future_data_tamper": tamper,
                     "reconcile": {f: reconcile_port(w_main, f) for f in FREQS},
                     "reconcile_control": {f: reconcile_port(w_ctrl, f) for f in FREQS},
                     "reconcile_spy_only": reconcile_port({STOCK: 1.0}, "none"),
                     "tie_to_repo_engine_max_gap": tie, "tie_passed": tie < 1e-9}

    # --- headline numbers ---------------------------------------------------------------------
    res["portfolios"] = {f: stats(research.window(runs[f], idx[0]), rf) for f in FREQS}
    res["portfolios_control"] = {f: stats(research.window(runs_ief[f], idx[0]), rf) for f in FREQS}
    res["solo"] = {s: stats(research.window(solo[s], idx[0]), rf) for s in solo}
    a, nr, spy = res["portfolios"]["annual"], res["portfolios"]["none"], res["solo"][STOCK]

    # --- independent replay -------------------------------------------------------------------
    o_d, c_d = {k: opens[k].tolist() for k in opens}, {k: closes[k].tolist() for k in closes}
    dates = [d.to_pydatetime() for d in idx]
    r0 = float(rf.iloc[0])
    replay = {"annual": independent_replay(o_d, c_d, dates, w_main, "annual", r0, COST_BPS),
              "none": independent_replay(o_d, c_d, dates, w_main, "none", r0, COST_BPS),
              "quarterly": independent_replay(o_d, c_d, dates, w_main, "quarterly", r0, COST_BPS),
              "monthly": independent_replay(o_d, c_d, dates, w_main, "monthly", r0, COST_BPS),
              "annual_control": independent_replay(o_d, c_d, dates, w_ctrl, "annual", r0, COST_BPS),
              "spy": independent_replay(o_d, c_d, dates, {STOCK: 1.0}, "none", r0, COST_BPS)}
    fr = lambda key, eq: float(abs(replay[key]["final_multiple"] / (eq.iloc[-1] / eq.iloc[0]) - 1))
    annual_full_year = lambda key, year: replay[key]["last_by_year"][year] / replay[key]["last_by_year"][year - 1] - 1
    spy_direct = float(c_d[STOCK][-1] / o_d[STOCK][1] * (1 - COST_BPS / 1e4))
    res["independent"] = {"annual": {k: replay["annual"][k] for k in ["final_multiple", "cagr", "max_drawdown", "rebalances"]},
                          "none": {k: replay["none"][k] for k in ["final_multiple", "cagr", "max_drawdown", "rebalances"]},
                          "gaps": {"annual": fr("annual", runs["annual"]["equity"]), "none": fr("none", runs["none"]["equity"]),
                                   "quarterly": fr("quarterly", runs["quarterly"]["equity"]), "monthly": fr("monthly", runs["monthly"]["equity"]),
                                   "annual_control": fr("annual_control", runs_ief["annual"]["equity"]), "spy": fr("spy", solo[STOCK]["equity"])},
                          "rebalance_count_matches": replay["annual"]["rebalances"] == a["rebalances"] and replay["monthly"]["rebalances"] == res["portfolios"]["monthly"]["rebalances"],
                          "max_dd_gap": abs(replay["annual"]["max_drawdown"] - a["max_drawdown"]),
                          "spy_direct_price_ratio": spy_direct, "spy_direct_gap": abs(spy_direct / spy["final_multiple"] - 1),
                          "annual_2022_replay": annual_full_year("annual", CRISIS_YEAR),
                          "annual_2022_engine": float((1 + runs["annual"]["ret"].loc[str(CRISIS_YEAR)]).prod() - 1)}

    # --- every rebalance -----------------------------------------------------------------------
    lg = rb.ledger(opens, closes, w_main, s_all)
    lg.to_csv(OUT / "rebalances.csv", index=False)
    lg_ok = lg.iloc[:-1] if len(lg) and lg["next_end"].iloc[-1] == idx[-1] else lg
    res["rebalance_ledger"] = {"n": int(len(lg)), "n_complete_years": int(len(lg_ok)), "n_sold_stocks": int((lg.action == "sold_stocks").sum()),
                               "n_bought_stocks": int((lg.action == "bought_stocks").sum()),
                               "n_helped": int((lg.effect > 0).sum()), "n_hurt": int((lg.effect < 0).sum()),
                               "share_helped": float((lg.effect > 0).mean()),
                               "best_effect": float(lg.effect.max()), "best_date": str(lg.loc[lg.effect.idxmax(), "date"].date()),
                               "worst_effect": float(lg.effect.min()), "worst_date": str(lg.loc[lg.effect.idxmin(), "date"].date()),
                               "median_effect": float(lg.effect.median()), "mean_turnover": float(lg.turnover.mean()),
                               "max_stock_weight_before": float(lg.stock_weight_before.max()), "max_stock_weight_date": str(lg.loc[lg.stock_weight_before.idxmax(), "date"].date()),
                               "min_stock_weight_before": float(lg.stock_weight_before.min()), "min_stock_weight_date": str(lg.loc[lg.stock_weight_before.idxmin(), "date"].date()),
                               "rows": [{"date": str(r.date.date()), "before": r.stock_weight_before, "effect": r.effect, "action": r.action, "turnover": r.turnover} for r in lg.itertuples()]}

    # added after seeing monthly fall deeper than no rebalancing: what did the monthly trades do during that fall?
    lg_m = rb.ledger(opens, closes, w_main, runs["monthly"])
    m_st = res["portfolios"]["monthly"]
    in_fall = lg_m[(lg_m.date > m_st["max_dd_peak"]) & (lg_m.date <= m_st["max_dd_trough"])]
    res["monthly_fall"] = {"peak": m_st["max_dd_peak"], "trough": m_st["max_dd_trough"], "n_rebalances": int(len(in_fall)),
                           "n_bought_stocks": int((in_fall.action == "bought_stocks").sum()),
                           "share_bought_stocks": float((in_fall.action == "bought_stocks").mean()),
                           "note": "explanation added after seeing the result, not a pre-set test"}
    res["rebalance_ledger"]["n_within_1pct"] = int((lg.effect.abs() < 0.01).sum())
    res["rebalance_ledger"]["second_best_effect"] = float(lg.effect.nlargest(2).iloc[-1])

    # --- drift: what never rebalancing does to the weights --------------------------------------
    d_weight = runs["none"][f"w_{STOCK}"]
    res["drift"] = {"start_stock_weight": float(d_weight.iloc[1]), "end_stock_weight": float(d_weight.iloc[-1]),
                    "max_stock_weight": float(d_weight.max()), "max_stock_weight_date": str(d_weight.idxmax().date()),
                    "weight_at_2007_peak": float(d_weight.loc["2007-10-09"]), "weight_at_2021_end": float(d_weight.loc["2021-12-31"]),
                    "mean_stock_weight": float(d_weight.iloc[1:].mean()), "mean_stock_weight_annual": float(s_all[f"w_{STOCK}"].iloc[1:].mean()),
                    "share_of_days_over_70": float((d_weight.iloc[1:] > 0.70).mean()),
                    "share_of_days_over_80": float((d_weight.iloc[1:] > 0.80).mean()),
                    "first_day_over_70": str(d_weight[d_weight > 0.70].index[0].date()),
                    "first_day_over_80": str(d_weight[d_weight > 0.80].index[0].date()) if (d_weight > 0.80).any() else None}

    # --- annual table -----------------------------------------------------------------------------
    yrly = lambda bt: (1 + bt["ret"]).groupby(bt.index.year).prod() - 1
    yr = pd.DataFrame({"spy": yrly(solo[STOCK]), "tlt": yrly(solo[BOND]), "ief": yrly(solo[CONTROL]), "annual": yrly(runs["annual"]),
                       "none": yrly(runs["none"]), "quarterly": yrly(runs["quarterly"]), "monthly": yrly(runs["monthly"]),
                       "annual_ief": yrly(runs_ief["annual"])})
    yr["excess_vs_spy"] = yr.annual - yr.spy
    yr["excess_vs_none"] = yr.annual - yr.none
    corr_by_year = pd.Series({y: float(g.spy.corr(g.tlt)) for y, g in pd.DataFrame({"spy": closes[STOCK].pct_change(), "tlt": closes[BOND].pct_change()}).groupby(lambda d: d.year)})
    yr["spy_tlt_corr"] = corr_by_year
    full_years = yr.loc[(yr.index > idx[0].year) & (yr.index < idx[-1].year)]
    yr.to_csv(OUT / "annual.csv")
    down = full_years[(full_years.spy < 0) & (full_years.tlt < 0)]
    spy_down = full_years[full_years.spy < 0]
    res["annual"] = {"full_years": int(len(full_years)), "first_full": int(full_years.index[0]), "last_full": int(full_years.index[-1]),
                     "partial_years": [int(idx[0].year), int(idx[-1].year)],
                     "years_annual_beat_spy": int((full_years.excess_vs_spy > 0).sum()),
                     "years_annual_beat_none": int((full_years.excess_vs_none > 0).sum()),
                     "n_spy_down": int(len(spy_down)), "spy_down_years": [int(y) for y in spy_down.index],
                     "annual_beat_spy_in_spy_down": int((spy_down.excess_vs_spy > 0).sum()),
                     "n_both_down": int(len(down)), "both_down_years": [int(y) for y in down.index],
                     "both_down_rows": {int(y): down.loc[y].to_dict() for y in down.index},
                     "crisis": {k: float(yr.loc[CRISIS_YEAR, k]) for k in ["spy", "tlt", "ief", "annual", "none", "quarterly", "monthly", "annual_ief"]},
                     "crisis_rank_annual_worst": int((full_years.annual < yr.loc[CRISIS_YEAR, "annual"]).sum()) + 1,
                     "best_year_annual": int(full_years.annual.idxmax()), "worst_year_annual": int(full_years.annual.idxmin()),
                     "worst_year_annual_value": float(full_years.annual.min()), "worst_year_spy": int(full_years.spy.idxmin()),
                     "worst_year_spy_value": float(full_years.spy.min()),
                     "y2008": {k: float(yr.loc[2008, k]) for k in ["spy", "tlt", "ief", "annual", "none", "annual_ief"]}}

    # --- drawdown episodes of SPY, and of the portfolio itself --------------------------------------
    spy_w = research.window(solo[STOCK], idx[0])
    a_w, n_w = research.window(runs["annual"], idx[0]), research.window(runs["none"], idx[0])
    m_w = research.window(runs["monthly"], idx[0])
    i_w = research.window(runs_ief["annual"], idx[0])

    def segment(eq, p, t):
        s = eq.loc[p:t]
        return float(s.iloc[-1] / s.iloc[0] - 1)

    eps = []
    for e in episodes(spy_w["equity"], MIN_DD):
        p, t = e["peak"], e["trough"]
        row = {"peak": str(p.date()), "trough": str(t.date()), "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
               "spy": e["depth"], "annual": segment(a_w["equity"], p, t), "none": segment(n_w["equity"], p, t), "monthly": segment(m_w["equity"], p, t),
               "annual_ief": segment(i_w["equity"], p, t), "tlt": segment(solo[BOND]["equity"], p, t),
               "drift_stock_weight_at_peak": float(d_weight.loc[p]), "annual_stock_weight_at_peak": float(s_all[f"w_{STOCK}"].loc[p])}
        row["annual_saved_vs_none"] = row["annual"] - row["none"]
        row["annual_saved_vs_spy"] = row["annual"] - row["spy"]
        eps.append(row)
    res["episodes"] = eps
    pd.DataFrame(eps).to_csv(OUT / "episodes.csv", index=False)
    own = {}
    for name, frame in {"annual": a_w, "none": n_w, "spy": spy_w, "annual_ief": i_w}.items():
        own[name] = [{"peak": str(e["peak"].date()), "trough": str(e["trough"].date()), "recovered": str(e["recovered"].date()) if e["recovered"] is not None else None,
                      "depth": e["depth"]} for e in episodes(frame["equity"], -0.10)]
    res["own_drawdowns"] = own

    # the 2022 fall, on each portfolio's own peak and trough around that year
    def crisis_dd(eq):
        dd = eq / eq.cummax() - 1
        win = dd.loc["2021-06-01":"2024-06-30"]
        trough = win.idxmin()
        peak = eq.loc[:trough].idxmax()
        rec = eq.loc[trough:]
        rec = rec[rec >= eq.loc[peak]]
        return {"peak": str(peak.date()), "trough": str(trough.date()), "depth": float(dd.loc[trough]),
                "recovered": str(rec.index[0].date()) if len(rec) else None,
                "still_under_at_end": bool(eq.iloc[-1] < eq.loc[peak]), "days_to_recover": int((rec.index[0] - peak).days) if len(rec) else None}

    res["crisis_dd"] = {"annual": crisis_dd(a_w["equity"]), "none": crisis_dd(n_w["equity"]), "spy": crisis_dd(spy_w["equity"]),
                        "annual_ief": crisis_dd(i_w["equity"]), "tlt": crisis_dd(research.window(solo[BOND], idx[0])["equity"]),
                        "ief": crisis_dd(research.window(solo[CONTROL], idx[0])["equity"])}

    # --- stock/bond correlation ------------------------------------------------------------------
    r_s, r_b, r_c = closes[STOCK].pct_change().dropna(), closes[BOND].pct_change().dropna(), closes[CONTROL].pct_change().dropna()
    res["correlation"] = {"full": float(r_s.corr(r_b)), "before_2022": float(r_s.loc[:"2021-12-31"].corr(r_b.loc[:"2021-12-31"])),
                          "from_2022": float(r_s.loc["2022-01-01":].corr(r_b.loc["2022-01-01":])),
                          "year_2022": float(r_s.loc["2022"].corr(r_b.loc["2022"])),
                          "control_full": float(r_s.corr(r_c)), "control_before_2022": float(r_s.loc[:"2021-12-31"].corr(r_c.loc[:"2021-12-31"])),
                          "control_from_2022": float(r_s.loc["2022-01-01":].corr(r_c.loc["2022-01-01":]))}
    rolling = r_s.rolling(252).corr(r_b)

    # --- health 1: costs --------------------------------------------------------------------------
    rows = []
    for b in COST_GRID:
        row = {"cost_bps": b}
        for f in FREQS:
            row[f"{f}_cagr"] = metrics.cagr(research.window(sim(w_main, f, cost=b), idx[0])["equity"])
        row["spy_cagr"] = metrics.cagr(research.window(sim({STOCK: 1.0}, "none", cost=b), idx[0])["equity"])
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "cost_sweep.csv", index=False)
    res["cost_sweep"] = sweep.to_dict("records")
    res["breakeven"] = {"annual_vs_none_bps": research.breakeven_bps(sweep, "annual_cagr", "none_cagr"),
                        "annual_vs_spy_bps": research.breakeven_bps(sweep, "annual_cagr", "spy_cagr"),
                        "monthly_vs_annual_bps": research.breakeven_bps(sweep, "monthly_cagr", "annual_cagr")}
    cagr0 = sweep.set_index("cost_bps").loc[0, "annual_cagr"]
    res["cost_drag"] = {"annual_vs_zero_cagr": cagr0 - a["cagr"], "total_cost_multiplier": a["total_cost_multiplier"],
                        "monthly_vs_zero_cagr": sweep.set_index("cost_bps").loc[0, "monthly_cagr"] - res["portfolios"]["monthly"]["cagr"]}

    # --- health 2: parameter grid ------------------------------------------------------------------
    def grid_for(bond):
        rows = []
        for sw in STOCK_WEIGHTS:
            for f in FREQS:
                bt = research.window(sim({STOCK: sw, bond: 1 - sw}, f), idx[0])
                st = stats(bt, rf)
                rows.append({"bond": bond, "stock_weight": sw, "freq": f, "sharpe": st["sharpe"], "cagr": st["cagr"], "max_drawdown": st["max_drawdown"], "ann_vol": st["ann_vol"], "trades": st["trades"]})
        return pd.DataFrame(rows)

    g, g_ctrl = grid_for(BOND), grid_for(CONTROL)
    pd.concat([g, g_ctrl]).to_csv(OUT / "param_grid.csv", index=False)
    pivot_dd = g[g.stock_weight == 0.6].set_index("freq")
    res["param_grid"] = {"n": int(len(g)), "n_control": int(len(g_ctrl)), "rows": g.to_dict("records"), "rows_control": g_ctrl.to_dict("records"),
                         "rank_60_annual_by_sharpe": int(g.sort_values("sharpe", ascending=False).reset_index(drop=True).query("stock_weight == 0.6 and freq == 'annual'").index[0]) + 1,
                         "best_by_sharpe": g.sort_values("sharpe", ascending=False).iloc[0].to_dict(),
                         "worst_by_sharpe": g.sort_values("sharpe").iloc[0].to_dict(),
                         "n_beat_spy_cagr": int((g.cagr > spy["cagr"]).sum()), "n_beat_spy_sharpe": int((g.sharpe > spy["sharpe"]).sum()),
                         "n_shallower_than_spy": int((g.max_drawdown > spy["max_drawdown"]).sum()),
                         "by_freq_at_60": {f: {"cagr": float(pivot_dd.loc[f, "cagr"]), "sharpe": float(pivot_dd.loc[f, "sharpe"]), "max_drawdown": float(pivot_dd.loc[f, "max_drawdown"])} for f in FREQS}}
    by_w = {}
    for sw in STOCK_WEIGHTS:
        p = g[g.stock_weight == sw].set_index("freq")
        by_w[str(int(sw * 100))] = {"none_cagr": float(p.loc["none", "cagr"]), "annual_cagr": float(p.loc["annual", "cagr"]),
                                    "none_max_drawdown": float(p.loc["none", "max_drawdown"]), "annual_max_drawdown": float(p.loc["annual", "max_drawdown"]),
                                    "annual_sharpe": float(p.loc["annual", "sharpe"]), "none_sharpe": float(p.loc["none", "sharpe"])}
    res["param_grid"]["by_weight"] = by_w

    # which month the yearly rebalance happens in
    months = []
    for m in range(1, 13):
        st = stats(research.window(sim(w_main, "annual", month=m), idx[0]), rf)
        months.append({"month": m, "cagr": st["cagr"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"], "final_multiple": st["final_multiple"]})
    mdf = pd.DataFrame(months)
    mdf.to_csv(OUT / "month_sweep.csv", index=False)
    res["month_sweep"] = {"rows": months, "cagr_min": float(mdf.cagr.min()), "cagr_max": float(mdf.cagr.max()),
                          "cagr_range": float(mdf.cagr.max() - mdf.cagr.min()),
                          "dd_min": float(mdf.max_drawdown.min()), "dd_max": float(mdf.max_drawdown.max()),
                          "dd_range": float(mdf.max_drawdown.max() - mdf.max_drawdown.min()),
                          "best_month": int(mdf.loc[mdf.sharpe.idxmax(), "month"]), "worst_month": int(mdf.loc[mdf.sharpe.idxmin(), "month"]),
                          "n_shallower_than_none": int((mdf.max_drawdown > nr["max_drawdown"]).sum())}

    # --- health 3: before and after 2010 -----------------------------------------------------------
    before, after = (idx[0], "2009-12-31"), (SPLIT, None)
    is_rows = []
    for sw in STOCK_WEIGHTS:
        for f in FREQS:
            bt = research.window(sim({STOCK: sw, BOND: 1 - sw}, f), *before)
            is_rows.append({"stock_weight": sw, "freq": f, "sharpe": metrics.sharpe(bt["ret"], rf.loc[bt.index])})
    is_df = pd.DataFrame(is_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    best_is = is_df.iloc[0]
    best_w = {STOCK: float(best_is.stock_weight), BOND: 1 - float(best_is.stock_weight)}
    split = {"in_sample": f"{idx[0].date()} to 2009-12-31", "out_of_sample": f"{SPLIT} to {idx[-1].date()}",
             "best_in_sample": {"stock_weight": float(best_is.stock_weight), "freq": best_is.freq},
             "rank_60_annual_in_sample": int(is_df.query("stock_weight == 0.6 and freq == 'annual'").index[0]) + 1}
    for name, frame in {"annual": s_all, "none": runs["none"], "monthly": runs["monthly"], "quarterly": runs["quarterly"], "spy": solo[STOCK],
                        "best_in_sample_run": sim(best_w, best_is.freq), "annual_control": runs_ief["annual"], "none_control": runs_ief["none"]}.items():
        split[name] = {"in_sample": stats(research.window(frame, *before), rf), "out_of_sample": stats(research.window(frame, *after), rf, whole=False)}
    res["split"] = split

    # --- save --------------------------------------------------------------------------------------
    curves = pd.DataFrame({"spy": spy_w["equity"], "tlt": research.window(solo[BOND], idx[0])["equity"], "ief": research.window(solo[CONTROL], idx[0])["equity"],
                           "annual": a_w["equity"], "none": n_w["equity"], "quarterly": research.window(runs["quarterly"], idx[0])["equity"], "monthly": m_w["equity"],
                           "annual_ief": i_w["equity"], "none_ief": research.window(runs_ief["none"], idx[0])["equity"]})
    for c in ["spy", "annual", "none", "monthly"]:
        curves[f"{c}_dd"] = metrics.drawdown(curves[c])
    curves["stock_weight_annual"] = s_all[f"w_{STOCK}"]
    curves["stock_weight_none"] = d_weight
    curves["spy_tlt_corr_252d"] = rolling
    curves.to_csv(OUT / "curves.csv")
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, color in {"spy": "grey", "none": "tab:blue", "annual": "black"}.items():
        ax[0].plot(curves[name], color=color, label={"spy": "SPY buy & hold", "none": "60/40 no rebalancing", "annual": "60/40 yearly rebalance"}[name])
    ax[0].set_yscale("log"); ax[0].legend()
    ax[1].plot(curves["stock_weight_none"], color="tab:blue", label="stock weight, no rebalancing")
    ax[1].plot(curves["stock_weight_annual"], color="black", label="stock weight, yearly rebalance"); ax[1].legend()
    fig.text(0.01, 0.01, "SPY, TLT daily, historical backtest, not a forecast. ETFs are data samples, not recommendations.", fontsize=7)
    fig.savefig(OUT / "equity.png", dpi=150); plt.close(fig)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: res[k] for k in ["window", "checks", "portfolios", "portfolios_control", "solo", "independent", "drift", "breakeven", "cost_drag", "month_sweep", "correlation", "crisis_dd", "episodes"]}, indent=0, default=str)[:12000])
    print(json.dumps({k: v for k, v in res["rebalance_ledger"].items() if k != "rows"}, indent=0)); print(json.dumps(res["annual"], indent=0)); print(json.dumps({k: v for k, v in res["param_grid"].items() if k not in ("rows", "rows_control")}, indent=0, default=str)); print(json.dumps({k: v for k, v in res["split"].items()}, indent=0, default=str)[:6000])


if __name__ == "__main__":
    main()
