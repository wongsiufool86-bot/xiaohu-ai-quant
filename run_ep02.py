"""Episode 2: lump sum vs DCA on SPY, every monthly start date. Writes results/ep02/results.json.

    DATA_DIR=/path/to/csv_folder python run_ep02.py   # data format: see backtest/data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import data, metrics
from strategies import dca

OUT = Path(__file__).parent / "results" / "ep02"
SYMBOL, COST_BPS, N_DEFAULT, HORIZON_YEARS = "SPY", 5.0, 12, 10
NS = [3, 6, 12, 24, 36]


def end_of_month_n(px, ms, k, n):
    """Last trading day of the n-th month counted from month k (k is month 1)."""
    if k + n < len(ms):
        return px.index[px.index < ms[k + n]][-1]
    return None


def end_after_years(px, start, years):
    target = start + pd.DateOffset(years=years)
    if target > px.index[-1]:
        return None
    return px.index[px.index <= target][-1]


def compare(px, rf, n, horizon, cost):
    ms = dca.month_starts(px.index)
    rows = []
    for k, s in enumerate(ms[:-1]):
        e = end_of_month_n(px, ms, k, n) if horizon == "n_months" else end_after_years(px, s, HORIZON_YEARS)
        if e is None or e < s:
            continue
        try:
            d = dca.dca_final_formula(px, rf, s, e, n, cost)
        except ValueError:
            continue
        ls = float(px["close"].loc[e] / px["open"].loc[s] * (1 - cost / 1e4))
        rows.append({"start": s, "end": e, "lump": ls, "dca": d, "ratio": ls / d})
    return pd.DataFrame(rows)


def summarize(df):
    if df.empty:
        return {}
    worst = df.loc[df.ratio.idxmin()]
    best = df.loc[df.ratio.idxmax()]
    return {
        "n_starts": int(len(df)),
        "first_start": str(df.start.iloc[0].date()), "last_start": str(df.start.iloc[-1].date()),
        "lump_win_share": float((df.ratio > 1).mean()),
        "median_ratio": float(df.ratio.median()), "mean_ratio": float(df.ratio.mean()),
        "worst_ratio": float(worst.ratio), "worst_start": str(worst.start.date()),
        "best_ratio": float(best.ratio), "best_start": str(best.start.date()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = data.load_prices(SYMBOL)
    rf = data.daily_rate(data.load_rate("%3MTCM"), px.index)
    res: dict = {"symbol": SYMBOL, "rules": {"n_default": N_DEFAULT, "cost_bps_per_buy": COST_BPS,
                 "fill": "open of first trading day of each month", "cash": "3-month Treasury",
                 "horizon_years": HORIZON_YEARS}}
    ms = dca.month_starts(px.index)

    # reconcile: day-by-day path vs closed form, on a sample of starts
    gaps = []
    for k in range(0, len(ms) - 40, 17):
        s = ms[k]
        e = end_of_month_n(px, ms, k, N_DEFAULT)
        path = dca.dca_path(px, rf, s, e, N_DEFAULT, COST_BPS)
        gaps.append(abs(path.iloc[-1] / dca.dca_final_formula(px, rf, s, e, N_DEFAULT, COST_BPS) - 1))
    res["checks"] = {"reconcile": {"samples": len(gaps), "max_relative_gap": float(max(gaps)),
                                   "passed": bool(max(gaps) < 1e-6)}}

    # main comparison: N=12, at end of DCA period and at 10 years
    main12 = compare(px, rf, N_DEFAULT, "n_months", COST_BPS)
    main10y = compare(px, rf, N_DEFAULT, "years", COST_BPS)
    main12.to_csv(OUT / "rolling_n12_at12m.csv", index=False)
    main10y.to_csv(OUT / "rolling_n12_at10y.csv", index=False)
    res["n12_at_end_of_dca"] = summarize(main12)
    res["n12_at_10y"] = summarize(main10y)

    # pain during the 12 investment months: max drawdown of each account (from its own peak)
    dd = []
    for k in range(0, len(ms) - N_DEFAULT - 1):
        s = ms[k]
        e = end_of_month_n(px, ms, k, N_DEFAULT)
        ls = dca.lump_sum_path(px, s, e, COST_BPS)
        dc = dca.dca_path(px, rf, s, e, N_DEFAULT, COST_BPS)
        dd.append({"start": s, "lump_mdd": metrics.max_drawdown(ls), "dca_mdd": metrics.max_drawdown(dc),
                   "lump_min_vs_start": float(ls.min() - 1), "dca_min_vs_start": float(dc.min() - 1)})
    dd = pd.DataFrame(dd)
    dd.to_csv(OUT / "pain_first12m.csv", index=False)
    res["pain_first_12m"] = {
        "median_lump_mdd": float(dd.lump_mdd.median()), "median_dca_mdd": float(dd.dca_mdd.median()),
        "worst_lump_mdd": float(dd.lump_mdd.min()), "worst_dca_mdd": float(dd.dca_mdd.min()),
        "worst_lump_start": str(dd.loc[dd.lump_mdd.idxmin(), "start"].date()),
        "share_lump_below_start_at_some_point_by_10pct": float((dd.lump_min_vs_start < -0.10).mean()),
        "share_dca_below_start_at_some_point_by_10pct": float((dd.dca_min_vs_start < -0.10).mean()),
    }

    # health check 1: costs (both pay the same rate on every purchase)
    res["cost_sweep_n12_at12m"] = []
    for b in [0, 5, 20, 50]:
        res["cost_sweep_n12_at12m"].append({"cost_bps": b, **{k: v for k, v in summarize(compare(px, rf, N_DEFAULT, "n_months", b)).items()
                                                               if k in ("lump_win_share", "median_ratio")}})

    # health check 2: DCA length and horizon
    res["by_n"] = {}
    for n in NS:
        res["by_n"][str(n)] = {"at_end_of_dca": summarize(compare(px, rf, n, "n_months", COST_BPS)),
                               "at_10y": summarize(compare(px, rf, n, "years", COST_BPS))}

    # health check 3: starts before 2010 vs from 2010
    for name, df in {"at_end_of_dca": main12, "at_10y": main10y}.items():
        res.setdefault("split", {})[name] = {
            "before_2010": summarize(df[df.start < "2010-01-01"]),
            "from_2010": summarize(df[df.start >= "2010-01-01"]),
        }

    # why: how often did a month beat cash?
    me = px["close"].groupby([px.index.year, px.index.month]).last()
    mret = me.pct_change().dropna()
    rf_m = rf.groupby([rf.index.year, rf.index.month]).apply(lambda x: (1 + x).prod() - 1).reindex(mret.index)
    res["monthly"] = {"n_months": int(len(mret)), "share_up": float((mret > 0).mean()),
                      "share_beat_cash": float((mret > rf_m).mean()),
                      "avg_month_spy": float(mret.mean()), "avg_month_cash": float(rf_m.mean())}

    # when did DCA win? count by start year
    lose = main12[main12.ratio < 1]
    by_year = lose.groupby(lose.start.dt.year).size().sort_values(ascending=False)
    res["dca_win_years"] = {"top": {int(k): int(v) for k, v in by_year.head(5).items()},
                            "share_in_top5_years": float(by_year.head(5).sum() / len(lose))}

    # sensitivity: idle cash earns nothing
    zero_rf = pd.Series(0.0, index=px.index)
    res["idle_cash_zero_rate"] = {k: v for k, v in summarize(compare(px, zero_rf, N_DEFAULT, "n_months", COST_BPS)).items()
                                  if k in ("lump_win_share", "median_ratio")}

    # the worst start: how long until each account is back to its starting money?
    ws = pd.Timestamp(res["n12_at_end_of_dca"]["worst_start"])
    ls_path = dca.lump_sum_path(px, ws, px.index[-1], COST_BPS)
    dc_path = dca.dca_path(px, rf, ws, px.index[-1], N_DEFAULT, COST_BPS)
    def back_to_start(path):
        """First day after the account's lowest point on which it is back to its starting money."""
        low = path.idxmin()
        after = path.loc[low:]
        rec = after[after >= 1.0]
        return str(rec.index[0].date()) if len(rec) else None
    pd.DataFrame({"lump_sum": ls_path, "dca": dc_path}).to_csv(OUT / "worst_start_paths.csv")
    res["worst_start_recovery"] = {"start": str(ws.date()),
                                   "lump_back_to_start": back_to_start(ls_path),
                                   "dca_back_to_start": back_to_start(dc_path),
                                   "lump_min": float(ls_path.min()), "dca_min": float(dc_path.min())}

    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(main12.start, (main12.ratio - 1) * 100, color="black")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Lump sum vs 12-month DCA, at end of DCA period (% better for lump sum)")
    fig.text(0.01, 0.01, "SPY daily, historical backtest, not a forecast. Teaching example, not investment advice.", fontsize=7)
    fig.savefig(OUT / "rolling_ratio.png", dpi=150); plt.close(fig)
    print(json.dumps({k: res[k] for k in ["checks", "n12_at_end_of_dca", "n12_at_10y", "pain_first_12m"]}, indent=2))


if __name__ == "__main__":
    main()
