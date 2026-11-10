"""Hand-checkable tests for interstitial x01. Synthetic data and synthetic model paths only:
no market data, and the model is never run here (torch is not even imported)."""
import sys

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from strategies import kronos_signal as ks
import run_x01 as rx


def bars(closes, opens=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1000.0}, index=idx)


def walk(seed, n=300):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    return bars(c, opens=c * (1 + rng.normal(0, 0.004, n)))


def fake_frame(px, p_up, start=5):
    """A daily frame as daily_frame would build it, with P(up) supplied by the test."""
    idx = px.index
    asof, target = idx[start:-1], idx[start + 1:]
    c = px["close"]
    df = pd.DataFrame({"target": target, "close_t": c.loc[asof].to_numpy(),
                       "actual_close": c.loc[target].to_numpy()}, index=asof)
    df["actual_ret"] = df["actual_close"] / df["close_t"] - 1
    df["p_up"] = np.asarray(p_up, float)[: len(df)]
    df["pred_ret"] = df["p_up"] - 0.5
    return df


# ---------------------------------------------------------------- no model, no future
def test_importing_run_x01_does_not_load_torch():
    assert "torch" not in sys.modules


def test_window_is_last_n_bars_ending_at_asof():
    px = walk(1, 50)
    w = ks.asof_window(px, px.index[29], n=10)
    assert len(w) == 10 and w.index[-1] == px.index[29] and w.index[0] == px.index[20]
    assert list(w.columns) == ks.COLS


def test_window_ignores_everything_after_asof():
    px = walk(2, 60)
    bad = px.copy()
    bad.iloc[31:] = bad.iloc[31:] * 7.0
    a, b = ks.asof_window(px, px.index[30], n=20), ks.asof_window(bad, px.index[30], n=20)
    pd.testing.assert_frame_equal(a, b)


def test_window_needs_enough_history():
    px = walk(3, 30)
    with pytest.raises(ValueError):
        ks.asof_window(px, px.index[8], n=10)


def test_segments_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=40)
    seg = ks.segment_days(idx, str(idx[30].date()), str(idx[34].date()), str(idx[25].date()), context=5)
    assert [t for _, t in seg["test"]] == list(idx[30:35])
    assert [a for a, _ in seg["test"]] == list(idx[29:34])
    assert [t for _, t in seg["control"]] == list(idx[21:26])     # same count, ends at control_end
    assert seg["control"][-1][1] <= seg["test"][0][0]


def test_day_seed_fixed_per_day():
    d = pd.Timestamp("2025-01-02")
    assert ks.day_seed(0, "SPY", d) == ks.day_seed(0, "SPY", "2025-01-02")
    assert len({ks.day_seed(0, "SPY", d), ks.day_seed(1, "SPY", d), ks.day_seed(0, "QQQ", d),
                ks.day_seed(0, "SPY", "2025-01-03")}) == 4


# ---------------------------------------------------------------- per-day numbers
def test_summarize_by_hand():
    s = ks.summarize([98, 99, 101, 102], 100.0)
    assert s["pred_mean"] == 100.0 and s["pred_ret"] == 0.0 and s["p_up"] == 0.5
    assert s["q50"] == 100.0
    # a path exactly at close_t is not "up"
    assert ks.summarize([100, 100, 101], 100.0)["p_up"] == pytest.approx(1 / 3)


def test_daily_frame_by_hand():
    px = bars([10, 11, 12, 13, 12, 12.6])
    pred = pd.DataFrame({"asof": [px.index[4]], "target": [px.index[5]], "close_t": [12.0],
                         "paths": [np.array([12.6, 11.4, 13.2, 99.0])]})
    f = ks.daily_frame(pred, px, n_paths=3, context=3)
    r = f.iloc[0]
    assert r["pred_mean"] == pytest.approx(12.4) and r["p_up"] == pytest.approx(2 / 3)   # 4th path ignored
    assert r["actual_ret"] == pytest.approx(0.05) and r["last_ret"] == pytest.approx(12 / 13 - 1)
    w = np.array([12.0, 13.0, 12.0])
    assert r["z"] == pytest.approx((12 - w.mean()) / w.std())


def test_window_z_uses_population_std_of_the_window():
    px = bars([1, 2, 3, 4, 5])
    assert ks.window_z(px, px.index[4], n=3) == pytest.approx((5 - 4) / np.std([3, 4, 5]))


# ---------------------------------------------------------------- statistics
def test_binomial_and_wilson_by_hand():
    assert ks.binom_two_sided(5, 10) == pytest.approx(1.0)
    assert ks.binom_two_sided(0, 5) == pytest.approx(2 / 32)
    assert ks.binom_two_sided(9, 10) == pytest.approx(22 / 1024)
    lo, hi = ks.wilson(50, 100)
    assert lo == pytest.approx(1 - hi) and 0.40 < lo < 0.41


def test_direction_table_by_hand():
    df = pd.DataFrame({"pred_ret": [0.1, 0.1, -0.1, -0.1, 0.1], "p_up": [0.6, 0.6, 0.4, 0.4, 0.6],
                       "actual_ret": [0.01, -0.01, -0.01, 0.01, 0.0],
                       "last_ret": [0.02, 0.02, -0.02, 0.02, 0.0]})
    d = ks.direction_table(df)
    assert d["days"] == 4 and d["model_hits"] == 2                  # flat day dropped
    assert d["always_up_hit_rate"] == 0.5
    assert d["momentum_hit_rate"] == 0.75                           # +,+ / +,- / -,- / +,+
    assert d["vs_momentum"]["only_model_right"] == 0 and d["vs_momentum"]["only_baseline_right"] == 1


def test_rank_ic_and_cross_section_by_hand():
    assert ks.rank_corr([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert ks.rank_corr([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    fr = {"A": pd.DataFrame({"pred_ret": [1, 1], "actual_ret": [1, 3]}, index=idx),
          "B": pd.DataFrame({"pred_ret": [2, 2], "actual_ret": [2, 2]}, index=idx),
          "C": pd.DataFrame({"pred_ret": [3, 3], "actual_ret": [3, 1]}, index=idx)}
    x = ks.cross_sectional_rank_ic(fr)
    assert x["days"] == 2 and x["mean"] == pytest.approx(0.0) and x["share_positive"] == 0.5


def test_calibration_and_coverage_by_hand():
    df = pd.DataFrame({"p_up": [0.65, 0.65, 0.25, 0.55], "actual_ret": [0.01, -0.01, -0.02, 0.0],
                       "actual_close": [101, 99, 98, 100], "q10": [99.5, 99.5, 99, 99],
                       "q90": [100.5, 100.5, 101, 101], "close_t": [100, 100, 100, 100]})
    cal = {c["bucket"]: c for c in ks.calibration(df)}
    assert cal["0.6-0.7"]["days"] == 2 and cal["0.6-0.7"]["actual_up"] == 0.5
    assert cal["0.0-0.3"]["actual_up"] == 0.0 and cal["0.5-0.6"]["actual_up"] == 0.0
    cov = ks.coverage(df)
    assert cov["inside_10_90"] == 0.25 and cov["above_q90"] == 0.25 and cov["below_q10"] == 0.5


def test_mean_pull_detects_a_forecast_that_leans_on_z():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 200)
    df = pd.DataFrame({"z": z, "pred_ret": -0.001 * z, "p_up": 0.5 - 0.1 * z, "actual_ret": rng.normal(0, 0.01, 200)})
    m = ks.mean_pull(df)
    assert m["corr_pred_z"] == pytest.approx(-1.0) and abs(m["corr_actual_z"]) < 0.2
    assert m["pred_ret_z_above_1"] < 0 < m["pred_ret_z_below_m1"]


# ---------------------------------------------------------------- trading
def test_threshold_is_strict():
    df = pd.DataFrame({"p_up": [0.5, 0.51, 0.6]})
    assert ks.target_weights(df).tolist() == [0.0, 1.0, 1.0]
    assert ks.target_weights(df, 0.55).tolist() == [0.0, 0.0, 1.0]


def test_signal_fills_at_next_open_by_hand():
    # decide on day 0 (P(up)=0.9), hold from day 1 open 100 -> close 110; decide out on day 1.
    px = bars([100, 110, 121, 121], opens=[100, 100, 110, 130])
    df = fake_frame(px, [0.9, 0.1, 0.1], start=0)
    rf = pd.Series(0.0, index=px.index)
    bt, tgt, _ = rx.rule_bt(px, df, rf, cost=0)
    assert bt["weight"].tolist() == [0, 1, 0, 0]
    # day1 open->close +10%; day2 overnight 110->110 belongs to the old position, then sold at the open
    assert bt["equity"].iloc[-1] / bt["equity"].iloc[0] == pytest.approx(1.10)


def test_close_to_close_upper_bound_by_hand():
    px = bars([100, 110, 99])
    df = fake_frame(px, [0.9, 0.1], start=0)
    rf = pd.Series(0.0, index=px.index)
    out = ks.close_to_close(df, ks.target_weights(df), 10, rf)
    # day 1: buy at close 100 (cost 0.1%), earn +10%; day 2: sell at close 110 (cost 0.1%), earn cash 0
    assert out["ret"].tolist() == pytest.approx([0.999 * 1.10 - 1, 0.999 - 1])


def test_engines_and_independent_ledger_agree_on_a_random_path():
    px = walk(7, 300)
    rng = np.random.default_rng(7)
    df = fake_frame(px, rng.uniform(0, 1, 300), start=10)
    rf = pd.Series(0.0001, index=px.index)
    bt, tgt, bars_ = rx.rule_bt(px, df, rf, cost=5)
    assert engine.event_loop(bars_, tgt, 5, rf).iloc[-1] == pytest.approx(bt["equity"].iloc[-1], rel=1e-10)
    ind = rx.independent_ledger(px, df, rf, cost_bps=5)
    perf = rx.perf(bt, rf)
    assert ind["final_multiple"] == pytest.approx(perf["final_multiple"], rel=1e-10)
    assert ind["orders"] == perf["orders"]


def test_ideal_and_next_open_cover_the_same_span():
    px = walk(8, 60)
    df = fake_frame(px, np.full(60, 0.9), start=5)
    rf = pd.Series(0.0, index=px.index)
    t = rx.trading(px, df, rf, cost=0)
    assert t["ideal_close"]["start"] == t["strategy"]["start"] and t["ideal_close"]["end"] == t["strategy"]["end"]
    # always in: the close-to-close version is exactly buy and hold from close_0
    assert t["ideal_close"]["final_multiple"] == pytest.approx(px["close"].iloc[-1] / px["close"].iloc[5])


def test_independent_hits_matches_direction_table():
    px = walk(9, 40)
    rng = np.random.default_rng(9)
    rows = []
    for a, t in zip(px.index[20:-1], px.index[21:]):
        c = px["close"].loc[a]
        rows.append({"asof": a, "target": t, "close_t": c, "paths": c * (1 + rng.normal(0, 0.01, 8))})
    pred = pd.DataFrame(rows)
    f = ks.daily_frame(pred, px, n_paths=6, context=10)
    assert rx.independent_hits(pred, px, 6)["hits"] == ks.direction_table(f)["model_hits"]


def test_fan_stats_by_hand():
    paths = np.array([[101, 102], [99, 98], [100, 104], [102, 106]], float)
    st = ks.fan_stats(paths, 100.0, np.array([100.5, 103.0]))
    assert st["end_pred_ret"] == pytest.approx(0.025) and st["end_actual_ret"] == pytest.approx(0.03)
    assert st["direction_right"] and st["end_p_up"] == 0.75
    assert st["share_days_inside_10_90"] == 1.0


def test_rescale_cache_to_current_units():
    # cache made on data that was later divided by 1.01 everywhere (a new dividend adjustment)
    px = walk(10, 30)
    old = px["close"] * 1.01
    rows = [{"symbol": "X", "seed": 0, "asof": a, "target": t, "close_t": old.loc[a],
             "paths": np.array([old.loc[a] * 1.02, old.loc[a] * 0.99])} for a, t in zip(px.index[5:8], px.index[6:9])]
    pred = pd.DataFrame(rows)
    out, fac = rx.to_current_units(pred, {"X": px})
    assert fac["X"]["rescaled"] and fac["X"]["min"] == pytest.approx(1 / 1.01)
    r0 = out.iloc[0]
    assert r0["close_t"] == px["close"].loc[r0["asof"]]
    assert np.asarray(r0["paths"]) == pytest.approx([r0["close_t"] * 1.02, r0["close_t"] * 0.99])
    # a factor that is not constant means the history really changed: refuse
    bad = pred.copy()
    bad.loc[0, "close_t"] *= 1.05
    with pytest.raises(SystemExit):
        rx.to_current_units(bad, {"X": px})
