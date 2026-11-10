"""Hand-checkable tests for episode 8. Synthetic data only; no real market data in this repo."""
import numpy as np
import pandas as pd
import pytest

from backtest import checks, engine
from strategies import rebalance_6040 as rb
from run_ep08 import independent_replay


def frame(dates, a_open, a_close, b_open=None, b_close=None):
    idx = pd.DatetimeIndex(dates)
    b_open = b_open if b_open is not None else [100.0] * len(idx)
    b_close = b_close if b_close is not None else [100.0] * len(idx)
    return pd.DataFrame({"A": a_open, "B": b_open}, index=idx, dtype=float), pd.DataFrame({"A": a_close, "B": b_close}, index=idx, dtype=float)


DATES = ["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04"]
W = {"A": 0.5, "B": 0.5}


def paths(n=1500, seed=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-05", periods=n)
    closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, (n, 2)), axis=0)
    opens = closes * (1 + rng.normal(0, 0.004, (n, 2)))
    return pd.DataFrame(opens, index=idx, columns=["A", "B"]), pd.DataFrame(closes, index=idx, columns=["A", "B"])


def test_flags_first_trading_day_of_each_period():
    idx = pd.DatetimeIndex(["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04", "2022-02-01", "2022-04-01", "2022-04-04", "2023-01-03"])
    assert rb.rebalance_flags(idx, "annual").tolist() == [False, False, True, False, False, False, False, True]
    assert rb.rebalance_flags(idx, "quarterly").tolist() == [False, False, True, False, False, True, False, True]
    assert rb.rebalance_flags(idx, "monthly").tolist() == [False, False, True, False, True, True, False, True]
    assert rb.rebalance_flags(idx, "none").tolist() == [False] * 8


def test_annual_month_is_a_parameter():
    idx = pd.DatetimeIndex(["2022-03-31", "2022-04-01", "2022-05-02", "2023-04-03"])
    assert rb.rebalance_flags(idx, "annual", month=4).tolist() == [False, True, False, True]
    assert rb.rebalance_flags(idx, "annual").tolist() == [False, False, False, False]


def test_first_bar_only_decides_and_earns_cash():
    # day 0: nothing is held, it earns the rate. Day 1: buys at the open, 50/50, 10 bp on the whole amount.
    o, c = frame(DATES[:2], [100, 100], [100, 110])
    rf = pd.Series([0.01, 0.5], index=o.index)    # the second rate must not be used: the money is invested
    bt = rb.simulate(o, c, W, "none", 10.0, rf)
    assert bt["equity"].iloc[0] == pytest.approx(1.01)
    expected = 1.01 * 0.999 * (0.5 * 110 / 100 + 0.5)
    assert bt["equity"].iloc[1] == pytest.approx(expected)


def test_rebalance_by_hand():
    # A opens 120 on the first trading day of January, B stays 100. After the first buy each sleeve is 0.4995.
    o, c = frame(DATES, [100, 100, 120, 120], [100, 100, 130, 130])
    bt = rb.simulate(o, c, W, "annual", 10.0)
    assert bt["equity"].iloc[1] == pytest.approx(0.999)
    # at the rebalance open: A 0.4995 x 1.2 = 0.5994, B 0.4995, total 1.0989
    drift_a = 0.5994 / 1.0989
    turnover = abs(0.5 - drift_a) + abs(0.5 - (1 - drift_a))
    assert bt["turnover"].iloc[2] == pytest.approx(turnover)
    assert bt["cost"].iloc[2] == pytest.approx(turnover * 10 / 1e4)
    after = 1.0989 * (1 - turnover * 10 / 1e4)
    # the day ends with A up from 120 to 130, B flat
    assert bt["equity"].iloc[2] == pytest.approx(0.5 * after * 130 / 120 + 0.5 * after)
    assert bt["w_A"].iloc[2] == pytest.approx((130 / 120) / (130 / 120 + 1))
    # no trade on the next day
    assert bt["turnover"].iloc[3] == 0.0
    assert bt["traded"].tolist() == [False, True, True, False]


def test_never_rebalanced_weights_drift_and_cost_once():
    o, c = frame(DATES, [100, 100, 120, 120], [100, 100, 120, 150])
    bt = rb.simulate(o, c, W, "none", 10.0)
    assert bt["cost"].tolist() == pytest.approx([0.0, 0.001, 0.0, 0.0])
    assert bt["w_A"].iloc[-1] == pytest.approx(1.5 / (1.5 + 1.0))


def test_no_cost_when_already_on_target():
    o, c = frame(DATES, [100, 100, 100, 100], [100, 100, 100, 100])
    bt = rb.simulate(o, c, W, "annual", 10.0)
    assert bt["turnover"].iloc[2] == 0.0
    assert bt["traded"].iloc[2]


def test_weights_must_sum_to_one():
    o, c = frame(DATES, [100] * 4, [100] * 4)
    with pytest.raises(ValueError):
        rb.simulate(o, c, {"A": 0.6, "B": 0.5}, "annual")
    with pytest.raises(ValueError):
        rb.rebalance_flags(o.index, "weekly")


def test_ledger_by_hand():
    # A opens 120 at the January rebalance, then 150 on the last bar; the ledger measures from open to the last close.
    o, c = frame(DATES, [100, 100, 120, 120], [100, 100, 120, 150])
    bt = rb.simulate(o, c, W, "annual", 0.0)
    row = rb.ledger(o, c, W, bt, stock="A").iloc[0]
    drift_a = 0.6 / 1.1                        # 0.5 x 1.2 against 0.5
    assert row["stock_weight_before"] == pytest.approx(drift_a)
    assert row["action"] == "sold_stocks"
    traded = 0.5 * 150 / 120 + 0.5
    left = drift_a * 150 / 120 + (1 - drift_a)
    assert row["effect"] == pytest.approx(traded / left - 1)
    assert row["effect"] < 0                   # A kept rising, so trimming it hurt here


def test_ledger_counts_a_buy_side_rebalance():
    o, c = frame(DATES, [100, 100, 80, 80], [100, 100, 80, 120])
    bt = rb.simulate(o, c, W, "annual", 0.0)
    row = rb.ledger(o, c, W, bt, stock="A").iloc[0]
    assert row["action"] == "bought_stocks"
    assert row["effect"] > 0                   # A rebounded after the top-up


def test_portfolio_engine_matches_repo_engine_for_one_asset():
    o, c = paths()
    rf = pd.Series(0.0002, index=o.index)
    px = pd.DataFrame({"open": o["A"], "high": c["A"], "low": c["A"], "close": c["A"], "volume": 1.0})
    a = rb.simulate(o, c, {"A": 1.0}, "none", 5.0, rf)["equity"]
    b = engine.vectorized(px, pd.Series(1.0, index=o.index), 5.0, rf)["equity"]
    assert (a / b - 1).abs().max() < 1e-12


@pytest.mark.parametrize("freq", ["none", "annual", "quarterly", "monthly"])
def test_two_engines_reconcile(freq):
    o, c = paths()
    rf = pd.Series(0.0002, index=o.index)
    v = rb.simulate(o, c, {"A": 0.6, "B": 0.4}, freq, 5.0, rf)["equity"]
    e = rb.event_loop(o, c, {"A": 0.6, "B": 0.4}, freq, 5.0, rf)
    assert (v / e - 1).abs().max() < 1e-12


@pytest.mark.parametrize("freq", ["none", "annual", "quarterly", "monthly"])
def test_independent_replay_matches_engine(freq):
    o, c = paths()
    rf = pd.Series(0.0002, index=o.index)
    w = {"A": 0.6, "B": 0.4}
    bt = rb.simulate(o, c, w, freq, 5.0, rf)
    r = independent_replay({k: o[k].tolist() for k in o}, {k: c[k].tolist() for k in c},
                           [d.to_pydatetime() for d in o.index], w, freq, float(rf.iloc[0]), 5.0)
    assert r["final_multiple"] == pytest.approx(bt["equity"].iloc[-1] / bt["equity"].iloc[0], rel=1e-10)
    assert r["rebalances"] == int(bt["traded"].sum()) - 1


def test_no_lookahead_after_cutting_the_data():
    o, c = paths()
    joint = pd.concat({"o": o, "c": c}, axis=1)
    fn = lambda p: rb.simulate(p["o"], p["c"], {"A": 0.6, "B": 0.4}, "annual", 5.0)["equity"]
    assert checks.lookahead_check(fn, joint)["passed"]


def test_future_prices_do_not_change_the_past():
    o, c = paths()
    base = rb.simulate(o, c, {"A": 0.6, "B": 0.4}, "monthly", 5.0)
    o2, c2 = o.copy(), c.copy()
    o2.iloc[800:] *= 3.0
    c2.iloc[800:] *= 0.2
    alt = rb.simulate(o2, c2, {"A": 0.6, "B": 0.4}, "monthly", 5.0)
    assert np.allclose(alt["equity"].iloc[:800], base["equity"].iloc[:800])
    assert (alt["traded"].iloc[:800].values == base["traded"].iloc[:800].values).all()
