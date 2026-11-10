"""Episode 5 hand-checkable tests. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

import run_ep05 as ep
from strategies import donchian_breakout as db


def bars(closes, highs=None, lows=None, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    h = np.array(highs if highs is not None else np.maximum(o, c), float)
    l = np.array(lows if lows is not None else np.minimum(o, c), float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def test_channel_excludes_today_by_hand():
    px = bars([1, 3, 2, 5, 4])
    ch = db.channels(px, entry=2, exit=2)
    assert np.isnan(ch["upper"].iloc[1])            # only one earlier day is known
    assert ch["upper"].iloc[2] == 3                 # max(1, 3), day 2's own close of 2 not used
    assert ch["upper"].iloc[3] == 3                 # max(3, 2), day 3's 5 not used
    assert ch["lower"].iloc[4] == 2                 # min(2, 5)


def test_no_signal_during_warmup():
    px = bars([1, 2, 3, 4])
    assert db.target(px, entry=3, exit=2).tolist() == [0, 0, 0, 1]   # day 3 is the first with 3 prior closes


def test_breakout_enter_and_exit_by_hand():
    # entry=2, exit=2
    #        0  1  2  3  4  5  6
    closes = [1, 2, 3, 4, 3, 2, 3]
    # day 2: 3 > max(1,2)=2 -> buy. day 3: stay. day 4: 3 < min(3,4)? no. day 5: 2 < min(4,3)=3 -> sell.
    # day 6: flat, 3 > max(3,2)=3? no (tie) -> stay flat.
    t = db.target(bars(closes), entry=2, exit=2)
    assert t.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_ties_do_nothing():
    # entry tie: day 2 close equals the upper channel, no entry.
    assert db.target(bars([2, 2, 2]), entry=2, exit=2).tolist() == [0, 0, 0]
    # exit tie: long from day 2 (3 > 2); day 4 close 3 equals lower min(3,4)=3, stays long.
    assert db.target(bars([1, 2, 3, 4, 3]), entry=2, exit=2).tolist() == [0, 0, 1, 1, 1]


def test_can_reenter_the_day_after_exit():
    # entry=1, exit=1. day 1 buys (2>1); day 2 sells (1<2); day 3 re-enters (2>1).
    t = db.target(bars([1, 2, 1, 2]), entry=1, exit=1)
    assert t.tolist() == [0, 1, 0, 1]


def test_exit_and_entry_cannot_both_fire_on_one_day():
    # Both channels contain yesterday, so a close cannot be above the upper and below the lower
    # on the same day. "No entry on the exit day" is therefore impossible, not merely forbidden.
    rng = np.random.default_rng(5)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 3000))
    px = bars(c)
    for a, b in [(1, 1), (2, 1), (20, 10), (55, 20)]:
        ch = db.channels(px, a, b)
        assert not ((px["close"] > ch["upper"]) & (px["close"] < ch["lower"])).any()


def test_use_high_low_channels():
    px = bars([10, 10, 11], highs=[10, 12, 11], lows=[9, 9, 10])
    # close 11 is above the highest close (10) but not above the highest high (12).
    assert db.target(px, entry=2, exit=1).tolist() == [0, 0, 1]
    assert db.target(px, entry=2, exit=1, use_hl=True).tolist() == [0, 0, 0]


def test_ledger_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=5)
    o = pd.Series([10, 10, 20, 30, 15.0], index=idx)
    w = pd.Series([0, 1, 1, 0, 1.0], index=idx)   # buy day 1 at 10, sell day 3 at 30, buy day 4 at 15 (open)
    led = ep.ledger(w, o, 5.0)
    assert len(led) == 2
    assert led.iloc[0]["ret"] == pytest.approx(3 * (1 - 5e-4) ** 2 - 1)
    assert led.iloc[0]["hold_days"] == 2
    assert bool(led.iloc[1]["open"]) is True


def test_streaks_by_hand():
    s = pd.DataFrame({"entry": pd.bdate_range("2020-01-01", periods=5), "exit": pd.bdate_range("2020-01-02", periods=5),
                      "ret": [0.1, -0.1, -0.1, 0.2, -0.05], "open": [False] * 5, "hold_days": [1] * 5,
                      "entry_i": range(5), "exit_i": range(1, 6)})
    st = ep.streaks(s)
    assert st["max_len"] == 2
    assert st["longest"]["n"] == 2
    assert st["longest"]["compound_return"] == pytest.approx(0.9 * 0.9 - 1)


def test_portfolio_sleeves_are_the_average():
    idx = pd.bdate_range("2020-01-01", periods=3)
    legs = {"A": pd.Series([1.0, 1.1, 1.21], index=idx), "B": pd.Series([1.0, 0.9, 0.9], index=idx)}
    avg = ep.combine(legs)
    assert avg.iloc[-1] == pytest.approx((1.21 + 0.9) / 2)


def test_basket_rebalance_by_hand():
    # Two assets, flat on day 0, month end on day 1 forces a rebalance on the next open.
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    o = pd.DataFrame({"A": [1.0, 1.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    c = pd.DataFrame({"A": [1.0, 2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    rf = pd.Series(0.0, index=idx)
    # Day 1 (open buy): A 0.5/1, B 0.5/1; close A=2 -> value 1.5.
    # Day 2 (Feb open, rebalance): value 1.5 with drifted weights A 2/3, B 1/3; targets 1/2 each.
    # Turnover = |1/2-2/3|+|1/2-1/3| = 1/3; cost 10bp on 1/3 of value = 1.5*0.001/3 = 0.0005.
    eq = ep.basket_monthly(o, c, rf, cost_bps=10.0)
    assert eq.iloc[1] == pytest.approx((1 - 0.001) * 1.5)
    assert eq.iloc[2] == pytest.approx((1 - 0.001) * 1.5 * (1 - 0.001 / 3))


def test_basket_two_implementations_agree():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=400)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, (400, 3)), axis=0)
    c = pd.DataFrame(closes, index=idx, columns=list("XYZ"))
    o = c * (1 + rng.normal(0, 0.003, (400, 3)))
    rf = pd.Series(1e-4, index=idx)
    a = ep.basket_monthly(o, c, rf, 5.0)
    b = ep.basket_monthly_recursive(o, c, rf, 5.0)
    assert (a / b - 1).abs().max() < 1e-9
