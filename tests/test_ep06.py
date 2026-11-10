"""Hand-checkable tests for episode 6. Synthetic data only; no real market data in this repo."""
import numpy as np
import pandas as pd
import pytest

from backtest import checks, engine
from strategies import stop_loss
from run_ep06 import independent_replay


def bars(closes, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1.0}, index=idx)


def test_starts_invested_and_stops_on_tenth_percent_from_peak():
    # bought at the open of day 1; peak close 100; 90 is exactly -10%, so the stop fires on day 3's close
    px = bars([100, 100, 95, 90, 90])
    t = stop_loss.target(px, stop=0.10, reentry=3)
    assert t.tolist() == [1, 1, 1, 0, 0]


def test_ten_percent_is_not_nine_percent():
    px = bars([100, 100, 91, 91, 91])
    assert stop_loss.target(px, stop=0.10, reentry=3).tolist() == [1, 1, 1, 1, 1]


def test_peak_follows_higher_closes():
    # peak moves to 120; 108 is exactly -10% from 120, while 100 would never be hit
    px = bars([100, 120, 110, 108, 108])
    assert stop_loss.target(px, stop=0.10, reentry=3).tolist() == [1, 1, 1, 0, 0]


def test_peak_counts_only_while_held():
    # day 0 close 200 happens before the position exists (bought at open of day 1), so it must not be the peak.
    px = bars([200, 100, 95, 91, 91])
    assert stop_loss.target(px, stop=0.10, reentry=3).tolist() == [1, 1, 1, 1, 1]


def test_reentry_needs_a_new_high_above_the_previous_window():
    # stopped on day 2 (90). Window 3: day 4 close 100 = max(100,90,90)? equal does not count; day 5 close 101 > 100 does.
    px = bars([100, 100, 90, 90, 100, 101, 101])
    t = stop_loss.target(px, stop=0.10, reentry=3)
    assert t.tolist() == [1, 1, 0, 0, 0, 1, 1]


def test_no_reentry_before_window_is_filled():
    # stop on day 2. Window 5 needs 5 earlier closes: not available on days 3-4, so the jump to 150 cannot fire.
    # Day 5: 150 is not above max(100,100,90,150,150). Day 6: 160 is above 150.
    px = bars([100, 100, 90, 150, 150, 150, 160])
    assert stop_loss.target(px, stop=0.10, reentry=5).tolist() == [1, 1, 0, 0, 0, 0, 1]


def test_peak_resets_after_coming_back():
    # stop day 2. Back in signal day 4 (96 > max(90,95)), bought at the open of day 5.
    # Day 5 close 87 becomes the new peak: against the old peak 100 it would have stopped out at once.
    # Day 7: 78 <= 87 * 0.9 = 78.3, stopped again.
    px = bars([100, 100, 90, 95, 96, 87, 87, 78])
    assert stop_loss.target(px, stop=0.10, reentry=2).tolist() == [1, 1, 0, 0, 1, 1, 1, 0]


def test_cost_basis_uses_entry_open_not_peak():
    # bought at open of day 1 = 100; close climbs to 110; 97 is -11.8% from the peak but only -3% from cost
    px = bars([100, 110, 105, 97, 97], opens=[100, 100, 110, 105, 97])
    assert stop_loss.target(px, stop=0.10, reentry=3).tolist() == [1, 1, 1, 0, 0]
    assert stop_loss.target(px, stop=0.10, reentry=3, basis="cost").tolist() == [1, 1, 1, 1, 1]


def test_cost_basis_triggers_on_entry_day():
    # entry open 100, entry-day close 89: already -11% from cost
    px = bars([100, 89, 89, 89], opens=[100, 100, 89, 89])
    assert stop_loss.target(px, stop=0.10, reentry=3, basis="cost").tolist() == [1, 0, 0, 0]


def test_wait_exactly_n_days_flat():
    # stop signal day 2; flat days 3 and 4 (wait=2); target flips on day 4, bought at the open of day 5
    px = bars([100, 100, 90, 85, 80, 75, 70])
    t = stop_loss.target(px, stop=0.10, wait=2)
    assert t.tolist() == [1, 1, 0, 0, 1, 1, 1]
    bt = engine.vectorized(px, t)
    assert bt["weight"].tolist() == [0, 1, 1, 0, 0, 1, 1]


def test_never_back_stays_flat():
    px = bars([100, 100, 90, 100, 200, 300])
    assert stop_loss.target(px, stop=0.10, never_back=True).tolist() == [1, 1, 0, 0, 0, 0]


def test_ledger_by_hand():
    # stop signal day 2 (close 90); sold at the open of day 3 = 85; low close afterwards 80;
    # back-in signal day 4 (101 > max(100,90,80)); bought at the open of day 5 = 100.
    px = bars([100, 100, 90, 80, 101, 101], opens=[100, 100, 90, 85, 88, 100])
    t = stop_loss.target(px, stop=0.10, reentry=3)
    w = engine.vectorized(px, t)["weight"]
    row = stop_loss.ledger(px, w).iloc[0]
    assert row["exit_open"] == 85 and row["reentry_open"] == 100 and row["days_out"] == 2
    assert row["price_change"] == pytest.approx(100 / 85 - 1)
    assert row["low_vs_exit"] == pytest.approx(80 / 85 - 1)
    assert row["higher"] and not row["whipsaw"]    # it did fall 5.9% below the sale price


def test_ledger_whipsaw():
    # sold at 90; closes never below 90; bought back at 100: higher and a whipsaw, by both definitions
    px = bars([100, 100, 90, 95, 101, 101], opens=[100, 100, 90, 90, 95, 100])
    t = stop_loss.target(px, stop=0.10, reentry=3)
    row = stop_loss.ledger(px, engine.vectorized(px, t)["weight"]).iloc[0]
    assert row["higher"] and row["whipsaw"] and row["whipsaw_strict"]


def test_ledger_open_position_has_no_reentry():
    px = bars([100, 100, 90, 80])
    t = stop_loss.target(px, stop=0.10, reentry=2)
    row = stop_loss.ledger(px, engine.vectorized(px, t)["weight"]).iloc[0]
    assert row["reentry"] is None or pd.isna(row["reentry"])
    assert row["higher"] is None or pd.isna(row["higher"])


def test_no_lookahead_on_random_path():
    rng = np.random.default_rng(6)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 800))
    px = bars(c, opens=c * (1 + rng.normal(0, 0.005, 800)))
    assert checks.lookahead_check(lambda p: stop_loss.target(p), px)["passed"]


def test_engines_reconcile_on_random_path():
    rng = np.random.default_rng(7)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 800))
    px = bars(c, opens=c * (1 + rng.normal(0, 0.005, 800)))
    rf = pd.Series(0.0001, index=px.index)
    assert checks.reconcile(px, stop_loss.target(px), 5.0, rf)["passed"]


def test_independent_replay_matches_engine():
    rng = np.random.default_rng(8)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 1500))
    px = bars(c, opens=c * (1 + rng.normal(0, 0.005, 1500)))
    rf = pd.Series(0.0001, index=px.index)
    bt = engine.vectorized(px, stop_loss.target(px), 5.0, rf)
    r = independent_replay(px, rf, 0.10, 20, 5.0)
    assert r["final_multiple"] == pytest.approx(bt["equity"].iloc[-1] / bt["equity"].iloc[0], rel=1e-9)
    assert r["stops"] == int(((bt["weight"] == 0) & (bt["weight"].shift(1) > 0)).sum())
