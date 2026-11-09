"""Hand-checkable tests for episode 4. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from backtest import checks, engine
from strategies import rsi_dip
import run_ep04


def bars(closes, opens=None, lows=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    l = np.array(lows, float) if lows is not None else np.minimum(o, c)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": l, "close": c, "volume": 1.0}, index=idx)


def test_rsi_period_3_by_hand():
    # changes +1, -1, +2, then -1
    r = rsi_dip.rsi(pd.Series([10.0, 11, 10, 12, 11]), 3)
    assert r.iloc[:3].isna().all()
    # seed: gain (1+0+2)/3 = 1, loss (0+1+0)/3 = 1/3, RS = 3, RSI = 75
    assert r.iloc[3] == pytest.approx(75.0)
    # next: gain (1*2+0)/3 = 2/3, loss (1/3*2+1)/3 = 5/9, RS = 6/5, RSI = 100 - 100/2.2
    assert r.iloc[4] == pytest.approx(100 - 100 / 2.2)


def test_rsi_seed_is_simple_average_not_ewm():
    c = pd.Series([10.0, 11, 10, 12, 11])
    wrong = c.diff().clip(lower=0).ewm(alpha=1 / 3, adjust=False).mean() / (-c.diff().clip(upper=0)).ewm(alpha=1 / 3, adjust=False).mean()
    assert 100 - 100 / (1 + wrong.iloc[3]) != pytest.approx(rsi_dip.rsi(c, 3).iloc[3])


def test_rsi_all_up_all_down_flat():
    assert rsi_dip.rsi(pd.Series(np.arange(1.0, 9.0)), 3).iloc[-1] == 100.0
    assert rsi_dip.rsi(pd.Series(np.arange(9.0, 1.0, -1.0)), 3).iloc[-1] == 0.0
    assert rsi_dip.rsi(pd.Series([5.0] * 8), 3).iloc[-1] == 50.0


def test_first_rsi_is_bar_14():
    r = rsi_dip.rsi(pd.Series(np.linspace(100, 80, 30)), 14)
    assert r.iloc[:14].isna().all() and r.iloc[14:].notna().all()


def test_state_machine_buy_hold_through_oversold_sell():
    # all down: RSI 0 -> buy on the first valid bar, stay in; then all up -> RSI eventually above 70 -> sell
    c = [20.0, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20]
    t = rsi_dip.target(bars(c), period=3)
    assert t.iloc[:3].sum() == 0
    assert t.iloc[3] == 1.0
    assert t.iloc[4:6].eq(1.0).all()
    sell_day = t.index[(t == 0) & (t.index > t.index[3])][0]
    assert rsi_dip.rsi(bars(c)["close"], 3).loc[sell_day] > 70


def test_equal_thresholds_do_nothing():
    # flat prices give RSI 50; buy=50 must not trigger because the rule is strictly below
    t = rsi_dip.target(bars([5.0] * 10), period=3, buy=50, sell=70)
    assert t.sum() == 0
    # sell threshold equal to RSI 100: strictly above never fires, position stays open
    c = [20.0, 19, 18, 17, 16, 17, 18, 19, 20, 21, 22]
    t = rsi_dip.target(bars(c), period=3, buy=30, sell=100)
    assert t.iloc[-1] == 1.0


def test_ma_exit_variant():
    # first valid RSI(2) at bar 2; falls so buy; close above 5-day average sells
    c = [10.0, 9, 8, 7, 6, 5, 4, 8, 9, 9]
    px = bars(c)
    t = rsi_dip.target_ma_exit(px, period=2, buy=10, ma=5)
    assert t.iloc[2] == 1.0
    assert t.loc[px.index[7]] == 0.0              # bar 7 close 8 > mean(6,5,4,8,7)=6


def test_ma_exit_ma_not_ready_blocks_nothing_false():
    c = [10.0, 9, 8, 9, 10]
    px = bars(c)
    # the 5-day average does not exist until bar 4; a NaN average must not sell
    t = rsi_dip.target_ma_exit(px, period=2, buy=10, ma=5)
    assert t.iloc[3] == 1.0


def test_lookahead_passes_and_engines_reconcile():
    rng = np.random.default_rng(3)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))
    o = c * (1 + rng.normal(0, 0.003, 600))
    px = bars(c, o)
    assert checks.lookahead_check(lambda p: rsi_dip.target(p), px)["passed"]
    t = rsi_dip.target(px)
    assert t.diff().abs().sum() > 2
    assert checks.reconcile(px, t, 5.0, pd.Series(0.0001, index=px.index))["passed"]


def test_trade_log_by_hand():
    # bars: signal on bar 1, buy bar 2 at open 100, closes 95 / 90 / 105, sell bar 5
    opens = [100, 100, 100, 96, 91, 106, 106]
    closes = [100, 100, 95, 90, 105, 106, 106]
    lows = [100, 100, 94, 88, 91, 106, 106]
    px = bars(closes, opens, lows)
    t = pd.Series([0, 1, 1, 1, 0, 0, 0], index=px.index, dtype=float)
    bt = engine.vectorized(px, t)
    log = rsi_dip.trade_log(px, bt["weight"], bt["ret"])
    assert len(log) == 1
    row = log.iloc[0]
    assert row["days_held"] == 3
    assert row["mae_close"] == pytest.approx(-0.10)
    assert row["mae_low"] == pytest.approx(-0.12)
    assert row["days_to_worst"] == 1
    assert row["return"] == pytest.approx(106 / 100 - 1)


def test_independent_check_matches_engine():
    rng = np.random.default_rng(5)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 800))
    o = c * (1 + rng.normal(0, 0.003, 800))
    px = bars(c, o)
    rf = pd.Series(0.0001, index=px.index)
    t = rsi_dip.target(px)
    start = rsi_dip.first_signal_date(px)
    eng = engine.vectorized(px, t, 5.0, rf).loc[start:]
    mult = run_ep04.independent_multiple(px, rf, 14, 30.0, 70.0, 5.0)
    assert mult == pytest.approx(float((1 + eng["ret"]).prod()), rel=1e-9)
