"""Hand-checkable tests for episode 7. Synthetic data only; no real market data in this repo."""
import numpy as np
import pandas as pd
import pytest

from backtest import checks, engine
from strategies import macd
from run_ep07 import independent_replay


def bars(closes, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1.0}, index=idx)


def random_px(seed, n=1500):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    return bars(c, opens=c * (1 + rng.normal(0, 0.004, n)))


def frame(dif, dea):
    """Prices are irrelevant here: crosses() is exercised through a patched lines()."""
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": np.array(dif, float) - np.array(dea, float)},
                        index=pd.bdate_range("2020-01-01", periods=len(dif)))


def patched_target(monkeypatch, dif, dea, **kw):
    fr = frame(dif, dea)
    monkeypatch.setattr(macd, "lines", lambda *a, **k: fr.copy())
    return macd.target(bars([100.0] * len(dif)), warmup=1, **kw).tolist()


def test_ema_by_hand():
    # span 3: alpha = 0.5. 10, then .5*20+.5*10=15, then .5*30+.5*15=22.5
    out = macd.ema(pd.Series([10.0, 20.0, 30.0]), 3)
    assert out.tolist() == pytest.approx([10, 15, 22.5])


def test_ema_sma_seed_by_hand():
    # span 3: first value is the average of the first 3 inputs (20); then .5*40+.5*20=30
    out = macd.ema(pd.Series([10.0, 20.0, 30.0, 40.0]), 3, seed="sma")
    assert np.isnan(out.iloc[1]) and out.iloc[2:].tolist() == pytest.approx([20, 30])


def test_dif_dea_by_hand():
    # fast 1 means EMA = close; slow 3 alpha .5; signal 3 alpha .5
    # close 10, 20, 30; slow 10, 15, 22.5; dif 0, 5, 7.5; dea 0, 2.5, 5
    ln = macd.lines(bars([10, 20, 30]), fast=1, slow=3, signal=3)
    assert ln["dif"].tolist() == pytest.approx([0, 5, 7.5])
    assert ln["dea"].tolist() == pytest.approx([0, 2.5, 5])
    assert ln["hist"].tolist() == pytest.approx([0, 2.5, 2.5])


def test_golden_cross_needs_strictly_above(monkeypatch):
    # day 1 equals (no cross yet), day 2 strictly above: yesterday <= and today >
    assert patched_target(monkeypatch, [-1, 0, 1, 1], [0, 0, 0, 0]) == [0, 0, 1, 1]


def test_touching_without_crossing_is_nothing(monkeypatch):
    # DIF comes up to DEA and goes back down: no golden cross, nothing bought
    assert patched_target(monkeypatch, [-1, 0, -1, -1], [0, 0, 0, 0]) == [0, 0, 0, 0]


def test_death_cross_sells_and_position_holds_between(monkeypatch):
    assert patched_target(monkeypatch, [-1, 1, 2, 1, -1, -2], [0, 0, 0, 0, 0, 0]) == [0, 1, 1, 1, 0, 0]


def test_starting_above_waits_for_next_golden_cross(monkeypatch):
    # DIF is above DEA from the start: no buy until it has fallen back and crossed up again
    assert patched_target(monkeypatch, [1, 2, 1, -1, 1], [0, 0, 0, 0, 0]) == [0, 0, 0, 0, 1]


def test_zero_above_ignores_cross_below_zero(monkeypatch):
    # golden cross on day 1 at DIF -1 > DEA -2: below zero, ignored. Later DIF rises above 0 with no new cross: still flat.
    dif, dea = [-3, -1, 1, 2], [-2, -2, -1, 0]
    assert patched_target(monkeypatch, dif, dea, mode="zero_above") == [0, 0, 0, 0]
    assert patched_target(monkeypatch, dif, dea, mode="cross") == [0, 1, 1, 1]


def test_zero_above_needs_both_lines_above_zero(monkeypatch):
    # cross with DIF +1 but DEA -0.5: DEA is below zero, so it does not count
    assert patched_target(monkeypatch, [-1, 1, 1], [-0.5, -0.5, -0.5], mode="zero_above") == [0, 0, 0]


def test_zero_above_buys_above_zero_and_death_cross_sells_anywhere(monkeypatch):
    # cross above zero on day 1; death cross on day 3 is below zero and still sells
    dif, dea = [1, 3, 3, -2, -2], [2, 2, 2, -1, -1]
    assert patched_target(monkeypatch, dif, dea, mode="zero_above") == [0, 1, 1, 0, 0]


def test_dif_zero_is_a_cross_of_the_two_emas():
    px = random_px(1, 800)
    t = macd.target(px, mode="dif_zero")
    fast, slow = macd.ema(px["close"], 12), macd.ema(px["close"], 26)
    f, s = fast.copy(), slow.copy()
    f.iloc[:33], s.iloc[:33] = np.nan, np.nan
    up = (f.shift(1) <= s.shift(1)) & (f > s)
    down = (f.shift(1) >= s.shift(1)) & (f < s)
    sig = pd.Series(np.nan, index=px.index)
    sig[up], sig[down] = 1.0, 0.0
    assert t.tolist() == sig.ffill().fillna(0.0).tolist()


def test_warmup_blocks_early_signals():
    # a strong V shape makes an early golden cross; a warm-up past it must suppress the buy
    c = [100 - i for i in range(12)] + [88 + 3 * i for i in range(12)]
    px = bars(c)
    no_block = macd.target(px, fast=2, slow=4, signal=2, warmup=2)
    first = int(np.argmax(no_block.to_numpy() == 1.0))
    assert first > 2
    blocked = macd.target(px, fast=2, slow=4, signal=2, warmup=first + 1)
    assert blocked.iloc[first] == 0.0


def test_trade_table_by_hand():
    # held on days 2, 3, 4 and sold at the open of day 5: 3 days held. Second entry is still open at the end.
    px = bars([100] * 9)
    w = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 1], index=px.index, dtype=float)
    ret = pd.Series([0, 0, .01, .02, -.01, 0, 0, .03, 0], index=px.index, dtype=float)
    tr = macd.trade_table(px, w, ret, short_days=3)
    assert len(tr) == 2
    first = tr.iloc[0]
    assert first["days_held"] == 3 and first["short"] and not first["open"]
    assert first["return"] == pytest.approx(1.01 * 1.02 * 0.99 * 1.0 - 1)
    assert tr.iloc[1]["open"] and not tr.iloc[1]["short"]


def test_trade_stats_by_hand():
    tr = pd.DataFrame({"return": [.1, -.05, .3], "days_held": [5, 20, 40], "short": [True, False, False], "open": [False, False, True]})
    st = macd.trade_stats(tr, years=2.0)
    assert st["n"] == 3 and st["closed"] == 2 and st["per_year"] == 1.5
    assert st["win_rate"] == 0.5 and st["n_short"] == 1 and st["median_days_held"] == 12.5
    assert st["avg_win"] == pytest.approx(.1) and st["avg_loss"] == pytest.approx(-.05)


def test_future_change_check_catches_a_cheat():
    px = random_px(2, 600)
    cheat = lambda p: (p["close"].shift(-1) > p["close"]).astype(float)
    assert not macd.future_change_check(cheat, px, n_cuts=5)["passed"]


def test_no_lookahead_on_random_path():
    px = random_px(3, 800)
    for mode in ("cross", "zero_above", "dif_zero"):
        assert checks.lookahead_check(lambda p: macd.target(p, mode=mode), px)["passed"]
        assert macd.future_change_check(lambda p: macd.target(p, mode=mode), px)["passed"]


def test_engines_reconcile_on_random_path():
    px = random_px(4, 800)
    rf = pd.Series(0.0001, index=px.index)
    assert checks.reconcile(px, macd.target(px), 5.0, rf)["passed"]


def test_independent_replay_matches_engine():
    px = random_px(5)
    rf = pd.Series(0.0001, index=px.index)
    warm = 60
    t = macd.target(px, warmup=warm)
    bt = engine.vectorized(px, t, 5.0, rf).iloc[warm - 1:]
    r = independent_replay(px, rf, 12, 26, 9, warm, 5.0)
    assert r["final_multiple"] == pytest.approx(bt["equity"].iloc[-1] / bt["equity"].iloc[0], rel=1e-9)
    assert r["entries"] == int(((bt["weight"] > 0) & (bt["weight"].shift(1) == 0)).sum())
