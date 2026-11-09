"""Hand-checkable tests. Synthetic data only; no real market data in this repo."""
import numpy as np
import pandas as pd
import pytest

from backtest import checks, data, engine, metrics
from strategies import ma_cross


def bars(opens, closes):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    o, c = np.array(opens, float), np.array(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1.0}, index=idx)


def test_next_open_execution_by_hand():
    # day0 close 100; signal buy at day0 close -> buy at day1 open 110; day1 close 121.
    px = bars([100, 110, 121], [100, 121, 121])
    tgt = pd.Series([1.0, 1.0, 1.0], index=px.index)
    bt = engine.vectorized(px, tgt)
    # day0: flat. day1: overnight 100->110 not ours, intraday 110->121 = +10%. day2: overnight 0, intraday 0.
    assert bt["equity"].round(10).tolist() == [1.0, 1.1, 1.1]


def test_cost_charged_on_weight_change():
    px = bars([100, 100, 100], [100, 100, 100])
    tgt = pd.Series([1.0, 0.0, 0.0], index=px.index)
    bt = engine.vectorized(px, tgt, cost_bps=10)
    # buy on day1 (10bp), sell on day2 (10bp)
    assert bt["equity"].iloc[-1] == pytest.approx(0.999 * 0.999)


def test_cash_interest_when_flat():
    px = bars([100, 100], [100, 100])
    rf = pd.Series([0.001, 0.001], index=px.index)
    bt = engine.vectorized(px, pd.Series(0.0, index=px.index), rf_daily=rf)
    assert bt["equity"].iloc[-1] == pytest.approx(1.001 ** 2)


def test_vectorized_matches_event_loop_random():
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 500))
    o = c * (1 + rng.normal(0, 0.003, 500))
    px = bars(o, c)
    tgt = pd.Series((rng.random(500) > 0.5).astype(float), index=px.index)
    rf = pd.Series(0.0001, index=px.index)
    assert checks.reconcile(px, tgt, 7.0, rf)["passed"]


def test_cross_rules():
    # flat, then a clear uptrend (golden cross), then a downtrend (death cross)
    c = [10.0] * 6 + [11, 12, 13, 14, 15] + [14, 12, 10, 8, 6, 4]
    px = bars(c, c)
    t = ma_cross.target(px, fast=2, slow=4)
    first_buy = t.index[t == 1][0]
    assert first_buy == px.index[6]          # first day fast > slow after equal
    assert t.iloc[-1] == 0.0                 # death cross reached


def test_starts_above_waits_for_next_cross():
    c = [1, 2, 3, 4, 5, 6, 7, 8]            # fast above slow from the first valid day
    px = bars(c, c)
    t = ma_cross.target(px, fast=2, slow=4)
    assert (t == 0).all()


def test_lookahead_check_catches_cheater():
    rng = np.random.default_rng(2)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    px = bars(c, c)
    honest = checks.lookahead_check(lambda p: ma_cross.target(p, 5, 20), px)
    cheater = checks.lookahead_check(lambda p: (p["close"].shift(-1) > p["close"]).astype(float), px)
    assert honest["passed"] and not cheater["passed"]


def test_metrics_by_hand():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    eq = pd.Series([1.0, 2.0, 1.0], index=idx)
    assert metrics.max_drawdown(eq) == pytest.approx(-0.5)
    assert metrics.longest_underwater(eq)["recovered"] is False
    eq2 = pd.Series([1.0, 1.21], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    assert metrics.cagr(eq2) == pytest.approx(0.1, abs=2e-4)   # 2 years (731 days) of +21%


# ---- episode 2: lump sum vs DCA ----
from strategies import dca as dcam


def monthly_bars(prices_by_month, days_per_month=2):
    """Each month: first day open=close=p, second day the same. Easy to hand-check."""
    idx, o = [], []
    for m, p in enumerate(prices_by_month):
        for d in range(days_per_month):
            idx.append(pd.Timestamp(2020, 1, 1) + pd.DateOffset(months=m, days=d)); o.append(p)
    o = np.array(o, float)
    return pd.DataFrame({"open": o, "high": o, "low": o, "close": o, "volume": 1.0}, index=pd.DatetimeIndex(idx))


def test_lump_sum_by_hand():
    px = monthly_bars([100, 50, 200])
    v = dcam.lump_sum_path(px, px.index[0], px.index[-1])
    assert v.iloc[-1] == pytest.approx(2.0)


def test_dca_by_hand_no_interest():
    # buy 1/3 at 100, 50, 200 -> shares = (1/3)(1/100 + 1/50 + 1/200) = 0.011667; value at 200 = 2.3333
    px = monthly_bars([100, 50, 200])
    rf = pd.Series(0.0, index=px.index)
    v = dcam.dca_path(px, rf, px.index[0], px.index[-1], 3)
    assert v.iloc[-1] == pytest.approx((1 / 3) * (1 / 100 + 1 / 50 + 1 / 200) * 200)
    assert dcam.dca_final_formula(px, rf, px.index[0], px.index[-1], 3) == pytest.approx(v.iloc[-1])


def test_dca_interest_and_formula_match():
    rng = np.random.default_rng(3)
    px = monthly_bars(list(100 * np.cumprod(1 + rng.normal(0, 0.05, 15))), days_per_month=3)
    rf = pd.Series(0.0002, index=px.index)
    s, e = px.index[0], px.index[-1]
    for n in (1, 4, 12):
        path = dcam.dca_path(px, rf, s, e, n, cost_bps=7)
        assert dcam.dca_final_formula(px, rf, s, e, n, cost_bps=7) == pytest.approx(path.iloc[-1], rel=1e-9)


def test_dca_one_month_equals_lump_sum():
    px = monthly_bars([100, 120, 90])
    rf = pd.Series(0.0, index=px.index)
    a = dcam.dca_path(px, rf, px.index[0], px.index[-1], 1, cost_bps=5).iloc[-1]
    b = dcam.lump_sum_path(px, px.index[0], px.index[-1], cost_bps=5).iloc[-1]
    assert a == pytest.approx(b)


def write_csv(folder, name, text):
    (folder / name).write_text(text)


def test_csv_prices_sorted_and_adjusted_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    write_csv(tmp_path, "SPY.csv",
              "date,open,high,low,close,volume\n"
              "2020-01-03,2,3,1,2.5,20\n"
              "2020-01-02,1,2,1,1.5,10\n")
    px = data.load_prices("SPY")
    assert list(px.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    assert list(px.columns) == data.PRICE_COLS
    assert px["close"].tolist() == [1.5, 2.5]


def test_csv_missing_column_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    write_csv(tmp_path, "SPY.csv", "date,open,high,low,close\n2020-01-02,1,2,1,1.5\n")
    with pytest.raises(ValueError, match="volume"):
        data.load_prices("SPY")


def test_csv_rate_no_lookahead_by_hand(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    write_csv(tmp_path, "rate.csv", "date,rate_pct\n2020-01-02,2.52\n2020-01-06,5.04\n")
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    rf = data.daily_rate(data.load_rate(), idx)
    assert rf.tolist() == pytest.approx([0.0, 0.0001, 0.0001, 0.0002])


def test_csv_without_rate_warns_and_earns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    with pytest.warns(UserWarning, match="rate.csv"):
        rate = data.load_rate()
    assert data.daily_rate(rate, idx).tolist() == [0.0, 0.0]


def test_ma_filter_by_hand():
    from strategies import ma_filter
    # window 3. closes 1,2,3 -> ma 2 on day2, close 3 above -> 1. day3 close 1: ma 2 -> below -> 0.
    # day4 close 2: ma (3+1+2)/3=2, equal -> keep 0. day5 close 9: ma 4 -> above -> 1.
    px = bars([1, 2, 3, 1, 2, 9], [1, 2, 3, 1, 2, 9])
    assert ma_filter.target(px, 3).tolist() == [0, 0, 1, 0, 0, 1]


def test_ma_filter_band_keeps_state_inside_band():
    from strategies import ma_filter
    # window 2, band 10%. day1 close 100 ma 100 -> nothing. day2 close 120 ma 110: 120>121? no -> still 0.
    # day3 close 200 ma 160 -> 200>176 -> 1. day4 close 190 ma 195: 190<175.5? no -> keep 1.
    px = bars([100, 120, 200, 190], [100, 120, 200, 190])
    assert ma_filter.target(px, 2, band=0.10).tolist() == [0, 0, 1, 1]


def test_ma_filter_monthly_only_changes_at_month_end():
    from strategies import ma_filter
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-27", "2020-02-28"])
    c = np.array([1.0, 2.0, 1.0, 1.0, 9.0])
    px = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    # window 1 gives close == ma (keep). Use window 2: ma = [nan,1.5,1.5,1,5]
    # Jan-31 close 2 > 1.5 -> 1; Feb-3 close 1 < 1.5 ignored (not month end); Feb-28 9 > 5 -> 1.
    assert ma_filter.target(px, 2, monthly=True).tolist() == [0, 1, 1, 1, 1]
