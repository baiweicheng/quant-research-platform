"""Tests for the audit-fix batch (M1, m1-m4): SPY total-return benchmark,
delisted force-close, membership-versioned cache key, removal-date convention.
"""

import os

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.engine import run_backtest
from qrp.backtest.metrics import benchmark_equity


# ---------------------------------------------------------------------------
# M1: SPY benchmark credits its own distributions (total-return basis)
# ---------------------------------------------------------------------------

def _flat_spy(periods=10, price=100.0):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame({"SPY": [price] * periods}, index=dates), dates


def test_benchmark_equity_credits_spy_dividend_on_ex_date():
    prices, dates = _flat_spy()
    divs = pd.DataFrame(
        {"ticker": ["SPY"], "ex_date": [dates[4]], "amount": [1.0]}
    )
    eq = benchmark_equity(prices, 100_000.0, dividends=divs)
    vals = eq["portfolio_value"].to_numpy()
    # 1,000 shares x $1 credited from the ex-date onward — not before.
    assert vals[3] == pytest.approx(100_000.0)
    assert vals[4] == pytest.approx(101_000.0)
    assert vals[-1] == pytest.approx(101_000.0)


def test_benchmark_equity_price_only_without_dividends():
    prices, _ = _flat_spy()
    eq = benchmark_equity(prices, 100_000.0)
    assert np.allclose(eq["portfolio_value"].to_numpy(), 100_000.0)
    eq2 = benchmark_equity(
        prices, 100_000.0,
        dividends=pd.DataFrame(columns=["ticker", "ex_date", "amount"]),
    )
    assert np.allclose(eq2["portfolio_value"].to_numpy(), 100_000.0)


def test_benchmark_equity_ignores_ex_dates_outside_grid():
    prices, dates = _flat_spy()
    divs = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY", "XXX"],
            "ex_date": [dates[0] - pd.Timedelta(days=10), dates[-1] + pd.Timedelta(days=10), dates[4]],
            "amount": [5.0, 5.0, 5.0],
        }
    )
    eq = benchmark_equity(prices, 100_000.0, dividends=divs)
    # Pre-grid and post-grid ex-dates credit nothing; XXX isn't held.
    assert np.allclose(eq["portfolio_value"].to_numpy(), 100_000.0)


def test_spy_benchmark_ui_credits_spy_dividends(tmp_path):
    """End-to-end: seeded SPY bars + a dividend row flow through the UI helper."""
    from qrp.data.store import upsert_bars, upsert_dividends
    from qrp.ui.helpers import spy_benchmark_for_ui

    db = tmp_path / "m.db"
    dates = pd.bdate_range("2024-01-02", periods=5)
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 100.0, "High": 101.0, "Low": 99.0,
                "Close": [100.0, 101.0, 102.0, 101.0, 110.0],
                "Volume": 1_000_000,
            }
        ),
        "SPY",
        db,
    )
    upsert_dividends(
        pd.DataFrame({"ex_date": ["2024-01-05"], "amount": [0.5]}),
        "SPY",
        db,
    )
    equity, _ = spy_benchmark_for_ui(
        "2024-01-02", "2024-01-08", 100_000.0, db_path=db
    )
    vals = equity["portfolio_value"].to_numpy()
    # 1,000 shares x $0.50 credited on the 2024-01-05 ex-date onward.
    assert vals[2] == pytest.approx(102_000.0)
    assert vals[3] == pytest.approx(101_000.0 + 500.0)
    assert vals[4] == pytest.approx(110_000.0 + 500.0)


# ---------------------------------------------------------------------------
# m2: delisted holdings force-close on the first date after last_valid,
#     at the last real CLOSE (never a forward-filled open)
# ---------------------------------------------------------------------------

def test_delisted_force_closes_first_date_after_last_valid_at_last_close():
    dates = pd.bdate_range("2024-01-02", periods=60)
    raw = [100.0 + i for i in range(40)] + [float("nan")] * 20
    prices = pd.DataFrame({"AAA": 100.0, "BBB": raw}, index=dates).ffill()
    # Opens deliberately differ from closes: the exit must use the close.
    opens = pd.DataFrame(
        {"AAA": 100.0, "BBB": [50.0] * 60}, index=dates
    )
    last_valid = {"AAA": dates[-1], "BBB": dates[39]}

    def always_bbb(_signals, _day):
        return {"BBB": 0.5}

    res = run_backtest(
        prices, pd.DataFrame(index=dates), always_bbb,
        frequency="monthly", cost_bps=0.0, last_valid=last_valid,
        opens=opens, execution="open",
    )
    bbb = res.trades[res.trades["ticker"] == "BBB"]
    sells = bbb[bbb["action"] == "SELL"]
    assert not sells.empty, "the delisted holding must be closed"
    # First trading date after last_valid — not the next monthly rebalance.
    assert sells["date"].iloc[0] == dates[40]
    # Last real CLOSE (139.0), not the forward-filled open (50.0).
    assert sells["price"].iloc[0] == pytest.approx(139.0)
    # Never reopened afterwards.
    later_buys = bbb[(bbb["action"] == "BUY") & (bbb["date"] > dates[40])]
    assert later_buys.empty


# ---------------------------------------------------------------------------
# m3: Compare cache key versions the membership table
# ---------------------------------------------------------------------------

def test_compare_cache_key_changes_with_membership_csv(tmp_path, monkeypatch):
    import qrp.data.sp500_history as sh
    import qrp.ui.helpers as H

    p = tmp_path / "hist.csv"
    p.write_text("ticker,date_added,date_removed\nAAA,2020-01-01,\n")
    monkeypatch.setattr(sh, "CSV_PATH", p)
    sh.load_history.cache_clear()
    try:
        kw = dict(
            strategy_names=["momentum"], tickers=["AAA"],
            start="2024-01-01", end="2024-02-01", capital=100_000,
            max_weight=0.05, cost_bps=10.0, execution="open",
        )
        k1 = H.compare_cache_key(**kw)
        p.write_text("ticker,date_added,date_removed\nAAA,2020-01-01,\nBBB,2020-01-01,\n")
        os.utime(p, (p.stat().st_atime_ns, p.stat().st_mtime_ns + 2_000_000_000))
        sh.load_history.cache_clear()
        k2 = H.compare_cache_key(**kw)
        assert k1 != k2, "a refreshed membership table must invalidate the cache"
    finally:
        sh.load_history.cache_clear()


def test_membership_version_absent_never_crashes(tmp_path, monkeypatch):
    import qrp.data.sp500_history as sh
    import qrp.ui.helpers as H

    monkeypatch.setattr(sh, "CSV_PATH", tmp_path / "missing.csv")
    assert H._membership_version() == "absent"
    # ...and the key still builds.
    k = H.compare_cache_key(
        ["momentum"], ["AAA"], "2024-01-01", "2024-02-01",
        100_000, 0.05, 10.0, "open",
    )
    assert isinstance(k, str) and len(k) == 32


# ---------------------------------------------------------------------------
# m4: removal-date convention pinned (d < date_removed: unbuyable ON day R)
# ---------------------------------------------------------------------------

def test_removal_day_is_exclusive():
    import qrp.data.sp500_history as sh

    df = sh.load_history()
    assert df is not None and not df.empty
    removed = df[df["date_removed"] != ""].copy()
    removed["span"] = (
        pd.to_datetime(removed["date_removed"])
        - pd.to_datetime(removed["date_added"])
    ).dt.days
    row = removed[removed["span"] > 60].iloc[0]
    ticker, r = row["ticker"], row["date_removed"]
    assert ticker not in sh.members_on(r), (
        f"{ticker} must be ineligible on its removal date {r}"
    )
    assert ticker in sh.members_on(pd.Timestamp(r) - pd.Timedelta(days=1)), (
        f"{ticker} must still be a member the day before removal"
    )
