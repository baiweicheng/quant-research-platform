"""Tests for Phase 2: SPY benchmark, relative metrics, one-way turnover.

All tests use SYNTHETIC data (no network). Every metric is checked
against hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.metrics import (
    benchmark_equity,
    compute_metrics,
    relative_metrics,
)


def _equity(values, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"date": dates, "portfolio_value": list(values)})


# --- turnover: one-way convention --------------------------------------------

def test_turnover_counts_each_leg_once():
    """Buy $1,000 then sell $1,000 of a flat $100k portfolio over exactly
    one year (253 daily values = 252 returns): one-way turnover = 1%."""
    equity = _equity([100_000.0] * 253)
    trades = pd.DataFrame(
        [
            {"shares": 10.0, "price": 100.0},  # buy $1,000
            {"shares": 10.0, "price": 100.0},  # sell $1,000
        ]
    )
    m = compute_metrics(equity, trades)
    # Gross $2,000 / 2 = $1,000 one-way; 1000 / (100000 * 1yr) = 0.01.
    assert m["turnover"]["value"] == pytest.approx(0.01)
    assert "one-way" in m["turnover"]["label"].lower()


def test_turnover_full_rebalance_is_100_percent():
    """Selling the whole $100k portfolio and buying a new one = 100%."""
    equity = _equity([100_000.0] * 253)
    trades = pd.DataFrame(
        [
            {"shares": 1000.0, "price": 100.0},  # sell everything: $100k
            {"shares": 1000.0, "price": 100.0},  # buy everything: $100k
        ]
    )
    m = compute_metrics(equity, trades)
    assert m["turnover"]["value"] == pytest.approx(1.0)


# --- benchmark_equity ----------------------------------------------------------

def test_benchmark_equity_is_spy_buy_and_hold():
    dates = pd.bdate_range("2024-01-02", periods=5)
    spy = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 101.0, 110.0]}, index=dates)
    eq = benchmark_equity(spy, 100_000.0)
    assert list(eq.columns) == ["date", "portfolio_value"]
    # 1000 shares bought at $100, never traded: final = 1000 x $110.
    assert eq["portfolio_value"].iloc[-1] == pytest.approx(110_000.0)
    assert eq["portfolio_value"].iloc[0] == pytest.approx(100_000.0)


def test_benchmark_equity_rejects_non_spy_frame():
    dates = pd.bdate_range("2024-01-02", periods=5)
    with pytest.raises(ValueError, match="exactly one price column"):
        benchmark_equity(pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=dates), 100_000.0)


# --- relative_metrics ------------------------------------------------------------

def _curve(daily_returns, start_value=100_000.0, start="2024-01-02"):
    values = [start_value]
    for r in daily_returns:
        values.append(values[-1] * (1 + r))
    return _equity(values, start)


def test_relative_metrics_double_the_market():
    """Strategy returns exactly 2x the market's: beta 2, alpha 0,
    up/down capture 200%."""
    r_b = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.015])
    m = relative_metrics(_curve(2 * r_b), _curve(r_b))
    assert m["beta"]["value"] == pytest.approx(2.0)
    assert m["alpha"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert m["up_capture"]["value"] == pytest.approx(2.0)
    assert m["down_capture"]["value"] == pytest.approx(2.0)
    # Tracking error = std of (2x - 1x) = std of market, annualized.
    assert m["tracking_error"]["value"] == pytest.approx(
        r_b.std(ddof=1) * np.sqrt(252)
    )


def test_relative_metrics_half_participation():
    """Half the market's moves: beta 0.5, captures 50% / 50%."""
    r_b = np.array([0.02, -0.02, 0.01, -0.01, 0.03, -0.03])
    m = relative_metrics(_curve(0.5 * r_b), _curve(r_b))
    assert m["beta"]["value"] == pytest.approx(0.5)
    assert m["up_capture"]["value"] == pytest.approx(0.5)
    assert m["down_capture"]["value"] == pytest.approx(0.5)


def test_relative_metrics_alpha_is_extra_return():
    """Market + 0.001/day extra: beta 1, alpha ~= 0.001 x 252 = 25.2%."""
    r_b = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.02, 0.01, 0.005])
    m = relative_metrics(_curve(r_b + 0.001), _curve(r_b))
    assert m["beta"]["value"] == pytest.approx(1.0)
    assert m["alpha"]["value"] == pytest.approx(0.252, abs=1e-6)


def test_relative_metrics_downside_protection():
    """Captures all the upside, none of the downside: up 100%, down 0%."""
    r_b = np.array([0.02, -0.02, 0.03, -0.01])
    r_p = np.array([0.02, 0.0, 0.03, 0.0])
    m = relative_metrics(_curve(r_p), _curve(r_b))
    assert m["up_capture"]["value"] == pytest.approx(1.0)
    assert m["down_capture"]["value"] == pytest.approx(0.0)


def test_relative_metrics_plain_english_everywhere():
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    m = relative_metrics(_curve(1.5 * r), _curve(r))
    assert set(m) == {
        "alpha", "beta", "information_ratio",
        "tracking_error", "up_capture", "down_capture",
    }
    for key, entry in m.items():
        assert {"value", "label", "what", "why", "good"} <= set(entry), key
        assert entry["label"] and entry["what"] and entry["why"] and entry["good"], key


def test_relative_metrics_short_history_is_nan_not_crash():
    m = relative_metrics(_equity([100_000.0]), _equity([100_000.0]))
    assert np.isnan(m["beta"]["value"])
    assert np.isnan(m["alpha"]["value"])


def test_relative_metrics_flat_benchmark_is_nan_not_crash():
    strat = _curve([0.01, 0.02, -0.01, 0.03])
    bench = _equity([50_000.0] * 5)  # zero variance
    m = relative_metrics(strat, bench)
    assert np.isnan(m["beta"]["value"])
    assert np.isnan(m["alpha"]["value"])


# --- UI helper: SPY benchmark -----------------------------------------------------

def test_spy_benchmark_helper_names_which_is_which(tmp_path):
    """The helper's error message names the Market data page fix (no jargon)."""
    from qrp.ui.helpers import spy_benchmark_for_ui

    with pytest.raises(ValueError, match="Market data page"):
        spy_benchmark_for_ui("2024-01-02", "2024-01-10", 100_000.0, db_path=tmp_path / "empty.db")


def test_spy_benchmark_helper_with_data(tmp_path):
    from qrp.data.store import upsert_bars
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
    equity, metrics = spy_benchmark_for_ui(
        "2024-01-02", "2024-01-08", 100_000.0, db_path=db
    )
    assert equity["portfolio_value"].iloc[-1] == pytest.approx(110_000.0)
    assert metrics["total_return"]["value"] == pytest.approx(0.10)


def _seed_prices(db, tickers_closes):
    from qrp.data.store import upsert_bars

    dates = pd.bdate_range("2024-01-02", periods=5)
    for ticker, closes in tickers_closes.items():
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": closes, "High": closes, "Low": closes,
                    "Close": closes, "Volume": 1_000_000,
                }
            ),
            ticker,
            db,
        )


def test_compare_benchmark_rows_names_which_is_which(tmp_path):
    """Universe benchmark and SPY benchmark are clearly distinct rows."""
    from qrp.ui.helpers import compare_benchmark_rows

    db = tmp_path / "m.db"
    _seed_prices(db, {"AAA": [100.0, 101, 102, 101, 110.0],
                      "SPY": [400.0, 402, 404, 403, 420.0]})
    bench_row, spy_row = compare_benchmark_rows(
        ["AAA"], "2024-01-02", "2024-01-08", 100_000.0, db_path=db
    )
    assert bench_row["Strategy"] == "Market benchmark (buy-and-hold)"
    assert spy_row["Strategy"] == "S&P 500 (SPY)"
    # SPY +5%: 400 -> 420.
    assert spy_row["total_return"] == pytest.approx(0.05)
    # Universe benchmark is AAA-only here: +10%.
    assert bench_row["total_return"] == pytest.approx(0.10)


def test_compare_benchmark_rows_spy_missing_is_none(tmp_path):
    """No SPY downloaded -> spy row is None, universe row still there."""
    from qrp.ui.helpers import compare_benchmark_rows

    db = tmp_path / "m.db"
    _seed_prices(db, {"AAA": [100.0, 101, 102, 101, 110.0]})
    bench_row, spy_row = compare_benchmark_rows(
        ["AAA"], "2024-01-02", "2024-01-08", 100_000.0, db_path=db
    )
    assert bench_row is not None
    assert spy_row is None


def test_benchmark_equity_reinvests_dividends_total_return():
    """Dividends buy more SPY shares on the ex-date (total-return basis),
    matching a hand computation — not idle cash."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    spy = pd.DataFrame(
        {"SPY": [100.0, 100.0, 100.0, 100.0, 110.0, 110.0]}, index=dates
    )
    divs = pd.DataFrame({"ticker": ["SPY"], "ex_date": [dates[3]], "amount": [2.0]})
    eq = benchmark_equity(spy, 10_000.0, dividends=divs)
    vals = eq["portfolio_value"].to_numpy()
    # 100 shares; the $200 payout on dates[3] buys 2 more shares at $100.
    assert vals[2] == pytest.approx(10_000.0)
    assert vals[3] == pytest.approx(10_200.0)      # 102 shares x $100
    assert vals[4] == pytest.approx(102 * 110.0)   # 11,220 — the shares compound
    assert vals[5] == pytest.approx(11_220.0)
    # Idle cash would have frozen at 10,200 after the ex-date.
    assert vals[4] > 10_200.0


def test_turnover_excludes_first_day_deployment():
    """Day-1's 100%-of-equity initial buying is setup, not ongoing trading:
    it must not count toward annualized turnover."""
    dates = pd.bdate_range("2024-01-02", periods=253)
    equity = pd.DataFrame({"date": dates, "portfolio_value": [100_000.0] * 253})
    trades = pd.DataFrame(
        [
            {"date": dates[0], "shares": 1000.0, "price": 100.0},   # day-1: buy $100k
            {"date": dates[100], "shares": 500.0, "price": 100.0},  # rebalance buy $50k
            {"date": dates[100], "shares": 500.0, "price": 100.0},  # rebalance sell $50k
        ]
    )
    m = compute_metrics(equity, trades)
    # Only the $100k rebalance round-trip counts: gross $100k / 2 = $50k
    # one-way; 50_000 / (100_000 x 1yr) = 0.5. (With day-1 included it
    # would be 1.0.)
    assert m["turnover"]["value"] == pytest.approx(0.5)
    assert "first day" in m["turnover"]["what"].lower()
