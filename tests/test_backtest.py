"""Tests for the long-only backtesting engine (Phase 3).

All tests use SYNTHETIC data (no network). They pin down the engine's most
important correctness properties:
  1. A perfect-foresight signal makes money (the engine can capture an edge).
  2. Higher trading costs never help (costs are actually charged).
  3. NO LOOKAHEAD: a signal that "knows" about a price jump dated T can only
     trade at T+1 prices — the jump itself is never captured.
  4. Long-only invariant: target weights are never negative and never sum to
     more than 1 on any rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.engine import rebalance_dates, run_backtest
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import top_n_by_signal

INITIAL = 100_000.0


def _trend_setup(periods: int = 120):
    """A climbs steadily, B is flat, C falls steadily; foresight signal."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    n = len(dates)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.005 ** np.arange(n)),
            "B": np.full(n, 100.0),
            "C": 100.0 * (0.995 ** np.arange(n)),
        },
        index=dates,
    )
    # "Perfect foresight": the signal on date d is the 20-day forward return.
    # (Synthetic only — a real signal could never know this.)
    foresight = prices.shift(-20) / prices - 1.0
    return prices, foresight


def test_perfect_foresight_is_profitable():
    prices, foresight = _trend_setup()
    res = run_backtest(
        prices,
        foresight,
        top_n_by_signal("foresight", 1),
        frequency="monthly",
        cost_bps=0.0,
        initial_capital=INITIAL,
    )
    assert res.equity["portfolio_value"].notna().all()
    assert res.equity["portfolio_value"].iloc[-1] > INITIAL


def test_higher_costs_reduce_return():
    dates = pd.bdate_range("2024-01-02", periods=120)
    n = len(dates)
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.002 ** np.arange(n)),
            "B": 100.0 * (1.001 ** np.arange(n)),
        },
        index=dates,
    )
    # Alternating signal forces real turnover every month.
    parity = (dates.month % 2).to_numpy().astype(float)
    signals = pd.DataFrame({"A": parity, "B": 1.0 - parity}, index=dates)

    cheap = run_backtest(
        prices, signals, top_n_by_signal("alt", 1),
        frequency="monthly", cost_bps=0.0, initial_capital=INITIAL,
    )
    pricey = run_backtest(
        prices, signals, top_n_by_signal("alt", 1),
        frequency="monthly", cost_bps=500.0, initial_capital=INITIAL,
    )
    assert len(cheap.trades) > 2  # sanity: turnover actually happened
    assert (
        cheap.equity["portfolio_value"].iloc[-1]
        > pricey.equity["portfolio_value"].iloc[-1]
    )


def test_no_lookahead_on_jump():
    """A signal dated T that 'knows' about a jump on T must trade at T+1."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    jump_day = pd.Timestamp("2024-02-01")  # a month-start => a rebalance date
    assert jump_day in dates
    x = np.where(dates >= jump_day, 200.0, 100.0)
    prices = pd.DataFrame({"X": x, "Y": np.full(len(dates), 100.0)}, index=dates)

    # The signal "knows" the jump, but only with the jump day's date on it.
    sig_x = pd.Series(np.nan, index=dates)
    sig_x.loc[jump_day] = 1.0
    signals = pd.DataFrame({"X": sig_x, "Y": np.nan}, index=dates)

    res = run_backtest(
        prices,
        signals,
        top_n_by_signal("jump", 1),
        frequency="monthly",
        cost_bps=0.0,
        initial_capital=INITIAL,
    )
    buys = res.trades[(res.trades["ticker"] == "X") & (res.trades["action"] == "BUY")]
    assert not buys.empty, "expected the strategy to buy X after the jump signal"
    # The trade happens strictly after the jump date, at the post-jump price.
    assert (buys["date"] > jump_day).all()
    assert np.allclose(buys["price"].to_numpy(), 200.0)
    # The jump itself (100 -> 200) was NOT captured: bought at 200, still 200.
    assert res.equity["portfolio_value"].iloc[-1] == pytest.approx(INITIAL)


def test_long_only_invariant():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=120)
    tickers = ["A", "B", "C", "D"]
    data = {}
    for t in tickers:
        rets = rng.normal(0.0005, 0.02, len(dates))
        data[t] = 100.0 * np.cumprod(1.0 + rets)
    prices = pd.DataFrame(data, index=dates)
    # Point-in-time signal by construction (uses only past prices).
    signals = pd.DataFrame(
        {t: prices[t].pct_change(20).fillna(0.0) for t in tickers}, index=dates
    )

    res = run_backtest(
        prices,
        signals,
        top_n_by_signal("s", 2),
        frequency="monthly",
        cost_bps=10.0,
        initial_capital=INITIAL,
    )
    assert res.target_weights, "expected at least one rebalance"
    for day, weights in res.target_weights.items():
        assert all(w >= 0 for w in weights.values()), f"negative weight on {day}"
        assert sum(weights.values()) <= 1.0 + 1e-9, f"weights sum > 1 on {day}"
    assert (res.equity["portfolio_value"] >= 0).all()
    assert res.equity["portfolio_value"].notna().all()


def test_rebalance_date_schedules():
    dates = pd.bdate_range("2024-01-02", periods=90)
    monthly = rebalance_dates(dates, "monthly")
    weekly = rebalance_dates(dates, "weekly")
    # First trading day of each calendar month present in the data.
    assert monthly[0] == pd.Timestamp("2024-01-02")
    assert monthly[1] == pd.Timestamp("2024-02-01")
    assert monthly[2] == pd.Timestamp("2024-03-01")
    assert len(weekly) > len(monthly)
    with pytest.raises(ValueError):
        rebalance_dates(dates, "yearly")


def test_metrics_explanations_present():
    prices, foresight = _trend_setup()
    res = run_backtest(
        prices,
        foresight,
        top_n_by_signal("foresight", 1),
        frequency="monthly",
        cost_bps=10.0,
        initial_capital=INITIAL,
    )
    metrics = compute_metrics(res.equity, res.trades)
    expected = {
        "final_value", "total_return", "cagr", "volatility", "sharpe",
        "max_drawdown", "turnover", "win_rate", "num_trades",
    }
    assert expected <= set(metrics.keys())
    for key, m in metrics.items():
        # Every metric carries a plain-English explanation for non-experts.
        assert {"value", "label", "what", "why", "good"} <= set(m.keys()), key
        assert m["label"] and m["what"] and m["why"] and m["good"], key
    assert metrics["final_value"]["value"] > 0
