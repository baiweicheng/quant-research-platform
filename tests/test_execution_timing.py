"""Tests for the trade-execution setting: next-day OPEN vs next-day CLOSE.

All tests use SYNTHETIC data (no network). They pin down the behavior of the
``execution`` setting on ``run_backtest`` (default "open", option "close"):

  1. With "open" execution, trades fill at T+1's OPENING price — not the close.
  2. The default (no ``execution`` given) is "open".
  3. "close" reproduces the old behavior exactly (the ``opens`` matrix is inert).
  4. NO LOOKAHEAD: a signal dated T cannot see T+1's open — it only affects
     the trade that fills on T+2.
  5. Dividend ex-date entitlement under open execution: whoever holds the
     stock overnight BEFORE the ex-date gets paid; buying at the open ON the
     ex-date is not entitled.
  6. A new position's first-day return is open-to-close of its fill day.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qrp.backtest.engine import run_backtest
from qrp.backtest.strategies import top_n_by_signal

INITIAL = 100_000.0


def _open_setup(periods: int = 5):
    """A's closes are flat at 100 but its OPEN on day index 1 gaps up to 110.

    Synthetic on purpose: opens and closes differ so the fill convention is
    observable. B is a flat $50 placeholder.
    """
    dates = pd.bdate_range("2024-01-02", periods=periods)
    closes = pd.DataFrame(
        {"A": [100.0] * periods, "B": [50.0] * periods},
        index=dates,
    )
    opens = pd.DataFrame(
        {"A": [99.0, 110.0] + [99.0] * (periods - 2), "B": [50.0] * periods},
        index=dates,
    )
    # A always wins the signal, every day.
    signals = pd.DataFrame(
        {"A": [1.0] * periods, "B": [0.0] * periods},
        index=dates,
    )
    return dates, closes, opens, signals


def _run(strategy, **kwargs):
    """Run the synthetic fixture with sensible defaults; kwargs override."""
    dates, closes, opens, signals = _open_setup(kwargs.pop("periods", 5))
    kwargs.setdefault("execution", "open")
    kwargs.setdefault("opens", opens)
    return (
        run_backtest(
            closes,
            signals,
            strategy,
            frequency="daily",
            cost_bps=0.0,
            initial_capital=INITIAL,
            **kwargs,
        ),
        dates,
        closes,
        opens,
    )


def test_trades_fill_at_next_day_open():
    # First rebalance is decided on Jan 2 -> fills on Jan 3.
    res, dates, closes, opens = _run(top_n_by_signal("s", 1))
    first = res.trades.iloc[0]
    assert first["date"] == pd.Timestamp("2024-01-03")
    assert first["action"] == "BUY"
    # Fills at Jan 3's OPEN (110), not Jan 3's close (100).
    assert first["price"] == pytest.approx(110.0)
    assert first["shares"] == pytest.approx(INITIAL / 110.0)


def test_default_execution_is_open():
    dates, closes, opens, signals = _open_setup()
    res = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        opens=opens,  # no execution= given
    )
    assert res.execution == "open"
    first = res.trades.iloc[0]
    assert first["date"] == pd.Timestamp("2024-01-03")
    assert first["price"] == pytest.approx(110.0)


def test_close_execution_reproduces_old_behavior():
    dates, closes, opens, signals = _open_setup()
    # Passing an opens matrix must change nothing under "close".
    res_with = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        execution="close",
        opens=opens,
    )
    res_without = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        execution="close",
    )
    assert res_with.execution == "close"
    pd.testing.assert_frame_equal(res_with.trades, res_without.trades)
    pd.testing.assert_frame_equal(res_with.equity, res_without.equity)
    # And fills are at the close: Jan 3 close is 100, not the open 110.
    first = res_with.trades.iloc[0]
    assert first["date"] == pd.Timestamp("2024-01-03")
    assert first["price"] == pytest.approx(100.0)
    assert first["shares"] == pytest.approx(INITIAL / 100.0)


def test_no_lookahead_signal_cannot_see_next_day_open():
    # "Foresight" signal: A is known (on Jan 3 itself) to be the winner.
    # Under the timing rule it may only drive the trade that FILLS on Jan 4
    # — never the Jan 3 fill, whose decision was made on Jan 2.
    dates, closes, opens, _ = _open_setup()
    signals = pd.DataFrame(
        {"A": [0.0, 1.0, 1.0, 1.0, 1.0], "B": [1.0, 0.0, 0.0, 0.0, 0.0]},
        index=dates,
    )
    res = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        execution="open",
        opens=opens,
    )
    a_buys = res.trades[(res.trades["ticker"] == "A") & (res.trades["action"] == "BUY")]
    # The decision on Jan 3 (signal A=1) fills on Jan 4 — never on Jan 3.
    assert not a_buys.empty
    assert a_buys["date"].min() == pd.Timestamp("2024-01-04")
    # And the Jan 3 fill (decided Jan 2) went to B, not A.
    jan3 = res.trades[res.trades["date"] == pd.Timestamp("2024-01-03")]
    assert set(jan3["ticker"]) == {"B"}


def test_dividend_entitlement_held_overnight_open_execution():
    # Position opened at Jan 3's open, held overnight into the Jan 4
    # ex-date -> entitled to the dividend.
    dates, closes, opens, signals = _open_setup()
    dividends = pd.DataFrame(
        {"ticker": ["A"], "ex_date": [dates[2]], "amount": [1.0]}
    )
    res = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        execution="open",
        opens=opens,
        dividends=dividends,
    )
    assert len(res.dividends) == 1
    row = res.dividends.iloc[0]
    assert row["date"] == dates[2]
    assert row["ticker"] == "A"
    shares_held = INITIAL / 110.0  # bought at Jan 3's open
    assert row["shares_held"] == pytest.approx(shares_held)
    assert row["amount_paid"] == pytest.approx(shares_held * 1.0)


def test_dividend_not_entitled_when_bought_at_open_on_ex_date():
    # A only becomes the winner on Jan 4's signal, so the buy is decided
    # Jan 4 and fills at Jan 5's OPEN — the ex-date itself. The buyer on
    # the ex-date is not entitled to the dividend.
    dates, closes, opens, _ = _open_setup(periods=6)
    signals = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]},
        index=dates,
    )
    dividends = pd.DataFrame(
        {"ticker": ["A"], "ex_date": [dates[4]], "amount": [1.0]}
    )
    res = run_backtest(
        closes,
        signals,
        top_n_by_signal("s", 1),
        frequency="daily",
        cost_bps=0.0,
        initial_capital=INITIAL,
        execution="open",
        opens=opens,
        dividends=dividends,
    )
    a_buy = res.trades[(res.trades["ticker"] == "A") & (res.trades["action"] == "BUY")]
    assert not a_buy.empty
    assert a_buy["date"].min() == dates[4]  # filled at the ex-date's open
    assert res.dividends.empty  # ...and NOT entitled


def test_first_day_return_is_open_to_close():
    # A is bought at Jan 3's open (110) and held; the equity that evening
    # is marked at Jan 3's close (100). The first-day loss is the
    # overnight gap the strategy could not have known about.
    res, dates, closes, opens = _run(top_n_by_signal("s", 1))
    jan3_equity = res.equity.loc[
        res.equity["date"] == pd.Timestamp("2024-01-03"), "portfolio_value"
    ].iloc[0]
    expected = (INITIAL / 110.0) * 100.0  # shares bought x Jan 3 close
    assert jan3_equity == pytest.approx(expected)
    assert jan3_equity < INITIAL  # the gap-up open was a loss vs the close mark
