"""The universe buy-and-hold benchmark credits dividends (total-return basis).

The "Market benchmark (buy-and-hold)" line must sit on the same total-return
basis as every strategy curve: each held stock's cash distributions are
credited on their ex-dates. Same entitlement rule as the backtest engine and
the SPY benchmark — whoever held the stock overnight into the ex-date gets
paid. With no dividend rows, output must be identical to the old price-only
behavior.
"""

import pandas as pd
import pytest

from qrp.backtest.metrics import buy_and_hold_equity


def _prices():
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAA": [10.0, 10.0, 10.0, 10.0, 10.0],
            "BBB": [20.0, 20.0, 20.0, 20.0, 20.0],
        },
        index=idx,
    )


def _divs(rows):
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "amount"])


def test_dividend_credited_on_ex_date():
    prices = _prices()
    # Day one: $50k into AAA @ $10 -> 5000 shares. A $1 dividend on the
    # third day must add exactly $5,000 of cash from that day onward.
    divs = _divs([("AAA", prices.index[2], 1.0)])
    base = buy_and_hold_equity(prices, 100_000.0)
    with_div = buy_and_hold_equity(prices, 100_000.0, dividends=divs)
    delta = with_div["portfolio_value"] - base["portfolio_value"]
    assert (delta.iloc[:2] == 0.0).all()
    assert (abs(delta.iloc[2:] - 5000.0) < 1e-6).all()
    assert with_div["portfolio_value"].iloc[-1] == pytest.approx(105_000.0)


def test_ex_date_on_first_day_matches_spy_convention():
    # Same convention as benchmark_equity (the SPY fix): an ex-date on the
    # first grid date is credited. Pinned so the two benchmarks never drift
    # apart in entitlement semantics.
    prices = _prices()
    divs = _divs([("BBB", prices.index[0], 2.0)])
    with_div = buy_and_hold_equity(prices, 100_000.0, dividends=divs)
    # 2500 BBB shares x $2 = $5,000 credited from day one.
    assert with_div["portfolio_value"].iloc[0] == pytest.approx(105_000.0)


def test_unheld_or_out_of_range_dividends_ignored():
    prices = _prices()
    divs = _divs(
        [
            # CCC has no day-one price, so it is never held: not entitled.
            ("CCC", prices.index[2], 5.0),
            # Ex-dates outside the price grid are ignored.
            ("AAA", prices.index[0] - pd.Timedelta(days=10), 5.0),
            ("BBB", prices.index[-1] + pd.Timedelta(days=10), 5.0),
            # Non-positive amounts are ignored.
            ("AAA", prices.index[2], 0.0),
        ]
    )
    base = buy_and_hold_equity(prices, 100_000.0)
    with_div = buy_and_hold_equity(prices, 100_000.0, dividends=divs)
    pd.testing.assert_frame_equal(with_div, base)


def test_no_dividends_identical_to_price_only():
    prices = _prices()
    base = buy_and_hold_equity(prices, 100_000.0)
    pd.testing.assert_frame_equal(
        buy_and_hold_equity(prices, 100_000.0, dividends=None), base
    )
    empty = pd.DataFrame(columns=["ticker", "ex_date", "amount"])
    pd.testing.assert_frame_equal(
        buy_and_hold_equity(prices, 100_000.0, dividends=empty), base
    )
