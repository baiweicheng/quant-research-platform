"""Tests for the paper-trading pending-orders (nightly) workflow.

All synthetic — no network, no real money. Each test uses a throwaway
SQLite database so the real project database is never touched.

The nightly workflow: stage_orders() records decisions made from data
through the latest close WITHOUT trading; fill_pending_orders() fills
them at the next market open once new price data arrives (idempotent —
a second call never double-fills).
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from qrp.backtest.costs import SpreadCostModel
from qrp.data.store import upsert_bars, upsert_dividends
from qrp.data.yahoo import normalize_ticker
from qrp.execution.paper import (
    PaperBroker,
    resolve_paper_timing,
    resolve_staging_timing,
)
from qrp.execution.rebalance import orders_to_target
from qrp.risk.guards import Order
from qrp.signals.store import load_bars
from qrp.ui import helpers as H


def trading_days(periods: int = 7) -> list[str]:
    return (
        pd.date_range("2024-01-02", periods=periods, freq="B")
        .strftime("%Y-%m-%d")
        .tolist()
    )


def seed_bars(
    path,
    ticker: str = "AAA",
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: int = 100_000,
    periods: int = 7,
    wide_on_last: bool = False,
) -> list[str]:
    """Seed `periods` trading days of bars; returns the date list."""
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    highs = [high] * periods
    lows = [low] * periods
    if wide_on_last:
        # A wild final day: proves spread costs are priced off the day
        # BEFORE the fill, not the fill day itself.
        highs[-1] = close * 1.10
        lows[-1] = close * 0.90
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": [open_] * periods,
            "High": highs,
            "Low": lows,
            "Close": [close] * periods,
            "Volume": [volume] * periods,
        }
    )
    upsert_bars(df, ticker, path)
    return dates.strftime("%Y-%m-%d").tolist()


def trade_count(path) -> int:
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()
    return n


def pending_count(path) -> int:
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM paper_pending_orders"
        ).fetchone()
    return n


def test_staging_creates_pending_orders_without_touching_account(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    seed_bars(db, ticker="BBB", open_=50.0, high=50.5, low=49.5, close=50.0)
    broker = PaperBroker(db, cost_model="flat", cost_bps=10.0)
    cash_before = broker.cash

    orders = [
        Order(ticker="AAA", action="BUY", shares=10.123456,
              price=100.0, reason="Bought AAA: new position"),
        Order(ticker="BBB", action="SELL", shares=5.0,
              price=50.0, reason="Trimmed BBB"),
    ]
    result = broker.stage_orders(orders, staged_date="2024-01-08")

    assert result == {"staged": 2, "skipped_duplicates": 0}
    pendings = broker.pending_orders()
    assert len(pendings) == 2
    assert pendings[0]["ticker"] == "AAA"
    assert pendings[0]["side"] == "BUY"
    # Full precision survives — staging never rounds share counts.
    assert pendings[0]["shares"] == pytest.approx(10.123456)
    assert pendings[0]["staged_date"] == "2024-01-08"
    assert pendings[0]["reason"] == "Bought AAA: new position"
    # Staging trades nothing.
    assert broker.cash == pytest.approx(cash_before)
    assert broker.positions == {}
    assert trade_count(db) == 0


def test_duplicate_staging_is_skipped(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, cost_model="flat")

    orders = [
        Order(ticker="AAA", action="BUY", shares=10.0, price=100.0,
              reason="x"),
    ]
    assert broker.stage_orders(orders, "2024-01-08")["staged"] == 1
    again = broker.stage_orders(orders, "2024-01-08")
    assert again == {"staged": 0, "skipped_duplicates": 1}
    assert pending_count(db) == 1  # double-clicking can't double-trade


def test_stage_orders_validates_input(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, cost_model="flat")
    with pytest.raises(ValueError):
        broker.stage_orders([], "2024-01-08")
    with pytest.raises(ValueError):
        broker.stage_orders(
            [Order(ticker="AAA", action="HOLD", shares=1.0, price=1.0)],
            "2024-01-08",
        )


def test_fill_waits_when_no_new_data_exists(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    broker = PaperBroker(db, cost_model="flat")

    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=10.0, price=100.0,
               reason="x")],
        staged_date=days[-1],  # latest stored day: no "next open" yet
    )
    report = broker.fill_pending_orders()

    assert report["filled"] == []
    assert report["skipped"] == []
    assert report["pending_remaining"] == 1  # still waiting, untouched
    assert trade_count(db) == 0
    assert broker.positions == {}


def test_fill_happens_at_next_open_exactly_once(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db, open_=100.0, close=100.0)
    # The fill day opens higher: proves the fill uses the OPEN price.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE daily_bars SET open = 102.0 WHERE ticker = 'AAA'"
            f" AND date = '{days[-1]}'"
        )
    broker = PaperBroker(db, cost_model="flat", cost_bps=10.0)

    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=10.0, price=100.0,
               reason="Bought AAA: new position")],
        staged_date=days[-2],
    )
    report = broker.fill_pending_orders()

    assert len(report["filled"]) == 1
    fill = report["filled"][0]
    assert fill["date"] == days[-1]
    assert fill["ticker"] == "AAA"
    assert fill["price"] == pytest.approx(102.0)  # the day's OPEN
    assert fill["cost"] == pytest.approx(10 * 102.0 * 0.001)
    assert report["pending_remaining"] == 0
    assert trade_count(db) == 1
    assert broker.positions == {"AAA": pytest.approx(10.0)}
    assert broker.cash == pytest.approx(100_000.0 - 1020.0 - 1.02)

    # A second fill pass is a no-op — never double-fills.
    report2 = broker.fill_pending_orders()
    assert report2["filled"] == []
    assert trade_count(db) == 1


def test_sells_fill_before_buys(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    seed_bars(db, ticker="BBB", open_=50.0, high=50.5, low=49.5, close=50.0)
    broker = PaperBroker(db, cost_model="flat", cost_bps=0.0)
    broker.place_order("BBB", "BUY", 100.0, days[-3])  # hold BBB first

    broker.stage_orders(
        [
            Order(ticker="AAA", action="BUY", shares=10.0, price=100.0,
                  reason="buy"),
            Order(ticker="BBB", action="SELL", shares=100.0, price=50.0,
                  reason="sell"),
        ],
        staged_date=days[-2],
    )
    report = broker.fill_pending_orders()
    assert len(report["filled"]) == 2
    assert report["filled"][0]["side"] == "SELL"
    assert report["filled"][1]["side"] == "BUY"


def test_cancel_pending_order(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    broker = PaperBroker(db, cost_model="flat")

    broker.stage_orders(
        [
            Order(ticker="AAA", action="BUY", shares=10.0, price=100.0,
                  reason="keep"),
            Order(ticker="AAA", action="SELL", shares=1.0, price=100.0,
                  reason="drop"),
        ],
        staged_date=days[-2],
    )
    pendings = broker.pending_orders()
    assert broker.cancel_pending_order(pendings[1]["id"]) is True
    assert broker.cancel_pending_order(999_999) is False
    assert pending_count(db) == 1

    report = broker.fill_pending_orders()
    assert len(report["filled"]) == 1
    assert report["filled"][0]["side"] == "BUY"


def test_unfillable_order_is_reported_not_retried(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    broker = PaperBroker(db, starting_capital=1_000.0, cost_model="flat")

    # Far more shares than the cash can cover — the fill must fail cleanly.
    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=1_000_000.0, price=100.0,
               reason="too big")],
        staged_date=days[-2],
    )
    report = broker.fill_pending_orders()

    assert report["filled"] == []
    assert len(report["skipped"]) == 1
    assert "cash" in report["skipped"][0]["reason"].lower()
    assert report["pending_remaining"] == 0  # dropped, not retried forever
    assert trade_count(db) == 0
    # Second pass stays quiet.
    assert broker.fill_pending_orders()["skipped"] == []


def test_ex_date_buyer_gets_no_dividend_but_overnight_holder_does(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    ex_date = days[-1]
    upsert_dividends(
        pd.DataFrame({"ex_date": [ex_date], "amount": [1.0]}), "AAA", db
    )
    broker = PaperBroker(db, cost_model="flat", cost_bps=0.0)
    # Overnight holder: bought days ago.
    broker.place_order("AAA", "BUY", 10.0, days[-4])

    # Pending buyer staged the day before the ex-date; fills ON the ex-date.
    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=5.0, price=100.0,
               reason="buy")],
        staged_date=days[-2],
    )
    report = broker.fill_pending_orders()
    assert len(report["filled"]) == 1
    assert report["filled"][0]["date"] == ex_date

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT shares_held, amount_paid FROM paper_dividends"
            " WHERE ticker = 'AAA' AND ex_date = ?",
            (ex_date,),
        ).fetchone()
    # Only the 10 overnight shares are paid — the 5 bought at the open ON
    # the ex-date are not entitled, like a real brokerage.
    assert row is not None
    assert row[0] == pytest.approx(10.0)
    assert row[1] == pytest.approx(10.0)


def test_replay_path_still_fills_immediately(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, cost_model="flat", cost_bps=10.0)

    # The asof/CLI replay path: an explicit past trading day fills at once.
    trading_day, signal_cutoff = resolve_paper_timing(
        db, "2024-01-05", asof_is_trading_day=True
    )
    assert trading_day == "2024-01-05"
    assert signal_cutoff == "2024-01-04"
    result = broker.place_order("AAA", "BUY", 10.0, trading_day)

    assert result.filled
    assert result.price == pytest.approx(100.0)  # that day's open
    assert pending_count(db) == 0  # replay never stages
    with sqlite3.connect(db) as conn:
        (d,) = conn.execute(
            "SELECT date FROM paper_trades"
        ).fetchone()
    assert d == "2024-01-05"


def test_resolve_staging_timing_uses_latest_close(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db)
    decision_date, signal_cutoff = resolve_staging_timing(db)
    assert decision_date == days[-1]
    assert signal_cutoff == days[-1]  # signals through the latest close

    with pytest.raises(ValueError, match="No price data"):
        resolve_staging_timing(tmp_path / "empty.db")


def test_reset_clears_pending_orders(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, cost_model="flat")
    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=1.0, price=100.0,
               reason="x")],
        "2024-01-08",
    )
    assert pending_count(db) == 1
    broker.reset()
    assert pending_count(db) == 0


def test_spread_cost_is_priced_off_the_day_before_the_fill(tmp_path):
    db = tmp_path / "t.db"
    days = seed_bars(db, wide_on_last=True)
    fill_day = days[-1]
    prior_day = days[-2]
    broker = PaperBroker(db, cost_model="spread")

    broker.stage_orders(
        [Order(ticker="AAA", action="BUY", shares=100.0, price=100.0,
               reason="x")],
        staged_date=prior_day,
    )
    report = broker.fill_pending_orders()
    assert len(report["filled"]) == 1
    notional = 100.0 * 100.0  # filled at the $100 open

    bars = load_bars("AAA", path=db)
    model = SpreadCostModel({"AAA": bars})
    expected_prior = notional * model.rate(
        "AAA", pd.Timestamp(prior_day), notional
    )
    priced_off_fill_day = notional * model.rate(
        "AAA", pd.Timestamp(fill_day), notional
    )
    # The wild fill day really does change the estimate — so this test can
    # tell the two pricings apart.
    assert priced_off_fill_day != pytest.approx(expected_prior)
    assert report["filled"][0]["cost"] == pytest.approx(expected_prior)


def test_spread_cost_falls_back_without_history(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db, periods=1)  # a single bar: no history to estimate from
    broker = PaperBroker(db, cost_model="spread")
    result = broker.place_order("AAA", "BUY", 10.0, "2024-01-02")
    assert result.filled
    assert result.cost == pytest.approx(
        10.0 * 100.0 * SpreadCostModel.FALLBACK_RATE
    )


def test_flat_cost_model_still_available(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, cost_model="flat", cost_bps=10.0)
    result = broker.place_order("AAA", "BUY", 10.0, "2024-01-08")
    assert result.filled
    assert result.cost == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown cost_model"):
        PaperBroker(db, cost_model="nonsense")


def test_orders_to_target_close_pricing(tmp_path):
    db = tmp_path / "t.db"
    # Open $100, close $110: the two pricings must size differently.
    seed_bars(db, open_=100.0, close=110.0)
    broker = PaperBroker(db, cost_model="flat", cost_bps=0.0)

    by_execution = orders_to_target(
        broker, {"AAA": 1.0}, "2024-01-08", 11_000.0, cost_bps=0.0
    )
    by_close = orders_to_target(
        broker, {"AAA": 1.0}, "2024-01-08", 11_000.0, cost_bps=0.0,
        price_source="close",
    )
    assert len(by_execution) == len(by_close) == 1
    assert by_execution[0].shares == pytest.approx(110.0)  # 11000 / 100
    assert by_execution[0].price == pytest.approx(100.0)
    assert by_close[0].shares == pytest.approx(100.0)  # 11000 / 110
    assert by_close[0].price == pytest.approx(110.0)

    with pytest.raises(ValueError, match="unknown price_source"):
        orders_to_target(
            broker, {"AAA": 1.0}, "2024-01-08", 11_000.0,
            price_source="nonsense",
        )


def test_ticker_normalization_in_pending_flow(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db, ticker="BRK-B")
    broker = PaperBroker(db, cost_model="flat")
    broker.stage_orders(
        [Order(ticker="brk.b", action="buy", shares=2.0, price=100.0,
               reason="x")],
        staged_date="2024-01-09",
    )
    pendings = broker.pending_orders()
    assert pendings[0]["ticker"] == "BRK-B"
    assert pendings[0]["side"] == "BUY"
    report = broker.fill_pending_orders()
    assert len(report["filled"]) == 1
    assert broker.positions == {"BRK-B": pytest.approx(2.0)}


def test_preview_matches_what_staging_will_stage(tmp_path):
    """The preview shows exactly what "Stage orders" will stage.

    The dashboard preview (staging mode: signals through the latest close,
    orders sized off the close) and the staged orders must describe the
    same trades — otherwise the dry run lies about the real run.
    """
    db = tmp_path / "t.db"
    days = seed_bars(db)
    seed_bars(db, ticker="BBB", open_=50.0, high=50.5, low=49.5, close=50.0)
    broker = PaperBroker(db, cost_model="flat", cost_bps=10.0)

    decision_date, _ = resolve_staging_timing(db)
    weights = {"AAA": 0.5, "BBB": 0.5}
    orders = H.compute_rebalance_orders(
        broker, weights, decision_date, 100_000.0, cost_bps=10.0,
        price_source="close",
    )
    rows, notes = H.preview_rebalance_orders(
        broker, weights, decision_date, 100_000.0, cost_bps=10.0,
        price_source="close",
    )
    assert len(rows) == len(orders) == 2
    assert notes[0].startswith(
        "Prices shown are the day's CLOSING prices, used to size"
    )
    staged = broker.stage_orders(orders, staged_date=decision_date)
    assert staged["staged"] == 2
    for pending, order, row in zip(broker.pending_orders(), orders, rows):
        assert pending["ticker"] == order.ticker == row["Ticker"]
        assert pending["side"] == order.action == row["Action"]
        assert pending["shares"] == pytest.approx(order.shares)
        assert abs(row["Shares"] - order.shares) < 0.01
