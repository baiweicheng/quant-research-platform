"""Tests for the paper-trading execution engine (Phase 5).

All synthetic — no network, no real money. Each test uses a throwaway
SQLite database so the real project database is never touched.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from qrp.data.store import upsert_bars
from qrp.execution import cli as exec_cli
from qrp.execution.paper import PaperBroker
from qrp.execution.rebalance import orders_to_target
from qrp.risk.guards import RiskConfig
from qrp.signals.store import upsert_signals

DAY = "2024-01-08"  # a Monday


def seed_bars(path) -> None:
    """Five trading days of flat prices for AAA ($100) and BBB ($50)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": base,
                "High": base * 1.01,
                "Low": base * 0.99,
                "Close": base,
                "Volume": 1_000,
            }
        )
        upsert_bars(df, ticker, path)


def seed_signals(path) -> None:
    """One signal value per ticker on the trading day."""
    idx = pd.to_datetime([DAY])
    upsert_signals(pd.Series([0.5], index=idx), "AAA", "momentum", path)
    upsert_signals(pd.Series([0.3], index=idx), "BBB", "momentum", path)


def trade_count(path) -> int:
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()
    return n


def test_buy_reduces_cash_and_creates_position(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=10.0)

    result = broker.place_order("AAA", "BUY", 10, DAY)

    assert result.filled
    assert result.price == pytest.approx(100.0)
    # 10 shares @ $100 = $1,000 plus 10 bps fee = $1.00
    assert result.cost == pytest.approx(1.0)
    assert broker.cash == pytest.approx(100_000.0 - 1_001.0)
    assert broker.positions == {"AAA": pytest.approx(10.0)}
    assert trade_count(db) == 1


def test_order_bigger_than_cash_rejected(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0)

    result = broker.place_order("AAA", "BUY", 100_000, DAY)

    assert not result.filled
    # Plain-English reason, and it must mention the actual problem: cash.
    assert "cash" in result.reason.lower()
    assert broker.cash == pytest.approx(100_000.0)
    assert broker.positions == {}
    assert trade_count(db) == 0


def test_risk_guard_blocks_overweight_position(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    config = RiskConfig(max_position_weight=0.20)
    broker = PaperBroker(db, starting_capital=100_000.0, risk_config=config)

    # $50,000 of AAA would be 50% of the account — over the 20% cap.
    result = broker.place_order("AAA", "BUY", 500, DAY)

    assert not result.filled
    assert "20%" in result.reason
    assert broker.cash == pytest.approx(100_000.0)
    assert broker.positions == {}


def test_sell_more_than_held_rejected_no_shorting(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0)
    broker.place_order("AAA", "BUY", 10, DAY)

    result = broker.place_order("AAA", "SELL", 11, DAY)

    assert not result.filled
    assert "short" in result.reason.lower()
    assert broker.positions["AAA"] == pytest.approx(10.0)


def test_dry_run_changes_nothing(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    seed_signals(db)

    rc = exec_cli.main(
        ["--signal", "momentum", "--method", "equal", "--date", DAY,
         "--db", str(db), "--dry-run"]
    )

    assert rc == 0
    broker = PaperBroker(db)
    assert broker.cash == pytest.approx(100_000.0)
    assert broker.positions == {}
    assert trade_count(db) == 0


def test_full_rebalance_to_two_stock_target(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    # Permissive caps here: this test checks rebalance math, while the
    # position-cap guard has its own dedicated test above.
    broker = PaperBroker(
        db,
        starting_capital=100_000.0,
        cost_bps=10.0,
        risk_config=RiskConfig(max_position_weight=1.0, max_positions=10),
    )

    orders = orders_to_target(
        broker, {"AAA": 0.5, "BBB": 0.5}, DAY, 100_000.0, cost_bps=10.0
    )
    # Sells (none here) come before buys.
    assert all(o.action == "BUY" for o in orders)
    for order in orders:
        result = broker.place_order(order.ticker, order.action, order.shares, DAY)
        assert result.filled, result.reason

    positions = broker.positions
    # Expected math, mirroring orders_to_target: positions are sized off
    # capital net of expected fees (10 bps of $100k = $100 -> $99,900
    # effective), and the $99,999.90 fee-inclusive total fits in $100k
    # cash, so no further scaling kicks in.
    fee_rate = 10.0 / 10_000.0
    raw_notional = 0.5 * 100_000.0 * (1 - fee_rate)  # per-stock target dollars
    assert positions["AAA"] == pytest.approx(raw_notional / 100.0)  # $100/share
    assert positions["BBB"] == pytest.approx(raw_notional / 50.0)    # $50/share
    assert broker.cash == pytest.approx(
        100_000.0 - 2 * raw_notional * (1 + fee_rate), abs=0.01
    )
    assert 0 <= broker.cash < 200  # only dust left over
    # Penny tolerance: float rounding can leave -1e-11 instead of exactly 0.
    assert broker.cash >= -0.01

    # Rebalance again toward AAA only: BBB is sold off completely.
    total = (
        broker.cash + positions["AAA"] * 100.0 + positions["BBB"] * 50.0
    )
    orders2 = orders_to_target(broker, {"AAA": 1.0}, DAY, total, cost_bps=10.0)
    actions = {(o.ticker, o.action) for o in orders2}
    assert ("BBB", "SELL") in actions
    # Sells come before buys in the order list.
    seen_buy = False
    for o in orders2:
        if o.action == "BUY":
            seen_buy = True
        assert not (seen_buy and o.action == "SELL")
    for order in orders2:
        result = broker.place_order(order.ticker, order.action, order.shares, DAY)
        assert result.filled, result.reason
    assert "BBB" not in broker.positions
    assert broker.cash >= -0.01  # penny tolerance for float rounding
