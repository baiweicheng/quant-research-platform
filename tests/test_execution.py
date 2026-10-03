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
from qrp.execution.paper import PaperBroker, resolve_paper_timing
from qrp.execution.rebalance import orders_to_target, priceable_closes
from qrp.risk.guards import RiskConfig
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H

DAY = "2024-01-08"  # a Monday
PREV_DAY = "2024-01-05"  # the trading day before DAY


def trading_days() -> list[str]:
    return pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist()


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


def seed_expensive_bar(path, ticker: str = "EXP", price: float = 3_000.0) -> None:
    """Five trading days of flat prices for an expensive stock."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": 100,
        }
    )
    upsert_bars(df, ticker, path)


def seed_signals(path, spike_on_last: bool = False) -> None:
    """A momentum value per ticker on each trading day.

    With spike_on_last, AAA's value on the final day is a huge spike —
    a stand-in for a signal that "knows" the future (used to prove the
    paper path never sees it).
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = [0.5] * 5
    if spike_on_last:
        aaa[-1] = 99.0
    upsert_signals(pd.Series(aaa, index=dates), "AAA", "momentum", path)
    upsert_signals(pd.Series([0.3] * 5, index=dates), "BBB", "momentum", path)


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


# --- audit fixes: timing, precision, robustness ----------------------------


def test_resolve_paper_timing_defaults_to_latest_and_previous_day(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)

    trading_day, signal_cutoff = resolve_paper_timing(db)

    assert trading_day == DAY
    assert signal_cutoff == PREV_DAY  # strictly before the trading day


def test_resolve_paper_timing_with_asof(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)

    # Signal-cutoff meaning (dashboard): trade at the first price date
    # after the cutoff.
    trading_day, signal_cutoff = resolve_paper_timing(db, "2024-01-03")
    assert (trading_day, signal_cutoff) == ("2024-01-04", "2024-01-03")

    # asof on the latest price date: no later trading day exists, so no
    # honest trade is possible — expect a plain-English error, not a crash.
    with pytest.raises(ValueError, match="next trading day"):
        resolve_paper_timing(db, DAY)


def test_resolve_paper_timing_with_trading_day(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)

    # Trading-day meaning (CLI --date): signals come from the day before.
    trading_day, signal_cutoff = resolve_paper_timing(
        db, DAY, asof_is_trading_day=True
    )
    assert (trading_day, signal_cutoff) == (DAY, PREV_DAY)

    with pytest.raises(ValueError, match="No price data stored"):
        resolve_paper_timing(db, "2024-01-06", asof_is_trading_day=True)


def test_resolve_paper_timing_needs_two_trading_days(tmp_path):
    db = tmp_path / "t.db"
    dates = pd.date_range("2024-01-02", periods=1, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000,
        }
    )
    upsert_bars(df, "AAA", db)

    with pytest.raises(ValueError, match="at least two"):
        resolve_paper_timing(db)


def test_paper_path_never_uses_trading_day_signals(tmp_path):
    """No-lookahead pin for the paper path (MAJOR 2).

    AAA's signal spikes on the trading day itself — as if it "knew" the
    future. The paper timing rule must keep that spike invisible: only
    signals dated strictly before the trading day may influence the trade.
    """
    db = tmp_path / "t.db"
    seed_bars(db)
    seed_signals(db, spike_on_last=True)

    trading_day, signal_cutoff = resolve_paper_timing(db)
    assert trading_day == DAY

    scores, signal_day = H.latest_scores(
        "momentum", ["AAA", "BBB"], signal_cutoff, str(db)
    )
    assert signal_day == PREV_DAY
    # The day-8 spike (99.0) is not used; AAA keeps its honest day-5 score.
    assert scores["AAA"] == pytest.approx(0.5)
    assert scores["BBB"] == pytest.approx(0.3)


def test_compute_rebalance_orders_keeps_full_precision(tmp_path):
    """MAJOR 1 regression: execute the raw orders, not rounded display text.

    A $12 buy of a $3,000 stock is 0.004 shares — above the $1 dust filter,
    so it is a legitimate order, but the preview table rounds it to "0.00".
    Executing from the raw Order must fill it; the old re-parsed path
    turned it into 0.0 shares and rejected it.
    """
    db = tmp_path / "t.db"
    seed_bars(db)
    seed_expensive_bar(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    orders = H.compute_rebalance_orders(
        broker, {"EXP": 0.00012}, DAY, 100_000.0, cost_bps=0.0
    )

    assert len(orders) == 1
    assert orders[0].shares == pytest.approx(0.004)  # full precision kept
    # This is what the preview table shows — and what the old code executed:
    assert float(f"{orders[0].shares:.2f}") == 0.0

    result = broker.place_order(
        orders[0].ticker, orders[0].action, orders[0].shares, DAY
    )
    assert result.filled, result.reason
    assert broker.positions["EXP"] == pytest.approx(0.004)


def test_preview_matches_execute_contract(tmp_path):
    """Preview rows and raw orders describe the same trades."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    orders = H.compute_rebalance_orders(
        broker, {"AAA": 0.5, "BBB": 0.5}, DAY, 100_000.0, cost_bps=0.0
    )
    rows, notes = H.preview_rebalance_orders(
        broker, {"AAA": 0.5, "BBB": 0.5}, DAY, 100_000.0, cost_bps=0.0
    )

    assert len(rows) == len(orders)
    # First note always states the execution convention in plain English.
    assert notes[0].startswith("Prices shown are the day's OPENING prices")
    for row, order in zip(rows, orders):
        assert row["Ticker"] == order.ticker
        assert row["Action"] == order.action
        # Display rounding only ever hides precision; the order is exact.
        assert abs(row["Shares"] - order.shares) < 0.01


def test_fee_shrink_keeps_every_buy_executable(tmp_path):
    """High fees scale buys down instead of producing unfillable orders."""
    db = tmp_path / "t.db"
    seed_bars(db)
    # Relaxed cap: this test is about fee math, not the 20% position cap.
    broker = PaperBroker(
        db, starting_capital=10_000.0, cost_bps=500.0,
        risk_config=RiskConfig(max_position_weight=1.0),
    )

    # 120% target: fees would push the naive orders over available cash.
    orders = orders_to_target(
        broker, {"AAA": 0.6, "BBB": 0.6}, DAY, 10_000.0, cost_bps=500.0
    )
    assert orders, "expected scaled-down buys, not an empty list"

    for order in orders:
        result = broker.place_order(
            order.ticker, order.action, order.shares, DAY
        )
        assert result.filled, result.reason
    assert broker.cash >= -0.01  # never negative, penny tolerance


def test_missing_price_tickers_are_skipped_not_crashed(tmp_path):
    """Delisted/halted tickers: no KeyError tracebacks in the paper path."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    closes, missing = priceable_closes(broker, {"AAA", "DEL"}, DAY)
    assert closes == {"AAA": pytest.approx(100.0)}
    assert missing == ["DEL"]

    # DEL has no bars: it is skipped, AAA is still traded.
    orders = orders_to_target(
        broker, {"AAA": 0.5, "DEL": 0.5}, DAY, 100_000.0, cost_bps=0.0
    )
    assert {o.ticker for o in orders} == {"AAA"}

    # A held position with no price data is left untouched, not crashed on.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO paper_positions (ticker, shares)"
            " VALUES ('GONE', 10)"
        )
    orders = orders_to_target(
        broker, {"AAA": 1.0}, DAY, 100_000.0, cost_bps=0.0
    )
    assert "GONE" not in {o.ticker for o in orders}
    assert broker.positions["GONE"] == pytest.approx(10.0)


# --- Feature 5: plain-English per-trade reasons ------------------------------


def buy_aaa_shares(db, shares: float) -> PaperBroker:
    """A broker holding `shares` of AAA at $100 (cost-free buys).

    Permissive position caps: this helper is about reason strings, while
    the position-cap guard has its own dedicated test above.
    """
    broker = PaperBroker(
        db,
        starting_capital=100_000.0,
        cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0, max_positions=10),
    )
    result = broker.place_order("AAA", "BUY", shares, DAY)
    assert result.filled, result.reason
    return broker


def test_order_reason_new_position_without_rank(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    orders = orders_to_target(broker, {"AAA": 0.5}, DAY, 100_000.0, cost_bps=0.0)

    assert len(orders) == 1
    assert orders[0].action == "BUY"
    assert orders[0].reason == "Bought AAA: new position at 50.0% of the portfolio"


def test_order_reason_new_position_with_rank(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    orders = orders_to_target(
        broker, {"AAA": 0.5}, DAY, 100_000.0, cost_bps=0.0, ranks={"AAA": 1}
    )

    assert len(orders) == 1
    assert orders[0].reason == (
        "Bought AAA: new position at 50.0% of the portfolio, ranked #1"
    )


def test_order_reason_adding_with_and_without_rank(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = buy_aaa_shares(db, 500)  # 50% of a $100k account

    orders = orders_to_target(broker, {"AAA": 0.7}, DAY, 100_000.0, cost_bps=0.0)

    assert len(orders) == 1
    assert orders[0].reason == (
        "Bought more AAA: raised from 50.0% to 70.0% of the portfolio"
    )

    orders = orders_to_target(
        broker, {"AAA": 0.7}, DAY, 100_000.0, cost_bps=0.0, ranks={"AAA": 3}
    )

    assert len(orders) == 1
    assert orders[0].reason == (
        "Bought more AAA: raised from 50.0% to 70.0% of the portfolio, "
        "ranked #3"
    )


def test_order_reason_trimming_with_and_without_rank(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = buy_aaa_shares(db, 500)  # 50% of a $100k account

    orders = orders_to_target(broker, {"AAA": 0.3}, DAY, 100_000.0, cost_bps=0.0)

    assert len(orders) == 1
    assert orders[0].action == "SELL"
    assert orders[0].reason == (
        "Trimmed AAA: reduced from 50.0% to 30.0% of the portfolio"
    )

    orders = orders_to_target(
        broker, {"AAA": 0.3}, DAY, 100_000.0, cost_bps=0.0, ranks={"AAA": 7}
    )

    assert len(orders) == 1
    assert orders[0].reason == (
        "Trimmed AAA: reduced from 50.0% to 30.0% of the portfolio, "
        "ranked #7"
    )


def test_order_reason_full_exit_with_and_without_rank(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = buy_aaa_shares(db, 500)

    orders = orders_to_target(broker, {}, DAY, 100_000.0, cost_bps=0.0)

    assert len(orders) == 1
    assert orders[0].action == "SELL"
    assert orders[0].reason == "Sold AAA: it left the target portfolio"

    orders = orders_to_target(
        broker, {}, DAY, 100_000.0, cost_bps=0.0, ranks={"AAA": 42}
    )

    assert len(orders) == 1
    assert orders[0].reason == (
        "Sold AAA: it left the target portfolio — rank fell to #42"
    )


def test_fee_shrink_keeps_reasons_on_buys(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(
        db, starting_capital=10_000.0, cost_bps=500.0,
        risk_config=RiskConfig(max_position_weight=1.0),
    )

    # 120% target: fees force the shrink path that rebuilds the buys.
    orders = orders_to_target(
        broker, {"AAA": 0.6, "BBB": 0.6}, DAY, 10_000.0, cost_bps=500.0
    )

    assert orders
    assert all(o.reason for o in orders)
    assert all(o.reason.startswith("Bought") for o in orders)


def test_place_order_persists_reason_and_trades_returns_it(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    result = broker.place_order(
        "AAA", "BUY", 10, DAY,
        reason="Bought AAA: new position at 10.0% of the portfolio",
    )
    assert result.filled

    (trade,) = broker.trades()
    assert trade["ticker"] == "AAA"
    assert trade["reason"] == (
        "Bought AAA: new position at 10.0% of the portfolio"
    )


def test_rebalance_reason_round_trips_through_place_order(tmp_path):
    """The engine path: orders_to_target reason -> stored -> trades()."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(
        db,
        starting_capital=100_000.0,
        cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0, max_positions=10),
    )

    orders = orders_to_target(
        broker, {"AAA": 0.5, "BBB": 0.5}, DAY, 100_000.0, cost_bps=0.0,
        ranks={"AAA": 1, "BBB": 2},
    )
    for order in orders:
        result = broker.place_order(
            order.ticker, order.action, order.shares, DAY, reason=order.reason
        )
        assert result.filled, result.reason

    by_ticker = {t["ticker"]: t["reason"] for t in broker.trades(limit=10)}
    assert by_ticker["AAA"] == (
        "Bought AAA: new position at 50.0% of the portfolio, ranked #1"
    )
    assert by_ticker["BBB"] == (
        "Bought BBB: new position at 50.0% of the portfolio, ranked #2"
    )


def test_place_order_default_reason_for_manual_orders(tmp_path):
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    result = broker.place_order("AAA", "BUY", 10, DAY)
    assert result.filled

    (trade,) = broker.trades()
    assert trade["reason"] == "Manual order — no reason was recorded."


def test_migration_adds_reason_column_to_old_db(tmp_path):
    """Databases created before Feature 5 upgrade without losing trades."""
    db = tmp_path / "mig.db"
    seed_bars(db)
    # The old schema: same table, no reason column, with one stored trade.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE paper_trades ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "date TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,"
            "shares REAL NOT NULL, price REAL NOT NULL, cost REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO paper_trades (date, ticker, side, shares, price, cost)"
            " VALUES ('2024-01-08', 'AAA', 'BUY', 10, 100.0, 1.0)"
        )

    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    with sqlite3.connect(db) as conn:
        cols = [
            row[1] for row in conn.execute("PRAGMA table_info(paper_trades)")
        ]
    assert "reason" in cols
    # No data loss: the pre-migration trade survives, with an empty reason.
    old = broker.trades(limit=10)
    assert len(old) == 1
    assert old[0]["ticker"] == "AAA"
    assert old[0]["reason"] == ""

    # The upgraded DB stores reasons on new trades too.
    broker.place_order(
        "BBB", "BUY", 10, DAY,
        reason="Bought BBB: new position at 5.0% of the portfolio",
    )
    trades = broker.trades(limit=10)
    assert trades[0]["ticker"] == "BBB"
    assert trades[0]["reason"] == (
        "Bought BBB: new position at 5.0% of the portfolio"
    )
    assert trades[1]["ticker"] == "AAA"  # the old row, untouched


def seed_gap_bars(path) -> None:
    """AAA opens at $96 but closes at $100 — open != close is observable.

    The 4% gap stays inside the 5% price-collar guard so the test orders
    are not blocked by the safety check."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": 96.0,
            "High": 101.0,
            "Low": 95.0,
            "Close": 100.0,
            "Volume": 1_000,
        }
    )
    upsert_bars(df, "AAA", path)


def test_paper_default_execution_is_open(tmp_path):
    db = tmp_path / "t.db"
    seed_gap_bars(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)
    assert broker.execution == "open"

    result = broker.place_order("AAA", "BUY", 10, DAY)
    assert result.filled
    # Fills at the day's OPEN (96), not the close (100).
    assert result.price == pytest.approx(96.0)
    assert broker.positions == {"AAA": pytest.approx(10.0)}
    # Valuation still marks at the close: cash = 100,000 - 960 (the fill),
    # and the position is worth 10 x $100 = $1,000 at the close.
    assert broker.cash == pytest.approx(100_000.0 - 960.0)


def test_paper_close_execution_fills_at_close(tmp_path):
    db = tmp_path / "t.db"
    seed_gap_bars(db)
    broker = PaperBroker(
        db, starting_capital=100_000.0, cost_bps=0.0, execution="close"
    )

    result = broker.place_order("AAA", "BUY", 10, DAY)
    assert result.filled
    # The old convention: fills at the day's CLOSE (100).
    assert result.price == pytest.approx(100.0)


def test_paper_rejects_unknown_execution(tmp_path):
    db = tmp_path / "t.db"
    seed_gap_bars(db)
    with pytest.raises(ValueError, match="unknown execution"):
        PaperBroker(db, execution="noon")


def test_preview_note_states_execution_convention(tmp_path):
    from qrp.ui import helpers as H

    db = str(tmp_path / "u.db")
    seed_gap_bars(db)
    open_broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)
    _, open_notes = H.preview_rebalance_orders(
        open_broker, {"AAA": 1.0}, DAY, 100_000.0, cost_bps=0.0
    )
    assert any("OPENING prices" in n for n in open_notes)

    close_broker = PaperBroker(
        db, starting_capital=100_000.0, cost_bps=0.0, execution="close"
    )
    _, close_notes = H.preview_rebalance_orders(
        close_broker, {"AAA": 1.0}, DAY, 100_000.0, cost_bps=0.0
    )
    assert any("CLOSING prices" in n for n in close_notes)
