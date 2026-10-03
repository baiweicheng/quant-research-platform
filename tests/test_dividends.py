"""Tests for Phase 1b: dividend-adjusted total returns.

All tests use SYNTHETIC data (no network). Covered:
- fetch_dividends parses Yahoo's events.dividends payload (mocked session).
- upsert/load round-trip, backfill idempotency, old-DB migration.
- Backtest engine: exact ex-div cash credit; ex-date buyers get nothing;
  no dividends -> behavior unchanged.
- Paper broker: trade-replay share counts (bought before / on / sold
  before ex-date), credit-once semantics.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from qrp.backtest.engine import run_backtest
from qrp.data.store import init_db, load_dividends, upsert_bars, upsert_dividends
from qrp.data.yahoo import fetch_dividends


# --- fetch_dividends (mocked Yahoo session) ---------------------------------

class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, params=None, timeout=None):
        return _FakeResp(self._payload)


def _chart_payload(dividends: dict | None):
    """Minimal v8 chart payload with (optional) dividend events."""
    result = {
        "timestamp": [1704067200, 1704153600],
        "indicators": {
            "quote": [
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [1000, 2000],
                }
            ]
        },
    }
    if dividends is not None:
        result["events"] = {"dividends": dividends}
    return {"chart": {"result": [result]}}


def test_fetch_dividends_parses_events():
    payload = _chart_payload(
        {
            "1704153600": {"amount": 0.50, "date": 1704153600},
            "1704240000": {"amount": 0.75, "date": 1704240000},
        }
    )
    df = fetch_dividends("AAA", "2024-01-01", "2024-02-01",
                         session=_FakeSession(payload))
    assert list(df.columns) == ["ex_date", "amount"]
    assert len(df) == 2
    assert df["amount"].tolist() == [0.50, 0.75]
    assert df["ex_date"].iloc[0] <= df["ex_date"].iloc[-1]  # oldest first


def test_fetch_dividends_empty_when_none_paid():
    df = fetch_dividends("AAA", "2024-01-01", "2024-02-01",
                         session=_FakeSession(_chart_payload(None)))
    assert list(df.columns) == ["ex_date", "amount"]
    assert df.empty


# --- store round-trip / idempotency / migration ------------------------------

def _div_df(rows):
    return pd.DataFrame(rows, columns=["ex_date", "amount"])


def test_dividend_upsert_load_roundtrip(tmp_path):
    db = tmp_path / "d.db"
    init_db(db)
    n = upsert_dividends(
        _div_df([("2024-01-10", 0.50), ("2024-04-10", 0.55)]), "AAA", db
    )
    assert n == 2
    df = load_dividends(["AAA"], path=db)
    assert len(df) == 2
    assert df["amount"].tolist() == [0.50, 0.55]
    assert str(df["ex_date"].iloc[0].date()) == "2024-01-10"


def test_dividend_backfill_is_idempotent(tmp_path):
    """Re-running a download rewrites stored dividends, never duplicates."""
    db = tmp_path / "d.db"
    init_db(db)
    df = _div_df([("2024-01-10", 0.50), ("2024-04-10", 0.55)])
    upsert_dividends(df, "AAA", db)
    upsert_dividends(df, "AAA", db)  # same download again
    assert len(load_dividends(["AAA"], path=db)) == 2
    # Earlier start: only the newly seen ex-date is added.
    upsert_dividends(_div_df([("2023-10-10", 0.45)]), "AAA", db)
    got = load_dividends(["AAA"], path=db)
    assert len(got) == 3
    assert got["ex_date"].min().strftime("%Y-%m-%d") == "2023-10-10"


def test_dividend_migration_on_old_db(tmp_path):
    """A database created before the dividends table existed gains it."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE daily_bars (ticker TEXT, date TEXT, open REAL,"
            " high REAL, low REAL, close REAL, volume INTEGER,"
            " PRIMARY KEY (ticker, date))"
        )
    # No dividends table yet: reads as "no dividends", never an error.
    assert load_dividends(["AAA"], path=db).empty
    init_db(db)  # the migration: CREATE TABLE IF NOT EXISTS
    assert upsert_dividends(_div_df([("2024-01-10", 0.50)]), "AAA", db) == 1
    assert len(load_dividends(["AAA"], path=db)) == 1


# --- backtest engine ---------------------------------------------------------

def _engine_prices(days=20):
    dates = pd.bdate_range("2024-01-02", periods=days)
    return pd.DataFrame({"AAA": [100.0] * days}, index=dates)


def test_engine_credits_dividend_cash_exactly():
    """1000 shares x $2 dividend -> exactly $2,000 cash on the ex-date."""
    prices = _engine_prices()
    ex = prices.index[10]
    signals = pd.DataFrame(index=prices.index)
    result = run_backtest(
        prices,
        signals,
        lambda signals, day: {"AAA": 1.0},
        frequency="daily",
        cost_bps=0.0,
        initial_capital=100_000.0,
        dividends=pd.DataFrame(
            [{"ticker": "AAA", "ex_date": ex, "amount": 2.0}]
        ),
    )
    assert len(result.dividends) == 1
    row = result.dividends.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["shares_held"] == pytest.approx(1000.0)
    assert row["amount_paid"] == pytest.approx(2000.0)
    # Flat prices + no costs: final equity = capital + dividend cash.
    assert result.equity["portfolio_value"].iloc[-1] == pytest.approx(102_000.0)
    # The payout lands on the ex-date's equity, not before.
    before = result.equity.set_index("date").loc[ex - pd.Timedelta(days=1),
                                                 "portfolio_value"]
    on = result.equity.set_index("date").loc[ex, "portfolio_value"]
    assert on - before == pytest.approx(2000.0)


def test_engine_ex_date_buyer_gets_nothing():
    """A buy executing ON the ex-date is not entitled to the dividend."""
    prices = _engine_prices()
    ex = prices.index[10]
    signals = pd.DataFrame(index=prices.index)
    # Decide the buy the day before the ex-date: it executes at the
    # ex-date's close, after the dividend credit step for that day.
    buy_day = prices.index[9]
    result = run_backtest(
        prices,
        signals,
        lambda signals, day: {"AAA": 1.0} if day == buy_day else {},
        frequency="daily",
        cost_bps=0.0,
        initial_capital=100_000.0,
        dividends=pd.DataFrame(
            [{"ticker": "AAA", "ex_date": ex, "amount": 2.0}]
        ),
    )
    assert result.dividends.empty
    assert result.equity["portfolio_value"].iloc[-1] == pytest.approx(100_000.0)


def test_engine_no_dividends_unchanged():
    """dividends=None and an empty frame both reproduce the old behavior."""
    prices = _engine_prices()
    signals = pd.DataFrame(index=prices.index)
    fn = lambda signals, day: {"AAA": 1.0}  # noqa: E731
    base = run_backtest(prices, signals, fn, cost_bps=10.0)
    empty = run_backtest(
        prices, signals, fn, cost_bps=10.0,
        dividends=pd.DataFrame(columns=["ticker", "ex_date", "amount"]),
    )
    pd.testing.assert_frame_equal(base.equity, empty.equity)
    pd.testing.assert_frame_equal(base.trades, empty.trades)
    assert empty.dividends.empty


# --- paper broker ------------------------------------------------------------

def _paper_db(tmp_path):
    from qrp.execution.paper import PaperBroker

    db = tmp_path / "paper.db"
    init_db(db)
    dates = pd.bdate_range("2024-01-02", periods=12)
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 1_000_000,
            }
        ),
        "AAA",
        db,
    )
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 50.0,
                "High": 51.0,
                "Low": 49.0,
                "Close": 50.0,
                "Volume": 1_000_000,
            }
        ),
        "BBB",
        db,
    )
    upsert_dividends(_div_df([("2024-01-10", 0.50)]), "AAA", db)
    upsert_dividends(_div_df([("2024-01-10", 1.00)]), "BBB", db)
    broker = PaperBroker(db)
    return db, broker


def _insert_trade(db, date, ticker, side, shares):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO paper_trades (date, ticker, side, shares, price,"
            " cost, reason) VALUES (?, ?, ?, ?, 100.0, 0.0, '')",
            (date, ticker, side, shares),
        )


def test_paper_credit_dividends_trade_replay(tmp_path):
    """Only shares held at the prior close get paid; credit happens once."""
    db, broker = _paper_db(tmp_path)
    # AAA: bought BEFORE the ex-date -> paid on 100 shares.
    _insert_trade(db, "2024-01-05", "AAA", "BUY", 100.0)
    # BBB: bought ON the ex-date -> not entitled.
    _insert_trade(db, "2024-01-10", "BBB", "BUY", 100.0)

    credited = broker.credit_dividends()
    assert len(credited) == 1
    assert credited[0]["ticker"] == "AAA"
    assert credited[0]["shares_held"] == pytest.approx(100.0)
    assert credited[0]["amount_paid"] == pytest.approx(50.0)
    assert broker.cash == pytest.approx(100_000.0 + 50.0)

    # Second call: nothing new — each (ticker, ex-date) credits once.
    assert broker.credit_dividends() == []
    assert broker.cash == pytest.approx(100_050.0)

    # The log records what was considered, including the unpaid BBB.
    log = {d["ticker"]: d for d in broker.dividends()}
    assert log["AAA"]["amount_paid"] == pytest.approx(50.0)
    assert log["BBB"]["shares_held"] == pytest.approx(0.0)


def test_paper_sold_before_ex_date_gets_nothing(tmp_path):
    db, broker = _paper_db(tmp_path)
    _insert_trade(db, "2024-01-05", "AAA", "BUY", 100.0)
    _insert_trade(db, "2024-01-09", "AAA", "SELL", 100.0)  # out before ex-date
    assert broker.credit_dividends() == []
    assert broker.cash == pytest.approx(100_000.0)
