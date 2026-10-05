"""Tests for the one-click "Refresh new data" incremental download.

All synthetic — network calls are monkeypatched. Each test uses a
throwaway SQLite database.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pandas as pd
import pytest

from qrp.data.store import upsert_bars
from qrp.ui import helpers as H


def seed_bars(path, ticker="AAA", end=None, periods=5, price=100.0):
    """Seed `periods` business days ending on `end` (default: 7 days ago)."""
    if end is None:
        end = (date.today() - timedelta(days=7)).isoformat()
    dates = pd.date_range(end=end, periods=periods, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": 100_000,
        }
    )
    upsert_bars(df, ticker, path)
    return dates.strftime("%Y-%m-%d").tolist()


def bar_count(path) -> int:
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()
    return n


def duplicate_bars(path) -> int:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT ticker, date, COUNT(*) FROM daily_bars"
            " GROUP BY ticker, date HAVING COUNT(*) > 1"
        ).fetchall()
    return len(rows)


@pytest.fixture
def no_network(monkeypatch):
    """Patch every network touchpoint with deterministic fakes."""
    calls = {"prices": [], "dividends": [], "filings": []}
    facts_to_return = {"n": 0}

    def fake_bars(ticker, start, end, session=None, timeout=30):
        calls["prices"].append((ticker, start, end))
        dates = pd.date_range(start=start, end=end, freq="B")
        return pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 101.0,
                "High": 102.0,
                "Low": 100.0,
                "Close": 101.0,
                "Volume": 100_000,
            }
        )

    def fake_dividends(ticker, start, end, session=None, timeout=30):
        calls["dividends"].append(ticker)
        return pd.DataFrame({"ex_date": [], "amount": []})

    def fake_ingest(ticker, path, session=None, ticker_map=None):
        calls["filings"].append(ticker)
        return facts_to_return["n"]

    monkeypatch.setattr(H, "fetch_daily_bars", fake_bars)
    monkeypatch.setattr(H, "fetch_dividends", fake_dividends)
    monkeypatch.setattr(H, "ingest_ticker", fake_ingest)
    monkeypatch.setattr(H, "load_ticker_map", lambda: {})
    monkeypatch.setattr(
        H, "fundamentals_schema_is_current", lambda db_path: True
    )
    return calls, facts_to_return


def test_refresh_with_nothing_new_is_a_noop(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    # Prices already run through the latest trading day: nothing to fetch.
    seed_bars(db, end=date.today().isoformat(), periods=5)
    before = bar_count(db)
    facts["n"] = 0

    summary = H.refresh_new_data(db)

    assert summary["up_to_date"] is True
    assert summary["new_bars"] == 0
    assert summary["new_facts"] == 0
    assert summary["price_failures"] == []
    assert "up to date" in summary["lines"][0].lower()
    assert bar_count(db) == before


def test_refresh_skips_tickers_already_current(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, end=(date.today() - timedelta(days=10)).isoformat(),
              periods=3)
    # A bar dated today: this ticker needs no price refresh at all.
    today = date.today().isoformat()
    upsert_bars(
        pd.DataFrame(
            {
                "Date": [today],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [100_000],
            }
        ),
        "AAA",
        db,
    )
    facts["n"] = 0

    summary = H.refresh_new_data(db)

    # start (tomorrow) > today, so Yahoo is never even asked.
    assert calls["prices"] == []
    assert summary["new_bars"] == 0


def test_refresh_picks_up_only_new_bars_without_duplicates(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seeded = seed_bars(
        db, end=(date.today() - timedelta(days=7)).isoformat(), periods=5
    )
    last_before = seeded[-1]
    before = bar_count(db)
    facts["n"] = 0

    summary = H.refresh_new_data(db)

    assert summary["up_to_date"] is False
    assert summary["new_bars"] > 0
    assert summary["tickers_with_new_bars"] == 1
    # The fake only serves business days in the requested window.
    assert bar_count(db) == before + summary["new_bars"]
    assert duplicate_bars(db) == 0
    # Every fetched window starts the day after the last stored date.
    for ticker, start, end in calls["prices"]:
        assert start > last_before

    # A second refresh finds nothing new and duplicates nothing.
    summary2 = H.refresh_new_data(db)
    assert summary2["new_bars"] == 0
    assert bar_count(db) == before + summary["new_bars"]
    assert duplicate_bars(db) == 0


def test_refresh_counts_new_filings(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db)
    facts["n"] = 12

    summary = H.refresh_new_data(db)

    assert summary["new_facts"] == 12
    assert summary["tickers_with_new_facts"] == 1
    assert calls["filings"] == ["AAA"]
    assert any("12" in line for line in summary["lines"])


def test_refresh_survives_a_weekend_window(tmp_path, no_network, monkeypatch):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, end=(date.today() - timedelta(days=3)).isoformat())
    facts["n"] = 0

    def no_rows(ticker, start, end, session=None, timeout=30):
        raise ValueError(f"Yahoo returned no rows for {ticker}")

    monkeypatch.setattr(H, "fetch_daily_bars", no_rows)
    summary = H.refresh_new_data(db)

    # A "no rows" window (e.g. weekend) is a no-op, not a failure.
    assert summary["price_failures"] == []
    assert summary["new_bars"] == 0


def test_refresh_reports_per_ticker_failures_without_aborting(
    tmp_path, no_network, monkeypatch
):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, ticker="AAA")
    seed_bars(db, ticker="BBB")
    facts["n"] = 0

    real_fake = H.fetch_daily_bars

    def flaky(ticker, start, end, session=None, timeout=30):
        if ticker == "AAA":
            raise RuntimeError("boom")
        return real_fake(ticker, start, end, session, timeout)

    monkeypatch.setattr(H, "fetch_daily_bars", flaky)
    summary = H.refresh_new_data(db)

    assert len(summary["price_failures"]) == 1
    assert summary["price_failures"][0][0] == "AAA"
    # BBB still refreshed fine.
    assert summary["tickers_with_new_bars"] == 1
    assert any("trouble" in line for line in summary["lines"])


def test_refresh_with_empty_database(tmp_path, no_network):
    db = tmp_path / "t.db"
    summary = H.refresh_new_data(db)
    assert summary["tickers"] == 0
    assert summary["up_to_date"] is True
    assert "no stocks" in summary["lines"][0].lower()
