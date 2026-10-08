"""Tests for the scoped incremental refreshes ("Refresh prices" and
"Refresh SEC filings").

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

    def fake_actions(ticker, start, end, session=None, timeout=30):
        # Same dividends as fake_dividends, plus no splits by default.
        calls.setdefault("actions", []).append((ticker, start, end))
        return {
            "splits": pd.DataFrame(columns=["date", "numerator", "denominator"]),
            "dividends": pd.DataFrame({"ex_date": [], "amount": []}),
            "prices": pd.DataFrame(columns=["Date", "Close", "AdjClose"]),
        }

    def fake_ingest(ticker, path, session=None, ticker_map=None):
        calls["filings"].append(ticker)
        return facts_to_return["n"]

    monkeypatch.setattr(H, "fetch_daily_bars", fake_bars)
    monkeypatch.setattr(H, "fetch_dividends", fake_dividends)
    monkeypatch.setattr(H, "fetch_actions", fake_actions)
    monkeypatch.setattr(H, "ingest_ticker", fake_ingest)
    monkeypatch.setattr(H, "load_ticker_map", lambda: {})
    monkeypatch.setattr(
        H, "fundamentals_schema_is_current", lambda db_path: True
    )
    # The refresh builds one HTTP session per worker thread; keep the
    # fakes fully offline by stubbing session construction too.
    monkeypatch.setattr(H, "_thread_yahoo_session", lambda _tls: None)
    monkeypatch.setattr(H, "_thread_sec_session", lambda _tls: None)
    return calls, facts_to_return


def test_refresh_with_nothing_new_is_a_noop(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    # Prices already run through the latest trading day: nothing to fetch.
    seed_bars(db, end=date.today().isoformat(), periods=5)
    before = bar_count(db)
    facts["n"] = 0

    summary = H.refresh_prices(db)

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

    summary = H.refresh_prices(db)

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

    summary = H.refresh_prices(db)

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
    summary2 = H.refresh_prices(db)
    assert summary2["new_bars"] == 0
    assert bar_count(db) == before + summary["new_bars"]
    assert duplicate_bars(db) == 0


def test_refresh_counts_new_filings(tmp_path, no_network):
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db)
    facts["n"] = 12

    summary = H.refresh_filings(db)

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
    summary = H.refresh_prices(db)

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
    summary = H.refresh_prices(db)

    assert len(summary["price_failures"]) == 1
    assert summary["price_failures"][0][0] == "AAA"
    # BBB still refreshed fine.
    assert summary["tickers_with_new_bars"] == 1
    assert any("trouble" in line for line in summary["lines"])


def test_refresh_with_empty_database(tmp_path, no_network):
    db = tmp_path / "t.db"
    for fn in (H.refresh_prices, H.refresh_filings):
        summary = fn(db)
        assert summary["tickers"] == 0
        assert summary["up_to_date"] is True
        assert "no stocks" in summary["lines"][0].lower()


def test_refresh_does_not_count_revised_old_bars_as_new(
    tmp_path, no_network, monkeypatch
):
    """Yahoo sometimes re-sends the latest bar with revised values.

    The upsert still applies the revision (data stays accurate), but the
    summary must not claim anything new arrived.
    """
    calls, facts = no_network
    db = tmp_path / "t.db"
    # Last stored date strictly before today, so the fetch runs — but the
    # fake only re-sends that already-stored date with revised values.
    seeded = seed_bars(
        db, end=(date.today() - timedelta(days=3)).isoformat(), periods=5
    )
    last = seeded[-1]
    facts["n"] = 0

    def revised(ticker, start, end, session=None, timeout=30):
        # Only the already-stored last date, with tweaked values.
        return pd.DataFrame(
            {
                "Date": [last],
                "Open": [999.0],
                "High": [999.0],
                "Low": [999.0],
                "Close": [999.0],
                "Volume": [1],
            }
        )

    monkeypatch.setattr(H, "fetch_daily_bars", revised)
    summary = H.refresh_prices(db)

    assert summary["new_bars"] == 0
    assert summary["up_to_date"] is True
    assert "up to date" in summary["lines"][0].lower()
    # ...but the revision WAS applied.
    with sqlite3.connect(db) as conn:
        (o,) = conn.execute(
            "SELECT open FROM daily_bars WHERE ticker='AAA' AND date=?",
            (last,),
        ).fetchone()
    assert o == pytest.approx(999.0)


def test_last_trading_day():
    # 2026-10-10 is a Saturday, 2026-10-11 a Sunday.
    assert H._last_trading_day(date(2026, 10, 10)) == date(2026, 10, 9)
    assert H._last_trading_day(date(2026, 10, 11)) == date(2026, 10, 9)
    assert H._last_trading_day(date(2026, 10, 9)) == date(2026, 10, 9)
    assert H._last_trading_day(date(2026, 10, 7)) == date(2026, 10, 7)
    assert H._last_trading_day(date(2026, 10, 12)) == date(2026, 10, 12)


def test_refresh_skips_tickers_current_through_last_trading_day(
    tmp_path, no_network, monkeypatch
):
    """On a Saturday, a ticker current through Friday needs no price fetch.

    No Yahoo request at all — this is what makes weekend refreshes
    nearly free.
    """

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 10)  # Saturday

    monkeypatch.setattr(H, "date", FakeDate)
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, end="2026-10-09", periods=5)  # through Friday
    facts["n"] = 0

    summary = H.refresh_prices(db)

    assert calls["prices"] == []
    assert summary["new_bars"] == 0
    assert summary["up_to_date"] is True


def test_refresh_still_fetches_stale_ticker_on_weekend(
    tmp_path, no_network, monkeypatch
):
    """A ticker that is genuinely behind still refreshes on a weekend."""

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 10)  # Saturday

    monkeypatch.setattr(H, "date", FakeDate)
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, end="2026-10-01", periods=5)  # stale: before last Friday
    facts["n"] = 0

    summary = H.refresh_prices(db)

    assert len(calls["prices"]) == 1
    assert summary["new_bars"] > 0


def test_refresh_parallel_failure_isolation_many_tickers(
    tmp_path, no_network, monkeypatch
):
    """With several tickers, one bad apple never spoils the batch."""
    calls, facts = no_network
    db = tmp_path / "t.db"
    for t in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]:
        seed_bars(db, ticker=t)
    facts["n"] = 0

    real_fake = H.fetch_daily_bars

    def flaky(ticker, start, end, session=None, timeout=30):
        if ticker in ("BBB", "EEE"):
            raise RuntimeError("boom")
        return real_fake(ticker, start, end, session, timeout)

    monkeypatch.setattr(H, "fetch_daily_bars", flaky)
    summary = H.refresh_prices(db)

    failed = sorted(t for t, _ in summary["price_failures"])
    assert failed == ["BBB", "EEE"]
    assert summary["tickers_with_new_bars"] == 4


def test_chart_payload_retries_on_429_then_succeeds(monkeypatch):
    """A 429 from Yahoo is retried with backoff, then succeeds."""
    import requests as rq
    from qrp.data import yahoo as Y

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code == 429:
                raise rq.HTTPError("429", response=self)

        def json(self):
            return {"chart": {"result": [{"meta": {}}]}}

    class FakeSess:
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            return FakeResp(429 if calls["n"] <= 2 else 200)

    monkeypatch.setattr(Y.time, "sleep", lambda s: None)
    result, _sess = Y._chart_payload(
        "AAA", "2026-10-01", "2026-10-09", session=FakeSess()
    )
    assert calls["n"] == 3
    assert result == {"meta": {}}


def test_chart_payload_gives_up_after_429_retries(monkeypatch):
    """Persistent 429s surface as HTTPError after the retries run out."""
    import requests as rq
    from qrp.data import yahoo as Y

    calls = {"n": 0}

    class FakeResp:
        status_code = 429

        def raise_for_status(self):
            raise rq.HTTPError("429", response=self)

        def json(self):
            return {}

    class FakeSess:
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            return FakeResp()

    monkeypatch.setattr(Y.time, "sleep", lambda s: None)
    with pytest.raises(rq.HTTPError):
        Y._chart_payload("AAA", "2026-10-01", "2026-10-09", session=FakeSess())
    assert calls["n"] == 4  # 1 initial + 3 retries


def test_refresh_prices_summary_mentions_no_filings(tmp_path, no_network):
    """The prices scope reports prices only — no filings content."""
    db = tmp_path / "t.db"
    seed_bars(db, ticker="AAA")
    summary = H.refresh_prices(db)
    text = " ".join(summary["lines"]).lower()
    assert "sec" not in text and "filing" not in text
    assert summary["new_facts"] == 0
    assert summary["fund_failures"] == []


def test_refresh_filings_summary_mentions_no_prices(tmp_path, no_network):
    """The filings scope reports filings only — no price content."""
    calls, facts = no_network
    db = tmp_path / "t.db"
    seed_bars(db, ticker="AAA")
    facts["n"] = 4
    summary = H.refresh_filings(db)
    assert summary["new_facts"] == 4
    assert summary["tickers_with_new_facts"] == 1
    assert not summary["up_to_date"]
    text = " ".join(summary["lines"]).lower()
    assert "daily prices" not in text
    assert summary["new_bars"] == 0
    assert summary["price_failures"] == []
    # And the price fetcher was never touched.
    assert calls["prices"] == []


# --- 1.6: split-straddling refresh -------------------------------------------

def test_refresh_redownloads_full_history_on_split(tmp_path, no_network, monkeypatch):
    """A 2:1 split straddling the refresh must re-download the full history
    so no fake -50% seam appears — not just append post-split bars."""
    calls, _ = no_network
    db = tmp_path / "t.db"
    # Pre-split history at $200, 10 business days ending Friday 2024-06-07.
    first_dates = seed_bars(db, end="2024-06-07", periods=10, price=200.0)
    first_date = first_dates[0]

    bar_calls = []

    def fake_bars_split(ticker, start, end, session=None, timeout=30):
        bar_calls.append((ticker, start, end))
        dates = pd.date_range(start=start, end=end, freq="B")
        # Post-split Yahoo bars are all on the $100 basis.
        return pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 100.0, "High": 101.0, "Low": 99.0,
                "Close": 100.0, "Volume": 200_000,
            }
        )

    def fake_actions_split(ticker, start, end, session=None, timeout=30):
        return {
            "splits": pd.DataFrame(
                [{"date": "2024-06-10", "numerator": 2.0, "denominator": 1.0}]
            ),
            "dividends": pd.DataFrame({"ex_date": [], "amount": []}),
            "prices": pd.DataFrame(columns=["Date", "Close", "AdjClose"]),
        }

    monkeypatch.setattr(H, "fetch_daily_bars", fake_bars_split)
    monkeypatch.setattr(H, "fetch_actions", fake_actions_split)

    summary = H.refresh_prices(db)

    # Full re-download: bars were requested from the first stored date.
    starts = [c[1] for c in bar_calls]
    assert first_date in starts, (
        f"expected full re-download from {first_date}, got {starts}"
    )
    # Every stored bar is now on the post-split $100 basis — no seam, no
    # fake crash in the series.
    with sqlite3.connect(db) as conn:
        closes = [
            r[0] for r in conn.execute(
                "SELECT close FROM daily_bars WHERE ticker = 'AAA' ORDER BY date"
            )
        ]
    assert closes and all(c == pytest.approx(100.0) for c in closes)
    # Plain-English note in the summary.
    assert any("split" in line.lower() for line in summary["lines"])


def test_refresh_falls_back_to_incremental_when_actions_fail(
    tmp_path, no_network, monkeypatch
):
    """If the corporate-actions fetch fails, the price refresh still works
    (incremental bars + plain dividend fetch) — never a fatal error."""
    calls, _ = no_network
    db = tmp_path / "t.db"
    seed_bars(db, end=(date.today() - timedelta(days=5)).isoformat(), periods=5)

    def boom(ticker, start, end, session=None, timeout=30):
        raise RuntimeError("yahoo hiccup")

    monkeypatch.setattr(H, "fetch_actions", boom)
    summary = H.refresh_prices(db)

    assert summary["price_failures"] == []
    assert summary["new_bars"] > 0


# --- 2.4: WAL-aware _db_mtime -------------------------------------------------

def test_db_mtime_missing_files_is_zero(tmp_path):
    assert H._db_mtime(tmp_path / "nope.db") == 0.0


def test_db_mtime_uses_newest_of_db_wal_shm(tmp_path):
    import os

    db = tmp_path / "t.db"
    db.write_bytes(b"x")
    os.utime(db, (1000, 1000))
    assert H._db_mtime(db) == 1000.0
    # A newer WAL file wins (writes land there first in WAL mode).
    wal = tmp_path / "t.db-wal"
    wal.write_bytes(b"y")
    os.utime(wal, (2000, 2000))
    assert H._db_mtime(db) == 2000.0
    # ...as does a newer SHM file.
    shm = tmp_path / "t.db-shm"
    shm.write_bytes(b"z")
    os.utime(shm, (3000, 3000))
    assert H._db_mtime(db) == 3000.0
    # Sidecars gone: falls back to the db file itself.
    wal.unlink()
    shm.unlink()
    assert H._db_mtime(db) == 1000.0
