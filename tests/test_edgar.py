"""Tests for the SEC EDGAR fundamentals pipeline (Phase 7).

Mostly synthetic (no network): parsing, tag fallbacks, annual preference,
restatement handling, and the point-in-time rule. One live smoke test that
skips gracefully without network.
"""

import json

import pandas as pd
import pytest

from qrp.data.edgar import (
    cik_for_ticker,
    extract_facts,
    facts_as_of,
    fetch_companyfacts,
    ingest_ticker,
    upsert_facts,
)


def _fact(end, filed, val, form="10-K", fp="FY", accn="x"):
    return {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": 2023,
        "fp": fp,
        "form": form,
        "filed": filed,
    }


SAMPLE_PAYLOAD = {
    "cik": 1234567,
    "entityName": "Test Corp",
    "facts": {
        "us-gaap": {
            # Primary revenue tag missing -> must fall back to Revenues.
            "Revenues": {
                "units": {"USD": [_fact("2023-12-31", "2024-02-01", 1000)]}
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        # Original filing...
                        _fact("2022-12-31", "2023-02-01", 100, accn="a1"),
                        # ...later restated via 10-K/A: the later filing wins.
                        _fact(
                            "2022-12-31", "2023-05-01", 110, form="10-K/A", accn="a2"
                        ),
                        _fact("2023-12-31", "2024-02-01", 150, accn="a3"),
                        # Same period covered by 10-Q and 10-K: 10-K wins.
                        _fact(
                            "2023-12-31",
                            "2024-01-20",
                            999,
                            form="10-Q",
                            fp="Q4",
                            accn="q4",
                        ),
                    ]
                }
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [_fact("2023-12-31", "2024-02-01", 50)]}
            }
        },
    },
}


def test_tag_fallback_and_dei_namespace():
    df = extract_facts(SAMPLE_PAYLOAD)
    rev = df[df["tag"] == "revenue"]
    assert len(rev) == 1
    assert rev.iloc[0]["value"] == 1000  # fell back to Revenues
    shares = df[df["tag"] == "shares_outstanding"]
    assert len(shares) == 1
    assert shares.iloc[0]["value"] == 50
    assert shares.iloc[0]["unit"] == "shares"


def test_restatement_kept_as_separate_rows():
    """Append-only: the original filing and the restatement are both kept.

    Two filings covering the same period become two rows with their own
    filing dates. The query helpers (facts_as_of) pick the latest filing
    knowable as of the requested date — never the future's number early.
    """
    df = extract_facts(SAMPLE_PAYLOAD)
    ni = df[df["tag"] == "net_income"]
    orig = ni[(ni["period_end"] == "2022-12-31") & (ni["filed_date"] == "2023-02-01")]
    rest = ni[(ni["period_end"] == "2022-12-31") & (ni["filed_date"] == "2023-05-01")]
    assert orig.iloc[0]["value"] == 100
    assert rest.iloc[0]["value"] == 110
    assert rest.iloc[0]["form"] == "10-K/A"
    # The 10-Q and the 10-K covering 2023-12-31 are separate filings, so
    # both rows exist; the query helpers pick the latest filed as of a date.
    q4_10q = ni[(ni["period_end"] == "2023-12-31") & (ni["form"] == "10-Q")]
    fy_10k = ni[(ni["period_end"] == "2023-12-31") & (ni["form"] == "10-K")]
    assert q4_10q.iloc[0]["value"] == 999
    assert fy_10k.iloc[0]["value"] == 150


def test_point_in_time_rule(tmp_path):
    db = tmp_path / "test.db"
    df = extract_facts(SAMPLE_PAYLOAD)
    upsert_facts(df, "TEST", db)

    # Before the restatement was filed, the ORIGINAL value is visible —
    # the restatement is invisible (no peeking at the future).
    early = facts_as_of("TEST", "net_income", "2023-04-01", db)
    row = early[early["period_end"] == "2022-12-31"].iloc[0]
    assert row["value"] == 100
    assert row["filed_date"] == "2023-02-01"

    # After the restatement filing, the corrected value is visible.
    late = facts_as_of("TEST", "net_income", "2024-01-01", db)
    row = late[late["period_end"] == "2022-12-31"].iloc[0]
    assert row["value"] == 110
    assert row["filed_date"] == "2023-05-01"

    # The 10-Q filed 2024-01-20 is what was knowable before the 10-K landed...
    mid = facts_as_of("TEST", "net_income", "2024-01-25", db)
    row = mid[mid["period_end"] == "2023-12-31"].iloc[0]
    assert row["value"] == 999
    # ...and the 10-K supersedes it once filed.
    later = facts_as_of("TEST", "net_income", "2024-03-01", db)
    row = later[later["period_end"] == "2023-12-31"].iloc[0]
    assert row["value"] == 150

    # A query before any filing returns nothing (no peeking at the future).
    assert facts_as_of("TEST", "net_income", "2022-01-01", db).empty


def test_unknown_ticker_has_no_cik(tmp_path):
    cache = tmp_path / "tickers.json"
    cache.write_text(json.dumps({"0": {"cik_str": 1, "ticker": "AAA", "title": "x"}}))
    assert cik_for_ticker("ZZZZ", cache) is None
    assert cik_for_ticker("aaa", cache) == "0000000001"


def test_ingest_unknown_ticker_raises_plainly(tmp_path):
    cache = tmp_path / "tickers.json"
    cache.write_text(json.dumps({"0": {"cik_str": 1, "ticker": "AAA", "title": "x"}}))
    db = tmp_path / "test.db"
    with pytest.raises(ValueError, match="no SEC filing record"):
        ingest_ticker("ZZZZ", db, cache_path=cache)


def test_fetch_aapl_companyfacts_smoke():
    try:
        payload = fetch_companyfacts("0000320193")
    except Exception as exc:  # any network/payload failure -> skip, not fail
        pytest.skip(f"network unavailable for EDGAR test: {exc}")
    df = extract_facts(payload)
    assert not df.empty
    assert (df["tag"] == "net_income").any()
    assert (df["tag"] == "revenue").any()


def test_old_schema_rebuilt_with_plain_message(tmp_path, capsys):
    """A DB with the old replace-style PK is rebuilt, with a plain-English
    notice and the exact command to re-ingest (old rows are unrecoverable,
    because restatements had already overwritten the originals)."""
    import sqlite3

    from qrp.data.edgar import init_db

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE fundamentals (ticker TEXT NOT NULL, tag TEXT NOT NULL,"
            " period_end TEXT NOT NULL, filed_date TEXT NOT NULL, form TEXT,"
            " fp TEXT, unit TEXT, value REAL NOT NULL,"
            " PRIMARY KEY (ticker, tag, period_end))"
        )
        conn.execute(
            "INSERT INTO fundamentals VALUES ('TEST','net_income','2022-12-31',"
            " '2023-02-01','10-K','FY','USD',100)"
        )
    init_db(db)
    out = capsys.readouterr().out
    assert "older format" in out
    assert "edgar_cli" in out  # the exact re-ingest command is printed
    with sqlite3.connect(db) as conn:
        pk = [
            r[1]
            for r in conn.execute("PRAGMA table_info(fundamentals)").fetchall()
            if r[5] > 0
        ]
    assert set(pk) == {"ticker", "tag", "period_end", "filed_date"}


def test_new_schema_is_not_rebuilt(tmp_path, capsys):
    """init_db on a current-schema DB is a no-op (no scare message)."""
    import sqlite3

    from qrp.data.edgar import init_db

    db = tmp_path / "new.db"
    init_db(db)
    out = capsys.readouterr().out
    assert "older format" not in out
    with sqlite3.connect(db) as conn:
        pk = [
            r[1]
            for r in conn.execute("PRAGMA table_info(fundamentals)").fetchall()
            if r[5] > 0
        ]
    assert set(pk) == {"ticker", "tag", "period_end", "filed_date"}


def test_find_cik_ticker_variants():
    from qrp.data.edgar import _find_cik

    mapping = {"BRK-B": "0001067983", "AAPL": "0000320193"}
    assert _find_cik(mapping, "brk.b") == "0001067983"  # dot vs dash
    assert _find_cik(mapping, "BRK-B") == "0001067983"
    assert _find_cik(mapping, "aapl") == "0000320193"
    assert _find_cik(mapping, "MSFT") is None


def test_ticker_map_refreshes_when_stale(tmp_path, monkeypatch):
    """A cache older than TICKER_MAP_MAX_AGE_DAYS is re-downloaded (ticker renames)."""
    import os
    import time

    from qrp.data import edgar

    cache = tmp_path / "company_tickers.json"
    cache.write_text(json.dumps({"0": {"cik_str": 1, "ticker": "OLDCO"}}))
    old_mtime = time.time() - 60 * 86400  # 60 days old
    os.utime(cache, (old_mtime, old_mtime))

    new_payload = {"0": {"cik_str": 2, "ticker": "NEWCO"}}

    class FakeResp:
        text = json.dumps(new_payload)

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, timeout=30):
            assert "company_tickers.json" in url
            return FakeResp()

    monkeypatch.setattr(edgar, "_session", lambda: FakeSession())
    assert edgar.load_ticker_map(cache) == {"NEWCO": "0000000002"}

    # A fresh cache is NOT re-downloaded.
    calls = []

    class CountingSession(FakeSession):
        def get(self, url, timeout=30):
            calls.append(url)
            return super().get(url, timeout)

    monkeypatch.setattr(edgar, "_session", lambda: CountingSession())
    assert edgar.load_ticker_map(cache) == {"NEWCO": "0000000002"}
    assert calls == []
