"""Tests for the SEC EDGAR fundamentals pipeline (Phase 7).

Mostly synthetic (no network): parsing, tag fallbacks, annual preference,
restatement handling, and the point-in-time rule. One live smoke test that
skips gracefully without network.
"""

import json

import pandas as pd
import pytest

from qrp.data import edgar
from qrp.data.edgar import (
    cik_for_ticker,
    clear_fetch_state,
    count_facts,
    extract_facts,
    facts_as_of,
    fetch_companyfacts,
    get_last_fetch,
    ingest_ticker,
    last_fetch_summary,
    set_last_fetch,
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


def test_dotted_tickers_share_one_db_key(tmp_path):
    """BRK.B must use the same DB key ('BRK-B') for bars and fundamentals.

    Yahoo stores share classes with dashes; without normalization the two
    tables would key Berkshire differently and fundamentals would never
    join to prices (silently dropping the stock from fundamental screens).
    """
    from qrp.data import store
    from qrp.data.yahoo import normalize_ticker

    db = tmp_path / "t.db"
    bars = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [100],
        }
    )
    store.upsert_bars(bars, "BRK.B", db)
    facts = pd.DataFrame(
        {
            "tag": ["net_income"],
            "period_end": ["2023-12-31"],
            "filed_date": ["2024-02-26"],
            "form": ["10-K"],
            "fp": ["FY"],
            "unit": ["USD"],
            "value": [1.0],
        }
    )
    upsert_facts(facts, "BRK.B", db)

    assert normalize_ticker("BRK.B") == "BRK-B"
    assert store.count_bars("BRK-B", db) == 1
    assert store.count_bars("BRK.B", db) == 1  # read path normalizes too
    from qrp.data.edgar import count_facts

    assert count_facts("BRK-B", db) == 1
    assert count_facts("BRK.B", db) == 1
    assert facts_as_of("BRK-B", "net_income", "2024-06-01", db).shape[0] == 1


# -- incremental updates (fetch-state bookkeeping) ---------------------------


def _mock_companyfacts(monkeypatch, payload):
    """Make ingest_ticker use a synthetic payload instead of the network.

    The payload object is read on every call, so tests can swap it between
    ingests by mutating a holder dict.
    """
    holder = {"payload": payload}
    monkeypatch.setattr(
        edgar,
        "fetch_companyfacts",
        lambda cik, session=None, timeout=60: holder["payload"],
    )
    return holder


def test_fetch_state_roundtrip_and_summary(tmp_path):
    """get/set/clear the last-download record; the summary joins fact counts."""
    db = tmp_path / "state.db"
    assert get_last_fetch("AAA", db) is None  # never downloaded

    stamp = set_last_fetch("aaa", db)  # normalized: stored as "AAA"
    assert get_last_fetch("AAA", db) == stamp
    assert get_last_fetch("aaa", db) == stamp

    # An explicit timestamp (backdating) is stored verbatim.
    set_last_fetch("BBB", db, when="2020-01-01T00:00:00+00:00")
    assert get_last_fetch("BBB", db) == "2020-01-01T00:00:00+00:00"

    # The summary lists every downloaded ticker with its fact count.
    df = extract_facts(SAMPLE_PAYLOAD)
    upsert_facts(df, "BBB", db)
    summary = last_fetch_summary(db)
    assert list(summary.columns) == ["ticker", "last_fetch", "facts"]
    assert list(summary["ticker"]) == ["AAA", "BBB"]
    assert summary.loc[summary["ticker"] == "AAA", "facts"].iloc[0] == 0
    assert summary.loc[summary["ticker"] == "BBB", "facts"].iloc[0] == len(df)

    # Clearing one ticker leaves the other alone.
    assert clear_fetch_state(["aaa"], db) == 1
    assert get_last_fetch("AAA", db) is None
    assert get_last_fetch("BBB", db) == "2020-01-01T00:00:00+00:00"

    # Clearing with None forgets every ticker.
    assert clear_fetch_state(None, db) == 1
    assert last_fetch_summary(db).empty


def test_incremental_second_ingest_stores_zero_new_facts(tmp_path, monkeypatch):
    """Re-ingesting an identical payload stores 0 new facts, no duplicates."""
    db = tmp_path / "incr.db"
    _mock_companyfacts(monkeypatch, SAMPLE_PAYLOAD)
    ticker_map = {"TEST": "0000000001"}

    n1 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    assert n1 == len(extract_facts(SAMPLE_PAYLOAD))
    assert get_last_fetch("TEST", db) is not None

    # Every filing in the payload is older than the last download, so the
    # second run has nothing new — and the row count does not grow.
    n2 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    assert n2 == 0
    assert count_facts("TEST", db) == n1


def test_incremental_picks_up_newly_filed_facts(tmp_path, monkeypatch):
    """Only filings newer than the last download are stored as new."""
    db = tmp_path / "incr2.db"
    _mock_companyfacts(monkeypatch, SAMPLE_PAYLOAD)
    ticker_map = {"TEST": "0000000001"}

    n1 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    # Pretend the last download happened before the 2024 filings landed.
    set_last_fetch("TEST", db, when="2024-01-01T00:00:00+00:00")

    n2 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    # Facts with filed_date after 2024-01-01: revenue (2024-02-01),
    # net_income 10-K (2024-02-01), net_income 10-Q (2024-01-20),
    # shares_outstanding (2024-02-01) -> 4 new facts.
    assert n2 == 4
    # The re-written rows are upserts, not duplicates.
    assert count_facts("TEST", db) == n1


def test_restatement_ingested_as_new_filing(tmp_path, monkeypatch):
    """Same period + new filing date (a restatement) is stored as its own row."""
    payload_v1 = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"USD": [_fact("2022-12-31", "2023-02-01", 100)]}
                }
            }
        }
    }
    payload_v2 = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _fact("2022-12-31", "2023-02-01", 100),
                            _fact(
                                "2022-12-31",
                                "2023-05-01",
                                110,
                                form="10-K/A",
                            ),
                        ]
                    }
                }
            }
        }
    }
    db = tmp_path / "rest.db"
    holder = _mock_companyfacts(monkeypatch, payload_v1)
    ticker_map = {"TEST": "0000000001"}

    n1 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    assert n1 == 1

    # The company later restates; the restatement is the only "new" filing.
    set_last_fetch("TEST", db, when="2023-03-01T00:00:00+00:00")
    holder["payload"] = payload_v2
    n2 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    assert n2 == 1
    # Original and restatement live side by side (append-only, no peeking).
    assert count_facts("TEST", db) == 2
    late = facts_as_of("TEST", "net_income", "2024-01-01", db)
    assert late.iloc[0]["value"] == 110
    assert late.iloc[0]["filed_date"] == "2023-05-01"


def test_incremental_false_reingests_everything(tmp_path, monkeypatch):
    """incremental=False keeps the old full-ingest behavior (and updates state)."""
    db = tmp_path / "full.db"
    _mock_companyfacts(monkeypatch, SAMPLE_PAYLOAD)
    ticker_map = {"TEST": "0000000001"}

    n1 = ingest_ticker("TEST", db, ticker_map=ticker_map)
    n2 = ingest_ticker("TEST", db, ticker_map=ticker_map, incremental=False)
    assert n2 == n1  # every filing re-stored, nothing skipped
    assert count_facts("TEST", db) == n1  # still no duplicates
    assert get_last_fetch("TEST", db) is not None  # state updated too


def test_cli_full_refresh_clears_state_and_reports(tmp_path, monkeypatch, capsys):
    """--full-refresh forgets last-download records and prints the summary."""
    import qrp.data.edgar_cli as cli

    db = str(tmp_path / "cli.db")
    set_last_fetch("AAA", db, when="2020-01-01T00:00:00+00:00")

    calls = []

    def fake_ingest(ticker, path, session=None, ticker_map=None, incremental=True):
        calls.append({"ticker": ticker, "incremental": incremental})
        return 3

    monkeypatch.setattr(cli, "load_ticker_map", lambda: {"AAA": "0000000001"})
    monkeypatch.setattr(cli, "ingest_ticker", fake_ingest)

    rc = cli.main(["--tickers", "AAA", "--db", db, "--full-refresh"])
    assert rc == 0
    # State was cleared before the loop (the fake ingest doesn't re-set it).
    assert get_last_fetch("AAA", db) is None
    assert calls == [{"ticker": "AAA", "incremental": False}]
    out = capsys.readouterr().out
    assert "AAA: stored 3 new facts" in out
    assert "Full refresh complete: 3 new facts across 1 ticker" in out


def test_cli_default_is_incremental(tmp_path, monkeypatch, capsys):
    """Without --full-refresh the CLI passes incremental=True through."""
    import qrp.data.edgar_cli as cli

    db = str(tmp_path / "cli2.db")
    calls = []

    def fake_ingest(ticker, path, session=None, ticker_map=None, incremental=True):
        calls.append(incremental)
        return 5

    monkeypatch.setattr(cli, "load_ticker_map", lambda: {"AAA": "0000000001"})
    monkeypatch.setattr(cli, "ingest_ticker", fake_ingest)

    assert cli.parse_args([]).full_refresh is False
    assert cli.parse_args(["--full-refresh"]).full_refresh is True

    rc = cli.main(["--tickers", "AAA", "--db", db])
    assert rc == 0
    assert calls == [True]
    out = capsys.readouterr().out
    assert "AAA: stored 5 new facts" in out
    assert "Incremental update complete: 5 new facts across 1 ticker" in out


# ---------------------------------------------------------------------------
# Audit 1.3: quarterly vs YTD duration handling
# ---------------------------------------------------------------------------

def _collision_payload():
    """A 10-Q filing reporting NetIncomeLoss in two durations.

    Q2 2024 10-Q: 3-month fact (2024-04-01 -> 2024-06-30, val=30) and
    6-month YTD fact (2024-01-01 -> 2024-06-30, val=50). Same end date,
    same filing — the old code picked one arbitrarily via pool[0].
    """
    def entry(start, end, val):
        d = {"end": end, "val": val, "filed": "2024-07-25",
             "form": "10-Q", "fp": "Q2"}
        if start:
            d["start"] = start
        return d
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            entry("2024-04-01", "2024-06-30", 30.0),
                            entry("2024-01-01", "2024-06-30", 50.0),
                        ]
                    }
                }
            }
        }
    }


def test_extract_facts_prefers_ytd_over_quarterly():
    """The YTD fact wins when a filing reports both durations."""
    df = extract_facts(_collision_payload())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["tag"] == "net_income"
    assert float(row["value"]) == 50.0, (
        "must keep the 6-month YTD figure, not the 3-month figure"
    )
    assert row["period_start"] == "2024-01-01"
    assert row["period_end"] == "2024-06-30"


def test_extract_facts_records_start_for_flow_facts():
    """period_start distinguishes durations; instant facts get None."""
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"USD": [
                        {"start": "2024-01-01", "end": "2024-03-31",
                         "val": 10.0, "filed": "2024-04-25",
                         "form": "10-Q", "fp": "Q1"},
                    ]}
                },
                "Assets": {
                    "units": {"USD": [
                        {"end": "2024-03-31", "val": 1000.0,
                         "filed": "2024-04-25", "form": "10-Q", "fp": "Q1"},
                    ]}
                },
            }
        }
    }
    df = extract_facts(payload)
    ni = df[df["tag"] == "net_income"].iloc[0]
    assert ni["period_start"] == "2024-01-01"
    assets = df[df["tag"] == "assets"].iloc[0]
    assert pd.isna(assets["period_start"])


def test_period_start_migration_preserves_rows(tmp_path):
    """Old DBs gain period_start via ADD COLUMN; rows survive as NULL."""
    import sqlite3
    from qrp.data.edgar import init_db
    db = tmp_path / "old.db"
    # Build an old-schema table by hand (no period_start).
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fundamentals (ticker TEXT, tag TEXT, period_end TEXT,"
        " filed_date TEXT, form TEXT, fp TEXT, unit TEXT, value REAL,"
        " PRIMARY KEY (ticker, tag, period_end, filed_date))"
    )
    conn.execute(
        "INSERT INTO fundamentals VALUES"
        " ('AAA', 'net_income', '2024-03-31', '2024-04-25',"
        "  '10-Q', 'Q1', 'USD', 10.0)"
    )
    conn.commit()
    conn.close()
    # init_db must add the column without touching the row.
    init_db(db)
    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")]
    assert "period_start" in cols
    row = conn.execute(
        "SELECT value, period_start FROM fundamentals"
    ).fetchone()
    conn.close()
    assert row == (10.0, None)


def test_upsert_and_query_roundtrip_with_start(tmp_path):
    """period_start survives upsert -> facts_as_of."""
    from qrp.data.edgar import init_db
    db = tmp_path / "rt.db"
    init_db(db)
    df = extract_facts(_collision_payload())
    assert upsert_facts(df, "AAA", db) == 1
    got = facts_as_of("AAA", "net_income", "2024-08-01", db)
    assert len(got) == 1
    assert got.iloc[0]["period_start"] == "2024-01-01"
    assert float(got.iloc[0]["value"]) == 50.0


def test_no_peeking_with_duration_rows(tmp_path):
    """A restatement's period_start is also point-in-time gated."""
    db = tmp_path / "np.db"
    df = extract_facts(_collision_payload())
    upsert_facts(df, "AAA", db)
    # Before the filing date: nothing is knowable.
    assert facts_as_of("AAA", "net_income", "2024-07-24", db).empty
    # On/after: the YTD row with its period_start.
    got = facts_as_of("AAA", "net_income", "2024-07-25", db)
    assert len(got) == 1
    assert got.iloc[0]["period_start"] == "2024-01-01"
