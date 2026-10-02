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


def test_latest_filing_wins_and_annual_preferred():
    df = extract_facts(SAMPLE_PAYLOAD)
    ni = df[df["tag"] == "net_income"].set_index("period_end")
    # Restatement: later filing date and corrected value.
    assert ni.loc["2022-12-31", "value"] == 110
    assert ni.loc["2022-12-31", "filed_date"] == "2023-05-01"
    # 10-K beats the 10-Q covering the same period.
    assert ni.loc["2023-12-31", "value"] == 150
    assert ni.loc["2023-12-31", "form"] == "10-K"


def test_point_in_time_rule(tmp_path):
    db = tmp_path / "test.db"
    df = extract_facts(SAMPLE_PAYLOAD)
    upsert_facts(df, "TEST", db)

    # Before the restatement was filed, the 2022 row is invisible entirely:
    # its latest filing date (2023-05-01) is after the query date.
    early = facts_as_of("TEST", "net_income", "2023-04-01", db)
    assert "2022-12-31" not in set(early["period_end"])

    # After the restatement filing, the corrected value is visible.
    late = facts_as_of("TEST", "net_income", "2024-01-01", db)
    row = late[late["period_end"] == "2022-12-31"].iloc[0]
    assert row["value"] == 110

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
