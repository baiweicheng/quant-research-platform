"""Tests for qrp/data/sectors.py — GICS sector ingestion."""

import sqlite3

import pytest

from qrp.data.sectors import (
    GICS_SECTORS,
    SectorFetchError,
    parse_sectors,
    refresh_sectors,
    sector_of,
    sectors_for,
    status,
)

# Minimal Wikipedia-style constituent table (header names match the live
# page: Symbol / Security / GICS Sector / GICS Sub-Industry).
_SECTORS_8 = [
    "Information Technology", "Financials", "Energy", "Industrials",
    "Health Care", "Consumer Discretionary", "Utilities", "Materials",
]
_FIXTURE_ROWS = "\n".join(
    f"<tr><td>{sym}</td><td>{name}</td><td>{sec}</td><td>{sub}</td></tr>"
    for sym, name, sec, sub in [
        ("AAPL", "Apple Inc.", "Information Technology", "Consumer Electronics"),
        ("JPM", "JPMorgan Chase & Co.", "Financials", "Diversified Banks"),
        ("XOM", "Exxon Mobil Corp.", "Energy", "Integrated Oil & Gas"),
    ]
    + [
        (f"T{i:03d}", f"Test Corp {i}", _SECTORS_8[i % 8], "Conglomerates")
        for i in range(500)
    ]
)

FIXTURE_HTML = f"""
<html><body><table class="wikitable">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
<th>GICS Sub-Industry</th><th>Headquarters</th></tr>
{_FIXTURE_ROWS}
</table></body></html>
"""


def test_parse_sectors_reads_gics_columns():
    parsed = parse_sectors(FIXTURE_HTML)
    assert parsed["AAPL"] == {
        "sector": "Information Technology",
        "sub_industry": "Consumer Electronics",
    }
    assert parsed["JPM"]["sector"] == "Financials"
    assert len(parsed) == 503


def test_parse_sectors_rejects_wrong_table():
    with pytest.raises(SectorFetchError):
        parse_sectors("<html><body><p>no tables here</p></body></html>")


def test_parse_sectors_rejects_implausible_count():
    tiny = FIXTURE_HTML.replace("</table>", "", 1)  # unbalanced -> few rows
    with pytest.raises(SectorFetchError):
        parse_sectors("<html><body><table><tr><th>Symbol</th><th>GICS Sector</th></tr>"
                      "<tr><td>A</td><td>Energy</td></tr></table></body></html>")


def test_refresh_and_lookup_roundtrip(tmp_path):
    db = tmp_path / "m.db"
    summary = refresh_sectors(db, html=FIXTURE_HTML)
    assert summary["tickers"] == 503
    assert sector_of("aapl", db) == "Information Technology"
    assert sector_of("NOPE", db) is None
    batch = sectors_for(["AAPL", "JPM", "NOPE"], db)
    assert batch == {
        "AAPL": "Information Technology",
        "JPM": "Financials",
        "NOPE": None,
    }
    st = status(db)
    assert st["tickers"] == 503 and st["fetched_at"]


def test_refresh_is_idempotent(tmp_path):
    db = tmp_path / "m.db"
    refresh_sectors(db, html=FIXTURE_HTML)
    refresh_sectors(db, html=FIXTURE_HTML)
    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM gics_sectors").fetchone()[0]
    assert n == 503


def test_gics_sector_names_are_canonical():
    assert len(GICS_SECTORS) == 11
    assert "Information Technology" in GICS_SECTORS
    assert "Financials" in GICS_SECTORS
