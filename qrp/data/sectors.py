"""GICS sector data for the S&P 500 (free source: Wikipedia).

What this is: a small table mapping each current S&P 500 ticker to its
GICS sector (Information Technology, Financials, …) and sub-industry.
It powers things that need to compare stocks against their own industry:
the deferred accruals strategy (balance-sheet bloat ranks within sectors),
sector-neutral signal variants, and Brinson attribution later.

Source: Wikipedia's "List of S&P 500 companies", which carries GICS Sector
+ GICS Sub-Industry columns for every constituent. Same page the membership
checker already reads — no new dependency, no API key, no quota.

Honest limitation (point-in-time): Wikipedia gives CURRENT sectors only.
GICS assignments do change — the big one was September 2018, when Telecom
Services became Communication Services. Using today's labels on a 2015
backtest is therefore a mild look-ahead bias: rare but real. It belongs in
the platform's plain-English realism footnote next to the bankruptcy
limitation, and any strategy that ranks within sectors must disclose it.
Tickers that are not current members (delisted names, historical-only
tickers) are simply absent from the table — callers treat a miss as
"Unknown" and fall back to universe-wide ranking.

Refresh cadence: sectors change a few times a year at most. Refresh from the
Market data page (or ``python -m qrp.data.sectors_cli``) whenever you refresh
the membership list — monthly is plenty.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from qrp.data import sp500_update as _sp500u
from qrp.data.store import DEFAULT_DB

SECTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS gics_sectors (
    ticker       TEXT PRIMARY KEY,
    sector       TEXT NOT NULL,
    sub_industry TEXT,
    fetched_at   TEXT NOT NULL
);
"""

# The 11 canonical GICS sectors. Used to sanity-check a fresh parse: a page
# that returns only 3 distinct sectors is a layout change, not data.
GICS_SECTORS = {
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
}


class SectorFetchError(RuntimeError):
    """The sector table couldn't be refreshed — local data is unchanged."""


def parse_sectors(html: str) -> dict[str, dict[str, str]]:
    """Wikipedia constituent HTML -> {ticker: {sector, sub_industry}}.

    Raises SectorFetchError when the table can't be found or looks wrong
    (Wikipedia layout change) — never returns a partial guess.
    """
    parser = _sp500u._TableParser()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        header = [c.lower() for c in table[0]]
        if "symbol" not in header or "gics sector" not in header:
            continue
        sym_i = header.index("symbol")
        sec_i = header.index("gics sector")
        sub_i = header.index("gics sub-industry") if "gics sub-industry" in header else None
        result: dict[str, dict[str, str]] = {}
        for row in table[1:]:
            if len(row) <= max(sym_i, sec_i):
                continue
            as_listed = row[sym_i].strip()
            sector = row[sec_i].strip()
            if not as_listed or not sector:
                continue
            norm = _sp500u._builder.normalize(as_listed)
            result[norm] = {
                "sector": sector,
                "sub_industry": row[sub_i].strip() if sub_i is not None and len(row) > sub_i else "",
            }
        if not (495 <= len(result) <= 510):
            raise SectorFetchError(
                "Wikipedia's S&P 500 table doesn't look the way we expected "
                f"(found {len(result)} tickers, expected about 503). The page "
                "layout may have changed — your sector data is unchanged."
            )
        seen = {v["sector"] for v in result.values()}
        if len(seen) < 8:
            raise SectorFetchError(
                f"Only {len(seen)} distinct sectors parsed — the GICS column "
                "may have moved. Your sector data is unchanged."
            )
        return result
    raise SectorFetchError(
        "Couldn't find the S&P 500 constituent table with a GICS sector "
        "column on Wikipedia's page — the layout may have changed. Your "
        "sector data is unchanged."
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SECTORS_SCHEMA)


def refresh_sectors(
    db_path: str | Path = DEFAULT_DB,
    html: str | None = None,
) -> dict:
    """Fetch the current GICS table and store it. Returns a summary dict.

    Pass html (for tests) to skip the download. Raises SectorFetchError on
    any problem — the existing table is left untouched.
    """
    if html is None:
        html = _sp500u._fetch_page(_sp500u.WIKI_LIST_URL)
        if not html:
            raise SectorFetchError(
                "Couldn't download Wikipedia's S&P 500 page — check your "
                "connection and try again. Your sector data is unchanged."
            )
    parsed = parse_sectors(html)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO gics_sectors "
            "(ticker, sector, sub_industry, fetched_at) VALUES (?, ?, ?, ?)",
            [
                (t, v["sector"], v["sub_industry"], fetched_at)
                for t, v in parsed.items()
            ],
        )
    return {"tickers": len(parsed), "fetched_at": fetched_at}


def sector_of(ticker: str, db_path: str | Path = DEFAULT_DB) -> str | None:
    """GICS sector for one ticker, or None when unknown (not a current
    member, or the table was never refreshed)."""
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT sector FROM gics_sectors WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
    return row[0] if row else None


def sectors_for(
    tickers: list[str], db_path: str | Path = DEFAULT_DB
) -> dict[str, str | None]:
    """{ticker: sector or None} for a batch (one query, not one per ticker)."""
    tickers = [t.upper() for t in tickers]
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT ticker, sector FROM gics_sectors WHERE ticker IN (%s)"
            % ",".join("?" * len(tickers)),
            tickers,
        ).fetchall()
    found = dict(rows)
    return {t: found.get(t) for t in tickers}


def status(db_path: str | Path = DEFAULT_DB) -> dict:
    """How much sector data we hold: {'tickers': n, 'fetched_at': iso|None}."""
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        n = conn.execute("SELECT COUNT(*) FROM gics_sectors").fetchone()[0]
        row = conn.execute(
            "SELECT MAX(fetched_at) FROM gics_sectors"
        ).fetchone()
    return {"tickers": n, "fetched_at": row[0] if row else None}


def is_financial_or_utility(ticker: str, db_path: str | Path = DEFAULT_DB) -> bool:
    """True for Financials/Utilities members — the sectors Greenblatt's
    Magic Formula excludes (enterprise value is meaningless for banks)."""
    return sector_of(ticker, db_path) in ("Financials", "Utilities")
