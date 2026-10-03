"""Download company fundamentals from the SEC EDGAR companyfacts API (free, no key).

What "fundamentals" are, in plain English
------------------------------------------
Fundamentals are the numbers inside a company's official financial reports:
how much money it made (net income), how much it sold (revenue), what it owns
(assets), what it owes (debt), how many shares exist, and so on. They come
from the reports every US public company must file with the SEC (the
government agency that oversees the stock market) — mostly the annual 10-K
and quarterly 10-Q filings.

Why filing dates matter (the no-peeking rule)
---------------------------------------------
This is the single most important rule in this module: a number is only
"known" on the day the company actually *filed* it with the SEC, not on the
day the business period ended. Example: Apple's fiscal year ends in
September, but it files its annual report in late October / early November.
A strategy pretending to look at "2023 earnings" in March 2024 may only use
numbers filed on or before that March date. Using the final, possibly
restated numbers before they were filed would be peeking at the future —
backtests built that way lie. Every fact stored here therefore carries its
real SEC filing date, and the query helpers only return facts whose filing
date is on or before the date you ask about.

Endpoints used (all free, no API key, no login):
    Ticker map:    https://www.sec.gov/files/company_tickers.json
    Company facts: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

The SEC asks callers to send a descriptive User-Agent header and stay under
about 10 requests per second; this module does both.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from qrp.data.yahoo import normalize_ticker
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = (
    "QuantResearchPlatform/1.0 (personal research project; "
    "contact: research@example.com)"
)
REQUEST_DELAY_S = 0.2  # stay well under the SEC's ~10 requests/second guidance

CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "company_tickers.json"

# Canonical tag -> XBRL tags to try, in order. Filers use inconsistent tags,
# so for each reporting period we take the first tag in the list that has a
# fact for that period (this also handles companies that switch tags over time).
TAG_FALLBACKS: dict[str, list[str]] = {
    "net_income": ["NetIncomeLoss"],
    "revenue": [
        "RevenueFromContractWithCustomersExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "debt_longterm": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cashflow_ops": ["NetCashProvidedByUsedInOperatingActivities"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "ebit": ["OperatingIncomeLoss"],
    # lives in the "dei" namespace, not "us-gaap"
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "dividends_per_share": ["CommonStockDividendsPerShareDeclared"],
}
# Canonical tags whose XBRL tags live in the "dei" namespace.
DEI_TAGS = {"shares_outstanding"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker     TEXT NOT NULL,
    tag        TEXT NOT NULL,
    period_end TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    form       TEXT,
    fp         TEXT,
    unit       TEXT,
    value      REAL NOT NULL,
    -- Append-only: every filing keeps its own row, keyed by when it was
    -- filed. A restatement therefore never erases the original filing, so
    -- a query "as of" a date between the two sees exactly what a real
    -- investor knew then (see the no-peeking rule in the module docstring).
    PRIMARY KEY (ticker, tag, period_end, filed_date)
);
"""

# When each ticker's fundamentals were last successfully downloaded, so
# repeat runs only fetch filings newer than the last download instead of
# re-downloading everything. Plain English: "only fetch what's new since
# last time". last_fetch is a UTC ISO-8601 timestamp like
# "2026-10-03T14:22:31.123456+00:00".
FETCH_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS edgar_fetch_state (
    ticker     TEXT PRIMARY KEY,
    last_fetch TEXT NOT NULL
);
"""

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "market.db"


def _pk_columns(conn: sqlite3.Connection) -> list[str]:
    """Primary-key column names of the fundamentals table (empty if none)."""
    rows = conn.execute("PRAGMA table_info(fundamentals)").fetchall()
    # row layout: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows if r[5] > 0]


def fundamentals_schema_is_current(path: str | Path = DEFAULT_DB) -> bool:
    """True when there is no fundamentals table yet, or it is already in
    the append-only format (primary key includes filed_date).

    The dashboard checks this *before* ingesting so it can warn in plain
    English inside the UI. init_db() still does the actual rebuild — its
    printed notice only reaches the server log, which the user never sees.
    """
    path = Path(path)
    if not path.exists():
        return True
    with sqlite3.connect(path) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type = 'table' AND name = 'fundamentals'"
        ).fetchone()
        if not has_table:
            return True
        return "filed_date" in _pk_columns(conn)


def init_db(path: str | Path = DEFAULT_DB) -> Path:
    """Create the database file and fundamentals table if they do not exist.

    If the table exists in the old replace-style format (primary key
    without filed_date), it is rebuilt in the new append-only format. The
    old rows cannot be salvaged — restatements had already overwritten the
    originals — so they are cleared and re-downloaded on the next ingest
    (SEC data is free and always re-fetchable). A plain-English notice
    says so, with the exact command to rebuild.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type = 'table' AND name = 'fundamentals'"
        ).fetchone()
        if has_table and "filed_date" not in _pk_columns(conn):
            conn.execute("DROP TABLE fundamentals")
            print(
                "Note: your saved company financials used an older format, "
                "so they were cleared and will be downloaded fresh on this "
                "run — nothing else in the database was touched. To rebuild "
                "them yourself later, run: python -m qrp.data.edgar_cli "
                f"--tickers YOUR,TICKERS --db {path}"
            )
        conn.execute(SCHEMA)
        conn.execute(FETCH_STATE_SCHEMA)
    return path


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    return sess


# Re-download the ticker→CIK map when the cache is older than this, so ticker
# renames (and newly listed companies) are picked up automatically instead of
# failing until someone manually deletes the cache file.
TICKER_MAP_MAX_AGE_DAYS = 30


def load_ticker_map(cache_path: str | Path = CACHE_FILE) -> dict[str, str]:
    """Return {TICKER: zero-padded-10-digit-CIK}, downloading when missing or stale.

    The SEC publishes one JSON file mapping every US-listed ticker to its
    CIK (the SEC's ID number for the company). It is cached locally and
    re-downloaded when the cache is older than TICKER_MAP_MAX_AGE_DAYS, so
    ticker renames and new listings are picked up without manual cleanup.
    """
    cache_path = Path(cache_path)
    stale = cache_path.exists() and (
        time.time() - cache_path.stat().st_mtime > TICKER_MAP_MAX_AGE_DAYS * 86400
    )
    if not cache_path.exists() or stale:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = _session().get(TICKER_MAP_URL, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
    raw = json.loads(cache_path.read_text())
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).strip().upper()
        cik = str(entry.get("cik_str", "")).strip()
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    return mapping


def _find_cik(mapping: dict[str, str], ticker: str) -> str | None:
    """Look up a ticker's CIK in an already-loaded map (with variants).

    Tries the ticker as written plus dot/dash variants ("BRK.B" vs
    "BRK-B"), because different sources spell share classes differently.
    """
    t = normalize_ticker(ticker)
    for variant in (t, t.replace("-", ".")):
        if variant in mapping:
            return mapping[variant]
    return None


def cik_for_ticker(
    ticker: str, cache_path: str | Path = CACHE_FILE
) -> str | None:
    """Return the 10-digit CIK for a ticker, or None if it is not listed."""
    return _find_cik(load_ticker_map(cache_path), ticker)


def fetch_companyfacts(
    cik: str, session: requests.Session | None = None, timeout: int = 60
) -> dict:
    """Download the full companyfacts JSON for one CIK from SEC EDGAR."""
    sess = session or _session()
    url = COMPANYFACTS_URL.format(cik=cik)
    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _coerce_fact(entry: dict) -> tuple[str, str, str, str, str, float] | None:
    """Pull (end, filed, form, fp, unit-agnostic, value) from one XBRL fact.

    Returns None when the entry is missing anything essential.
    """
    try:
        end = str(entry["end"])
        filed = str(entry["filed"])
        value = float(entry["val"])
    except (KeyError, TypeError, ValueError):
        return None
    form = str(entry.get("form", ""))
    fp = str(entry.get("fp", ""))
    return end, filed, form, fp, value


def extract_facts(payload: dict) -> pd.DataFrame:
    """Turn a companyfacts JSON payload into a tidy facts DataFrame.

    Columns: tag (canonical name), period_end, filed_date, form, fp, unit,
    value — one row per (tag, period_end, filed_date), i.e. one row per
    *filing*, not per period (append-only storage).

    Rules applied, in order, within each individual filing:
      1. Tag fallback: the first XBRL tag in TAG_FALLBACKS with a fact for
         that period wins.
      2. Annual preference: if both an annual (10-K, including amended
         10-K/A restatements) fact and other facts were filed together for
         the same period, the 10-K fact wins; otherwise quarterly (10-Q)
         facts fill the gaps.

    Restatements are NOT collapsed: the original filing and the later
    restatement become separate rows with their own filing dates. Queries
    "as of" a date then see exactly what was knowable then — the original
    before the restatement was filed, the restated number after (see the
    module docstring on the no-peeking rule).
    """
    namespaces = payload.get("facts", {})
    rows: list[dict] = []
    for canonical, xbrl_tags in TAG_FALLBACKS.items():
        ns = "dei" if canonical in DEI_TAGS else "us-gaap"
        ns_facts = namespaces.get(ns, {})
        # Gather every candidate fact across the fallback tags.
        candidates: list[tuple[str, str, str, str, str, str, float]] = []
        for xbrl_tag in xbrl_tags:
            tag_node = ns_facts.get(xbrl_tag, {})
            for unit, entries in tag_node.get("units", {}).items():
                for entry in entries or []:
                    parsed = _coerce_fact(entry)
                    if parsed is None:
                        continue
                    end, filed, form, fp, value = parsed
                    candidates.append(
                        (xbrl_tag, end, filed, form, fp, unit, value)
                    )
        # Group by (reporting period, filing date): what each individual
        # filing reported for each period. Two filings covering the same
        # period (original + restatement, or 10-Q then 10-K) stay separate
        # rows — the query helpers pick the latest filing knowable as of
        # the requested date.
        by_filing: dict[tuple[str, str], list] = {}
        for cand in candidates:
            by_filing.setdefault((cand[1], cand[2]), []).append(cand)
        for (end, filed), group in by_filing.items():
            chosen = None
            for xbrl_tag in xbrl_tags:  # fallback order
                tagged = [c for c in group if c[0] == xbrl_tag]
                if not tagged:
                    continue
                # Annual reports win; 10-K/A is an amended annual report
                # (a restatement) and counts as annual too.
                annual = [c for c in tagged if c[3] in ("10-K", "10-K/A")]
                pool = annual or tagged
                chosen = pool[0]
                break
            if chosen is None:
                continue
            _, _, _, form, fp, unit, value = chosen
            rows.append(
                {
                    "tag": canonical,
                    "period_end": end,
                    "filed_date": filed,
                    "form": form,
                    "fp": fp,
                    "unit": unit,
                    "value": value,
                }
            )
    df = pd.DataFrame(
        rows,
        columns=["tag", "period_end", "filed_date", "form", "fp", "unit", "value"],
    )
    if not df.empty:
        df = df.sort_values(["tag", "period_end", "filed_date"]).reset_index(
            drop=True
        )
    return df


def upsert_facts(
    df: pd.DataFrame, ticker: str, path: str | Path = DEFAULT_DB
) -> int:
    """Insert or replace fundamental facts for one ticker. Returns rows written."""
    path = init_db(path)
    ticker = normalize_ticker(ticker)
    rows = [
        (
            ticker,
            r.tag,
            r.period_end,
            r.filed_date,
            r.form or None,
            r.fp or None,
            r.unit or None,
            float(r.value),
        )
        for r in df.itertuples()
    ]
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def facts_as_of(
    ticker: str,
    tag: str,
    asof: str,
    path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Return one tag's facts knowable on or before `asof` (YYYY-MM-DD).

    This is the point-in-time query: a fact is only returned when its SEC
    filing date is on or before `asof`. When several filings cover the
    same period (e.g. an original 10-K and a later 10-K/A restatement),
    only the latest filing knowable as of `asof` is returned — so a query
    between the two sees the original, and a query after sees the
    restatement. No peeking at the future, ever.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["tag", "period_end", "filed_date", "form", "fp", "unit", "value"]
        )
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT tag, period_end, filed_date, form, fp, unit, value FROM ("
            "  SELECT *, ROW_NUMBER() OVER ("
            "    PARTITION BY tag, period_end ORDER BY filed_date DESC"
            "  ) AS rn FROM fundamentals"
            "  WHERE ticker = ? AND tag = ? AND filed_date <= ?"
            ") WHERE rn = 1 ORDER BY period_end",
            conn,
            params=(normalize_ticker(ticker), tag, asof),
        )
    return df


def latest_facts(
    ticker: str, asof: str, path: str | Path = DEFAULT_DB
) -> pd.DataFrame:
    """Return every stored tag for a ticker as knowable on or before `asof`.

    One row per (tag, period): the latest filing knowable as of `asof`.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["tag", "period_end", "filed_date", "form", "fp", "unit", "value"]
        )
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT tag, period_end, filed_date, form, fp, unit, value FROM ("
            "  SELECT *, ROW_NUMBER() OVER ("
            "    PARTITION BY tag, period_end ORDER BY filed_date DESC"
            "  ) AS rn FROM fundamentals"
            "  WHERE ticker = ? AND filed_date <= ?"
            ") WHERE rn = 1 ORDER BY tag, period_end",
            conn,
            params=(normalize_ticker(ticker), asof),
        )
    return df


def count_facts(ticker: str, path: str | Path = DEFAULT_DB) -> int:
    """Return how many fundamental facts are stored for one ticker."""
    path = Path(path)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM fundamentals WHERE ticker = ?",
            (normalize_ticker(ticker),),
        ).fetchone()
    return n


def fact_date_range(
    ticker: str, path: str | Path = DEFAULT_DB
) -> tuple[str | None, str | None]:
    """Return (earliest period_end, latest period_end) stored for a ticker."""
    path = Path(path)
    if not path.exists():
        return None, None
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT MIN(period_end), MAX(period_end) FROM fundamentals"
            " WHERE ticker = ?",
            (normalize_ticker(ticker),),
        ).fetchone()
    return row[0], row[1]


def get_last_fetch(ticker: str, path: str | Path = DEFAULT_DB) -> str | None:
    """Return when a ticker's fundamentals were last downloaded successfully.

    The answer is a UTC timestamp in ISO-8601 format (for example
    "2026-10-03T14:22:31.123456+00:00"), or None when this ticker has
    never been downloaded — meaning "everything is new, fetch it all".
    Tickers are normalized the same way as the facts table ("BRK.B" and
    "BRK-B" share one record).
    """
    path = init_db(path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT last_fetch FROM edgar_fetch_state WHERE ticker = ?",
            (normalize_ticker(ticker),),
        ).fetchone()
    return row[0] if row else None


def set_last_fetch(
    ticker: str, path: str | Path = DEFAULT_DB, when: str | None = None
) -> str:
    """Record a successful download for one ticker.

    `when` is a UTC ISO-8601 timestamp; leave it out and "right now" in
    UTC is used. Any previous record for the ticker is overwritten.
    Returns the stored timestamp (handy for tests, and for backdating
    the record in scripts — e.g. pretending the last download was a
    year ago so the next run re-fetches recent filings).
    """
    path = init_db(path)
    if when is None:
        when = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO edgar_fetch_state (ticker, last_fetch)"
            " VALUES (?, ?)",
            (normalize_ticker(ticker), when),
        )
    return when


def clear_fetch_state(
    tickers: list[str] | None, path: str | Path = DEFAULT_DB
) -> int:
    """Forget when tickers were last downloaded.

    Pass a list of tickers to forget just those, or None to forget every
    ticker. The next ingest then treats everything as new and does a full
    re-download. Only the "when did I last download" record is removed —
    the stored facts are NOT deleted. Returns how many records were
    removed.
    """
    path = init_db(path)
    if tickers is not None and len(tickers) == 0:
        return 0
    with sqlite3.connect(path) as conn:
        if tickers is None:
            cur = conn.execute("DELETE FROM edgar_fetch_state")
        else:
            cur = conn.execute(
                "DELETE FROM edgar_fetch_state WHERE ticker IN (%s)"
                % ",".join("?" for _ in tickers),
                [normalize_ticker(t) for t in tickers],
            )
    return cur.rowcount


def last_fetch_summary(path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """One row per ticker ever downloaded: ticker, last_fetch, facts.

    `last_fetch` is the UTC ISO-8601 timestamp of the last successful
    download; `facts` is how many fundamental facts are currently stored
    for the ticker (0 when the download record exists but no facts do).
    Built for the dashboard's "data last updated" display. Sorted by
    ticker. Empty (no rows) when nothing has been downloaded yet.
    """
    path = init_db(path)
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT s.ticker AS ticker, s.last_fetch AS last_fetch,"
            " COUNT(f.ticker) AS facts"
            " FROM edgar_fetch_state s"
            " LEFT JOIN fundamentals f ON f.ticker = s.ticker"
            " GROUP BY s.ticker, s.last_fetch"
            " ORDER BY s.ticker",
            conn,
        )
    return df


def ingest_ticker(
    ticker: str,
    path: str | Path = DEFAULT_DB,
    session: requests.Session | None = None,
    cache_path: str | Path = CACHE_FILE,
    ticker_map: dict[str, str] | None = None,
    incremental: bool = True,
) -> int:
    """Download and store fundamentals for one ticker. Returns NEW facts stored.

    By default (incremental=True) only filings *filed after* the ticker's
    last successful download are stored — in plain English: "only fetch
    what's new since last time". This makes repeat runs fast: re-fetching
    can't create duplicates anyway (the table key includes the filing
    date), but skipping already-seen filings avoids re-writing thousands
    of unchanged rows.

    Pass incremental=False for a full ingest (every filing, as before);
    the "last downloaded" record is still updated either way, so the next
    incremental run starts from this download. On a ticker with no
    previous download, incremental mode stores everything — there is
    nothing to compare against yet.

    The "last downloaded" record is only updated when the download and
    storage succeed. A failed ticker keeps its old record (or none), so
    the next run retries it from scratch instead of skipping it.

    Pass `ticker_map` (from load_ticker_map) and `session` when ingesting
    many tickers in a loop, so the ~1MB ticker map is parsed once and the
    HTTP connection is reused instead of re-read per ticker.

    Raises ValueError when the ticker has no SEC filing record, and
    requests exceptions on network/HTTP failures.
    """
    t = normalize_ticker(ticker)
    mapping = ticker_map if ticker_map is not None else load_ticker_map(cache_path)
    cik = _find_cik(mapping, t)
    if cik is None:
        raise ValueError(
            f"{t} has no SEC filing record (no CIK found) — it may be "
            "delisted, foreign, or not a real ticker."
        )
    payload = fetch_companyfacts(cik, session)
    df = extract_facts(payload)
    if df.empty:
        raise ValueError(f"SEC returned no usable facts for {t} (CIK {cik})")
    if incremental:
        last = get_last_fetch(t, path)
        if last is not None:
            # filed_date is stored as "YYYY-MM-DD" text, so a plain string
            # comparison correctly keeps only the newer filings.
            df = df[df["filed_date"] > last[:10]].reset_index(drop=True)
    # An empty frame writes 0 rows: still a success (nothing new to store).
    n = upsert_facts(df, t, path)
    set_last_fetch(t, path)
    return n
