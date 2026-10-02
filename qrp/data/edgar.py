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
from pathlib import Path

import pandas as pd
import requests

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

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "market.db"


def _pk_columns(conn: sqlite3.Connection) -> list[str]:
    """Primary-key column names of the fundamentals table (empty if none)."""
    rows = conn.execute("PRAGMA table_info(fundamentals)").fetchall()
    # row layout: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows if r[5] > 0]


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
    return path


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    return sess


def load_ticker_map(cache_path: str | Path = CACHE_FILE) -> dict[str, str]:
    """Return {TICKER: zero-padded-10-digit-CIK}, downloading the map once.

    The SEC publishes one JSON file mapping every US-listed ticker to its
    CIK (the SEC's ID number for the company). It is cached locally so we
    only download it the first time.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
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
    t = ticker.strip().upper()
    for variant in (t, t.replace(".", "-"), t.replace("-", ".")):
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
    ticker = ticker.strip().upper()
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
            params=(ticker.strip().upper(), tag, asof),
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
            params=(ticker.strip().upper(), asof),
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
            (ticker.strip().upper(),),
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
            (ticker.strip().upper(),),
        ).fetchone()
    return row[0], row[1]


def ingest_ticker(
    ticker: str,
    path: str | Path = DEFAULT_DB,
    session: requests.Session | None = None,
    cache_path: str | Path = CACHE_FILE,
    ticker_map: dict[str, str] | None = None,
) -> int:
    """Download and store fundamentals for one ticker. Returns facts stored.

    Pass `ticker_map` (from load_ticker_map) and `session` when ingesting
    many tickers in a loop, so the ~1MB ticker map is parsed once and the
    HTTP connection is reused instead of re-read per ticker.

    Raises ValueError when the ticker has no SEC filing record, and
    requests exceptions on network/HTTP failures.
    """
    t = ticker.strip().upper()
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
    return upsert_facts(df, t, path)
