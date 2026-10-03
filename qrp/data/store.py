"""SQLite storage for ingested daily bars.

Database file lives at <project-root>/data/market.db by default (git-ignored).
Table:
    daily_bars(ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
               close REAL, volume INTEGER, PRIMARY KEY(ticker, date))
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sqlite3
from qrp.data.yahoo import normalize_ticker

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS dividends (
    ticker  TEXT NOT NULL,
    ex_date TEXT NOT NULL,
    amount  REAL NOT NULL,
    PRIMARY KEY (ticker, ex_date)
);
"""

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "market.db"


def init_db(path: str | Path = DEFAULT_DB) -> Path:
    """Create the database file and tables if they do not exist.

    The dividends table is created here too, so existing databases gain
    it automatically on the next ingest (plain CREATE TABLE IF NOT EXISTS
    migration — no data is ever touched).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


def upsert_bars(df: pd.DataFrame, ticker: str, path: str | Path = DEFAULT_DB) -> int:
    """Insert or replace daily bars for one ticker. Returns rows written."""
    path = init_db(path)
    ticker = normalize_ticker(ticker)
    rows = [
        (
            ticker,
            row.Date,
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            int(row.Volume),
        )
        for row in df.itertuples()
    ]
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_bars"
            " (ticker, date, open, high, low, close, volume)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def count_bars(ticker: str, path: str | Path = DEFAULT_DB) -> int:
    """Return how many bars are stored for one ticker."""
    path = Path(path)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM daily_bars WHERE ticker = ?",
            (normalize_ticker(ticker),),
        ).fetchone()
    return n


def upsert_dividends(
    df: pd.DataFrame, ticker: str, path: str | Path = DEFAULT_DB
) -> int:
    """Store cash dividends for one ticker. Returns rows written.

    df has columns ex_date (YYYY-MM-DD) and amount (dollars per share).
    INSERT OR REPLACE makes re-running a download idempotent: dividends
    already stored are simply rewritten, never duplicated — so backfilling
    an older start date only adds the newly seen ex-dates.
    """
    path = init_db(path)
    ticker = normalize_ticker(ticker)
    if df is None or df.empty:
        return 0
    rows = [
        (ticker, str(row.ex_date), float(row.amount))
        for row in df.itertuples()
        if float(row.amount) > 0
    ]
    if not rows:
        return 0
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO dividends (ticker, ex_date, amount)"
            " VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def load_dividends(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Cash dividends as a DataFrame: ticker, ex_date (datetime), amount.

    Empty DataFrame (same columns) when none are stored — callers treat
    "no dividends" as "nothing to credit", never an error.
    """
    path = Path(path)
    cols = ["ticker", "ex_date", "amount"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    tickers = [normalize_ticker(t) for t in tickers]
    if not tickers:
        return pd.DataFrame(columns=cols)
    placeholders = ",".join("?" for _ in tickers)
    query = (
        "SELECT ticker, ex_date, amount FROM dividends"
        f" WHERE ticker IN ({placeholders})"
    )
    params: list = list(tickers)
    if start:
        query += " AND ex_date >= ?"
        params.append(start)
    if end:
        query += " AND ex_date <= ?"
        params.append(end)
    with sqlite3.connect(path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='dividends'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return pd.DataFrame(columns=cols)
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df.sort_values(["ex_date", "ticker"]).reset_index(drop=True)
