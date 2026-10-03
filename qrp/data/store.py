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
"""

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "market.db"


def init_db(path: str | Path = DEFAULT_DB) -> Path:
    """Create the database file and daily_bars table if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA)
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
