"""SQLite storage for computed signals.

Table:
    signals(ticker TEXT, date TEXT, signal TEXT, value REAL,
            PRIMARY KEY(ticker, date, signal))

Dates are YYYY-MM-DD strings so range filters work with plain comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sqlite3

from qrp.data.store import DEFAULT_DB

SCHEMA_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    signal TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (ticker, date, signal)
);
"""


def init_signals_table(path: str | Path = DEFAULT_DB) -> Path:
    """Create the database file and signals table if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA_SIGNALS)
    return path


def list_tickers(path: str | Path = DEFAULT_DB) -> list[str]:
    """Tickers present in daily_bars, sorted alphabetically."""
    path = Path(path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker"
        ).fetchall()
    return [r[0] for r in rows]


def load_bars(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Load daily bars for one ticker from SQLite.

    Args:
        ticker: US ticker symbol (case-insensitive).
        start/end: optional YYYY-MM-DD bounds (inclusive).
        path: SQLite file.

    Returns:
        DataFrame with a DatetimeIndex (oldest first) and columns
        open, high, low, close, volume. Empty DataFrame if nothing stored.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    query = "SELECT date, open, high, low, close, volume FROM daily_bars WHERE ticker = ?"
    params: list[str] = [ticker.strip().upper()]
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def upsert_signals(
    values: pd.Series,
    ticker: str,
    signal_name: str,
    path: str | Path = DEFAULT_DB,
) -> int:
    """Insert or replace signal values for one ticker. NaNs are skipped.

    Args:
        values: Series indexed by date (DatetimeIndex or YYYY-MM-DD strings).
        ticker: US ticker symbol.
        signal_name: registered signal name, e.g. "momentum".

    Returns:
        Number of rows written.
    """
    path = init_signals_table(path)
    ticker = ticker.strip().upper()
    rows = []
    for ts, val in values.items():
        if pd.isna(val):
            continue
        day = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
        rows.append((ticker, day, signal_name, float(val)))
    if not rows:
        return 0
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO signals (ticker, date, signal, value)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def count_signals(ticker: str, signal_name: str, path: str | Path = DEFAULT_DB) -> int:
    """Return how many signal values are stored for one ticker+signal."""
    path = Path(path)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE ticker = ? AND signal = ?",
            (ticker.strip().upper(), signal_name),
        ).fetchone()
    return n
