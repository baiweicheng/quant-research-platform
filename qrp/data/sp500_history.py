"""Point-in-time S&P 500 membership (survivorship-bias fix).

The backtest engine used to build its universe from today's ticker list, so
every historical backtest quietly benefited from knowing which companies
would survive — a +227 percentage-point flattery versus the real index over
2015-2026. This module answers "which tickers were index members on date T?"
from a membership table built out of Wikipedia's dated S&P 500 change log
(see qrp/data/build_sp500_history.py for the source, retrieval date and the
full method).

Tickers the table has never heard of (e.g. a user's custom list) are always
treated as eligible: the filter only constrains tickers known to the table.
If the CSV is absent, every helper degrades gracefully and the UI says so
plainly instead of silently producing survivorship-biased numbers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "sp500_history.csv"
SOURCE = "Wikipedia's 'Historical components of the S&P 500' change log"
RETRIEVED = "2026-10-04"
COVERAGE_START = "1976-07-01"
COVERAGE_END = "2026-09-21"


def history_available() -> bool:
    """True when the membership CSV is present."""
    return CSV_PATH.exists()


@lru_cache(maxsize=1)
def load_history() -> pd.DataFrame | None:
    """The membership table, or None when the CSV is absent.

    Columns: ticker, date_added, date_removed ("" = still a member).
    """
    if not history_available():
        return None
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    return df


def is_known(ticker: str) -> bool:
    """Whether the table has ever seen this ticker."""
    df = load_history()
    return df is not None and bool((df["ticker"] == ticker).any())


def members_on(day: pd.Timestamp | str) -> set[str]:
    """Tickers that were S&P 500 members on the given date.

    A stock removed with effective date R counts as a member only for
    dates strictly before R (d < date_removed), so it is ineligible on R
    itself — conservative by one day versus S&P's typical
    effective-after-close timing, and documented in
    build_sp500_history.py.
    """
    df = load_history()
    if df is None:
        return set()
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    mask = (df["date_added"] <= d) & (
        (df["date_removed"] == "") | (d < df["date_removed"])
    )
    return set(df.loc[mask, "ticker"])


def eligible_on(ticker: str, day: pd.Timestamp | str) -> bool:
    """Whether `ticker` may be held on `day`.

    Unknown tickers pass through unfiltered; known tickers must have been
    index members on that date.
    """
    if not is_known(ticker):
        return True
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    df = load_history()
    rows = df[df["ticker"] == ticker]
    return bool(
        ((rows["date_added"] <= d)
         & ((rows["date_removed"] == "") | (d < rows["date_removed"]))).any()
    )


def make_universe_filter(
    tickers: list[str],
) -> "callable | None":
    """Build the engine's per-date eligibility function, or None.

    Returns None when the history table is absent (callers then fall back
    to the static list and must show the survivorship-bias note).
    """
    if load_history() is None:
        return None
    known = {t for t in tickers if is_known(t)}

    def universe_filter(day: pd.Timestamp) -> set[str]:
        members = members_on(day)
        return {t for t in tickers if t not in known or t in members}

    return universe_filter


def all_historical_tickers() -> list[str]:
    """Every ticker the table has ever seen (for the download flow)."""
    df = load_history()
    if df is None:
        return []
    return sorted(df["ticker"].unique().tolist())


def coverage_note() -> str:
    """One plain-English paragraph describing the membership source."""
    n = len(all_historical_tickers())
    return (
        f"Index membership comes from {SOURCE} (retrieved {RETRIEVED}), "
        f"covering {COVERAGE_START} to {COVERAGE_END} across {n} tickers. "
        "On any backtest date, only stocks that were actually in the index "
        "on that date can be bought — the backtest never gets to peek at "
        "which companies survived. The log is comprehensive from about "
        "2007 on; earlier years are partial, which does not affect "
        "backtests (data here starts in 2013)."
    )


def missing_history_note() -> str:
    """Plain-English warning shown when the CSV is absent."""
    return (
        "The historical index-membership table is missing, so this backtest "
        "used today's stock list for the whole period. That flatters the "
        "results: it quietly benefits from knowing which companies survived "
        "(survivorship bias). Reinstall the data files or re-download to "
        "restore point-in-time membership."
    )
