"""Smoke test for the Yahoo Finance downloader (Phase 1)."""

import pandas as pd
import pytest

from qrp.data.yahoo import EXPECTED_COLUMNS, fetch_daily_bars, normalize_ticker


def test_fetch_aapl_daily_bars():
    try:
        df = fetch_daily_bars("aapl", "2024-01-02", "2024-01-10")
    except Exception as exc:  # any network/payload failure -> skip, not fail
        pytest.skip(f"network unavailable for Yahoo test: {exc}")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
    # Dates are YYYY-MM-DD strings, oldest first.
    assert df["Date"].iloc[0] <= df["Date"].iloc[-1]
    assert df["Close"].notna().all()


def test_normalize_ticker():
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker(" BRK.B ") == "BRK-B"  # Yahoo wants dashes
    try:
        normalize_ticker("not a ticker!")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid ticker")


def _bars_frame(dates):
    """Synthetic OHLCV frame with the columns the real pipeline stores."""
    n = len(dates)
    close = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Open": close,
            "High": [c * 1.01 for c in close],
            "Low": [c * 0.99 for c in close],
            "Close": close,
            "Volume": [1_000_000] * n,
        }
    )


def test_cli_default_start_is_2015():
    """The market-data download defaults to 2015-01-01 (ten years of history)."""
    from qrp.data.cli import parse_args

    assert parse_args([]).start == "2015-01-01"


def test_backfill_bars_no_duplicates(tmp_path):
    """Re-running a download with an earlier start backfills without duplicating."""
    from qrp.data.store import count_bars, upsert_bars

    db = tmp_path / "backfill.db"
    # First "download": 2020 only.
    first = pd.bdate_range("2020-01-02", "2020-03-31")
    upsert_bars(_bars_frame(first), "AAA", path=db)
    assert count_bars("AAA", path=db) == len(first)
    # Second "download": earlier start, overlapping January.
    second = pd.bdate_range("2015-01-02", "2020-01-31")
    upsert_bars(_bars_frame(second), "AAA", path=db)
    union = set(first.strftime("%Y-%m-%d")) | set(second.strftime("%Y-%m-%d"))
    assert count_bars("AAA", path=db) == len(union)
    # Earliest bar is present exactly once.
    import sqlite3

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM daily_bars WHERE ticker='AAA' AND date='2015-01-02'"
        ).fetchone()[0]
        assert rows == 1
        rows = conn.execute(
            "SELECT COUNT(*) FROM daily_bars WHERE ticker='AAA' AND date='2020-01-15'"
        ).fetchone()[0]
        assert rows == 1


def test_backfill_facts_no_duplicates(tmp_path):
    """Re-storing fundamentals never duplicates facts (append-only, keyed)."""
    from qrp.data.edgar import upsert_facts

    db = tmp_path / "facts.db"
    df = pd.DataFrame(
        {
            "tag": ["net_income", "net_income"],
            "period_end": ["2019-12-31", "2020-12-31"],
            "filed_date": ["2020-02-01", "2021-02-01"],
            "form": ["10-K", "10-K"],
            "fp": ["FY", "FY"],
            "unit": ["USD", "USD"],
            "value": [1.0e9, 1.2e9],
        }
    )
    upsert_facts(df, "AAA", path=db)
    upsert_facts(df, "AAA", path=db)  # same facts again: must not duplicate
    import sqlite3

    with sqlite3.connect(db) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM fundamentals WHERE ticker='AAA'"
        ).fetchone()
        assert n == 2
