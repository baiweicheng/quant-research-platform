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
