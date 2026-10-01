"""Smoke test for the Yahoo Finance downloader (Phase 1)."""

import pandas as pd
import pytest
import requests

from qrp.data.yahoo import EXPECTED_COLUMNS, fetch_daily_bars


def test_fetch_aapl_daily_bars():
    try:
        df = fetch_daily_bars("aapl", "2024-01-02", "2024-01-10")
    except requests.RequestException as exc:
        pytest.skip(f"network unavailable for Yahoo test: {exc}")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
    # Dates are YYYY-MM-DD strings, oldest first.
    assert df["Date"].iloc[0] <= df["Date"].iloc[-1]
    assert df["Close"].notna().all()
