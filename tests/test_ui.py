"""Tests for the Phase 6 Streamlit UI helpers and app structure.

The full Streamlit app cannot be exercised interactively in tests, so this
suite checks (a) that qrp.ui.app imports cleanly and exposes all pages, and
(b) that the pure helper functions it relies on behave correctly.
"""

import importlib

import pytest

streamlit = pytest.importorskip("streamlit")

from qrp.risk.guards import RiskConfig
from qrp.ui import helpers as H


def test_app_imports_cleanly():
    app = importlib.import_module("qrp.ui.app")
    assert hasattr(app, "main")
    assert hasattr(app, "PAGES")
    expected_pages = {
        "Home",
        "Market data",
        "Signals",
        "Backtest",
        "Portfolio",
        "Paper trading",
        "Risk",
    }
    assert set(app.PAGES) == expected_pages
    for name, func in app.PAGES.items():
        assert callable(func), f"page {name!r} is not callable"
        assert func.__name__.startswith("page_"), (
            f"page function name mismatch for {name!r}"
        )
    # Every PAGES entry must reference a real page_* function on the module.
    for func in app.PAGES.values():
        assert getattr(app, func.__name__) is func


def test_fmt_money():
    assert H.fmt_money(12345.678) == "$12,345.68"
    assert H.fmt_money(0) == "$0.00"


def test_fmt_pct():
    assert H.fmt_pct(0.1234) == "+12.34%"
    assert H.fmt_pct(float("nan")) == "n/a"
    assert H.fmt_pct(1.0) == "+100.00%"
    assert H.fmt_pct(-0.5) == "-50.00%"


def test_parse_tickers():
    assert H.parse_tickers("aapl, MSFT , NVDA") == ["AAPL", "MSFT", "NVDA"]
    assert H.parse_tickers("  , , ") == []
    assert H.parse_tickers("BRK.B, tsla") == ["BRK.B", "TSLA"]


def test_risk_config_rows_covers_every_guard():
    rows = H.risk_config_rows(RiskConfig())
    settings = {r["Setting"] for r in rows}
    assert "Max weight per stock" in settings
    assert "Max number of stocks" in settings
    assert "Drawdown stop" in settings
    assert "Max dollars per order" in settings
    assert "Price safety margin" in settings
    for row in rows:
        assert row["Value"], row
        assert row["What it means"], row


def test_latest_scores_on_empty_db_raises(tmp_path):
    db = str(tmp_path / "empty.db")
    with pytest.raises(ValueError, match="No stored values"):
        H.latest_scores("momentum", ["AAPL"], db_path=db)


def test_ticker_summary_empty_db(tmp_path):
    db = str(tmp_path / "empty.db")
    df = H.ticker_summary(db_path=db)
    assert df.empty
    assert list(df.columns) == ["Ticker", "First day", "Last day", "Daily prices"]


def test_metric_display_formats():
    assert H.metric_display("final_value", 1_234_567.8) == "$1,234,567.80"
    assert H.metric_display("total_return", 0.5) == "+50.00%"
    assert H.metric_display("max_drawdown", 0.123) == "-12.30%"
    assert H.metric_display("num_trades", 7.0) == "7"
    assert H.metric_display("num_trades", float("nan")) == "n/a"
    assert H.metric_display("sharpe", float("nan")) == "n/a"
