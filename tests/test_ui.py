"""Tests for the Phase 6 Streamlit UI helpers and app structure.

The full Streamlit app cannot be exercised interactively in tests, so this
suite checks (a) that qrp.ui.app imports cleanly and exposes all pages, and
(b) that the pure helper functions it relies on behave correctly.
"""

import importlib

import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit")

from qrp.data.store import upsert_bars
from qrp.execution.paper import PaperBroker
from qrp.risk.guards import RiskConfig
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H


def _seed_paper_db(path) -> None:
    """Five trading days of flat prices for AAA ($100) and BBB ($50)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": base,
                    "High": base * 1.01,
                    "Low": base * 0.99,
                    "Close": base,
                    "Volume": 1_000,
                }
            ),
            ticker,
            path,
        )
    sig_dates = pd.date_range("2024-01-02", periods=5, freq="B")
    upsert_signals(
        pd.Series([0.5] * 5, index=sig_dates), "AAA", "momentum", path
    )
    upsert_signals(
        pd.Series([0.3] * 5, index=sig_dates), "BBB", "momentum", path
    )


def test_app_imports_cleanly():
    app = importlib.import_module("qrp.ui.app")
    assert hasattr(app, "main")
    assert hasattr(app, "PAGES")
    expected_pages = {
        "Home",
        "Market data",
        "Signals",
        "Strategies",
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
    # max_drawdown is stored negative (a fall from the peak), e.g. -0.123.
    assert H.metric_display("max_drawdown", -0.123) == "-12.30%"
    assert H.metric_display("num_trades", 7.0) == "7"
    assert H.metric_display("num_trades", float("nan")) == "n/a"
    assert H.metric_display("sharpe", float("nan")) == "n/a"


def test_latest_scores_asof_none_uses_latest_date(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)

    scores, day = H.latest_scores("momentum", ["AAA", "BBB"], None, db_path=db)
    assert day == "2024-01-08"
    assert scores == {"AAA": 0.5, "BBB": 0.3}

    scores, day = H.latest_scores(
        "momentum", ["AAA", "BBB"], "2024-01-04", db_path=db
    )
    assert day == "2024-01-04"
    assert scores == {"AAA": 0.5, "BBB": 0.3}


def test_preview_rebalance_orders_rows_and_notes(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)
    # Relaxed cap: this test is about preview rows/notes, not the 20% cap.
    broker = PaperBroker(
        db, starting_capital=100_000.0, cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0),
    )

    rows, notes = H.preview_rebalance_orders(
        broker, {"AAA": 0.5, "BBB": 0.5}, "2024-01-08", 100_000.0,
        cost_bps=0.0,
    )
    assert len(rows) == 2
    assert notes == []
    assert {r["Ticker"] for r in rows} == {"AAA", "BBB"}
    assert all(r["Safety check"].startswith("Pass") for r in rows)


def test_preview_notes_missing_price_ticker(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)
    broker = PaperBroker(db, starting_capital=100_000.0, cost_bps=0.0)

    rows, notes = H.preview_rebalance_orders(
        broker, {"AAA": 0.5, "DEL": 0.5}, "2024-01-08", 100_000.0,
        cost_bps=0.0,
    )
    # DEL has no price data: skipped from the rows, called out in a note.
    assert {r["Ticker"] for r in rows} == {"AAA"}
    assert any("DEL" in n and "price" in n for n in notes)


def test_cached_price_matrix_invalidates_on_db_change(tmp_path, monkeypatch):
    """The UI cache key includes the DB file's mtime: newly ingested data
    is never hidden behind a stale cache, and repeat reads hit the cache."""
    import os

    db = str(tmp_path / "u.db")
    _seed_paper_db(db)

    calls = []
    real = H.load_price_matrix

    def counting(tickers, **kwargs):
        calls.append(1)
        return real(tickers, **kwargs)

    monkeypatch.setattr(H, "load_price_matrix", counting)

    m1 = H.cached_load_price_matrix(["AAA"], db_path=db)
    assert len(calls) == 1
    m2 = H.cached_load_price_matrix(["AAA"], db_path=db)
    assert len(calls) == 1  # second read served from cache
    assert m1.equals(m2)

    # Simulate a fresh ingest by bumping the DB file's mtime.
    stat = os.stat(db)
    os.utime(db, (stat.st_atime, stat.st_mtime + 5))
    m3 = H.cached_load_price_matrix(["AAA"], db_path=db)
    assert len(calls) == 2  # cache miss: reloaded from the database
    assert m3.equals(m1)
