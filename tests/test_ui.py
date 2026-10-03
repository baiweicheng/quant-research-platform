"""Tests for the Phase 6 Streamlit UI helpers and app structure.

The full Streamlit app cannot be exercised interactively in tests, so this
suite checks (a) that qrp.ui.app imports cleanly and exposes all pages, and
(b) that the pure helper functions it relies on behave correctly.
"""

import importlib
import sqlite3

import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

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
        "Compare strategies",
        "Backtest",
        "Backtest wizard",
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
    # The two Wave 2 pages must be the expected functions by name.
    assert app.PAGES["Compare strategies"] is app.page_compare
    assert app.PAGES["Backtest wizard"] is app.page_wizard
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
    assert H.parse_tickers("BRK.B, tsla") == ["BRK-B", "TSLA"]


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


def test_load_sp500_tickers():
    tickers = H.load_sp500_tickers()
    assert len(tickers) == 503
    assert all(t and t == t.upper() for t in tickers)
    assert not any(t.startswith("#") for t in tickers)
    assert "AAPL" in tickers
    assert "BRK.B" in tickers  # kept as listed; Yahoo side normalizes to BRK-B


def test_fundamentals_summary_empty_db(tmp_path):
    db = str(tmp_path / "empty.db")
    df = H.fundamentals_summary(db_path=db)
    assert df.empty
    assert list(df.columns) == [
        "Ticker", "Facts stored", "First period", "Last period",
    ]


def test_fundamentals_summary_with_data(tmp_path):
    import sqlite3

    from qrp.data.edgar import init_db

    db = str(tmp_path / "f.db")
    init_db(db)
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO fundamentals (ticker, tag, period_end, filed_date, value)"
            " VALUES (?,?,?,?,?)",
            [
                ("AAA", "revenue", "2023-12-31", "2024-02-01", 100.0),
                ("AAA", "revenue", "2024-12-31", "2025-02-01", 110.0),
                ("BBB", "assets", "2024-12-31", "2025-03-01", 500.0),
            ],
        )
    df = H.fundamentals_summary(db_path=db)
    assert list(df.columns) == [
        "Ticker", "Facts stored", "First period", "Last period",
    ]
    assert len(df) == 2
    aaa = df[df["Ticker"] == "AAA"].iloc[0]
    assert aaa["Facts stored"] == 2
    assert aaa["First period"] == "2023-12-31"
    assert aaa["Last period"] == "2024-12-31"


def test_fundamentals_schema_is_current(tmp_path):
    import sqlite3

    from qrp.data.edgar import fundamentals_schema_is_current, init_db

    assert fundamentals_schema_is_current(str(tmp_path / "nope.db")) is True
    db = str(tmp_path / "new.db")
    init_db(db)
    assert fundamentals_schema_is_current(db) is True
    old = str(tmp_path / "old.db")
    with sqlite3.connect(old) as conn:
        conn.execute(
            "CREATE TABLE fundamentals (ticker TEXT, tag TEXT, period_end TEXT,"
            " value REAL, PRIMARY KEY (ticker, tag, period_end))"
        )
    assert fundamentals_schema_is_current(old) is False


def test_download_market_data_prices_only(tmp_path, monkeypatch):
    db = str(tmp_path / "d.db")

    def fake_fetch(ticker, start, end):
        if ticker == "BAD":
            raise ValueError("no such ticker")
        return pd.DataFrame(
            {
                "Date": ["2024-01-02"],
                "Open": [1.0], "High": [1.0], "Low": [1.0],
                "Close": [1.0], "Volume": [100],
            }
        )

    monkeypatch.setattr(H, "fetch_daily_bars", fake_fetch)
    monkeypatch.setattr(H, "upsert_bars", lambda df, ticker, path: 5)

    rows, notices = H.download_market_data(
        ["AAA", "BAD"], "2024-01-01", "2024-01-31",
        download_prices=True, download_fundamentals=False, db_path=db,
    )
    assert len(rows) == 2
    assert rows[0]["Ticker"] == "AAA"
    assert rows[0]["Prices"].startswith("OK")
    assert rows[0]["Fundamentals"] == "—"
    # A failing ticker never aborts the run.
    assert rows[1]["Prices"].startswith("Failed:")
    assert notices == []


def _seed_all_strategy_signals(path) -> None:
    """_seed_paper_db plus the other signals the six strategies need."""
    _seed_paper_db(path)
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    for sig in ("trend", "low_volatility", "trending_value",
                "value_composite", "profitability"):
        upsert_signals(pd.Series([0.4] * 5, index=dates), "AAA", sig, path)
        upsert_signals(pd.Series([0.2] * 5, index=dates), "BBB", sig, path)


def test_run_backtest_for_ui_cost_model_spread(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)

    result, metrics = H.run_backtest_for_ui(
        "momentum", ["AAA", "BBB"], "equal", 0.5, "monthly",
        10.0, 100_000.0, "2024-01-02", "2024-01-08", db,
        cost_model="spread",
    )
    assert result.cost_model_name == "spread"
    assert metrics["total_return"]["value"] == metrics["total_return"]["value"]

    # The default stays the flat model — results match earlier versions.
    result_flat, _ = H.run_backtest_for_ui(
        "momentum", ["AAA", "BBB"], "equal", 0.5, "monthly",
        10.0, 100_000.0, "2024-01-02", "2024-01-08", db,
    )
    assert result_flat.cost_model_name == "flat"


def test_run_backtest_for_ui_cost_model_invalid(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)
    with pytest.raises(ValueError, match="unknown cost model"):
        H.run_backtest_for_ui(
            "momentum", ["AAA", "BBB"], "equal", 0.5, "monthly",
            10.0, 100_000.0, "2024-01-02", "2024-01-08", db,
            cost_model="fancy",
        )


def test_run_strategy_backtest_for_ui_cost_model_spread(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)

    result, _ = H.run_strategy_backtest_for_ui(
        "momentum", ["AAA", "BBB"], 10.0, 100_000.0,
        "2024-01-02", "2024-01-08", db, cost_model="spread",
    )
    assert result.cost_model_name == "spread"


def test_compare_strategies_for_ui_runs_all_strategies(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_all_strategy_signals(db)

    rows = H.compare_strategies_for_ui(
        ["AAA", "BBB"], "2024-01-02", "2024-01-08",
        100_000.0, 0.5, db_path=db,
    )
    assert len(rows) == 6
    expected_columns = {
        "Strategy", "Total return", "Yearly growth (CAGR)", "Sharpe",
        "Worst fall", "Trades",
    }
    names = set()
    for row in rows:
        assert expected_columns <= set(row)
        assert "total_return" in row and "sharpe" in row  # raw chart values
        names.add(row["Strategy"])
    assert len(names) == 6


def test_preview_rebalance_orders_includes_why(tmp_path):
    db = str(tmp_path / "u.db")
    _seed_paper_db(db)
    broker = PaperBroker(
        db, starting_capital=100_000.0, cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0),
    )

    rows, _ = H.preview_rebalance_orders(
        broker, {"AAA": 0.5, "BBB": 0.5}, "2024-01-08", 100_000.0,
        cost_bps=0.0,
    )
    assert len(rows) == 2
    for row in rows:
        assert "Why" in row
        assert row["Why"], row  # plain-English reason, never empty
        assert row["Ticker"] in row["Why"]


def test_fundamentals_fetch_summary_empty_db(tmp_path):
    db = str(tmp_path / "empty.db")
    df = H.fundamentals_fetch_summary(db_path=db)
    assert df.empty
    assert list(df.columns) == ["Ticker", "Last checked", "Facts stored"]


def test_fundamentals_fetch_summary_with_data(tmp_path):
    import sqlite3

    from qrp.data.edgar import init_db, set_last_fetch

    db = str(tmp_path / "f.db")
    init_db(db)
    set_last_fetch("AAA", db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO fundamentals (ticker, tag, period_end, filed_date, value)"
            " VALUES (?,?,?,?,?)",
            ("AAA", "revenue", "2024-12-31", "2025-02-01", 100.0),
        )
    df = H.fundamentals_fetch_summary(db_path=db)
    assert list(df.columns) == ["Ticker", "Last checked", "Facts stored"]
    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "AAA"
    assert df.iloc[0]["Facts stored"] == 1
    assert df.iloc[0]["Last checked"], "last-check timestamp must be shown"


def test_download_market_data_fundamentals_old_schema_notice(
    tmp_path, monkeypatch
):
    import sqlite3

    db = str(tmp_path / "d.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE fundamentals (ticker TEXT, tag TEXT, period_end TEXT,"
            " value REAL, PRIMARY KEY (ticker, tag, period_end))"
        )
    monkeypatch.setattr(H, "load_ticker_map", lambda: {})
    monkeypatch.setattr(
        H, "ingest_ticker",
        lambda t, path, session=None, ticker_map=None: 10,
    )

    rows, notices = H.download_market_data(
        ["AAA"], None, None,
        download_prices=False, download_fundamentals=True, db_path=db,
    )
    assert len(notices) == 1
    assert "older format" in notices[0]
    assert "nothing else" in notices[0]
    assert rows[0]["Fundamentals"].startswith("OK")
    assert rows[0]["Prices"] == "—"


# -- friendly empty states (Fix 1) -------------------------------------------

def test_data_status_missing_file(tmp_path):
    s = H.data_status(tmp_path / "does-not-exist.db")
    assert s == {
        "has_prices": False,
        "has_signals": False,
        "has_fundamentals": False,
        "tickers": [],
    }


def test_data_status_empty_db_file(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    s = H.data_status(path)
    assert not s["has_prices"] and not s["has_signals"]
    assert not s["has_fundamentals"] and s["tickers"] == []


def test_data_status_tables_present_but_empty(tmp_path):
    path = tmp_path / "m.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_bars (ticker TEXT)")
        conn.execute("CREATE TABLE signals (ticker TEXT)")
        conn.execute("CREATE TABLE fundamentals (ticker TEXT)")
    s = H.data_status(path)
    assert not s["has_prices"] and not s["has_signals"]
    assert not s["has_fundamentals"] and s["tickers"] == []


def test_data_status_with_prices_and_signals(tmp_path):
    path = tmp_path / "m.db"
    _seed_paper_db(path)
    s = H.data_status(path)
    assert s["has_prices"] and s["has_signals"]
    assert not s["has_fundamentals"]
    assert s["tickers"] == ["AAA", "BBB"]


def test_data_status_with_fundamentals_only(tmp_path):
    path = tmp_path / "m.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE fundamentals (ticker TEXT, tag TEXT)")
        conn.execute("INSERT INTO fundamentals VALUES ('AAPL', 'x')")
    s = H.data_status(path)
    assert s["has_fundamentals"]
    assert not s["has_prices"] and not s["has_signals"]
    assert s["tickers"] == []


def test_require_data_rejects_unknown_need(tmp_path):
    with pytest.raises(ValueError, match="Unknown data need"):
        H.require_data("moon", db_path=tmp_path / "m.db")


# -- Fix 1 + Fix 2 through the real Streamlit app -----------------------------

def _app_without_data():
    """AppTest on a working dir whose data/market.db does not exist."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(str(root / "qrp" / "ui" / "app.py"))


def test_empty_state_no_traceback_on_signals_page():
    at = _app_without_data()
    at.session_state[H.NAV_PAGE_KEY] = "Signals"
    at.run()
    assert not at.exception
    assert any(
        "no market data" in str(i.value).lower() for i in at.info
    ), "expected the friendly empty-state message"
    assert any(b.label == "Go to Market data" for b in at.button)


def test_empty_state_button_jumps_to_market_data():
    at = _app_without_data()
    at.session_state[H.NAV_PAGE_KEY] = "Signals"
    at.run()
    at.button(key="require_data_goto_prices").click().run()
    assert not at.exception
    assert at.session_state[H.NAV_PAGE_KEY] == "Market data"


def test_sidebar_buttons_match_pages_and_navigate():
    import qrp.ui.app as app

    at = _app_without_data()
    at.run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    assert labels == list(app.PAGES.keys())
    # The active page renders as a filled (primary) button, the rest subtle.
    kinds = {b.label: b.proto.type for b in at.sidebar.button}
    assert kinds["Home"] == "primary"
    assert all(k == "secondary" for n, k in kinds.items() if n != "Home")
    at.sidebar.button(key="nav_Risk").click().run()
    assert not at.exception
    assert at.session_state[H.NAV_PAGE_KEY] == "Risk"


# -- full page sweep: no tracebacks on a fresh install -------------------------

def _default_db_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "data" / "market.db"


@pytest.fixture
def clean_db():
    p = _default_db_path()
    if p.exists():
        p.unlink()
    yield p
    if p.exists():
        p.unlink()


@pytest.mark.parametrize(
    "page",
    [
        "Home",
        "Market data",
        "Signals",
        "Strategies",
        "Compare strategies",
        "Backtest",
        "Backtest wizard",
        "Portfolio",
        "Paper trading",
        "Risk",
    ],
)
def test_every_page_renders_with_no_data(page, clean_db):
    """Fresh install: every page renders, none raises, data pages are friendly."""
    import qrp.ui.app as app

    assert page in app.PAGES
    at = _app_without_data()
    at.session_state[H.NAV_PAGE_KEY] = page
    at.run()
    assert not at.exception, f"{page}: {at.exception}"
    if page not in ("Home", "Market data"):
        assert any(
            "no market data" in str(i.value).lower() for i in at.info
        ), f"{page} should show the friendly empty state"


def test_prices_only_partial_db(clean_db):
    """Prices downloaded, signals not computed: pages degrade gracefully."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 1000,
            }
        ),
        "AAA",
        clean_db,
    )
    # Signals page works: it prompts to compute signals, no crash.
    at = _app_without_data()
    at.session_state[H.NAV_PAGE_KEY] = "Signals"
    at.run()
    assert not at.exception
    assert any("compute signals" in str(i.value).lower() for i in at.info)
    # Pages needing signals get the friendly message naming signals.
    for page in ("Compare strategies", "Backtest", "Portfolio"):
        at = _app_without_data()
        at.session_state[H.NAV_PAGE_KEY] = page
        at.run()
        assert not at.exception, page
        assert any("signal" in str(i.value).lower() for i in at.info), page
    # The wizard's Run step guides the user to the Signals page.
    at = _app_without_data()
    at.session_state[H.NAV_PAGE_KEY] = "Backtest wizard"
    at.session_state["wizard_step"] = 3
    at.run()
    assert not at.exception
    assert any(b.label == "Go to Signals" for b in at.button)


def test_fundamentals_only_partial_db(clean_db):
    """Fundamentals without prices: every data page asks for prices first."""
    from qrp.data.edgar import init_db as init_edgar_db

    init_edgar_db(clean_db)
    for page in ("Signals", "Backtest", "Paper trading", "Risk"):
        at = _app_without_data()
        at.session_state[H.NAV_PAGE_KEY] = page
        at.run()
        assert not at.exception, page
        assert any(
            "daily stock prices" in str(i.value).lower() for i in at.info
        ), page
