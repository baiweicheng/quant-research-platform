"""UI-agnostic helpers for the Streamlit dashboard (Phase 6).

These functions contain no Streamlit calls, so they are easy to unit-test.
The dashboard (qrp/ui/app.py) is a thin presentation layer over them and
the existing qrp modules (data, signals, backtest, portfolio, risk,
execution). All user-facing text here is plain English: the person using
the dashboard is not assumed to have a finance background.

Performance note: the expensive SQLite loads below go through small
@st.cache_data wrappers (cached_load_price_matrix and friends), because
Streamlit re-runs the whole script on every widget click. The cache key
always includes the database file's modification time, so newly ingested
data is never hidden behind a stale cache. The cached functions remain
pure and unit-testable outside Streamlit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from qrp.backtest.costs import SpreadCostModel
from qrp.backtest.engine import (
    BacktestResult,
    last_valid_dates,
    load_open_matrix,
    load_price_matrix,
    load_signal_matrix,
    run_backtest,
)
from qrp.data.sp500_history import (
    history_available,
    make_universe_filter,
    missing_history_note,
)
from qrp.backtest.metrics import (
    benchmark_equity,
    buy_and_hold_equity,
    compute_metrics,
    relative_metrics,
)
from qrp.backtest.strategies import WEIGHTING_METHODS, weighted_top_n
from qrp.data.edgar import (
    REQUEST_DELAY_S as EDGAR_REQUEST_DELAY_S,
    USER_AGENT,
    fundamentals_schema_is_current,
    ingest_ticker,
    last_fetch_summary,
    load_ticker_map,
)
from qrp.data.store import (
    DEFAULT_DB,
    connect_db,
    load_dividends,
    upsert_bars,
    upsert_dividends,
)
from qrp.data.yahoo import fetch_daily_bars, fetch_dividends, normalize_ticker
from qrp.execution.paper import resolve_paper_timing
from qrp.execution.rebalance import (
    orders_to_target,
    priceable_closes,
)
from qrp.portfolio.cli import METHOD_CHOICES
from qrp.portfolio.construction import (
    INVERSE_VOLATILITY,
    METHOD_DESCRIPTIONS,
    TargetPortfolio,
    build_target_weights,
    implied_stock_count,
)
from qrp.risk.guards import Order, PortfolioState, RiskConfig, check_order, check_portfolio
from qrp.signals.store import count_signals, list_tickers, load_bars, upsert_signals
from qrp.strategies import (
    cutoff_description,
    display_name,
    era_label,
    get_strategy,
    list_strategies,
    make_strategy_fn,
    strategy_signal_matrices,
)

# Short CLI keys ("equal") -> human labels shown in the dashboard.
WEIGHTING_LABELS = {
    "equal": "Equal weight",
    "signal": "Signal weight",
    "inverse_vol": "Inverse volatility",
}

logger = logging.getLogger(__name__)


# -- cached loaders --------------------------------------------------------

def _db_mtime(db_path: str | Path) -> float:
    """Modification time of the database file; 0.0 when it doesn't exist."""
    try:
        return Path(db_path).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _cached_price_matrix(
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None,
    db: str,
    mtime: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(prices, valid): closes plus the pre-fill validity mask.

    Cached once; the two wrappers below split the tuple so existing call
    sites keep working. `valid` marks real downloaded bars, used to spot
    delisted stocks whose history ends mid-backtest.
    """
    return load_price_matrix(
        list(tickers), start=start, end=end, path=db, with_validity=True
    )


@st.cache_data(show_spinner=False)
def _cached_signal_matrix(
    signal_name: str,
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None,
    db: str,
    mtime: float,
) -> pd.DataFrame:
    return load_signal_matrix(
        signal_name, list(tickers), start=start, end=end, path=db
    )


@st.cache_data(show_spinner=False)
def _cached_strategy_matrices(
    strategy_name: str,
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None,
    db: str,
    mtime: float,
) -> dict[str, pd.DataFrame]:
    return strategy_signal_matrices(
        get_strategy(strategy_name),
        list(tickers),
        start=start,
        end=end,
        db_path=db,
    )


def cached_load_price_matrix(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """load_price_matrix, cached per (tickers, dates, db, db-mtime)."""
    db = str(db_path)
    return _cached_price_matrix(
        tuple(tickers), start, end, db, _db_mtime(db_path)
    )[0]


def cached_price_validity(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Validity mask for cached_load_price_matrix (True = real bar).

    Shares its cache entry with cached_load_price_matrix, so this costs no
    extra download or DB work.
    """
    db = str(db_path)
    return _cached_price_matrix(
        tuple(tickers), start, end, db, _db_mtime(db_path)
    )[1]


def backtest_realism_context(
    tickers: list[str],
    start: str | None,
    end: str | None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[object | None, dict | None]:
    """Point-in-time membership + delisting info for one backtest.

    Returns (universe_filter, last_valid) for run_backtest:
    only index members on each date may be held, and a stock whose price
    history ends (delisted) is closed at its last real price. When the
    membership table is absent, universe_filter is None and the caller
    must show the survivorship-bias note.
    """
    universe_filter = make_universe_filter(tickers)
    valid = cached_price_validity(tickers, start=start, end=end, db_path=db_path)
    last_valid = last_valid_dates(valid) if not valid.empty else None
    return universe_filter, last_valid


@st.cache_data(show_spinner=False)
def _cached_open_matrix(
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None,
    db: str,
    mtime: float,
) -> pd.DataFrame:
    return load_open_matrix(list(tickers), start=start, end=end, path=db)


def cached_load_open_matrix(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """load_open_matrix, cached per (tickers, dates, db, db-mtime)."""
    db = str(db_path)
    return _cached_open_matrix(
        tuple(tickers), start, end, db, _db_mtime(db_path)
    )


def cached_load_signal_matrix(
    signal_name: str,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """load_signal_matrix, cached per (signal, tickers, dates, db, db-mtime)."""
    db = str(db_path)
    return _cached_signal_matrix(
        signal_name, tuple(tickers), start, end, db, _db_mtime(db_path)
    )


def cached_strategy_signal_matrices(
    strategy_name: str,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> dict[str, pd.DataFrame]:
    """strategy_signal_matrices, cached per (strategy, tickers, dates, db, db-mtime)."""
    db = str(db_path)
    return _cached_strategy_matrices(
        strategy_name, tuple(tickers), start, end, db, _db_mtime(db_path)
    )


# -- formatting ------------------------------------------------------------

def fmt_money(value: float) -> str:
    """$1,234.56 style; 'n/a' for missing values."""
    if value is None or value != value:  # NaN check without importing numpy
        return "n/a"
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    """+12.34% style; 'n/a' for missing values."""
    if value is None or value != value:
        return "n/a"
    return f"{value:+.2%}"


def parse_tickers(text: str) -> list[str]:
    """Turn 'aapl, MSFT; nvda' into ['AAPL', 'MSFT', 'NVDA']."""
    return [
        normalize_ticker(t)
        for t in text.replace(";", ",").split(",")
        if t.strip()
    ]


# -- data page -------------------------------------------------------------

def db_tickers(db_path: str | Path = DEFAULT_DB) -> list[str]:
    """Tickers with stored price data, sorted alphabetically."""
    return list_tickers(db_path)


# -- friendly empty states ---------------------------------------------------
# Two deliberate exceptions to this module's "no Streamlit calls" rule:
# require_data() and go_to_page() below are thin presentation helpers so
# that every data-dependent page shares one friendly empty state.

#: Session-state key the sidebar navigation uses to track the active page.
NAV_PAGE_KEY = "qrp_page"


def data_status(db_path: str | Path = DEFAULT_DB) -> dict:
    """Describe which data the local database actually holds.

    Pure and unit-testable: never raises on a missing, empty, or freshly
    created database — having no data yet is a normal state, not an error.
    Returns ``{"has_prices", "has_signals", "has_fundamentals", "tickers"}``.
    """
    status = {
        "has_prices": False,
        "has_signals": False,
        "has_fundamentals": False,
        "tickers": [],
    }
    path = Path(db_path)
    if not path.exists():
        return status
    try:
        with connect_db(path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "daily_bars" in tables:
                count = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
                if count:
                    status["has_prices"] = True
                    status["tickers"] = [
                        row[0]
                        for row in conn.execute(
                            "SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker"
                        )
                    ]
            if "signals" in tables:
                status["has_signals"] = bool(
                    conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
                )
            if "fundamentals" in tables:
                status["has_fundamentals"] = bool(
                    conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
                )
    except sqlite3.Error:
        return status
    return status


def go_to_page(page: str) -> None:
    """Jump the dashboard to another page (used by empty-state buttons)."""
    st.session_state[NAV_PAGE_KEY] = page
    st.rerun()


def data_date_range(db_path: str | Path = DEFAULT_DB) -> tuple[str | None, str | None]:
    """(first_date, last_date) of stored daily prices as YYYY-MM-DD strings.

    Pure and unit-testable: returns (None, None) when the database has no
    price data yet. Used to default the Backtest/Compare/Wizard date inputs
    to the actual data range.
    """
    path = Path(db_path)
    if not path.exists():
        return None, None
    try:
        with connect_db(path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "daily_bars" not in tables:
                return None, None
            first, last = conn.execute(
                "SELECT MIN(date), MAX(date) FROM daily_bars"
            ).fetchone()
    except sqlite3.Error:
        return None, None
    if not first or not last:
        return None, None
    return str(first)[:10], str(last)[:10]


def benchmark_for_ui(
    tickers: list[str],
    start: str | None,
    end: str | None,
    capital: float,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[pd.DataFrame, dict]:
    """S&P-500-style buy-and-hold benchmark over the same inputs.

    "Buy every stock in equal amounts on day one, never trade" — computed
    from the same downloaded prices over the same dates, so it is directly
    comparable to any backtest. Returns (equity_curve, metrics).
    """
    prices = cached_load_price_matrix(
        tickers, start=start, end=end, db_path=db_path
    )
    if prices.empty:
        raise ValueError("No price data in that date range.")
    equity = buy_and_hold_equity(prices, capital)
    return equity, compute_metrics(equity, pd.DataFrame())


def spy_benchmark_for_ui(
    start: str | None,
    end: str | None,
    capital: float,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[pd.DataFrame, dict]:
    """S&P 500 (SPY) buy-and-hold benchmark over the same inputs.

    "Put all the starting cash into SPY — the ETF tracking the S&P 500 —
    on day one, never trade." Directly comparable to any backtest over the
    same dates. Returns (equity_curve, metrics). Raises ValueError when no
    SPY prices are stored (tick "Also download SPY" on the Market data
    page to get them).
    """
    prices = cached_load_price_matrix(
        ["SPY"], start=start, end=end, db_path=db_path
    )
    if prices.empty:
        raise ValueError(
            "No SPY data in that date range. On the Market data page, tick "
            "'Also download SPY (S&P 500 benchmark)' and download prices — "
            "then this comparison can use it."
        )
    equity = benchmark_equity(prices, capital)
    return equity, compute_metrics(equity, pd.DataFrame())


def spy_relative_for_ui(
    result_equity: pd.DataFrame,
    start: str | None,
    end: str | None,
    capital: float,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[pd.DataFrame, dict, dict]:
    """SPY benchmark equity, its metrics, and strategy-vs-SPY metrics.

    Returns (spy_equity, spy_metrics, relative_metrics). Raises ValueError
    when no SPY prices are stored — callers show the friendly hint instead
    of a traceback.
    """
    spy_equity, spy_metrics = spy_benchmark_for_ui(start, end, capital, db_path)
    rel = relative_metrics(result_equity, spy_equity)
    return spy_equity, spy_metrics, rel


def compare_benchmark_rows(
    tickers: list[str],
    start: str | None,
    end: str | None,
    capital: float,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[dict | None, dict | None]:
    """Benchmark rows for the Compare page: (universe_row, spy_row).

    Each row is a dict with the display columns Strategy | Total return |
    Yearly growth (CAGR) | Sharpe | Worst fall | Trades plus raw
    "total_return"/"sharpe" keys for the chart and winner logic — the same
    shape _race_one_strategy returns. A row is None when its data is
    missing (SPY needs the "Also download SPY" checkbox on Market data).
    """
    def row(name: str, metrics: dict) -> dict:
        return {
            "Strategy": name,
            "Total return": metric_display("total_return", metrics["total_return"]["value"]),
            "Yearly growth (CAGR)": metric_display("cagr", metrics["cagr"]["value"]),
            "Sharpe": metric_display("sharpe", metrics["sharpe"]["value"]),
            "Worst fall": metric_display("max_drawdown", metrics["max_drawdown"]["value"]),
            "Trades": "0",
            "total_return": metrics["total_return"]["value"],
            "sharpe": metrics["sharpe"]["value"],
        }

    try:
        _, bench_metrics = benchmark_for_ui(tickers, start, end, capital, db_path)
        bench_row: dict | None = row("Market benchmark (buy-and-hold)", bench_metrics)
    except ValueError:
        bench_row = None
    try:
        _, spy_metrics = spy_benchmark_for_ui(start, end, capital, db_path)
        spy_row: dict | None = row("S&P 500 (SPY)", spy_metrics)
    except ValueError:
        spy_row = None
    return bench_row, spy_row


def warn_if_no_trades(trades: pd.DataFrame) -> None:
    """Plain-English warning when a backtest executed zero trades.

    A $100,000.00 / +0.00% result with no trades is almost never what the
    user intended — it usually means the signals don't cover the chosen
    dates. This says so instead of failing silently.
    """
    if trades is not None and not trades.empty:
        return
    st.warning(
        "This backtest made **no trades**, so the account just sat in cash. "
        "The usual causes: the signals haven't been computed for these dates "
        "yet (compute them on the Signals page first), or the strategy needs "
        "more price history than the date range holds — for example, the "
        "Momentum signal needs a full year of prices before its first score "
        "appears. Try widening the dates."
    )
    """Jump the dashboard to another page (used by empty-state buttons)."""
    st.session_state[NAV_PAGE_KEY] = page
    st.rerun()


def require_data(*needs: str, db_path: str | Path = DEFAULT_DB) -> dict:
    """Gate a data-dependent page behind one friendly empty state.

    ``needs`` names any of ``"prices"``, ``"signals"``, ``"fundamentals"``.
    When everything requested is present, the ``data_status`` dict is
    returned so callers can reuse the ticker list. Otherwise a plain-English
    message is shown with a button that jumps straight to the Market data
    page, and the page stops — a raw database error is never surfaced.
    """
    labels = {
        "prices": "daily stock prices",
        "signals": "computed signal scores",
        "fundamentals": "company fundamentals from SEC filings",
    }
    unknown = [n for n in needs if n not in labels]
    if unknown:
        raise ValueError(f"Unknown data need(s): {', '.join(unknown)}")
    status = data_status(db_path)
    missing = [n for n in needs if not status[f"has_{n}"]]
    if not missing:
        return status
    st.info(
        "There's no market data here yet. This page needs "
        + " and ".join(labels[n] for n in missing)
        + " — download them first on the Market data page. "
        "It's free and takes a minute or two."
    )
    if st.button(
        "Go to Market data",
        type="primary",
        key="require_data_goto_" + "_".join(needs),
    ):
        go_to_page("Market data")
    st.stop()
    return status  # unreachable; keeps type checkers calm


def ticker_summary(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """One row per stored ticker: symbol, first/last trading day, bar count."""
    path = Path(db_path)
    columns = ["Ticker", "First day", "Last day", "Daily prices"]
    tickers = list_tickers(path)
    if not tickers:
        return pd.DataFrame(columns=columns)
    rows = []
    with connect_db(path) as conn:
        for ticker in tickers:
            first, last, n = conn.execute(
                "SELECT MIN(date), MAX(date), COUNT(*) FROM daily_bars "
                "WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            rows.append(
                {
                    "Ticker": ticker,
                    "First day": first,
                    "Last day": last,
                    "Daily prices": n,
                }
            )
    return pd.DataFrame(rows, columns=columns)


SP500_TICKERS_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "sp500_tickers.txt"
)


def load_sp500_tickers() -> list[str]:
    """S&P 500 constituent tickers from the checked-in list.

    Reads qrp/data/sp500_tickers.txt (comment lines and blanks skipped).
    Symbols come back as listed, e.g. "BRK.B": the Yahoo downloader
    normalizes share-class dots to dashes ("BRK-B") and the SEC lookup
    accepts both spellings, so callers don't need to translate.
    """
    tickers = []
    for line in SP500_TICKERS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line.upper())
    return tickers


def load_sp500_universe(include_historical: bool = True) -> list[str]:
    """S&P 500 tickers for downloading: current plus past members.

    Past members matter because backtests need price history for stocks
    that were in the index at the time (many delisted tickers still serve
    history on Yahoo; ones that don't are skipped per-ticker). Falls back
    to the current list when the membership table is absent.
    """
    from qrp.data import sp500_universe

    return sp500_universe(include_historical=include_historical)


def fundamentals_summary(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """One row per ticker with stored fundamentals: fact count, period range."""
    columns = ["Ticker", "Facts stored", "First period", "Last period"]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    with connect_db(path) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type = 'table' AND name = 'fundamentals'"
        ).fetchone()
        if not has_table:
            return pd.DataFrame(columns=columns)
        df = pd.read_sql_query(
            'SELECT ticker AS "Ticker", COUNT(*) AS "Facts stored", '
            'MIN(period_end) AS "First period", '
            'MAX(period_end) AS "Last period" '
            "FROM fundamentals GROUP BY ticker ORDER BY ticker",
            conn,
        )
    return df


def fundamentals_fetch_summary(
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """When each ticker's fundamentals were last checked for new filings.

    One display-ready row per ticker with "Ticker", "Last checked" (the UTC
    timestamp of the last successful fundamentals download) and "Facts
    stored". Built on the incremental-ingest state table, so it always
    reflects reality — empty when nothing was ever fetched.
    """
    df = last_fetch_summary(db_path)
    return df.rename(
        columns={
            "ticker": "Ticker",
            "last_fetch": "Last checked",
            "facts": "Facts stored",
        }
    )


def download_market_data(
    tickers: list[str],
    start: str | None,
    end: str | None,
    download_prices: bool,
    download_fundamentals: bool,
    db_path: str | Path = DEFAULT_DB,
    progress_cb=None,
) -> tuple[list[dict], list[str]]:
    """Download prices and/or fundamentals for tickers.

    Returns (rows, notices). One result dict per ticker with keys "Ticker",
    "Prices" and "Fundamentals", each a plain-English status ("OK — …",
    "Failed: …", or "—" when that download wasn't selected). A failing
    ticker never aborts the run. `notices` carries plain-English messages
    the UI should show (e.g. the old-format fundamentals rebuild note).
    `progress_cb(done, total)` is called after each ticker when given.

    `start`/`end` are only used when `download_prices` is true.
    """
    rows: list[dict] = []
    notices: list[str] = []
    total = len(tickers)
    session = None
    ticker_map = None
    if download_fundamentals:
        if not fundamentals_schema_is_current(db_path):
            notices.append(
                "Your saved company financials used an older format, so they "
                "were cleared and are downloading fresh on this run — "
                "nothing else in your database was touched."
            )
        # Load once up front (mirror edgar_cli): the ticker map is ~1MB of
        # JSON, and one HTTP session reuses the connection per ticker.
        ticker_map = load_ticker_map()
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
    for i, raw in enumerate(tickers):
        t = raw.strip().upper()
        if download_prices:
            try:
                yt = normalize_ticker(t)  # e.g. "BRK.B" -> "BRK-B" for Yahoo
                n = upsert_bars(fetch_daily_bars(yt, start, end), yt, db_path)
                # Dividends ride along in the same chart response: creditable
                # cash on ex-dates. Stored idempotently, so re-running a
                # download backfills new ex-dates without duplicating bars.
                try:
                    n_div = upsert_dividends(
                        fetch_dividends(yt, start, end), yt, db_path
                    )
                    div_note = f", {n_div} dividends"
                except Exception:
                    div_note = ""  # never fail the price ingest for this
                prices_status = f"OK — {n:,} new daily prices{div_note}"
            except Exception as exc:  # keep going on per-ticker failures
                prices_status = f"Failed: {exc}"
        else:
            prices_status = "—"
        if download_fundamentals:
            try:
                n = ingest_ticker(
                    t, db_path, session=session, ticker_map=ticker_map
                )
                fund_status = f"OK — {n:,} new facts"
            except Exception as exc:  # keep going on per-ticker failures
                fund_status = f"Failed: {exc}"
        else:
            fund_status = "—"
        rows.append(
            {"Ticker": t, "Prices": prices_status, "Fundamentals": fund_status}
        )
        if progress_cb is not None:
            progress_cb(i + 1, total)
        # Be polite to the free SEC endpoint between tickers.
        if download_fundamentals and i < total - 1:
            time.sleep(EDGAR_REQUEST_DELAY_S)
    return rows, notices


# -- signals page ----------------------------------------------------------

def load_signal_series(
    signal_name: str,
    ticker: str,
    db_path: str | Path = DEFAULT_DB,
) -> pd.Series:
    """One signal's values over time for one ticker (for charting)."""
    path = Path(db_path)
    if not path.exists():
        return pd.Series(dtype=float)
    with connect_db(path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'signals'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return pd.Series(dtype=float)
        df = pd.read_sql_query(
            "SELECT date, value FROM signals WHERE signal = ? AND ticker = ? "
            "ORDER BY date",
            conn,
            params=[signal_name, normalize_ticker(ticker)],
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


def latest_signal_table(
    signal_name: str,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Latest stored score per ticker for one signal, best score first."""
    matrix = cached_load_signal_matrix(signal_name, tickers, db_path=db_path)
    if matrix.empty:
        return pd.DataFrame(columns=["Ticker", "As of", "Score"])
    rows = []
    for ticker in matrix.columns:
        col = matrix[ticker].dropna()
        if col.empty:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "As of": col.index[-1].strftime("%Y-%m-%d"),
                "Score": round(float(col.iloc[-1]), 4),
            }
        )
    table = pd.DataFrame(rows, columns=["Ticker", "As of", "Score"])
    return table.sort_values("Score", ascending=False).reset_index(drop=True)


def compute_signal_rows(
    signal_name: str,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
) -> list[dict]:
    """Compute one signal for tickers and store the results.

    This is the exact computation the Signals page runs, and it matches
    ``python -m qrp.signals.cli`` row for row: results land in the signals
    table in the same shape the Backtest and Compare pages expect.

    Fundamental (cross-sectional) signals — value_composite, profitability,
    trending_value — are scored for the whole ticker set at once via
    ``compute_all()``, because "cheap" only means something compared to
    other stocks. Price-only signals use the per-ticker ``compute()`` loop.
    The routing is decided by the signal's class (FundamentalSignal), never
    by a hardcoded list of names.

    Returns one row dict per ticker with "Ticker", "Values stored",
    "Total in database" and "Status" ("OK" or "No price data"). Missing
    data is reported per ticker — this function never raises for it.
    """
    # Lazy imports: qrp.signals.fundamental pulls in the backtest engine
    # lazily itself, and keeping these local avoids any import-order
    # surprises for modules that import helpers early.
    from qrp.signals import default_registry
    from qrp.signals.fundamental import FundamentalSignal

    sig = default_registry().get(signal_name)
    tickers = [normalize_ticker(t) for t in tickers]
    rows: list[dict] = []

    def _row(ticker: str, stored: int, status: str) -> dict:
        return {
            "Ticker": ticker,
            "Values stored": stored,
            "Total in database": count_signals(ticker, signal_name, db_path),
            "Status": status,
        }

    if isinstance(sig, FundamentalSignal):
        matrix = sig.compute_all(tickers, db_path)
        if matrix.empty:
            return [_row(t, 0, "No price data") for t in tickers]
        for ticker in tickers:
            if ticker not in matrix.columns:
                rows.append(_row(ticker, 0, "No price data"))
                continue
            n = upsert_signals(matrix[ticker].dropna(), ticker, signal_name, db_path)
            rows.append(_row(ticker, n, "OK"))
        return rows

    for ticker in tickers:
        df = load_bars(ticker, path=db_path)
        if df.empty:
            rows.append(_row(ticker, 0, "No price data"))
            continue
        values = sig.compute(df)
        n = upsert_signals(values, ticker, signal_name, db_path)
        rows.append(_row(ticker, n, "OK"))
    return rows


def signal_data_note(
    signal_name: str,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
) -> str | None:
    """Plain-English data-gap note for a signal, if it defines one.

    A signal may define an optional ``data_gap_note(tickers, db_path)``
    method (see TwinMomentum): text shown when the signal ran fine but
    its key input data is missing entirely — e.g. "no earnings data
    yet, re-run the fundamentals download" instead of a bare "OK" with
    zero rows stored. Returns None for signals without the hook, or
    when there is no gap. Never raises.
    """
    # Lazy import (matches compute_signal_rows): keeps import order safe.
    from qrp.signals import default_registry

    try:
        sig = default_registry().get(signal_name)
    except KeyError:
        return None
    gap = getattr(sig, "data_gap_note", None)
    if not callable(gap):
        return None
    try:
        return gap(tickers, db_path)
    except Exception:
        logger.exception("data_gap_note failed for %s", signal_name)
        return None


def compute_all_signals(
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
    compute_rows=None,
    progress=None,
) -> list[dict]:
    """Compute every registered signal and summarize the run.

    Runs each registered signal with the same per-signal computation the
    Signals page's single dropdown uses (``compute_signal_rows``), one
    after another, and returns one summary dict per signal: "Signal",
    "Tickers OK" (e.g. "3/5"), "Rows stored", "Time (s)", and "Status"
    ("OK" or a plain-English failure note — a failure never raises, and
    the database is left unchanged for that signal).

    `compute_rows` overrides the per-signal computation; it exists so
    tests can verify the iteration without doing the heavy work.
    `progress` is an optional callback ``progress(done, total, label)``
    called before each signal and once at the end, so a UI can draw a
    progress bar.
    """
    # Lazy import (matches compute_signal_rows): keeps import order safe.
    from qrp.signals import default_registry

    compute = compute_rows if compute_rows is not None else compute_signal_rows
    names = [name for name, _ in default_registry().list_signals()]
    total = len(names)
    summary: list[dict] = []
    for i, name in enumerate(names):
        label = name.replace("_", " ").title()
        if progress is not None:
            progress(i, total, label)
        t0 = time.perf_counter()
        try:
            rows = compute(name, tickers, db_path)
        except Exception:
            # The user sees a plain-English note; the real traceback goes
            # to the log so the next "Failed" is diagnosable without
            # code-reading.
            logger.exception("compute_all_signals: signal %s failed", name)
            summary.append(
                {
                    "Signal": label,
                    "Tickers OK": f"0/{len(tickers)}",
                    "Rows stored": 0,
                    "Time (s)": round(time.perf_counter() - t0, 1),
                    "Status": "Failed — saved data unchanged",
                }
            )
            continue
        ok = sum(1 for r in rows if r.get("Status") == "OK")
        written = sum(int(r.get("Values stored", 0) or 0) for r in rows)
        # A signal whose key data is missing entirely says so plainly
        # instead of a bare "OK" with zero rows stored.
        note = signal_data_note(name, tickers, db_path)
        summary.append(
            {
                "Signal": label,
                "Tickers OK": f"{ok}/{len(rows)}",
                "Rows stored": written,
                "Time (s)": round(time.perf_counter() - t0, 1),
                "Status": note or "OK",
            }
        )
    if progress is not None:
        progress(total, total, "")
    return summary


def latest_scores(
    signal_name: str,
    tickers: list[str],
    asof: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[dict[str, float], str]:
    """Point-in-time signal scores: latest value on or before `asof`.

    Returns (scores, day) where day is the YYYY-MM-DD the scores are as of.
    Never uses data from after `asof` — the same no-peeking rule as the
    backtest engine.
    """
    matrix = cached_load_signal_matrix(signal_name, tickers, db_path=db_path)
    if matrix.empty:
        raise ValueError(
            f"No stored values for signal {signal_name!r}. "
            "Compute signals first (Signals page)."
        )
    # None means "latest available" — spelled out explicitly rather than
    # relying on slicing with NaT.
    cutoff = matrix.index.max() if asof is None else pd.Timestamp(asof)
    history = matrix.loc[:cutoff].ffill()
    if history.empty:
        raise ValueError(f"No signal data on or before {asof}.")
    scores = history.iloc[-1].dropna().to_dict()
    if not scores:
        raise ValueError(f"No scored stocks on or before {asof}.")
    return scores, history.index[-1].strftime("%Y-%m-%d")


# -- portfolio page --------------------------------------------------------

def build_target_for_ui(
    signal_name: str,
    method_key: str,
    asof: str,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
    max_weight: float = 0.20,
) -> tuple[TargetPortfolio, dict[str, float], str]:
    """Build a target portfolio from a signal (dashboard-friendly wrapper).

    method_key is one of "equal", "signal", "inverse_vol" (the short CLI
    names). Returns (portfolio, scores, day). The stock count is implied
    by max_weight (n = round(1 / max_weight)) — a single sizing knob.
    """
    if method_key not in METHOD_CHOICES:
        raise ValueError(f"unknown weighting {method_key!r}")
    scores, day = latest_scores(signal_name, tickers, asof, db_path)
    method = METHOD_CHOICES[method_key]
    volatilities: dict[str, float] | None = None
    if method == INVERSE_VOLATILITY:
        # The stored low_volatility signal IS the negative of recent price
        # volatility, so negating it gives point-in-time volatilities.
        low_vol = cached_load_signal_matrix("low_volatility", tickers, db_path=db_path)
        if low_vol.empty:
            raise ValueError(
                "Inverse-vol weighting needs low_volatility values — "
                "compute signals first (Signals page)."
            )
        v_hist = (-low_vol).loc[: pd.Timestamp(asof)].ffill()
        if not v_hist.empty:
            volatilities = v_hist.iloc[-1].dropna().to_dict()
    portfolio = build_target_weights(
        scores,
        method=method,
        max_weight=max_weight,
        max_positions=implied_stock_count(max_weight),
        volatilities=volatilities,
    )
    return portfolio, scores, day


def weighting_description(method_key: str) -> str:
    """Plain-English description of a weighting method (short CLI key)."""
    return METHOD_DESCRIPTIONS[METHOD_CHOICES[method_key]]


# -- strategies ------------------------------------------------------------

def strategy_options() -> list[tuple[str, str]]:
    """(name, description) for every registered strategy, for dropdowns."""
    return list_strategies()


def strategy_display(name: str) -> str:
    """'trending_value' -> 'Trending Value'."""
    return display_name(name)


def strategy_era_label(name: str) -> str:
    """Build-era group label for a strategy, e.g. 'Fundamental strategies'."""
    return era_label(name)


def strategy_settings_summary(name: str) -> str:
    """One plain-English line summarizing a strategy's settings."""
    s = get_strategy(name)
    n = implied_stock_count(s.max_weight)
    return (
        f"Holds up to {n} stocks (no stock over {s.max_weight:.0%}), "
        f"rebalanced {s.rebalance}, sized by "
        f"{s.weighting.replace('_', ' ')}, and skips trades that would move "
        f"a position by less than {s.drift_tolerance:.0%} (drift buffer)."
    )


def strategy_calculation(name: str) -> str:
    """The strategy's step-by-step calculation text (plain English)."""
    return get_strategy(name).calculation


def strategy_comparison_rows() -> list[dict]:
    """Every strategy as a comparison row for the Strategies page."""
    rows = []
    for name, _ in list_strategies():
        s = get_strategy(name)
        n = implied_stock_count(s.max_weight)
        if s.cutoffs:
            cutoff_text = "; ".join(
                cutoff_description(sig, c) for sig, c in s.cutoffs.items()
            )
            cutoff_text = f"Holds {cutoff_text}."
        else:
            cutoff_text = "—"
        rows.append(
            {
                "name": name,
                "Era": era_label(name),
                "Strategy": display_name(name),
                "What it does": s.description,
                "Calculation": s.calculation,
                "Data used": s.data_used,
                "Rebalance": s.rebalance,
                "Up to": f"{n} stocks (max {s.max_weight:.0%} each)",
                "Cutoff filter": cutoff_text,
                "Good at": s.good_at,
                "Main risk": s.main_risk,
            }
        )
    return rows


def _build_cost_model(
    cost_model: str,
    tickers: list[str],
    start: str | None,
    end: str | None,
    db_path: str | Path,
):
    """Pick the transaction-cost model for a backtest.

    Returns a ``CostModel`` for run_backtest, or None for the flat default
    (fixed fee per trade — identical to earlier versions). "spread" builds
    a SpreadCostModel from each ticker's daily high/low/close/volume
    history, which estimates the real cost of each trade from recent price
    ranges. Anything else raises ValueError.
    """
    if cost_model == "flat":
        return None
    if cost_model == "spread":
        bars = {
            ticker: load_bars(ticker, start, end, db_path)
            for ticker in tickers
        }
        return SpreadCostModel(bars)
    raise ValueError(f"unknown cost model {cost_model!r}")


def run_strategy_backtest_for_ui(
    strategy_name: str,
    tickers: list[str],
    cost_bps: float,
    capital: float,
    start: str | None,
    end: str | None,
    db_path: str | Path = DEFAULT_DB,
    cost_model: str = "spread",
    execution: str = "open",
) -> tuple[BacktestResult, dict]:
    """Run a named registry strategy; returns (result, metrics).

    cost_model is "spread" (the default: realistic cost estimated from each
    stock's recent high-low price ranges) or "flat" (fixed fee per trade —
    keeps results identical to earlier versions). execution is "open"
    (fills at the next trading day's open — the default, matching the real
    nightly workflow) or "close" (next day's close, the older convention).
    The backtest only ever holds stocks that were index members on each
    date (no survivorship lookahead), and delisted stocks are closed at
    their last real price.
    """
    strategy = get_strategy(strategy_name)
    prices = cached_load_price_matrix(
        tickers, start=start, end=end, db_path=db_path
    )
    if prices.empty:
        raise ValueError("No price data in that date range.")
    matrices = cached_strategy_signal_matrices(
        strategy_name, tickers, start=start, end=end, db_path=db_path
    )
    universe_filter, last_valid = backtest_realism_context(
        tickers, start, end, db_path
    )
    fn = make_strategy_fn(
        strategy, tickers, start=start, end=end, db_path=db_path,
        matrices=matrices, universe_filter=universe_filter,
    )
    # No signal matrix to load here: the strategy function carries its own
    # pre-loaded point-in-time matrices, and the engine only needs the date
    # grid — an empty frame with the right index aligns it.
    signals = pd.DataFrame(index=prices.index)
    result = run_backtest(
        prices,
        signals,
        fn,
        frequency=strategy.rebalance,
        cost_bps=cost_bps,
        initial_capital=capital,
        drift_tolerance=strategy.drift_tolerance,
        cost_model=_build_cost_model(
            cost_model, tickers, start, end, db_path
        ),
        # Dividend cash lands in the portfolio on ex-dates, so the equity
        # curve is a true total return (price moves + payouts), not just
        # price moves.
        dividends=load_dividends(tickers, start, end, db_path),
        universe_filter=universe_filter,
        last_valid=last_valid,
    )
    return result, compute_metrics(result.equity, result.trades)


def _race_one_strategy(
    strategy_name: str,
    tickers: list[str],
    start: str | None,
    end: str | None,
    capital: float,
    max_weight: float,
    cost_bps: float,
    db_path: str,
    frequency: str | None = None,
    cost_model: str = "spread",
    _cost_model_obj=None,
    _universe_filter=None,
    _last_valid=None,
) -> dict:
    """Run one strategy over the shared inputs; return its comparison row.

    The strategy keeps its own recipe (which signals it uses, how it sizes
    positions, its drift buffer); only the max weight is overridden by the
    shared comparison knob, and `frequency` (when given) overrides every
    strategy's own rebalance preset uniformly, so every strategy races
    under identical conditions. Returns a display row with the raw
    "total_return"/"sharpe" numbers kept under lowercase keys for the bar
    chart and the winner logic (the caller drops those from the table).
    """
    strategy = replace(get_strategy(strategy_name), max_weight=max_weight)
    if frequency is not None:
        # Uniform override: this comparison rebalances every strategy at
        # the chosen frequency, ignoring each registry preset.
        strategy = replace(strategy, rebalance=frequency)
    db = str(db_path)
    prices = _cached_price_matrix(
        tuple(tickers), start, end, db, _db_mtime(db_path)
    )[0]
    if prices.empty:
        raise ValueError("No price data in that date range.")
    matrices = _cached_strategy_matrices(
        strategy_name, tuple(tickers), start, end, db, _db_mtime(db_path)
    )
    fn = make_strategy_fn(
        strategy, tickers, start=start, end=end, db_path=db_path,
        matrices=matrices, universe_filter=_universe_filter,
    )
    # Same as run_strategy_backtest_for_ui: the strategy carries its own
    # point-in-time matrices, so an empty frame with the right date grid
    # aligns the engine.
    signals = pd.DataFrame(index=prices.index)
    if _cost_model_obj is None:
        _cost_model_obj = _build_cost_model(
            cost_model, tickers, start, end, db_path
        )
    result = run_backtest(
        prices,
        signals,
        fn,
        frequency=strategy.rebalance,
        cost_bps=cost_bps,
        initial_capital=capital,
        drift_tolerance=strategy.drift_tolerance,
        cost_model=_cost_model_obj,
        dividends=load_dividends(tickers, start, end, db_path),
        # The Compare page holds everything constant across strategies:
        # next-day-open fills, like the real nightly workflow.
        execution="open",
        opens=_cached_open_matrix(
            tuple(tickers), start, end, db, _db_mtime(db_path)
        ),
        universe_filter=_universe_filter,
        last_valid=_last_valid,
    )
    m = compute_metrics(result.equity, result.trades)
    total_return = m["total_return"]["value"]
    sharpe = m["sharpe"]["value"]
    return {
        "Strategy": display_name(strategy_name),
        "Total return": metric_display("total_return", total_return),
        "Yearly growth (CAGR)": metric_display("cagr", m["cagr"]["value"]),
        "Sharpe": metric_display("sharpe", sharpe),
        "Worst fall": metric_display("max_drawdown", m["max_drawdown"]["value"]),
        "Trades": metric_display("num_trades", m["num_trades"]["value"]),
        "total_return": total_return,
        "sharpe": sharpe,
    }


@st.cache_data(show_spinner=False)
def _cached_compare_rows(
    tickers: tuple[str, ...],
    start: str | None,
    end: str | None,
    capital: float,
    max_weight: float,
    cost_bps: float,
    db: str,
    mtime: float,
    frequency: str | None,
    cost_model: str,
) -> list[dict]:
    # Shared across all strategies in this race: one spread-cost model
    # (building it per strategy would reload every ticker's bars nine
    # times), one membership filter, one delisting map.
    tickers_list = list(tickers)
    cost_model_obj = _build_cost_model(cost_model, tickers_list, start, end, db)
    universe_filter = make_universe_filter(tickers_list)
    valid = _cached_price_matrix(tickers, start, end, db, mtime)[1]
    last_valid = last_valid_dates(valid) if not valid.empty else None
    return [
        _race_one_strategy(
            name, tickers_list, start, end, capital, max_weight, cost_bps,
            db, frequency=frequency, cost_model=cost_model,
            _cost_model_obj=cost_model_obj,
            _universe_filter=universe_filter, _last_valid=last_valid,
        )
        for name, _ in list_strategies()
    ]


def compare_cache_key(
    strategy_names: list[str],
    tickers: list[str],
    start: str | None,
    end: str | None,
    capital: float,
    max_weight: float,
    cost_bps: float,
    execution: str,
    cost_model: str = "spread",
    frequency: str | None = None,
) -> str:
    """Stable hash identifying one comparison configuration.

    The Compare page remembers its results in the Streamlit session state
    under this key, so leaving the page and coming back restores the
    results instead of forcing a re-run — but only when every input still
    matches. Any changed input (including the rebalancing-frequency
    override and the cost model) produces a different key, so stale results
    are never shown.
    """
    payload = {
        "strategies": sorted(strategy_names),
        "tickers": sorted(tickers),
        "start": start or "",
        "end": end or "",
        "capital": float(capital),
        "max_weight": float(max_weight),
        "cost_bps": float(cost_bps),
        "execution": execution,
        "cost_model": cost_model,
        "frequency": frequency or "",
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_cached_compare(cache: dict, key: str) -> dict | None:
    """Return the cached comparison payload for `key`, or None."""
    return cache.get(key)


def store_cached_compare(cache: dict, key: str, payload: dict) -> None:
    """Remember a comparison result under its cache key."""
    cache[key] = payload


def compare_strategies_for_ui(
    tickers: list[str],
    start: str | None,
    end: str | None,
    capital: float,
    max_weight: float,
    cost_bps: float = 10.0,
    db_path: str | Path = DEFAULT_DB,
    cost_model: str = "spread",
    frequency: str | None = None,
) -> list[dict]:
    """Run every registered strategy over identical inputs; returns rows.

    Same stocks, dates, starting cash and max weight for all — a fair
    head-to-head. Each strategy keeps its own recipe (which signals it
    uses, how it sizes positions, its drift buffer); `frequency`, when
    given, overrides every strategy's rebalance preset uniformly
    ("daily"/"weekly"/"monthly"). Costs use the spread-based model by
    default. One dict per strategy with display columns Strategy | Total
    return | Yearly growth (CAGR) | Sharpe | Worst fall | Trades, plus raw
    "total_return"/"sharpe" numbers for the chart and the winner logic.
    The whole run is cached per (inputs, db-mtime), so clicking around the
    dashboard never re-runs it.
    """
    db = str(db_path)
    return _cached_compare_rows(
        tuple(tickers), start, end, capital, max_weight, cost_bps, db,
        _db_mtime(db_path), frequency, cost_model,
    )


def realism_footnote(cost_model_name: str | None = None) -> str:
    """Plain-English "How realistic are these numbers?" footnote.

    Shown under Compare and Backtest results. Honest about the method and
    the remaining limitations.
    """
    if history_available():
        membership = (
            "it only ever buys stocks that were actually in the S&P 500 on "
            "each date — no peeking at which companies survived"
        )
    else:
        membership = (
            "NOTE: the historical index-membership table is missing, so "
            "this run used today's stock list for the whole period — that "
            "flatters results (survivorship bias)"
        )
    if cost_model_name == "flat":
        costs = (
            "a simple flat fee per trade (the older convention — switch to "
            "the realistic price-gap estimate for a harsher, truer number)"
        )
    else:
        costs = (
            "a realistic price-gap cost estimated from each stock's recent "
            "high-low ranges"
        )
    return (
        "How realistic are these numbers? The backtest replays history "
        f"with: {membership}; trades fill at the next day's open, like "
        "market-on-open orders placed the night before; trading costs use "
        f"{costs}; and cash dividends are credited to whoever held the "
        "stock. What it still does NOT model: market impact (a large "
        "portfolio moving prices when it trades), and bankruptcies that go "
        "to zero (a delisted stock is closed at its last available price, "
        "but a true wipeout can't be detected automatically). Past "
        "performance never predicts the future."
    )


def strategy_preset_rows() -> list[dict]:
    """One row per strategy: its registry preset defaults.

    Used by the Compare page's "Strategy defaults" table. Stocks is the
    target count implied by the max-weight cap.
    """
    rows = []
    for name, _ in list_strategies():
        s = get_strategy(name)
        rows.append(
            {
                "Strategy": s.formal_name or display_name(name),
                "Stocks": f"~{implied_stock_count(s.max_weight)}",
                "Rebalance": s.rebalance.capitalize(),
                "Weighting": WEIGHTING_LABELS.get(s.weighting, s.weighting),
                "Max weight": f"{s.max_weight:.0%}",
            }
        )
    return rows


def build_strategy_target_for_ui(
    strategy_name: str,
    asof: str,
    tickers: list[str],
    db_path: str | Path = DEFAULT_DB,
) -> tuple[TargetPortfolio, dict[str, float], str, dict[str, str]]:
    """Target portfolio for a named strategy as of a date (dashboard wrapper).

    Returns (TargetPortfolio, composite_scores, day, leg_dates).
    Point-in-time: only signal values dated on or before `asof` are used —
    never from the future. `day` is the honest "as of": the OLDEST of the
    per-leg latest data dates (the blended score is only as fresh as its
    stalest input). `leg_dates` maps each underlying signal name to its own
    latest data date, so the UI can show exactly how fresh each input is
    instead of a single date that might overstate one leg.
    """
    from qrp.strategies import strategy_target_portfolio

    strategy = get_strategy(strategy_name)
    matrices = cached_strategy_signal_matrices(
        strategy_name, tickers, db_path=db_path
    )
    cutoff = pd.Timestamp(asof)
    leg_dates: dict[str, str] = {}
    dated: dict[str, pd.DataFrame] = {}
    for sig_name, _ in strategy.signals:
        m = matrices[sig_name]
        leg_dates[sig_name] = m.index.max().strftime("%Y-%m-%d")
        if not m.loc[:cutoff].empty:
            dated[sig_name] = m
    if not dated:
        raise ValueError(f"No signal data on or before {asof}.")
    day = min(m.loc[:cutoff].index.max() for m in dated.values())
    portfolio, scores = strategy_target_portfolio(strategy, dated, day)
    return portfolio, scores, day.strftime("%Y-%m-%d"), leg_dates


# -- backtest page ---------------------------------------------------------

def run_backtest_for_ui(
    signal_name: str,
    tickers: list[str],
    weighting_key: str,
    max_weight: float,
    rebalance: str,
    cost_bps: float,
    capital: float,
    start: str | None,
    end: str | None,
    db_path: str | Path = DEFAULT_DB,
    cost_model: str = "spread",
    drift_tolerance: float = 0.02,
    execution: str = "open",
) -> tuple[BacktestResult, dict]:
    """Run a long-only backtest; returns (result, metrics).

    The stock count is implied by max_weight (n = round(1 / max_weight)) —
    a single sizing knob. cost_model is "spread" (the default: realistic
    cost estimated from each stock's recent high-low price ranges) or
    "flat" (fixed fee per trade — keeps results identical to earlier
    versions). execution is "open" (fills at the next trading day's open —
    the default) or "close" (next day's close, the older convention).
    drift_tolerance skips trades that would move a position by less than
    this fraction of portfolio value (0.02 = 2%); it is passed explicitly
    so the Backtest page agrees with the Compare page and the wizard.
    Only index members on each date may be held (no survivorship
    lookahead); delisted stocks are closed at their last real price.
    """
    prices = cached_load_price_matrix(
        tickers, start=start, end=end, db_path=db_path
    )
    if prices.empty:
        raise ValueError("No price data in that date range.")
    signals = cached_load_signal_matrix(
        signal_name, tickers, start=start, end=end, db_path=db_path
    )
    if signals.empty:
        raise ValueError(
            f"No stored values for signal {signal_name!r} in that range. "
            "Compute signals first (Signals page)."
        )
    method = WEIGHTING_METHODS[weighting_key]
    vol_matrix = None
    if method == INVERSE_VOLATILITY:
        low_vol = cached_load_signal_matrix(
            "low_volatility", tickers, start=start, end=end, db_path=db_path
        )
        if low_vol.empty:
            raise ValueError(
                "Inverse-vol weighting needs low_volatility values — "
                "compute signals first (Signals page)."
            )
        vol_matrix = -low_vol
    n = implied_stock_count(max_weight)
    strategy = weighted_top_n(
        signal_name, n, method=method, max_weight=max_weight,
        vol_matrix=vol_matrix,
    )
    universe_filter, last_valid = backtest_realism_context(
        tickers, start, end, db_path
    )
    result = run_backtest(
        prices,
        signals,
        strategy,
        frequency=rebalance,
        cost_bps=cost_bps,
        initial_capital=capital,
        drift_tolerance=drift_tolerance,
        cost_model=_build_cost_model(
            cost_model, tickers, start, end, db_path
        ),
        # Dividend cash lands in the portfolio on ex-dates, so the equity
        # curve is a true total return (price moves + payouts).
        dividends=load_dividends(tickers, start, end, db_path),
        execution=execution,
        opens=(
            cached_load_open_matrix(
                tickers, start=start, end=end, db_path=db_path
            )
            if execution == "open"
            else None
        ),
        universe_filter=universe_filter,
        last_valid=last_valid,
    )
    return result, compute_metrics(result.equity, result.trades)


def metric_display(key: str, value: float) -> str:
    """Short display string for one metric value."""
    if key == "final_value":
        return fmt_money(value)
    if key == "sharpe":
        return "n/a" if value != value else f"{value:.2f}"
    if key in ("beta", "information_ratio"):
        # Unit-free ratios: 1.05x market sensitivity, 0.75 consistency.
        return "n/a" if value != value else f"{value:.2f}"
    if key == "num_trades":
        return "n/a" if value != value else f"{int(round(value))}"
    if key == "max_drawdown":
        # Shown as a fall, e.g. "-12.30%" — the stored value is negative
        # (a drop from the peak), so -abs() just normalizes the sign.
        return fmt_pct(-abs(value))
    return fmt_pct(value)


# -- risk page -------------------------------------------------------------

def risk_config_rows(config: RiskConfig | None = None) -> list[dict]:
    """Current safety settings with plain-English explanations."""
    config = config or RiskConfig()
    stop = (
        "Off"
        if config.max_drawdown_stop is None
        else f"{config.max_drawdown_stop:.0%}"
    )
    per_order = (
        "No cap"
        if config.max_order_value is None
        else fmt_money(config.max_order_value)
    )
    return [
        {
            "Setting": "Max weight per stock",
            "Value": f"{config.max_position_weight:.0%}",
            "What it means": (
                "No single stock may ever take more than this slice of the "
                "portfolio. Keeps one bad pick from sinking you."
            ),
        },
        {
            "Setting": "Max number of stocks",
            "Value": str(config.max_positions),
            "What it means": (
                "Never hold more stocks than this at once. Keeps the "
                "portfolio focused instead of a hundred tiny bets."
            ),
        },
        {
            "Setting": "Drawdown stop",
            "Value": stop,
            "What it means": (
                "If the portfolio's value falls this far below its own peak, "
                "new buying pauses until you review and lift the halt. "
                "Selling always still works — nothing is ever sold automatically."
            ),
        },
        {
            "Setting": "Max dollars per order",
            "Value": per_order,
            "What it means": (
                "Any single order costing more than this is refused. Catches "
                "'fat finger' mistakes like ordering 1,000 shares instead of 100."
            ),
        },
        {
            "Setting": "Price safety margin",
            "Value": f"{config.price_collar:.0%}",
            "What it means": (
                "An order priced further than this from the stock's last "
                "close is refused. Catches typos and stale prices."
            ),
        },
    ]


def check_target_portfolio(
    portfolio: TargetPortfolio, config: RiskConfig | None = None
) -> tuple[bool, list[str]]:
    """Run the portfolio-level safety checks; returns (passed, reasons)."""
    return check_portfolio(portfolio.weights, config or RiskConfig())


# -- paper trading page ----------------------------------------------------

def compute_rebalance_orders(
    broker,
    target_weights: dict[str, float],
    day: str,
    capital: float,
    cost_bps: float = 10.0,
) -> list[Order]:
    """Raw executable orders for a paper rebalance (sells first, then buys).

    These are the exact Order objects the preview table is built from —
    always execute THESE, never re-parsed display strings (the preview
    rounds shares for readability, which would silently change the orders).
    """
    return orders_to_target(broker, target_weights, day, capital, cost_bps)


def preview_rebalance_orders(
    broker,
    target_weights: dict[str, float],
    day: str,
    capital: float,
    cost_bps: float = 10.0,
) -> tuple[list[dict], list[str]]:
    """Dry-run orders for a rebalance, each with its safety-check verdict.

    Nothing is traded. Returns (rows, notes): one display dict per order
    with Action, Ticker, Shares, Price, Est. value, the plain-English
    "Why" (the order's reason — e.g. "Bought AAA: new position at 10.0% of
    the portfolio"), and the safety check result in plain English; plus
    plain-English notes about anything that was skipped (e.g. tickers with
    no price data that day).
    """
    orders = compute_rebalance_orders(
        broker, target_weights, day, capital, cost_bps
    )
    closes, missing = priceable_closes(
        broker, set(target_weights) | set(broker.positions), day
    )
    state = PortfolioState(
        cash=broker.cash, positions=broker.positions, prices=closes
    )
    preview = []
    for order in orders:
        ok, reason = check_order(order, state, broker.risk_config)
        preview.append(
            {
                "Action": order.action,
                "Ticker": order.ticker,
                "Shares": round(order.shares, 2),
                "Price": fmt_money(order.price),
                "Est. value": fmt_money(order.shares * order.price),
                "Why": order.reason,
                "Safety check": (
                    "Pass — every check cleared."
                    if ok
                    else f"Blocked: {reason}"
                ),
            }
        )
    notes = []
    execution = getattr(broker, "execution", "close")
    if execution == "open":
        notes.append(
            "Prices shown are the day's OPENING prices — these orders would "
            "fill at the next market open, like real market-on-open orders "
            "staged the night before."
        )
    else:
        notes.append(
            "Prices shown are the day's CLOSING prices — these orders would "
            "fill at the day's close."
        )
    if missing:
        notes.append(
            "No price data on " + day + " for " + ", ".join(missing) +
            " — left untouched (delisted or halted stocks are never guessed at)."
        )
    return preview, notes


__all__ = [
    "METHOD_DESCRIPTIONS",
    "WEIGHTING_LABELS",
    "fmt_money",
    "fmt_pct",
    "implied_stock_count",
    "parse_tickers",
    "ticker_summary",
    "load_sp500_tickers",
    "load_sp500_universe",
    "fundamentals_summary",
    "fundamentals_fetch_summary",
    "download_market_data",
    "load_signal_series",
    "latest_signal_table",
    "latest_scores",
    "build_target_for_ui",
    "weighting_description",
    "run_backtest_for_ui",
    "metric_display",
    "risk_config_rows",
    "check_target_portfolio",
    "compute_rebalance_orders",
    "preview_rebalance_orders",
    "resolve_paper_timing",
    "strategy_options",
    "strategy_display",
    "strategy_era_label",
    "strategy_settings_summary",
    "strategy_calculation",
    "strategy_comparison_rows",
    "run_strategy_backtest_for_ui",
    "compare_cache_key",
    "get_cached_compare",
    "store_cached_compare",
    "compute_signal_rows",
    "compute_all_signals",
    "compare_strategies_for_ui",
    "build_strategy_target_for_ui",
    "realism_footnote",
    "strategy_preset_rows",
    "backtest_realism_context",
    "cached_price_validity",
]
