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

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from qrp.backtest.engine import (
    BacktestResult,
    load_price_matrix,
    load_signal_matrix,
    run_backtest,
)
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import WEIGHTING_METHODS, weighted_top_n
from qrp.data.store import DEFAULT_DB
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
)
from qrp.risk.guards import Order, PortfolioState, RiskConfig, check_order, check_portfolio
from qrp.signals.store import list_tickers
from qrp.strategies import (
    display_name,
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
) -> pd.DataFrame:
    return load_price_matrix(list(tickers), start=start, end=end, path=db)


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
        t.strip().upper()
        for t in text.replace(";", ",").split(",")
        if t.strip()
    ]


# -- data page -------------------------------------------------------------

def db_tickers(db_path: str | Path = DEFAULT_DB) -> list[str]:
    """Tickers with stored price data, sorted alphabetically."""
    return list_tickers(db_path)


def ticker_summary(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """One row per stored ticker: symbol, first/last trading day, bar count."""
    path = Path(db_path)
    columns = ["Ticker", "First day", "Last day", "Daily prices"]
    tickers = list_tickers(path)
    if not tickers:
        return pd.DataFrame(columns=columns)
    rows = []
    with sqlite3.connect(path) as conn:
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
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT date, value FROM signals WHERE signal = ? AND ticker = ? "
            "ORDER BY date",
            conn,
            params=[signal_name, ticker.strip().upper()],
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
    max_positions: int = 20,
) -> tuple[TargetPortfolio, dict[str, float], str]:
    """Build a target portfolio from a signal (dashboard-friendly wrapper).

    method_key is one of "equal", "signal", "inverse_vol" (the short CLI
    names). Returns (portfolio, scores, day).
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
        max_positions=max_positions,
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


def strategy_settings_summary(name: str) -> str:
    """One plain-English line summarizing a strategy's settings."""
    s = get_strategy(name)
    return (
        f"Holds up to {s.top_n} stocks, rebalanced {s.rebalance}, sized by "
        f"{s.weighting.replace('_', ' ')}, no stock over "
        f"{s.max_weight:.0%}, and skips trades that would move a position "
        f"by less than {s.drift_tolerance:.0%} (drift buffer)."
    )


def strategy_comparison_rows() -> list[dict]:
    """Every strategy as a comparison row for the Strategies page."""
    rows = []
    for name, _ in list_strategies():
        s = get_strategy(name)
        rows.append(
            {
                "name": name,
                "Strategy": display_name(name),
                "What it does": s.description,
                "Data used": s.data_used,
                "Rebalance": s.rebalance,
                "Up to": f"{s.top_n} stocks",
                "Good at": s.good_at,
                "Main risk": s.main_risk,
            }
        )
    return rows


def run_strategy_backtest_for_ui(
    strategy_name: str,
    tickers: list[str],
    cost_bps: float,
    capital: float,
    start: str | None,
    end: str | None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[BacktestResult, dict]:
    """Run a named registry strategy; returns (result, metrics)."""
    strategy = get_strategy(strategy_name)
    prices = cached_load_price_matrix(
        tickers, start=start, end=end, db_path=db_path
    )
    if prices.empty:
        raise ValueError("No price data in that date range.")
    matrices = cached_strategy_signal_matrices(
        strategy_name, tickers, start=start, end=end, db_path=db_path
    )
    fn = make_strategy_fn(
        strategy, tickers, start=start, end=end, db_path=db_path,
        matrices=matrices,
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
    )
    return result, compute_metrics(result.equity, result.trades)


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
    top_n: int,
    max_weight: float,
    rebalance: str,
    cost_bps: float,
    capital: float,
    start: str | None,
    end: str | None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[BacktestResult, dict]:
    """Run a long-only backtest; returns (result, metrics)."""
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
    n = max(1, min(top_n, 20))
    strategy = weighted_top_n(
        signal_name, n, method=method, max_weight=max_weight,
        vol_matrix=vol_matrix,
    )
    result = run_backtest(
        prices,
        signals,
        strategy,
        frequency=rebalance,
        cost_bps=cost_bps,
        initial_capital=capital,
    )
    return result, compute_metrics(result.equity, result.trades)


def metric_display(key: str, value: float) -> str:
    """Short display string for one metric value."""
    if key == "final_value":
        return fmt_money(value)
    if key == "sharpe":
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
    with Action, Ticker, Shares, Price, Est. value, and the safety check
    result in plain English; plus plain-English notes about anything that
    was skipped (e.g. tickers with no price data that day).
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
                "Safety check": (
                    "Pass — every check cleared."
                    if ok
                    else f"Blocked: {reason}"
                ),
            }
        )
    notes = []
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
    "parse_tickers",
    "ticker_summary",
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
    "strategy_settings_summary",
    "strategy_comparison_rows",
    "run_strategy_backtest_for_ui",
    "build_strategy_target_for_ui",
]
