"""UI-agnostic helpers for the Streamlit dashboard (Phase 6).

These functions contain no Streamlit calls, so they are easy to unit-test.
The dashboard (qrp/ui/app.py) is a thin presentation layer over them and
the existing qrp modules (data, signals, backtest, portfolio, risk,
execution). All user-facing text here is plain English: the person using
the dashboard is not assumed to have a finance background.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

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

# Short CLI keys ("equal") -> human labels shown in the dashboard.
WEIGHTING_LABELS = {
    "equal": "Equal weight",
    "signal": "Signal weight",
    "inverse_vol": "Inverse volatility",
}


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
    matrix = load_signal_matrix(signal_name, tickers, path=db_path)
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
    matrix = load_signal_matrix(signal_name, tickers, path=db_path)
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
        low_vol = load_signal_matrix("low_volatility", tickers, path=db_path)
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
    prices = load_price_matrix(tickers, start=start, end=end, path=db_path)
    if prices.empty:
        raise ValueError("No price data in that date range.")
    signals = load_signal_matrix(
        signal_name, tickers, start=start, end=end, path=db_path
    )
    if signals.empty:
        raise ValueError(
            f"No stored values for signal {signal_name!r} in that range. "
            "Compute signals first (Signals page)."
        )
    method = WEIGHTING_METHODS[weighting_key]
    vol_matrix = None
    if method == INVERSE_VOLATILITY:
        low_vol = load_signal_matrix(
            "low_volatility", tickers, start=start, end=end, path=db_path
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
]
