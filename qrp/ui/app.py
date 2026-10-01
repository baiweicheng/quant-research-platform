"""Streamlit dashboard for the quantitative research platform (Phase 6).

Run from the project root:

    streamlit run qrp/ui/app.py

The dashboard is a thin orchestration layer: all the real work lives in
qrp.data, qrp.signals, qrp.backtest, qrp.portfolio, qrp.risk and
qrp.execution. This file only presents forms, tables and charts — and
explains everything in plain English, because the person using it is not
assumed to have a finance background.

Nothing here places real orders. Trade execution is paper-trading only
(simulated money); no live broker exists in this codebase.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Streamlit puts the script's own folder on sys.path, not the project root,
# so make `import qrp...` work no matter how the app is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from qrp.data.store import DEFAULT_DB, upsert_bars
from qrp.data.yahoo import fetch_daily_bars
from qrp.execution.paper import PaperBroker
from qrp.risk.guards import RiskConfig
from qrp.signals import default_registry
from qrp.signals.store import count_signals, load_bars, upsert_signals
from qrp.ui import helpers as H


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.title("Quantitative Research Platform")
    st.write(
        "A personal toolkit for researching stock ideas with real data. "
        "It downloads free daily stock prices, scores stocks with simple "
        "signals, tests strategies against history, builds diversified "
        "portfolios with automatic safety checks, and lets you practice "
        "trading with fake money. **US stocks, long positions only.**"
    )

    st.subheader("How it works — the pipeline")
    st.markdown(
        """
1. **Get data** (Market data page) — download free daily prices for the
   stocks you care about.
2. **Score stocks** (Signals page) — compute a score for every stock, e.g.
   *momentum* (has it been climbing?), *trend* (above its long-term
   average?), *low volatility* (calm or jumpy?). Higher score = more
   attractive to buy and hold.
3. **Test ideas on history** (Backtest page) — replay the past to see what
   a signal-based strategy *would* have done, with trading costs included.
4. **Build a portfolio** (Portfolio page) — turn scores into "how much of
   each stock", with automatic caps so no single stock can dominate.
5. **Practice with fake money** (Paper trading page) — rebalance a simulated
   $100,000 account. Every order passes the safety checks first.
6. **Stay safe** (Risk page) — review the safety settings and run the
   checks on any portfolio.
"""
    )

    st.subheader("Quick start")
    st.markdown(
        """
1. Go to **Market data** and download prices (start with a few tickers,
   e.g. `AAPL, MSFT, NVDA`).
2. Go to **Signals** and press *Compute signals*.
3. Go to **Backtest** and press *Run backtest* to see how a simple
   strategy would have performed.
4. Go to **Paper trading** and preview a rebalance — still fake money.
"""
    )

    st.subheader("Safety first")
    st.info(
        "This platform only paper-trades: the account, the orders and the "
        "fees are real math, but no real dollars move and no real broker is "
        "connected. Connecting a real broker is a future step and will only "
        "happen with your explicit approval."
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def page_data() -> None:
    st.title("Market data")
    st.write(
        "Stock prices are the raw material for everything else here. This "
        "page downloads free daily prices (open, high, low, close, volume) "
        "from Yahoo Finance and stores them on your computer. No API key "
        "needed — just be polite and don't hammer the free service."
    )

    st.subheader("Download prices")
    tickers_text = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="US stock symbols, e.g. AAPL for Apple. Separate several with commas.",
    )
    col1, col2 = st.columns(2)
    start = col1.text_input(
        "From date",
        value="2020-01-01",
        help="First day to download, as YYYY-MM-DD.",
    )
    end = col2.text_input(
        "To date",
        value=date.today().isoformat(),
        help="Last day to download, as YYYY-MM-DD.",
    )
    if st.button("Download", type="primary"):
        tickers = H.parse_tickers(tickers_text)
        if not tickers:
            st.error("Type at least one ticker symbol first.")
        else:
            results = []
            with st.spinner(f"Downloading {len(tickers)} ticker(s)..."):
                for ticker in tickers:
                    try:
                        df = fetch_daily_bars(ticker, start, end)
                        n = upsert_bars(df, ticker, DEFAULT_DB)
                        results.append(
                            {"Ticker": ticker, "New prices": n, "Status": "OK"}
                        )
                    except Exception as exc:  # keep going on per-ticker failures
                        results.append(
                            {
                                "Ticker": ticker,
                                "New prices": 0,
                                "Status": f"Failed: {exc}",
                            }
                        )
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("What is stored right now")
    summary = H.ticker_summary()
    if summary.empty:
        st.info(
            "No data yet. Download some tickers above to get started — the "
            "rest of the platform needs prices first."
        )
    else:
        st.dataframe(summary, use_container_width=True)
        st.caption(
            "One row per stock: the first and last trading day stored, and "
            "how many daily prices there are."
        )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _signal_picker(label: str = "Signal", key: str = "signal") -> tuple[str, str]:
    """Selectbox of registered signals; returns (name, description)."""
    registry = default_registry()
    options = registry.list_signals()
    names = [name for name, _ in options]
    descriptions = dict(options)
    name = st.selectbox(
        label,
        names,
        key=key,
        format_func=lambda n: n.replace("_", " ").title(),
    )
    st.caption(descriptions[name])
    return name, descriptions[name]


def page_signals() -> None:
    st.title("Signals")
    st.write(
        "A **signal** is a score for every stock. The scores are built so "
        "you can compare stocks against each other: **a higher score means "
        "the stock looks more attractive to buy and hold.** These are "
        "classic starting points used by investment researchers — not "
        "guarantees. The Backtest page checks how they would actually have "
        "performed in the past."
    )

    tickers = H.db_tickers()
    if not tickers:
        st.info(
            "No price data yet — go to the Market data page and download "
            "some tickers first."
        )
        return

    signal_name, _ = _signal_picker(key="signals_page_signal")
    chosen = st.multiselect("Stocks", tickers, default=tickers)

    if st.button("Compute signals", type="primary"):
        if not chosen:
            st.error("Pick at least one stock.")
        else:
            sig = default_registry().get(signal_name)
            rows = []
            with st.spinner("Scoring stocks..."):
                for ticker in chosen:
                    df = load_bars(ticker, path=DEFAULT_DB)
                    if df.empty:
                        rows.append(
                            {"Ticker": ticker, "Values stored": 0,
                             "Status": "No price data"}
                        )
                        continue
                    values = sig.compute(df)
                    n = upsert_signals(values, ticker, signal_name, DEFAULT_DB)
                    total = count_signals(ticker, signal_name, DEFAULT_DB)
                    rows.append(
                        {"Ticker": ticker, "Values stored": n,
                         "Total in database": total, "Status": "OK"}
                    )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Latest scores")
    table = H.latest_signal_table(signal_name, tickers)
    if table.empty:
        st.info("No computed values yet — press *Compute signals* above.")
    else:
        st.dataframe(table, use_container_width=True)
        st.caption(
            "The most recent score for each stock and the date it is as of. "
            "Higher = more attractive."
        )

    st.subheader("Score over time")
    one = st.selectbox("Stock for the chart", tickers, key="signals_chart_ticker")
    series = H.load_signal_series(signal_name, one)
    if series.empty:
        st.info(f"No {signal_name.replace('_', ' ')} values stored for {one} yet.")
    else:
        st.line_chart(series)
        st.caption(
            f"How {one}'s {signal_name.replace('_', ' ')} score moved over time."
        )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

_METRIC_ORDER = [
    "final_value", "total_return", "cagr", "volatility", "sharpe",
    "max_drawdown", "turnover", "win_rate", "num_trades",
]


def page_backtest() -> None:
    st.title("Backtest")
    st.write(
        "A **backtest** replays history: it pretends you followed a strategy "
        "in the past and shows what *would* have happened, with trading "
        "costs included. It is the standard way to check an idea *before* "
        "trusting it — but it is not a crystal ball (see the honest "
        "reminder at the bottom)."
    )

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    signal_name, _ = _signal_picker(label="Rank stocks by", key="backtest_signal")
    c1, c2, c3 = st.columns(3)
    top_n = c1.slider(
        "Hold the top N stocks",
        1, min(20, len(tickers)), min(5, len(tickers)),
        help="On each rebalance day, buy the N highest-scoring stocks.",
    )
    weighting_key = c2.selectbox(
        "Position sizing",
        ["equal", "signal", "inverse_vol"],
        format_func=lambda k: H.WEIGHTING_LABELS[k],
    )
    rebalance = c3.radio("Rebalance", ["monthly", "weekly"], horizontal=True,
                         help="How often the portfolio is adjusted back to its targets.")
    st.caption(H.weighting_description(weighting_key))

    c4, c5, c6 = st.columns(3)
    cost_bps = c4.slider(
        "Cost per trade",
        0.0, 50.0, 10.0, 1.0,
        help="Trading cost in basis points. 10 = 0.10% of each trade's value lost to costs.",
    )
    capital = c5.number_input(
        "Starting cash ($)", value=100_000.0, step=10_000.0, min_value=1_000.0,
    )
    max_weight = c6.slider(
        "Max weight per stock",
        0.05, 0.50, 0.20, 0.05,
        help="No single stock may exceed this share of the portfolio.",
    )
    c7, c8 = st.columns(2)
    start = c7.text_input("From (optional, YYYY-MM-DD)", value="",
                          help="Leave empty to start at the earliest available data.")
    end = c8.text_input("To (optional, YYYY-MM-DD)", value="",
                        help="Leave empty to end at the latest available data.")

    if st.button("Run backtest", type="primary"):
        with st.spinner("Replaying history..."):
            try:
                result, metrics = H.run_backtest_for_ui(
                    signal_name, tickers, weighting_key, top_n, max_weight,
                    rebalance, cost_bps, capital,
                    start.strip() or None, end.strip() or None, DEFAULT_DB,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state["last_equity"] = result.equity

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final value",
                  H.metric_display("final_value", metrics["final_value"]["value"]))
        m2.metric("Total return",
                  H.metric_display("total_return", metrics["total_return"]["value"]))
        m3.metric("Yearly growth (CAGR)",
                  H.metric_display("cagr", metrics["cagr"]["value"]))
        m4.metric("Worst fall",
                  H.metric_display("max_drawdown", metrics["max_drawdown"]["value"]))

        st.subheader("Portfolio value over time")
        equity = result.equity.copy()
        equity["date"] = pd.to_datetime(equity["date"])
        st.line_chart(equity.set_index("date")["portfolio_value"])
        st.caption("The simulated account value on each trading day.")

        st.subheader("All metrics, explained")
        for key in _METRIC_ORDER:
            m = metrics[key]
            with st.expander(f"{m['label']}: {H.metric_display(key, m['value'])}"):
                st.write(f"**What it means:** {m['what']}")
                st.write(f"**Why it matters:** {m['why']}")
                st.write(f"**What good looks like:** {m['good']}")

        st.warning(
            "Honest reminder: a backtest shows what *would* have happened, "
            "using today's knowledge of the past. It does not guarantee "
            "anything about the future — markets change, popular strategies "
            "stop working once everyone uses them, and real-life costs (taxes, "
            "wider price gaps on big orders) are usually higher. Treat "
            "backtest results as a filter for bad ideas, not proof of future "
            "profits."
        )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def _portfolio_controls(prefix: str) -> dict:
    """Shared signal/method/date controls; returns the chosen settings."""
    signal_name, _ = _signal_picker(label="Rank stocks by", key=f"{prefix}_signal")
    method_key = st.selectbox(
        "Position sizing",
        ["equal", "signal", "inverse_vol"],
        key=f"{prefix}_method",
        format_func=lambda k: H.WEIGHTING_LABELS[k],
    )
    st.caption(H.weighting_description(method_key))
    asof = st.text_input(
        "As of date (YYYY-MM-DD)",
        value="",
        key=f"{prefix}_date",
        help="Leave empty to use the latest date with signal data.",
    )
    return {"signal_name": signal_name, "method_key": method_key, "asof": asof}


def page_portfolio() -> None:
    st.title("Target portfolio")
    st.write(
        'A **target portfolio** answers "how much of each stock should I '
        'hold right now?" — ticker by ticker, as a percentage of your money. '
        "Safety caps apply automatically: no stock may exceed the max "
        "weight, tiny 'dust' positions are dropped, and whatever cannot be "
        "placed stays as **cash** — the portfolio never forces money into "
        "stocks just to stay invested."
    )

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    settings = _portfolio_controls("portfolio")
    c1, c2 = st.columns(2)
    max_weight = c1.slider("Max weight per stock", 0.05, 0.50, 0.20, 0.05)
    max_positions = c2.slider("Max number of stocks", 1, 30, 20)

    if st.button("Build portfolio", type="primary"):
        asof = settings["asof"].strip() or None
        try:
            if asof is None:
                from qrp.backtest.engine import load_signal_matrix
                matrix = load_signal_matrix(
                    settings["signal_name"], tickers, path=DEFAULT_DB
                )
                if matrix.empty:
                    raise ValueError("No signal data — compute signals first.")
                asof = matrix.index.max().strftime("%Y-%m-%d")
            portfolio, scores, day = H.build_target_for_ui(
                settings["signal_name"], settings["method_key"], asof,
                tickers, DEFAULT_DB, max_weight=max_weight,
                max_positions=max_positions,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.subheader(f"Target portfolio as of {day}")
        if portfolio.weights:
            rows = [
                {
                    "Ticker": t,
                    "Signal score": f"{scores[t]:+.4f}",
                    "Target weight": f"{w:.1%}",
                }
                for t, w in sorted(
                    portfolio.weights.items(), key=lambda kv: -kv[1]
                )
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            chart_df = pd.DataFrame(
                {"weight": portfolio.weights}
            ).sort_values("weight", ascending=False)
            st.bar_chart(chart_df)
            st.write(f"**Cash:** {portfolio.cash_weight:.1%} stays uninvested.")
            st.info(
                f"What this means: if you invested $10,000 today, you would "
                f"put ${10_000 * (1 - portfolio.cash_weight):,.0f} into "
                f"{len(portfolio.weights)} stock(s) as shown above and keep "
                f"${10_000 * portfolio.cash_weight:,.0f} in cash."
            )
        else:
            st.info("No positions — the whole portfolio would stay in cash.")
        for note in portfolio.notes:
            st.caption(f"Note: {note}")


# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------

def _paper_account_tables(broker: PaperBroker) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Positions table, recent-trades table, and total account value."""
    closes: dict[str, float] = {}
    for ticker in broker.positions:
        try:
            closes[ticker] = broker.last_price(ticker)
        except KeyError:
            closes[ticker] = 0.0
    pos_value = sum(
        shares * closes.get(ticker, 0.0)
        for ticker, shares in broker.positions.items()
    )
    total = broker.cash + pos_value
    pos_rows = []
    for ticker, shares in sorted(broker.positions.items()):
        value = shares * closes.get(ticker, 0.0)
        pos_rows.append(
            {
                "Ticker": ticker,
                "Shares": f"{shares:,.2f}",
                "Price": H.fmt_money(closes.get(ticker, 0.0)),
                "Value": H.fmt_money(value),
                "Share of account": H.fmt_pct(value / total if total > 0 else 0.0),
            }
        )
    trade_rows = [
        {
            "Date": t["date"],
            "Ticker": t["ticker"],
            "Side": t["side"],
            "Shares": f"{t['shares']:,.2f}",
            "Price": H.fmt_money(t["price"]),
            "Fees": H.fmt_money(t["cost"]),
        }
        for t in broker.trades(limit=20)
    ]
    return (
        pd.DataFrame(pos_rows),
        pd.DataFrame(trade_rows),
        total,
    )


def page_paper() -> None:
    st.title("Paper trading")
    st.write(
        "**Paper trading** means practicing with fake money. The account, "
        "the orders, the fees and the risk checks are all real math — but "
        "no real dollars move and nothing touches a real brokerage. It is "
        "the flight simulator before the real plane."
    )

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    broker = PaperBroker(DEFAULT_DB)
    pos_df, trade_df, total = _paper_account_tables(broker)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cash", H.fmt_money(broker.cash))
    c2.metric("Positions value", H.fmt_money(total - broker.cash))
    c3.metric("Total account", H.fmt_money(total))

    st.subheader("Positions")
    if pos_df.empty:
        st.info("No positions yet — the account is all cash.")
    else:
        st.dataframe(pos_df, use_container_width=True)

    st.subheader("Recent trades")
    if trade_df.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(trade_df, use_container_width=True)

    st.subheader("Preview a rebalance")
    st.write(
        "See exactly what *would* be traded to move the paper account to a "
        "target portfolio — including which orders the safety checks would "
        "block — without trading anything."
    )
    settings = _portfolio_controls("paper")
    if st.button("Preview rebalance"):
        try:
            asof = settings["asof"].strip() or None
            if asof is None:
                from qrp.backtest.engine import load_signal_matrix
                matrix = load_signal_matrix(
                    settings["signal_name"], tickers, path=DEFAULT_DB
                )
                if matrix.empty:
                    raise ValueError("No signal data — compute signals first.")
                asof = matrix.index.max().strftime("%Y-%m-%d")
            portfolio, _, day = H.build_target_for_ui(
                settings["signal_name"], settings["method_key"], asof,
                tickers, DEFAULT_DB,
            )
            preview = H.preview_rebalance_orders(
                broker, portfolio.weights, day, total,
                cost_bps=broker.cost_bps,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        if not preview:
            st.success("Already at target — no trades needed.")
        else:
            st.dataframe(pd.DataFrame(preview), use_container_width=True)
            st.caption(
                "Dry run complete — the paper account was not touched. "
                "Blocked orders explain why in plain English."
            )

    st.subheader("Execute rebalance (paper money only)")
    confirm_exec = st.checkbox(
        "I understand this trades the PAPER account with fake money.",
        key="paper_exec_confirm",
    )
    if st.button("Execute paper rebalance", type="primary",
                 disabled=not confirm_exec):
        try:
            asof = settings["asof"].strip() or None
            if asof is None:
                from qrp.backtest.engine import load_signal_matrix
                matrix = load_signal_matrix(
                    settings["signal_name"], tickers, path=DEFAULT_DB
                )
                if matrix.empty:
                    raise ValueError("No signal data — compute signals first.")
                asof = matrix.index.max().strftime("%Y-%m-%d")
            portfolio, _, day = H.build_target_for_ui(
                settings["signal_name"], settings["method_key"], asof,
                tickers, DEFAULT_DB,
            )
            preview = H.preview_rebalance_orders(
                broker, portfolio.weights, day, total,
                cost_bps=broker.cost_bps,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        if not preview:
            st.success("Already at target — no trades needed.")
        else:
            for row in preview:
                result = broker.place_order(
                    row["Ticker"],
                    row["Action"],
                    float(str(row["Shares"]).replace(",", "")),
                    day,
                )
                if result.filled:
                    verb = "Bought" if result.action == "BUY" else "Sold"
                    st.write(
                        f"{verb} {result.shares:,.2f} shares of "
                        f"{result.ticker} at {H.fmt_money(result.price)} "
                        f"(fees {H.fmt_money(result.cost)})."
                    )
                else:
                    st.warning(f"Skipped {row['Action']} {row['Ticker']}: "
                               f"{result.reason}")
            st.success("Rebalance executed in the paper account. "
                       "Reminder: simulated money only.")
            st.rerun()

    st.subheader("Reset paper account")
    st.write("Wipe the paper account and start over with fresh play money. "
             "This erases positions and trade history.")
    confirm_reset = st.checkbox(
        "I understand this wipes the paper account and its history.",
        key="paper_reset_confirm",
    )
    reset_capital = st.number_input("Fresh play money ($)", value=100_000.0,
                                    step=10_000.0, min_value=1_000.0)
    if st.button("Reset paper account", disabled=not confirm_reset):
        broker.reset(reset_capital)
        st.success(f"Paper account reset with {H.fmt_money(reset_capital)} "
                   "of play money. No real money involved.")
        st.rerun()


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

def page_risk() -> None:
    st.title("Risk safeguards")
    st.write(
        "Think of these as the seatbelts and airbags of the platform: "
        "automatic checks that refuse to let a mistake — a typo'd price, an "
        "order bigger than your cash, one stock taking over the whole "
        "portfolio — turn into real damage. They run before every order, "
        "everywhere in the platform."
    )

    st.subheader("Current safety settings")
    st.dataframe(
        pd.DataFrame(H.risk_config_rows(RiskConfig())),
        use_container_width=True,
    )

    st.subheader("Check a target portfolio")
    st.write(
        "Build a target portfolio from a signal and run the portfolio-level "
        "safety checks on it: no negative weights, no stock above the cap, "
        "no more stocks than the maximum, weights never above 100%."
    )
    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return
    settings = _portfolio_controls("risk")
    if st.button("Run safety checks", type="primary"):
        try:
            asof = settings["asof"].strip() or None
            if asof is None:
                from qrp.backtest.engine import load_signal_matrix
                matrix = load_signal_matrix(
                    settings["signal_name"], tickers, path=DEFAULT_DB
                )
                if matrix.empty:
                    raise ValueError("No signal data — compute signals first.")
                asof = matrix.index.max().strftime("%Y-%m-%d")
            portfolio, _, day = H.build_target_for_ui(
                settings["signal_name"], settings["method_key"], asof,
                tickers, DEFAULT_DB,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        ok, reasons = H.check_target_portfolio(portfolio)
        st.write(f"Checked the target portfolio as of {day}.")
        if ok:
            st.success(
                "All checks passed — every weight is within the caps and "
                "the portfolio is long-only."
            )
        else:
            st.error("Some checks failed:")
            for reason in reasons:
                st.write(f"- {reason}")
        for note in portfolio.notes:
            st.caption(f"Note: {note}")

    st.subheader("The drawdown stop, in plain English")
    st.write(
        "A **drawdown** is how far the portfolio's value has fallen from its "
        "own highest point so far. If that fall gets bigger than the "
        "drawdown stop above (15% by default), **new buying pauses** until "
        "you review what is happening and lift the halt. Two things it does "
        "*not* do: it never sells anything by itself, and it never stops "
        "you from selling — the guard never traps you in a falling position."
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Every page function referenced here must exist — covered by tests/test_ui.py.
PAGES: dict[str, object] = {
    "Home": page_home,
    "Market data": page_data,
    "Signals": page_signals,
    "Backtest": page_backtest,
    "Portfolio": page_portfolio,
    "Paper trading": page_paper,
    "Risk": page_risk,
}


def main() -> None:
    st.set_page_config(
        page_title="Quant Research Platform",
        page_icon="📈",
        layout="wide",
    )
    st.sidebar.title("Quant Research Platform")
    st.sidebar.caption("US stocks · long only · paper trading")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.info(
        "Paper trading only — no real broker is connected and no real "
        "money moves."
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
