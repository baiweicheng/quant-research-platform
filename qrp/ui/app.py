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

from qrp.data.store import DEFAULT_DB
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

    st.divider()

    with st.container(border=True):
        st.subheader("How it works — the pipeline")
        st.markdown(
            """
1. **Get data** (Market data page) — download free daily prices for the
   stocks you care about.
2. **Score stocks** (Signals page) — compute a score for every stock, e.g.
   *momentum* (has it been climbing?), *trend* (above its long-term
   average?), *low volatility* (calm or jumpy?). Higher score = more
   attractive to buy and hold.
3. **Compare strategies** (Strategies page) — read what each ready-made
   strategy does, what data it needs, and what it's good (and bad) at,
   side by side.
4. **Test ideas on history** (Backtest page) — replay the past to see what
   a signal-based strategy *would* have done, with trading costs included.
5. **Build a portfolio** (Portfolio page) — turn scores into "how much of
   each stock", with automatic caps so no single stock can dominate.
6. **Practice with fake money** (Paper trading page) — rebalance a simulated
   $100,000 account. Every order passes the safety checks first.
7. **Stay safe** (Risk page) — review the safety settings and run the
   checks on any portfolio.
"""
        )

    with st.container(border=True):
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

    with st.container(border=True):
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
        "Everything here starts with data, downloaded free and stored on "
        "your computer — no API key needed.\n\n"
        "**Prices (daily)** are the day-by-day record — open, high, low, "
        "close, volume — from Yahoo Finance. Every signal, backtest and "
        "strategy needs prices.\n\n"
        "**Fundamentals** are the numbers from a company's official "
        "financial reports — revenue, profit, assets, debt, shares "
        "outstanding — filed with the SEC (the US markets regulator). They "
        "only update **quarterly**, when companies publish results, not "
        "daily like prices. Three strategies need them: **Trending Value**, "
        "**Value Momentum** and **Quality Lowvol** (the Strategies page "
        "says which)."
    )

    st.divider()

    with st.container(border=True):
        st.subheader("Download data")
        sp500 = H.load_sp500_tickers()
        universe = st.radio(
            "Which stocks?",
            options=["sp500", "custom"],
            format_func=lambda k: (
                f"S&P 500 — all {len(sp500)} companies"
                if k == "sp500"
                else "Custom list"
            ),
            help=(
                "The S&P 500 is the 500 largest US companies — the full hunting "
                "ground the strategies were designed for. 'Custom list' lets you "
                "type your own symbols."
            ),
        )
        tickers_text = ""
        if universe == "custom":
            tickers_text = st.text_input(
                "Tickers (comma-separated)",
                value="AAPL, MSFT, NVDA",
                help="US stock symbols, e.g. AAPL for Apple. Separate several with commas.",
            )
        col1, col2 = st.columns(2)
        want_prices = col1.checkbox(
            "Prices (daily)",
            value=True,
            help=(
                "Daily open, high, low, close and volume from Yahoo Finance. "
                "Free, no key. Everything on this platform needs prices."
            ),
        )
        want_fund = col2.checkbox(
            "Fundamentals (SEC filings)",
            value=True,
            help=(
                "Revenue, profit, assets, debt and more from official SEC "
                "filings. Free, no key. Updates quarterly. Needed by Trending "
                "Value, Value Momentum and Quality Lowvol."
            ),
        )
        start = end = None
        if want_prices:
            d1, d2 = st.columns(2)
            start = d1.text_input(
                "From date",
                value="2020-01-01",
                help="First day to download, as YYYY-MM-DD.",
            )
            end = d2.text_input(
                "To date",
                value=date.today().isoformat(),
                help="Last day to download, as YYYY-MM-DD.",
            )
        if st.button("Download", type="primary"):
            if universe == "custom":
                tickers = H.parse_tickers(tickers_text)
            else:
                tickers = list(sp500)
            if not tickers:
                st.error("Type at least one ticker symbol first.")
            elif not (want_prices or want_fund):
                st.error("Tick at least one of Prices or Fundamentals first.")
            else:
                if want_fund and len(tickers) > 100:
                    st.info(
                        f"Downloading fundamentals for {len(tickers)} companies "
                        "takes a while (tens of minutes) — the SEC asks everyone "
                        "to go slowly. It's fine to leave this page open while "
                        "it works."
                    )
                progress = st.progress(0.0, text="Starting…")

                def _cb(done: int, total: int) -> None:
                    progress.progress(
                        done / total, text=f"{done} of {total} stocks…"
                    )

                with st.spinner(f"Downloading for {len(tickers)} stock(s)..."):
                    rows, notices = H.download_market_data(
                        tickers,
                        start,
                        end,
                        want_prices,
                        want_fund,
                        DEFAULT_DB,
                        progress_cb=_cb,
                    )
                progress.empty()
                for notice in notices:
                    st.warning(notice)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.container(border=True):
        st.subheader("Update fundamentals (only what's new)")
        st.write(
            "Checking for new filings only downloads what's new since your last "
            "check — much faster than starting over. Companies file quarterly, "
            "so weekly checks are plenty."
        )
        tickers_in_db = H.db_tickers()
        if not tickers_in_db:
            st.info(
                "No stocks in the database yet — download some data above first."
            )
        elif st.button("Check for new filings"):
            progress = st.progress(0.0, text="Starting…")

            def _cb(done: int, total: int) -> None:
                progress.progress(
                    done / total, text=f"{done} of {total} stocks…"
                )

            with st.spinner(
                f"Checking SEC filings for {len(tickers_in_db)} stock(s)..."
            ):
                rows, notices = H.download_market_data(
                    tickers_in_db, None, None,
                    download_prices=False, download_fundamentals=True,
                    db_path=DEFAULT_DB, progress_cb=_cb,
                )
            progress.empty()
            for notice in notices:
                st.warning(notice)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.container(border=True):
        st.subheader("Last checked")
        fetch_df = H.fundamentals_fetch_summary()
        if fetch_df.empty:
            st.info(
                "Nothing checked yet — press *Check for new filings* after "
                "downloading fundamentals at least once."
            )
        else:
            st.dataframe(fetch_df, use_container_width=True)
            st.caption(
                "When each company's filings were last checked for new numbers "
                "(UTC), and how many reported facts are stored."
            )

    with st.container(border=True):
        st.subheader("What prices are stored right now")
        summary = H.ticker_summary()
        if summary.empty:
            st.info(
                "No prices yet. Download some tickers above to get started — the "
                "rest of the platform needs prices first."
            )
        else:
            st.dataframe(summary, use_container_width=True)
            st.caption(
                "One row per stock: the first and last trading day stored, and "
                "how many daily prices there are."
            )

    with st.container(border=True):
        st.subheader("What fundamentals are stored right now")
        fsum = H.fundamentals_summary()
        if fsum.empty:
            st.info(
                "No fundamentals yet. Tick 'Fundamentals (SEC filings)' above "
                "and press Download — the value and quality strategies need these."
            )
        else:
            st.dataframe(fsum, use_container_width=True)
            st.caption(
                "One row per stock: how many reported numbers are stored, and "
                "the earliest and latest reporting period covered."
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


def _strategy_picker(prefix: str, label: str = "Strategy") -> str | None:
    """Dropdown of named strategies; returns the chosen name or None for manual.

    Every strategy shows its plain-English description under the picker, so
    the choice is informed. The Strategies page compares them side by side.
    """
    options = H.strategy_options()
    names = ["manual"] + [name for name, _ in options]
    descriptions = dict(options)
    choice = st.selectbox(
        label,
        names,
        key=f"{prefix}_strategy",
        format_func=lambda n: (
            "Custom — I'll choose the settings myself"
            if n == "manual"
            else H.strategy_display(n)
        ),
        help=(
            "Pick a ready-made strategy, or Custom to set every dial "
            "yourself. The Strategies page compares them side by side."
        ),
    )
    if choice != "manual":
        st.caption(descriptions[choice])
        st.caption(H.strategy_settings_summary(choice))
    return None if choice == "manual" else choice


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def page_strategies() -> None:
    st.title("Strategies")
    st.write(
        "A **strategy** is a complete, named recipe: which signals rank the "
        "stocks, how positions are sized, and how often the portfolio is "
        "rebalanced. Pick one below, compare them, then run any of them "
        "from the Backtest page — every strategy here is backtestable with "
        "one click."
    )

    st.divider()

    rows = H.strategy_comparison_rows()
    with st.container(border=True):
        st.subheader("Side-by-side comparison")
        st.dataframe(
            pd.DataFrame(rows).drop(columns=["name", "What it does"]),
            use_container_width=True,
        )
        st.caption(
            "Data used tells you what each strategy needs: 'Prices only' runs "
            "on daily stock prices; 'Prices + company financials' also uses "
            "numbers from SEC filings (slower to update, richer information)."
        )

    with st.container(border=True):
        st.subheader("What each strategy does, in plain English")
        for row in rows:
            with st.expander(row["Strategy"]):
                st.write(row["What it does"])
                st.write(f"**Data used:** {row['Data used']}")
                st.write(f"**Rebalance:** {row['Rebalance']} · "
                         f"**Holds:** {row['Up to']}")
                st.write(f"**Good at:** {row['Good at']}")
                st.write(f"**Main risk:** {row['Main risk']}")

    st.divider()
    st.info(
        "Honest note: every strategy here looked good *in the past*, on the "
        "data it was tested with. Markets change and popular ideas fade — "
        "use the Backtest page to check each one yourself before trusting it."
    )


# ---------------------------------------------------------------------------
# Compare strategies
# ---------------------------------------------------------------------------

def _years_between(start: str, end: str) -> float:
    """(end - start) in years; 0.0 when either date is unparseable."""
    try:
        return (
            date.fromisoformat(end) - date.fromisoformat(start)
        ).days / 365.25
    except ValueError:
        return 0.0


def page_compare() -> None:
    st.title("Compare strategies")
    st.write(
        "Every ready-made strategy runs **the same race**: the same stocks, "
        "the same dates, the same starting cash, and the same max weight per "
        "stock. Each strategy keeps its own recipe — which signals it uses, "
        "how often it rebalances, how it sizes positions, its drift buffer — "
        "so this is a fair head-to-head on the parts you choose."
    )

    st.divider()

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    three_years_ago = date.today().replace(
        year=date.today().year - 3
    ).isoformat()

    with st.container(border=True):
        c1, c2 = st.columns(2)
        start = c1.text_input(
            "From (YYYY-MM-DD)",
            value=three_years_ago,
            key="compare_start",
            help="First day of the comparison period for every strategy.",
        )
        end = c2.text_input(
            "To (YYYY-MM-DD)",
            value=date.today().isoformat(),
            key="compare_end",
            help="Last day of the comparison period for every strategy.",
        )
        c3, c4 = st.columns(2)
        capital = c3.number_input(
            "Starting cash ($)",
            value=100_000.0, step=10_000.0, min_value=1_000.0,
            key="compare_capital",
            help="The same starting cash every strategy gets.",
        )
        max_weight = c4.slider(
            "Max weight per stock (applies to all)",
            0.05, 0.50, 0.10, 0.05,
            key="compare_max_weight",
            help=(
                "Overrides each strategy's own preset for this comparison only: "
                "no single stock may exceed this share of the portfolio, which "
                "also decides how many stocks each holds. Everything else in "
                "each recipe stays untouched."
            ),
        )

        years = _years_between(start.strip(), end.strip())
        heavy = len(tickers) > 100 or years > 5
        confirmed = True
        if heavy:
            st.warning(
                f"Comparing {len(H.strategy_options())} strategies over "
                f"{len(tickers)} stocks and {years:.1f} years may take several "
                "minutes — the whole race runs before anything is shown."
            )
            confirmed = st.checkbox(
                "Yes, run the full comparison anyway.",
                key="compare_confirm",
            )

    if st.button("Compare strategies", type="primary", disabled=not confirmed):
        if years <= 0:
            st.error("The end date must be after the start date.")
            return
        with st.spinner("Racing every strategy over the same history..."):
            try:
                rows = H.compare_strategies_for_ui(
                    tickers, start.strip() or None, end.strip() or None,
                    capital, max_weight, db_path=DEFAULT_DB,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

        with st.container(border=True):
            st.subheader("Head-to-head results")
            table = pd.DataFrame(rows).drop(columns=["total_return", "sharpe"])
            st.dataframe(table, use_container_width=True)
            st.caption(
                "Every row raced the same stocks, dates and starting cash — "
                "only each strategy's own recipe differs."
            )

        chart_rows = [r for r in rows if r["total_return"] == r["total_return"]]
        if chart_rows:
            with st.container(border=True):
                st.subheader("Total return, side by side")
                chart = pd.DataFrame(
                    {
                        r["Strategy"]: r["total_return"]
                        for r in chart_rows
                    },
                    index=["Total return"],
                ).T
                st.bar_chart(chart)
                st.caption("Each bar is one strategy's total return over the period.")

        valid = chart_rows
        if valid:
            winner = max(valid, key=lambda r: r["total_return"])
            st.success(
                f"**{winner['Strategy']}** did best: it earned the highest "
                f"total return over this period ({winner['Total return']}, "
                f"about {winner['Yearly growth (CAGR)']} per year), with a "
                f"worst fall of {winner['Worst fall']} along the way. "
                "Remember: the past doesn't predict the future — check the "
                "Strategies page for what each recipe is actually betting on."
            )
        else:
            st.info("No strategy produced a measurable return over this period.")

        st.warning(
            "Honest reminder: a head-to-head on the past shows what *would* "
            "have happened, not what will. Treat the winner as a starting "
            "point for research, not a recommendation."
        )


# ---------------------------------------------------------------------------
# Signals (page)
# ---------------------------------------------------------------------------

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

    st.divider()

    tickers = H.db_tickers()
    if not tickers:
        st.info(
            "No price data yet — go to the Market data page and download "
            "some tickers first."
        )
        return

    with st.container(border=True):
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

    with st.container(border=True):
        st.subheader("Latest scores")
        table = H.latest_signal_table(signal_name, chosen)
        if table.empty:
            st.info("No computed values yet — press *Compute signals* above.")
        else:
            st.dataframe(table, use_container_width=True)
            st.caption(
                "The most recent score for each stock and the date it is as of. "
                "Higher = more attractive."
            )

    with st.container(border=True):
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

    st.divider()

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    with st.container(border=True):
        strategy_name = _strategy_picker("backtest")
        if strategy_name is None:
            signal_name, _ = _signal_picker(label="Rank stocks by", key="backtest_signal")
            c1, c2, c3 = st.columns(3)
            weighting_key = c1.selectbox(
                "Position sizing",
                ["equal", "signal", "inverse_vol"],
                format_func=lambda k: H.WEIGHTING_LABELS[k],
            )
            rebalance = c2.radio("Rebalance", ["monthly", "weekly"], horizontal=True,
                                 help="How often the portfolio is adjusted back to its targets.")
            max_weight = c3.slider(
                "Max weight per stock",
                0.05, 0.50, 0.20, 0.05,
                help="No single stock may exceed this share of the portfolio. "
                     "This one setting also decides how many stocks are held.",
            )
            st.caption(H.weighting_description(weighting_key))
            c3.caption(f"→ {H.implied_stock_count(max_weight)} stocks at this setting")
        else:
            st.info(
                "Running the ready-made strategy above with its own settings. "
                "Switch to *Custom* to set every dial yourself."
            )

    with st.container(border=True):
        c4, c5 = st.columns(2)
        cost_bps = c4.slider(
            "Cost per trade",
            0.0, 50.0, 10.0, 1.0,
            help="Trading cost in basis points. 10 = 0.10% of each trade's value lost to costs.",
        )
        capital = c5.number_input(
            "Starting cash ($)", value=100_000.0, step=10_000.0, min_value=1_000.0,
        )
        c7, c8 = st.columns(2)
        start = c7.text_input("From (optional, YYYY-MM-DD)", value="",
                              help="Leave empty to start at the earliest available data.")
        end = c8.text_input("To (optional, YYYY-MM-DD)", value="",
                            help="Leave empty to end at the latest available data.")

        cost_model = st.radio(
            "Trading cost model",
            options=["flat", "spread"],
            format_func=lambda k: (
                "Simple flat fee" if k == "flat"
                else "Realistic price-gap estimate"
            ),
            help=(
                "'Simple flat fee' charges a fixed fee on every trade — 10 means "
                "0.10% — and keeps results identical to earlier versions. "
                "'Realistic price-gap estimate' estimates the real cost of each "
                "trade from the stock's recent high-low price ranges, plus extra "
                "for large trades in jumpy stocks; more realistic, but needs "
                "enough price history to work."
            ),
        )

    if st.button("Run backtest", type="primary"):
        with st.spinner("Replaying history..."):
            try:
                if strategy_name is None:
                    result, metrics = H.run_backtest_for_ui(
                        signal_name, tickers, weighting_key, max_weight,
                        rebalance, cost_bps, capital,
                        start.strip() or None, end.strip() or None, DEFAULT_DB,
                        cost_model=cost_model,
                    )
                else:
                    result, metrics = H.run_strategy_backtest_for_ui(
                        strategy_name, tickers, cost_bps, capital,
                        start.strip() or None, end.strip() or None, DEFAULT_DB,
                        cost_model=cost_model,
                    )
            except ValueError as exc:
                st.error(str(exc))
                return

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final value",
                      H.metric_display("final_value", metrics["final_value"]["value"]))
            m2.metric("Total return",
                      H.metric_display("total_return", metrics["total_return"]["value"]))
            m3.metric("Yearly growth (CAGR)",
                      H.metric_display("cagr", metrics["cagr"]["value"]))
            m4.metric("Worst fall",
                      H.metric_display("max_drawdown", metrics["max_drawdown"]["value"]))
            st.caption(
                "Trading costs: "
                + (
                    "simple flat fee — 10 means 0.10% per trade"
                    if result.cost_model_name == "flat"
                    else "realistic price-gap estimate from each stock's "
                         "recent high-low price ranges"
                )
                + "."
            )

        with st.container(border=True):
            st.subheader("Portfolio value over time")
            equity = result.equity.copy()
            equity["date"] = pd.to_datetime(equity["date"])
            st.line_chart(equity.set_index("date")["portfolio_value"])
            st.caption("The simulated account value on each trading day.")

        with st.container(border=True):
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
# Backtest wizard
# ---------------------------------------------------------------------------

_WIZARD_STEPS = [
    "What is a backtest?",
    "Pick a strategy",
    "Pick dates",
    "Run",
    "Results",
]

_WIZARD_KEYS = (
    "wizard_step", "wizard_strategy", "wizard_start", "wizard_end",
    "wizard_result", "wizard_metrics",
)


def _wizard_default_strategy() -> str:
    """First registered strategy, or 'manual'-style None when none exist."""
    options = H.strategy_options()
    return options[0][0] if options else ""


def _wizard_reset() -> None:
    """Clear all wizard state and go back to step 1."""
    for key in _WIZARD_KEYS:
        st.session_state.pop(key, None)
    st.session_state["wizard_step"] = 0


def _wizard_explanation(
    strategy_label: str, start: str, end: str, metrics: dict
) -> str:
    """2–3 plain-English sentences describing what the wizard's backtest did."""
    final = H.metric_display("final_value", metrics["final_value"]["value"])
    total = H.metric_display("total_return", metrics["total_return"]["value"])
    cagr = H.metric_display("cagr", metrics["cagr"]["value"])
    fall = H.metric_display("max_drawdown", metrics["max_drawdown"]["value"])
    trades = metrics["num_trades"]["value"]
    trade_bit = (
        f"It made {int(round(trades))} trades along the way."
        if trades == trades
        else "It traded as the signals changed."
    )
    return (
        f"Following **{strategy_label}** from {start} to {end} would have "
        f"turned $100,000 into {final} — a total return of {total}, or about "
        f"{cagr} per year. Along the way the account fell as much as {fall} "
        f"from its own peak, which is the worst losing stretch you would "
        f"have had to sit through. {trade_bit}"
    )


def page_wizard() -> None:
    st.title("Backtest wizard")
    st.write(
        "New here? This walks you through your first backtest in five small "
        "steps — no finance background needed."
    )

    st.divider()

    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 0
    step = st.session_state["wizard_step"]
    st.progress(
        (step + 1) / len(_WIZARD_STEPS),
        text=f"Step {step + 1} of {len(_WIZARD_STEPS)}: {_WIZARD_STEPS[step]}",
    )

    tickers = H.db_tickers()

    with st.container(border=True):
        if step == 0:
            st.subheader("What is a backtest?")
            st.write(
                "A **backtest** replays history. You pick a recipe for choosing "
                "stocks (a *strategy*), and the computer pretends you followed "
                "it in the past — buying and selling at the prices of the day, "
                "with trading costs included — and shows what *would* have "
                "happened to your money. It is the standard way to check an "
                "idea *before* trusting it with real decisions."
            )
            st.write(
                "One honest caveat up front: a backtest uses today's knowledge "
                "of the past. It can filter out bad ideas, but it cannot prove "
                "an idea will work in the future."
            )
            if not tickers:
                st.info(
                    "You'll need some price data first — the Market data page "
                    "can download it in a minute."
                )
        elif step == 1:
            st.subheader("Pick a strategy")
            st.write(
                "A **strategy** is a complete, named recipe: which signals rank "
                "the stocks, how positions are sized, and how often the "
                "portfolio is rebalanced. Pick the one that sounds most "
                "sensible to you — you can compare them all later on the "
                "Compare strategies page."
            )
            options = H.strategy_options()
            if not options:
                st.error("No strategies are registered.")
            else:
                descriptions = dict(options)
                if "wizard_strategy" not in st.session_state:
                    st.session_state["wizard_strategy"] = options[0][0]
                choice = st.radio(
                    "Strategy",
                    [name for name, _ in options],
                    format_func=H.strategy_display,
                    key="wizard_strategy",
                    help="Pick a ready-made recipe. The description below says "
                         "what it does in plain English.",
                )
                st.write(descriptions[choice])
                st.caption(H.strategy_settings_summary(choice))
        elif step == 2:
            st.subheader("Pick dates")
            st.write(
                "Choose the stretch of history to replay. Longer is usually more "
                "informative — three years is a good starting point — but every "
                "day in the range needs price data for your stocks."
            )
            three_years_ago = date.today().replace(
                year=date.today().year - 3
            ).isoformat()
            st.session_state.setdefault("wizard_start", three_years_ago)
            st.session_state.setdefault("wizard_end", date.today().isoformat())
            st.text_input(
                "From (YYYY-MM-DD)",
                key="wizard_start",
                help="First day of the replay.",
            )
            st.text_input(
                "To (YYYY-MM-DD)",
                key="wizard_end",
                help="Last day of the replay.",
            )
        elif step == 3:
            st.subheader("Run")
            strategy_name = st.session_state.get("wizard_strategy") or \
                _wizard_default_strategy()
            start = (st.session_state.get("wizard_start") or "").strip() or None
            end = (st.session_state.get("wizard_end") or "").strip() or None
            st.write(
                f"Ready to replay **{H.strategy_display(strategy_name)}** from "
                f"{start or 'the earliest data'} to {end or 'the latest data'} "
                f"on {len(tickers)} stock(s), starting with $100,000 of pretend "
                "money."
            )
            if st.button("Run backtest", type="primary", key="wizard_run"):
                if not tickers:
                    st.error(
                        "No price data yet — go to the Market data page first."
                    )
                else:
                    with st.spinner("Replaying history..."):
                        try:
                            result, metrics = H.run_strategy_backtest_for_ui(
                                strategy_name, tickers, 10.0, 100_000.0,
                                start, end, DEFAULT_DB,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["wizard_result"] = result
                            st.session_state["wizard_metrics"] = metrics
                            st.session_state["wizard_step"] = 4
                            st.rerun()
        else:
            st.subheader("Results")
            metrics = st.session_state.get("wizard_metrics")
            if metrics is None:
                st.info("Nothing to show yet — go back and run the backtest.")
            else:
                strategy_name = st.session_state.get("wizard_strategy") or \
                    _wizard_default_strategy()
                start = st.session_state.get("wizard_start") or "the earliest data"
                end = st.session_state.get("wizard_end") or "the latest data"
                m1, m2 = st.columns(2)
                m1.metric(
                    "Final value",
                    H.metric_display("final_value", metrics["final_value"]["value"]),
                )
                m2.metric(
                    "Total return",
                    H.metric_display("total_return", metrics["total_return"]["value"]),
                )
                result = st.session_state.get("wizard_result")
                if result is not None:
                    equity = result.equity.copy()
                    equity["date"] = pd.to_datetime(equity["date"])
                    st.line_chart(equity.set_index("date")["portfolio_value"])
                    st.caption("The pretend account's value on each trading day.")
                st.write(
                    _wizard_explanation(
                        H.strategy_display(strategy_name), start, end, metrics
                    )
                )
                st.warning(
                    "Honest reminder: this is what *would* have happened, not a "
                    "prediction. Markets change, and popular ideas fade — use "
                    "backtests to filter out bad ideas, not to prove future "
                    "profits."
                )

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("← Back", key="wizard_back", disabled=step == 0):
        st.session_state["wizard_step"] = step - 1
        st.rerun()
    if step < 4 and b2.button("Next →", key="wizard_next", type="primary"):
        if step == 1 and not st.session_state.get("wizard_strategy"):
            st.error("Pick a strategy first.")
        elif not tickers:
            st.error("Download some price data first (Market data page).")
        else:
            st.session_state["wizard_step"] = step + 1
            st.rerun()
    if b3.button("Start over", key="wizard_reset"):
        _wizard_reset()
        st.rerun()


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def _portfolio_controls(prefix: str, asof_help: str | None = None) -> dict:
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
        help=asof_help or "Leave empty to use the latest date with signal data.",
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

    st.divider()

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    with st.container(border=True):
        strategy_name = _strategy_picker("portfolio")
        if strategy_name is None:
            settings = _portfolio_controls("portfolio")
            max_weight = st.slider(
                "Max weight per stock",
                0.05, 0.50, 0.20, 0.05,
                help="No single stock may exceed this share of the portfolio. "
                     "This one setting also decides how many stocks are held.",
            )
            st.caption(
                f"→ {H.implied_stock_count(max_weight)} stocks at this setting"
            )
        else:
            asof_only = st.text_input(
                "As of date (YYYY-MM-DD)",
                value="",
                key="portfolio_strategy_date",
                help="Only signals dated on or before this day are used. "
                     "Leave empty for the latest available date.",
            )

    if st.button("Build portfolio", type="primary"):
        try:
            freshness_caption = None
            if strategy_name is None:
                asof = settings["asof"].strip() or None
                if asof is None:
                    matrix = H.cached_load_signal_matrix(
                        settings["signal_name"], tickers, db_path=DEFAULT_DB
                    )
                    if matrix.empty:
                        raise ValueError("No signal data — compute signals first.")
                    asof = matrix.index.max().strftime("%Y-%m-%d")
                portfolio, scores, day = H.build_target_for_ui(
                    settings["signal_name"], settings["method_key"], asof,
                    tickers, DEFAULT_DB, max_weight=max_weight,
                )
            else:
                asof = asof_only.strip() or None
                if asof is None:
                    matrices = H.cached_strategy_signal_matrices(
                        strategy_name, tickers, db_path=DEFAULT_DB
                    )
                    latest = min(m.index.max() for m in matrices.values())
                    asof = latest.strftime("%Y-%m-%d")
                portfolio, scores, day, leg_dates = H.build_strategy_target_for_ui(
                    strategy_name, asof, tickers, DEFAULT_DB,
                )
                legs = "; ".join(
                    f"{H.display_name(n)}: {d}" for n, d in leg_dates.items()
                )
                freshness_caption = (
                    f"Shown as of the oldest input ({day}); "
                    f"each input's latest data: {legs}."
                )
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.container(border=True):
            st.subheader(f"Target portfolio as of {day}")
            if freshness_caption:
                st.caption(freshness_caption)
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
            # Plain-English reason the trade was made, stored by the broker
            # when the order was placed (Feature 5).
            "Why": t.get("reason", "") or "—",
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

    st.divider()

    tickers = H.db_tickers()
    if not tickers:
        st.info("No price data yet — go to the Market data page first.")
        return

    broker = PaperBroker(DEFAULT_DB)
    pos_df, trade_df, total = _paper_account_tables(broker)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash", H.fmt_money(broker.cash))
        c2.metric("Positions value", H.fmt_money(total - broker.cash))
        c3.metric("Total account", H.fmt_money(total))

    with st.container(border=True):
        st.subheader("Positions")
        if pos_df.empty:
            st.info("No positions yet — the account is all cash.")
        else:
            st.dataframe(pos_df, use_container_width=True)

    with st.container(border=True):
        st.subheader("Recent trades")
        if trade_df.empty:
            st.info("No trades yet.")
        else:
            st.dataframe(trade_df, use_container_width=True)

    with st.container(border=True):
        st.subheader("Preview a rebalance")
        st.write(
            "See exactly what *would* be traded to move the paper account to a "
            "target portfolio — including which orders the safety checks would "
            "block — without trading anything."
        )
        strategy_name = _strategy_picker("paper")
        if strategy_name is None:
            settings = _portfolio_controls(
                "paper",
                asof_help=(
                    "Signal cutoff date: only signals dated on or before this day are "
                    "used. The trade executes at the next trading day's close — a "
                    "signal isn't known until after its day's closing bell. Leave "
                    "empty to trade at the latest available close using prior-day "
                    "signals (the same no-peeking rule as the backtest)."
                ),
            )
        else:
            asof_only = st.text_input(
                "As of date (YYYY-MM-DD)",
                value="",
                key="paper_strategy_date",
                help=(
                    "Signal cutoff date: only signals dated on or before this day "
                    "are used. Leave empty to trade at the latest available close "
                    "using prior-day signals."
                ),
            )
            settings = {"asof": asof_only}
        if st.button("Preview rebalance"):
            try:
                asof = settings["asof"].strip() or None
                trading_day, signal_cutoff = H.resolve_paper_timing(
                    DEFAULT_DB, asof
                )
                if strategy_name is None:
                    portfolio, _, signal_day = H.build_target_for_ui(
                        settings["signal_name"], settings["method_key"],
                        signal_cutoff, tickers, DEFAULT_DB,
                    )
                else:
                    portfolio, _, signal_day, leg_dates = H.build_strategy_target_for_ui(
                        strategy_name, signal_cutoff, tickers, DEFAULT_DB,
                    )
                preview, notes = H.preview_rebalance_orders(
                    broker, portfolio.weights, trading_day, total,
                    cost_bps=broker.cost_bps,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            if strategy_name is not None:
                legs = "; ".join(
                    f"{H.display_name(n)}: {d}" for n, d in leg_dates.items()
                )
                st.caption(f"Each input's latest data: {legs}.")
            st.caption(
                f"Trading {trading_day} on signals as of {signal_day} "
                f"(signals dated before {trading_day} only — no peeking)."
            )
            if not preview:
                st.success("Already at target — no trades needed.")
            else:
                st.dataframe(pd.DataFrame(preview), use_container_width=True)
                st.caption(
                    "Dry run complete — the paper account was not touched. "
                    "Blocked orders explain why in plain English."
                )
            for note in notes:
                st.info(f"Note: {note}")

    with st.container(border=True):
        st.subheader("Execute rebalance (paper money only)")
        confirm_exec = st.checkbox(
            "I understand this trades the PAPER account with fake money.",
            key="paper_exec_confirm",
        )
        if st.button("Execute paper rebalance", type="primary",
                     disabled=not confirm_exec):
            try:
                asof = settings["asof"].strip() or None
                trading_day, signal_cutoff = H.resolve_paper_timing(
                    DEFAULT_DB, asof
                )
                if strategy_name is None:
                    portfolio, _, _ = H.build_target_for_ui(
                        settings["signal_name"], settings["method_key"],
                        signal_cutoff, tickers, DEFAULT_DB,
                    )
                else:
                    portfolio, _, _, _ = H.build_strategy_target_for_ui(
                        strategy_name, signal_cutoff, tickers, DEFAULT_DB,
                    )
                # Execute the RAW orders (full precision) — never re-parsed
                # display strings, which round shares and would silently alter
                # or break small orders.
                orders = H.compute_rebalance_orders(
                    broker, portfolio.weights, trading_day, total,
                    cost_bps=broker.cost_bps,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            if not orders:
                st.success("Already at target — no trades needed.")
            else:
                for order in orders:
                    result = broker.place_order(
                        order.ticker, order.action, order.shares, trading_day,
                        reason=order.reason,
                    )
                    if result.filled:
                        verb = "Bought" if result.action == "BUY" else "Sold"
                        st.write(
                            f"{verb} {result.shares:,.2f} shares of "
                            f"{result.ticker} at {H.fmt_money(result.price)} "
                            f"(fees {H.fmt_money(result.cost)})."
                        )
                    else:
                        st.warning(f"Skipped {order.action} {order.ticker}: "
                                   f"{result.reason}")
                st.success("Rebalance executed in the paper account. "
                           "Reminder: simulated money only.")
                st.rerun()

    with st.container(border=True):
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

    st.divider()

    with st.container(border=True):
        st.subheader("Current safety settings")
        st.dataframe(
            pd.DataFrame(H.risk_config_rows(RiskConfig())),
            use_container_width=True,
        )

    with st.container(border=True):
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
                    matrix = H.cached_load_signal_matrix(
                        settings["signal_name"], tickers, db_path=DEFAULT_DB
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
            st.write(f"Checked the target portfolio as of {day}.")
            ok, reasons = H.check_target_portfolio(portfolio)
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

    with st.container(border=True):
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
    "Strategies": page_strategies,
    "Compare strategies": page_compare,
    "Backtest": page_backtest,
    "Backtest wizard": page_wizard,
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
    st.sidebar.caption(
        "Every strategy is explained in plain English on the Strategies page."
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
