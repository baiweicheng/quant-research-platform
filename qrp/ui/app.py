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

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Streamlit puts the script's own folder on sys.path, not the project root,
# so make `import qrp...` work no matter how the app is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from qrp.data.store import DEFAULT_DB
from qrp.data import sp500_update as _SP500U
from qrp.execution.alpaca import AlpacaError
from qrp.execution.paper import PaperBroker, PaperBrokerAdapter, price_trading_days
from qrp.risk.guards import RiskConfig
from qrp.signals import default_registry
from qrp.ui import helpers as H
from qrp.ops import audit as OPS_AUDIT
from qrp.ops import killswitch as OPS_KS
from qrp.ops import pipeline as OPS_PIPE
from qrp.ops import pretrade as OPS_PRE


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
8. **Trust but verify** (Data health page) — scan your stored prices for
   stale data, suspicious moves and split/dividend mismatches, keep
   permanent stock IDs, and back up the database.
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

def _render_refresh_summary(status: dict, scope: str = "prices") -> None:
    """Show a finished background refresh's summary (and fill report).

    `scope` is "prices" or "filings". Only the prices job runs the
    pending-order fill pass, so only it shows a fill report.
    """
    summary = status.get("summary") or {}
    lines = summary.get("lines") or []
    if summary.get("up_to_date"):
        st.success(lines[0] if lines else "Everything is up to date.")
    else:
        for line in lines:
            # Plain if/else on purpose: a ternary here returns the
            # DeltaGenerator from st.success/st.info, and Streamlit's
            # magic would render that object onto the page.
            if line.startswith("Added"):
                st.success(line)
            else:
                st.info(line)
    if scope != "prices":
        return
    fill_report = status.get("fill_report") or {}
    if fill_report.get("error"):
        st.warning(fill_report["error"])
    if fill_report.get("filled"):
        st.success(
            f"Filled {len(fill_report['filled'])} pending paper "
            f"order(s) at the new market open — see the Paper "
            f"trading page for details."
        )
    if fill_report.get("skipped"):
        st.warning(
            f"{len(fill_report['skipped'])} pending paper order(s) "
            f"could not fill — see the Paper trading page."
        )


def _render_sp500_membership_section() -> None:
    """The "Check for S&P 500 changes" block on the Market data page.

    Check fetches Wikipedia's live pages and reports the diff in plain
    English; Apply appears only when changes were found and only ever
    runs after the user reviews the report. Check state (including the
    pending diff) persists in a status file so navigating away never
    loses it. Fetch failures show a plain-English note, never a
    traceback, and never write partial data.
    """
    status = H.sp500_check_status(DEFAULT_DB)
    last = status.get("last_checked")
    st.caption(
        f"Membership last checked: {last}"
        if last
        else "Membership last checked: not checked yet"
    )
    if st.button("Check for S&P 500 changes", key="sp500_check"):
        with st.spinner("Checking Wikipedia for index changes…"):
            try:
                diff = _SP500U.check_for_changes()
            except _SP500U.MembershipCheckError as exc:
                H.sp500_record_check(DEFAULT_DB, "error", str(exc))
                st.rerun()
            H.sp500_record_check(
                DEFAULT_DB,
                "up_to_date" if diff["up_to_date"] else "changes_found",
                pending=None if diff["up_to_date"] else diff,
            )
            st.rerun()

    status = H.sp500_check_status(DEFAULT_DB)
    outcome = status.get("outcome")
    if outcome == "error":
        st.error(
            status.get("detail") or "Couldn't check right now — try again later."
        )
    elif outcome == "up_to_date":
        st.success(
            "Your S&P 500 list is current — last checked "
            f"{status.get('last_checked')}."
        )
    elif outcome == "changes_found" and status.get("pending"):
        diff = status["pending"]
        st.warning(
            "Found changes to review. Nothing has been applied yet — "
            "read the list, then press Apply changes if it looks right."
        )
        for line in _SP500U.describe_diff(diff):
            st.info(line)
        rename_text = st.text_area(
            "Declare a ticker rename (optional)",
            key="sp500_renames",
            placeholder="e.g. PSKY -> SKYD",
            help=(
                "If one of the changes above was actually a ticker rename "
                "(the same company trading under a new symbol) rather than "
                "a company leaving and another joining, declare it here as "
                "'OLD -> NEW' (one per line). Renames keep the company's "
                "original start date."
            ),
        )
        if st.button("Apply changes", key="sp500_apply", type="primary"):
            with st.spinner("Applying membership changes…"):
                try:
                    summary = _SP500U.apply_updates(diff, rename_text)
                except _SP500U.MembershipCheckError as exc:
                    st.error(str(exc))
                    return
            H.sp500_record_check(
                DEFAULT_DB, "applied", json.dumps(summary), pending=None
            )
            st.rerun()
    elif outcome == "applied":
        try:
            summary = json.loads(status.get("detail") or "{}")
        except ValueError:
            summary = {}
        st.success("Membership changes applied.")
        for line in summary.get("summary_lines", []):
            st.info(line)
        if summary.get("new_tickers"):
            st.write(
                "**New tickers:** " + ", ".join(summary["new_tickers"])
            )
        st.write(summary.get("next_steps", ""))
        st.caption(summary.get("git_note", ""))


def _render_refresh_section(scope: str) -> None:
    """One half of the Refresh data block: prices or SEC filings.

    Starts the background job for `scope` and polls its status file, so
    navigating away never stops a run. The two scopes are independent —
    either may run while the other is going.
    """
    is_prices = scope == "prices"
    title = "Refresh prices" if is_prices else "Refresh SEC filings"
    st.markdown(f"**{title}**")
    if is_prices:
        st.write(
            "Your daily catch-up — new daily prices (and dividends) since "
            "your last pull. After prices land, any pending paper orders "
            "are checked for filling."
        )
    else:
        st.write(
            "Only needed before rebalancing or recomputing signals — "
            "company financials update quarterly, not daily."
        )
        last_checked = H.refresh_filings_last_checked(DEFAULT_DB)
        if last_checked:
            st.caption(f"Filings last checked: {last_checked}.")
        else:
            st.caption("Filings last checked: not checked yet.")
    # The job runs on a background thread in the server process (see
    # qrp/data/refresh_job.py): this page only starts it and polls the
    # status file, so navigating away never stops a run.
    status = H.refresh_job_status(DEFAULT_DB, scope)
    state = status.get("state", "idle")
    if state == "running":
        done = status.get("done", 0)
        total = status.get("total", 0) or 1
        st.progress(
            min(done / total, 1.0),
            text=status.get("step") or "Working…",
        )
        st.caption(
            "This keeps running even if you switch pages or close the "
            "browser — this page will show the summary when it's done."
        )
        time.sleep(2)
        st.rerun()
    elif state in ("error", "interrupted"):
        st.error(
            status.get("error")
            or "The refresh ran into a problem."
        )
        if st.button(f"Try the {scope} refresh again", type="primary"):
            H.refresh_job_start(DEFAULT_DB, scope)
            st.rerun()
    else:
        if state == "done":
            _render_refresh_summary(status, scope)
        if st.button(title, type="primary"):
            H.refresh_job_start(DEFAULT_DB, scope)
            st.rerun()


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
        "daily like prices. Four strategies need them: **Trending Value "
        "(O'Shaughnessy)**, **Value + Momentum Composite (AQR-Style)**, "
        "**Quality + Low Volatility Composite** and **Twin Momentum "
        "(Price + Earnings Surprise)** (the "
        "Strategies page says which)."
    )

    st.divider()

    with st.container(border=True):
        st.subheader("Download data")
        sp500 = H.load_sp500_tickers()
        sp500_all = H.load_sp500_universe(include_historical=True)
        universe = st.radio(
            "Which stocks?",
            options=["sp500", "custom"],
            format_func=lambda k: (
                f"S&P 500 — {len(sp500)} companies today"
                + (
                    f" (+{len(sp500_all) - len(sp500)} past members)"
                    if len(sp500_all) > len(sp500)
                    else ""
                )
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
        include_historical = True
        if universe == "sp500":
            include_historical = st.checkbox(
                "Include past index members (recommended for backtests)",
                value=True,
                help=(
                    "Also download price history for stocks that were in the "
                    "S&P 500 in the past but aren't today. Backtests need "
                    "them: on any historical date only that date's actual "
                    "index members can be bought. Many delisted tickers still "
                    "serve their history on Yahoo Finance; ones that don't "
                    "are simply skipped. Unticking downloads only today's "
                    "500 — faster, but historical backtests will then carry "
                    "survivorship bias."
                ),
            )
        col1, col2 = st.columns(2)
        want_prices = col1.checkbox(
            "Prices (daily)",
            value=True,
            help=(
                "Daily open, high, low, close and volume from Yahoo Finance. "
                "Free, no key. Everything on this platform needs prices. "
                "Dividend payouts come along automatically, so backtests "
                "and paper trading count the cash you would really receive."
            ),
        )
        want_fund = col2.checkbox(
            "Fundamentals (SEC filings)",
            value=True,
            help=(
                "Revenue, profit, assets, debt and more from official SEC "
                "filings. Free, no key. Updates quarterly. Needed by Trending "
                "Value (O'Shaughnessy), Value + Momentum Composite "
                "(AQR-Style), Quality + Low Volatility Composite and Twin "
                "Momentum (Price + Earnings Surprise)."
            ),
        )
        want_spy = st.checkbox(
            "Also download SPY (S&P 500 benchmark)",
            value=True,
            help=(
                "SPY is the exchange-traded fund that tracks the S&P 500 — "
                "the 500 largest US companies. Downloading it lets the "
                "Backtest and Compare pages show your strategy next to "
                "'just owning the whole market'. It's one extra ticker, "
                "so it adds almost no download time."
            ),
        )
        start = end = None
        if want_prices:
            d1, d2 = st.columns(2)
            start = d1.text_input(
                "From date",
                value="2013-01-01",
                help=(
                    "First day to download, as YYYY-MM-DD. 2013 gives the "
                    "signal lookbacks (up to 253 trading days) enough history "
                    "so a backtest starting in 2015 is fully populated from "
                    "day one."
                ),
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
                tickers = H.load_sp500_universe(
                    include_historical=include_historical
                )
            if not tickers:
                st.error("Type at least one ticker symbol first.")
            elif not (want_prices or want_fund):
                st.error("Tick at least one of Prices or Fundamentals first.")
            else:
                if want_prices and want_spy and "SPY" not in tickers:
                    # The S&P 500 ETF is the benchmark ticker the Backtest
                    # and Compare pages measure strategies against.
                    tickers = [*tickers, "SPY"]
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
        st.subheader("S&P 500 membership")
        st.write(
            "Which companies count as S&P 500 members on any date — this "
            "decides which stocks your backtests may buy, so they never "
            "peek at future winners. The index changes a few times a month "
            "(buyouts, spinoffs, ticker renames), and the list saved in "
            "this repo can lag behind. This checks Wikipedia's live pages "
            "for changes. Nothing is applied without your review."
        )
        _render_sp500_membership_section()

    with st.container(border=True):
        st.subheader("Refresh data")
        st.write(
            "Two quick catch-ups — no dates to type, and each only "
            "downloads what arrived since your last pull. Both run in the "
            "background, so you can keep using the app — or even close the "
            "browser — and come back here for the summary. (For a full "
            "fresh download of all history, use Download data above.)"
        )
        # Each refresh runs on a background thread in the server process
        # (see qrp/data/refresh_job.py): this page only starts the jobs
        # and polls their status files, so navigating away never stops a
        # run. Prices and filings are independent jobs — either may run
        # while the other is going.
        col_prices, col_filings = st.columns(2)
        with col_prices:
            _render_refresh_section("prices")
        with col_filings:
            _render_refresh_section("filings")

    with st.container(border=True):
        st.subheader("Factor data (Ken French)")
        st.write(
            "Academic researchers found that most of what stocks do can be "
            "explained by a handful of broad **tilts**: the market as a "
            "whole, small vs big companies, cheap vs expensive stocks, "
            "profitable vs unprofitable companies, careful vs aggressive "
            "spenders, and momentum (recent winners vs losers). This free "
            "daily dataset from Professor Ken French's data library at "
            "Dartmouth (no key needed) lets the Compare page show which "
            "tilts each strategy was actually riding — and whether anything "
            "was left over beyond them. Download it once; re-downloading "
            "refreshes it with the latest data."
        )
        status = H.french_factors_status_for_ui(DEFAULT_DB)
        if status["downloaded"]:
            st.success(
                f"Factor data stored: {status['first']} to {status['last']} "
                f"({status['rows']:,} trading days)."
            )
        else:
            st.info("Factor data not downloaded yet.")
        if st.button("Download factor data", type="primary"):
            with st.spinner("Downloading the free factor data from Dartmouth..."):
                try:
                    info = H.download_french_factors_for_ui(DEFAULT_DB)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        f"Stored {info['rows']:,} days of factor data "
                        f"({info['first']} to {info['last']}). The Compare "
                        "page can now explain each strategy's returns."
                    )
                    st.rerun()

    with st.container(border=True):
        st.subheader("Last checked")
        fetch_df = H.fundamentals_fetch_summary()
        if fetch_df.empty:
            st.info(
                "Nothing checked yet — press *Refresh SEC filings* above "
                "after downloading fundamentals at least once."
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
        format_func=lambda n: registry.display_name(n),
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
        with st.expander("How it is calculated"):
            st.write(H.strategy_calculation(choice))
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

    H.require_data("prices")

    rows = H.strategy_comparison_rows()
    with st.container(border=True):
        st.subheader("Side-by-side comparison")
        st.dataframe(
            pd.DataFrame(rows).drop(
                columns=["name", "What it does", "Calculation"]
            ),
            use_container_width=True,
        )
        st.caption(
            "Data used tells you what each strategy needs: 'Prices only' runs "
            "on daily stock prices; 'Prices + company financials' also uses "
            "numbers from SEC filings (slower to update, richer information)."
        )

    with st.container(border=True):
        st.subheader("What each strategy does, in plain English")
        st.caption(
            "Grouped by when each batch was added, oldest first — the "
            "original technical strategies, then the fundamental blends, "
            "then the newest research batch."
        )
        current_era = None
        for row in rows:
            if row["Era"] != current_era:
                current_era = row["Era"]
                st.markdown(f"**{current_era}**")
            with st.expander(row["Strategy"]):
                st.write(row["What it does"])
                st.write(f"**Data used:** {row['Data used']}")
                st.write(f"**Rebalance:** {row['Rebalance']} · "
                         f"**Holds:** {row['Up to']}")
                if row["Cutoff filter"] != "—":
                    st.write(f"**Cutoff filter:** {row['Cutoff filter']}")
                st.write(f"**Good at:** {row['Good at']}")
                st.write(f"**Main risk:** {row['Main risk']}")
                with st.expander("How it is calculated"):
                    st.write(row["Calculation"])

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


# Default first day of a comparison. Market data downloads from 2013 so the
# signals' lookbacks are warmed up, but comparisons start at 2015 — the first
# year every signal can actually score.
_COMPARE_DEFAULT_START = "2015-01-01"

# Session-state key holding the Compare page's last-used inputs as a plain
# dict (NOT widget state). Streamlit wipes keyed widget values when a page is
# left, so without this the input boxes would reset to defaults on return
# and the remembered results below could never match again.
_COMPARE_INPUTS_KEY = "compare_inputs"


def _compare_input_defaults(
    data_start: str | None, data_end: str | None
) -> dict:
    """Default Compare inputs (used on a first visit with no saved inputs)."""
    default_start = (
        data_start
        if data_start and data_start > _COMPARE_DEFAULT_START
        else _COMPARE_DEFAULT_START
    )
    return {
        "compare_start": default_start,
        "compare_end": data_end or "",
        "compare_capital": 100_000.0,
        "compare_max_weight": 0.05,
        "compare_frequency": "monthly",
    }


def _restore_compare_inputs(defaults: dict) -> None:
    """Re-seed Compare widget keys from the last run's saved inputs.

    Must run BEFORE the widgets below are created. ``setdefault`` only fills
    keys that are missing (i.e. widget state was wiped by leaving the page),
    so it never clobbers live widget values — changing an input still
    produces a new cache key and invalidates the remembered results.
    """
    saved = st.session_state.get(_COMPARE_INPUTS_KEY) or {}
    for key, default in defaults.items():
        st.session_state.setdefault(key, saved.get(key, default))


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

    tickers = H.require_data("prices", "signals")["tickers"]

    data_start, data_end = H.data_date_range(DEFAULT_DB)

    # Sticky inputs: re-seed the widget keys from the last run before the
    # widgets below are created, so leaving the page and coming back keeps
    # your inputs (and the remembered results that match them).
    defaults = _compare_input_defaults(data_start, data_end)
    _restore_compare_inputs(defaults)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        start = c1.text_input(
            "From (YYYY-MM-DD)",
            value=defaults["compare_start"],
            key="compare_start",
            help="First day of the comparison period for every strategy. "
                 "Defaults to 2015-01-01: market data downloads from 2013 "
                 "so the signals are warmed up, and comparisons start at "
                 "2015.",
        )
        end = c2.text_input(
            "To (YYYY-MM-DD)",
            value=data_end or "",
            key="compare_end",
            help="Last day of the comparison period for every strategy. "
                 "Defaults to your latest downloaded data.",
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
            0.01, 0.50, 0.05, 0.01,
            key="compare_max_weight",
            help=(
                "Overrides each strategy's own preset for this comparison only: "
                "no single stock may exceed this share of the portfolio, which "
                "also decides how many stocks each holds (5% ≈ 20 stocks). "
                "Everything else in each recipe stays untouched."
            ),
        )
        frequency = st.selectbox(
            "Rebalancing frequency (applies to all)",
            options=["daily", "weekly", "monthly"],
            index=2,
            format_func=str.capitalize,
            key="compare_frequency",
            help=(
                "Overrides every strategy's own rebalancing preset for this "
                "comparison only, so the whole race runs at the same rhythm. "
                "Monthly (the default) is the academic convention and keeps "
                "trading costs down; try Weekly to see what more frequent "
                "trading costs net of fees."
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

    # Remember these inputs in plain (non-widget) session state on every
    # visit, so they survive leaving the page even though Streamlit wipes
    # the widget keys themselves. The restore above then re-seeds the
    # widgets, and the cache key below matches the remembered results.
    st.session_state[_COMPARE_INPUTS_KEY] = {
        "compare_start": start.strip(),
        "compare_end": end.strip(),
        "compare_capital": float(capital),
        "compare_max_weight": float(max_weight),
        "compare_frequency": frequency,
    }

    # Remembered results: the cache key covers every input, so coming back
    # to this page with unchanged inputs restores the previous results
    # instead of forcing a re-run. Changed inputs -> different key ->
    # the old results simply don't show.
    strategy_names = [name for name, _ in H.strategy_options()]
    cache = st.session_state.setdefault(_COMPARE_CACHE_KEY, {})
    cache_key = H.compare_cache_key(
        strategy_names,
        tickers,
        start.strip() or None,
        end.strip() or None,
        capital,
        max_weight,
        10.0,  # the shared trading-cost setting used by this race
        "open",  # the shared execution timing used by this race
        cost_model="spread",
        frequency=frequency,
        db_path=DEFAULT_DB,
    )
    cached = H.get_cached_compare(cache, cache_key)

    with st.container(border=True):
        st.subheader("Strategy defaults")
        st.dataframe(
            pd.DataFrame(H.strategy_preset_rows()),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"This comparison applies **{frequency.capitalize()}** rebalancing "
            f"and **{max_weight:.0%}** max weight to all strategies "
            "(overrides shown) — everything else is each strategy's own "
            "registry preset from the table above."
        )

    button_label = "Recompute" if cached else "Compare strategies"
    if st.button(button_label, type="primary", disabled=not confirmed):
        if years <= 0:
            st.error("The end date must be after the start date.")
            return
        with st.spinner("Racing every strategy over the same history..."):
            try:
                rows = H.compare_strategies_for_ui(
                    tickers, start.strip() or None, end.strip() or None,
                    capital, max_weight, db_path=DEFAULT_DB,
                    cost_model="spread", frequency=frequency,
                )
                equities = H.compare_equities_for_ui(
                    tickers, start.strip() or None, end.strip() or None,
                    capital, max_weight, db_path=DEFAULT_DB,
                    cost_model="spread", frequency=frequency,
                )
                bench_row, spy_row = H.compare_benchmark_rows(
                    tickers, start.strip() or None, end.strip() or None,
                    capital, db_path=DEFAULT_DB,
                )
                try:
                    attributions = H.compare_attribution_for_ui(
                        equities, db_path=DEFAULT_DB
                    )
                except ValueError:
                    # No factor data downloaded yet — the table still
                    # shows; the attribution section explains what's missing.
                    attributions = None
            except ValueError as exc:
                st.error(str(exc))
                return
        H.store_cached_compare(
            cache,
            cache_key,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "rows": rows,
                "bench_row": bench_row,
                "spy_row": spy_row,
                "attributions": attributions,
            },
        )
        _render_compare_results(rows, bench_row, spy_row, attributions)
    elif cached is not None:
        _render_compare_results(
            cached["rows"],
            cached["bench_row"],
            cached["spy_row"],
            cached.get("attributions"),
            caption=(
                f"Results from {cached['time']} — these match your current "
                "inputs, so nothing was re-run. Press **Recompute** above to "
                "run the race again."
            ),
        )

def _render_compare_results(
    rows: list[dict],
    bench_row: dict | None,
    spy_row: dict | None,
    attributions: dict[str, dict | None] | None = None,
    caption: str | None = None,
) -> None:
    """Render a finished head-to-head comparison (fresh or restored).

    Shared by the live "Compare strategies" run and the session-state
    restore, so the two always look identical. `attributions` maps each
    strategy's display name to its factor-attribution result (or None when
    that strategy had too little overlapping history); None overall means
    no factor data was downloaded, and the section says so plainly.
    """
    if caption:
        st.caption(caption)
    with st.container(border=True):
        st.subheader("Head-to-head results")
        all_rows = rows + [r for r in (bench_row, spy_row) if r]
        table = pd.DataFrame(all_rows).drop(columns=["total_return", "sharpe"])
        st.dataframe(table, use_container_width=True)
        st.caption(
            "Every row raced the same stocks, dates and starting cash — "
            "only each strategy's own recipe differs. "
            "**Market benchmark (buy-and-hold)** buys every one of *your* "
            "stocks in equal amounts on day one and never trades — "
            "dividends included, like the strategies. "
            "**S&P 500 (SPY)** puts everything into the S&P 500 index "
            "fund on day one and never trades — the 'just own the whole "
            "market' alternative."
        )
        if spy_row is None:
            st.info(
                "The **S&P 500 (SPY)** row is missing because no SPY prices "
                "are stored yet. On the Market data page, tick 'Also "
                "download SPY (S&P 500 benchmark)' and download prices — "
                "then re-run the comparison."
            )

    chart_rows = [r for r in rows if r["total_return"] == r["total_return"]]
    if chart_rows:
        with st.container(border=True):
            st.subheader("Total return, side by side")
            bars = {
                r["Strategy"]: r["total_return"] for r in chart_rows
            }
            if bench_row and bench_row["total_return"] == bench_row["total_return"]:
                bars["Market benchmark"] = bench_row["total_return"]
            if spy_row and spy_row["total_return"] == spy_row["total_return"]:
                bars["S&P 500 (SPY)"] = spy_row["total_return"]
            chart = pd.DataFrame(bars, index=["Total return"]).T
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

    with st.container(border=True):
        st.subheader("What explains each strategy's returns?")
        st.write(
            "Each strategy's daily returns are compared against six "
            "well-known market patterns — the market itself, company size, "
            "value vs growth, profitability, investment style, and momentum "
            "(the Fama-French factors, free from Dartmouth). This shows "
            "whether a strategy was doing something of its own, or mostly "
            "riding familiar tilts."
        )
        if attributions is None:
            st.info(
                "Factor data isn't downloaded yet, so this section can't "
                "run. On the **Market data** page, press *Download factor "
                "data (Ken French)* — it's free — then re-run the comparison."
            )
        else:
            for name, res in attributions.items():
                if res is None:
                    st.write(
                        f"**{name}**: not enough overlapping history with "
                        "the factor data for an attribution."
                    )
                    continue
                with st.expander(f"{name}: {res['short_line']}"):
                    st.write(res["summary"])
                    st.caption(
                        f"R² = {res['r_squared']:.2f} over {res['n_obs']} "
                        f"trading days ({res['start']} to {res['end']}). "
                        "A tilt counts as convincing when its t-stat is "
                        "beyond ±2."
                    )

    with st.container(border=True):
        st.subheader("How realistic are these numbers?")
        st.write(H.realism_footnote(cost_model_name="spread"))


# Session-state key holding remembered comparison results.
_COMPARE_CACHE_KEY = "compare_results_cache"


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

    tickers = H.require_data("prices")["tickers"]

    with st.container(border=True):
        signal_name, _ = _signal_picker(key="signals_page_signal")
        use_all = st.checkbox(
            f"Compute for all {len(tickers)} stocks in the database",
            value=True,
            key="signals_all_stocks",
            help=(
                "Score every stock with downloaded prices at once. "
                "Uncheck to pick specific stocks instead."
            ),
        )
        if use_all:
            chosen = tickers
        else:
            chosen = st.multiselect(
                "Stocks", tickers, default=tickers, key="signals_tickers"
            )

        if st.button("Compute signals", type="primary", key="signals_compute"):
            if not chosen:
                st.error("Pick at least one stock.")
            else:
                with st.spinner("Scoring stocks..."):
                    try:
                        rows = H.compute_signal_rows(signal_name, chosen, DEFAULT_DB)
                    except Exception:
                        st.error(
                            "Something went wrong while scoring the stocks. "
                            "Your saved data is unchanged. Please try again — "
                            "or run the same step from a terminal with:\n\n"
                            f"`python -m qrp.signals.cli --signals {signal_name} "
                            "--tickers all`"
                        )
                    else:
                        note = H.signal_data_note(signal_name, chosen, DEFAULT_DB)
                        if note:
                            st.info(note)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.container(border=True):
        st.subheader("Compute all signals at once")
        st.write(
            "Scores every chosen stock on every signal in one go — the "
            "same math as computing them one by one above, just without the "
            "repeated clicking. This can take a few minutes on a large "
            "universe; the summary below shows what each signal stored."
        )
        if st.button("Compute all signals", type="primary",
                     key="signals_compute_all"):
            if not chosen:
                st.error("Pick at least one stock.")
            else:
                bar = st.progress(0, text="Starting…")

                def _tick(done: int, total: int, label: str) -> None:
                    if done < total:
                        bar.progress(
                            done / total,
                            text=f"Computing {label} "
                                 f"({done + 1} of {total})…",
                        )
                    else:
                        bar.progress(1.0, text="All signals computed.")

                summary = H.compute_all_signals(
                    chosen, DEFAULT_DB, progress=_tick
                )
                st.dataframe(pd.DataFrame(summary), use_container_width=True)
                st.caption(
                    "Rows stored is how many per-stock daily scores each "
                    "signal wrote to the database — the Backtest, Compare "
                    "and Portfolio pages read from these."
                )

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
        pretty = H.signal_display_name(signal_name)
        if series.empty:
            st.info(f"No {pretty} values stored for {one} yet.")
        else:
            st.line_chart(series)
            st.caption(
                f"How {one}'s {pretty} score moved over time."
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

    tickers = H.require_data("prices", "signals")["tickers"]

    with st.container(border=True):
        strategy_name = _strategy_picker("backtest")
        if strategy_name is None:
            signal_name, _ = _signal_picker(label="Rank stocks by", key="backtest_signal")
            c1, c2, c3 = st.columns(3)
            weighting_key = c1.selectbox(
                "Position sizing",
                ["equal", "signal", "inverse_vol", "optimized"],
                format_func=lambda k: H.WEIGHTING_LABELS[k],
            )
            rebalance = c2.radio("Rebalance", ["daily", "weekly", "monthly"],
                                 horizontal=True,
                                 help=(
                                     "How often the portfolio is re-evaluated. "
                                     "'Daily' (the default) checks every trading day, "
                                     "but the drift buffer below still decides whether "
                                     "anything actually trades — small day-to-day "
                                     "changes don't cost you anything."
                                 ))
            optimizer_settings = None
            if weighting_key == "optimized":
                st.caption(H.weighting_description(weighting_key))
                with st.expander("Optimizer settings", expanded=False):
                    st.caption(
                        "How the optimizer balances return, risk, trading, and "
                        "closeness to the market. Defaults are sensible — change "
                        "them only if you know what each knob does."
                    )
                    risk_aversion = st.slider(
                        "Risk aversion",
                        0.5, 10.0, float(H.OPTIMIZER_DEFAULTS["risk_aversion"]), 0.5,
                        key="backtest_opt_risk",
                        help="How much the optimizer dislikes ups and downs. Higher "
                             "means a calmer portfolio, usually at the cost of some "
                             "expected return.",
                    )
                    turnover_bps = st.slider(
                        "Turnover penalty (basis points)",
                        0.0, 200.0,
                        float(H.OPTIMIZER_DEFAULTS["turnover_penalty_bps"]), 5.0,
                        key="backtest_opt_turnover",
                        help="The 'only trade when it's worth it' knob. Trading a "
                             "dollar of the portfolio is penalized as if it cost "
                             "this much — higher values make the optimizer sit "
                             "tighter and trade less.",
                    )
                    tracking_error = st.slider(
                        "Tracking-error budget vs S&P 500 (%)",
                        1.0, 15.0,
                        float(H.OPTIMIZER_DEFAULTS["tracking_error"]) * 100.0, 1.0,
                        key="backtest_opt_te",
                        help="How far the portfolio's returns may stray from the "
                             "S&P 500's, per year. Tighter keeps you closer to the "
                             "market; looser lets the optimizer chase the signal "
                             "harder.",
                    )
                    opt_max_weight = st.slider(
                        "Max weight per stock (%)",
                        5.0, 50.0,
                        float(H.OPTIMIZER_DEFAULTS["max_weight"]) * 100.0, 5.0,
                        key="backtest_opt_maxw",
                        help="No single stock may exceed this share of the "
                             "portfolio — the same safety cap as everywhere else.",
                    )
                    cov_choice = st.selectbox(
                        "Risk model",
                        ["shrinkage", "factor"],
                        key="backtest_opt_cov",
                        format_func=lambda k: H.COVARIANCE_LABELS[k],
                        help="How the optimizer learns which stocks move together. "
                             "'Shrinkage sample' is the simple, sturdy default. "
                             "'Statistical factor model' learns the handful of "
                             "shared market patterns behind past movement — usually "
                             "a sharper read of the real risk. The Risk model page "
                             "explains it in plain English.",
                    )
                    st.caption(H.COVARIANCE_DESCRIPTION)
                    optimizer_settings = {
                        "risk_aversion": risk_aversion,
                        "turnover_penalty_bps": turnover_bps,
                        "tracking_error": tracking_error / 100.0,
                        "max_weight": opt_max_weight / 100.0,
                        "covariance": cov_choice,
                    }
                max_weight = optimizer_settings["max_weight"]
                st.caption(
                    "How optimized backtests work: at every rebalance, the "
                    "optimizer re-solves from scratch using only data that "
                    "existed on that date — scores, prices, and company data "
                    "are all dated on or before it, so nothing peeks at the "
                    "future. Its turnover penalty compares against the "
                    "previous rebalance's target weights, standing in for "
                    "the holdings the backtest would actually hold. And "
                    "expect these to run slower than the rule-based "
                    "backtests: every rebalance is a fresh optimization."
                )
            else:
                max_weight = c3.slider(
                    "Max weight per stock",
                    0.01, 0.50, 0.05, 0.01,
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
        data_start, data_end = H.data_date_range(DEFAULT_DB)
        st.write(
            "**Quick stress windows** — jump to a famous rough patch, or "
            "back to your full history:"
        )
        s1, s2, s3 = st.columns(3)
        if s1.button("COVID crash (2020)", key="preset_covid"):
            st.session_state["backtest_start"] = "2020-01-01"
            st.session_state["backtest_end"] = "2020-12-31"
            st.rerun()
        if s2.button("2022 bear market", key="preset_2022"):
            st.session_state["backtest_start"] = "2022-01-01"
            st.session_state["backtest_end"] = "2022-12-31"
            st.rerun()
        if s3.button("Full history", key="preset_full"):
            st.session_state["backtest_start"] = data_start or ""
            st.session_state["backtest_end"] = data_end or ""
            st.rerun()
        start = c7.text_input("From (YYYY-MM-DD)", value=data_start or "",
                              key="backtest_start",
                              help="Defaults to your earliest downloaded data.")
        end = c8.text_input("To (YYYY-MM-DD)", value=data_end or "",
                            key="backtest_end",
                            help="Defaults to your latest downloaded data.")
        drift_pct = st.slider(
            "Drift buffer (skip tiny trades)",
            0, 10, 2, 1,
            format="%d%%",
            help=(
                "Don't pay trading costs to fix wobbles nobody would notice: "
                "a position is only traded when bringing it back to target "
                "would move it by more than this share of the portfolio. "
                "2% matches the ready-made strategies, so this page agrees "
                "with the Compare page."
            ),
        )
        drift_tolerance = drift_pct / 100.0

        cost_model = st.radio(
            "Trading cost model",
            options=["flat", "spread"],
            index=1,
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
                "enough price history to work. One caveat: the estimate is a "
                "daily-average spread, while trades filling right at the open "
                "tend to pay a bit more — so with next-day-open execution it "
                "may slightly understate costs."
            ),
        )

        execution = st.radio(
            "Trade execution",
            options=["open", "close"],
            format_func=lambda k: (
                "Next day's open" if k == "open"
                else "Next day's close"
            ),
            help=(
                "When a trade decided today actually fills. 'Next day's open' "
                "(the default) matches the real workflow: run the rebalance at "
                "night, stage the orders, and they fill at the next morning's "
                "open like market-on-open orders. 'Next day's close' is the "
                "older convention — results from before this setting existed "
                "used it, so they will differ slightly from new runs."
            ),
        )

    if strategy_name is None:
        def _run_current_config(start: str | None, end: str | None):
            """This page's backtest with the current settings, any window."""
            return H.run_backtest_for_ui(
                signal_name, tickers, weighting_key, max_weight,
                rebalance, cost_bps, capital,
                start, end, DEFAULT_DB,
                cost_model=cost_model,
                drift_tolerance=drift_tolerance,
                execution=execution,
                optimizer_settings=optimizer_settings,
            )
    else:
        def _run_current_config(start: str | None, end: str | None):
            """This page's strategy backtest with the current settings."""
            return H.run_strategy_backtest_for_ui(
                strategy_name, tickers, cost_bps, capital,
                start, end, DEFAULT_DB,
                cost_model=cost_model,
                execution=execution,
            )

    b1, b2 = st.columns(2)
    run_main = b1.button("Run backtest", type="primary", key="run_backtest")
    run_suite = b2.button(
        "Run all three stress windows",
        key="run_stress_suite",
        help=(
            "Runs this exact backtest — same strategy and settings — over "
            "the COVID crash (2020), the 2022 bear market, and your full "
            "history, then shows the three side by side."
        ),
    )
    if run_suite:
        with st.spinner("Replaying the three stress windows..."):
            suite_rows = H.run_stress_suite_for_ui(_run_current_config)
        with st.container(border=True):
            st.subheader("Stress windows, side by side")
            st.write(
                "The same strategy and settings, replayed over three very "
                "different periods — the fastest way to see whether an idea "
                "only worked in good times."
            )
            st.dataframe(
                pd.DataFrame(suite_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Each row is a full backtest with identical settings; only "
                "the dates change. A window your downloaded data doesn't "
                "cover says so instead of failing."
            )

    if run_main:
        with st.spinner("Replaying history..."):
            try:
                result, metrics = _run_current_config(
                    start.strip() or None, end.strip() or None
                )
            except ValueError as exc:
                st.error(str(exc))
                return

        H.warn_if_no_trades(result.trades)

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
            st.subheader("Risk — how bumpy was the ride?")
            st.write(
                "Beyond the headline return: how much this strategy could "
                "lose on a bad day, and how wobbly it was day to day. "
                "Every term is explained in plain English below."
            )
            risk = H.risk_metrics_for_ui(result.equity, metrics)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Bad-day loss (95%)",
                      H.metric_display("var_95", risk["var_95"]["value"]))
            r2.metric("Average of the worst days",
                      H.metric_display("cvar_95", risk["cvar_95"]["value"]))
            r3.metric("Worst fall",
                      H.metric_display("max_drawdown",
                                       risk["max_drawdown"]["value"]))
            r4.metric("Yearly bumpiness",
                      H.metric_display("volatility",
                                       risk["volatility"]["value"]))
            for key in ("var_95", "cvar_95", "var_99", "cvar_99",
                        "daily_volatility", "max_drawdown", "volatility"):
                m = risk[key]
                with st.expander(
                    f"{m['label']}: {H.metric_display(key, m['value'])}"
                ):
                    st.write(f"**What it means:** {m['what']}")
                    st.write(f"**Why it matters:** {m['why']}")
                    st.write(f"**What good looks like:** {m['good']}")

        with st.container(border=True):
            st.subheader("Portfolio value over time")
            st.caption(
                "This is a **total return** curve: it includes both price moves "
                "and any dividend cash the portfolio received along the way. "
                "When a company you held paid a dividend, that cash was added "
                "to the account on the payout date — so the curve never shows "
                "a fake loss on dividend days."
            )
            equity = result.equity.copy()
            equity["date"] = pd.to_datetime(equity["date"])
            lines = {
                "Strategy": equity.set_index("date")["portfolio_value"],
            }
            try:
                bench_equity, bench_metrics = H.benchmark_for_ui(
                    tickers, start.strip() or None, end.strip() or None,
                    capital, db_path=DEFAULT_DB,
                )
                lines["Market benchmark (buy-and-hold)"] = (
                    bench_equity.set_index("date")["portfolio_value"]
                )
                bench_ret = H.metric_display(
                    "total_return", bench_metrics["total_return"]["value"]
                )
                bench_note = (
                    f" The **Market benchmark (buy-and-hold)** line buys every "
                    f"stock in equal amounts on day one and never trades "
                    f"(it returned {bench_ret})."
                )
            except ValueError:
                bench_note = ""
            try:
                spy_equity, spy_metrics, rel_metrics = H.spy_relative_for_ui(
                    result.equity, start.strip() or None, end.strip() or None,
                    capital, db_path=DEFAULT_DB,
                )
                lines["S&P 500 (SPY)"] = (
                    spy_equity.set_index("date")["portfolio_value"]
                )
                spy_ret = H.metric_display(
                    "total_return", spy_metrics["total_return"]["value"]
                )
                spy_note = (
                    f" The **S&P 500 (SPY)** line puts everything into the "
                    f"S&P 500 index fund on day one and never trades (it "
                    f"returned {spy_ret}) — the 'just own the whole market' "
                    f"alternative."
                )
            except ValueError:
                rel_metrics = None
                spy_note = (
                    " (Tip: tick 'Also download SPY (S&P 500 benchmark)' on "
                    "the Market data page and download prices to add the "
                    "S&P 500 line and the strategy-vs-market comparison below.)"
                )
            st.line_chart(pd.DataFrame(lines))
            st.caption(
                "The simulated account value on each trading day."
                + bench_note + spy_note
            )

        if rel_metrics is not None:
            with st.container(border=True):
                st.subheader("Strategy vs the S&P 500 (SPY)")
                st.write(
                    "How this backtest compares with simply owning the whole "
                    "US stock market through the S&P 500 index fund — the "
                    "simplest thing you could have done instead."
                )
                for key in (
                    "alpha", "beta", "information_ratio",
                    "tracking_error", "up_capture", "down_capture",
                ):
                    m = rel_metrics[key]
                    with st.expander(f"{m['label']}: {H.metric_display(key, m['value'])}"):
                        st.write(f"**What it means:** {m['what']}")
                        st.write(f"**Why it matters:** {m['why']}")
                        st.write(f"**What good looks like:** {m['good']}")

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

        with st.container(border=True):
            st.subheader("How realistic are these numbers?")
            st.write(H.realism_footnote(cost_model_name=cost_model))


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

    tickers = H.require_data("prices")["tickers"]

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
            data_start, data_end = H.data_date_range(DEFAULT_DB)
            st.session_state.setdefault("wizard_start", data_start or three_years_ago)
            st.session_state.setdefault("wizard_end", data_end or date.today().isoformat())
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
            if not H.data_status()["has_signals"]:
                st.info(
                    "Signals haven't been computed yet — the Signals page "
                    "scores your stocks from their prices in one click. "
                    "Come back here afterwards to run."
                )
                if st.button("Go to Signals", type="primary",
                             key="wizard_goto_signals"):
                    H.go_to_page("Signals")
            elif st.button("Run backtest", type="primary", key="wizard_run"):
                with st.spinner("Replaying history..."):
                    try:
                        result, metrics = H.run_strategy_backtest_for_ui(
                            strategy_name, tickers, 10.0, 100_000.0,
                            start, end, DEFAULT_DB,
                            # Wizard backtests fill at the next day's open,
                            # matching the real nightly workflow.
                            execution="open",
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
                result = st.session_state.get("wizard_result")
                if result is not None:
                    H.warn_if_no_trades(result.trades)
                m1, m2 = st.columns(2)
                m1.metric(
                    "Final value",
                    H.metric_display("final_value", metrics["final_value"]["value"]),
                )
                m2.metric(
                    "Total return",
                    H.metric_display("total_return", metrics["total_return"]["value"]),
                )
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
        else:
            st.session_state["wizard_step"] = step + 1
            st.rerun()
    if b3.button("Start over", key="wizard_reset"):
        _wizard_reset()
        st.rerun()


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def _portfolio_controls(
    prefix: str, asof_help: str | None = None, show_asof: bool = True
) -> dict:
    """Shared signal/method/date controls; returns the chosen settings.

    When "optimized" is chosen, four extra sliders appear for the
    optimizer's settings; they travel in settings["optimizer"] (None for
    the rule-based methods).

    show_asof=False hides the "As of date" input (for views like the
    Alpaca paper view, where a replay date is meaningless) — the returned
    settings still carry "asof": "" so callers need no special-casing.
    """
    signal_name, _ = _signal_picker(label="Rank stocks by", key=f"{prefix}_signal")
    method_key = st.selectbox(
        "Position sizing",
        ["equal", "signal", "inverse_vol", "optimized"],
        key=f"{prefix}_method",
        format_func=lambda k: H.WEIGHTING_LABELS[k],
    )
    st.caption(H.weighting_description(method_key))
    optimizer_settings = None
    if method_key == "optimized":
        with st.expander("Optimizer settings", expanded=False):
            st.caption(
                "How the optimizer balances return, risk, trading, and "
                "closeness to the market. Defaults are sensible — change "
                "them only if you know what each knob does."
            )
            risk_aversion = st.slider(
                "Risk aversion",
                0.5, 10.0, float(H.OPTIMIZER_DEFAULTS["risk_aversion"]), 0.5,
                key=f"{prefix}_opt_risk",
                help="How much the optimizer dislikes ups and downs. Higher "
                     "means a calmer portfolio, usually at the cost of some "
                     "expected return.",
            )
            turnover_bps = st.slider(
                "Turnover penalty (basis points)",
                0.0, 200.0,
                float(H.OPTIMIZER_DEFAULTS["turnover_penalty_bps"]), 5.0,
                key=f"{prefix}_opt_turnover",
                help="The 'only trade when it's worth it' knob. Trading a "
                     "dollar of the portfolio is penalized as if it cost "
                     "this much — higher values make the optimizer sit "
                     "tighter and trade less.",
            )
            tracking_error = st.slider(
                "Tracking-error budget vs S&P 500 (%)",
                1.0, 15.0,
                float(H.OPTIMIZER_DEFAULTS["tracking_error"]) * 100.0, 1.0,
                key=f"{prefix}_opt_te",
                help="How far the portfolio's returns may stray from the "
                     "S&P 500's, per year. Tighter keeps you closer to the "
                     "market; looser lets the optimizer chase the signal "
                     "harder.",
            )
            max_weight = st.slider(
                "Max weight per stock (%)",
                5.0, 50.0,
                float(H.OPTIMIZER_DEFAULTS["max_weight"]) * 100.0, 5.0,
                key=f"{prefix}_opt_maxw",
                help="No single stock may exceed this share of the "
                     "portfolio — the same safety cap as everywhere else.",
            )
            cov_choice = st.selectbox(
                "Risk model",
                ["shrinkage", "factor"],
                key=f"{prefix}_opt_cov",
                format_func=lambda k: H.COVARIANCE_LABELS[k],
                help="How the optimizer learns which stocks move together. "
                     "'Shrinkage sample' is the simple, sturdy default. "
                     "'Statistical factor model' learns the handful of "
                     "shared market patterns behind past movement — usually "
                     "a sharper read of the real risk. The Risk model page "
                     "explains it in plain English.",
            )
            st.caption(H.COVARIANCE_DESCRIPTION)
            optimizer_settings = {
                "risk_aversion": risk_aversion,
                "turnover_penalty_bps": turnover_bps,
                "tracking_error": tracking_error / 100.0,
                "max_weight": max_weight / 100.0,
                "covariance": cov_choice,
            }
    asof = ""
    if show_asof:
        asof = st.text_input(
            "As of date (YYYY-MM-DD)",
            value="",
            key=f"{prefix}_date",
            help=asof_help or "Leave empty to use the latest date with signal data.",
        )
    return {
        "signal_name": signal_name,
        "method_key": method_key,
        "asof": asof,
        "optimizer": optimizer_settings,
    }


def _render_optimizer_results(portfolio, scores, day, stats,
                               covariance_choice: str = "shrinkage") -> None:
    """Show what the mean-variance optimizer decided, in plain English.

    One paragraph explaining the optimizer, four headline numbers
    (expected return, predicted volatility, tracking error vs the S&P 500,
    estimated turnover), and a before/after per-stock weight table. On
    this page the "before" portfolio is cash (w_prev is zeros), so the
    table shows what gets bought from scratch.
    """
    st.write(
        "**What the optimizer did:** it picked weights to get the most "
        "expected return for the risk taken, while limiting trading and "
        "staying close to the market (the S&P 500). The expected returns "
        "are a rough mapping from the signal scores — a guess about the "
        "signal's edge, not a measured forecast."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected annual return", f"{stats.expected_return_ann:.1%}")
    c2.metric("Predicted annual volatility", f"{stats.predicted_vol_ann:.1%}")
    c3.metric("Tracking error vs S&P 500", f"{stats.tracking_error_ann:.1%}")
    c4.metric("Est. turnover to build", f"{stats.turnover_one_way:.1%}")
    st.caption(
        "Expected return: the signal-implied gain per year (rough guess). "
        "Predicted volatility: how much the portfolio value is expected to "
        "swing per year. Tracking error: how far its returns may stray "
        "from the S&P 500's per year. Turnover: what share of the money "
        "gets traded to build this from cash."
    )
    st.caption(stats.benchmark_note)
    st.caption(
        "Risk model used: "
        f"{H.COVARIANCE_LABELS.get(covariance_choice, covariance_choice)}."
    )
    rows = [
        {
            "Ticker": t,
            "Signal score": f"{scores[t]:+.4f}",
            "Before": "0.0%",
            "After": f"{w:.1%}",
        }
        for t, w in sorted(portfolio.weights.items(), key=lambda kv: -kv[1])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    chart_df = pd.DataFrame(
        {"weight": portfolio.weights}
    ).sort_values("weight", ascending=False)
    st.bar_chart(chart_df)
    st.info(
        f"What this means: if you invested $10,000 today, you would put "
        f"${10_000:,.0f} into {len(portfolio.weights)} stock(s) as shown "
        f"above — the optimizer stays fully invested, so nothing is kept "
        f"in cash."
    )
    for note in portfolio.notes:
        st.caption(f"Note: {note}")


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

    tickers = H.require_data("prices", "signals")["tickers"]

    with st.container(border=True):
        strategy_name = _strategy_picker("portfolio")
        if strategy_name is None:
            settings = _portfolio_controls("portfolio")
            if settings["method_key"] == "optimized":
                # The optimizer carries its own max-weight setting (in the
                # "Optimizer settings" expander above); the rule-based
                # slider below does not apply to it.
                st.caption(
                    "The optimizer sets its own max weight per stock "
                    "(see Optimizer settings above)."
                )
                max_weight = None
            else:
                max_weight = st.slider(
                    "Max weight per stock",
                    0.05, 0.50, 0.10, 0.05,
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
            stats = None
            if strategy_name is None:
                asof = settings["asof"].strip() or None
                if asof is None:
                    matrix = H.cached_load_signal_matrix(
                        settings["signal_name"], tickers, db_path=DEFAULT_DB
                    )
                    if matrix.empty:
                        raise ValueError("No signal data — compute signals first.")
                    asof = matrix.index.max().strftime("%Y-%m-%d")
                if settings["method_key"] == "optimized":
                    # Fresh portfolio here: w_prev is zeros, so the turnover
                    # penalty has nothing to anchor to and the reported
                    # turnover is the cost of building from cash.
                    portfolio, scores, day, stats = H.build_optimized_target_for_ui(
                        settings["signal_name"], asof, tickers, DEFAULT_DB,
                        optimizer_settings=settings["optimizer"],
                    )
                else:
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
                    f"{H.signal_display_name(n)}: {d}" for n, d in leg_dates.items()
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
            if stats is not None:
                # The optimizer renders its own results block (metrics,
                # before/after table, and notes).
                cov_choice = (
                    (settings.get("optimizer") or {}).get(
                        "covariance", H.COVARIANCE_DEFAULT
                    )
                )
                _render_optimizer_results(
                    portfolio, scores, day, stats,
                    covariance_choice=cov_choice,
                )
            else:
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
            "Cost": H.fmt_money(t["cost"]),
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


def _alpaca_status_cached() -> dict:
    """Alpaca paper connection status, checked once per session.

    The live check is a network call, so it runs at most once per browser
    session (saving keys on the Brokers page, or the "Reconnect" button,
    clears the cache). When no keys are saved this is a cheap local file
    check and never touches the network.
    """
    if not H.alpaca_paper_keys_saved(DEFAULT_DB):
        return {
            "configured": False, "ok": False, "broker": None,
            "account": None, "error": "",
        }
    if "alpaca_conn" not in st.session_state:
        with st.spinner("Checking the Alpaca paper connection..."):
            st.session_state["alpaca_conn"] = H.alpaca_connection_for_ui(
                DEFAULT_DB
            )
    return st.session_state["alpaca_conn"]


def page_brokers() -> None:
    """Broker connections: Alpaca paper keys + connection test."""
    st.title("Brokers")
    st.write(
        "Connect a broker to paper-trade through it instead of only the "
        "built-in simulator."
    )
    st.divider()

    st.subheader("Alpaca paper trading")
    st.write(
        "Alpaca offers free paper-trading accounts — real market mechanics "
        "with fake money. Connect yours here and the Paper trading page "
        "will drive Alpaca instead of the built-in simulator."
    )
    st.caption(
        "Get free paper keys: sign up at alpaca.markets, open the paper "
        "trading dashboard, and generate a paper API key pair. Your keys "
        "stay on this computer in a private file — never in the code, "
        "never sent anywhere except Alpaca."
    )
    keys_saved = H.alpaca_paper_keys_saved(DEFAULT_DB)
    if keys_saved:
        st.success("Paper API keys are saved on this computer.")
    key_id = st.text_input(
        "Paper key ID",
        type="password",
        key="brokers_key_id",
        help="From your Alpaca paper dashboard — paste it here; it stays on this computer.",
    )
    secret = st.text_input(
        "Paper secret key",
        type="password",
        key="brokers_secret",
        help="Pairs with the key ID above. Never shared, never logged.",
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Save keys", type="primary", key="brokers_save"):
        if not key_id.strip() or not secret.strip():
            st.error(
                "Enter both the key ID and the secret key before saving."
            )
        else:
            try:
                keys_path = H.save_alpaca_paper_keys_for_ui(
                    key_id.strip(), secret.strip(), DEFAULT_DB
                )
            except (AlpacaError, OSError, ValueError) as exc:
                # save_paper_keys raises AlpacaError (not OSError) on write
                # failure — catch it so a disk problem shows plain English,
                # never a raw traceback.
                st.error(str(exc))
            else:
                # P6: record that keys were saved — never the keys themselves.
                OPS_AUDIT.log_event(
                    "keys_saved",
                    {"location": str(keys_path)},
                    actor="dashboard",
                    db_path=DEFAULT_DB,
                )
                st.session_state.pop("alpaca_conn", None)
                st.session_state["brokers_test_result"] = None
                # Don't keep the typed secrets in session state now that
                # they're saved — the fields come back empty.
                st.session_state.pop("brokers_key_id", None)
                st.session_state.pop("brokers_secret", None)
                st.success(
                    "Keys saved — stored in a private file on this "
                    "computer only, readable just by you."
                )
                st.rerun()
    if c2.button("Test connection", key="brokers_test"):
        kid = key_id.strip()
        sec = secret.strip()
        with st.spinner("Talking to Alpaca's paper endpoint…"):
            if kid and sec:
                # Test what was typed, without saving it.
                ok, payload = H.test_alpaca_key_pair(kid, sec)
            elif keys_saved:
                status = H.alpaca_connection_for_ui(DEFAULT_DB)
                ok = status["ok"]
                payload = (
                    status["account"]
                    if ok
                    else status["error"] or "The saved keys didn't work."
                )
            else:
                ok, payload = (
                    False,
                    "No keys to test — paste your paper key ID and secret "
                    "above first.",
                )
        st.session_state["brokers_test_result"] = (ok, payload)
        st.rerun()
    if keys_saved and c3.button("Remove keys", key="brokers_remove"):
        removed = H.delete_alpaca_paper_keys_for_ui(DEFAULT_DB)
        # P6: record the removal — never the keys themselves.
        OPS_AUDIT.log_event(
            "keys_removed",
            {"keys_file_removed": bool(removed)},
            actor="dashboard",
            db_path=DEFAULT_DB,
        )
        st.session_state.pop("alpaca_conn", None)
        st.session_state["brokers_test_result"] = None
        st.success("Keys removed from this computer.")
        st.rerun()

    result = st.session_state.get("brokers_test_result")
    if result is not None:
        ok, payload = result
        if ok:
            st.success(
                "Connected — your Alpaca PAPER account shows "
                f"{H.fmt_money(payload.equity)} equity, "
                f"{H.fmt_money(payload.cash)} cash and "
                f"{H.fmt_money(payload.buying_power)} buying power. "
                "Fake money, real mechanics."
            )
        else:
            st.error(f"Couldn't connect: {payload}")

    st.divider()
    st.info(
        "Live (real-money) trading is intentionally not available here — "
        "paper trading only, until the safety checks for real money are "
        "built."
    )


def page_paper() -> None:
    st.title("Paper trading")
    status = _alpaca_status_cached()
    if status["ok"]:
        st.write(
            "**Paper trading through Alpaca** — your staged orders go to "
            "Alpaca's paper-trading simulator: real market mechanics, fake "
            "money. Nothing here touches real dollars."
        )
        _paper_alpaca_view(status)
        st.divider()
        with st.expander(
            "Local simulated broker (offline testing)", expanded=False
        ):
            st.write(
                "The built-in simulator from before Alpaca — the same math, "
                "no internet needed. Handy for offline testing and for "
                "replaying past dates, which Alpaca can't do."
            )
            _paper_local_view()
    else:
        st.write(
            "**Paper trading** means practicing with fake money. The account, "
            "the orders, the fees and the risk checks are all real math — but "
            "no real dollars move and nothing touches a real brokerage. It is "
            "the flight simulator before the real plane."
        )
        if status["configured"]:
            st.warning(
                "Alpaca paper keys are saved, but the connection failed: "
                f"{status['error']} Showing the local simulated broker "
                "instead — check the Brokers page."
            )
            if st.button("Retry Alpaca connection", key="paper_retry_alpaca"):
                st.session_state.pop("alpaca_conn", None)
                st.rerun()
        else:
            st.info(
                "Tip: you can paper-trade through Alpaca's simulator instead "
                "of just this built-in one — connect it on the Brokers page."
            )
        st.divider()
        _paper_local_view()


def _paper_alpaca_view(status: dict) -> None:
    """Paper trading through the Alpaca paper account.

    The default view once Alpaca paper keys are saved and the connection
    tests OK. Staged orders are sent to Alpaca immediately; Alpaca queues
    them overnight and fills them at the next market open.
    """
    alpaca = status["broker"]
    account = status["account"]
    tickers = H.require_data("prices")["tickers"]

    if st.button("Reconnect to Alpaca", key="alpaca_reconnect"):
        st.session_state.pop("alpaca_conn", None)
        st.rerun()

    # Orders Alpaca hasn't accepted yet stay staged here (never silently
    # converted to an immediate fill) — retry sending them now for the
    # next valid market open.
    try:
        queued = H.alpaca_queued_staging_orders()
    except Exception:  # noqa: BLE001 - queue read is best-effort
        queued = []
    if queued:
        with st.container(border=True):
            st.subheader("Waiting to go to Alpaca")
            st.write(
                "These orders are staged but Alpaca hasn't accepted them "
                "yet. They'll be sent automatically for the next valid "
                "market open — nothing here will ever fill immediately."
            )
            for i, entry in enumerate(queued):
                cols = st.columns([1, 1, 2, 4, 1])
                cols[0].write(str(entry.get("side", "")))
                try:
                    shares_txt = f"{float(entry.get('shares', 0)):,.0f}"
                except (TypeError, ValueError):
                    shares_txt = str(entry.get("shares", ""))
                cols[1].write(f"{shares_txt} sh")
                cols[2].write(str(entry.get("ticker", "")))
                last_err = str(entry.get("last_error", "") or "")
                cols[3].caption(
                    f"Waiting since {str(entry.get('staged_at', ''))[:10]}"
                    + (f" — {last_err}" if last_err else "")
                )
                if cols[4].button("Discard", key=f"paper_alpaca_drop_{i}"):
                    H.drop_alpaca_queued_order(i)
                    st.rerun()
        with st.spinner("Sending waiting orders to Alpaca…"):
            # P6: the kill switch blocks the flush too — waiting orders are
            # still submissions.
            if _ops_trading_allowed():
                try:
                    flush = H.flush_alpaca_staging_queue(alpaca)
                except AlpacaError as exc:
                    st.error(str(exc))
                else:
                    if flush["sent"]:
                        st.success(
                            f"Sent {flush['sent']} waiting order(s) to Alpaca "
                            "for the next market open."
                        )
                        OPS_AUDIT.log_event(
                            "queue_event",
                            {
                                "action": "flushed",
                                "sent": flush["sent"],
                                "waiting": flush["waiting"],
                                "notes": flush["notes"],
                                "broker": "alpaca-paper",
                            },
                            actor="dashboard",
                            db_path=DEFAULT_DB,
                        )
                        st.rerun()
                    for note in flush["notes"]:
                        st.warning(note)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", H.fmt_money(account.equity))
        c2.metric("Cash", H.fmt_money(account.cash))
        c3.metric("Buying power", H.fmt_money(account.buying_power))
        st.caption(
            "Live snapshot of your Alpaca PAPER account — simulated money "
            "on Alpaca's servers, not real dollars."
        )

    with st.container(border=True):
        st.subheader("Positions at Alpaca")
        try:
            details = alpaca.get_position_details()
        except AlpacaError as exc:
            st.error(str(exc))
            details = None
        if details is None:
            pass
        elif not details:
            st.info("No positions — the Alpaca paper account is all cash.")
        else:
            rows = [
                {
                    "Ticker": d["symbol"],
                    "Shares": f"{d['qty']:,.2f}",
                    "Price": H.fmt_money(d["current_price"]),
                    "Value": H.fmt_money(d["market_value"]),
                    "Unrealized P&L": H.fmt_money(d["unrealized_pl"]),
                }
                for d in sorted(details, key=lambda r: r["symbol"])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.container(border=True):
        st.subheader("Open orders at Alpaca")
        st.write(
            "Orders you staged — Alpaca holds them overnight and fills them "
            "at the next market open. Cancel any of them below any time "
            "before they fill."
        )
        try:
            opens = alpaca.list_orders("open")
        except AlpacaError as exc:
            st.error(str(exc))
            opens = None
        if opens is None:
            pass
        elif not opens:
            st.info("No open orders at Alpaca.")
        else:
            for o in opens:
                oc1, oc2 = st.columns([5, 1])
                with oc1:
                    st.write(
                        f"**{o.side}** {o.shares:,.2f} shares of "
                        f"**{o.ticker}** — {o.status}"
                    )
                    if o.reason:
                        st.caption(o.reason)
                with oc2:
                    if st.button("Cancel", key=f"alpaca_cancel_{o.id}"):
                        try:
                            ok = alpaca.cancel_order(o.id)
                        except AlpacaError as exc:
                            st.error(str(exc))
                        else:
                            if ok:
                                st.success(
                                    f"Canceled the {o.side} order for "
                                    f"{o.ticker}."
                                )
                            else:
                                st.info("That order was already gone.")
                            st.rerun()

    with st.container(border=True):
        st.subheader("Stage orders at Alpaca for the next market open")
        stale_note = H.data_staleness_note()
        if stale_note:
            st.warning(stale_note)
        st.write(
            "Run the numbers now, trade later — the nightly routine, through "
            "Alpaca. Orders are sent the moment you stage them; Alpaca "
            "queues them overnight and fills them at the next market open."
        )
        st.caption(
            "How orders go out: every order is sent as a market-on-open "
            "order for the next trading day's open — staged at any time, "
            "it fills at the next open and never immediately. Share counts "
            "are rounded down to whole shares (Alpaca's market-on-open "
            "orders don't take fractions); the leftover stays as cash. If "
            "Alpaca can't take an order right now, it stays staged here "
            "and goes out automatically for the next valid open."
        )
        strategy_name = _strategy_picker("paper_alpaca")
        if strategy_name is None:
            # No as-of input here: the Alpaca view always stages for the
            # next market open (replay lives in the local simulator).
            settings = _portfolio_controls("paper_alpaca", show_asof=False)
        else:
            settings = {"asof": ""}
        if st.button("Preview rebalance", key="paper_alpaca_preview"):
            _alpaca_preview(alpaca, account, strategy_name, settings, tickers)
        confirm = st.checkbox(
            "I understand this sends orders to my Alpaca PAPER account "
            "(simulated money) — they will fill at the next market open.",
            key="paper_alpaca_confirm",
        )
        if st.button(
            "Stage orders at Alpaca",
            type="primary",
            disabled=not confirm,
            key="paper_alpaca_stage",
        ):
            _alpaca_stage(alpaca, account, strategy_name, settings, tickers)

    with st.container(border=True):
        st.subheader("Recent Alpaca orders")
        try:
            recent = alpaca.list_orders("all")[:20]
        except AlpacaError as exc:
            st.error(str(exc))
            recent = None
        if recent is None:
            pass
        elif not recent:
            st.info("No orders yet at Alpaca.")
        else:
            rows = [
                {
                    "Ticker": o.ticker,
                    "Side": o.side,
                    "Shares": f"{o.shares:,.2f}",
                    "Status": o.status,
                    "Avg fill": (
                        H.fmt_money(o.filled_avg_price)
                        if o.filled_avg_price
                        else "—"
                    ),
                    "Why": o.reason or "—",
                }
                for o in recent
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _alpaca_dust_note(dust: int) -> str:
    """Plain-English note for sub-1-share orders skipped in Alpaca staging."""
    return (
        f"{dust} tiny order(s) under one whole share were skipped — Alpaca's "
        "market-on-open orders need whole shares. If a leftover under one "
        "share is stuck in your Alpaca account (for example from trading "
        "outside this dashboard), it can't be sold from here — sell it in "
        "Alpaca's own paper dashboard."
    )


def _alpaca_sendable_orders(orders):
    """Whole-share orders for Alpaca's market-on-open staging.

    The staging flow is market-on-open only (time_in_force="opg"), and
    Alpaca only accepts fractions on plain day orders — so share counts
    are rounded down to whole shares, the safe direction (buys can't
    overshoot cash, sells can't overshoot what's held). Returns
    (sendable, dust): sendable is a list of dicts with ticker, side,
    shares (int), price and reason; dust counts orders too small for
    one whole share.
    """
    sendable = []
    dust = 0
    for order in orders:
        whole = int(order.shares)
        if whole <= 0:
            dust += 1
            continue
        sendable.append(
            {
                "ticker": order.ticker,
                "side": order.action,
                "shares": whole,
                "price": order.price,
                "reason": order.reason,
            }
        )
    return sendable, dust


def _alpaca_portfolio(strategy_name, settings, tickers, w_prev=None):
    """Target portfolio + timing for an Alpaca staging run.

    Returns (portfolio, decision_date, signal_day). Raises ValueError with
    a plain-English message when the target can't be built. w_prev (the
    Alpaca account's current holdings as weights) is only used by the
    "optimized" method; rule-based methods ignore it.
    """
    decision_date, signal_cutoff = H.resolve_staging_timing(DEFAULT_DB)
    if strategy_name is None:
        portfolio, _, signal_day = H.build_target_for_ui(
            settings["signal_name"], settings["method_key"],
            signal_cutoff, tickers, DEFAULT_DB,
            max_weight=0.05,
            w_prev=w_prev, optimizer_settings=settings.get("optimizer"),
        )
    else:
        (portfolio, _, signal_day, _) = H.build_strategy_target_for_ui(
            strategy_name, signal_cutoff, tickers, DEFAULT_DB,
        )
    return portfolio, decision_date, signal_day


def _warn_on_stale_signals(signal_day: str | None) -> None:
    """Warn (never block) when signals are older than the latest prices.

    The user's rule: any signal date that isn't the same day as the latest
    price data gets a warning — no grace threshold. Replay mode never
    calls this (old signals are the whole point of a replay).
    """
    note = H.signal_staleness_note(signal_day, DEFAULT_DB)
    if note:
        st.warning(note)


def _alpaca_preview(alpaca, account, strategy_name, settings, tickers) -> None:
    """Dry run: show what staging would send to Alpaca, without sending."""
    w_prev = (
        H.current_weights(alpaca, account.equity)
        if settings.get("method_key") == "optimized"
        else None
    )
    try:
        portfolio, decision_date, signal_day = _alpaca_portfolio(
            strategy_name, settings, tickers, w_prev=w_prev
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    _warn_on_stale_signals(signal_day)
    try:
        orders = H.compute_rebalance_orders(
            alpaca, portfolio.weights, decision_date, account.equity,
            cost_bps=0.0, price_source="close",
        )
    except (AlpacaError, ValueError) as exc:
        st.error(str(exc))
        return
    sendable, dust = _alpaca_sendable_orders(orders)
    st.caption(
        f"Decided {decision_date} on signals as of {signal_day} — this is "
        "exactly what staging would send to Alpaca (market-on-open orders, "
        "filled at the next market open)."
    )
    if not sendable:
        if dust:
            st.info(
                "Every order came out under one whole share — nothing to "
                "send. Alpaca's market-on-open orders need whole shares."
            )
        else:
            st.success("Already at target — no trades needed.")
        return
    # Safety-check verdicts from the standard preview machinery.
    verdict_by_key: dict = {}
    notes: list = []
    try:
        preview_rows, notes = H.preview_rebalance_orders(
            alpaca, portfolio.weights, decision_date, account.equity,
            cost_bps=0.0, price_source="close",
        )
    except (AlpacaError, ValueError):
        pass
    else:
        for row in preview_rows:
            verdict_by_key[(row["Ticker"], row["Action"])] = row.get(
                "Safety check", ""
            )
    table = [
        {
            "Action": s["side"],
            "Ticker": s["ticker"],
            "Shares": s["shares"],
            "Est. value": H.fmt_money(s["shares"] * s["price"]),
            "Why": s["reason"],
            "Safety check": verdict_by_key.get((s["ticker"], s["side"]), ""),
        }
        for s in sendable
    ]
    st.dataframe(pd.DataFrame(table), use_container_width=True)
    st.caption(
        "Dry run complete — nothing was sent to Alpaca. Share counts are "
        "rounded down to whole shares (Alpaca's market-on-open orders "
        "don't take fractions); the leftover stays as cash. Blocked "
        "orders explain why in plain English."
    )
    if dust:
        st.info(_alpaca_dust_note(dust))
    for note in notes:
        st.info(f"Note: {note}")
    # P6: the staging preview shows the pre-trade checklist inline —
    # per-check green/red with plain-English reasons. Blocked orders never
    # reach the broker at staging time.
    with st.expander("Pre-trade checklist (what staging will verify)",
                     expanded=False):
        _show_pretrade_report(
            OPS_PRE.run_pretrade_checks(
                sendable, alpaca, db_path=DEFAULT_DB,
                decision_date=decision_date, actor="dashboard (preview)",
            )
        )


def _ops_trading_allowed() -> bool:
    """Kill-switch-first gate for every staging/submission path.

    Returns True when trading may proceed; shows the plain-English reason
    and returns False when the kill switch is engaged (or the flag file
    is unreadable — fail-closed). The order pipeline checks this FIRST.
    """
    try:
        OPS_PIPE.check_trading_allowed(DEFAULT_DB)
    except OPS_KS.KillSwitchEngaged as exc:
        st.error(str(exc))
        return False
    return True


def _show_pretrade_report(report) -> None:
    """Per-check green/red table with plain-English reasons."""
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    rows = [
        {
            "Check": c.name,
            "Result": f"{icon.get(c.status, '')} "
                      f"{c.status.capitalize()}",
            "Why": c.detail,
        }
        for c in report.checks
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)
    if not report.ok:
        st.error(report.plain_english_summary())


def _alpaca_stage(alpaca, account, strategy_name, settings, tickers) -> None:
    """Send staged orders to Alpaca as market-on-open.

    Next-open only: every order goes out as time_in_force="opg" and is
    never converted into an immediate fill. P6 guards: the kill switch is
    checked first, the pre-trade checklist must pass (any failure blocks
    the whole batch with a plain-English reason), every order carries its
    deterministic client_order_id, and the submit path auto-engages the
    kill switch on mass rejections / auth failure / unexpected errors.
    If Alpaca rejects an order, it is kept staged in the local queue for
    the next valid open instead.
    """
    # P6: kill-switch-first — refuse before anything is computed.
    if not _ops_trading_allowed():
        return
    w_prev = (
        H.current_weights(alpaca, account.equity)
        if settings.get("method_key") == "optimized"
        else None
    )
    try:
        portfolio, decision_date, signal_day = _alpaca_portfolio(
            strategy_name, settings, tickers, w_prev=w_prev
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    _warn_on_stale_signals(signal_day)
    try:
        orders = H.compute_rebalance_orders(
            alpaca, portfolio.weights, decision_date, account.equity,
            cost_bps=0.0, price_source="close",
        )
    except (AlpacaError, ValueError) as exc:
        st.error(str(exc))
        return
    sendable, dust = _alpaca_sendable_orders(orders)
    if not sendable:
        if dust:
            st.info(
                "Every order came out under one whole share — nothing to "
                "send. Alpaca's market-on-open orders need whole shares."
            )
        else:
            st.success("Already at target — no trades needed.")
        return
    try:
        queued_keys = {
            (str(e.get("ticker")), str(e.get("side")))
            for e in H.alpaca_queued_staging_orders()
        }
    except Exception:  # noqa: BLE001 - queue read is best-effort
        queued_keys = set()
    already_waiting = 0
    to_submit = []
    for s in sendable:
        if (s["ticker"], s["side"]) in queued_keys:
            # Same order is already staged and waiting for the next valid
            # open — don't create a confusing duplicate queue entry.
            already_waiting += 1
            continue
        to_submit.append(s)
    if not to_submit:
        # Nothing new to send — say so and stop before the checklist and
        # the audit noise of a zero-order batch.
        if already_waiting:
            st.info(
                f"{already_waiting} order(s) already staged and waiting for "
                "Alpaca — skipped, so nothing is double-staged."
            )
        if dust:
            st.info(_alpaca_dust_note(dust))
        return
    # P6: the pre-trade checklist gates the batch — any failure blocks it
    # with a plain-English reason (blocked orders never reach the broker).
    # This also covers the old "already open at Alpaca" skip: a duplicate
    # now blocks with its reason instead of being silently skipped.
    report = OPS_PRE.run_pretrade_checks(
        to_submit, alpaca, db_path=DEFAULT_DB,
        decision_date=decision_date, actor="dashboard",
    )
    _show_pretrade_report(report)
    if not report.ok:
        return
    OPS_AUDIT.log_event(
        "signals_used",
        {
            "decision_date": decision_date,
            "signal_day": signal_day,
            "strategy": strategy_name,
            "settings": {
                k: v for k, v in (settings or {}).items()
                if k in ("signal_name", "method_key", "asof")
            },
            "n_orders": len(to_submit),
            "broker": "alpaca-paper",
        },
        actor="dashboard",
        db_path=DEFAULT_DB,
    )

    def _send(s: dict):
        # P6: every order carries its deterministic client_order_id, so a
        # retried staging run cannot double-trade (Alpaca dedupes on it).
        return alpaca.place_order(
            s["ticker"], s["side"], float(s["shares"]),
            reason=s["reason"], time_in_force="opg",
            client_order_id=OPS_PRE.client_order_id(
                s["ticker"], s["side"], decision_date
            ),
        )

    def _on_rejected(s: dict, error: str) -> None:
        # Never silently convert to an immediate fill — keep the order
        # staged for the next valid open.
        H.queue_alpaca_staged_order(
            s["ticker"], s["side"], float(s["shares"]),
            reason=s["reason"],
        )

    # P6: the submit path watches for mass rejections, auth failure, and
    # unexpected exceptions — any of them engages the kill switch itself.
    result = OPS_PIPE.submit_with_auto_guards(
        alpaca, to_submit, _send, db_path=DEFAULT_DB, actor="dashboard",
        on_rejected=_on_rejected, decision_date=decision_date,
    )
    if result["notice"]:
        # Either the kill switch was already engaged (fail-closed) or one
        # of the automatic hooks fired mid-batch — say so loudly, in
        # plain English, and stop.
        st.error(result["notice"])
        return
    staged = len(result["submitted"])
    held_back = len(result["rejected"])
    if staged:
        st.success(
            f"Sent {staged} order(s) to Alpaca (decided {decision_date}, "
            f"signals as of {signal_day}). Alpaca will fill them at the "
            "next market open — nothing has traded yet."
        )
        OPS_AUDIT.log_event(
            "orders_staged",
            {
                "decision_date": decision_date,
                "n_submitted": staged,
                "n_held_back": held_back,
                "broker": "alpaca-paper",
            },
            actor="dashboard",
            db_path=DEFAULT_DB,
        )
    if already_waiting:
        st.info(
            f"{already_waiting} order(s) already staged and waiting for "
            "Alpaca — skipped, so nothing is double-staged."
        )
    if held_back:
        st.warning(
            f"{held_back} order(s) couldn't be sent right now — "
            "they stay staged for the next valid open "
            "(see 'Waiting to go to Alpaca' above):"
        )
        for order, error in result["rejected"]:
            st.warning(
                f"{order['side']} {order['ticker']}: {error}"
            )
    if dust:
        st.info(_alpaca_dust_note(dust))
    # Note: no rerun for already_waiting alone — nothing changed, and
    # rerunning would wipe the "already staged" note before it's seen.
    if staged or held_back:
        st.rerun()


def _optimizer_target_kwargs(settings: dict, broker, total: float) -> dict:
    """Extra build_target_for_ui kwargs when "optimized" is the method.

    Rule-based methods ignore these entirely ({} keeps their behavior
    byte-for-byte identical). The optimizer receives the current holdings
    as w_prev — so its turnover penalty measures real trading — plus the
    user's optimizer settings. Callers in replay mode pass total=None to
    leave w_prev at zeros: a replay answers "what would I have done back
    then", where today's holdings don't apply.
    """
    if settings.get("method_key") != "optimized":
        return {}
    w_prev = H.current_weights(broker, total) if total else None
    return {"w_prev": w_prev, "optimizer_settings": settings.get("optimizer")}


def _paper_local_view() -> None:
    """The local simulated broker UI.

    The default view when Alpaca isn't connected; tucked inside the "Local
    simulated broker (offline testing)" expander when Alpaca is the default.
    """
    tickers = H.require_data("prices")["tickers"]

    # Paper page uses next-day-open execution with the spread cost model —
    # the same approach as the backtests, so results stay comparable.
    broker = H.paper_broker(DEFAULT_DB)

    # Morning-after step, on every visit: fill staged orders whose next
    # market open has arrived (idempotent — a refresh never double-fills,
    # and orders with no new data yet simply stay pending).
    try:
        _fill_report = broker.fill_pending_orders()
    except Exception as exc:
        st.error(f"Could not check pending orders: {exc}")
        _fill_report = {
            "filled": [], "skipped": [], "pending_remaining": 0,
            "dividends_credited": 0.0,
        }
    for _f in _fill_report["filled"]:
        _verb = "Bought" if _f["side"] == "BUY" else "Sold"
        st.success(
            f"{_verb} {_f['shares']:,.2f} shares of {_f['ticker']} at "
            f"{H.fmt_money(_f['price'])} on {_f['date']} "
            f"(trading costs {H.fmt_money(_f['cost'])})."
        )
    for _s in _fill_report["skipped"]:
        st.warning(
            f"Could not fill {_s['side']} {_s['ticker']}: {_s['reason']}"
        )
    if _fill_report["dividends_credited"]:
        st.info(
            f"Collected {H.fmt_money(_fill_report['dividends_credited'])} "
            f"in dividend cash while filling pending orders."
        )

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
        st.subheader("Dividends received")
        st.write(
            "When a company you hold pays a dividend, the cash lands here. "
            "Press the button after downloading fresh market data — each "
            "payout is collected once, and the button tells you what arrived."
        )
        if st.button("Collect dividends"):
            try:
                credited = broker.credit_dividends()
                if credited:
                    total_div = sum(p["amount_paid"] for p in credited)
                    st.success(
                        f"Collected ${total_div:,.2f} from "
                        f"{len(credited)} payout(s) — added to your cash."
                    )
                else:
                    st.info("Nothing new — no unpaid dividends found.")
            except Exception as exc:
                st.error(f"Could not collect dividends: {exc}")
        divs = broker.dividends()
        if divs:
            div_df = pd.DataFrame(divs)
            div_df["Paid"] = div_df["amount_paid"].map(H.fmt_money)
            div_df["Per share"] = div_df["amount_per_share"].map(
                lambda a: f"${a:,.4f}"
            )
            st.dataframe(
                div_df[["ex_date", "ticker", "shares_held", "Per share", "Paid"]],
                use_container_width=True,
            )
        else:
            st.info(
                "No dividends collected yet. (Most payouts appear after you "
                "download market data that includes them.)"
            )

    with st.container(border=True):
        st.subheader("Pending orders")
        st.write(
            "Orders you staged but that haven't filled yet — the 'run the "
            "numbers at night' half of paper trading. They fill automatically "
            "at the next market open once new price data arrives: this page "
            "checks on every visit, and the Market data page's 'Refresh new "
            "data' button fills them right after downloading."
        )
        _pendings = broker.pending_orders()
        if not _pendings:
            st.info(
                "No pending orders — nothing waiting for the next market open."
            )
        else:
            try:
                _latest_day = price_trading_days(DEFAULT_DB)[-1]
            except Exception:
                _latest_day = None
            st.write(
                f"**{len(_pendings)} order(s) pending — they will fill at the "
                f"next market open once new price data arrives.**"
            )
            if _latest_day:
                st.caption(
                    f"Prices in the database run through {_latest_day}. A "
                    f"staged order fills at the first opening price after its "
                    f"staged date — share counts below were sized off that "
                    f"date's close, so the fill price (the next open) will "
                    f"differ slightly."
                )
            for _p in _pendings:
                try:
                    _est = broker.last_price(_p["ticker"], _p["staged_date"])
                    _est_txt = H.fmt_money(_est)
                except KeyError:
                    _est_txt = "—"
                _c1, _c2 = st.columns([5, 1])
                with _c1:
                    st.write(
                        f"**{_p['side']}** {_p['shares']:,.2f} shares of "
                        f"**{_p['ticker']}** (about {_est_txt}/share at the "
                        f"{_p['staged_date']} close)"
                    )
                    if _p["reason"]:
                        st.caption(_p["reason"])
                with _c2:
                    if st.button("Cancel", key=f"cancel_pending_{_p['id']}"):
                        broker.cancel_pending_order(_p["id"])
                        st.rerun()
            if st.button("Fill pending orders"):
                # Usually a no-op — the page already checked on load — but
                # explicit and harmless.
                try:
                    _rep = broker.fill_pending_orders()
                except Exception as exc:
                    st.error(f"Could not fill pending orders: {exc}")
                    _rep = None
                if _rep and _rep["filled"]:
                    st.success(f"Filled {len(_rep['filled'])} order(s).")
                elif _rep:
                    st.info(
                        "Nothing to fill yet — still waiting on new price data."
                    )
                st.rerun()

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
                    "Two modes. Leave empty (normal nightly use): signals "
                    "through the latest close are used, orders are STAGED as "
                    "pending, and they fill at the next market open once new "
                    "price data arrives. Set a past date to REPLAY it: orders "
                    "fill immediately at that date's open, the 'what would I "
                    "have done back then' tool. Either way, only signals dated "
                    "strictly before the fill day are used — the same "
                    "no-peeking rule as the backtest."
                ),
            )
        else:
            asof_only = st.text_input(
                "As of date (YYYY-MM-DD)",
                value="",
                key="paper_strategy_date",
                help=(
                    "Two modes. Leave empty (normal nightly use): signals "
                    "through the latest close are used, orders are STAGED as "
                    "pending, and they fill at the next market open once new "
                    "price data arrives. Set a past date to REPLAY it: orders "
                    "fill immediately at that date's open."
                ),
            )
            settings = {"asof": asof_only}
        if st.button("Preview rebalance"):
            try:
                asof = settings["asof"].strip() or None
                if asof is None:
                    # Mirror exactly what "Stage orders" would stage.
                    price_day, signal_cutoff = H.resolve_staging_timing(
                        DEFAULT_DB
                    )
                    price_source = "close"
                else:
                    # Replay preview: what would have traded on the past date.
                    price_day, signal_cutoff = H.resolve_paper_timing(
                        DEFAULT_DB, asof
                    )
                    price_source = "execution"
                if strategy_name is None:
                    portfolio, _, signal_day = H.build_target_for_ui(
                        settings["signal_name"], settings["method_key"],
                        signal_cutoff, tickers, DEFAULT_DB,
                        max_weight=0.05,
                        **_optimizer_target_kwargs(settings, broker, total),
                    )
                else:
                    portfolio, _, signal_day, leg_dates = H.build_strategy_target_for_ui(
                        strategy_name, signal_cutoff, tickers, DEFAULT_DB,
                    )
                preview, notes = H.preview_rebalance_orders(
                    broker, portfolio.weights, price_day, total,
                    cost_bps=broker.cost_bps, price_source=price_source,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            if strategy_name is not None:
                legs = "; ".join(
                    f"{H.signal_display_name(n)}: {d}" for n, d in leg_dates.items()
                )
                st.caption(f"Each input's latest data: {legs}.")
            if asof is None:
                st.caption(
                    f"Decided {price_day} on signals as of {signal_day} — "
                    f"this is exactly what 'Stage orders' would stage below "
                    f"(fills at the next market open)."
                )
                _warn_on_stale_signals(signal_day)
            else:
                st.caption(
                    f"Trading {price_day} on signals as of {signal_day} "
                    f"(signals dated before {price_day} only — no peeking)."
                )
            if not preview:
                st.success("Already at target — no trades needed.")
            else:
                st.dataframe(pd.DataFrame(preview), use_container_width=True)
                st.caption(
                    "Dry run complete — the paper account was not touched. "
                    "Blocked orders explain why in plain English."
                )
                if asof is None:
                    # P6: staging preview shows the pre-trade checklist
                    # inline (nightly mode only — the checklist verifies
                    # today's data, which a historical replay doesn't use).
                    with st.expander(
                        "Pre-trade checklist (what staging will verify)",
                        expanded=False,
                    ):
                        _show_pretrade_report(
                            OPS_PRE.run_pretrade_checks(
                                H.compute_rebalance_orders(
                                    broker, portfolio.weights, price_day,
                                    total, cost_bps=broker.cost_bps,
                                    price_source=price_source,
                                ),
                                PaperBrokerAdapter(broker),
                                db_path=DEFAULT_DB,
                                decision_date=price_day,
                                actor="dashboard (preview)",
                            )
                        )
            for note in notes:
                st.info(f"Note: {note}")

    with st.container(border=True):
        st.subheader("Stage orders for the next market open (paper money only)")
        st.write(
            "Run the numbers now, trade later — the real nightly routine. "
            "Orders are **staged as pending** from data through the latest "
            "close and **fill at the next market open** once new price data "
            "arrives, so the account value doesn't move tonight. (Set an 'As "
            "of date' above to replay a past date instead — replay orders "
            "fill immediately at that date's open.)"
        )
        _asof = settings["asof"].strip() or None
        if _asof:
            st.warning(
                "Replay mode: 'As of date' is set, so orders fill IMMEDIATELY "
                "at that past date's open — the 'what would I have done back "
                "then' tool. Clear the date to stage orders for the next "
                "market open instead."
            )
            _button_label = "Run replay (fills immediately)"
            _confirm_label = (
                "I understand this trades the PAPER account with fake money, "
                "filling immediately at the past date's open."
            )
        else:
            _button_label = "Stage orders for the next open"
            _confirm_label = (
                "I understand this stages orders in the PAPER account with "
                "fake money — nothing trades until the next market open."
            )
        confirm_exec = st.checkbox(
            _confirm_label,
            key="paper_exec_confirm",
        )
        if st.button(_button_label, type="primary",
                     disabled=not confirm_exec):
            if _asof is None:
                # --- nightly staging flow: decide now, fill at next open ---
                # P6: kill-switch-first — refuse before anything is computed.
                if not _ops_trading_allowed():
                    return
                try:
                    decision_date, signal_cutoff = H.resolve_staging_timing(
                        DEFAULT_DB
                    )
                    if strategy_name is None:
                        portfolio, _, signal_day = H.build_target_for_ui(
                            settings["signal_name"], settings["method_key"],
                            signal_cutoff, tickers, DEFAULT_DB,
                            max_weight=0.05,
                            **_optimizer_target_kwargs(settings, broker, total),
                        )
                    else:
                        (portfolio, _, signal_day,
                         _) = H.build_strategy_target_for_ui(
                            strategy_name, signal_cutoff, tickers, DEFAULT_DB,
                        )
                    # Stage the RAW orders (full precision) — never re-parsed
                    # display strings, which round shares and would silently
                    # alter or break small orders. Sized off the latest close
                    # (the fill open isn't known yet); fills re-price at the
                    # actual open.
                    _warn_on_stale_signals(signal_day)
                    orders = H.compute_rebalance_orders(
                        broker, portfolio.weights, decision_date, total,
                        cost_bps=broker.cost_bps, price_source="close",
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    return
                if not orders:
                    st.success("Already at target — no trades needed.")
                else:
                    # P6: the pre-trade checklist gates staging — any
                    # failure blocks with a plain-English reason.
                    _adapter = PaperBrokerAdapter(broker)
                    _report = OPS_PRE.run_pretrade_checks(
                        orders, _adapter, db_path=DEFAULT_DB,
                        decision_date=decision_date, actor="dashboard",
                    )
                    _show_pretrade_report(_report)
                    if not _report.ok:
                        return
                    OPS_AUDIT.log_event(
                        "signals_used",
                        {
                            "decision_date": decision_date,
                            "signal_day": signal_day,
                            "strategy": strategy_name,
                            "n_orders": len(orders),
                            "broker": "local-paper",
                        },
                        actor="dashboard",
                        db_path=DEFAULT_DB,
                    )
                    try:
                        staged = broker.stage_orders(
                            orders, staged_date=decision_date
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        return
                    except Exception as exc:  # noqa: BLE001 - fail-safe
                        # P6 automatic hook: an unexpected exception in the
                        # submit path engages the kill switch itself.
                        OPS_KS.trigger_kill_switch(
                            _adapter,
                            "An unexpected error escaped the local staging "
                            f"path ({exc}).",
                            actor="auto-guard",
                            db_path=DEFAULT_DB,
                        )
                        st.error(
                            "Something unexpected went wrong while staging "
                            f"the orders ({exc}). The kill switch engaged "
                            "itself as a precaution: all open orders were "
                            "canceled and trading is now stopped until you "
                            "manually re-enable it on the Operations page. "
                            "Your positions were not sold."
                        )
                        return
                    OPS_AUDIT.log_event(
                        "orders_staged",
                        {
                            "decision_date": decision_date,
                            "staged": staged["staged"],
                            "skipped_duplicates": staged[
                                "skipped_duplicates"],
                            "broker": "local-paper",
                        },
                        actor="dashboard",
                        db_path=DEFAULT_DB,
                    )
                    st.success(
                        f"Staged {staged['staged']} order(s) decided on "
                        f"{decision_date} (signals as of {signal_day}). They "
                        f"will fill at the next market open once new price "
                        f"data arrives — nothing has been traded yet, and "
                        f"the account value hasn't moved."
                    )
                    if staged["skipped_duplicates"]:
                        st.info(
                            f"{staged['skipped_duplicates']} identical "
                            f"order(s) were already pending — skipped, so "
                            f"nothing is double-staged."
                        )
                    st.rerun()
            else:
                # --- replay flow: immediate fills at the past date's open ---
                # P6: the kill switch gates replay too (it still trades the
                # paper account). The full pre-trade checklist does NOT run
                # here — it checks today's data freshness, which is
                # meaningless for a historical replay.
                if not _ops_trading_allowed():
                    return
                try:
                    trading_day, signal_cutoff = H.resolve_paper_timing(
                        DEFAULT_DB, _asof
                    )
                    if strategy_name is None:
                        # Replay mode: w_prev stays zeros (today's holdings
                        # don't apply to a "back then" replay), so pass
                        # total=None and only the optimizer settings.
                        portfolio, _, _ = H.build_target_for_ui(
                            settings["signal_name"], settings["method_key"],
                            signal_cutoff, tickers, DEFAULT_DB,
                            max_weight=0.05,
                            **_optimizer_target_kwargs(settings, broker, None),
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
                                f"(trading costs {H.fmt_money(result.cost)})."
                            )
                        else:
                            st.warning(f"Skipped {order.action} {order.ticker}: "
                                       f"{result.reason}")
                    st.success("Replay executed in the paper account. "
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

    _risk_status = H.require_data("prices")

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
        tickers = _risk_status["tickers"]
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
# Risk model
# ---------------------------------------------------------------------------

def page_risk_model() -> None:
    st.title("Risk model")
    st.write(
        "A **risk model** learns the handful of shared market patterns that "
        "move groups of stocks together — the broad market rising or "
        "falling, tech stocks moving as a pack, and so on. The optimizer "
        "needs this because without it, it can't tell which bets are really "
        "the *same* bet in disguise: ten stocks that all fall together are "
        "one big risk, not ten small ones, even if each one looks calm on "
        "its own.\n\n"
        "**Tracking error** is how far your portfolio's returns are expected "
        "to wander from the market's (the S&P 500). A 5% tracking error "
        "means roughly ±5% a year vs the market — tighter means you hug "
        "the market, looser means you live or die by your own picks.\n\n"
        "This page fits the platform's own statistical factor model on a "
        "portfolio you pick, shows the patterns it found, forecasts the "
        "tracking error and volatility, and then honestly checks how its "
        "past forecasts held up."
    )

    st.divider()

    _risk_model_status = H.require_data("prices", "signals")
    tickers = _risk_model_status["tickers"]

    with st.container(border=True):
        st.subheader("Pick a portfolio to analyze")
        settings = _portfolio_controls("riskmodel")
        max_weight = None
        if settings["method_key"] != "optimized":
            max_weight = st.slider(
                "Max weight per stock",
                0.05, 0.50, 0.10, 0.05,
                key="riskmodel_maxw",
                help="No single stock may exceed this share of the portfolio.",
            )
        analyze = st.button("Analyze risk", type="primary", key="riskmodel_go")

    if not analyze:
        return

    try:
        asof = settings["asof"].strip() or None
        if asof is None:
            matrix = H.cached_load_signal_matrix(
                settings["signal_name"], tickers, db_path=DEFAULT_DB
            )
            if matrix.empty:
                raise ValueError("No signal data — compute signals first.")
            asof = matrix.index.max().strftime("%Y-%m-%d")
        if settings["method_key"] == "optimized":
            portfolio, _scores, day, _stats = H.build_optimized_target_for_ui(
                settings["signal_name"], asof, tickers, DEFAULT_DB,
                optimizer_settings=settings["optimizer"],
            )
        else:
            portfolio, _scores, day = H.build_target_for_ui(
                settings["signal_name"], settings["method_key"], asof,
                tickers, DEFAULT_DB, max_weight=max_weight,
            )
        if not portfolio.weights:
            raise ValueError(
                "The portfolio came out empty — nothing to analyze."
            )
        fit = H.fit_factor_report_for_ui(
            list(portfolio.weights), day, DEFAULT_DB
        )
        bench, bench_note = H.benchmark_weights_for_ui(
            list(portfolio.weights), day, DEFAULT_DB
        )
        te, vol = H.factor_te_forecast_for_ui(
            portfolio.weights, bench, day, DEFAULT_DB
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.container(border=True):
        st.subheader(f"What the risk model found (as of {day})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shared patterns", f"{fit.n_factors}")
        c2.metric(
            "Past movement explained",
            f"{fit.total_variance_explained:.0%}",
        )
        c3.metric("Forecast tracking error vs S&P 500", f"{te:.1%}")
        c4.metric("Forecast annual volatility", f"{vol:.1%}")
        st.caption(
            "Tracking error: how far this portfolio's returns are expected "
            "to wander from the S&P 500's per year. Volatility: how much "
            "its value is expected to swing per year. Both are forecasts "
            "built from past patterns — not guarantees."
        )
        st.caption(bench_note)
        st.write("**The shared patterns, in plain English**")
        for line in H.factor_descriptions_for_ui(fit):
            st.write(f"- {line}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Pattern": f"Factor {i + 1}",
                        "Share of past movement": f"{v:.1%}",
                    }
                    for i, v in enumerate(fit.variance_explained)
                ]
            ),
            use_container_width=True,
        )
        for note in fit.notes:
            st.caption(f"Note: {note}")

    with st.container(border=True):
        st.subheader("Honest check: past forecasts vs what actually happened")
        st.write(
            "The model stands at four past dates, forecasts the tracking "
            "error using only what was knowable *then*, and compares it "
            "against what this portfolio actually did over the next three "
            "months."
        )
        with st.spinner(
            "Checking past forecasts (this fits the model four times)..."
        ):
            try:
                check = H.factor_calibration_for_ui(
                    portfolio.weights, bench, day, DEFAULT_DB
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "As of": r["asof"],
                        "Forecast": f"{r['forecast_te']:.1%}",
                        "Actually realized": f"{r['realized_te']:.1%}",
                    }
                    for r in check.rows
                ]
            ),
            use_container_width=True,
        )
        st.info(f"Verdict: {check.verdict}")
        for note in check.notes:
            st.caption(f"Note: {note}")

    with st.container(border=True):
        st.subheader("What this model can't do")
        st.write(
            "- The patterns are **anonymous**: the model knows 'pattern 2 "
            "moves these stocks together' but not *why* — it has no sector "
            "or style labels. A licensed risk model (Barra/Axioma) adds "
            "those curated labels.\n"
            "- It learns from the **past year**, so a brand-new market "
            "regime is invisible until it shows up in the data.\n"
            "- Research benchmarks put DIY statistical models at **~80–90% "
            "of a licensed model's forecasting value** — most of the "
            "benefit, free."
        )


# ---------------------------------------------------------------------------
# Data health (P5)
# ---------------------------------------------------------------------------

_SEVERITY_RENDER = {"error": "error", "warning": "warning", "info": "info"}


def page_data_health() -> None:
    st.title("Data health")
    st.write(
        "Everything on this platform — signals, backtests, paper trades — "
        "trusts the prices stored on your Mac. This page checks that trust: "
        "it scans for stale prices, suspicious one-day moves, split and "
        "dividend records that don't match the price history, and local "
        "corruption. **It never changes your data** — it only flags things "
        "worth a look, in plain English. It also keeps permanent stock IDs "
        "and database backups."
    )

    st.divider()

    _ = H.require_data("prices")

    with st.container(border=True):
        st.subheader("Automated checks")
        st.write(
            "Runs four checks over your stored prices: **freshness** (is any "
            "ticker's newest price older than expected?), **outliers** (any "
            "suspicious one-day jumps, or impossible price bars?), "
            "**corporate actions** (do Yahoo's listed splits and dividends "
            "match what the price history shows?), and **spot checks** (do "
            "a few stored prices match a fresh download?). The network "
            "checks sample tickers — flagged ones first — to stay polite to "
            "Yahoo's free service."
        )
        if st.button(
            "Run data-health checks", type="primary", key="dh_run_checks"
        ):
            with st.spinner(
                "Checking prices, splits and dividends "
                "(the network checks take a little while)..."
            ):
                try:
                    report = H.run_data_health_checks_for_ui()
                except Exception as exc:  # never show a traceback
                    st.error(
                        "The checks couldn't run: "
                        f"{exc}. Your data is untouched — try again."
                    )
                    return
            st.session_state["data_health_report"] = report

        report = st.session_state.get("data_health_report")
        if report is None:
            st.caption(
                "No check has been run yet in this session — press the "
                "button above."
            )
        else:
            counts = report.counts()
            st.caption(f"Last checked: {report.checked_at} (UTC)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers checked", f"{report.tickers_checked}")
            c2.metric("Problems", f"{counts['error']}")
            c3.metric("Worth a look", f"{counts['warning']}")
            c4.metric("Notes", f"{counts['info']}")
            st.write(f"**{report.plain_english_summary()}**")
            if not report.findings:
                st.success(
                    "Nothing looks off — prices are fresh, no suspicious "
                    "moves, and the corporate-action records line up."
                )
            else:
                for severity in ("error", "warning", "info"):
                    items = [
                        f for f in report.findings if f.severity == severity
                    ]
                    if not items:
                        continue
                    header = {
                        "error": "Problems — need attention",
                        "warning": "Worth a look",
                        "info": "Notes",
                    }[severity]
                    with st.expander(f"{header} ({len(items)})", expanded=(severity == "error")):
                        for f in items:
                            render = getattr(st, _SEVERITY_RENDER[severity])
                            render(f.summary)
                            if f.ticker:
                                st.caption(f"Ticker: {f.ticker}")
                            if f.detail:
                                st.caption(f.detail)
                            if f.action:
                                st.caption(f"What you can do: {f.action}")

    with st.container(border=True):
        st.subheader("Permanent stock IDs")
        st.write(
            "A **FIGI** is a permanent ID for a stock — tickers change when "
            "companies rebrand or merge, but the FIGI stays the same, so "
            "your history always points at the right company. Mapped once "
            "via the free OpenFIGI service and cached locally; lookups run "
            "about 25 a minute, so a full universe takes a few minutes the "
            "first time and never again."
        )
        notice = st.session_state.pop("dh_figi_notice", None)
        if notice:
            st.info(notice)
        try:
            figi_status = H.figi_mapping_status_for_ui()
        except Exception as exc:
            st.error(f"Couldn't read the ID mappings: {exc}.")
            figi_status = None
        if figi_status is not None:
            st.write(
                f"**{figi_status['mapped']} of {figi_status['total']}** "
                "tickers mapped."
            )
            if figi_status["unmapped"]:
                st.caption(
                    "Not yet mapped: "
                    + ", ".join(figi_status["unmapped"][:20])
                    + (
                        f" and {len(figi_status['unmapped']) - 20} more"
                        if len(figi_status["unmapped"]) > 20
                        else ""
                    )
                )
            table = figi_status["table"]
            if not table.empty:
                st.dataframe(
                    table[["ticker", "figi", "name"]].rename(
                        columns={
                            "ticker": "Ticker",
                            "figi": "FIGI",
                            "name": "Company",
                        }
                    ),
                    use_container_width=True,
                )
            if st.button("Map tickers to FIGIs", key="dh_figi_map"):
                with st.spinner(
                    "Looking up IDs (free service, a few seconds per batch)..."
                ):
                    st.session_state["dh_figi_notice"] = (
                        H.refresh_figi_mapping_for_ui()
                    )
                st.rerun()

    with st.container(border=True):
        st.subheader("Backups")
        st.write(
            "Your database holds the downloaded prices, fundamentals, and "
            "the paper-trading account. Back it up before big changes — "
            "restoring brings everything back. The newest 5 backups are "
            "kept automatically."
        )
        notice = st.session_state.pop("dh_backup_notice", None)
        if notice:
            st.success(notice)
        if st.button("Back up now", type="primary", key="dh_backup_now"):
            with st.spinner("Copying and verifying the database..."):
                st.session_state["dh_backup_notice"] = H.backup_now_for_ui()
            st.rerun()
        backups = H.backup_list_for_ui()
        if not backups:
            st.caption("No backups yet — press “Back up now” to make one.")
        else:
            for b in backups:
                cols = st.columns([3, 2, 2, 3])
                cols[0].write(f"**{b['name']}**")
                cols[1].write(b["created_at"][:16].replace("T", " "))
                cols[2].write(b["size"])
                confirm_key = f"dh_restore_confirm_{b['name']}"
                confirmed = cols[3].checkbox(
                    "I understand this replaces my current database",
                    key=confirm_key,
                )
                if cols[3].button(
                    "Restore",
                    key=f"dh_restore_{b['name']}",
                    disabled=not confirmed,
                ):
                    with st.spinner("Verifying and restoring..."):
                        st.session_state["dh_backup_notice"] = (
                            H.restore_backup_for_ui(
                                b["name"], confirmed=True
                            )
                        )
                    st.rerun()


# ---------------------------------------------------------------------------
# Operations (P6: ops hardening)
# ---------------------------------------------------------------------------

def _ops_broker() -> tuple:
    """The broker the kill switch acts on: Alpaca paper when connected,
    else the local simulated broker. Returns (broker, label)."""
    status = _alpaca_status_cached()
    if status["ok"] and status["broker"] is not None:
        return status["broker"], "Alpaca paper"
    return PaperBrokerAdapter(H.paper_broker(DEFAULT_DB)), "local paper broker"


def page_operations() -> None:
    st.title("Operations")
    st.write(
        "The safety layer: kill switch, audit log, pre-trade checklist, "
        "and data-health alerts. Everything here is **paper trading only** "
        "— fake money, real guardrails."
    )

    broker, broker_label = _ops_broker()

    # -- system status -----------------------------------------------------
    with st.container(border=True):
        st.subheader("System status")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", "Paper trading")
        try:
            engaged = OPS_KS.is_engaged(DEFAULT_DB)
            ks_error = ""
        except OPS_KS.KillSwitchEngaged as exc:
            engaged = True
            ks_error = str(exc)
        c2.metric("Kill switch", "ENGAGED 🛑" if engaged else "Off ✅")
        # P6: the quick local check (stale prices + outliers, no network)
        # — the same report the pre-trade checklist consumes. The full
        # suite with network spot-checks lives on the Data health page.
        try:
            from qrp.data import quality as _quality
            health = _quality.run_data_health_checks(
                DEFAULT_DB, network_sample=0)
            worst = health.worst_severity()
        except Exception:  # noqa: BLE001 - status must never break the page
            health, worst = None, "unknown"
        c3.metric(
            "Data health",
            {"ok": "OK ✅", "info": "Notes ℹ️",
             "warning": "Warnings ⚠️", "error": "Problems ❌"}.get(
                worst, str(worst)),
        )
        if engaged:
            st.error(
                ks_error or OPS_KS.describe_state(DEFAULT_DB),
            )
        else:
            st.success(
                "Trading is allowed — the kill switch is off and staging "
                "may proceed (it still has to pass the pre-trade checklist)."
            )
        if health is not None:
            st.caption(
                f"Data health: {health.plain_english_summary()} "
                f"(checked {health.checked_at[:10]})."
            )

    # -- kill switch --------------------------------------------------------
    with st.container(border=True):
        st.subheader("Kill switch")
        st.write(
            "**What it does:** cancels ALL open orders at the "
            f"{broker_label} immediately, then blocks all staging and "
            "submission until you manually re-enable it."
        )
        st.write(
            "**What it does NOT do:** it never sells your positions. "
            "Canceling open orders and selling everything are different "
            "risks — a sell-everything decision is yours to make "
            "deliberately, never an automatic guard's job. Your holdings "
            "stay exactly as they are."
        )
        st.caption(
            "It also engages itself if a batch sees too many broker "
            f"rejections (more than {OPS_KS.AUTO_TRIGGER_MAX_REJECTIONS}), "
            "if the API keys stop working, or if something unexpected "
            "happens while submitting orders. Re-enabling is always manual "
            "— nothing here resets itself."
        )
        if not engaged:
            if st.session_state.get("ops_confirm_stop"):
                st.warning(
                    "This cancels every open order at the "
                    f"{broker_label} and stops all trading until you "
                    "re-enable it. Your positions will NOT be sold."
                )
                cc1, cc2 = st.columns(2)
                if cc1.button("Yes, stop all trading", type="primary",
                              key="ops_stop_confirm"):
                    with st.spinner("Canceling open orders…"):
                        result = OPS_KS.trigger_kill_switch(
                            broker,
                            "Manual stop from the Operations page",
                            actor="dashboard",
                            db_path=DEFAULT_DB,
                        )
                    st.session_state.pop("ops_confirm_stop", None)
                    n = len(result["canceled"])
                    st.success(
                        f"Kill switch ENGAGED — canceled {n} open order(s). "
                        "Trading is now stopped; positions were not sold."
                    )
                    for failure in result["failed"]:
                        st.warning(
                            f"Could not cancel {failure['id']}: "
                            f"{failure['error']}"
                        )
                    st.rerun()
                if cc2.button("Cancel", key="ops_stop_cancel"):
                    st.session_state.pop("ops_confirm_stop", None)
                    st.rerun()
            elif st.button("🛑 STOP trading", type="primary",
                           key="ops_stop"):
                st.session_state["ops_confirm_stop"] = True
                st.rerun()
            st.caption(
                "No dashboard? Create the STOP file next to the market "
                "database (`touch data/STOP`) — the pipeline honors it "
                "the same way. Command line: "
                "`python -m qrp.ops.killswitch --engage`."
            )
        else:
            st.error(
                "Trading is stopped. Re-enabling is a deliberate manual "
                "action — it never happens on its own."
            )
            confirm = st.checkbox(
                "I understand re-enabling lets order staging run again.",
                key="ops_release_confirm",
            )
            if st.button("Re-enable trading", disabled=not confirm,
                         key="ops_release"):
                OPS_KS.release(actor="dashboard", db_path=DEFAULT_DB)
                st.session_state.pop("ops_release_confirm", None)
                st.success("Kill switch released — trading is allowed again.")
                st.rerun()

    # -- data-health alerts ---------------------------------------------------
    with st.container(border=True):
        st.subheader("Data-health alerts")
        if health is None:
            st.warning("Could not run the data-health checks right now.")
        else:
            counts = health.counts()
            a1, a2, a3 = st.columns(3)
            a1.metric("Problems", counts.get("error", 0))
            a2.metric("Warnings", counts.get("warning", 0))
            a3.metric("Notes", counts.get("info", 0))
            try:
                has_prices = bool(
                    H.data_status(DEFAULT_DB).get("has_prices"))
            except Exception:  # noqa: BLE001 - best-effort status
                has_prices = True
            if not has_prices:
                st.info(
                    "There's no market data here yet — download daily "
                    "stock prices on the Market data page first. The "
                    "pre-trade checklist will block staging until you do."
                )
            bad = [f for f in health.findings
                   if f.severity in ("warning", "error")]
            if bad:
                for f in bad[:10]:
                    (st.error if f.severity == "error" else st.warning)(
                        f.summary
                    )
                    if f.action:
                        st.caption(f.action)
                if len(bad) > 10:
                    st.caption(f"…and {len(bad) - 10} more.")
            else:
                st.success(health.plain_english_summary())
            st.caption(
                "This is the quick local check (stale prices + outliers) — "
                "the same report the pre-trade checklist consumes before "
                "staging. The full check with network spot-checks lives on "
                "the Data health page."
            )
            if st.button("Open the Data health page",
                         key="ops_goto_data_health"):
                H.go_to_page("Data health")

    # -- audit log --------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Audit log")
        st.write(
            "Append-only record of every decision: signals used, staged "
            "orders, pre-trade check results, broker responses, fills, "
            "rejections, queue events, kill-switch actions, and key "
            "save/remove (never the keys themselves). Entries are "
            "hash-chained — tampering or gaps are detected below."
        )
        integrity = OPS_AUDIT.verify_chain(DEFAULT_DB)
        if integrity["ok"]:
            st.success(f"Integrity: {integrity['message']}")
        else:
            st.error(f"Integrity: {integrity['message']}")
        events = OPS_AUDIT.read_events(DEFAULT_DB, limit=5000)
        if not events:
            st.info("Nothing recorded yet — stage an order and it will "
                    "show up here.")
        else:
            types = sorted({e.get("event_type", "") for e in events})
            f1, f2 = st.columns(2)
            picked = f1.multiselect(
                "Event type", types, default=types,
                key="ops_audit_types",
            )
            query = f2.text_input(
                "Search details", value="",
                key="ops_audit_search",
                help="Filters the details text, e.g. a ticker.",
            ).strip().lower()
            shown = [
                e for e in events
                if e.get("event_type") in picked
                and (not query or query in
                     json.dumps(
                         e.get("details", {})).lower())
            ]
            shown = shown[-500:]
            rows = [
                {
                    "Time (UTC)": e.get("ts"),
                    "#": e.get("seq"),
                    "Event": e.get("event_type"),
                    "Actor": e.get("actor"),
                    "Details": json.dumps(
                        e.get("details", {}), sort_keys=True)[:300],
                }
                for e in reversed(shown)
            ]
            st.caption(f"Showing {len(rows)} of {len(events)} recorded "
                       "events (newest first).")
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            def _export_bytes(export_fn, suffix: str) -> bytes | None:
                """Build an export in a temp dir and return its bytes.

                Nothing extra is left on disk — the download button
                serves the bytes directly.
                """
                import shutil
                import tempfile as _tempfile

                tmpdir = _tempfile.mkdtemp(prefix="qrp-audit-")
                try:
                    dest = Path(tmpdir) / f"audit-log.{suffix}"
                    return export_fn(DEFAULT_DB, dest).read_bytes()
                except OSError:
                    return None
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            e1, e2 = st.columns(2)
            with e1:
                csv_bytes = _export_bytes(OPS_AUDIT.export_csv, "csv")
                if csv_bytes is not None:
                    st.download_button(
                        "Download audit log (CSV)", csv_bytes,
                        file_name="audit-log.csv", mime="text/csv",
                        key="ops_audit_csv",
                    )
            with e2:
                jsonl_bytes = _export_bytes(OPS_AUDIT.export_jsonl,
                                            "jsonl")
                if jsonl_bytes is not None:
                    st.download_button(
                        "Download audit log (JSONL)", jsonl_bytes,
                        file_name="audit-log.jsonl",
                        mime="application/jsonl",
                        key="ops_audit_jsonl",
                    )

    # -- pre-trade checklist explainer -------------------------------------------
    with st.container(border=True):
        st.subheader("Pre-trade checklist, in plain English")
        st.write(
            "Every order batch passes these seven checks before anything "
            "is staged or submitted — for Alpaca paper and the local paper "
            "broker alike. Any failure blocks the batch, and the reason is "
            "shown instead of a technical error."
        )
        st.markdown(
            "1. **Kill switch** — trading must not be stopped. (If the "
            "flag file can't even be read, that counts as stopped.)\n"
            "2. **Data freshness** — the database as a whole must be "
            "current, and every stock in the batch must have a current "
            "price. Lagging names outside the batch are left out of the "
            "target and only noted, never blocking.\n"
            "3. **Buying power** — planned buys must fit within "
            f"{OPS_PRE.BUYING_POWER_FACTOR:.0%} of buying power.\n"
            "4. **Position limits** — no stock over "
            f"{OPS_PRE.MAX_POSITION_WEIGHT:.0%} of the account afterwards, "
            "and no duplicate orders.\n"
            "5. **Fat-finger guard** — one order over "
            f"{OPS_PRE.FAT_FINGER_MEDIAN_MULTIPLE:g}× its usual size "
            "(from this account's own order history), over "
            f"${OPS_PRE.FAT_FINGER_MAX_ORDER_VALUE:,.0f}, or (buys only) "
            f"priced more than {OPS_PRE.FAT_FINGER_MAX_PRICE_DRIFT:.0%} "
            "away from the last close is blocked. Sells skip the "
            "price check so an exit is never trapped.\n"
            "6. **Duplicate protection** — every order gets a deterministic "
            "ID (`rebalance-YYYYMMDD-TICKER-side`); an order with an ID "
            "that's already open never goes out twice.\n"
            "7. **No shorting / no margin** — long-only; a sell can never "
            "exceed what you hold."
        )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Every page function referenced here must exist — covered by tests/test_ui.py.
PAGES: dict[str, object] = {
    "Home": page_home,
    "Market data": page_data,
    "Data health": page_data_health,
    "Signals": page_signals,
    "Strategies": page_strategies,
    "Compare strategies": page_compare,
    "Backtest": page_backtest,
    "Backtest wizard": page_wizard,
    "Portfolio": page_portfolio,
    "Paper trading": page_paper,
    "Brokers": page_brokers,
    "Risk": page_risk,
    "Risk model": page_risk_model,
    "Operations": page_operations,
}


_NAV_CSS = """
<style>
/* Modern sidebar navigation: rounded full-width buttons with a clear
   active state in the theme's teal. The active page renders as a
   primary (filled) button; the rest are subtle secondary buttons. */
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px;
    padding: 0.45rem 0.9rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    border-color: #0E7C7B;
    color: #0E7C7B;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Quant Research Platform",
        page_icon="📈",
        layout="wide",
    )
    st.sidebar.title("Quant Research Platform")
    st.sidebar.caption("US stocks · long only · paper trading")
    if H.NAV_PAGE_KEY not in st.session_state:
        st.session_state[H.NAV_PAGE_KEY] = "Home"
    st.sidebar.markdown(_NAV_CSS, unsafe_allow_html=True)
    st.sidebar.caption("Go to")
    for name in PAGES:
        is_active = st.session_state[H.NAV_PAGE_KEY] == name
        if st.sidebar.button(
            name,
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state[H.NAV_PAGE_KEY] = name
            st.rerun()
    st.sidebar.divider()
    _sidebar_status = _alpaca_status_cached()
    if _sidebar_status["ok"]:
        st.sidebar.info(
            "Paper trading through Alpaca's simulator — fake money, real "
            "market mechanics. No real dollars move."
        )
    else:
        st.sidebar.info(
            "Paper trading only — no real broker is connected and no real "
            "money moves."
        )
    st.sidebar.caption(
        "Every strategy is explained in plain English on the Strategies page."
    )
    PAGES[st.session_state[H.NAV_PAGE_KEY]]()


if __name__ == "__main__":
    main()
