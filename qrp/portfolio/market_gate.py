"""Dual-momentum market gate overlay (backlog #3, built 2026-10-11).

What it is: a single on/off switch that sits ON TOP of any strategy. When
the broad market trend is healthy (risk-on), the strategy runs untouched.
When the trend turns negative (risk-off), equity exposure goes to zero and
the portfolio sits in cash until the trend recovers. It never picks stocks
— it only decides *how much* of the strategy's picks to hold.

The rule (a simplified Antonacci dual momentum): risk-on when the S&P 500
(SPY) trailing 12-month total return is positive, risk-off when it is not.
"Total return" means dividends are included — SPY's cash distributions are
in the dividends table, credited on ex-dates. Antonacci's full version also
compares stocks vs bonds and parks in T-bills when both fall; this
simplified gate only asks "is the market's own 12-month trend positive?",
which is the leg that does the crash-protection work.

Point-in-time: the 12-month window ends on the decision date and uses only
prices/distributions known on or before that date — never future data.

Data needed: SPY price history ("Also download SPY" on the Market data
page). Without it the gate raises a plain-English error naming the fix;
it never silently assumes risk-on.

How it plugs in: Strategy has a `market_gate` flag (default False). When
True, strategy_target_portfolio() scales the built weights to zero on a
risk-off date and records a note. The backtest engine, the dashboard
preview, and paper staging all flow through that one function, so the
gate behaves identically everywhere. Precedence (per the #16 design):
safety wins — a risk-off gate overrides the drift buffer when de-risking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qrp.data.store import DEFAULT_DB

MARKET_TICKER = "SPY"
_LOOKBACK_DAYS = 252  # ~12 months of trading days


class MarketGateError(RuntimeError):
    """The gate can't be evaluated — usually missing SPY data."""


def _spy_closes(db_path: str | Path) -> pd.DataFrame:
    """SPY daily closes (+ ex-date dividends) from the local DB."""
    import sqlite3

    try:
        with sqlite3.connect(db_path) as conn:
            bars = pd.read_sql_query(
                "SELECT date, close FROM daily_bars WHERE ticker = ? ORDER BY date",
                conn,
                params=(MARKET_TICKER,),
                parse_dates=["date"],
            )
    except Exception:
        bars = pd.DataFrame()
    if bars.empty:
        raise MarketGateError(
            "The market gate needs S&P 500 price history, but no SPY data "
            "is stored. On the Market data page, tick 'Also download SPY "
            "(S&P 500 benchmark)' and download, then try again — or turn "
            "the market gate off."
        )
    with sqlite3.connect(db_path) as conn:
        try:
            divs = pd.read_sql_query(
                "SELECT ex_date AS date, amount FROM dividends WHERE ticker = ? "
                "ORDER BY ex_date",
                conn,
                params=(MARKET_TICKER,),
                parse_dates=["date"],
            )
        except Exception:
            divs = pd.DataFrame()
    bars = bars.set_index("date").sort_index()
    if not divs.empty:
        # Total return: chain daily price changes plus the day's cash
        # distribution as a fraction of the prior close (the standard
        # ex-date treatment — whoever held SPY overnight is entitled).
        divs = divs.set_index("date").sort_index()
        joined = bars.join(divs, how="left").fillna({"amount": 0.0})
        price_ret = joined["close"].pct_change().fillna(0.0)
        div_ret = (joined["amount"] / joined["close"].shift(1)).fillna(0.0)
        bars["close"] = joined["close"].iloc[0] * (1 + price_ret + div_ret).cumprod()
    return bars


def gate_state(
    date: str | pd.Timestamp, db_path: str | Path = DEFAULT_DB
) -> dict:
    """Risk-on / risk-off as of one date. Returns a plain-data dict.

    {"risk_on": bool, "spy_12m_return": float|None, "asof": "YYYY-MM-DD",
     "reason": str}
    """
    date = pd.Timestamp(date)
    bars = _spy_closes(db_path)
    hist = bars[bars.index <= date]
    if len(hist) < _LOOKBACK_DAYS + 1:
        raise MarketGateError(
            f"The market gate needs 12 months of SPY history before {date.date()} "
            f"but only {len(hist)} trading days are stored. Download a longer "
            "SPY history on the Market data page, or turn the market gate off."
        )
    start_px = hist["close"].iloc[-_LOOKBACK_DAYS - 1]
    end_px = hist["close"].iloc[-1]
    ret = end_px / start_px - 1.0
    risk_on = bool(ret > 0)
    asof = hist.index[-1].date().isoformat()
    return {
        "risk_on": risk_on,
        "spy_12m_return": round(float(ret), 4),
        "asof": asof,
        "reason": (
            f"S&P 500 (SPY) 12-month total return as of {asof} is "
            f"{ret:+.1%} — "
            + ("trend is positive: fully invested."
               if risk_on else
               "trend is negative: moving to cash until it recovers.")
        ),
    }


def apply_gate(
    weights: dict[str, float], date: str | pd.Timestamp,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[dict[str, float], str]:
    """Scale weights by the gate. Returns (new_weights, note).

    Risk-on: weights unchanged, note describes the check. Risk-off: every
    weight goes to zero (100% cash) and the note says so plainly.
    """
    state = gate_state(date, db_path)
    if state["risk_on"]:
        return weights, f"Market gate: {state['reason']}"
    note = (
        f"Market gate: {state['reason']} All equity positions scaled to "
        "zero — the portfolio sits in cash until the 12-month trend turns "
        "positive again."
    )
    return {}, note
