"""Fundamental + blended signals (Phase 8).

What "fundamental" signals are, in plain English
------------------------------------------------
Price-only signals (Phase 2) look at what a stock's price *did*. Fundamental
signals look at what the company *is*: how cheap it is for its earnings,
how profitable its business is. They come from the official financial
reports companies file with the SEC (Phase 7), not from prices.

THE TOP CORRECTNESS RULE — NO PEEKING:
    A company number is only "known" on the day it was actually *filed*
    with the SEC, never on the day the business period ended. Every value
    in this module is computed from facts whose filing date is on or before
    the signal date. A restated number filed later is invisible to earlier
    dates. This is enforced by `_latest_known_asof()` below and pinned by
    tests/test_strategies.py::test_fundamentals_are_point_in_time.

Why these signals are cross-sectional
--------------------------------------
Price signals (momentum, trend) are computed one ticker at a time. Value
signals cannot be: "cheap" only means something *compared to other stocks*.
So these signals compute the whole cross-section at once — every ticker,
every date — then store one per-ticker series each in the signals table,
exactly like the price signals. The backtest engine, the strategy registry
and the UI all read them the same way.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from qrp.data.store import DEFAULT_DB
from qrp.signals.base import Signal

# NOTE: qrp.backtest.engine imports qrp.signals.store at module level, so
# importing load_price_matrix here at module level would be circular. The
# helper below imports it lazily (always safe: by call time both modules
# are fully initialized).
def _load_prices(tickers: list[str], db_path: str | Path) -> pd.DataFrame:
    from qrp.backtest.engine import load_price_matrix

    return load_price_matrix(tickers, path=db_path)

# Canonical fundamental tags (see qrp/data/edgar.py TAG_FALLBACKS).
TAG_EQUITY = "equity"
TAG_NET_INCOME = "net_income"
TAG_CASHFLOW_OPS = "cashflow_ops"
TAG_GROSS_PROFIT = "gross_profit"
TAG_ASSETS = "assets"
TAG_SHARES = "shares_outstanding"

# A stock needs at least this many of the three value components to get a
# value_composite score; with fewer, there is not enough to judge cheapness.
MIN_VALUE_COMPONENTS = 2


class FundamentalSignal(Signal):
    """A signal that needs company fundamentals as well as prices.

    Unlike price-only signals, these cannot be computed from one ticker's
    bars alone: fundamentals live in the SQLite `fundamentals` table
    (Phase 7, SEC EDGAR) and value composites need the whole cross-section
    of tickers at once.

    Subclasses implement ``compute_all(tickers, db_path)`` returning a
    dates-x-tickers DataFrame of scores (NaN where unknowable). The
    per-ticker ``compute(df)`` raises — the signals CLI calls
    ``compute_all`` for these instead.
    """

    def compute(self, df: pd.DataFrame) -> pd.Series:  # noqa: D102
        raise NotImplementedError(
            f"{self.name} needs fundamentals and the full ticker cross-section; "
            "use compute_all(tickers, db_path) instead."
        )

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
    ) -> pd.DataFrame:
        """Scores for every ticker on every trading day (dates x tickers)."""
        raise NotImplementedError


def _load_fact_frames(
    tickers: list[str],
    tags: list[str],
    db_path: str | Path = DEFAULT_DB,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Bulk-load fundamental facts for (ticker, tag) pairs.

    One SQL query for everything (fast), then pandas does the
    point-in-time work. Each frame has datetime columns filed_date,
    period_end and float column value, sorted by filed_date.
    """
    tickers = [t.strip().upper() for t in tickers]
    path = Path(db_path)
    if not path.exists() or not tickers or not tags:
        return {}
    tq = ",".join("?" for _ in tickers)
    gq = ",".join("?" for _ in tags)
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT ticker, tag, period_end, filed_date, value FROM fundamentals"
            f" WHERE ticker IN ({tq}) AND tag IN ({gq})",
            conn,
            params=[*tickers, *tags],
        )
    if df.empty:
        return {}
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for (ticker, tag), grp in df.groupby(["ticker", "tag"]):
        frames[(ticker, tag)] = grp.sort_values("filed_date").reset_index(drop=True)
    return frames


def latest_known_asof(
    dates: pd.DatetimeIndex, facts: pd.DataFrame
) -> pd.Series:
    """For each date, the latest-known value of one (ticker, tag) fact series.

    "Latest-known" means: among facts whose SEC filing date is on or before
    the date, take the one with the most recent reporting period. A fact
    filed *after* the date never influences the value — this is the
    no-peeking rule, and it matches qrp.data.edgar.facts_as_of() exactly
    (pinned by test).

    Args:
        dates: the trading-day grid to produce values for.
        facts: DataFrame with datetime columns filed_date, period_end and
            float column value (one row per reporting period, as stored).

    Returns:
        Series indexed like `dates`; NaN before the first filing.
    """
    out = pd.Series(np.nan, index=dates, dtype=float)
    if facts.empty:
        return out
    # Walk filings in chronological order, remembering the value belonging
    # to the most recent reporting period seen so far. A restatement of an
    # *older* period that arrives later does not move the needle: the
    # newest period's number is still the latest known picture.
    events: list[tuple[pd.Timestamp, float]] = []
    best_end: pd.Timestamp | None = None
    best_val = np.nan
    for filed, grp in facts.groupby("filed_date", sort=True):
        row = grp.loc[grp["period_end"].idxmax()]
        if best_end is None or row["period_end"] >= best_end:
            best_end = row["period_end"]
            best_val = float(row["value"])
        events.append((filed, best_val))
    ev = pd.DataFrame(events, columns=["filed_date", "value"]).sort_values(
        "filed_date"
    )
    grid = pd.DataFrame({"date": dates}).sort_values("date")
    merged = pd.merge_asof(
        grid, ev, left_on="date", right_on="filed_date", direction="backward"
    )
    return pd.Series(merged["value"].to_numpy(), index=dates)


def fundamental_matrix(
    tickers: list[str],
    tag: str,
    dates: pd.DatetimeIndex,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Latest-known values of one fundamental tag: dates x tickers."""
    frames = _load_fact_frames(tickers, [tag], db_path)
    cols = {
        t: latest_known_asof(
            dates, frames.get((t.strip().upper(), tag), pd.DataFrame())
        )
        for t in tickers
    }
    return pd.DataFrame(cols, index=dates)


def _market_cap_matrix(
    tickers: list[str], dates: pd.DatetimeIndex, db_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(market_cap, prices): shares_outstanding(latest filed) x close."""
    prices = _load_prices(tickers, db_path)
    prices = prices.reindex(index=dates, columns=[t.strip().upper() for t in tickers])
    shares = fundamental_matrix(
        [t.strip().upper() for t in tickers], TAG_SHARES, dates, db_path
    )
    return shares * prices, prices


def _value_percentiles(
    tickers: list[str], dates: pd.DatetimeIndex, db_path: str | Path
) -> pd.DataFrame:
    """Cross-sectional cheapness percentiles (0..1; higher = cheaper).

    Three classic value ratios, each using the latest *filed* numbers:
      book_to_market  = equity / market_cap
      earnings_yield  = net_income / market_cap
      cashflow_yield  = cashflow_ops / market_cap
    Each ratio is ranked across stocks (higher ratio = cheaper = better);
    the score is the average percentile rank over the components a stock
    has. Stocks missing 2+ components get NaN (not enough to judge).
    """
    tickers = [t.strip().upper() for t in tickers]
    mcap, _ = _market_cap_matrix(tickers, dates, db_path)
    ratios: dict[str, pd.DataFrame] = {}
    for tag in (TAG_EQUITY, TAG_NET_INCOME, TAG_CASHFLOW_OPS):
        vals = fundamental_matrix(tickers, tag, dates, db_path)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[tag] = vals / mcap.replace(0, np.nan)
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for day in dates:
        row = pd.DataFrame({k: v.loc[day] for k, v in ratios.items()})
        enough = row.notna().sum(axis=1) >= MIN_VALUE_COMPONENTS
        pct = row.rank(pct=True)  # higher ratio -> higher percentile -> cheaper
        score = pct.mean(axis=1, skipna=True)
        score[~enough] = np.nan
        out.loc[day] = score
    return out


class ValueComposite(FundamentalSignal):
    """Averaged cheapness rank from book value, earnings and cash flow.

    For each stock, three "how much company do I get per dollar of price"
    ratios (book-to-market, earnings yield, cash-flow yield) are ranked
    across all stocks; the score is the average percentile rank. A high
    score means the stock is cheap on all three classic value measures at
    once — combining them so no single accounting quirk can mislead.

    Direction: higher = cheaper for its fundamentals = more attractive long.
    """

    name = "value_composite"
    description = (
        "How cheap the stock looks next to its real business numbers — its "
        "book value, earnings, and cash flow, each compared to its price. "
        "A high score means you pay less for each dollar of what the "
        "company is worth. It blends three classic cheapness measures so "
        "one odd accounting number can't fool it."
    )
    lookback_days = 0  # NaN appears naturally before the first filing

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
    ) -> pd.DataFrame:
        tickers = [t.strip().upper() for t in tickers]
        prices = _load_prices(tickers, db_path)
        if prices.empty:
            return pd.DataFrame()
        return _value_percentiles(tickers, prices.index, db_path)


class Profitability(FundamentalSignal):
    """Gross profit divided by total assets (latest filed numbers).

    How much profit the business squeezes out of everything it owns.
    Quality companies — the ones that earn a lot from their assets — tend
    to keep outperforming, because investors underpay for boring,
    reliable earners.

    Direction: higher = more profitable = more attractive long.
    """

    name = "profitability"
    description = (
        "How profitable the company is: its gross profit (sales minus what "
        "it cost to make them) divided by everything the company owns. A "
        "high score means the business earns a lot from its assets — "
        "quality businesses like that tend to keep doing well."
    )
    lookback_days = 0

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
    ) -> pd.DataFrame:
        tickers = [t.strip().upper() for t in tickers]
        prices = _load_prices(tickers, db_path)
        if prices.empty:
            return pd.DataFrame()
        dates = prices.index
        gp = fundamental_matrix(tickers, TAG_GROSS_PROFIT, dates, db_path)
        assets = fundamental_matrix(tickers, TAG_ASSETS, dates, db_path)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = gp / assets.replace(0, np.nan)
        return out


class TrendingValue(FundamentalSignal):
    """Cheap AND climbing: value rank blended with 6-month momentum rank.

    The average of the value_composite percentile and the 6-month price
    return percentile. The idea (O'Shaughnessy's "Trending Value"): value
    tells you *what* to buy, momentum tells you *when* — it avoids "value
    traps," cheap stocks that keep getting cheaper because the market
    knows something you don't.

    Direction: higher = cheap and rising = more attractive long.
    """

    name = "trending_value"
    description = (
        "A blend of cheapness and momentum: it favors stocks that are both "
        "inexpensive for their business numbers AND already climbing. The "
        "idea is simple — cheap tells you what to buy, a rising price tells "
        "you the market is starting to agree — which sidesteps 'value "
        "traps', cheap stocks that just keep sinking."
    )
    lookback_days = 0

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
    ) -> pd.DataFrame:
        tickers = [t.strip().upper() for t in tickers]
        prices = _load_prices(tickers, db_path)
        if prices.empty:
            return pd.DataFrame()
        dates = prices.index
        value_pct = _value_percentiles(tickers, dates, db_path)
        ret6 = prices / prices.shift(126) - 1.0  # ~6 months of trading days
        mom_pct = ret6.rank(axis=1, pct=True)
        # Average over whichever halves are available; NaN only when a
        # stock has neither a value score nor a 6-month return yet.
        have = value_pct.notna().astype(float) + mom_pct.notna().astype(float)
        out = (value_pct.fillna(0.0) + mom_pct.fillna(0.0)) / have.replace(
            0.0, np.nan
        )
        return out


__all__ = [
    "FundamentalSignal",
    "ValueComposite",
    "Profitability",
    "TrendingValue",
    "fundamental_matrix",
    "latest_known_asof",
]
