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
from qrp.data.yahoo import normalize_ticker
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
    tickers = [normalize_ticker(t) for t in tickers]
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
            dates, frames.get((normalize_ticker(t), tag), pd.DataFrame())
        )
        for t in tickers
    }
    return pd.DataFrame(cols, index=dates)


def _market_cap_matrix(
    tickers: list[str], dates: pd.DatetimeIndex, db_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(market_cap, prices): shares_outstanding(latest filed) x close."""
    prices = _load_prices(tickers, db_path)
    prices = prices.reindex(index=dates, columns=[normalize_ticker(t) for t in tickers])
    shares = fundamental_matrix(
        [normalize_ticker(t) for t in tickers], TAG_SHARES, dates, db_path
    )
    return shares * prices, prices


class _SharedInputs:
    """Expensive intermediates shared across fundamental signals in one run.

    Loading the price matrix and computing value percentiles costs seconds
    on a large universe; the signals CLI computes several fundamental
    signals back-to-back, so it builds one of these and hands it to each
    signal instead of every signal redoing the same work. Safe because one
    run never modifies prices or fundamentals mid-run — each property is
    computed at most once, lazily, on first use.
    """

    def __init__(self, tickers: list[str], db_path: str | Path):
        self.tickers = [normalize_ticker(t) for t in tickers]
        self.db_path = db_path
        self._prices: pd.DataFrame | None = None
        self._mcap: pd.DataFrame | None = None
        self._value_pct: pd.DataFrame | None = None
        self._facts: dict[str, pd.DataFrame] = {}

    @property
    def prices(self) -> pd.DataFrame:
        """Close-price matrix (dates x tickers), loaded once."""
        if self._prices is None:
            self._prices = _load_prices(self.tickers, self.db_path)
        return self._prices

    @property
    def market_cap(self) -> pd.DataFrame:
        """Market cap = latest-filed shares outstanding x close, dates x tickers."""
        if self._mcap is None:
            self._mcap = self.fact_frame(TAG_SHARES) * self.prices
        return self._mcap

    def fact_frame(self, tag: str) -> pd.DataFrame:
        """Latest-known values of one fundamental tag, dates x tickers."""
        if tag not in self._facts:
            self._facts[tag] = fundamental_matrix(
                self.tickers, tag, self.prices.index, self.db_path
            )
        return self._facts[tag]

    @property
    def value_percentiles(self) -> pd.DataFrame:
        """Cross-sectional cheapness percentiles (higher = cheaper)."""
        if self._value_pct is None:
            self._value_pct = _value_percentiles(
                self.tickers, self.prices.index, self.db_path, shared=self
            )
        return self._value_pct


def _value_percentiles(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    db_path: str | Path,
    shared: "_SharedInputs | None" = None,
) -> pd.DataFrame:
    """Cross-sectional cheapness percentiles (0..1; higher = cheaper).

    Three classic value ratios, each using the latest *filed* numbers:
      book_to_market  = equity / market_cap
      earnings_yield  = net_income / market_cap
      cashflow_yield  = cashflow_ops / market_cap
    Each ratio is ranked across stocks (higher ratio = cheaper = better);
    the score is the average percentile rank over the components a stock
    has. Stocks missing 2+ components get NaN (not enough to judge).

    The ranking is fully vectorized: each ratio frame is ranked across
    tickers in one call, then the frames are averaged — no per-day Python
    loop (the old loop took ~5s for 10 tickers x 1696 days; this is ~50x
    faster and the gap grows with universe size). Pass `shared` (a
    _SharedInputs) to reuse already-loaded prices/fundamentals instead of
    reloading them.
    """
    tickers = [normalize_ticker(t) for t in tickers]
    if shared is not None:
        mcap = shared.market_cap.reindex(index=dates, columns=tickers)
        get_frame = lambda tag: shared.fact_frame(tag).reindex(
            index=dates, columns=tickers
        )
    else:
        mcap, _ = _market_cap_matrix(tickers, dates, db_path)
        get_frame = lambda tag: fundamental_matrix(tickers, tag, dates, db_path)  # noqa: E731
    ratios: dict[str, pd.DataFrame] = {}
    for tag in (TAG_EQUITY, TAG_NET_INCOME, TAG_CASHFLOW_OPS):
        vals = get_frame(tag)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[tag] = vals / mcap.replace(0, np.nan)
    # Rank each ratio across tickers (axis=1) in one vectorized call per
    # ratio — identical to ranking each day's cross-section separately.
    ranked = {tag: frame.rank(axis=1, pct=True) for tag, frame in ratios.items()}
    n_have = sum(f.notna() for f in ranked.values())
    total = sum(f.fillna(0.0) for f in ranked.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        out = total / n_have.where(n_have != 0)
    out[n_have < MIN_VALUE_COMPONENTS] = np.nan
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
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = shared if shared is not None else _SharedInputs(tickers, db_path)
        if sh.prices.empty:
            return pd.DataFrame()
        return sh.value_percentiles


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
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = shared if shared is not None else _SharedInputs(tickers, db_path)
        if sh.prices.empty:
            return pd.DataFrame()
        gp = sh.fact_frame(TAG_GROSS_PROFIT)
        assets = sh.fact_frame(TAG_ASSETS)
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
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = shared if shared is not None else _SharedInputs(tickers, db_path)
        prices = sh.prices
        if prices.empty:
            return pd.DataFrame()
        value_pct = sh.value_percentiles
        # NOTE on the momentum convention: this uses the FULL trailing
        # 6 months (~126 trading days) including the most recent month,
        # matching O'Shaughnessy's published 6-month relative-strength
        # spec. The platform's standalone `momentum` signal instead skips
        # the most recent ~21 trading days (12-1) to dodge short-term
        # reversal contamination. Both conventions are deliberate; they
        # differ because the two signals follow different published specs.
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
