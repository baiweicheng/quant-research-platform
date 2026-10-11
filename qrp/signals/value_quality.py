"""Value & quality classic signals (backlog #2, built 2026-10-11).

Five well-known academic/practitioner signals, implemented from their
canonical definitions (the standing rule when a source text is garbled):
Greenblatt's Magic Formula, Piotroski's F-score, a lite Quality-Minus-Junk,
shareholder yield, and long-horizon reversal. All follow the FundamentalSignal
contract: compute_all(tickers, db_path, shared) -> dates x tickers scores,
higher = more attractive long, NaN where unknowable. All point-in-time: only
SEC filings published on or before each date ever enter a score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qrp.data.sectors import is_financial_or_utility
from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.signals.fundamental import (
    TAG_ASSETS,
    TAG_CASHFLOW_OPS,
    TAG_EQUITY,
    TAG_GROSS_PROFIT,
    TAG_NET_INCOME,
    TAG_SHARES,
    TTM_TAGS,
    FundamentalSignal,
    _SharedInputs,
)

# Extra canonical tags these signals need (see qrp/data/edgar.py).
TAG_EBIT = "ebit"
TAG_CASH = "cash"
TAG_DEBT_LT = "debt_longterm"
TAG_ASSETS_CUR = "assets_current"
TAG_LIAB_CUR = "liabilities_current"
TAG_PPE_NET = "ppe_net"
TAG_REVENUE = "revenue"
TAG_DIV_PAID = "dividends_paid"

_TRADING_YEAR = 252  # one year of daily bars, for YoY comparisons

# Flow tags that must be standardized to trailing-twelve-months before any
# cross-sectional comparison (a 10-Q quarter is never ranked against a
# 10-K year). sh.ttm_frame() handles any tag; these are the extra ones
# beyond fundamental.py's TTM_TAGS that these signals need.
# (TAG_EBIT is defined with the other tag constants above.)


def _pct_rank_higher_better(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks, 0..1, higher = better."""
    return frame.rank(axis=1, pct=True)


def _average_available(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean percentile over whichever components a stock actually has."""
    have = sum(f.notna().astype(float) for f in frames)
    total = sum(f.fillna(0.0) for f in frames)
    return total / have.replace(0.0, np.nan)


def _require_shared(
    tickers: list[str], db_path: str | Path, shared: "_SharedInputs | None"
) -> "_SharedInputs":
    tickers = [normalize_ticker(t) for t in tickers]
    return shared if shared is not None else _SharedInputs(tickers, db_path)


def _ttm(sh: "_SharedInputs", tag: str) -> pd.DataFrame:
    """TTM frame for a flow tag (ttm_frame is generic over tags)."""
    return sh.ttm_frame(tag)


class MagicFormula(FundamentalSignal):
    """Greenblatt's Magic Formula: cheap AND good.

    Two ranks, averaged: (1) earnings yield = trailing-12-month EBIT ÷
    enterprise value, where EV = market cap + (long-term debt − cash)
    (net cash reduces EV, raising the yield for cash-rich firms, as in
    Greenblatt); (2) return on capital = TTM EBIT ÷ invested capital,
    where invested capital = (current assets − current liabilities)
    + net property/plant/equipment — Greenblatt's net working capital +
    net fixed assets, using the balance-sheet tags ingested from SEC
    filings. Financials and Utilities are excluded — enterprise value is
    meaningless for a bank — using the GICS table when it has been
    refreshed; a stock whose sector is unknown is kept (excluding on a
    guess would be worse).

    Missing data: a missing long-term-debt or cash tag is treated as zero
    (debt is a disclosed liability — absent means none reported); a missing
    current-assets, current-liabilities, or PPE tag leaves the ROC leg
    unscorable, so the stock is ranked on earnings yield alone.

    Direction: higher = cheap for its earnings power AND earns a lot on
    its capital = more attractive long.
    """

    name = "magic_formula"
    formal_name = "Magic Formula (Greenblatt: Earnings Yield + Return on Capital)"
    era = 4  # value & quality classics (chronological display order)
    description = (
        "Joel Greenblatt's two-part screen: it favors stocks that are both "
        "cheap for their operating earnings (earnings yield) and genuinely "
        "good businesses (high return on the capital tied up in them). "
        "Banks and utilities are left out, since the math doesn't apply to "
        "them."
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
        sh = _require_shared(tickers, db_path, shared)
        if sh.prices.empty:
            return pd.DataFrame()
        # Extra TTM tags through the shared cache.
        ebit = _ttm(sh, TAG_EBIT)
        mcap = sh.market_cap
        debt = sh.fact_frame(TAG_DEBT_LT).fillna(0.0)
        cash = sh.fact_frame(TAG_CASH).fillna(0.0)
        # Canonical Greenblatt invested capital: net working capital
        # (current assets − current liabilities) + net fixed assets (PPE).
        nwc = sh.fact_frame(TAG_ASSETS_CUR) - sh.fact_frame(TAG_LIAB_CUR)
        ppe = sh.fact_frame(TAG_PPE_NET)
        invested = nwc + ppe
        # Canonical EV: market cap + net debt (net cash reduces EV).
        net_debt = debt - cash
        ev = mcap + net_debt
        with np.errstate(divide="ignore", invalid="ignore"):
            ey = ebit / ev.replace(0, np.nan)
            roc = ebit / invested.replace(0, np.nan)
        out = _average_available(
            [_pct_rank_higher_better(ey), _pct_rank_higher_better(roc)]
        )
        # Greenblatt excludes financials/utilities (EV is meaningless for a
        # bank). Only when the sector table knows the ticker.
        for t in tickers:
            if is_financial_or_utility(t, db_path):
                out[t] = np.nan
        return out


class PiotroskiFScore(FundamentalSignal):
    """Piotroski (2000) F-score: nine yes/no tests of financial strength.

    Each stock gets one point per test passed (0–9):
    profitability — (1) ROA > 0, (2) operating cash flow > 0,
    (3) ROA higher than a year ago, (4) cash flow exceeds net income
    (accruals check: earnings backed by cash, not accounting);
    leverage/liquidity — (5) long-term-debt-to-assets lower than a year
    ago, (6) current ratio higher than a year ago, (7) no new shares
    issued over the year; operating efficiency — (8) gross margin higher
    than a year ago, (9) asset turnover higher than a year ago.
    "A year ago" is the latest-known filing 252 trading days earlier, so
    every comparison is point-in-time. Piotroski's finding: among cheap
    (high book-to-market) stocks, high F-scores kept winning and low
    scores kept losing — it separates real recoveries from value traps.

    Direction: higher score (more tests passed) = stronger finances =
    more attractive long.
    """

    name = "piotroski"
    formal_name = "Piotroski F-Score (9-Point Financial Strength)"
    era = 4
    description = (
        "Nine yes/no health checks on the financial statements — is it "
        "profitable, is the profit real cash, is debt shrinking, is it "
        "getting more efficient? Each yes is a point, out of nine. High "
        "scores mark financially strong companies; it was designed to tell "
        "real bargains apart from value traps."
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
        sh = _require_shared(tickers, db_path, shared)
        if sh.prices.empty:
            return pd.DataFrame()
        ni = sh.ttm_frame(TAG_NET_INCOME)
        cfo = sh.ttm_frame(TAG_CASHFLOW_OPS)
        gp = sh.ttm_frame(TAG_GROSS_PROFIT)
        rev = _ttm(sh, TAG_REVENUE)
        assets = sh.fact_frame(TAG_ASSETS)
        debt = sh.fact_frame(TAG_DEBT_LT)
        ca = sh.fact_frame(TAG_ASSETS_CUR)
        cl = sh.fact_frame(TAG_LIAB_CUR)
        shares = sh.fact_frame(TAG_SHARES)

        def ago(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.shift(_TRADING_YEAR)

        with np.errstate(divide="ignore", invalid="ignore"):
            roa = ni / assets.replace(0, np.nan)
            lev = debt / assets.replace(0, np.nan)
            liq = ca / cl.replace(0, np.nan)
            gm = gp / rev.replace(0, np.nan)
            ato = rev / assets.replace(0, np.nan)

        def both(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            return a.notna() & b.notna()

        # (passed, answerable): a NaN comparison is False in pandas, so
        # answerability is tracked from the inputs, not the test result —
        # otherwise "no data" would score as "failed".
        specs = [
            (roa > 0, roa.notna()),
            (cfo > 0, cfo.notna()),
            (roa > ago(roa), both(roa, ago(roa))),
            (cfo > ni, both(cfo, ni)),  # accruals: cash beats accounting
            (lev < ago(lev), both(lev, ago(lev))),
            (liq > ago(liq), both(liq, ago(liq))),
            (shares <= ago(shares), both(shares, ago(shares))),  # no dilution
            (gm > ago(gm), both(gm, ago(gm))),
            (ato > ago(ato), both(ato, ago(ato))),
        ]
        score = sum(p.fillna(False).astype(float) for p, _ in specs)
        answerable = sum(ok.astype(float) for _, ok in specs)
        # A stock needs at least 5 answerable tests to be scored at all.
        return score.where(answerable >= 5)


class QMJLite(FundamentalSignal):
    """Quality-Minus-Junk, lite version (Asness–Frazzini–Pedersen 2019).

    Quality = profitable + growing + safe. Profitability: gross profits ÷
    assets, ROE, ROA, operating cash flow ÷ assets, and low accruals
    (accounting profit far above cash profit is a red flag). Growth: the
    one-year change in gross profitability and in ROE. Safety: low trailing
    one-year price volatility and low leverage (debt ÷ equity). Each
    component is ranked across stocks (higher = better) and the score is
    the average over the components a stock actually has. The academic
    finding: high-quality stocks earn higher returns — investors
    systematically underpay for boring, reliable earners.

    Direction: higher = higher quality = more attractive long.
    """

    name = "qmj_lite"
    formal_name = "Quality-Minus-Junk Lite (Profitable + Growing + Safe)"
    era = 4
    description = (
        "A quality screen in the spirit of AQR's research: it favors "
        "companies that are profitable, getting more profitable, and safe "
        "— low debt, calm share price. Quality businesses tend to keep "
        "outperforming because investors underpay for reliable earners."
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
        sh = _require_shared(tickers, db_path, shared)
        if sh.prices.empty:
            return pd.DataFrame()
        gp = sh.ttm_frame(TAG_GROSS_PROFIT)
        ni = sh.ttm_frame(TAG_NET_INCOME)
        cfo = sh.ttm_frame(TAG_CASHFLOW_OPS)
        assets = sh.fact_frame(TAG_ASSETS)
        equity = sh.fact_frame(TAG_EQUITY)
        debt = sh.fact_frame(TAG_DEBT_LT)
        prices = sh.prices
        with np.errstate(divide="ignore", invalid="ignore"):
            gp_a = gp / assets.replace(0, np.nan)
            roe = ni / equity.replace(0, np.nan)
            roa = ni / assets.replace(0, np.nan)
            cfo_a = cfo / assets.replace(0, np.nan)
            accruals = (ni - cfo) / assets.replace(0, np.nan)
            lev = debt / equity.replace(0, np.nan)
        rets = prices / prices.shift(1) - 1.0
        vol = rets.rolling(_TRADING_YEAR, min_periods=126).std()

        def ago(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.shift(_TRADING_YEAR)

        growth_gp = gp_a - ago(gp_a)
        growth_roe = roe - ago(roe)
        comps = [
            _pct_rank_higher_better(gp_a),
            _pct_rank_higher_better(roe),
            _pct_rank_higher_better(roa),
            _pct_rank_higher_better(cfo_a),
            _pct_rank_higher_better(-accruals),  # low accruals = better
            _pct_rank_higher_better(growth_gp),
            _pct_rank_higher_better(growth_roe),
            _pct_rank_higher_better(-vol),  # calm price = safer
            _pct_rank_higher_better(-lev),  # low leverage = safer
        ]
        return _average_available(comps)


class ShareholderYield(FundamentalSignal):
    """Cash returned to shareholders: dividends + buybacks, ÷ market cap.

    Shareholder yield = (trailing-12-month dividends paid + net buybacks)
    ÷ market cap. Net buybacks are measured from shrinking shares
    outstanding over the trailing year: (shares a year ago − shares now) ×
    price, floored at zero (issuance is not negative yield here — it is
    captured by the Piotroski dilution test instead). The idea: a company
    steadily handing cash back is usually a disciplined, shareholder-minded
    business, and the yield itself is a bird-in-the-hand return.

    Direction: higher = more cash returned per dollar of price =
    more attractive long.
    """

    name = "shareholder_yield"
    formal_name = "Shareholder Yield (Dividends + Buybacks ÷ Market Cap)"
    era = 4
    description = (
        "How much cash the company hands back to its owners — dividends "
        "plus share buybacks — compared to its price. A high yield means "
        "you're paid well just for holding, and steady buybacks usually "
        "signal a disciplined management team."
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
        sh = _require_shared(tickers, db_path, shared)
        if sh.prices.empty:
            return pd.DataFrame()
        div = _ttm(sh, TAG_DIV_PAID).fillna(0.0)
        shares = sh.fact_frame(TAG_SHARES)
        prices = sh.prices
        mcap = sh.market_cap
        with np.errstate(divide="ignore", invalid="ignore"):
            retired = (shares.shift(_TRADING_YEAR) - shares).clip(lower=0.0)
            buybacks = retired * prices
            s_yield = (div + buybacks) / mcap.replace(0, np.nan)
        return _pct_rank_higher_better(s_yield)


class LongHorizonReversal(FundamentalSignal):
    """Long-term reversal (De Bondt & Thaler 1985): buy the forgotten.

    Score from the trailing 5-year price return SKIPPING the most recent
    year (price 5 years ago → price 1 year ago). The worst past performers
    score highest. The academic finding: investors overreact — they pile
    into long-run winners and abandon long-run losers — so the abandoned
    "losers" tend to bounce back over the next few years. Skipping the
    most recent year keeps this distinct from ordinary momentum (which
    likes recent winners); the two are meant to be complementary.

    Direction: higher = worse 5-year-ex-last-year return = stronger
    reversal candidate = more attractive long.
    """

    name = "reversal_long"
    formal_name = "Long-Horizon Reversal (5Y ex-1Y Losers)"
    era = 4
    description = (
        "It buys yesterday's forgotten stocks: the worst performers over "
        "the past five years, ignoring the most recent year. Investors "
        "overreact — they abandon long-run losers beyond what the business "
        "deserves — and those stocks tend to bounce back. The natural "
        "complement to momentum, which likes recent winners."
    )
    lookback_days = 5 * _TRADING_YEAR

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = _require_shared(tickers, db_path, shared)
        prices = sh.prices
        if prices.empty:
            return pd.DataFrame()
        with np.errstate(divide="ignore", invalid="ignore"):
            past = prices.shift(_TRADING_YEAR) / prices.shift(5 * _TRADING_YEAR) - 1.0
        return _pct_rank_higher_better(-past)  # losers rank highest


__all__ = [
    "MagicFormula",
    "PiotroskiFScore",
    "QMJLite",
    "ShareholderYield",
    "LongHorizonReversal",
]
