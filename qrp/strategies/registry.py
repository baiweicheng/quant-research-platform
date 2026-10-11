"""Named investment strategies (Phase 8; research batch 3 added later).

A **strategy** is a complete, named recipe: which signal(s) rank the
stocks, how positions are sized, how often the portfolio is rebalanced,
and how much drift is tolerated before trading. Strategies are the unit
the user picks in the UI — one dropdown on the Backtest, Portfolio and
Paper-trading pages — so a new idea only needs a registry entry to become
backtestable, explainable, and comparable everywhere.

Strategies are listed chronologically by build era (see ERA_LABELS):
  era 1 — the three original price-only technical strategies;
  era 2 — the three fundamental-blended institutional strategies;
  era 3 — the three research-paper strategies.
`list_strategies()` always returns them in that order, so every picker,
page and table in the UI shows the same chronological sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from qrp.backtest.engine import StrategyFn, load_signal_matrix
from qrp.backtest.strategies import WEIGHTING_METHODS
from qrp.data.store import DEFAULT_DB
from qrp.portfolio.construction import (
    INVERSE_VOLATILITY,
    TargetPortfolio,
    build_target_weights,
    implied_stock_count,
)


@dataclass
class Strategy:
    """One named, backtestable investment recipe.

    name: machine-readable id, e.g. "trending_value".
    description: plain-English what-it-does and why (shown in the UI).
    signals: list of (signal_name, weight); multi-signal strategies blend
        the cross-sectional ranks of each signal by weight.
    weighting: "equal", "signal" or "inverse_vol" (position sizing).
    rebalance: "daily", "weekly" or "monthly" (default "monthly", the
        academic convention).
    max_weight: no single stock may exceed this fraction of capital. This is
        the single position-sizing knob: it also implies how many stocks are
        held, n = round(1 / max_weight) (e.g. 5% -> ~20 stocks).
    drift_tolerance: skip trading a position whose target weight is within
        this of its current weight (avoids churn on tiny moves).
    data_used: plain-English, e.g. "Prices only" or "Prices + company
        financials". Shown on the Strategies comparison page.
    good_at: one plain-English line on what the strategy is good at.
    main_risk: one plain-English line on its main risk.
    formal_name: the user-facing formal name, e.g. "12-1 Month
        Cross-Sectional Momentum". The machine-readable `name` above stays
        the stable internal key (CLI, tests, saved state); only display
        labels change.
    calculation: step-by-step plain-English math describing exactly how the
        strategy scores stocks and builds the portfolio — the real formulas
        from the code (lookback windows, ranking, weighting, cutoffs), kept
        readable for non-finance users.
    cutoffs: optional absolute filters on raw signal values, applied BEFORE
        cross-sectional ranking: {signal_name: minimum_value}. A stock whose
        raw signal is below the cutoff is not a candidate at all — it is
        never ranked, never bought, and its weight sits in cash. This is
        how a strategy says "only hold stocks that truly clear the bar"
        instead of settling for the least-bad stocks in a bear market.
    market_gate: when True, the dual-momentum market gate overlay applies:
        on a risk-off date (SPY 12-month total return negative) the built
        weights are scaled to zero and the portfolio sits in cash until
        the trend recovers. Safety wins over the drift buffer here.
    max_stocks: explicit stock-count override. When None (the default),
        the count is implied by the cap (round(1 / max_weight)) — one
        knob, as before. Variants may set it directly, e.g. 10 stocks
        with a 5% cap (which structurally leaves cash on the table —
        the variant validator warns about that).
    era: build era (1, 2, 3 or 4 — see ERA_LABELS). Controls the chronological
        display order everywhere strategies are listed; internal keys are
        unaffected.
    """

    name: str
    description: str
    formal_name: str = ""
    calculation: str = ""
    signals: list[tuple[str, float]] = field(default_factory=list)
    weighting: str = "equal"
    rebalance: str = "monthly"
    max_weight: float = 0.10
    drift_tolerance: float = 0.02
    data_used: str = "Prices only"
    good_at: str = ""
    main_risk: str = ""
    cutoffs: dict[str, float] = field(default_factory=dict)
    market_gate: bool = False
    max_stocks: int | None = None
    era: int = 1
    # Registration sequence within the file; assigned by _register. Together
    # with era this pins the chronological display order even if blocks are
    # ever reordered in the source.
    _seq: int = field(default=-1, repr=False, compare=False)


# Build eras, oldest first. Shown as group labels on the Strategies page.
ERA_LABELS = {
    1: "Original technical strategies",
    2: "Fundamental strategies",
    3: "New research strategies",
    4: "Value & quality classics",
}


def cutoff_description(signal_name: str, cutoff: float) -> str:
    """Plain-English meaning of one absolute cutoff filter."""
    if signal_name == "trend":
        return (
            "only stocks trading above their 200-day average price "
            f"(trend score of {cutoff:+.0%} or better)"
        )
    return f"only stocks with {signal_name} of {cutoff:+.4f} or better"


STRATEGIES: dict[str, Strategy] = {}

_SEQ_COUNTER = 0


def _register(s: Strategy) -> None:
    global _SEQ_COUNTER
    if s.weighting not in WEIGHTING_METHODS:
        raise ValueError(f"unknown weighting {s.weighting!r} for {s.name}")
    if s.rebalance not in ("daily", "weekly", "monthly"):
        raise ValueError(f"unknown rebalance {s.rebalance!r} for {s.name}")
    if s.era not in ERA_LABELS:
        raise ValueError(f"unknown era {s.era!r} for {s.name}")
    s._seq = _SEQ_COUNTER
    _SEQ_COUNTER += 1
    STRATEGIES[s.name] = s


_register(
    Strategy(
        name="trending_value",
        era=2,
        formal_name="Trending Value (O'Shaughnessy)",
        description=(
            "Buys stocks that are both cheap for their business numbers and "
            "already climbing. Value tells you *what* to buy; a rising price "
            "tells you the market is starting to agree — which sidesteps "
            "'value traps', cheap stocks that just keep sinking."
        ),
        calculation=(
            "1. Value half: each stock is ranked across all stocks on a value "
            "composite — book-to-market (shareholder equity ÷ market cap), "
            "earnings yield (trailing-12-month net income ÷ market cap) and "
            "cash-flow yield (trailing-12-month operating cash flow ÷ market "
            "cap), using only SEC filings published on or before each date. "
            "The three ranks are averaged into one cheapness percentile.\n"
            "2. Momentum half: each stock's total price return over the "
            "trailing 126 trading days (about 6 months, including the most "
            "recent month — O'Shaughnessy's published spec), ranked across "
            "all stocks.\n"
            "3. Composite score: the average of the two percentile ranks "
            "(averaged over whichever halves a stock actually has).\n"
            "4. Positions are sized by score — stronger scores get more — "
            "capped at 5% per stock (about 20 stocks, a concentrated value "
            "portfolio in line with O'Shaughnessy's published portfolios). "
            "Scores recompute monthly — the academic convention for "
            "fundamental strategies, since weekly rebalancing of slow-moving "
            "value signals just churns; a trade fires only when a position's "
            "target weight drifts more than 2% from its current weight."
        ),
        signals=[("trending_value", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Catches cheap stocks just as they start recovering.",
        main_risk="In a sharp market-wide selloff, even cheap winners fall together.",
    )
)
_register(
    Strategy(
        name="value_momentum",
        era=2,
        formal_name="Value + Momentum Composite (AQR-Style)",
        description=(
            "Ranks every stock on cheapness (book value, earnings and cash "
            "flow versus price) and on 12-month price strength, then holds "
            "the best of both. Cheap stocks and winning stocks tend to take "
            "turns leading the market, so owning both smooths the ride — "
            "the classic institutional 'value + momentum' blend."
        ),
        calculation=(
            "1. Value half: three 'how much company per dollar of price' "
            "ratios, each built from the latest SEC filings known on each "
            "historical date — book-to-market (shareholder equity ÷ market "
            "cap), earnings yield (trailing-12-month net income ÷ market "
            "cap), cash-flow yield (trailing-12-month operating cash flow ÷ "
            "market cap). Quarterly income numbers are rebuilt into "
            "trailing-12-month totals so a 3-month quarter is never ranked "
            "against a 12-month year. Each ratio is ranked across all "
            "stocks; the stock's value score is its average percentile rank. "
            "A stock missing two or more of the three gets no value score.\n"
            "2. Momentum half: the 12-month total return skipping the most "
            "recent month (the same Momentum score described below), ranked "
            "across all stocks.\n"
            "3. Composite score: 50% value rank + 50% momentum rank.\n"
            "4. Positions are sized by score, capped at 5% per stock (about "
            "20 stocks — concentrated enough for the value and momentum "
            "tilts to matter, diversified enough to avoid single-stock "
            "bets). Recomputed monthly (AQR-style composites are published "
            "as monthly-rebalanced); trades fire only past the 2% drift "
            "buffer."
        ),
        signals=[("value_composite", 0.5), ("momentum", 0.5)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Diversified across two return drivers that rarely slump together.",
        main_risk="Momentum can crash hard when markets snap back after a panic.",
    )
)
_register(
    Strategy(
        name="quality_lowvol",
        era=2,
        formal_name="Quality + Low Volatility Composite",
        description=(
            "Buys profitable, financially healthy companies with calm stock "
            "prices — quality businesses that don't swing wildly. The most "
            "defensive of the new strategies: it aims for steadier, "
            "less scary returns rather than the highest possible gains."
        ),
        calculation=(
            "1. Quality half: trailing-12-month gross profit (sales minus the "
            "cost of making them, from SEC filings known on each date) ÷ "
            "total assets (the latest filed snapshot) — how much profit the "
            "business squeezes out of everything it owns. Ranked across all "
            "stocks.\n"
            "2. Calmness half: the 3-month low-volatility score described "
            "below (how jumpy the price was day to day), ranked across all "
            "stocks.\n"
            "3. Composite score: 50% quality rank + 50% calmness rank.\n"
            "4. Positions are sized by score, capped at 5% per stock (about "
            "20 stocks — a defensive, diversified quality portfolio). "
            "Recomputed monthly, the academic convention for fundamental "
            "strategies; trades fire only past the 2% drift buffer."
        ),
        signals=[("profitability", 0.5), ("low_volatility", 0.5)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Holds up better than the market when things get rough.",
        main_risk="Lags in roaring bull markets led by speculative stocks.",
    )
)
_register(
    Strategy(
        name="momentum",
        era=1,
        formal_name="12-1 Month Cross-Sectional Momentum",
        description=(
            "Holds the past year's biggest winners, skipping the most recent "
            "month. The classic momentum approach: stocks on a strong run "
            "tend to keep climbing for a while."
        ),
        calculation=(
            "1. For each stock, take its closing price 252 trading days ago "
            "(about 12 months) and 21 trading days ago (about 1 month). The "
            "score is price(21 days ago) ÷ price(252 days ago) − 1 — the "
            "total return over the past 12 months, skipping the most recent "
            "month. (Skipping avoids very short-term jumps, which often snap "
            "back, contaminating the trend reading.)\n"
            "2. Rank every stock from lowest to highest score; each stock's "
            "percentile rank (0% = worst, 100% = best) becomes its composite "
            "score.\n"
            "3. Hold the top-ranked stocks — up to about 50, implied by the "
            "2% maximum per stock (roughly the top tenth of the S&P 500, "
            "matching academic momentum studies; the old 10-stock version "
            "was an arbitrary concentration, not the published design) — "
            "each at an equal share of the portfolio.\n"
            "4. Scores recompute monthly from data available that day only — "
            "the academic convention (Jegadeesh-Titman momentum is a "
            "monthly-rebalanced strategy); a position trades only when its "
            "target weight drifts more than 2% from its current weight."
        ),
        signals=[("momentum", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.02,
        data_used="Prices only",
        good_at="Rides strong, persistent market trends.",
        main_risk="Suffers sharp reversals when trends suddenly break.",
    )
)
_register(
    Strategy(
        name="trend",
        era=1,
        formal_name="200-Day Moving-Average Trend Filter",
        description=(
            "Holds stocks trading above their 200-day average price — a "
            "simple long-term uptrend filter. If a stock is above where it "
            "has averaged over the past 10 months, the uptrend is intact. "
            "This is an absolute rule, not a relative one: stocks below "
            "their average are excluded outright, so in a broad downturn "
            "the portfolio can sit partly — or even entirely — in cash "
            "rather than holding the least-bad losers."
        ),
        calculation=(
            "1. For each stock, compute the average closing price over the "
            "past 200 trading days (about 10 months).\n"
            "2. Score = (today's close − 200-day average) ÷ 200-day average. "
            "A score of +0.10 means the price sits 10% above its long-run "
            "average; −0.10 means 10% below.\n"
            "3. Absolute filter first: any stock below its 200-day average "
            "(score under 0) is excluded outright — never ranked, never "
            "bought. That slice of the portfolio sits in cash, so in a "
            "broad downturn the portfolio can hold partly or even entirely "
            "cash instead of the least-bad losers.\n"
            "4. The surviving stocks are ranked by score; the top ones — up "
            "to about 25, implied by the 4% cap — are held equally. "
            "Recomputed monthly from data available that day only; trades "
            "fire only past the 2% drift buffer."
        ),
        signals=[("trend", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.04,
        data_used="Prices only",
        good_at="Simple, slow-moving; stays invested through healthy uptrends.",
        main_risk=(
            "Slow to react — gives back gains when trends turn. In a broad "
            "bear market it can sit mostly in cash, missing the eventual "
            "recovery until uptrends re-establish."
        ),
        # Absolute filter, applied before ranking: a stock trading below
        # its 200-day average is not a candidate at all — never ranked,
        # never bought. Without this, percentile ranking would still hold
        # the "least-bad" stocks in a bear market.
        cutoffs={"trend": 0.0},
    )
)
_register(
    Strategy(
        name="low_volatility",
        era=1,
        formal_name="Low Volatility (3-Month)",
        description=(
            "Holds the calmest stocks of the past 3 months. Calmer stocks "
            "have historically delivered steadier returns with smaller "
            "drops than jumpy ones."
        ),
        calculation=(
            "1. For each stock, take its daily returns over the past 63 "
            "trading days (about 3 months) and compute their standard "
            "deviation — how jumpy the price was day to day.\n"
            "2. Score = the negative of that number, so the calmest stocks "
            "score highest (a perfectly flat price scores 0).\n"
            "3. Rank every stock by score; the calmest ones — up to about "
            "25, implied by the 4% cap — are held equally.\n"
            "4. Recomputed monthly from data available that day only; "
            "trades fire only past the 2% drift buffer."
        ),
        signals=[("low_volatility", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.04,
        data_used="Prices only",
        good_at="Smaller drops and a smoother ride than the market.",
        main_risk="Lags when risky, speculative stocks are rallying.",
    )
)


# --- Era 3: research batch (user-approved paper review) --------------------
# Three long-only adaptations of published anomalies, implemented from the
# canonical academic definitions. See qrp/signals/research.py for the
# signal math; the presets below follow each paper's portfolio spec,
# translated into this platform's knobs.

_register(
    Strategy(
        name="continuous_momentum",
        era=3,
        formal_name="Continuous Momentum (Frog-in-the-Pan)",
        description=(
            "Holds past winners that climbed smoothly rather than spiking. "
            "A stock that rose on many small up days was quietly "
            "under-noticed by the market, so its trend tends to persist; "
            "a stock that jumped on a few big spikes tends to snap back. "
            "The frog-in-the-pan idea: gradual heating goes unnoticed."
        ),
        calculation=(
            "1. Formation return: each stock's total price return over the "
            "past 12 months, skipping the most recent month — "
            "close(21 trading days ago) ÷ close(252 trading days ago) − 1, "
            "exactly like the Momentum score.\n"
            "2. Information discreteness: ID = sign(formation return) × "
            "(% of down days − % of up days) over the same window, "
            "counting only the direction of each day (not its size). A "
            "smooth climber has far more up days than down days, so its ID "
            "is near −1; a choppy jumper's ID is near +1.\n"
            "3. Score = −ID for winners (formation return above zero), so "
            "the smoothest winners rank first. Stocks that are not winners "
            "get no score at all and are never bought — in a bear market "
            "with fewer winners than portfolio slots, the unfilled slice "
            "simply sits in cash instead of being padded with losers.\n"
            "4. Hold the top-ranked winners — up to about 50, implied by "
            "the 2% maximum per stock — sized by inverse volatility "
            "(calmer stocks get a bit more), so no shaky name dominates. "
            "Scores recompute monthly from data available that day only; "
            "trades fire only past the 2% drift buffer."
        ),
        signals=[("continuous_momentum", 1.0)],
        weighting="inverse_vol",
        rebalance="monthly",
        max_weight=0.02,
        data_used="Prices only",
        good_at="Avoids the fragile spike-driven winners that crash hardest.",
        main_risk="Still a momentum strategy — suffers when trends snap back.",
    )
)
_register(
    Strategy(
        name="overnight_intraday",
        era=3,
        formal_name="Intraday Institutional Flow (Overnight Divergence)",
        description=(
            "Buys stocks with steady buying during the trading day that the "
            "overnight crowd hasn't chased yet. Big institutions tend to "
            "buy patiently through the day, while overnight moves are "
            "driven by headlines and retail orders — daytime strength "
            "without overnight hype hints at quiet institutional "
            "accumulation."
        ),
        calculation=(
            "1. Split each day's return in two: the overnight leg "
            "(previous close → today's open) and the intraday leg "
            "(open → close). On dividend ex-dates the dividend is added "
            "back to the overnight leg, so the payout doesn't read as a "
            "fake overnight drop.\n"
            "2. Compound each leg over the past 126 trading days (about 6 "
            "months), skipping the most recent 5 sessions to dodge "
            "short-term noise.\n"
            "3. Each day, convert both compounded legs to Z-scores across "
            "all stocks (how many standard deviations above or below the "
            "average stock). Score = intraday Z − overnight Z.\n"
            "4. Hold the top ~50 scorers (2% cap each), split evenly. "
            "Recomputed monthly from data available that day only; trades "
            "fire only past the 2% drift buffer. (The paper's "
            "market-cap weighting isn't a knob this platform offers, so "
            "equal weight is used instead — documented, not hidden.)"
        ),
        signals=[("overnight_intraday", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.02,
        data_used="Prices + dividends",
        good_at="A genuinely different return driver — barely related to value or momentum.",
        main_risk="An exotic, less-studied signal; the edge could be weaker than the paper suggests.",
    )
)
_register(
    Strategy(
        name="twin_momentum",
        era=3,
        formal_name="Twin Momentum (Price + Earnings Surprise)",
        description=(
            "Buys stocks where the price trend AND the earnings surprise "
            "both point up. A rising price backed by a real earnings beat "
            "is much sturdier than a price spike with nothing behind it — "
            "stocks failing either test are skipped entirely, which "
            "filters out the speculative runs that crash hardest."
        ),
        calculation=(
            "1. Price leg: the 12-month return skipping the most recent "
            "month (the standard Momentum score), converted to a Z-score "
            "across all stocks each day.\n"
            "2. Earnings leg: standardized unexpected diluted EPS (SUE) "
            "from 10-Q filings. For the latest reported quarter, surprise "
            "= EPS this quarter − EPS the same quarter last year; SUE = "
            "(surprise − average surprise) ÷ spread of surprises, using "
            "the stock's own last 8 quarterly surprises (at least 8 "
            "quarters of 10-Q history required). Only filings published on "
            "or before each date are used. Converted to a Z-score across "
            "stocks.\n"
            "3. Both legs must sit above their cross-sectional average "
            "that day — a stock failing either is disqualified outright "
            "and never bought. Qualifiers are ranked by the average of "
            "the two Z-scores.\n"
            "4. Hold the top ~40 qualifiers (2.5% cap each), split evenly. "
            "Recomputed monthly from data available that day only; trades "
            "fire only past the 2% drift buffer."
        ),
        signals=[("twin_momentum", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.025,
        data_used="Prices + company financials",
        good_at="Catches earnings-backed trends while dodging hollow price spikes.",
        main_risk="Needs 8 quarters of 10-Q history, so young listings never qualify.",
    )
)


# ---------------------------------------------------------------------------
# Era 4 — value & quality classics (backlog #2, built 2026-10-11).
# ---------------------------------------------------------------------------

_register(
    Strategy(
        name="magic_formula",
        era=4,
        formal_name="Magic Formula (Greenblatt)",
        description=(
            "Joel Greenblatt's famous two-part screen: it buys stocks that "
            "are both cheap for their operating earnings and genuinely good "
            "businesses earning high returns on their capital. Cheap alone "
            "can be a trap and quality alone can be overpriced — the magic "
            "is demanding both at once. Banks and utilities are excluded, "
            "since the math doesn't apply to them."
        ),
        calculation=(
            "1. Earnings yield: trailing-12-month operating earnings (EBIT) "
            "divided by enterprise value, where enterprise value = market "
            "cap + long-term debt minus cash (never below market cap alone). "
            "Only SEC filings published on or before each date are used.\n"
            "2. Return on capital: TTM EBIT divided by invested capital, "
            "where invested capital = shareholder equity + long-term debt "
            "minus cash (Greenblatt's net-working-capital + net-fixed-assets, "
            "expressed with the balance-sheet tags the platform ingests).\n"
            "3. Each measure is ranked across all stocks (higher = better); "
            "the score is the average of the two percentile ranks.\n"
            "4. Financials and Utilities are excluded before ranking "
            "(enterprise value is meaningless for a bank), using the GICS "
            "sector table when it has been refreshed — a stock whose "
            "sector is unknown is kept rather than guessed about.\n"
            "5. Hold the top ~20 by score, sized by score and capped at 5% "
            "per stock. Recomputed monthly from data available that day "
            "only; trades fire only past the 2% drift buffer."
        ),
        signals=[("magic_formula", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Finds cheap stocks that are also genuinely good businesses.",
        main_risk="Needs full balance-sheet history (debt, cash, equity) — young or thinly-filed listings never qualify.",
    )
)
_register(
    Strategy(
        name="piotroski",
        era=4,
        formal_name="Piotroski F-Score",
        description=(
            "Nine yes/no health checks on the financial statements — is it "
            "profitable, is the profit real cash, is debt shrinking, is it "
            "getting more efficient? Each yes is a point, out of nine, and "
            "the strategy buys the highest scorers. Designed to separate "
            "real bargains from value traps: among cheap stocks, the "
            "financially strong keep winning and the weak keep losing."
        ),
        calculation=(
            "1. Nine tests, one point each: (a) return on assets > 0; "
            "(b) operating cash flow > 0; (c) return on assets higher than "
            "a year ago; (d) cash flow exceeds net income (the accruals "
            "check — earnings backed by cash, not accounting); (e) "
            "long-term-debt-to-assets lower than a year ago; (f) current "
            "ratio higher than a year ago; (g) shares outstanding not "
            "increased over the year; (h) gross margin higher than a year "
            "ago; (i) asset turnover higher than a year ago. 'A year ago' "
            "is the latest-known filing 252 trading days earlier — every "
            "comparison is point-in-time, using only filings published on "
            "or before each date.\n"
            "2. A stock needs at least 5 answerable tests to be scored at "
            "all; otherwise it is skipped.\n"
            "3. Hold the top ~20 by F-score, split evenly (5% cap each). "
            "Recomputed monthly; trades fire only past the 2% drift "
            "buffer.\n"
            "Note: Piotroski's original paper bought only 8-9 scores among "
            "the cheapest stocks. This version ranks the full universe by "
            "F-score so it always holds a diversified set — the Compare "
            "page can show both side by side."
        ),
        signals=[("piotroski", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Dodges value traps — financially shaky 'bargains' never make the cut.",
        main_risk="Financial-statement signals move slowly; in a fast momentum-driven market it can look asleep.",
    )
)
_register(
    Strategy(
        name="qmj_lite",
        era=4,
        formal_name="Quality-Minus-Junk Lite",
        description=(
            "A quality screen in the spirit of AQR's research: it favors "
            "companies that are profitable, getting more profitable, and "
            "safe — low debt, calm share price. Quality businesses tend to "
            "keep outperforming because investors systematically underpay "
            "for boring, reliable earners."
        ),
        calculation=(
            "1. Profitability leg: gross profits divided by assets, return "
            "on equity, return on assets, operating cash flow divided by "
            "assets, and low accruals (accounting profit far above cash "
            "profit is a red flag) — each ranked across stocks, higher = "
            "better.\n"
            "2. Growth leg: the one-year change in gross profitability and "
            "in return on equity, ranked the same way.\n"
            "3. Safety leg: low trailing one-year price volatility and low "
            "leverage (debt divided by equity) rank highest.\n"
            "4. The score is the average percentile rank over whichever "
            "components a stock actually has. All inputs use only SEC "
            "filings published on or before each date (flows as trailing "
            "12 months).\n"
            "5. Hold the top ~20 by score, sized by score and capped at 5% "
            "per stock. Recomputed monthly; trades fire only past the 2% "
            "drift buffer."
        ),
        signals=[("qmj_lite", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Steady compounders — it leans into businesses that earn reliably.",
        main_risk="Quality is often fully priced; in speculative rallies it lags the junk it avoids.",
    )
)
_register(
    Strategy(
        name="shareholder_yield",
        era=4,
        formal_name="Shareholder Yield",
        description=(
            "How much cash the company hands back to its owners — dividends "
            "plus share buybacks — compared to its price. A high yield pays "
            "you just for holding, and steady buybacks usually signal a "
            "disciplined management team that doesn't waste cash on "
            "empire-building."
        ),
        calculation=(
            "1. Dividend leg: trailing-12-month dividends paid divided by "
            "market cap, from SEC filings (only filings published on or "
            "before each date).\n"
            "2. Buyback leg: the shrink in shares outstanding over the "
            "trailing year times current price, divided by market cap, "
            "floored at zero (share issuance is not counted as negative "
            "yield here — the Piotroski dilution test covers that "
            "separately).\n"
            "3. Shareholder yield is the sum of the two legs, ranked across "
            "stocks (higher = better).\n"
            "4. Hold the top ~20 by yield, sized by score and capped at 5% "
            "per stock. Recomputed monthly; trades fire only past the 2% "
            "drift buffer."
        ),
        signals=[("shareholder_yield", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Pays you to wait — high cash-return businesses with disciplined management.",
        main_risk="A high yield can be a distress signal (price collapsed); buybacks stop exactly when you'd want them most.",
    )
)
_register(
    Strategy(
        name="reversal_long",
        era=4,
        formal_name="Long-Horizon Reversal",
        description=(
            "It buys yesterday's forgotten stocks: the worst performers "
            "over the past five years, ignoring the most recent year. "
            "Investors overreact — they abandon long-run losers beyond what "
            "the business deserves — and those stocks tend to bounce back. "
            "The natural complement to momentum, which likes recent winners."
        ),
        calculation=(
            "1. For each stock: the 5-year price return ending one year ago "
            "(price 1 year ago divided by price 5 years ago, minus 1), "
            "ranked across stocks with the worst performers scoring "
            "highest.\n"
            "2. Skipping the most recent year is deliberate: it keeps this "
            "distinct from ordinary momentum (which favors recent winners) "
            "so the two can sit side by side.\n"
            "3. Needs 5 years of price history — young listings never "
            "qualify.\n"
            "4. Hold the top ~20 reversal candidates, sized by score and "
            "capped at 5% per stock. Recomputed monthly; trades fire only "
            "past the 2% drift buffer."
        ),
        signals=[("reversal_long", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.05,
        data_used="Prices only",
        good_at="Catches hated stocks as they turn — the mirror image of momentum.",
        main_risk="Some 'losers' are losers for good reason (dying businesses); catching the falling knife is the occupational hazard.",
    )
)


def get_strategy(name: str) -> Strategy:
    """Return the strategy registered under name; KeyError if unknown."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy: {name!r}") from None


def list_strategies() -> list[tuple[str, str]]:
    """(name, description) for every registered strategy, chronological.

    Ordered by build era (original technicals, then fundamental blends,
    then research batch), oldest first — every picker, page and table in
    the UI shares this one sequence.
    """
    ordered = sorted(STRATEGIES.values(), key=lambda s: (s.era, s._seq))
    return [(s.name, s.description) for s in ordered]


def era_label(name: str) -> str:
    """Build-era group label for a strategy, e.g. 'Fundamental strategies'."""
    s = STRATEGIES.get(name)
    if s is None:
        return ""
    return ERA_LABELS.get(s.era, "")


def display_name(name: str) -> str:
    """Formal user-facing name for a registered strategy.

    Returns the strategy's formal name (e.g. "12-1 Month Cross-Sectional
    Momentum"). For names that are not registered strategies (e.g. a raw
    signal name reused as a label), falls back to the title-cased key so
    callers never have to special-case.
    """
    s = STRATEGIES.get(name)
    if s is not None and s.formal_name:
        return s.formal_name
    return name.replace("_", " ").title()


def strategy_signal_matrices(
    strategy: Strategy,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> dict[str, pd.DataFrame]:
    """Load every signal matrix a strategy needs (point-in-time).

    Keys are signal names; for "inverse_vol" weighting an extra "__vol__"
    key holds positive volatilities (the stored low_volatility signal is
    the negative of volatility, so it is negated here).
    """
    matrices: dict[str, pd.DataFrame] = {}
    for sig_name, _ in strategy.signals:
        m = load_signal_matrix(sig_name, tickers, start=start, end=end, path=db_path)
        if m.empty:
            raise ValueError(
                f"No stored values for signal {sig_name!r} in range — "
                "compute signals first (Signals page)."
            )
        matrices[sig_name] = m
    if strategy.weighting == "inverse_vol":
        low_vol = load_signal_matrix(
            "low_volatility", tickers, start=start, end=end, path=db_path
        )
        if low_vol.empty:
            raise ValueError(
                "Inverse-vol weighting needs low_volatility values — "
                "compute signals first (Signals page)."
            )
        matrices["__vol__"] = -low_vol
    return matrices


def composite_scores(
    strategy: Strategy,
    matrices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    excluded: dict[str, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Blend a strategy's signals into one score per ticker.

    Each signal is converted to cross-sectional percentile ranks (0..1,
    higher = more attractive — every signal in this platform follows that
    convention), then the weighted average is taken. A ticker missing a
    signal is simply averaged over the signals it has; a ticker with no
    signal values at all is excluded. Only values dated on or before
    `date` are used — never from the future.

    Absolute cutoffs (strategy.cutoffs) are applied BEFORE ranking: a
    ticker whose raw signal value is below the cutoff is not a candidate
    at all — it is never ranked and never bought, so its weight sits in
    cash. Pass a dict as `excluded` to also get {signal_name: (n_failed,
    n_total)} counts of what each cutoff filtered out.
    """
    ranked: list[tuple[pd.Series, float]] = []
    for sig_name, weight in strategy.signals:
        history = matrices[sig_name].loc[:date].ffill()
        if history.empty:
            continue
        latest = history.iloc[-1].dropna()
        if latest.empty:
            continue
        cutoff = strategy.cutoffs.get(sig_name)
        if cutoff is not None:
            total = int(len(latest))
            latest = latest[latest >= cutoff]
            if excluded is not None:
                excluded[sig_name] = (total - int(len(latest)), total)
        if latest.empty:
            continue
        ranked.append((latest.rank(pct=True), weight))
    if not ranked:
        return {}
    scores: dict[str, float] = {}
    tickers: set[str] = set()
    for pct, _ in ranked:
        tickers.update(pct.index)
    for ticker in tickers:
        num = sum(float(pct[ticker]) * w for pct, w in ranked if ticker in pct.index)
        den = sum(w for pct, w in ranked if ticker in pct.index)
        scores[ticker] = num / den
    return scores


def _volatilities_asof(
    matrices: dict[str, pd.DataFrame], date: pd.Timestamp
) -> dict[str, float] | None:
    if "__vol__" not in matrices:
        return None
    hist = matrices["__vol__"].loc[:date].ffill()
    if hist.empty:
        return None
    return hist.iloc[-1].dropna().to_dict()


def strategy_target_weights(
    strategy: Strategy,
    matrices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    max_weight: float | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[dict[str, float], dict[str, float]]:
    """Target weights + composite scores for a strategy at one date.

    Returns (weights, scores). Pure function of point-in-time data —
    shared by the backtest, the dashboard and the paper trader.
    """
    portfolio, scores = strategy_target_portfolio(
        strategy, matrices, date, max_weight=max_weight, db_path=db_path
    )
    return portfolio.weights, scores


def strategy_target_portfolio(
    strategy: Strategy,
    matrices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    max_weight: float | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[TargetPortfolio, dict[str, float]]:
    """Full target portfolio (+ composite scores) for a strategy at one date.

    Returns (TargetPortfolio, scores). The TargetPortfolio carries the
    safety-rail notes (caps applied, dust dropped) for display.
    """
    excluded: dict[str, tuple[int, int]] = {}
    scores = composite_scores(strategy, matrices, date, excluded=excluded)
    method = WEIGHTING_METHODS[strategy.weighting]
    mw = strategy.max_weight if max_weight is None else max_weight
    portfolio = build_target_weights(
        scores,
        method=method,
        max_weight=mw,
        # One knob by default: the stock count is implied by the
        # max-weight cap (n = round(1 / max_weight)), not a separate
        # control. Variants may set max_stocks explicitly.
        max_positions=(
            strategy.max_stocks
            if strategy.max_stocks is not None
            else implied_stock_count(mw)
        ),
        volatilities=_volatilities_asof(matrices, date)
        if method == INVERSE_VOLATILITY
        else None,
    )
    for sig_name, (n_failed, n_total) in excluded.items():
        if n_failed:
            portfolio.notes.append(
                f"Cutoff filter: {n_failed} of {n_total} stocks were below "
                f"the bar ({cutoff_description(sig_name, strategy.cutoffs[sig_name])}) "
                f"and were left out — that slice of the portfolio sits in cash."
            )
    if strategy.market_gate:
        # The dual-momentum overlay: on a risk-off date the built weights
        # are scaled to zero (100% cash). Safety wins over the drift
        # buffer — de-risking is never blocked by it.
        from qrp.portfolio.market_gate import apply_gate

        gated_weights, gate_note = apply_gate(portfolio.weights, date, db_path)
        portfolio.notes.append(gate_note)
        portfolio = TargetPortfolio(
            weights=gated_weights,
            cash_weight=1.0 - sum(gated_weights.values()),
            method=portfolio.method,
            notes=portfolio.notes,
        )
    return portfolio, scores


def make_strategy_fn(
    strategy: Strategy,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
    matrices: dict[str, pd.DataFrame] | None = None,
    universe_filter: Callable[[pd.Timestamp], set[str]] | None = None,
) -> StrategyFn:
    """Build a backtest-engine StrategyFn from a registered strategy.

    Signal matrices are loaded once up front (they are point-in-time, so
    slicing them at each rebalance date inside the returned function stays
    lookahead-free). The engine's own `signals` argument is ignored — the
    strategy carries its own data. Pass `matrices` (e.g. from a cache) to
    skip the load; otherwise they are loaded here. `universe_filter`, when
    given, maps each rebalance date to the eligible tickers (point-in-time
    index membership); matrices are sliced to eligible columns before
    scoring, so the strategy can never rank or buy a future member.
    """
    if matrices is None:
        matrices = strategy_signal_matrices(strategy, tickers, start, end, db_path)
    # Forward-fill ONCE up front (audit 2.6 / 1.2): composite_scores slices
    # `.loc[:date]` per rebalance date and calls `.ffill()` again, but ffill
    # is idempotent, so pre-filling here turns every per-date full-matrix
    # fill into a no-op. Results are unchanged to the last bit.
    matrices = {k: v.ffill() for k, v in matrices.items()}

    def fn(_signals: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict[str, float]:
        use_matrices = matrices
        if universe_filter is not None:
            allowed = universe_filter(rebalance_date)
            use_matrices = {
                k: v[[c for c in v.columns if c in allowed]]
                for k, v in matrices.items()
            }
        weights, _ = strategy_target_weights(
            strategy, use_matrices, rebalance_date, db_path=db_path
        )
        return weights

    fn.strategy_name = strategy.name  # type: ignore[attr-defined]
    return fn


__all__ = [
    "Strategy",
    "STRATEGIES",
    "ERA_LABELS",
    "cutoff_description",
    "get_strategy",
    "list_strategies",
    "era_label",
    "display_name",
    "strategy_signal_matrices",
    "composite_scores",
    "strategy_target_weights",
    "strategy_target_portfolio",
    "make_strategy_fn",
]
