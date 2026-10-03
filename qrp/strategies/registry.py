"""Named investment strategies (Phase 8).

A **strategy** is a complete, named recipe: which signal(s) rank the
stocks, how positions are sized, how often the portfolio is rebalanced,
and how much drift is tolerated before trading. Strategies are the unit
the user picks in the UI — one dropdown on the Backtest, Portfolio and
Paper-trading pages — so a new idea only needs a registry entry to become
backtestable, explainable, and comparable everywhere.

The three new institutional strategies each blend fundamentals (company
numbers from SEC filings) with technicals (price patterns), per the
research in the "Institutional Strategy Recommendations" report:
  - trending_value: cheap AND climbing (value rank + 6-month momentum rank)
  - value_momentum: cheapness + 12-month momentum, averaged
  - quality_lowvol: profitability + calm prices, averaged
The three original single-signal approaches sit beside them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    rebalance: "weekly" or "monthly".
    max_weight: no single stock may exceed this fraction of capital. This is
        the single position-sizing knob: it also implies how many stocks are
        held, n = round(1 / max_weight) (e.g. 5% -> ~20 stocks).
    drift_tolerance: skip trading a position whose target weight is within
        this of its current weight (avoids churn on tiny moves).
    data_used: plain-English, e.g. "Prices only" or "Prices + company
        financials". Shown on the Strategies comparison page.
    good_at: one plain-English line on what the strategy is good at.
    main_risk: one plain-English line on its main risk.
    """

    name: str
    description: str
    signals: list[tuple[str, float]] = field(default_factory=list)
    weighting: str = "equal"
    rebalance: str = "monthly"
    max_weight: float = 0.10
    drift_tolerance: float = 0.02
    data_used: str = "Prices only"
    good_at: str = ""
    main_risk: str = ""


STRATEGIES: dict[str, Strategy] = {}


def _register(s: Strategy) -> None:
    if s.weighting not in WEIGHTING_METHODS:
        raise ValueError(f"unknown weighting {s.weighting!r} for {s.name}")
    if s.rebalance not in ("weekly", "monthly"):
        raise ValueError(f"unknown rebalance {s.rebalance!r} for {s.name}")
    STRATEGIES[s.name] = s


_register(
    Strategy(
        name="trending_value",
        description=(
            "Buys stocks that are both cheap for their business numbers and "
            "already climbing. Value tells you *what* to buy; a rising price "
            "tells you the market is starting to agree — which sidesteps "
            "'value traps', cheap stocks that just keep sinking."
        ),
        signals=[("trending_value", 1.0)],
        weighting="signal",
        rebalance="weekly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Catches cheap stocks just as they start recovering.",
        main_risk="In a sharp market-wide selloff, even cheap winners fall together.",
    )
)
_register(
    Strategy(
        name="value_momentum",
        description=(
            "Ranks every stock on cheapness (book value, earnings and cash "
            "flow versus price) and on 12-month price strength, then holds "
            "the best of both. Cheap stocks and winning stocks tend to take "
            "turns leading the market, so owning both smooths the ride — "
            "the classic institutional 'value + momentum' blend."
        ),
        signals=[("value_composite", 0.5), ("momentum", 0.5)],
        weighting="signal",
        rebalance="weekly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Diversified across two return drivers that rarely slump together.",
        main_risk="Momentum can crash hard when markets snap back after a panic.",
    )
)
_register(
    Strategy(
        name="quality_lowvol",
        description=(
            "Buys profitable, financially healthy companies with calm stock "
            "prices — quality businesses that don't swing wildly. The most "
            "defensive of the new strategies: it aims for steadier, "
            "less scary returns rather than the highest possible gains."
        ),
        signals=[("profitability", 0.5), ("low_volatility", 0.5)],
        weighting="signal",
        rebalance="weekly",
        max_weight=0.05,
        data_used="Prices + company financials",
        good_at="Holds up better than the market when things get rough.",
        main_risk="Lags in roaring bull markets led by speculative stocks.",
    )
)
_register(
    Strategy(
        name="momentum",
        description=(
            "Holds the past year's biggest winners, skipping the most recent "
            "month. The classic momentum approach: stocks on a strong run "
            "tend to keep climbing for a while."
        ),
        signals=[("momentum", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.10,
        data_used="Prices only",
        good_at="Rides strong, persistent market trends.",
        main_risk="Suffers sharp reversals when trends suddenly break.",
    )
)
_register(
    Strategy(
        name="trend",
        description=(
            "Holds stocks trading above their 200-day average price — a "
            "simple long-term uptrend filter. If a stock is above where it "
            "has averaged over the past 10 months, the uptrend is intact."
        ),
        signals=[("trend", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.10,
        data_used="Prices only",
        good_at="Simple, slow-moving; stays invested through healthy uptrends.",
        main_risk="Slow to react — gives back gains when trends turn.",
    )
)
_register(
    Strategy(
        name="low_volatility",
        description=(
            "Holds the calmest stocks of the past 3 months. Calmer stocks "
            "have historically delivered steadier returns with smaller "
            "drops than jumpy ones."
        ),
        signals=[("low_volatility", 1.0)],
        weighting="equal",
        rebalance="monthly",
        max_weight=0.10,
        data_used="Prices only",
        good_at="Smaller drops and a smoother ride than the market.",
        main_risk="Lags when risky, speculative stocks are rallying.",
    )
)


def get_strategy(name: str) -> Strategy:
    """Return the strategy registered under name; KeyError if unknown."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy: {name!r}") from None


def list_strategies() -> list[tuple[str, str]]:
    """Return (name, description) for every registered strategy."""
    return [(name, s.description) for name, s in STRATEGIES.items()]


def display_name(name: str) -> str:
    """'trending_value' -> 'Trending Value' for dropdowns and headings."""
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
) -> dict[str, float]:
    """Blend a strategy's signals into one score per ticker.

    Each signal is converted to cross-sectional percentile ranks (0..1,
    higher = more attractive — every signal in this platform follows that
    convention), then the weighted average is taken. A ticker missing a
    signal is simply averaged over the signals it has; a ticker with no
    signal values at all is excluded. Only values dated on or before
    `date` are used — never from the future.
    """
    ranked: list[tuple[pd.Series, float]] = []
    for sig_name, weight in strategy.signals:
        history = matrices[sig_name].loc[:date].ffill()
        if history.empty:
            continue
        latest = history.iloc[-1].dropna()
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
) -> tuple[dict[str, float], dict[str, float]]:
    """Target weights + composite scores for a strategy at one date.

    Returns (weights, scores). Pure function of point-in-time data —
    shared by the backtest, the dashboard and the paper trader.
    """
    portfolio, scores = strategy_target_portfolio(
        strategy, matrices, date, max_weight=max_weight
    )
    return portfolio.weights, scores


def strategy_target_portfolio(
    strategy: Strategy,
    matrices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    max_weight: float | None = None,
) -> tuple[TargetPortfolio, dict[str, float]]:
    """Full target portfolio (+ composite scores) for a strategy at one date.

    Returns (TargetPortfolio, scores). The TargetPortfolio carries the
    safety-rail notes (caps applied, dust dropped) for display.
    """
    scores = composite_scores(strategy, matrices, date)
    method = WEIGHTING_METHODS[strategy.weighting]
    mw = strategy.max_weight if max_weight is None else max_weight
    portfolio = build_target_weights(
        scores,
        method=method,
        max_weight=mw,
        # One knob: the stock count is implied by the max-weight cap
        # (n = round(1 / max_weight)), not a separate control.
        max_positions=implied_stock_count(mw),
        volatilities=_volatilities_asof(matrices, date)
        if method == INVERSE_VOLATILITY
        else None,
    )
    return portfolio, scores


def make_strategy_fn(
    strategy: Strategy,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
    matrices: dict[str, pd.DataFrame] | None = None,
) -> StrategyFn:
    """Build a backtest-engine StrategyFn from a registered strategy.

    Signal matrices are loaded once up front (they are point-in-time, so
    slicing them at each rebalance date inside the returned function stays
    lookahead-free). The engine's own `signals` argument is ignored — the
    strategy carries its own data. Pass `matrices` (e.g. from a cache) to
    skip the load; otherwise they are loaded here.
    """
    if matrices is None:
        matrices = strategy_signal_matrices(strategy, tickers, start, end, db_path)

    def fn(_signals: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict[str, float]:
        weights, _ = strategy_target_weights(strategy, matrices, rebalance_date)
        return weights

    fn.strategy_name = strategy.name  # type: ignore[attr-defined]
    return fn


__all__ = [
    "Strategy",
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
    "display_name",
    "strategy_signal_matrices",
    "composite_scores",
    "strategy_target_weights",
    "strategy_target_portfolio",
    "make_strategy_fn",
]
