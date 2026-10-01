"""Ready-made long-only strategies built on the signal framework (Phase 3).

A strategy is a function ``(signal_matrix, rebalance_date) -> {ticker: weight}``
with weights >= 0 summing to <= 1. The engine calls it on each rebalance date
and executes the resulting trades at the next trading day's close.

Phase 4 adds weighted_top_n, which sizes positions with the portfolio
construction methods (equal / signal / inverse-volatility weighting) instead
of plain equal weights, and enforces the max_weight cap.
"""

from __future__ import annotations

import pandas as pd

from qrp.backtest.engine import StrategyFn
from qrp.portfolio.construction import (
    EQUAL_WEIGHT,
    INVERSE_VOLATILITY,
    SIGNAL_WEIGHT,
    build_target_weights,
)


def top_n_by_signal(signal_name: str, n: int) -> StrategyFn:
    """Hold the top-N tickers by signal score, equally weighted.

    On each rebalance date the strategy looks at every ticker's latest
    signal value dated on or before that date (never from the future),
    ranks tickers from highest score to lowest, and gives each of the top N
    an equal share of the portfolio (1/N each). Tickers with no signal value
    yet are skipped; if fewer than N qualify, the portfolio simply holds
    fewer stocks; if none qualify, it stays in cash.

    Ties are broken alphabetically so results are deterministic.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    def strategy(
        signal_matrix: pd.DataFrame, rebalance_date: pd.Timestamp
    ) -> dict[str, float]:
        # Latest available value on or before the rebalance date, per ticker.
        # Forward-filling means a ticker whose signal has a gap still uses its
        # most recent known value; tickers never scored are excluded.
        history = signal_matrix.loc[:rebalance_date].ffill()
        if history.empty:
            return {}
        latest = history.iloc[-1].dropna()
        if latest.empty:
            return {}
        # Deterministic ranking: score desc, ticker asc for ties.
        ranked = sorted(latest.index, key=lambda t: (-latest[t], t))
        picks = ranked[:n]
        weight = 1.0 / len(picks)
        return {ticker: weight for ticker in picks}

    strategy.signal_name = signal_name  # type: ignore[attr-defined]
    strategy.top_n = n  # type: ignore[attr-defined]
    return strategy


# Short CLI names ("equal", "signal", "inverse_vol") -> construction methods.
WEIGHTING_METHODS = {
    "equal": EQUAL_WEIGHT,
    "signal": SIGNAL_WEIGHT,
    "inverse_vol": INVERSE_VOLATILITY,
}


def weighted_top_n(
    signal_name: str,
    n: int,
    method: str = EQUAL_WEIGHT,
    max_weight: float = 0.20,
    vol_matrix: pd.DataFrame | None = None,
) -> StrategyFn:
    """Rank top-N by signal, then size positions with a weighting method.

    This is top_n_by_signal grown up: instead of always splitting money
    evenly, it delegates the sizing to qrp.portfolio.construction, so the
    max_weight cap, max_positions cap, and dust filter all apply.

    Args:
        signal_name: which signal to rank stocks by.
        n: consider at most the n highest-scoring stocks.
        method: "equal_weight", "signal_weight", or "inverse_volatility"
            (see qrp.portfolio.construction.METHOD_DESCRIPTIONS).
        max_weight: no single stock may exceed this fraction of capital.
        vol_matrix: required for "inverse_volatility" — a tickers-x-dates
            matrix of recent price volatilities as positive numbers,
            point-in-time like the signal matrix (e.g. the stored
            low_volatility signal values, negated). Each rebalance date
            uses values dated on or before that date: no lookahead.

    Ties are broken alphabetically so results are deterministic.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    def strategy(
        signal_matrix: pd.DataFrame, rebalance_date: pd.Timestamp
    ) -> dict[str, float]:
        history = signal_matrix.loc[:rebalance_date].ffill()
        if history.empty:
            return {}
        latest = history.iloc[-1].dropna()
        if latest.empty:
            return {}
        scores = latest.to_dict()

        vols: dict[str, float] | None = None
        if method == INVERSE_VOLATILITY:
            if vol_matrix is None:
                raise ValueError(
                    "inverse_volatility needs vol_matrix "
                    "(pass the low_volatility signal matrix, negated)"
                )
            v_history = vol_matrix.loc[:rebalance_date].ffill()
            if not v_history.empty:
                vols = v_history.iloc[-1].dropna().to_dict()

        portfolio = build_target_weights(
            scores,
            method=method,
            max_weight=max_weight,
            max_positions=n,
            volatilities=vols,
        )
        return portfolio.weights

    strategy.signal_name = signal_name  # type: ignore[attr-defined]
    strategy.top_n = n  # type: ignore[attr-defined]
    strategy.weighting_method = method  # type: ignore[attr-defined]
    return strategy
