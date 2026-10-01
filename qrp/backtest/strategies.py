"""Ready-made long-only strategies built on the signal framework (Phase 3).

A strategy is a function ``(signal_matrix, rebalance_date) -> {ticker: weight}``
with weights >= 0 summing to <= 1. The engine calls it on each rebalance date
and executes the resulting trades at the next trading day's close.
"""

from __future__ import annotations

import pandas as pd

from qrp.backtest.engine import StrategyFn


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
