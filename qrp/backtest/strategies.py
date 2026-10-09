"""Ready-made long-only strategies built on the signal framework (Phase 3).

A strategy is a function ``(signal_matrix, rebalance_date) -> {ticker: weight}``
with weights >= 0 summing to <= 1. The engine calls it on each rebalance date
and executes the resulting trades at the next trading day's close.

Phase 4 adds weighted_top_n, which sizes positions with the portfolio
construction methods (equal / signal / inverse-volatility weighting) instead
of plain equal weights, and enforces the max_weight cap.

The backtest page's "Optimized (mean-variance)" sizing adds
optimized_strategy, which re-solves the Phase P3 mean-variance optimizer
at every rebalance date (point-in-time: only data dated on or before that
date is used).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qrp.backtest.engine import StrategyFn
from qrp.data.store import DEFAULT_DB
from qrp.portfolio.construction import (
    EQUAL_WEIGHT,
    INVERSE_VOLATILITY,
    SIGNAL_WEIGHT,
    build_target_weights,
)
from qrp.portfolio.covariance import CovarianceModel
from qrp.portfolio.optimizer import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_AVERSION,
    DEFAULT_TRACKING_ERROR,
    DEFAULT_TURNOVER_PENALTY_BPS,
    optimize_weights,
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
    max_weight: float = 0.05,
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


def optimized_strategy(
    signal_name: str,
    *,
    db_path: str | Path = DEFAULT_DB,
    optimizer_settings: dict | None = None,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    covariance_model: CovarianceModel | None = None,
) -> StrategyFn:
    """Re-solve the mean-variance optimizer (Phase P3) at each rebalance.

    On every rebalance date this takes the latest signal scores (each
    ticker's most recent value dated on or before that date — never from
    the future, same as weighted_top_n) and hands them to
    qrp.portfolio.optimizer.optimize_weights with ``asof`` set to the
    rebalance date. Scores, the risk model, and the benchmark therefore
    all use only data knowable on or before the date: no peeking.

    Args:
        signal_name: which signal's scores feed the optimizer.
        db_path: SQLite market database the optimizer reads prices and
            company data from (always through its ``asof`` filter).
        optimizer_settings: overrides for "risk_aversion",
            "turnover_penalty_bps", "tracking_error", and "max_weight";
            anything omitted falls back to the optimizer's defaults.
            (The dashboard merges its slider defaults in before calling.)
        max_weight: per-stock cap — used when optimizer_settings does
            not carry one.
        covariance_model: the optimizer's risk model. None lets
            optimize_weights use its own default (Ledoit-Wolf shrinkage
            from the market DB); the dashboard builds one from the user's
            Risk model choice and passes it in.

    Two documented approximations, both forced by the strategy contract
    (a strategy function sees signal scores, never the engine's actual
    holdings):

    - The turnover penalty is fed the PREVIOUS TARGET weights returned
      by this function as a stand-in for current holdings. Real holdings
      drift with prices between rebalances, so the penalty is measured
      against a slightly stale picture — close, but not exact.
    - When the optimizer cannot solve yet (early rebalance dates with
      too little price history, etc. — it raises ValueError with a
      plain-English reason), the strategy returns {}: stay in cash,
      mirroring top_n_by_signal's warm-up behavior. The stand-in
      holdings reset to cash too, so the next successful solve measures
      turnover from cash.
    """
    settings = optimizer_settings or {}
    cap = float(settings.get("max_weight", max_weight))
    state: dict[str, dict[str, float] | None] = {"prev": None}

    def strategy(
        signal_matrix: pd.DataFrame, rebalance_date: pd.Timestamp
    ) -> dict[str, float]:
        history = signal_matrix.loc[:rebalance_date].ffill()
        if history.empty:
            return {}
        latest = history.iloc[-1].dropna()
        if latest.empty:
            return {}
        scores = {t: float(v) for t, v in latest.items()}
        try:
            portfolio, _stats = optimize_weights(
                scores,
                asof=rebalance_date.strftime("%Y-%m-%d"),
                tickers=list(scores),
                db_path=db_path,
                w_prev=state["prev"],
                risk_aversion=settings.get(
                    "risk_aversion", DEFAULT_RISK_AVERSION
                ),
                turnover_penalty_bps=settings.get(
                    "turnover_penalty_bps", DEFAULT_TURNOVER_PENALTY_BPS
                ),
                tracking_error=settings.get(
                    "tracking_error", DEFAULT_TRACKING_ERROR
                ),
                max_weight=cap,
                covariance_model=covariance_model,
            )
        except ValueError:
            state["prev"] = None
            return {}
        state["prev"] = dict(portfolio.weights)
        return dict(portfolio.weights)

    strategy.signal_name = signal_name  # type: ignore[attr-defined]
    strategy.weighting_method = "optimized"  # type: ignore[attr-defined]
    return strategy
