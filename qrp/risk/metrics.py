"""Risk metrics for backtest results: Value at Risk and friends (roadmap P1).

This module answers "how bad could a bad day be?" — the question the
performance metrics in qrp.backtest.metrics do not ask directly. It adds
only what is missing there:

- Historical Value at Risk (VaR) at 95% and 99%: the daily loss that the
  worst 5% (or 1%) of days reached or exceeded.
- Conditional VaR (CVaR, "expected shortfall"): the *average* loss on those
  worst days — what a bad day typically looks like once you are in one.
- 1-day volatility: the typical daily wobble, the daily cousin of the
  annualized volatility already in qrp.backtest.metrics.

"Historical" means no fancy distribution is assumed: the numbers come
straight from the strategy's own past daily returns, sorted worst first.
Every metric is returned with the same plain-English
label/what/why/good explanations as the rest of the platform, because the
person reading the report is not assumed to have a finance background.

Deliberately NOT duplicated here: max drawdown and annualized volatility —
those already live in qrp.backtest.metrics.compute_metrics and the UI
reuses those entries directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    """Daily simple returns from a backtest equity curve.

    Args:
        equity: DataFrame with a "portfolio_value" column (the shape the
            backtest engine produces).

    Returns:
        Series of daily returns (0.01 = +1%), NaN/infinite values dropped.
    """
    values = equity["portfolio_value"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    return pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()


def historical_var(returns: pd.Series, confidence: float) -> float:
    """Historical Value at Risk as a positive loss fraction.

    Plain English: sort every daily return from worst to best; VaR at 95%
    is how bad the day was at the 5%-worst mark. "VaR 95% = 2%" means: on
    95 days out of 100 the daily loss stayed below 2% — but on the worst
    5% of days it was 2% or worse.

    Args:
        returns: daily simple returns.
        confidence: 0.95 or 0.99.

    Returns:
        The loss threshold as a positive fraction (0.02 = 2%), or NaN when
        there are no returns to measure.
    """
    rets = pd.Series(returns, dtype=float).dropna()
    if rets.empty:
        return float("nan")
    # The (1 - confidence) quantile is a return (negative for a loss);
    # negate it so the answer reads as "lost at least X%".
    return float(-np.quantile(rets.to_numpy(), 1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float) -> float:
    """Historical Conditional VaR ("expected shortfall") as a positive loss.

    Plain English: the *average* loss across the worst (1 - confidence)
    slice of days. VaR tells you where the bad days start; CVaR tells you
    how bad they typically were once you were in one. Always at least as
    big as the matching VaR.

    Returns:
        Positive loss fraction, or NaN when there are no returns.
    """
    rets = pd.Series(returns, dtype=float).dropna()
    if rets.empty:
        return float("nan")
    cutoff = np.quantile(rets.to_numpy(), 1.0 - confidence)
    tail = rets[rets <= cutoff]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def compute_risk_metrics(equity: pd.DataFrame) -> dict[str, dict[str, object]]:
    """VaR/CVaR at 95%/99% plus 1-day volatility, from an equity curve.

    Args:
        equity: DataFrame with a "portfolio_value" column (the shape the
            backtest engine produces).

    Returns:
        Dict mapping metric key -> {"value": float, "label": str, "what":
        str, "why": str, "good": str} — the same shape as
        qrp.backtest.metrics.compute_metrics, so the UI renders them the
        same way. Values are NaN when the curve is too short to measure.
    """
    rets = daily_returns(equity)
    n = len(rets)
    var_95 = historical_var(rets, 0.95) if n else float("nan")
    cvar_95 = historical_cvar(rets, 0.95) if n else float("nan")
    var_99 = historical_var(rets, 0.99) if n else float("nan")
    cvar_99 = historical_cvar(rets, 0.99) if n else float("nan")
    # 1-day volatility: the typical daily wobble — std of daily returns,
    # the daily cousin of the annualized volatility in compute_metrics.
    daily_vol = float(rets.std(ddof=1)) if n > 1 else float("nan")

    return {
        "var_95": {
            "value": var_95,
            "label": "Bad-day loss threshold (VaR 95%)",
            "what": "On a bad day — one of the worst 5% of days — this "
                    "strategy lost at least this much. Said the other way: "
                    "95 days out of 100, the daily loss stayed below this number.",
            "why": "Returns alone don't tell you how scary the worst days "
                   "were. This is the 'how bad does bad get' number.",
            "good": "Smaller is calmer. Compare it with the strategy's "
                    "typical daily wobble below — a VaR far above it means "
                    "the worst days were real outliers.",
        },
        "cvar_95": {
            "value": cvar_95,
            "label": "Average loss on the worst days (CVaR 95%)",
            "what": "The average loss across the worst 5% of days. VaR "
                    "tells you where the bad days start; this tells you how "
                    "bad they typically were once you were in one.",
            "why": "Two strategies can share the same VaR while one's bad "
                   "days are much worse than the other's. This catches that "
                   "difference.",
            "good": "Smaller is better, and close to the VaR 95% above is "
                    "reassuring — it means the worst days were bad, but not "
                    "catastrophically worse than the threshold.",
        },
        "var_99": {
            "value": var_99,
            "label": "Very-bad-day loss threshold (VaR 99%)",
            "what": "Like the 95% version, but for the worst 1% of days — "
                    "the truly rare, ugly days. 99 days out of 100, the "
                    "daily loss stayed below this number.",
            "why": "The 95% number can hide how extreme the rarest days "
                   "get. This one looks one step further into the tail.",
            "good": "Smaller is calmer. With short backtests this number is "
                    "noisy — it is most meaningful over several years of data.",
        },
        "cvar_99": {
            "value": cvar_99,
            "label": "Average loss on the very worst days (CVaR 99%)",
            "what": "The average loss across the worst 1% of days — what a "
                    "genuinely terrible day typically cost.",
            "why": "The 'what if things really break' number. Useful for "
                   "sizing how much pain a strategy can inflict in a crash.",
            "good": "Smaller is better. Treat it as a rough guide, not a "
                    "promise — history's worst day is rarely the future's.",
        },
        "daily_volatility": {
            "value": daily_vol,
            "label": "Typical daily wobble (1-day volatility)",
            "what": "On a typical day, the portfolio's value moved about "
                    "this much up or down. The daily cousin of the yearly "
                    "'bumpiness' number — multiply by about 16 (the square "
                    "root of 252 trading days) to get the yearly version.",
            "why": "Puts the VaR numbers in context: a 2% VaR means "
                   "something very different when the typical day moves 1% "
                   "versus 0.2%.",
            "good": "Lower is calmer. Individual stocks usually wobble "
                    "1–3% a day; a diversified portfolio should wobble less.",
        },
    }
