"""Performance metrics for backtest results (Phase 3).

Every metric is returned with a plain-English explanation — what it is, why
it matters, and roughly what "good" looks like — because the person reading
the report is not assumed to have a finance background. The exact formulas
live in the code comments below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _daily_returns(equity: pd.DataFrame) -> pd.Series:
    values = equity["portfolio_value"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    return pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()


def compute_metrics(
    equity: pd.DataFrame, trades: pd.DataFrame
) -> dict[str, dict[str, object]]:
    """Compute performance metrics from a backtest equity curve.

    Returns:
        Dict mapping metric key -> {"value": float, "label": str,
        "what": str, "why": str, "good": str}. "what"/"why"/"good" are
        plain-English explanations.
    """
    values = equity["portfolio_value"].to_numpy(dtype=float)
    start_value = float(values[0])
    end_value = float(values[-1])
    rets = _daily_returns(equity)
    n_days = len(rets)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days else 0.0

    # Total return: end/start - 1.
    total_return = end_value / start_value - 1.0 if start_value > 0 else float("nan")

    # CAGR: (end/start)^(252/n_days) - 1 — the constant yearly growth rate
    # that would turn the start value into the end value.
    cagr = (
        (end_value / start_value) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
        if n_days > 0 and start_value > 0 and end_value > 0
        else float("nan")
    )

    # Annualized volatility: std of daily returns * sqrt(252).
    volatility = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n_days > 1 else 0.0

    # Sharpe ratio: mean(daily return) / std(daily return) * sqrt(252),
    # with a 0% risk-free baseline (we assume cash earns nothing).
    if n_days > 1 and rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    # Max drawdown: the worst peak-to-trough fall, max_t (peak_t - value_t)/peak_t.
    running_peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peak > 0, (values - running_peak) / running_peak, 0.0)
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    # Annualized turnover: total traded notional / (average portfolio value * years).
    if not trades.empty:
        notional = float((trades["shares"] * trades["price"]).sum())
    else:
        notional = 0.0
    avg_value = float(np.mean(values)) if len(values) else 0.0
    turnover = notional / (avg_value * years) if avg_value > 0 and years > 0 else 0.0

    # Win rate: fraction of calendar months whose month-end value exceeded
    # the previous month-end value. Grouped manually (not resample) so this
    # works on any pandas version.
    month_key = pd.DatetimeIndex(equity["date"]).strftime("%Y-%m")
    month_end = equity.groupby(month_key)["portfolio_value"].last()
    monthly_rets = month_end.pct_change().dropna()
    win_rate = float((monthly_rets > 0).mean()) if len(monthly_rets) else float("nan")

    num_trades = int(len(trades))

    return {
        "final_value": {
            "value": end_value,
            "label": "Final portfolio value",
            "what": "What the portfolio was worth on the last day, starting from the initial cash.",
            "why": "The simplest summary of the whole simulation.",
            "good": "Higher than what you started with is the basic goal.",
        },
        "total_return": {
            "value": total_return,
            "label": "Total return",
            "what": "How much the portfolio grew (or shrank) from start to finish, in percent. "
                    "Start with $100,000 and end with $120,000 and that is +20%.",
            "why": "The bottom line: did the strategy make money over this period?",
            "good": "Positive is good. Always compare it with just holding a broad market fund over the same period.",
        },
        "cagr": {
            "value": cagr,
            "label": "CAGR (yearly growth rate)",
            "what": "The steady yearly growth rate that would get you from the start value to the end value. "
                    "It smooths out all the ups and downs into one 'per year' number.",
            "why": "Lets you fairly compare strategies tested over different lengths of time.",
            "good": "Higher is better. Around 10% a year over many years is historically strong for stocks.",
        },
        "volatility": {
            "value": volatility,
            "label": "Volatility (yearly bumpiness)",
            "what": "How much the portfolio's value typically wobbles up and down, as a yearly percent. "
                    "20% volatility means swings of roughly ±20% in a year are normal.",
            "why": "Bigger wobbles mean a bumpier ride and a higher chance of panic-selling at the worst moment.",
            "good": "Lower is calmer. The US stock market as a whole usually sits around 15–20%.",
        },
        "sharpe": {
            "value": sharpe,
            "label": "Sharpe ratio (return per unit of risk)",
            "what": "How much return you got for each bit of roller-coaster ride. It divides the average gain "
                    "by the bumpiness, so wild-but-lucky strategies do not look better than they are.",
            "why": "A strategy can look profitable only because it took huge risks. Sharpe adjusts for that.",
            "good": "Above 1 is solid, above 2 is excellent. Below 0 means holding cash would have done better.",
        },
        "max_drawdown": {
            "value": max_drawdown,
            "label": "Worst fall from a peak (max drawdown)",
            "what": "The biggest drop from a previous high point. −30% means at one point the portfolio was "
                    "down 30% from its best value so far.",
            "why": "This is the pain you would actually have felt. Strategies live or die by whether you can "
                   "stomach their worst episodes.",
            "good": "Closer to zero (a smaller fall) is better. Even great strategies often fall 20% or more at some point.",
        },
        "turnover": {
            "value": turnover,
            "label": "Turnover (trading activity per year)",
            "what": "Roughly what fraction of the portfolio gets bought and sold each year. 100% means the "
                    "entire portfolio turns over about once a year.",
            "why": "More trading means more costs and, in a real taxable account, more taxes.",
            "good": "Lower is usually better, all else equal.",
        },
        "win_rate": {
            "value": win_rate,
            "label": "Winning months",
            "what": "The share of calendar months that ended higher than they started.",
            "why": "Tells you how often the strategy feels like it is 'working' month to month.",
            "good": "Above 50% feels good, but a strategy can still be profitable below 50% if its winning months are bigger than its losing months.",
        },
        "num_trades": {
            "value": float(num_trades),
            "label": "Number of trades",
            "what": "How many individual buy or sell orders the simulation placed.",
            "why": "A sanity check on activity — and a reminder that every trade paid the configured cost.",
            "good": "No universal target; it should roughly match the rebalance schedule (e.g. monthly rebalancing of a few stocks means a handful of trades per month).",
        },
    }
