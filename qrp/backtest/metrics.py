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

    # Annualized turnover (one-way convention): buys and sells are summed
    # and then halved, so a full "sell everything, buy everything else"
    # rebalance counts as 100% turnover, not 200%. Divided by average
    # portfolio value and years to annualize.
    if not trades.empty:
        gross_notional = float((trades["shares"] * trades["price"]).sum())
    else:
        gross_notional = 0.0
    notional = gross_notional / 2.0
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
            "label": "Turnover (one-way trading per year)",
            "what": "Roughly what fraction of the portfolio gets replaced each year. "
                    "Selling everything and buying a whole new portfolio counts as 100% "
                    "(each buy-or-sell leg is counted once, not twice).",
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


def benchmark_equity(
    prices: pd.DataFrame, initial_capital: float = 100_000.0
) -> pd.DataFrame:
    """S&P 500 (SPY) buy-and-hold benchmark over the backtest dates.

    Plain English: on the first day, put all the starting cash into SPY —
    the ETF that tracks the S&P 500 — and never trade again. The curve
    shows what "just owning the US stock market" would have done, so any
    strategy can be judged against the simplest alternative: doing nothing.

    Args:
        prices: DataFrame indexed by date with a single "SPY" close-price
            column (dividends are credited separately by the engine, so
            this is price-only — the same basis as every strategy curve).
        initial_capital: starting cash.

    Returns:
        DataFrame with "date" and "portfolio_value" columns, the same
        shape as a backtest equity curve, so the same metrics apply.
    """
    cols = list(prices.columns)
    if len(cols) != 1:
        raise ValueError(
            f"benchmark_equity needs exactly one price column (SPY), "
            f"got {cols}"
        )
    return buy_and_hold_equity(prices, initial_capital)


def _align_returns(
    equity: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Daily returns of strategy and benchmark on their shared dates."""
    e = equity.set_index("date")["portfolio_value"].astype(float)
    b = benchmark.set_index("date")["portfolio_value"].astype(float)
    common = e.index.intersection(b.index).sort_values()
    e, b = e.loc[common], b.loc[common]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_p = (e / e.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
        r_b = (b / b.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
    both = pd.DataFrame({"p": r_p, "b": r_b}).dropna()
    return both["p"], both["b"]


def relative_metrics(
    equity: pd.DataFrame, benchmark: pd.DataFrame
) -> dict[str, dict[str, object]]:
    """How a strategy did *compared with* a benchmark (e.g. S&P 500 via SPY).

    Every metric comes with the same plain-English label/what/why/good
    explanations as compute_metrics. Formulas use daily returns over the
    dates both curves share (252 trading days = 1 year).

    Returns:
        Dict mapping metric key -> {"value", "label", "what", "why", "good"}.
        Values are NaN when there is not enough overlapping history.
    """
    r_p, r_b = _align_returns(equity, benchmark)
    n = len(r_p)
    out: dict[str, dict[str, object]] = {}

    def entry(key, value, label, what, why, good):
        out[key] = {
            "value": float(value),
            "label": label,
            "what": what,
            "why": why,
            "good": good,
        }

    if n < 2 or r_b.std(ddof=1) == 0:
        nan = float("nan")
        beta = alpha = nan
    else:
        # Beta: how strongly the strategy moves with the benchmark —
        # cov(strategy, market) / var(market). 1.0 = moves in lockstep.
        beta = float(r_p.cov(r_b) / r_b.var(ddof=1))
        # Alpha (annualized): the average daily return left over after
        # removing what the market exposure explains, times 252. Positive
        # means the strategy earned more than its market sensitivity alone
        # would predict (0% cash baseline, like the Sharpe ratio here).
        alpha = float((r_p - beta * r_b).mean() * TRADING_DAYS_PER_YEAR)

    entry(
        "beta",
        beta,
        "Beta vs S&P 500 (market sensitivity)",
        "How strongly the strategy moves when the S&P 500 moves. 1.0 means "
        "it moves in lockstep with the market; 0.5 means about half as "
        "much; above 1.0 means its swings are amplified.",
        "Tells you how much of the ride is just 'the market going up and "
        "down' versus the strategy's own choices.",
        "No universal target — near 1.0 means market-like swings; well "
        "below 1.0 means a calmer ride than the market.",
    )
    entry(
        "alpha",
        alpha,
        "Alpha vs S&P 500 (extra yearly return)",
        "How much the strategy beat — or trailed — what you would expect "
        "from its market sensitivity alone, as a yearly percent. It answers: "
        "'after accounting for how closely it follows the market, did it "
        "actually add anything?'",
        "The purest single number for 'did this strategy earn its keep "
        "versus just owning the market'.",
        "Positive is good — it means the strategy added return beyond what "
        "its market exposure explains. Consistently negative means owning "
        "an index fund would have done better.",
    )

    # Tracking error: how wobbly the *difference* between strategy and
    # benchmark is — std of daily excess returns, annualized.
    active = r_p - r_b
    if n > 1 and active.std(ddof=1) > 0:
        tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        # Information ratio: average excess return per unit of tracking
        # error, annualized — a Sharpe ratio measured against the market
        # instead of cash.
        information_ratio = float(
            active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        tracking_error = float("nan")
        information_ratio = 0.0 if n > 1 else float("nan")

    entry(
        "tracking_error",
        tracking_error,
        "Tracking error (wobble vs S&P 500)",
        "How much the strategy's day-to-day returns typically differ from "
        "the S&P 500's, as a yearly percent. 5% means the gap between the "
        "two usually stays within a few percent a year.",
        "A strategy that claims to beat the market but swings wildly away "
        "from it is a bumpier bet than the number alone suggests.",
        "Lower means it hugs the benchmark more closely. Index funds sit "
        "near 0–1%; active strategies often land between 4% and 10%.",
    )
    entry(
        "information_ratio",
        information_ratio,
        "Information ratio (consistency of outperformance)",
        "The strategy's extra return over the S&P 500 divided by how wobbly "
        "that extra return was — like the Sharpe ratio, but measured "
        "against the market instead of cash.",
        "Beating the market by a lot in one lucky year is less impressive "
        "than beating it a little, steadily, every year. This tells them apart.",
        "Above 0.5 is solid, above 1.0 is excellent. Below 0 means it "
        "trailed the market.",
    )

    # Up/down-market capture: on days the benchmark rose (fell), what
    # fraction of its average move did the strategy capture (suffer)?
    up = r_b > 0
    down = r_b < 0
    if up.sum() > 0 and r_b[up].mean() != 0:
        up_capture = float(r_p[up].mean() / r_b[up].mean())
    else:
        up_capture = float("nan")
    if down.sum() > 0 and r_b[down].mean() != 0:
        down_capture = float(r_p[down].mean() / r_b[down].mean())
    else:
        down_capture = float("nan")

    entry(
        "up_capture",
        up_capture,
        "Up-market capture",
        "When the S&P 500 rose, what share of that rise the strategy "
        "captured. 110% means on up days it rose about 10% more than the "
        "market; 90% means it captured nine-tenths of the rally.",
        "Shows whether the strategy participates when times are good.",
        "Above 100% is great in rallies — but check down-market capture "
        "too, because the same boldness often hurts when markets fall.",
    )
    entry(
        "down_capture",
        down_capture,
        "Down-market capture",
        "When the S&P 500 fell, what share of that fall the strategy "
        "suffered. 80% means on down days it fell only four-fifths as much "
        "as the market — here, lower is better.",
        "This is the crash-protection score: the whole point of many "
        "strategies is losing less when markets tumble.",
        "Below 100% means it cushioned the falls; below 80% is strong "
        "downside protection.",
    )
    return out


def buy_and_hold_equity(
    prices: pd.DataFrame, initial_capital: float = 100_000.0
) -> pd.DataFrame:
    """Equal-weighted buy-and-hold benchmark over the same price grid.

    Plain English: on the first day, split the starting cash evenly across
    every stock that has a price that day, buy them, and then never trade
    again. The curve shows what "just owning the market and sitting still"
    would have done — the honest baseline every strategy is judged against.

    Details: stocks with no price on the first day (e.g. listed later) are
    left out; later price gaps are carried forward at the last known price.
    Returns a DataFrame with "date" and "portfolio_value" columns, the same
    shape as a backtest equity curve, so the same metrics apply.
    """
    if prices.empty:
        raise ValueError("no price data")
    first_prices = prices.iloc[0].dropna()
    first_prices = first_prices[first_prices > 0]
    if first_prices.empty:
        raise ValueError("no usable prices on the first date")
    shares = (initial_capital / len(first_prices)) / first_prices
    held = prices[first_prices.index].ffill()
    values = (held * shares).sum(axis=1)
    return pd.DataFrame(
        {"date": prices.index, "portfolio_value": values.to_numpy(dtype=float)}
    )
