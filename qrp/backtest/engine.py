"""Long-only backtesting engine (Phase 3).

Simulates, day by day, what would have happened if you had followed a
signal-based strategy in the past. Long only, no leverage, no shorting.

THE MOST IMPORTANT RULE IN THIS FILE — NO LOOKAHEAD:
    A backtest is only honest if the simulated trader never uses information
    from the future. This engine enforces that two ways:
      1. On a rebalance date T, the strategy may only see signal values dated
         on or before T (signals themselves are point-in-time: each value was
         computed from price data available on that date).
      2. Trades decided on date T execute at the NEXT trading day's close
         (T+1), never at T's close — because in real life you only know T's
         signal after T's market close.
    If either rule were broken, results would look better than any real
    trader could achieve. They are covered by tests/test_backtest.py.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


from qrp.backtest.costs import CostModel, FlatCostModel
from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.signals.store import load_bars
# Fraction of a basis point... no — basis points: 1 bp = 0.01%.
# cost_bps=10 means 0.10% of traded value is lost to costs on each trade.
# The flat model (FlatCostModel) is the engine default; see qrp/backtest/costs.py
# for the alternative volatility/spread-based model.
TRADING_DAYS_PER_YEAR = 252

# Strategy: given the full signal matrix and a rebalance date, return target
# weights {ticker: weight}. Weights must be >= 0 and sum <= 1 (long only);
# whatever is not invested stays in cash.
StrategyFn = Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]]


@dataclass
class BacktestResult:
    """Everything a backtest run produces."""

    equity: pd.DataFrame  # columns: date (datetime), portfolio_value (float)
    trades: pd.DataFrame  # columns: date, ticker, action, shares, price, cost
    # Rebalance date (YYYY-MM-DD) -> {ticker: target weight}; the long-only
    # invariant (weights >= 0, sum <= 1) holds for every entry by construction.
    target_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    start_capital: float = 0.0
    cost_bps: float = 0.0
    # Which cost model priced the trades: "flat" or "spread" (see costs.py).
    cost_model_name: str = "flat"
    signal_name: str = ""
    drift_tolerance: float = 0.0
    # Cash dividends credited during the run: one row per payout with
    # columns date, ticker, shares_held, amount_per_share, amount_paid.
    dividends: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "shares_held",
                "amount_per_share",
                "amount_paid",
            ]
        )
    )


def load_price_matrix(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Load close prices as a tickers-x-dates matrix.

    Returns:
        DataFrame indexed by date (DatetimeIndex, oldest first) with one
        column per ticker of close prices. Prices are forward-filled so a
        missing day keeps the last known price (documented assumption).
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = load_bars(ticker, start=start, end=end, path=path)
        if not df.empty:
            frames[normalize_ticker(ticker)] = df["close"]
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames).sort_index()
    return prices.ffill()


def load_signal_matrix(
    signal_name: str,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Load one signal as a tickers-x-dates matrix from the signals table.

    Values are point-in-time: each was computed from data available on its
    date, so reading values dated <= T on rebalance date T is lookahead-free.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    tickers = [normalize_ticker(t) for t in tickers]
    placeholders = ",".join("?" for _ in tickers)
    query = (
        "SELECT ticker, date, value FROM signals WHERE signal = ? "
        f"AND ticker IN ({placeholders})"
    )
    params: list[str] = [signal_name, *tickers]
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    with sqlite3.connect(path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'signals'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return pd.DataFrame()
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot(index="date", columns="ticker", values="value").sort_index()
    return matrix


def rebalance_dates(
    dates: pd.DatetimeIndex, frequency: str = "daily"
) -> list[pd.Timestamp]:
    """Trading dates on which the portfolio is rebalanced.

    "monthly" -> first trading day of each calendar month.
    "weekly"  -> first trading day of each ISO week.
    "daily"   -> every trading day. The engine still only trades when a
        position's drift exceeds the drift buffer, so daily re-evaluation
        does not mean daily trading — it just means the check happens
        every day.
    """
    dates = dates.sort_values()
    if frequency == "monthly":
        keys = dates.strftime("%Y-%m")
    elif frequency == "weekly":
        # ISO year-week via isocalendar(): portable across macOS/Linux,
        # unlike strftime %G/%V which depend on the C library.
        iso = dates.isocalendar()
        keys = iso["year"].astype(str) + "-W" + iso["week"].astype(str)
    elif frequency == "daily":
        keys = dates.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"unknown rebalance frequency: {frequency!r}")
    seen: dict[str, pd.Timestamp] = {}
    for day, key in zip(dates, keys):
        if key not in seen:
            seen[key] = day
    # Sort by the actual calendar date, never by the label text: labels
    # like "2015-W10" sort before "2015-W2" as strings, which once caused
    # weekly backtests to silently stop rebalancing after ~11 weeks.
    return sorted(seen.values())


def _execute_rebalance(
    day: pd.Timestamp,
    price_row: pd.Series,
    target_weights: dict[str, float],
    shares: pd.Series,
    cash: float,
    cost_model: CostModel,
    drift_tolerance: float = 0.0,
) -> tuple[float, pd.Series, list[dict]]:
    """Trade toward target weights at today's close.

    Execution order keeps the economics honest:
      1. Sells go first; their proceeds (net of costs) return to cash.
      2. Buys are then sized to the cash actually available — a buy can
         never spend money the portfolio does not have, so cash can never
         go meaningfully negative (no hidden leverage, at any cost level).
      3. Trades smaller than $1 are skipped as dust (also filters out
         floating-point noise).
    Each trade is priced by ``cost_model``: the cost rate is computed per
    trade from the ticker, the trade day, and the trade's dollar value
    (``CostModel.rate``), so a spread/volatility model can charge different
    rates for different stocks and order sizes. The flat default keeps the
    old ``cost_bps`` behavior exactly.

    ``drift_tolerance`` (e.g. 0.02 = 2%) is the anti-churn filter: a trade
    is skipped when moving the position to its target would shift less
    than this fraction of the whole portfolio's value. In plain English:
    don't pay trading costs to fix a wobble nobody would notice. A
    position whose target weight is within 2% of its current weight is
    simply left alone.
    """
    portfolio_value = cash + float((shares * price_row).sum())
    new_shares = shares.copy()
    trades: list[dict] = []
    # The smallest trade worth paying costs for: $1 of dust, or the drift
    # tolerance as dollars — whichever is bigger. |target - current| as a
    # weight times portfolio value is exactly |delta_value|, so comparing
    # the dollar move against tolerance x portfolio value implements the
    # weight-based drift rule without any extra bookkeeping.
    min_trade_value = max(1.0, drift_tolerance * portfolio_value)

    def record(ticker: str, delta_shares: float, price: float) -> float:
        """Append a trade row; return its cost."""
        trade_value = abs(delta_shares) * price
        rate = cost_model.rate(ticker, day, trade_value)
        cost = trade_value * rate
        trades.append(
            {
                "date": day,
                "ticker": ticker,
                "action": "BUY" if delta_shares > 0 else "SELL",
                "shares": float(abs(delta_shares)),
                "price": float(price),
                "cost": float(cost),
            }
        )
        return cost

    # Desired signed share changes implied by the target weights.
    desired: dict[str, float] = {}
    for ticker in shares.index:
        price = price_row[ticker]
        if pd.isna(price) or price <= 0:
            continue  # cannot price it; hold as-is (leading NaNs are not
            # forward-filled, so a ticker listed after the window start can
            # hit this branch — holding is the safe behavior)
        delta_value = (
            target_weights.get(ticker, 0.0) * portfolio_value
            - shares[ticker] * price
        )
        if abs(delta_value) < min_trade_value:
            continue  # dust, or within the drift tolerance — not worth trading
        desired[ticker] = delta_value / price

    # Sells first: proceeds (net of costs) go back to cash.
    for ticker, delta_shares in desired.items():
        if delta_shares >= 0:
            continue
        price = price_row[ticker]
        cost = record(ticker, delta_shares, price)
        cash += -delta_shares * price - cost
        new_shares[ticker] += delta_shares

    # Buys: limited to the cash actually available after sells settled.
    # Each buy's cash need includes its own cost rate, because a
    # per-trade model (e.g. spread-based) can price buys differently.
    buys = {t: dsh for t, dsh in desired.items() if dsh > 0}
    if buys:
        need = sum(
            dsh * price_row[t] * (1.0 + cost_model.rate(t, day, dsh * price_row[t]))
            for t, dsh in buys.items()
        )
        scale = min(1.0, cash / need) if need > 0 else 0.0
        for ticker, delta_shares in buys.items():
            price = price_row[ticker]
            scaled = delta_shares * scale
            if abs(scaled * price) < 1.0:
                continue  # scaled down to dust (or to zero)
            cost = record(ticker, scaled, price)
            cash -= scaled * price + cost
            new_shares[ticker] += scaled

    # Guard: with cash-constrained buys, cash can only go negative through
    # floating-point rounding. Anything larger is a leverage bug.
    if cash < -1.0 - 1e-6 * portfolio_value:
        raise RuntimeError(
            f"cash went negative ({cash:.2f}) on {day.date()}: leverage bug"
        )
    return cash, new_shares, trades


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    strategy: StrategyFn,
    frequency: str = "daily",
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    drift_tolerance: float = 0.0,
    cost_model: CostModel | None = None,
    dividends: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the daily long-only simulation.

    Args:
        prices: DataFrame indexed by date with one close-price column per ticker.
        signals: DataFrame of the same shape with signal scores (NaN allowed).
        strategy: function returning target weights for a rebalance date.
        frequency: "daily" (default), "weekly" or "monthly" rebalance.
            "daily" re-evaluates every trading day, but the drift buffer
            still gates actual trades — small day-to-day changes don't trade.
        cost_bps: trading cost in basis points per trade (10 = 0.10%).
            Used only when ``cost_model`` is None.
        initial_capital: starting cash.
        drift_tolerance: skip trades that would move a position by less
            than this fraction of portfolio value (0.02 = 2%). In plain
            English: don't pay to fix wobbles nobody would notice.
        cost_model: how each trade's cost is priced (a ``CostModel`` from
            qrp/backtest/costs.py). ``None`` (the default) means
            ``FlatCostModel(cost_bps)`` — the flat fee per trade — so every
            backtest ever run behaves exactly as before. Pass
            ``SpreadCostModel`` for volatility/spread-based costs.
        dividends: optional DataFrame with columns ticker, ex_date
            (datetime-like), amount (dollars per share). On each ex-date
            the engine adds amount x shares_held to cash BEFORE executing
            that day's trades — because the buyer on the ex-date is not
            entitled to the dividend, only whoever already owned the stock
            gets paid. Prices stay raw (unadjusted) so execution stays
            realistic; the equity curve then reflects true total return.
            ``None`` (the default) means no dividends are credited and
            every backtest ever run behaves exactly as before.

    Returns:
        BacktestResult with the daily equity curve, trade log, and the target
        weights chosen on every rebalance date.
    """
    if prices.empty:
        raise ValueError("no price data")
    if cost_model is None:
        cost_model = FlatCostModel(cost_bps)
    dates = prices.index.sort_values()
    # Align the signal matrix to the price grid; missing => NaN => excluded.
    signals = signals.reindex(index=dates, columns=prices.columns)

    # Ex-date -> [(ticker, amount_per_share)]: credited on the shares held
    # at the start of that day, before any trades execute.
    div_map: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    if dividends is not None and not dividends.empty:
        d = dividends.copy()
        d["ex_date"] = pd.to_datetime(d["ex_date"]).dt.normalize()
        for row in d.itertuples():
            if float(row.amount) > 0:
                div_map.setdefault(row.ex_date, []).append(
                    (normalize_ticker(row.ticker), float(row.amount))
                )

    rb_dates = rebalance_dates(dates, frequency)
    rb_iter = iter(rb_dates)
    next_rb = next(rb_iter, None)

    shares = pd.Series(0.0, index=prices.columns)
    cash = float(initial_capital)
    pending: tuple[pd.Timestamp, dict[str, float]] | None = None
    target_weights: dict[str, dict[str, float]] = {}
    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    div_rows: list[dict] = []

    for i, day in enumerate(dates):
        # 0) Credit cash dividends whose ex-date is today. Whoever held the
        #    stock at yesterday's close gets amount x shares as cash, added
        #    before any trades — the ex-date buyer is not entitled to it.
        for ticker, amount in div_map.get(day.normalize(), ()):
            held = float(shares.get(ticker, 0.0))
            if held > 0:
                paid = held * amount
                cash += paid
                div_rows.append(
                    {
                        "date": day,
                        "ticker": ticker,
                        "shares_held": held,
                        "amount_per_share": amount,
                        "amount_paid": paid,
                    }
                )
        # 1) Execute any rebalance decided yesterday (at today's close).
        if pending is not None and day == pending[0]:
            _, weights = pending
            cash, shares, day_trades = _execute_rebalance(
                day, prices.loc[day], weights, shares, cash, cost_model,
                drift_tolerance,
            )
            trade_rows.extend(day_trades)
            pending = None
        # 2) On a rebalance date, decide new targets from signals <= today,
        #    to be executed at the NEXT trading day's close (no lookahead).
        if next_rb is not None and day == next_rb:
            weights = strategy(signals, day)
            for w in weights.values():
                if w < 0:
                    raise ValueError(f"strategy returned negative weight on {day}")
            if sum(weights.values()) > 1.0 + 1e-9:
                raise ValueError(f"strategy weights sum > 1 on {day}")
            target_weights[day.strftime("%Y-%m-%d")] = dict(weights)
            if i + 1 < len(dates):
                pending = (dates[i + 1], weights)
            next_rb = next(rb_iter, None)
        # 3) Mark the portfolio to market at today's close.
        portfolio_value = cash + float((shares * prices.loc[day]).sum())
        equity_rows.append({"date": day, "portfolio_value": portfolio_value})

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(
        trade_rows, columns=["date", "ticker", "action", "shares", "price", "cost"]
    )
    divs = pd.DataFrame(
        div_rows,
        columns=["date", "ticker", "shares_held", "amount_per_share", "amount_paid"],
    )
    return BacktestResult(
        equity=equity,
        trades=trades,
        target_weights=target_weights,
        start_capital=float(initial_capital),
        cost_bps=float(cost_bps),
        cost_model_name=cost_model.name,
        drift_tolerance=float(drift_tolerance),
        dividends=divs,
    )
