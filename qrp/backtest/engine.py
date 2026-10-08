"""Long-only backtesting engine (Phase 3).

Simulates, day by day, what would have happened if you had followed a
signal-based strategy in the past. Long only, no leverage, no shorting.

THE MOST IMPORTANT RULE IN THIS FILE — NO LOOKAHEAD:
    A backtest is only honest if the simulated trader never uses information
    from the future. This engine enforces that two ways:
      1. On a rebalance date T, the strategy may only see signal values dated
         on or before T (signals themselves are point-in-time: each value was
         computed from price data available on that date).
      2. Trades decided on date T execute at the NEXT trading day (T+1) —
         never on T itself — because in real life you only know T's signal
         after T's market close. *When* on T+1 they fill is the `execution`
         setting: "open" (the default — you run the rebalance at night and
         the orders fill at the next morning's open, like real market-on-open
         orders) or "close" (the older convention). Either way, nothing dated
         after the fill moment is ever used.
    If either rule were broken, results would look better than any real
    trader could achieve. They are covered by tests/test_backtest.py and
    tests/test_execution_timing.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


from qrp.backtest.costs import CostModel, FlatCostModel
from qrp.data.store import DEFAULT_DB, connect_db
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
    # When trades filled: "open" (next trading day's open — the default) or
    # "close" (next trading day's close — the older convention).
    execution: str = "open"
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
    with_validity: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load close prices as a tickers-x-dates matrix.

    Returns:
        DataFrame indexed by date (DatetimeIndex, oldest first) with one
        column per ticker of close prices. Prices are forward-filled so a
        missing day keeps the last known price (documented assumption).
        With with_validity=True, returns (prices, valid) where valid is a
        boolean frame that is True only on days with a real downloaded bar
        (before forward-filling) — used to detect delisted stocks whose
        price history ends mid-backtest.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = load_bars(ticker, start=start, end=end, path=path)
        if not df.empty:
            frames[normalize_ticker(ticker)] = df["close"]
    if not frames:
        empty = pd.DataFrame()
        return (empty, empty) if with_validity else empty
    prices = pd.DataFrame(frames).sort_index()
    valid = prices.notna()
    prices = prices.ffill()
    if with_validity:
        return prices, valid
    return prices


def load_open_matrix(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    path: str | Path = DEFAULT_DB,
    with_validity: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load opening prices as a tickers-x-dates matrix.

    Same shape and date grid as load_price_matrix, but with each day's
    opening price — the price a next-day-open execution fills at. Forward-
    filled like closes (documented assumption). With with_validity=True,
    returns (opens, valid) like load_price_matrix.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = load_bars(ticker, start=start, end=end, path=path)
        if not df.empty:
            frames[normalize_ticker(ticker)] = df["open"]
    if not frames:
        empty = pd.DataFrame()
        return (empty, empty) if with_validity else empty
    opens = pd.DataFrame(frames).sort_index()
    valid = opens.notna()
    opens = opens.ffill()
    if with_validity:
        return opens, valid
    return opens


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
    with connect_db(path) as conn:
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


def last_valid_dates(valid: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Map each ticker to its last date with a real (non-filled) bar.

    A stock whose price history ends mid-backtest was delisted (or its
    download stopped); the engine closes such positions at that last real
    price instead of trading on forward-filled ghosts.
    """
    out: dict[str, pd.Timestamp] = {}
    for ticker in valid.columns:
        col = valid[ticker]
        if bool(col.any()):
            out[ticker] = col[col].index.max()
    return out


def _execute_rebalance(
    day: pd.Timestamp,
    price_row: pd.Series,
    target_weights: dict[str, float],
    shares: pd.Series,
    cash: float,
    cost_model: CostModel,
    drift_tolerance: float = 0.0,
    cost_day: pd.Timestamp | None = None,
) -> tuple[float, pd.Series, list[dict]]:
    """Trade toward target weights at the given execution prices.

    `price_row` is the fill price for each ticker on `day` — the day's
    close under "close" execution, or the day's open under "open"
    execution (chosen by the caller in run_backtest). `cost_day` is the
    date the cost model may look at: for open execution it is the trading
    day *before* the fill, because a trader filling at the open cannot
    know that day's high/low/close yet. (None means "same as day".)

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
    # The date the cost model is allowed to see. For open execution this
    # is the day before the fill (no peeking at the fill day's range).
    cost_asof = cost_day if cost_day is not None else day

    def record(ticker: str, delta_shares: float, price: float) -> float:
        """Append a trade row; return its cost."""
        trade_value = abs(delta_shares) * price
        rate = cost_model.rate(ticker, cost_asof, trade_value)
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
            dsh * price_row[t] * (1.0 + cost_model.rate(t, cost_asof, dsh * price_row[t]))
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
    execution: str = "open",
    opens: pd.DataFrame | None = None,
    universe_filter: Callable[[pd.Timestamp], set[str]] | None = None,
    last_valid: dict[str, pd.Timestamp] | None = None,
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
        execution: when a rebalance decided on date T fills — "open"
            (default) fills at T+1's opening price, matching the real
            workflow of running the rebalance at night and staging
            market-on-open orders; "close" fills at T+1's close (the older
            convention — pre-existing backtest results used this).
        opens: DataFrame like ``prices`` but with each day's opening price,
            used as fill prices when execution="open". ``None`` means the
            closes stand in as opens (exact when they're equal, as in most
            test fixtures). Ignored when execution="close".
        universe_filter: optional function mapping a date to the set of
            tickers eligible on that date (point-in-time index membership).
            On each rebalance date the strategy only sees eligible tickers,
            so a stock can never be bought before it joined the index; a
            held stock that leaves the index gets sold at the next
            rebalance. ``None`` (the default) keeps the old behavior.
        last_valid: optional {ticker: last date with a real downloaded bar}.
            A stock whose price history has ended (delisted) can no longer
            be bought, and a held one is force-closed on the first trading
            date after its last real bar, at that last real close —
            regardless of the execution setting. ``None`` keeps the old
            behavior.

    Returns:
        BacktestResult with the daily equity curve, trade log, and the target
        weights chosen on every rebalance date.
    """
    if prices.empty:
        raise ValueError("no price data")
    if execution not in ("open", "close"):
        raise ValueError(
            f"unknown execution {execution!r} — use 'open' or 'close'"
        )
    if cost_model is None:
        cost_model = FlatCostModel(cost_bps)
    dates = prices.index.sort_values()
    # Align the signal matrix to the price grid; missing => NaN => excluded.
    signals = signals.reindex(index=dates, columns=prices.columns)
    # Forward-fill ONCE up front. Every strategy slices `.loc[:day]` per
    # rebalance date and calls `.ffill()` again, but ffill is idempotent —
    # on already-filled data it is a no-op — so pre-filling here turns D
    # per-date full-matrix fills into one. (Audit 2.6 / 1.2.)
    signals = signals.ffill()
    # Align the opens to the same grid (only needed for open execution).
    if execution == "open":
        if opens is None:
            opens = prices
        else:
            opens = opens.reindex(index=dates, columns=prices.columns)

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
        # 0b) Force-close delisted holdings: a stock whose price history
        #     ended (last_valid < today) is sold TODAY at its last real
        #     close — never carried at a stale forward-filled price until
        #     the next rebalance, and never at a forward-filled open. The
        #     position can never be reopened (the pending-execution filter
        #     below blocks rebuys).
        if last_valid is not None:
            for t in list(shares.index[shares > 0]):
                lv = last_valid.get(t)
                if lv is None or lv.normalize() >= day.normalize():
                    continue
                lv_day = lv.normalize()
                if lv_day in prices.index:
                    exit_price = float(prices.loc[lv_day, t])
                else:  # shouldn't happen; fall back to today's (filled) row
                    exit_price = float(prices.loc[day, t])
                held_shares = float(shares[t])
                if exit_price > 0 and held_shares > 0:
                    trade_value = held_shares * exit_price
                    # Priced off the last real day itself: the exit is
                    # booked at that day's close, whose range is known.
                    cost = trade_value * cost_model.rate(t, lv_day, trade_value)
                    cash += trade_value - cost
                    trade_rows.append(
                        {
                            "date": day,
                            "ticker": t,
                            "action": "SELL",
                            "shares": held_shares,
                            "price": exit_price,
                            "cost": float(cost),
                        }
                    )
                    shares[t] = 0.0
        # 1) Execute any rebalance decided yesterday. The fill price is
        # T+1's open under "open" execution, or T+1's close under "close".
        # Either way the fill uses only T+1's own bar — never the future.
        if pending is not None and day == pending[0]:
            _, weights = pending
            if last_valid is not None:
                # A delisted stock (price history ended) can no longer be
                # bought; any held one was already force-closed above at
                # its last real close.
                weights = {
                    t: w for t, w in weights.items()
                    if last_valid.get(t, day) >= day
                }
            exec_row = opens.loc[day] if execution == "open" else prices.loc[day]
            # Cost estimates may not peek at the fill day's range: under open
            # execution the trader at the open can't know the day's high/low/
            # close, so the cost model sees the previous trading day instead.
            cost_day = dates[i - 1] if (execution == "open" and i > 0) else day
            cash, shares, day_trades = _execute_rebalance(
                day, exec_row, weights, shares, cash, cost_model,
                drift_tolerance, cost_day=cost_day,
            )
            trade_rows.extend(day_trades)
            pending = None
        # 2) On a rebalance date, decide new targets from signals <= today,
        #    to be executed at the NEXT trading day (no lookahead).
        if next_rb is not None and day == next_rb:
            if universe_filter is not None:
                # Point-in-time membership: the strategy only ranks stocks
                # that were actually in the index on this date, so it can
                # never buy a future member (no survivorship lookahead).
                allowed = universe_filter(day)
                use_signals = signals[
                    [c for c in signals.columns if c in allowed]
                ]
            else:
                allowed = None
                use_signals = signals
            weights = strategy(use_signals, day)
            if allowed is not None:
                weights = {
                    t: w for t, w in weights.items() if t in allowed
                }
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
        execution=execution,
        dividends=divs,
    )
