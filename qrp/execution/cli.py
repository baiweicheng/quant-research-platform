"""CLI: rebalance the PAPER account toward a signal-based target portfolio.

Paper trading means simulated trading with fake money: the account, the
orders, and the fees are all real math, but no real dollars move and nothing
touches a real brokerage. It is how you practice and validate before risking
anything real.

What this command does, step by step:

1. Builds a target portfolio from a signal (like the portfolio CLI does).
2. Compares it to what the paper account currently holds.
3. Runs every resulting order through the risk safeguards.
4. Executes the surviving orders in the paper account (unless --dry-run).

Examples:
    python -m qrp.execution.cli --dry-run
        # show what WOULD be traded today, without trading anything
    python -m qrp.execution.cli --signal momentum --method signal
        # actually rebalance the paper account (still fake money)
    python -m qrp.execution.cli --reset --capital 50000
        # wipe the paper account and start over with $50,000 of play money
        # (does not trade — run again without --reset to trade)
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from qrp.backtest.engine import load_signal_matrix
from qrp.data.store import DEFAULT_DB
from qrp.execution.paper import PaperBroker
from qrp.execution.rebalance import orders_to_target
from qrp.portfolio.cli import METHOD_CHOICES
from qrp.portfolio.construction import METHOD_DESCRIPTIONS, build_target_weights
from qrp.risk.guards import PortfolioState, check_order
from qrp.signals import default_registry
from qrp.signals.store import list_tickers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebalance the PAPER trading account (simulated money only)."
    )
    p.add_argument(
        "--signal",
        default="momentum",
        help="Signal name to rank stocks by (default: momentum)",
    )
    p.add_argument(
        "--method",
        default="equal",
        choices=sorted(METHOD_CHOICES),
        help=(
            "How to size positions: equal (even split), signal (more money to "
            "higher scores), inverse_vol (more money to calmer stocks). "
            "Default: equal"
        ),
    )
    p.add_argument(
        "--date",
        default=None,
        help="Trading day YYYY-MM-DD (default: latest trading day in the database)",
    )
    p.add_argument(
        "--tickers",
        default="all",
        help="Comma-separated US tickers, or 'all' in the db (default: all)",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting play-money cash — only used together with --reset (default: 100000)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the paper account and start over with --capital play money",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the orders and risk-check results without executing anything",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def _latest_trading_day(db: str) -> str:
    import sqlite3

    with sqlite3.connect(db) as conn:
        (day,) = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
    if not day:
        raise SystemExit("No price data — run the data CLI first.")
    return str(day)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = default_registry()
    try:
        registry.get(args.signal)
    except KeyError:
        available = ", ".join(n for n, _ in registry.list_signals())
        print(f"Unknown signal {args.signal!r}. Available: {available}")
        return 1

    broker = PaperBroker(args.db)
    if args.reset:
        broker.reset(args.capital)
        print(
            f"Paper account reset: starting over with ${args.capital:,.2f} "
            "of play money. No real money involved."
        )
        print("Run without --reset to trade from this fresh account.")
        return 0

    day = args.date or _latest_trading_day(args.db)

    if args.tickers.strip().lower() == "all":
        tickers = list_tickers(args.db)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers found (is daily_bars populated?).")
        return 1

    # Same point-in-time discipline as the portfolio CLI: only signal values
    # dated on or before the trading day are used — never future data.
    matrix = load_signal_matrix(args.signal, tickers, path=args.db)
    if matrix.empty:
        print(
            f"No stored values for signal {args.signal!r}. "
            "Run the signals CLI first: python -m qrp.signals.cli"
        )
        return 1
    history = matrix.loc[: pd.Timestamp(day)].ffill()
    if history.empty:
        print(f"No signal data on or before {day}.")
        return 1
    scores = history.iloc[-1].dropna().to_dict()
    if not scores:
        print(f"No scored stocks on or before {day}.")
        return 1
    signal_day = history.index[-1].strftime("%Y-%m-%d")

    method = METHOD_CHOICES[args.method]
    volatilities = None
    if method == "inverse_volatility":
        low_vol = load_signal_matrix("low_volatility", tickers, path=args.db)
        if low_vol.empty:
            print("No stored low_volatility values — needed for inverse-vol weighting.")
            return 1
        v_hist = (-low_vol).loc[: pd.Timestamp(day)].ffill()
        if not v_hist.empty:
            volatilities = v_hist.iloc[-1].dropna().to_dict()

    portfolio = build_target_weights(scores, method=method, volatilities=volatilities)

    closes = {t: broker.last_price(t, day) for t in set(portfolio.weights) | set(broker.positions)}
    total_value = broker.cash + sum(
        shares * closes[t] for t, shares in broker.positions.items()
    )
    cost_bps = getattr(broker, "cost_bps", 0.0)
    orders = orders_to_target(broker, portfolio.weights, day, total_value, cost_bps)

    print(f"Paper rebalance — signal '{args.signal}' as of {signal_day}, trading {day}")
    print(f"Method: {args.method} — {METHOD_DESCRIPTIONS[method]}")
    print(f"Account value before: ${total_value:,.2f} "
          f"(cash ${broker.cash:,.2f}, {len(broker.positions)} position(s))")
    print()
    if not orders:
        print("Already at target — no trades needed.")
        return 0

    if args.dry_run:
        print("DRY RUN — nothing will be traded. Risk-checking each order:")
        state = PortfolioState(
            cash=broker.cash, positions=broker.positions, prices=closes
        )
        for order in orders:
            ok, reason = check_order(order, state, broker.risk_config)
            target_pct = portfolio.weights.get(order.ticker, 0.0)
            if ok:
                print(
                    f"  WOULD {order.action} {order.shares:,.2f} shares of "
                    f"{order.ticker} at ${order.price:,.2f} "
                    f"(toward a {target_pct:.0%} target) — passed every safety check."
                )
            else:
                print(f"  WOULD {order.action} {order.ticker} — BLOCKED: {reason}")
        print()
        print("Dry run complete — the paper account was not touched.")
        return 0

    for order in orders:
        result = broker.place_order(order.ticker, order.action, order.shares, day)
        target_pct = portfolio.weights.get(order.ticker, 0.0)
        if result.filled:
            verb = "Bought" if result.action == "BUY" else "Sold"
            print(
                f"{verb} {result.shares:,.2f} shares of {result.ticker} at "
                f"${result.price:,.2f} (fees ${result.cost:,.2f}) — moving "
                f"toward your {target_pct:.0%} target."
            )
        else:
            print(f"Skipped {order.action} {order.ticker}: {result.reason}")

    closes = {t: broker.last_price(t, day) for t in broker.positions}
    total_value = broker.cash + sum(
        shares * closes[t] for t, shares in broker.positions.items()
    )
    print()
    print(f"Account value after: ${total_value:,.2f}")
    print(f"Cash: ${broker.cash:,.2f}")
    if broker.positions:
        print("Holdings:")
        for ticker, shares in sorted(broker.positions.items()):
            value = shares * closes[ticker]
            print(
                f"  {ticker:<6}{shares:>12,.2f} shares  "
                f"${value:>12,.2f} ({value / total_value:>6.1%} of account)"
            )
    print()
    print("Reminder: this was paper trading — simulated money only. "
          "No real brokerage is connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
