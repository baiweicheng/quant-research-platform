"""CLI: run a long-only backtest of a signal strategy and print the results.

Example:
    python -m qrp.backtest.cli --signal momentum --top 10 --rebalance monthly
    python -m qrp.backtest.cli --signal trend --top 5 --start 2024-01-01 --cost-bps 5
"""

from __future__ import annotations

import argparse
from datetime import date

from qrp.backtest.engine import load_price_matrix, load_signal_matrix, run_backtest
from qrp.backtest.metrics import compute_metrics
from qrp.backtest.strategies import top_n_by_signal
from qrp.data.store import DEFAULT_DB
from qrp.signals import default_registry
from qrp.signals.store import list_tickers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest a top-N long-only signal strategy."
    )
    p.add_argument(
        "--signal",
        default="momentum",
        help="Signal name to rank stocks by (default: momentum)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Hold the top N ranked stocks, equally weighted (default: 10)",
    )
    p.add_argument(
        "--rebalance",
        default="monthly",
        choices=["monthly", "weekly"],
        help="Rebalance schedule (default: monthly)",
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Trading cost in basis points per trade, 10 = 0.10%% (default: 10)",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting cash in dollars (default: 100000)",
    )
    p.add_argument(
        "--tickers",
        default="all",
        help="Comma-separated US tickers, or 'all' in the db (default: all)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def _fmt_pct(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:+.2%}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = default_registry()
    try:
        registry.get(args.signal)
    except KeyError:
        available = ", ".join(n for n, _ in registry.list_signals())
        print(f"Unknown signal {args.signal!r}. Available: {available}")
        return 1

    if args.tickers.strip().lower() == "all":
        tickers = list_tickers(args.db)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers found (is daily_bars populated?).")
        return 1

    prices = load_price_matrix(tickers, start=args.start, end=args.end, path=args.db)
    if prices.empty:
        print("No price data in range.")
        return 1
    signals = load_signal_matrix(
        args.signal, tickers, start=args.start, end=args.end, path=args.db
    )
    if signals.empty:
        print(f"No stored values for signal {args.signal!r} in range.")
        return 1

    result = run_backtest(
        prices,
        signals,
        top_n_by_signal(args.signal, args.top),
        frequency=args.rebalance,
        cost_bps=args.cost_bps,
        initial_capital=args.capital,
    )
    metrics = compute_metrics(result.equity, result.trades)

    first_day = result.equity["date"].iloc[0].strftime("%Y-%m-%d")
    last_day = result.equity["date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"Backtest: top-{args.top} by '{args.signal}', {args.rebalance} rebalance, "
          f"{first_day} -> {last_day}")
    print(f"Universe: {', '.join(prices.columns)} "
          f"({len(prices.columns)} stocks, cost {args.cost_bps:.1f} bps/trade)")
    if len(prices.columns) < 10:
        print("Note: this tiny universe is only for proving the engine runs — "
              "results on a handful of stocks are illustrative, not meaningful.")
    print()

    order = [
        "final_value", "total_return", "cagr", "volatility", "sharpe",
        "max_drawdown", "turnover", "win_rate", "num_trades",
    ]
    for key in order:
        m = metrics[key]
        value = m["value"]
        if key == "final_value":
            shown = f"${value:,.2f}"
        elif key in ("sharpe", "num_trades"):
            shown = f"{value:.2f}" if value == value else "n/a"
        else:
            shown = _fmt_pct(value)
        print(f"{m['label']}: {shown}")
        print(f"    What it means: {m['what']}")
        print(f"    Why it matters: {m['why']}")
        print(f"    What good looks like: {m['good']}")
        print()

    print("Honest reminder: a backtest shows what WOULD have happened, using "
          "today's knowledge of the past. It does not guarantee anything about "
          "the future — markets change, and costs in real life are often higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
