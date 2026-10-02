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
from qrp.backtest.strategies import WEIGHTING_METHODS, weighted_top_n
from qrp.data.store import DEFAULT_DB
from qrp.portfolio.construction import METHOD_DESCRIPTIONS
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
        help="Hold the top N ranked stocks (default: 10)",
    )
    p.add_argument(
        "--weighting",
        default="equal",
        choices=sorted(WEIGHTING_METHODS),
        help=(
            "How to size positions: equal (even split), signal (more money to "
            "higher scores), inverse_vol (more money to calmer stocks). "
            "Default: equal"
        ),
    )
    p.add_argument(
        "--max-weight",
        type=float,
        default=0.20,
        help="No single stock may exceed this fraction of capital, e.g. 0.2 = 20%% (default: 0.2)",
    )
    p.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Never hold more than this many stocks (default: 20)",
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

    method = WEIGHTING_METHODS[args.weighting]
    vol_matrix = None
    if method == "inverse_volatility":
        # The stored low_volatility signal IS the negative of recent price
        # volatility, so negating it gives point-in-time volatilities.
        low_vol = load_signal_matrix(
            "low_volatility", tickers, start=args.start, end=args.end, path=args.db
        )
        if low_vol.empty:
            print(
                "No stored low_volatility values in range — needed for "
                "inverse-vol weighting. Run the signals CLI first."
            )
            return 1
        vol_matrix = -low_vol

    n = max(1, min(args.top, args.max_positions))
    if n != args.top:
        print(f"Note: holding top {n} — limited by --max-positions "
              f"{args.max_positions} (you asked for --top {args.top}).")
    strategy = weighted_top_n(
        args.signal, n, method=method, max_weight=args.max_weight,
        vol_matrix=vol_matrix,
    )

    result = run_backtest(
        prices,
        signals,
        strategy,
        frequency=args.rebalance,
        cost_bps=args.cost_bps,
        initial_capital=args.capital,
    )
    metrics = compute_metrics(result.equity, result.trades)

    first_day = result.equity["date"].iloc[0].strftime("%Y-%m-%d")
    last_day = result.equity["date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"Backtest: top-{n} by '{args.signal}' ({args.weighting} weighting), "
          f"{args.rebalance} rebalance, {first_day} -> {last_day}")
    print(f"Universe: {', '.join(prices.columns)} "
          f"({len(prices.columns)} stocks, cost {args.cost_bps:.1f} bps/trade, "
          f"max weight {args.max_weight:.0%})")
    if len(prices.columns) < 10:
        print("Note: this tiny universe is only for proving the engine runs — "
              "results on a handful of stocks are illustrative, not meaningful.")
    print(f"Weighting: {METHOD_DESCRIPTIONS[method]}")
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
