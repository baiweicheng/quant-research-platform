"""CLI: print a target portfolio for a signal on a given date.

A "target portfolio" is the answer to "how much of each stock should I hold
right now?" — ticker by ticker, as a percentage of your money, with the
safety caps (max weight per stock, max number of stocks, dust filter)
already applied.

Example:
    python -m qrp.portfolio.cli --signal momentum --method signal
    python -m qrp.portfolio.cli --signal trend --method inverse_vol --max-weight 0.15
    python -m qrp.portfolio.cli --signal momentum --date 2024-06-28 --tickers AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from qrp.backtest.engine import load_signal_matrix
from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.portfolio.construction import (
    METHOD_DESCRIPTIONS,
    build_target_weights,
)
from qrp.signals import default_registry
from qrp.signals.store import list_tickers

METHOD_CHOICES = {
    "equal": "equal_weight",
    "signal": "signal_weight",
    "inverse_vol": "inverse_volatility",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a target portfolio from signal scores."
    )
    p.add_argument(
        "--signal",
        default="momentum",
        help="Signal name to rank stocks by (default: momentum)",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Portfolio date YYYY-MM-DD (default: latest date with signal data)",
    )
    p.add_argument(
        "--tickers",
        default="all",
        help="Comma-separated US tickers, or 'all' in the db (default: all)",
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
        "--min-weight",
        type=float,
        default=0.01,
        help="Drop positions smaller than this fraction as dust, e.g. 0.01 = 1%% (default: 0.01)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


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
        tickers = [normalize_ticker(t) for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers found (is daily_bars populated?).")
        return 1

    matrix = load_signal_matrix(args.signal, tickers, path=args.db)
    if matrix.empty:
        print(
            f"No stored values for signal {args.signal!r}. "
            "Run the signals CLI first: python -m qrp.signals.cli"
        )
        return 1

    asof = pd.Timestamp(args.date) if args.date else matrix.index.max()
    # Latest signal value on or before the date, per ticker (never from the future).
    history = matrix.loc[:asof].ffill()
    if history.empty:
        print(f"No signal data on or before {asof.strftime('%Y-%m-%d')}.")
        return 1
    scores = history.iloc[-1].dropna().to_dict()
    if not scores:
        print(f"No scored stocks on or before {asof.strftime('%Y-%m-%d')}.")
        return 1
    day = history.index[-1].strftime("%Y-%m-%d")

    method = METHOD_CHOICES[args.method]
    volatilities = None
    if method == "inverse_volatility":
        low_vol = load_signal_matrix("low_volatility", tickers, path=args.db)
        if low_vol.empty:
            print(
                "No stored low_volatility values — needed for inverse-vol "
                "weighting. Run the signals CLI first."
            )
            return 1
        v_hist = (-low_vol).loc[:asof].ffill()
        if not v_hist.empty:
            volatilities = v_hist.iloc[-1].dropna().to_dict()

    portfolio = build_target_weights(
        scores,
        method=method,
        max_weight=args.max_weight,
        max_positions=args.max_positions,
        min_weight=args.min_weight,
        volatilities=volatilities,
    )

    print(f"Target portfolio — signal '{args.signal}' as of {day}")
    print(f"Method: {args.method}")
    print(f"  {METHOD_DESCRIPTIONS[method]}")
    print()
    if portfolio.weights:
        print(f"{'Ticker':<8}{'Signal score':>14}{'Target weight':>15}")
        print("-" * 38)
        for ticker, weight in sorted(
            portfolio.weights.items(), key=lambda kv: -kv[1]
        ):
            print(f"{ticker:<8}{scores[ticker]:>+14.4f}{weight:>14.2%}")
        print("-" * 38)
        print(f"{'CASH':<8}{'':>14}{portfolio.cash_weight:>14.2%}")
        print()
        print(
            f"What this means: if you invested $10,000 today, you would put "
            f"${10_000 * (1 - portfolio.cash_weight):,.0f} into "
            f"{len(portfolio.weights)} stock(s) as shown above and keep "
            f"${10_000 * portfolio.cash_weight:,.0f} in cash."
        )
    else:
        print("No positions — the whole portfolio would stay in cash.")
    for note in portfolio.notes:
        print(f"Note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
