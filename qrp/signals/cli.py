"""CLI: compute signals from stored daily bars and persist them to SQLite.

Bars are loaded with a warm-up buffer before --start so every signal's
lookback is satisfied inside the requested window.

Example:
    python -m qrp.signals.cli --signals momentum,trend --tickers AAPL,MSFT
    python -m qrp.signals.cli --tickers all --start 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from qrp.data.store import DEFAULT_DB
from qrp.signals import default_registry
from qrp.signals.store import count_signals, list_tickers, load_bars, upsert_signals

# Calendar-day buffer loaded before --start so signal lookbacks (max 253
# trading days ~ 12 months) are warm. 400 days comfortably covers it.
HISTORY_BUFFER_DAYS = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute signals from stored bars.")
    p.add_argument(
        "--signals",
        default="all",
        help="Comma-separated signal names, or 'all' (default: all)",
    )
    p.add_argument(
        "--tickers",
        default="all",
        help="Comma-separated US tickers, or 'all' tickers in the db (default: all)",
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = default_registry()

    if args.signals.strip().lower() == "all":
        signals = [registry.get(name) for name, _ in registry.list_signals()]
    else:
        signals = []
        for raw in args.signals.split(","):
            name = raw.strip()
            if not name:
                continue
            try:
                signals.append(registry.get(name))
            except KeyError:
                available = ", ".join(n for n, _ in registry.list_signals())
                print(f"Unknown signal {name!r}. Available: {available}")
                return 1
    if not signals:
        print("No signals selected.")
        return 1

    if args.tickers.strip().lower() == "all":
        tickers = list_tickers(args.db)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers found (is daily_bars populated?).")
        return 1

    # Warm-up buffer so lookbacks are satisfied from the first stored date.
    load_start = None
    if args.start:
        load_start = (
            date.fromisoformat(args.start) - timedelta(days=HISTORY_BUFFER_DAYS)
        ).isoformat()
    start_ts = pd.Timestamp(args.start) if args.start else None

    for ticker in tickers:
        df = load_bars(ticker, start=load_start, end=args.end, path=args.db)
        if df.empty:
            print(f"{ticker}: no bars in db, skipped")
            continue
        for sig in signals:
            values = sig.compute(df)
            if start_ts is not None:
                values = values[values.index >= start_ts]
            n = upsert_signals(values, ticker, sig.name, args.db)
            total = count_signals(ticker, sig.name, args.db)
            latest = values.dropna()
            if not latest.empty:
                last_day = latest.index[-1].strftime("%Y-%m-%d")
                print(
                    f"{ticker} {sig.name}: stored {n} values "
                    f"({total} total), latest {latest.iloc[-1]:+.4f} on {last_day}"
                )
            else:
                print(f"{ticker} {sig.name}: no values in range, stored 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
