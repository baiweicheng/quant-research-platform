"""CLI: ingest daily bars from Yahoo Finance into local SQLite.

Example:
    python -m qrp.data.cli --tickers AAPL,MSFT,NVDA --start 2023-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from qrp.data.yahoo import fetch_daily_bars, normalize_ticker
from qrp.data.store import DEFAULT_DB, count_bars, upsert_bars

DEMO_TICKERS = ["AAPL", "MSFT", "NVDA"]
REQUEST_DELAY_S = 0.5  # be polite to the free endpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest daily bars from Yahoo Finance.")
    p.add_argument(
        "--tickers",
        default=",".join(DEMO_TICKERS),
        help="Comma-separated US tickers (default: %(default)s)",
    )
    p.add_argument(
        "--universe",
        choices=["sp500"],
        default=None,
        help="Use a named stock universe instead of --tickers: 'sp500' "
        "reads the S&P 500 list checked into qrp/data/sp500_tickers.txt.",
    )
    p.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.universe == "sp500":
        from qrp.data import sp500_tickers

        tickers = sp500_tickers()
    else:
        tickers = [t for t in (s.strip() for s in args.tickers.split(",")) if t]

    failures: list[str] = []
    for i, raw in enumerate(tickers):
        try:
            ticker = normalize_ticker(raw)  # e.g. "brk.b" -> "BRK-B"
            df = fetch_daily_bars(ticker, args.start, args.end)
            n = upsert_bars(df, ticker, args.db)
            total = count_bars(ticker, args.db)
            print(f"{ticker}: fetched {n} bars ({args.start}..{args.end}), "
                  f"{total} total in db")
        except Exception as exc:  # keep going on per-ticker failures
            failures.append(raw)
            print(f"{raw.upper()}: FAILED ({exc})")
        if i < len(tickers) - 1:
            time.sleep(REQUEST_DELAY_S)

    if failures:
        print(f"Failed tickers: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
