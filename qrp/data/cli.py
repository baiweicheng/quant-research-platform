"""CLI: ingest daily bars from Yahoo Finance into local SQLite.

Example:
    python -m qrp.data.cli --tickers AAPL,MSFT,NVDA --start 2023-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from qrp.data.yahoo import fetch_daily_bars, fetch_dividends, normalize_ticker
from qrp.data.store import DEFAULT_DB, count_bars, upsert_bars, upsert_dividends

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
        "reads the S&P 500 list checked into qrp/data/sp500_tickers.txt, "
        "plus past index members from qrp/data/sp500_history.csv so "
        "historical backtests have the price history they need "
        "(tickers with no downloadable history are skipped).",
    )
    p.add_argument(
        "--start",
        default="2013-01-01",
        help=(
            "First day to download, as YYYY-MM-DD (default: 2013-01-01 — "
            "enough history that the 253-trading-day signal lookbacks are "
            "fully populated from day one of a 2015 backtest)"
        ),
    )
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
        from qrp.data import sp500_universe

        tickers = sp500_universe(include_historical=True)
    else:
        tickers = [t for t in (s.strip() for s in args.tickers.split(",")) if t]

    failures: list[str] = []
    for i, raw in enumerate(tickers):
        try:
            ticker = normalize_ticker(raw)  # e.g. "brk.b" -> "BRK-B"
            df = fetch_daily_bars(ticker, args.start, args.end)
            n = upsert_bars(df, ticker, args.db)
            # Dividends ride along in the same chart response: creditable
            # cash on ex-dates. Stored idempotently, so re-running a
            # download backfills new ex-dates without duplicating anything.
            try:
                divs = fetch_dividends(ticker, args.start, args.end)
                n_div = upsert_dividends(divs, ticker, args.db)
            except Exception as div_exc:  # never fail the bar ingest for this
                n_div = 0
                print(f"{ticker}: dividends skipped ({div_exc})")
            total = count_bars(ticker, args.db)
            print(f"{ticker}: fetched {n} bars ({args.start}..{args.end}), "
                  f"{total} total in db, {n_div} dividends")
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
