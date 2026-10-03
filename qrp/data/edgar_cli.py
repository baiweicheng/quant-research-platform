"""CLI: ingest company fundamentals from SEC EDGAR into local SQLite.

Fundamentals are the numbers from a company's official financial reports
(revenue, profit, assets, debt, ...), filed with the SEC. Free, no API key.

By default each run is an *incremental update*: only filings newer than
your last download are fetched. Pass --full-refresh to re-download
everything instead (much slower).

Example:
    python -m qrp.data.edgar_cli --tickers AAPL,MSFT,NVDA
    python -m qrp.data.edgar_cli --universe sp500 --full-refresh
"""

from __future__ import annotations

import argparse
import time

import requests

from qrp.data.edgar import (
    DEFAULT_DB,
    REQUEST_DELAY_S,
    USER_AGENT,
    clear_fetch_state,
    count_facts,
    fact_date_range,
    ingest_ticker,
    load_ticker_map,
)

DEMO_TICKERS = ["AAPL", "MSFT", "NVDA"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest company fundamentals from SEC EDGAR (free, no key)."
    )
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
    p.add_argument(
        "--full-refresh",
        action="store_true",
        help="Download every filing fresh, ignoring the record of your last "
        "download. Much slower — only needed if you suspect missing data. "
        "Without it, the default is an incremental update: only filings "
        "newer than your last download are fetched.",
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
    new_total = 0
    done = 0
    if args.full_refresh:
        # Forget every ticker's last-download record, so each ingest below
        # treats every filing as new (the same as incremental=False).
        clear_fetch_state(tickers, args.db)
    # Load once up front: the ticker map is ~1MB of JSON (parsing it per
    # ticker would redo that work hundreds of times), and one HTTP session
    # reuses the connection instead of re-negotiating TLS for every ticker.
    ticker_map = load_ticker_map()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for i, raw in enumerate(tickers):
        try:
            n = ingest_ticker(
                raw,
                args.db,
                session=session,
                ticker_map=ticker_map,
                incremental=not args.full_refresh,
            )
            new_total += n
            done += 1
            total = count_facts(raw, args.db)
            start, end = fact_date_range(raw, args.db)
            print(
                f"{raw.upper()}: stored {n} new facts "
                f"(periods {start}..{end}), {total} total in db"
            )
        except Exception as exc:  # keep going on per-ticker failures
            failures.append(raw)
            print(f"{raw.upper()}: FAILED ({exc})")
        if i < len(tickers) - 1:
            time.sleep(REQUEST_DELAY_S)

    mode = "Full refresh" if args.full_refresh else "Incremental update"
    ticker_word = "ticker" if done == 1 else "tickers"
    print(f"{mode} complete: {new_total} new facts across {done} {ticker_word}")

    if failures:
        print(f"Failed tickers: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
