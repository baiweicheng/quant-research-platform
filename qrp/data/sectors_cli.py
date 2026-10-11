"""CLI: refresh the GICS sector table from Wikipedia's S&P 500 list.

Example:
    python -m qrp.data.sectors_cli
    python -m qrp.data.sectors_cli --db /path/to/market.db
"""

from __future__ import annotations

import argparse

from qrp.data.sectors import DEFAULT_DB, SectorFetchError, refresh_sectors, status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refresh GICS sector data from Wikipedia (free, no key)."
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = refresh_sectors(args.db)
    except SectorFetchError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(
        f"Stored GICS sectors for {summary['tickers']} tickers "
        f"(fetched {summary['fetched_at']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
