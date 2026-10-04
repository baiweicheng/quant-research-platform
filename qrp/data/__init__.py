"""Market data ingestion from Yahoo Finance and local storage (Phase 1)."""


def sp500_tickers() -> list[str]:
    """Tickers of the S&P 500 list checked into the repo.

    Source: Wikipedia "List of S&P 500 companies". The retrieval date is
    recorded in the header comment of qrp/data/sp500_tickers.txt.
    One ticker per line; "#" lines are comments.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent / "sp500_tickers.txt"
    tickers: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line)
    return tickers


def sp500_universe(include_historical: bool = True) -> list[str]:
    """S&P 500 tickers for downloading: current plus past members.

    Past members matter because backtests need price history for stocks
    that were in the index at the time (many delisted tickers still serve
    history on Yahoo; ones that don't simply fail per-ticker and are
    skipped). Falls back to the current list when the membership table
    is absent.
    """
    current = sp500_tickers()
    if not include_historical:
        return current
    try:
        from qrp.data.sp500_history import (
            all_historical_tickers,
            history_available,
        )
    except ImportError:
        return current
    if not history_available():
        return current
    return sorted(set(current) | set(all_historical_tickers()))
