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
