"""Download daily OHLCV bars for US tickers from Yahoo Finance (free, no API key).

Uses the v8 chart API:
    https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={unix}&period2={unix}&interval=1d

Yahoo requires a session cookie before the chart API will answer; one is
obtained from https://fc.yahoo.com first.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
COOKIE_URL = "https://fc.yahoo.com"
EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _clean_ticker(ticker: str) -> str:
    """Uppercase and validate a US ticker symbol."""
    t = ticker.strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return t


def _to_datetime(d: str | date) -> datetime:
    """Parse a YYYY-MM-DD date (or date object) as UTC midnight."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc)
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _new_session() -> requests.Session:
    """Build a session with the cookie Yahoo requires for the chart API."""
    sess = requests.Session()
    sess.headers["User-Agent"] = _BROWSER_UA
    sess.get(COOKIE_URL, timeout=30)
    return sess


def fetch_daily_bars(
    ticker: str,
    start: str | date,
    end: str | date,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily bars for one US ticker from Yahoo Finance.

    Returns a DataFrame with columns Date,Open,High,Low,Close,Volume and
    dates as YYYY-MM-DD strings, oldest first. Raises on HTTP errors or
    empty/unparseable responses.
    """
    t = _clean_ticker(ticker)
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end) + timedelta(days=1)  # period2 is exclusive
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    sess = session or _new_session()
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
    }
    resp = sess.get(CHART_URL.format(ticker=t), params=params, timeout=timeout)
    if resp.status_code == 401 and session is None:
        # Cookie may have expired; refresh once and retry.
        sess = _new_session()
        resp = sess.get(CHART_URL.format(ticker=t), params=params, timeout=timeout)
    resp.raise_for_status()

    try:
        payload = resp.json()
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        timestamps = result["timestamp"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Yahoo returned no usable data for {t}") from exc

    df = pd.DataFrame(
        {
            "Date": [
                datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                for ts in timestamps
            ],
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["Close"])  # skip days with no trading data
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
    df = df.sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Yahoo returned no rows for {t}")
    return df[EXPECTED_COLUMNS]
