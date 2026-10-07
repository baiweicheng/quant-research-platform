"""Download daily OHLCV bars for US tickers from Yahoo Finance (free, no API key).

Uses the v8 chart API:
    https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={unix}&period2={unix}&interval=1d

Yahoo requires a session cookie before the chart API will answer; one is
obtained from https://fc.yahoo.com first.
"""

from __future__ import annotations

import re
import time
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


def normalize_ticker(ticker: str) -> str:
    """Uppercase and validate a US ticker symbol for Yahoo Finance.

    Yahoo's chart API expects share-class dots as dashes, so "BRK.B"
    becomes "BRK-B" (this is the form stored in the database too).
    """
    t = ticker.strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return t.replace(".", "-")


def _clean_ticker(ticker: str) -> str:
    """Uppercase and validate a US ticker symbol (see normalize_ticker)."""
    return normalize_ticker(ticker)


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


def _get_with_429_retry(
    sess: requests.Session,
    url: str,
    params: dict,
    timeout: int,
    max_retries: int = 3,
) -> requests.Response:
    """GET with polite exponential backoff on HTTP 429 (rate limited).

    Bursts of parallel requests can trip Yahoo's throttling; a short
    backoff (1s, 2s, 4s) almost always clears it. Raises the final
    response's HTTPError via the caller if retries are exhausted.
    """
    resp = sess.get(url, params=params, timeout=timeout)
    for attempt in range(max_retries):
        if resp.status_code != 429:
            return resp
        time.sleep(2**attempt)
        resp = sess.get(url, params=params, timeout=timeout)
    return resp


def _chart_payload(
    ticker: str,
    start: str | date,
    end: str | date,
    session: requests.Session | None = None,
    timeout: int = 30,
    events: str = "div",
) -> tuple[dict, requests.Session]:
    """Raw v8 chart API payload for one ticker (shared by bars/dividends).

    ``events`` selects which corporate-action streams come back
    ("div", "div|split", ...). Returns (result_dict, session). Raises on
    HTTP errors or empty/unparseable responses.
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
        "events": events,
    }
    url = CHART_URL.format(ticker=t)
    resp = _get_with_429_retry(sess, url, params, timeout)
    if resp.status_code == 401 and session is None:
        # Cookie may have expired; refresh once and retry.
        sess = _new_session()
        resp = _get_with_429_retry(sess, url, params, timeout)
    resp.raise_for_status()

    try:
        payload = resp.json()
        result = payload["chart"]["result"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Yahoo returned no usable data for {t}") from exc
    return result, sess


def fetch_daily_bars(
    ticker: str,
    start: str | date,
    end: str | date,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily bars for one US ticker from Yahoo Finance.

    Returns a DataFrame with columns Date,Open,High,Low,Close,Volume and
    dates as YYYY-MM-DD strings, oldest first. Close prices are raw
    (split-adjusted by Yahoo, but NOT dividend-adjusted) — realistic
    execution prices. Raises on HTTP errors or empty/unparseable responses.
    """
    result, _sess = _chart_payload(ticker, start, end, session, timeout)
    t = _clean_ticker(ticker)
    try:
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


def fetch_dividends(
    ticker: str,
    start: str | date,
    end: str | date,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Cash dividends for one US ticker from Yahoo Finance.

    Returns a DataFrame with columns ex_date (YYYY-MM-DD) and amount
    (dollars per share), oldest first — one row per dividend paid in the
    window. Empty DataFrame (same columns) when the ticker paid none.
    Raises on HTTP errors like fetch_daily_bars.
    """
    result, _sess = _chart_payload(ticker, start, end, session, timeout)
    events = (result.get("events") or {}).get("dividends") or {}
    rows = []
    for _ts, evt in events.items():
        try:
            amount = float(evt["amount"])
            ex = datetime.fromtimestamp(int(evt["date"]), tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (KeyError, TypeError, ValueError):
            continue
        if amount > 0:
            rows.append({"ex_date": ex, "amount": amount})
    df = pd.DataFrame(rows, columns=["ex_date", "amount"])
    if df.empty:
        return df
    return df.sort_values("ex_date").reset_index(drop=True)


def fetch_actions(
    ticker: str,
    start: str | date,
    end: str | date,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict:
    """Splits, dividends and adjusted closes for one ticker (P5).

    Used by the data-health checks to cross-check Yahoo's declared
    corporate actions against what the price history actually shows.
    Returns a dict::

        {
            "splits":    DataFrame(date, numerator, denominator),
            "dividends": DataFrame(ex_date, amount),
            "prices":    DataFrame(Date, Close, AdjClose),
        }

    ``prices["AdjClose"]`` is Yahoo's dividend-and-split-adjusted close
    (may be all-NaN if Yahoo doesn't return it — callers must cope).
    Raises on HTTP errors like fetch_daily_bars.
    """
    result, _sess = _chart_payload(
        ticker, start, end, session, timeout, events="div|split"
    )
    t = _clean_ticker(ticker)

    split_rows = []
    splits = (result.get("events") or {}).get("splits") or {}
    for _ts, evt in splits.items():
        try:
            num = float(evt["numerator"])
            den = float(evt["denominator"])
            ex = datetime.fromtimestamp(int(evt["date"]), tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (KeyError, TypeError, ValueError):
            continue
        if num > 0 and den > 0 and num != den:
            split_rows.append(
                {"date": ex, "numerator": num, "denominator": den}
            )
    splits_df = pd.DataFrame(split_rows, columns=["date", "numerator", "denominator"])
    if not splits_df.empty:
        splits_df = splits_df.sort_values("date").reset_index(drop=True)

    div_rows = []
    dividends = (result.get("events") or {}).get("dividends") or {}
    for _ts, evt in dividends.items():
        try:
            amount = float(evt["amount"])
            ex = datetime.fromtimestamp(int(evt["date"]), tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (KeyError, TypeError, ValueError):
            continue
        if amount > 0:
            div_rows.append({"ex_date": ex, "amount": amount})
    divs_df = pd.DataFrame(div_rows, columns=["ex_date", "amount"])
    if not divs_df.empty:
        divs_df = divs_df.sort_values("ex_date").reset_index(drop=True)

    prices_df = pd.DataFrame(columns=["Date", "Close", "AdjClose"])
    try:
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = (result.get("indicators") or {}).get("adjclose") or [{}]
        adjclose = adj[0].get("adjclose") if adj else None
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Yahoo returned no usable data for {t}") from exc
    if timestamps and quote.get("close"):
        n = len(timestamps)
        adj_vals = list(adjclose) if adjclose else []
        adj_vals = (adj_vals + [None] * n)[:n]  # pad/truncate to bar count
        prices_df = pd.DataFrame(
            {
                "Date": [
                    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    for ts in timestamps
                ],
                "Close": pd.to_numeric(quote.get("close"), errors="coerce"),
                "AdjClose": pd.to_numeric(adj_vals, errors="coerce"),
            }
        ).dropna(subset=["Close"])
        prices_df = prices_df.sort_values("Date").reset_index(drop=True)

    return {"splits": splits_df, "dividends": divs_df, "prices": prices_df}
