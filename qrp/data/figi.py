"""Permanent security IDs via the free OpenFIGI API (P5: data quality).

Tickers change — companies rebrand, merge, or move listings — but a FIGI
(Financial Instrument Global Identifier) stays with the security forever.
Mapping every ticker to its FIGI once means the platform's history always
points at the right company even if the ticker printed on it changes.

The mapping comes from OpenFIGI's free API (https://api.openfigi.com),
no API key needed:

- Anonymous limit: **25 requests/minute, 10 tickers per request** (verified
  against OpenFIGI's published limits, October 2026). The client stays
  well under that (~2.6s between requests) and honors ``Retry-After``.
- Results are cached in the local ``security_ids`` table (see
  qrp.data.store) — each ticker is looked up once, ever, unless asked
  again. A full S&P 500 universe maps in ~50 requests, i.e. a few minutes
  the first time and zero afterwards.
- 429/5xx responses are retried with backoff; a ticker that can't be
  mapped is reported as failed, never faked.

If the API is unreachable, every function degrades to a clear message —
no invented data, ever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from qrp.data.store import (
    DEFAULT_DB,
    connect_db,
    get_security_id,
    load_security_ids,
    upsert_security_ids,
)
from qrp.data.yahoo import normalize_ticker

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"

#: Published anonymous limits (no API key). Verified October 2026.
ANON_REQUESTS_PER_MINUTE = 25
ANON_JOBS_PER_REQUEST = 10

#: Stay conservatively under the cap: ~23 requests/minute worst case.
MIN_INTERVAL_S = 2.6

MAX_RETRIES = 3
REQUEST_TIMEOUT_S = 30

USER_AGENT = "quant-research-platform/1.0 (personal research project)"


def _valid_figi(value: object) -> bool:
    """A FIGI is 12 characters, starts with BBG, and is alphanumeric."""
    return (
        isinstance(value, str)
        and len(value) == 12
        and value.startswith("BBG")
        and value.isalnum()
    )


@dataclass
class FigiMapping:
    """One ticker's permanent identifiers."""

    ticker: str
    figi: str  # exchange-level FIGI for this ticker on US exchanges
    composite_figi: str | None = None  # country-level: follows across exchanges
    share_class_figi: str | None = None  # share-class level
    name: str | None = None
    security_type: str | None = None

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "figi": self.figi,
            "composite_figi": self.composite_figi,
            "share_class_figi": self.share_class_figi,
            "name": self.name,
            "security_type": self.security_type,
            "mapped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


class OpenFigiClient:
    """Thin, polite client for OpenFIGI's /v3/mapping endpoint.

    ``post`` and ``sleep`` are injectable so tests can fake the network
    without touching it.
    """

    def __init__(
        self,
        post: Callable[..., requests.Response] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        min_interval_s: float = MIN_INTERVAL_S,
        max_retries: int = MAX_RETRIES,
        timeout_s: int = REQUEST_TIMEOUT_S,
    ) -> None:
        self._post = post or self._default_post
        self._sleep = sleep
        self._min_interval = min_interval_s
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._last_request_at = 0.0

    @staticmethod
    def _default_post(url, **kwargs) -> requests.Response:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        session.headers["Content-Type"] = "application/json"
        return session.post(url, **kwargs)

    def _throttled_post(self, jobs: list[dict]) -> requests.Response:
        """POST one batch, throttled and retried. Raises on hard failure."""
        wait = self._min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._last_request_at = time.monotonic()
            try:
                resp = self._post(
                    OPENFIGI_MAPPING_URL, json=jobs, timeout=self._timeout
                )
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep(min(2.0**attempt, 8.0))
                continue
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    pause = float(retry_after)
                except (TypeError, ValueError):
                    pause = min(2.0**attempt, 8.0)
                self._sleep(pause)
                last_exc = requests.HTTPError(
                    f"OpenFIGI rate limit (429); waited {pause}s"
                )
                continue
            if 500 <= resp.status_code < 600:
                last_exc = requests.HTTPError(
                    f"OpenFIGI server error ({resp.status_code})"
                )
                self._sleep(min(2.0**attempt, 8.0))
                continue
            if resp.status_code != 200:
                raise requests.HTTPError(
                    f"OpenFIGI returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
            return resp
        raise (
            last_exc
            if last_exc is not None
            else requests.HTTPError("OpenFIGI request failed")
        )

    def map_tickers(self, tickers: list[str]) -> dict[str, FigiMapping | None]:
        """Map tickers to FIGIs (US exchange). Missing/invalid -> None.

        Batches ``ANON_JOBS_PER_REQUEST`` tickers per HTTP request. A batch
        that fails after retries marks its tickers failed (None) and the
        mapping continues with the next batch — one bad batch never kills
        the whole run.
        """
        clean = [normalize_ticker(t) for t in tickers]
        out: dict[str, FigiMapping | None] = {}
        for i in range(0, len(clean), ANON_JOBS_PER_REQUEST):
            batch = clean[i : i + ANON_JOBS_PER_REQUEST]
            jobs = [
                {"idType": "TICKER", "idValue": t, "exchCode": "US"}
                for t in batch
            ]
            try:
                resp = self._throttled_post(jobs)
                payload = resp.json()
            except Exception:  # noqa: BLE001 - batch fails, tickers -> None
                for t in batch:
                    out[t] = None
                continue
            results = payload if isinstance(payload, list) else []
            for t, item in zip(batch, results):
                out[t] = self._parse_item(t, item)
            for t in batch[len(results) :]:
                out[t] = None
        # Fallback for share-class dashes: the platform stores Yahoo's
        # "BRK-B" spelling, but FIGI symbology uses "BRK/B". Retry the
        # misses with the slash form (one extra throttled batch at most).
        retry = {
            t: t.replace("-", "/") for t, m in out.items() if m is None and "-" in t
        }
        if retry:
            jobs = [
                {"idType": "TICKER", "idValue": v, "exchCode": "US"}
                for v in retry.values()
            ]
            try:
                resp = self._throttled_post(jobs)
                payload = resp.json()
            except Exception:  # noqa: BLE001 - fallback fails, stays None
                payload = []
            results = payload if isinstance(payload, list) else []
            for t, item in zip(retry.keys(), results):
                parsed = self._parse_item(t, item)
                if parsed is not None:
                    out[t] = parsed
        return out

    @staticmethod
    def _parse_item(ticker: str, item: object) -> FigiMapping | None:
        if not isinstance(item, dict):
            return None
        data = item.get("data")
        if not isinstance(data, list) or not data:
            return None  # includes {"error": "..."} responses
        first = data[0]
        if not isinstance(first, dict) or not _valid_figi(first.get("figi")):
            return None
        comp = first.get("compositeFIGI")
        sc = first.get("shareClassFIGI")
        return FigiMapping(
            ticker=ticker,
            figi=str(first["figi"]),
            composite_figi=str(comp) if _valid_figi(comp) else None,
            share_class_figi=str(sc) if _valid_figi(sc) else None,
            name=str(first["name"]) if first.get("name") else None,
            security_type=str(first["securityType"])
            if first.get("securityType")
            else None,
        )


@dataclass
class FigiRefreshResult:
    """Outcome of a mapping refresh, in plain English."""

    mapped: int
    already_cached: int
    failed: list[str] = field(default_factory=list)

    def message(self) -> str:
        parts = []
        if self.mapped:
            parts.append(f"Mapped {self.mapped} ticker(s) to permanent FIGI IDs")
        if self.already_cached:
            parts.append(
                f"{self.already_cached} already mapped (cached, no lookup needed)"
            )
        if self.failed:
            shown = ", ".join(self.failed[:5])
            more = f" and {len(self.failed) - 5} more" if len(self.failed) > 5 else ""
            parts.append(f"couldn't map {len(self.failed)}: {shown}{more}")
        if not parts:
            return "Nothing to map — every ticker already has a FIGI."
        text = "; ".join(parts) + "."
        if self.failed:
            text += " Unmapped tickers are usually delisted names or typos — safe to ignore."
        return text


def refresh_figi_mapping(
    db_path: str | Path = DEFAULT_DB,
    tickers: list[str] | None = None,
    client: OpenFigiClient | None = None,
    only_missing: bool = True,
) -> FigiRefreshResult:
    """Map tickers to FIGIs and cache the results in ``security_ids``.

    With ``only_missing=True`` (the default), tickers already in the table
    are skipped without any network call — a full universe maps once and
    never hits the API again.
    """
    db_path = Path(db_path)
    if tickers is None:
        with connect_db(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker"
            ).fetchall()
        tickers = [r[0] for r in rows]
    if not tickers:
        return FigiRefreshResult(mapped=0, already_cached=0)

    already = 0
    todo: list[str] = []
    if only_missing:
        for t in tickers:
            if get_security_id(t, db_path) is not None:
                already += 1
            else:
                todo.append(t)
    else:
        todo = list(tickers)

    if not todo:
        return FigiRefreshResult(mapped=0, already_cached=already)

    client = client or OpenFigiClient()
    try:
        mappings = client.map_tickers(todo)
    except Exception:  # noqa: BLE001 - total failure, nothing stored
        return FigiRefreshResult(
            mapped=0, already_cached=already, failed=list(todo)
        )
    rows = [m.as_dict() for m in mappings.values() if m is not None]
    if rows:
        upsert_security_ids(rows, db_path)
    failed = [t for t, m in mappings.items() if m is None]
    return FigiRefreshResult(
        mapped=len(rows), already_cached=already, failed=failed
    )


def figi_mapping_status(
    db_path: str | Path = DEFAULT_DB,
) -> dict:
    """How many stored tickers have a FIGI: for the dashboard."""
    db_path = Path(db_path)
    total = 0
    if db_path.exists():
        with connect_db(db_path) as conn:
            has_table = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table'"
                    " AND name='daily_bars'"
                ).fetchone()
                is not None
            )
            if has_table:
                (total,) = conn.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM daily_bars"
                ).fetchone()
    df = load_security_ids(path=db_path)
    mapped = len(df)
    unmapped: list[str] = []
    if total and not df.empty:
        have = set(df["ticker"])
        with connect_db(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker"
            ).fetchall()
        unmapped = [r[0] for r in rows if r[0] not in have]
    return {
        "mapped": mapped,
        "total": total,
        "unmapped": unmapped,
        "table": df,
    }
