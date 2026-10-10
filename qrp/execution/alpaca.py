"""Alpaca paper-trading adapter (Phase P2).

Talks to Alpaca's REST API directly with ``requests`` (no third-party SDK —
the repo pins only requests/pandas/numpy/streamlit). PAPER ONLY: the
default base URL is Alpaca's paper endpoint, and the dashboard never offers
any live-trading UI. The adapter *can* be pointed elsewhere (its tests do),
but constructing it against Alpaca's live server raises unless the caller
explicitly opts in with ``allow_live=True`` — tripping into live trading
by accident is meant to be impossible.

How our nightly workflow maps onto Alpaca
----------------------------------------
The platform stages orders for the next market open. Every staged order
goes out as a plain day market order (``type="market"``,
``time_in_force="day"``), fractional share quantities included — no
whole-share rounding (Alpaca accepts fractional quantities only on day
market orders; quantities here carry at most 4 decimal places). Staged
while the market is closed, orders are sent to Alpaca right away:
Alpaca holds day orders submitted after the close and releases them at
the next trading day's open, so they fill as market orders starting
right at the open — usually within a whisker of the official opening
print, but not the auction-guaranteed print the platform used before.

Market-hours gate: a day market order sent while the market is OPEN
would fill immediately, and this platform stages for the NEXT open,
never mid-day. So both staging paths (the dashboard send and the
staging-queue flush) check Alpaca's clock first: while the market is
open — or the clock can't be read at all (fail closed) — orders stay in
the local staging queue on the user's computer and are sent on the next
app visit after a market close, to fill at the open that follows.

The adapter's ``place_order`` default stays ``time_in_force="auto"``
for non-staging callers (whole shares -> market-on-open "opg";
fractional shares -> day market order); pass an explicit
``time_in_force`` to override. Live trading is never offered: see the
base-URL guard in ``AlpacaBroker``.

Key storage
-----------
Paper API keys live in ``data/alpaca_paper_keys.json`` (next to the market
database), written with mode 0600 and listed in ``.gitignore``. They are
never logged, never committed, and never leave the user's machine — the
dashboard's Brokers page is the only place they are entered.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from qrp.data.store import DEFAULT_DB, connect_db
from qrp.data.yahoo import normalize_ticker
from qrp.execution.broker import AccountInfo, Broker, BrokerError, OrderInfo
from qrp.risk.guards import RiskConfig

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

KEYS_FILENAME = "alpaca_paper_keys.json"
REASONS_FILENAME = "alpaca_order_reasons.json"
QUEUE_FILENAME = "alpaca_staging_queue.json"

# A queued staging order is dropped once it has waited this many calendar
# days without Alpaca accepting it (each failure is shown in plain English;
# the next staging run recomputes orders from scratch anyway). Time-based,
# not attempt-based: flush runs on every Paper-page visit, and a plain
# attempt counter would drop a legitimate order after a few casual daytime
# visits before the evening acceptance window even opens.
QUEUE_MAX_WAIT_DAYS = 3

_VALID_TIF = {"day", "gtc", "opg", "cls", "ioc", "fok", "auto"}

# Alpaca order statuses -> our simplified lifecycle.
_STATUS_MAP = {
    "new": "open",
    "accepted": "open",
    "pending_new": "open",
    "pending_cancel": "open",
    "pending_replace": "open",
    "partially_filled": "open",
    "calculated": "open",
    "held": "open",
    "filled": "filled",
    "canceled": "canceled",
    "expired": "expired",
    "rejected": "rejected",
    "done_for_day": "expired",
    "replaced": "canceled",
}


class AlpacaError(BrokerError):
    """Something went wrong talking to Alpaca.

    The message is always plain English and never contains API keys.
    ``status_code`` is the HTTP status when the failure came from an
    Alpaca response, else None (network trouble, bad input, ...).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Key storage
# ---------------------------------------------------------------------------


def default_keys_path(db_path: str | Path = DEFAULT_DB) -> Path:
    """Where the paper API keys live: next to the market database."""
    return Path(db_path).parent / KEYS_FILENAME


def _reasons_path(keys_path: str | Path | None = None) -> Path:
    base = Path(keys_path) if keys_path else default_keys_path()
    return base.parent / REASONS_FILENAME


def _queue_path(keys_path: str | Path | None = None) -> Path:
    base = Path(keys_path) if keys_path else default_keys_path()
    return base.parent / QUEUE_FILENAME


def save_paper_keys(
    key_id: str, secret: str, keys_path: str | Path | None = None
) -> Path:
    """Save paper API keys to disk, readable only by this user (mode 0600).

    Raises:
        AlpacaError with a plain-English message when either value is blank
        or the file cannot be written.
    """
    key_id = (key_id or "").strip()
    secret = (secret or "").strip()
    if not key_id or not secret:
        raise AlpacaError(
            "Both the paper API key ID and the paper API secret are needed — "
            "one of them was blank."
        )
    path = Path(keys_path) if keys_path else default_keys_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # os.open with an explicit mode sets 0600 atomically at creation;
        # the chmod covers files that already existed with wider permissions.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"key_id": key_id, "secret": secret}, fh)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise AlpacaError(
            f"Could not save the API keys to {path}: {exc.strerror or exc}."
        )
    return path


def load_paper_keys(
    keys_path: str | Path | None = None,
) -> tuple[str, str] | None:
    """(key_id, secret), or None when no usable keys are saved."""
    path = Path(keys_path) if keys_path else default_keys_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    key_id = str(data.get("key_id", "") or "").strip()
    secret = str(data.get("secret", "") or "").strip()
    if not key_id or not secret:
        return None
    return key_id, secret


def delete_paper_keys(keys_path: str | Path | None = None) -> bool:
    """Remove the saved keys. Returns True when a file was removed."""
    path = Path(keys_path) if keys_path else default_keys_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


def paper_keys_configured(keys_path: str | Path | None = None) -> bool:
    """True when a usable key pair is saved on this machine."""
    return load_paper_keys(keys_path) is not None


def record_order_reason(
    order_id: str, reason: str, keys_path: str | Path | None = None
) -> None:
    """Remember the plain-English reason for an Alpaca order id.

    Best-effort sidecar (Alpaca itself has no "reason" field): it never
    raises — a failed note must not break trading.
    """
    if not order_id or not reason:
        return
    path = _reasons_path(keys_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            notes = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notes, dict):
                notes = {}
        except (OSError, ValueError):
            notes = {}
        notes[str(order_id)] = reason
        # Keep the file small: only the most recent few hundred reasons.
        if len(notes) > 500:
            notes = dict(list(notes.items())[-500:])
        path.write_text(json.dumps(notes), encoding="utf-8")
    except OSError:
        pass


def lookup_order_reason(
    order_id: str, keys_path: str | Path | None = None
) -> str:
    """The recorded reason for an Alpaca order id, or "" when unknown."""
    if not order_id:
        return ""
    try:
        notes = json.loads(_reasons_path(keys_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(notes, dict):
        return ""
    return str(notes.get(str(order_id), "") or "")


# ---------------------------------------------------------------------------
# Staging queue
# ---------------------------------------------------------------------------


def load_staging_queue(
    keys_path: str | Path | None = None,
) -> list[dict]:
    """Orders staged locally that haven't been sent to Alpaca yet.

    Each entry: {ticker, side, shares, reason, staged_at, attempts,
    last_error, planned_open}. The queue is how "keep the order staged
    for the next valid open" works: when an order can't be sent yet —
    Alpaca rejected it at submission, or the market is open and the
    market-hours gate holds it on this computer (see
    flush_staging_queue) — the order waits here instead of being
    silently converted into an immediate fill, and the dashboard
    retries it on every visit after a market close. ``planned_open``
    (YYYY-MM-DD, when known) records the morning the order was
    originally meant to fill, so a late send can say so.
    """
    try:
        data = json.loads(_queue_path(keys_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def save_staging_queue(
    entries: list[dict], keys_path: str | Path | None = None
) -> None:
    """Persist the staging queue (best-effort; never raises)."""
    path = _queue_path(keys_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")
    except OSError:
        pass


def queue_staged_order(
    ticker: str,
    side: str,
    shares: float,
    reason: str = "",
    keys_path: str | Path | None = None,
    planned_open: str = "",
) -> None:
    """Keep a staging order waiting locally until it can be sent.

    ``planned_open`` (YYYY-MM-DD, "" when unknown) records the morning
    the order was staged to fill — if it ends up going out for a later
    open because the app wasn't open after a close, the flush note
    says so.
    """
    entries = load_staging_queue(keys_path)
    entries.append(
        {
            "ticker": normalize_ticker(ticker),
            "side": (side or "").strip().upper(),
            "shares": float(shares),
            "reason": reason or "",
            "staged_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "attempts": 0,
            "last_error": "",
            "planned_open": str(planned_open or ""),
        }
    )
    save_staging_queue(entries, keys_path)


def drop_queued_order(
    index: int, keys_path: str | Path | None = None
) -> bool:
    """Discard one queued staging order. Returns True when it existed."""
    entries = load_staging_queue(keys_path)
    if 0 <= index < len(entries):
        entries.pop(index)
        save_staging_queue(entries, keys_path)
        return True
    return False


def _queued_at(entry: dict) -> datetime | None:
    """When a queue entry was first staged (UTC), or None when unknown."""
    raw = str(entry.get("staged_at") or "")
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _may_keep_waiting(entry: dict) -> bool:
    """True when a rejected queue entry may keep waiting for the next open.

    Entries are dropped only after QUEUE_MAX_WAIT_DAYS days without being
    sent — never just because the dashboard was visited a few times.
    Entries written before timestamps existed are backfilled as new.
    """
    queued_at = _queued_at(entry)
    if queued_at is None:
        entry["staged_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return True
    return datetime.now(timezone.utc) - queued_at <= timedelta(
        days=QUEUE_MAX_WAIT_DAYS
    )


# ---------------------------------------------------------------------------
# Market clock helpers (shared by the staging gate and the queue flush)
# ---------------------------------------------------------------------------


def fetch_market_clock(broker) -> dict | None:
    """The broker's market clock, or None when it can't be read.

    Fail-closed by design: callers treat None as "not safe to send",
    never as "market closed".
    """
    get_clock = getattr(broker, "get_clock", None)
    if get_clock is None:
        return None
    try:
        clock = get_clock()
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return None
    return clock if isinstance(clock, dict) else None


def clock_market_open(clock: dict | None) -> bool | None:
    """True/False from a clock dict's ``is_open``; None when unreadable."""
    if not isinstance(clock, dict):
        return None
    is_open = clock.get("is_open")
    return is_open if isinstance(is_open, bool) else None


def clock_next_open_date(clock: dict | None) -> str:
    """YYYY-MM-DD of the clock's next market open ("" when unknown)."""
    if not isinstance(clock, dict):
        return ""
    raw = str(clock.get("next_open") or "")
    return raw[:10] if len(raw) >= 10 else ""


def flush_staging_queue(broker) -> dict:
    """Try to send every queued staging order to Alpaca as a day order.

    Called on every Paper trading page visit (after a market close, the
    next interaction sends the waiting orders — there is no background
    sender). Returns {"sent": n, "waiting": m, "held": bool,
    "target_open": "YYYY-MM-DD" or "", "notes": [...]} with
    plain-English notes.

    Market-hours gate: while the market is OPEN — or the clock can't be
    read (fail closed) — nothing is sent; every entry stays in the
    local queue on this computer, and the note names the morning the
    orders now target (the clock's next open, recomputed on every
    visit — never a stale date). When the market is closed, orders go
    out as plain day market orders, which Alpaca holds for the next
    open. An entry whose send lands on a later open than the one it was
    staged for gets a note saying it went out later than planned.

    Entries that Alpaca keeps rejecting are dropped only after
    QUEUE_MAX_WAIT_DAYS days of waiting (the next staging run recomputes
    orders from scratch anyway) — never merely because the page was
    visited a few times.
    """
    keys_path = broker._keys_path
    entries = load_staging_queue(keys_path)
    if not entries:
        return {"sent": 0, "waiting": 0, "held": False,
                "target_open": "", "notes": []}
    clock = fetch_market_clock(broker)
    is_open = clock_market_open(clock)
    target_open = clock_next_open_date(clock)
    if is_open is not False:
        # Market open (or clock unreadable): hold everything locally.
        count = len(entries)
        plural = "order" if count == 1 else "orders"
        if is_open is None:
            note = (
                f"Couldn't check whether the market is open, so "
                f"{count} waiting {plural} stayed held on this "
                "computer to be safe — they'll be sent on the next "
                "visit when the market is known to be closed."
            )
        else:
            when = (
                f"for the {target_open} open" if target_open
                else "for the next market open"
            )
            note = (
                f"The market is open right now, so {count} waiting "
                f"{plural} stayed held on this computer — they'll be "
                f"sent after today's close, {when}. Reopen this app any "
                "time after the 4 PM close to send them."
            )
        return {"sent": 0, "waiting": count, "held": True,
                "target_open": target_open, "notes": [note]}
    try:
        open_keys = {(o.ticker, o.side) for o in broker.list_orders("open")}
    except AlpacaError:
        open_keys = set()
    sent = 0
    kept: list[dict] = []
    notes: list[str] = []
    for entry in entries:
        key = (entry.get("ticker"), entry.get("side"))
        if key in open_keys:
            # Already open at Alpaca (e.g. staged twice) — not needed. Said
            # out loud rather than silently skipped: a different-size
            # top-up order would also match here, so the note names exactly
            # what was skipped.
            notes.append(
                f"Skipped the queued {entry.get('side')} order for "
                f"{entry.get('ticker')} — an open {entry.get('side')} order "
                f"for {entry.get('ticker')} already exists at Alpaca."
            )
            continue
        try:
            broker.place_order(
                str(entry.get("ticker")),
                str(entry.get("side")),
                float(entry.get("shares")),
                reason=str(entry.get("reason", "") or ""),
                time_in_force="day",
            )
        except AlpacaError as exc:
            entry["attempts"] = int(entry.get("attempts", 0) or 0) + 1
            entry["last_error"] = str(exc)
            if _may_keep_waiting(entry):
                kept.append(entry)
            else:
                notes.append(
                    f"Gave up on the queued {entry.get('side')} order for "
                    f"{entry.get('ticker')} — it waited "
                    f"{QUEUE_MAX_WAIT_DAYS} days without Alpaca accepting "
                    f"it. Last problem: {exc}"
                )
        except (TypeError, ValueError):
            # Corrupt entry — drop it rather than retrying forever.
            notes.append(
                f"Dropped a garbled queued order for {entry.get('ticker')}."
            )
        else:
            sent += 1
            open_keys.add(key)
            planned = str(entry.get("planned_open") or "")
            if planned and target_open and target_open > planned:
                # The app simply wasn't open after a close in time —
                # say so, with both mornings named.
                notes.append(
                    f"{entry.get('side')} {entry.get('ticker')}: sent "
                    "later than planned because the app wasn't open "
                    f"after a close — it was staged for the {planned} "
                    f"open and will fill at the {target_open} open "
                    "instead."
                )
    save_staging_queue(kept, keys_path)
    return {"sent": sent, "waiting": len(kept), "held": False,
            "target_open": target_open, "notes": notes}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _map_status(alpaca_status: str) -> str:
    return _STATUS_MAP.get(str(alpaca_status or "").lower(), "open")


def _to_float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _require_dict(data: object, what: str) -> dict:
    """An Alpaca JSON object, or a plain-English AlpacaError — never assert."""
    if isinstance(data, dict):
        return data
    raise AlpacaError(
        f"Alpaca's paper server answered {what} in a format we didn't "
        "expect. Try again in a bit."
    )


def _require_float(value: object, field: str) -> float:
    """A numeric field from Alpaca, or a plain-English AlpacaError."""
    parsed = _to_float_or_none(value)
    if parsed is None:
        raise AlpacaError(
            f"Alpaca's paper server sent a {field} value we couldn't read. "
            "Try again in a bit."
        )
    return parsed


def _qty_str(shares: float) -> str:
    """Share quantity as Alpaca wants it: a decimal string, at most 9
    decimal places (Alpaca rejects more). Trailing zeros are stripped, so
    whole shares stay whole ("10", not "10.000000000")."""
    return f"{shares:.9f}".rstrip("0").rstrip(".") or "0"


class AlpacaBroker(Broker):
    """Trade the Alpaca PAPER account through its REST API.

    Args:
        api_key / api_secret: the PAPER key pair (from the Brokers page).
        base_url: Alpaca server; defaults to the paper endpoint. Passing
            Alpaca's live server raises unless ``allow_live=True``.
        timeout: seconds to wait for Alpaca per request.
        time_in_force: default for placed orders — "auto" (whole shares ->
            market-on-open for the opening auction; fractional shares ->
            plain day market order, the only TIF Alpaca accepts with
            fractions; both fill at the next open when staged at night).
            An explicit TIF ("day", "opg", ...) overrides the automatic
            choice.
        db_path: local market database — used for ``last_price`` when
            sizing orders (fills still happen at Alpaca's prices) and to
            locate the key/reason files.
        keys_path: override for the key/reason file location (tests).
        risk_config: platform safety guards applied when previewing orders
            (defaults to the standard RiskConfig).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = PAPER_BASE_URL,
        timeout: float = 20.0,
        time_in_force: str = "auto",
        db_path: str | Path = DEFAULT_DB,
        keys_path: str | Path | None = None,
        risk_config: RiskConfig | None = None,
        allow_live: bool = False,
    ) -> None:
        base = (base_url or "").rstrip("/")
        if base == LIVE_BASE_URL and not allow_live:
            raise AlpacaError(
                "That is Alpaca's LIVE trading server, and live trading is "
                "not enabled in this app — paper trading only, by design. "
                "Use the paper endpoint instead."
            )
        if not api_key or not api_secret:
            raise AlpacaError(
                "An Alpaca API key pair is needed — save your paper keys on "
                "the Brokers page first."
            )
        tif = (time_in_force or "auto").lower()
        if tif not in _VALID_TIF:
            raise AlpacaError(
                f"Unknown time-in-force {time_in_force!r} — use one of "
                f"{sorted(_VALID_TIF)}."
            )
        self._key = api_key
        self._secret = api_secret
        self._base_url = base or PAPER_BASE_URL
        self._timeout = float(timeout)
        self._tif = tif
        self._db = Path(db_path)
        self._keys_path = Path(keys_path) if keys_path else default_keys_path(
            self._db
        )
        self.risk_config = risk_config or RiskConfig()
        self._session = requests.Session()
        # Asset fractionability, learned once per ticker (see
        # is_fractionable) so a staging run doesn't re-ask per order.
        self._fractionable_cache: dict[str, bool | None] = {}

    # -- transport ------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _translate_error(self, resp: requests.Response) -> AlpacaError:
        code = resp.status_code
        try:
            body = resp.json()
            message = str(body.get("message", "") or "").strip()
        except ValueError:
            message = ""
        lower = message.lower()
        if code == 401:
            return AlpacaError(
                "Alpaca didn't recognize these API keys. Double-check the "
                "key ID and secret on the Brokers page — make sure they are "
                "PAPER keys from your Alpaca paper account, not live keys.",
                code,
            )
        if code == 403:
            if "buying power" in lower or "insufficient" in lower:
                return AlpacaError(
                    "Alpaca refused the order: not enough buying power in "
                    "the paper account for it.",
                    code,
                )
            detail = f" Alpaca said: {message}." if message else ""
            return AlpacaError(
                "Alpaca refused the request." + detail, code
            )
        if code == 404:
            return AlpacaError(
                "Alpaca has no such order — it may already be filled, "
                "canceled, or expired.",
                code,
            )
        if code == 422:
            detail = message or "the order didn't meet Alpaca's rules"
            hint = (
                " (Alpaca only allows fractional shares on plain day orders.)"
                if "fraction" in lower
                else ""
            )
            return AlpacaError(
                f"Alpaca rejected the order: {detail}.{hint}", code
            )
        if code == 429:
            return AlpacaError(
                "Alpaca is rate-limiting requests right now. Wait a minute "
                "and try again.",
                code,
            )
        if code >= 500:
            return AlpacaError(
                "Alpaca's paper trading server is having trouble. Try again "
                "in a bit.",
                code,
            )
        if message:
            return AlpacaError(f"Alpaca returned an error: {message}.", code)
        return AlpacaError(
            f"Alpaca returned an unexpected error (HTTP {code}). Try again "
            "in a bit.",
            code,
        )

    def _request(
        self, method: str, path: str, **kwargs: object
    ) -> object:
        url = self._base_url + path
        try:
            resp = self._session.request(
                method, url, headers=self._headers(),
                timeout=self._timeout, **kwargs,  # type: ignore[arg-type]
            )
        except requests.ConnectionError:
            raise AlpacaError(
                "Could not reach Alpaca's paper trading server. Check your "
                "internet connection and try again."
            )
        except requests.Timeout:
            raise AlpacaError(
                "Alpaca's paper trading server took too long to answer. "
                "Try again in a bit."
            )
        except requests.RequestException:
            raise AlpacaError(
                "Something went wrong talking to Alpaca's paper server. "
                "Try again in a bit."
            )
        if resp.status_code >= 400:
            raise self._translate_error(resp)
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            raise AlpacaError(
                "Alpaca's paper server answered in a format we didn't "
                "expect. Try again in a bit."
            )

    # -- Broker interface ------------------------------------------------

    def get_account(self) -> AccountInfo:
        data = _require_dict(
            self._request("GET", "/v2/account"), "the account"
        )
        equity = _to_float_or_none(
            data.get("equity") or data.get("portfolio_value")
        )
        return AccountInfo(
            equity=equity if equity is not None else 0.0,
            cash=_require_float(data.get("cash") or 0.0, "cash"),
            buying_power=_require_float(
                data.get("buying_power") or 0.0, "buying power"
            ),
        )

    def get_positions(self) -> dict[str, float]:
        try:
            data = self._request("GET", "/v2/positions")
        except AlpacaError as exc:
            if exc.status_code == 404:
                return {}
            raise
        if not isinstance(data, list):
            return {}
        out: dict[str, float] = {}
        for pos in data:
            if not isinstance(pos, dict):
                continue
            qty = _to_float_or_none(pos.get("qty"))
            if qty:
                out[str(pos.get("symbol", "")).upper()] = qty
        return out

    def get_position_details(self) -> list[dict]:
        """Richer position rows for the dashboard table.

        Each row: symbol, qty, current_price, market_value,
        unrealized_pl (all floats; missing values become 0.0).
        """
        try:
            data = self._request("GET", "/v2/positions")
        except AlpacaError as exc:
            if exc.status_code == 404:
                return []
            raise
        if not isinstance(data, list):
            return []
        rows = []
        for pos in data:
            if not isinstance(pos, dict):
                continue
            qty = _to_float_or_none(pos.get("qty")) or 0.0
            if not qty:
                continue
            rows.append(
                {
                    "symbol": str(pos.get("symbol", "")).upper(),
                    "qty": qty,
                    "current_price": _to_float_or_none(
                        pos.get("current_price")
                    ) or 0.0,
                    "market_value": _to_float_or_none(
                        pos.get("market_value")
                    ) or 0.0,
                    "unrealized_pl": _to_float_or_none(
                        pos.get("unrealized_pl")
                    ) or 0.0,
                }
            )
        return rows

    def get_clock(self) -> dict:
        """Market clock from Alpaca: is_open, next_open, next_close,
        timestamp (raw values as Alpaca returns them)."""
        data = self._request("GET", "/v2/clock")
        return data if isinstance(data, dict) else {}

    def get_asset(self, ticker: str) -> dict:
        """Alpaca's asset record for a ticker (``fractionable`` etc.).

        Raises AlpacaError when Alpaca can't produce the record (e.g.
        an unknown symbol — Alpaca answers 404).
        """
        t = normalize_ticker(ticker)
        from urllib.parse import quote

        data = self._request("GET", "/v2/assets/" + quote(t, safe=""))
        return _require_dict(data, f"the asset record for {t}")

    def is_fractionable(self, ticker: str) -> bool | None:
        """Whether Alpaca allows fractional shares of a ticker.

        Reads the asset's ``fractionable`` flag (Alpaca's fractional-
        trading docs: a fractional order for a non-fractionable asset
        is rejected). Cached on this broker — one asset lookup per
        ticker per staging run. None when the check itself failed
        (network trouble, unknown symbol): callers fall back to whole
        shares with a note rather than sending an order Alpaca might
        refuse.
        """
        t = normalize_ticker(ticker)
        if t in self._fractionable_cache:
            return self._fractionable_cache[t]
        try:
            asset = self.get_asset(t)
        except AlpacaError:
            result: bool | None = None
        else:
            result = bool(asset.get("fractionable"))
        self._fractionable_cache[t] = result
        return result

    def place_order(
        self,
        ticker: str,
        side: str,
        shares: float,
        reason: str = "",
        time_in_force: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> OrderInfo:
        ticker = normalize_ticker(ticker)
        side_u = (side or "").strip().upper()
        if side_u not in ("BUY", "SELL"):
            raise AlpacaError(
                f"Cannot place an order for {ticker}: side must be BUY or "
                f"SELL, not {side!r}."
            )
        try:
            qty = float(shares)
        except (TypeError, ValueError):
            qty = 0.0
        if not qty > 0:
            raise AlpacaError(
                f"Cannot place an order for {ticker}: shares must be "
                "positive."
            )
        tif = (time_in_force or self._tif).lower()
        if tif == "auto":
            # Whole shares ride the opening auction (true market-on-open);
            # fractional shares ride a plain day order, the only TIF Alpaca
            # accepts with fractions. Staged at night, both fill at the
            # next market open.
            tif = "opg" if qty.is_integer() else "day"
        if tif not in _VALID_TIF or tif == "auto":
            raise AlpacaError(
                f"Unknown time-in-force {time_in_force!r} — use one of "
                f"{sorted(_VALID_TIF)}."
            )
        payload = {
            "symbol": ticker,
            # Alpaca wants qty as a decimal string (max 9 decimals);
            # full precision passes through untouched otherwise.
            "qty": _qty_str(qty),
            "side": "buy" if side_u == "BUY" else "sell",
            "type": "market",
            "time_in_force": tif,
            # P6 duplicate protection: the caller passes the deterministic
            # "rebalance-YYYYMMDD-TICKER-side" id; Alpaca dedupes
            # server-side on it, so a retried batch cannot double-trade.
            # A random id preserves the old behavior when none is given.
            "client_order_id": client_order_id or f"qrp-{uuid.uuid4().hex[:16]}",
        }
        data = _require_dict(
            self._request("POST", "/v2/orders", json=payload),
            "the order confirmation",
        )
        info = self._to_order_info(data)
        if reason:
            record_order_reason(info.id, reason, self._keys_path)
            info = replace(info, reason=reason)
        return info

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._request("DELETE", f"/v2/orders/{order_id}")
        except AlpacaError as exc:
            if exc.status_code == 404:
                return False
            raise
        return True

    def list_orders(self, status: str = "open") -> list[OrderInfo]:
        status = (status or "open").lower()
        if status == "open":
            alpaca_status, wanted = "open", {"open"}
        elif status == "all":
            alpaca_status, wanted = "all", set()
        elif status in ("filled", "canceled", "rejected", "expired"):
            alpaca_status, wanted = "closed", {status}
        else:
            raise AlpacaError(
                f"Unknown order status {status!r} — use open, filled, "
                "canceled, rejected, expired, or all."
            )
        data = self._request(
            "GET",
            "/v2/orders",
            params={
                "status": alpaca_status,
                "limit": 500,
                "direction": "desc",
            },
        )
        if not isinstance(data, list):
            return []
        orders = [self._to_order_info(o) for o in data if isinstance(o, dict)]
        if wanted:
            orders = [o for o in orders if o.status in wanted]
        return orders

    def get_order(self, order_id: str) -> OrderInfo | None:
        try:
            data = self._request("GET", f"/v2/orders/{order_id}")
        except AlpacaError as exc:
            if exc.status_code == 404:
                return None
            raise
        if not isinstance(data, dict):
            return None
        return self._to_order_info(data)

    # -- helpers ----------------------------------------------------------

    def _to_order_info(self, data: dict) -> OrderInfo:
        order_id = str(data.get("id", ""))
        return OrderInfo(
            id=order_id,
            ticker=str(data.get("symbol", "")).upper(),
            side="BUY"
            if str(data.get("side", "")).lower() == "buy"
            else "SELL",
            shares=_to_float_or_none(data.get("qty")) or 0.0,
            status=_map_status(str(data.get("status", ""))),
            filled_avg_price=_to_float_or_none(data.get("filled_avg_price")),
            submitted_at=data.get("submitted_at"),
            reason=lookup_order_reason(order_id, self._keys_path),
            # P6: surface Alpaca's client_order_id so the pre-trade
            # duplicate check can match on the deterministic id.
            client_order_id=str(data.get("client_order_id") or ""),
        )

    # -- price access for order sizing ------------------------------------
    # The dashboard sizes orders off the platform's downloaded prices;
    # fills still happen at Alpaca's prices.

    @property
    def cash(self) -> float:
        """Uninvested dollars (one account call)."""
        return self.get_account().cash

    @property
    def positions(self) -> dict[str, float]:
        """Ticker -> shares (one positions call)."""
        return self.get_positions()

    def last_price(self, ticker: str, date: str | None = None) -> float:
        """Latest stored close for a ticker from the local market database.

        Raises:
            KeyError: if no price is stored for that ticker/date.
        """
        t = normalize_ticker(ticker)
        with connect_db(self._db) as conn:
            if date is None:
                row = conn.execute(
                    "SELECT close FROM daily_bars WHERE ticker = ?"
                    " ORDER BY date DESC LIMIT 1",
                    (t,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT close FROM daily_bars WHERE ticker = ?"
                    " AND date <= ? ORDER BY date DESC LIMIT 1",
                    (t, date),
                ).fetchone()
        if not row or row[0] is None:
            raise KeyError(f"no stored price for {t}")
        return float(row[0])

    def exec_price(self, ticker: str, date: str | None = None) -> float:
        """Price the dashboard sizes orders off: the close (staging prices
        off the latest known close; the fill itself happens at Alpaca's
        next open)."""
        return self.last_price(ticker, date)


# ---------------------------------------------------------------------------
# Read-only verification CLI
# ---------------------------------------------------------------------------


def _verify_command(keys_path: str | Path | None = None) -> int:
    """Check the paper integration end to end without placing any order.

    Reads the saved paper keys, then hits the real paper endpoint for:
    authentication (via /v2/account), the account, the market clock, and
    positions. Prints plain-English results. Returns 0 when everything
    works, 2 when no keys are saved, 1 on any failure.
    """
    keys = load_paper_keys(keys_path)
    if keys is None:
        print(
            "No Alpaca paper keys saved yet.\n"
            "Save them on the dashboard's Brokers page first, then run "
            "this check again."
        )
        return 2
    key_id, secret = keys
    broker = AlpacaBroker(key_id, secret, keys_path=keys_path)
    print(f"Paper endpoint: {PAPER_BASE_URL}")
    try:
        account = broker.get_account()
    except AlpacaError as exc:
        print(f"Authentication / account: FAILED — {exc}")
        return 1
    print(
        f"Authentication: OK\n"
        f"Account: equity ${account.equity:,.2f}, cash "
        f"${account.cash:,.2f}, buying power ${account.buying_power:,.2f}"
    )
    try:
        clock = broker.get_clock()
    except AlpacaError as exc:
        print(f"Market clock: FAILED — {exc}")
        return 1
    if clock.get("is_open"):
        print("Market clock: OK — the market is OPEN right now.")
    else:
        print(
            "Market clock: OK — the market is closed right now "
            f"(next open: {clock.get('next_open', 'unknown')})."
        )
    try:
        positions = broker.get_positions()
    except AlpacaError as exc:
        print(f"Positions: FAILED — {exc}")
        return 1
    if positions:
        held = ", ".join(
            f"{t} {q:g}" for t, q in sorted(positions.items())
        )
        print(f"Positions: OK — holding: {held}.")
    else:
        print("Positions: OK — no open positions (all cash).")
    print(
        "\nAll checks passed. No orders were placed — this check never "
        "trades."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m qrp.execution.alpaca --verify [--keys PATH]."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Verify the Alpaca PAPER integration (read-only: authentication, "
            "account, market clock, positions). Never places an order."
        )
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the read-only verification against the paper endpoint.",
    )
    parser.add_argument(
        "--keys",
        default=None,
        help="Path to the paper-keys JSON file (default: next to the "
        "market database).",
    )
    args = parser.parse_args(argv)
    if not args.verify:
        parser.print_help()
        return 2
    return _verify_command(args.keys)


if __name__ == "__main__":
    raise SystemExit(main())
