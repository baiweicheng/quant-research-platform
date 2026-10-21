"""Alpaca paper-trading adapter (Phase P2).

Talks to Alpaca's REST API directly with ``requests`` (no third-party SDK —
the repo pins only requests/pandas/numpy/streamlit). PAPER ONLY: the
default base URL is Alpaca's paper endpoint, and the dashboard never offers
any live-trading UI. The adapter *can* be pointed elsewhere (its tests do),
but constructing it against Alpaca's live server raises unless the caller
explicitly opts in with ``allow_live=True`` — tripping into live trading
by accident is meant to be impossible.

How staging maps onto Alpaca
-----------------------------
The platform stages orders computed from data through the most recent
close. Every staged order goes out to Alpaca immediately as a plain
day market order (``type="market"``, ``time_in_force="day"``),
fractional share quantities included — no whole-share rounding
(Alpaca accepts fractional quantities only on day market orders;
quantities here carry at most 4 decimal places).

When the market is open, the order fills right away at current prices;
when the market is closed, Alpaca holds the order and releases it at
the next trading day's open, where it fills as a market order starting
right at the open — usually within a whisker of the official opening
print, but not auction-guaranteed the way market-on-open orders were.
Both timings are intended (the user's 2026-10-09 "allow mid-day fills"
decision): staging decisions still use only data through yesterday's
close — a mid-day fill just executes a little later than the open, at
intraday prices. Alpaca's clock (GET /v2/clock) is read only so the
dashboard can say plainly, before the user confirms, which of the two
will happen; it never gates a send.

Orders Alpaca refuses at submission wait in a small local queue and
are retried as day orders on Paper page visits; entries that keep
failing are dropped after QUEUE_MAX_WAIT_DAYS days (the next staging
run recomputes orders from scratch anyway).

Every staging run also leaves a snapshot of its target (weights,
reference prices, account equity, decision date) next to the queue.
Acceptance is not a fill — an order can be accepted at night and still
vanish before the open (the SUN failures of October 2026) — so the
dashboard's "Check fills & top up" button compares the account's real
positions against that snapshot and sends whatever is missing, as
ordinary day orders with the original decision date's order IDs.

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
SNAPSHOT_FILENAME = "alpaca_staging_snapshot.json"

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


#: Alpaca order statuses after which a client_order_id is spent: reusing
#: the id is refused, but no live order is at stake, so submitting a
#: follow-up revision ("<id>-r2") is safe.
_SPENT_STATUSES = {"canceled", "expired", "rejected"}


def _is_id_conflict(exc: AlpacaError) -> bool:
    """True when Alpaca refused with 422 client_order_id-must-be-unique."""
    if getattr(exc, "status_code", None) != 422:
        return False
    msg = str(exc).lower()
    return "client_order_id" in msg and "unique" in msg


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
    """Orders Alpaca refused at submission, waiting to be retried.

    Each entry: {ticker, side, shares, reason, staged_at, attempts,
    last_error}. Staging always sends straight to Alpaca, so the queue
    holds only orders Alpaca wouldn't accept (API errors and the
    like); the dashboard retries them on every Paper page visit as day
    orders, and entries that keep failing are dropped after
    QUEUE_MAX_WAIT_DAYS days.
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
    client_order_id: str | None = None,
) -> None:
    """Keep an order Alpaca refused in the local retry queue.

    The client_order_id is stored alongside so a later flush retries the
    SAME id (Alpaca dedupes on it) instead of minting a new one that
    could double-trade an order Alpaca actually accepted.
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
            "client_order_id": (client_order_id or "").strip() or None,
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
# Staging snapshot + post-open fill check
# ---------------------------------------------------------------------------


def _snapshot_path(keys_path: str | Path | None = None) -> Path:
    base = Path(keys_path) if keys_path else default_keys_path()
    return base.parent / SNAPSHOT_FILENAME


def save_staging_snapshot(
    snapshot: dict, keys_path: str | Path | None = None
) -> None:
    """Remember the most recent staged rebalance (best-effort).

    Written whenever a staging run submits at least one order, and read
    back by the post-open fill check ("Check fills & top up") so it can
    compare the account's actual positions against the rebalance's
    target. One snapshot only: a new staging overwrites the old one.
    Never raises — a failed note must not break staging.
    """
    path = _snapshot_path(keys_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot), encoding="utf-8")
    except OSError:
        pass


def load_staging_snapshot(
    keys_path: str | Path | None = None,
) -> dict | None:
    """The last staged rebalance's snapshot, or None when there is none
    (never staged, or the file is unreadable/garbled)."""
    try:
        data = json.loads(_snapshot_path(keys_path).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("target_weights"), dict):
        return None
    if not isinstance(data.get("reference_prices"), dict):
        return None
    return data


#: The fill check leaves a position alone when it is within this share
#: of its target value — the same 2% drift buffer the weekly rebalance
#: itself uses, so the check never fights the strategy's own tolerance
#: for small price moves since staging.
FILL_CHECK_DRIFT = 0.02

#: Gaps worth less than this are dust (Alpaca's minimum order size) and
#: are skipped, matching the staging convention.
FILL_CHECK_DUST_VALUE = 1.0


def compute_fill_gaps(
    snapshot: dict,
    positions: dict,
    open_orders=(),
) -> tuple[list[dict], list[str]]:
    """Gap orders needed to complete a staged rebalance.

    Compares what the account ACTUALLY holds now (``positions``:
    ticker -> shares, straight from the broker) against the staged
    rebalance's target (snapshot weights x snapshot equity, converted
    to shares at the snapshot's reference prices — the same prices the
    original orders were sized with, so a partial fill's remainder
    comes out exact). Because current positions are the truth, anything
    the user traded by hand since staging is respected automatically:
    no order is generated for a gap that no longer exists.

    Returns (gaps, notes): gap dicts {ticker, side, shares, price,
    reason} — sells first, then buys, like a rebalance — and
    plain-English notes for everything deliberately skipped (dust,
    within the drift buffer is silent by design; an already-open order
    and an unpriceable name are named).
    """
    try:
        equity = float(snapshot.get("equity") or 0.0)
    except (TypeError, ValueError):
        equity = 0.0
    weights = snapshot.get("target_weights") or {}
    prices = snapshot.get("reference_prices") or {}
    held_by: dict[str, float] = {}
    for t, q in (positions or {}).items():
        try:
            held_by[normalize_ticker(str(t))] = float(q or 0.0)
        except (TypeError, ValueError):
            continue
    open_keys = set()
    for o in open_orders or ():
        t = getattr(o, "ticker", None) or (
            o.get("ticker") if isinstance(o, dict) else None)
        s = getattr(o, "side", None) or (
            o.get("side") if isinstance(o, dict) else None)
        if t and s:
            open_keys.add((normalize_ticker(str(t)),
                           str(s).strip().upper()))
    gaps: list[dict] = []
    notes: list[str] = []
    tickers = sorted(set(weights) | set(held_by))
    for ticker in tickers:
        try:
            weight = float(weights.get(ticker, 0.0) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        held = held_by.get(ticker, 0.0)
        if weight <= 0 and held <= 0:
            continue
        try:
            price = float(prices.get(ticker, 0.0) or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            notes.append(
                f"Couldn't check {ticker} — no price was recorded for "
                "it when the rebalance was staged, so its target can't "
                "be reconstructed. Stage a fresh rebalance to cover it."
            )
            continue
        target_shares = weight * equity / price if equity > 0 else 0.0
        gap = target_shares - held
        gap_value = gap * price
        if abs(gap_value) < FILL_CHECK_DUST_VALUE:
            continue  # dust — under Alpaca's $1 minimum order size
        target_value = target_shares * price
        if target_value > 0 and abs(gap_value) <= (
            FILL_CHECK_DRIFT * target_value
        ):
            continue  # close enough — the strategy's own drift buffer
        if gap > 0:
            side, shares = "BUY", gap
            if held <= 0:
                reason = "staged order didn't fill — position is missing"
            else:
                reason = "staged order only partly filled — topping up"
        else:
            side, shares = "SELL", min(-gap, held)
            if weight <= 0:
                reason = (
                    "not in the last rebalance's target — selling the "
                    "leftover"
                )
            else:
                reason = "above the last rebalance's target — trimming back"
        if shares <= 0:
            continue
        if (ticker, side) in open_keys:
            notes.append(
                f"Skipped {ticker} — a {side} order for it is already "
                "open at Alpaca, so nothing new was sent for it."
            )
            continue
        gaps.append(
            {
                "ticker": ticker,
                "side": side,
                "shares": shares,
                "price": price,
                "reason": reason,
            }
        )
    gaps.sort(key=lambda g: 0 if g["side"] == "SELL" else 1)
    return gaps, notes


# ---------------------------------------------------------------------------
# Market clock helpers (disclosure only — the clock never gates a send)
# ---------------------------------------------------------------------------


def fetch_market_clock(broker) -> dict | None:
    """The broker's market clock, or None when it can't be read.

    Used only so the dashboard can tell the user, before they confirm,
    whether staged orders will fill right away (market open) or at the
    next open (market closed). None simply means "unknown" — staging
    sends either way.
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


def flush_staging_queue(broker) -> dict:
    """Retry every queued staging order to Alpaca as a day market order.

    The queue holds only orders Alpaca refused at submission; called
    on every Paper trading page visit. Sends go out immediately as day
    market orders — filling right away if the market is open, or at
    the next open if Alpaca is holding them for tomorrow. Returns
    {"sent": n, "waiting": m, "notes": [...]} with plain-English
    notes. Entries that Alpaca keeps rejecting are dropped only after
    QUEUE_MAX_WAIT_DAYS days of waiting (the next staging run
    recomputes orders from scratch anyway) — never merely because the
    page was visited a few times.
    """
    keys_path = broker._keys_path
    entries = load_staging_queue(keys_path)
    if not entries:
        return {"sent": 0, "waiting": 0, "notes": []}
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
            # Conflict-aware submit: if the stored id collides with a dead
            # order at Alpaca, a follow-up revision goes out instead of a
            # permanent 422; if the intent is already live/filled there,
            # nothing new is sent.
            _info, resolve_note = broker.place_order_resolving_conflict(
                str(entry.get("ticker")),
                str(entry.get("side")),
                float(entry.get("shares")),
                reason=str(entry.get("reason", "") or ""),
                time_in_force="day",
                client_order_id=entry.get("client_order_id"),
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
            if resolve_note:
                notes.append(resolve_note)
            sent += 1
            open_keys.add(key)
    save_staging_queue(kept, keys_path)
    return {"sent": sent, "waiting": len(kept), "notes": notes}


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

    def get_order_by_client_order_id(
        self, client_order_id: str
    ) -> OrderInfo | None:
        """The order Alpaca holds under this client_order_id, else None.

        Returns None (not an error) when Alpaca simply has no such order
        (HTTP 404). Transport trouble still raises AlpacaError.
        """
        cid = (client_order_id or "").strip()
        if not cid:
            return None
        try:
            data = _require_dict(
                self._request(
                    "GET",
                    "/v2/orders:by_client_order_id",
                    params={"client_order_id": cid},
                ),
                "the order lookup",
            )
        except AlpacaError as exc:
            if exc.status_code == 404:
                return None
            raise
        return self._to_order_info(data)

    def place_order_resolving_conflict(
        self,
        ticker: str,
        side: str,
        shares: float,
        *,
        reason: str = "",
        time_in_force: str = "day",
        client_order_id: str | None = None,
        max_revisions: int = 5,
    ) -> tuple[OrderInfo, str]:
        """Place an order, resolving client_order_id collisions with Alpaca.

        Alpaca refuses a reused client_order_id (HTTP 422
        "client_order_id must be unique") even when the original order is
        long dead (canceled/expired) — so a plain retry with the same id
        can never succeed, and the local retry queue would spin on it
        forever. This method:

        - places with the given id;
        - on a 422 id-conflict, reads back the order Alpaca holds;
        - if that order is spent (canceled/expired/rejected), resubmits as
          a follow-up revision "<id>-r2" (then -r3, ...);
        - if that order is still live (open) or already filled, returns it
          instead of sending a duplicate — the intent is already
          represented at Alpaca, and re-sending could double-trade.

        Returns (OrderInfo, note): note is "" on a clean first-try
        placement, else a plain-English explanation of what happened.
        Any other Alpaca refusal raises AlpacaError unchanged.
        """
        cid = (client_order_id or "").strip() or None
        if cid is None:
            return (
                self.place_order(
                    ticker, side, shares, reason=reason,
                    time_in_force=time_in_force,
                ),
                "",
            )

        def _try(submit_cid: str) -> OrderInfo:
            return self.place_order(
                ticker, side, shares, reason=reason,
                time_in_force=time_in_force, client_order_id=submit_cid,
            )

        try:
            return _try(cid), ""
        except AlpacaError as exc:
            if not _is_id_conflict(exc):
                raise
        existing = self.get_order_by_client_order_id(cid)
        if existing is not None and existing.status not in _SPENT_STATUSES:
            # Live or already filled: the intent is already at Alpaca —
            # send nothing new. (A filled order means the position already
            # reflects this trade; an open one is still working.)
            return existing, (
                f"Kept the {side} order for {normalize_ticker(ticker)} "
                f"as-is: an order with the same ID is already "
                f"{existing.status} at Alpaca, so no duplicate was sent."
            )
        # Spent id (or Alpaca claims a duplicate it won't show us):
        # follow-up revisions until one lands.
        spent = existing.status if existing is not None else "closed"
        for rev in range(2, max_revisions + 1):
            rev_cid = f"{cid}-r{rev}"
            try:
                return _try(rev_cid), (
                    f"The original order ID was already used by a {spent} "
                    f"order at Alpaca, so this went out as a follow-up "
                    f"({rev_cid})."
                )
            except AlpacaError as exc:
                if not _is_id_conflict(exc):
                    raise
        raise AlpacaError(
            f"Alpaca kept refusing the {side} order for "
            f"{normalize_ticker(ticker)}: its order ID {cid} (and "
            f"{max_revisions - 1} follow-ups) are all already in use. "
            "Cancel the old orders at Alpaca and try staging again.",
            422,
        )

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
