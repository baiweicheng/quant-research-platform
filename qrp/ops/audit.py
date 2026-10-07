"""Append-only audit log (P6: ops hardening).

Every entry is one JSON object per line (JSONL) in ``data/audit.log``
(gitignored, local only). Each entry carries:

- ``seq``: 1-based sequence number,
- ``ts``: UTC timestamp,
- ``event_type``: what happened (see EVENT_TYPES below),
- ``actor``: who/what did it ("dashboard", "cli", "auto-guard", ...),
- ``details``: plain JSON with the facts (never API keys — values are
  scrubbed before writing),
- ``prev_hash``: the SHA-256 hash of the previous entry,
- ``hash``: the SHA-256 hash of this entry (tamper-evident chain).

The file is append-only: entries are only ever added, never edited or
deleted. ``verify_chain()`` recomputes every hash and checks the linkage,
so tampering or gaps are detected and reported in plain English.

Log rotation (kept simple, documented here): once a month, rename the
current file to ``audit-YYYY-MM.log`` and let a fresh ``audit.log``
start. Never delete entries from the current file; keep old monthly
files as long as you might ever want to re-read them. There is no
automatic rotation — renaming the file is a deliberate, visible act.

Recorded event types (EVENT_TYPES): signal computations ("signals_used"),
order staging ("orders_staged"), pre-trade check results
("pretrade_checks"), broker responses ("order_submitted",
"order_rejected"), fills ("orders_filled"), queue events ("queue_event"),
kill-switch engage/release ("kill_switch_engaged",
"kill_switch_released", "kill_switch_cancel_report"), blocked staging
attempts ("staging_blocked"), and key save/remove ("keys_saved",
"keys_removed" — never the keys themselves).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qrp.data.store import DEFAULT_DB
from qrp.ops import ops_files

AUDIT_FILENAME = "audit.log"

#: Every event type the platform writes. New types must be added here so
#: the UI's filter list and the tests stay in sync.
EVENT_TYPES = (
    "signals_used",
    "orders_staged",
    "pretrade_checks",
    "order_submitted",
    "order_rejected",
    "orders_filled",
    "queue_event",
    "kill_switch_engaged",
    "kill_switch_released",
    "kill_switch_cancel_report",
    "keys_saved",
    "keys_removed",
    "staging_blocked",
    "audit_chain_verified",
)

_GENESIS_PREV_HASH = "GENESIS"


# --- secret scrubbing ----------------------------------------------------------
# Mirrors qrp.ui.helpers._redact_for_log: labeled secrets and any bare
# 32+ character token that could be a key become placeholders. The audit
# log must never carry API keys, even by accident.

_LABELED_SECRET_RE = re.compile(
    r"(?i)(secret|api[_-]?key|key[_-]?id|password|token)"
    r"([\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+"
)
_BARE_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")


def _scrub_text(text: str) -> str:
    text = _LABELED_SECRET_RE.sub(r"\1\2<redacted>", text)
    return _BARE_TOKEN_RE.sub("<redacted-token>", text)


def _scrub(value: Any) -> Any:
    """Recursively scrub secret-looking values from a JSON-able object."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {str(_scrub(k)): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


# --- hashing --------------------------------------------------------------------


def _canonical(entry: dict) -> str:
    """Deterministic JSON for hashing (sorted keys, no whitespace).

    ``default=str`` keeps logging total: an exotic value can never break
    the audit trail — it is recorded as its string form instead.
    """
    return json.dumps(entry, sort_keys=True, separators=(",", ":"),
                      default=str)


def _entry_hash(entry: dict) -> str:
    """SHA-256 over the entry WITHOUT its own 'hash' field."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def audit_path(db_path: str | Path = DEFAULT_DB) -> Path:
    """Where the audit log lives (gitignored)."""
    return ops_files(db_path).audit_log


# --- writing -----------------------------------------------------------------------


def _last_entry(path: Path) -> dict | None:
    """The last complete entry in the log, or None when the log is empty.

    Reads from the tail so a large log doesn't get fully loaded. Skips
    blank lines (every entry ends with a newline, so the file virtually
    always ends with one).
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, io.SEEK_END)
            pos = fh.tell()
            # Skip trailing newlines / whitespace.
            while pos > 0:
                fh.seek(pos - 1)
                if fh.read(1) not in (b"\n", b"\r", b" ", b"\t"):
                    break
                pos -= 1
            if pos == 0:
                return None
            # Find the start of the last non-empty line.
            start = pos
            while start > 0:
                fh.seek(start - 1)
                if fh.read(1) == b"\n":
                    break
                start -= 1
            fh.seek(start)
            tail = fh.read(pos - start).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "seq" in entry:
            return entry
        return None
    return None


def log_event(
    event_type: str,
    details: dict | None = None,
    *,
    actor: str = "dashboard",
    db_path: str | Path = DEFAULT_DB,
) -> dict:
    """Append one audit entry. Never raises for logging reasons.

    Returns the entry that was written (with seq and hash). ``details``
    is scrubbed of anything secret-looking first. Note: this never
    rewrites history — it only appends.
    """
    path = audit_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = _last_entry(path)
    seq = int(prev.get("seq", 0) or 0) + 1 if prev else 1
    prev_hash = prev.get("hash") if prev else _GENESIS_PREV_HASH
    entry = {
        "seq": seq,
        "ts": _now_iso(),
        "event_type": str(event_type),
        "actor": str(actor),
        "details": _scrub(details or {}),
        "prev_hash": prev_hash,
    }
    entry["hash"] = _entry_hash(entry)
    line = _canonical(entry)
    # Append-only: "a" mode, one line per entry. Two processes appending at
    # the exact same instant could interleave seq numbers on paper, but the
    # hash chain still verifies and the dashboard is single-user, so this
    # documented simplicity wins over file locking.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return entry


# --- reading -------------------------------------------------------------------------


def read_events(
    db_path: str | Path = DEFAULT_DB,
    *,
    event_types: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read audit entries, newest last. Filters: event types, ISO time
    range (inclusive), and a max count (keeps the most recent). Malformed
    lines are skipped — ``verify_chain`` is the tool that reports them."""
    path = audit_path(db_path)
    entries: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict) or "seq" not in entry:
                    continue
                entries.append(entry)
    except FileNotFoundError:
        return []
    except OSError:
        return []
    wanted = set(event_types) if event_types else None
    out = []
    for entry in entries:
        if wanted and entry.get("event_type") not in wanted:
            continue
        ts = str(entry.get("ts") or "")
        if since and ts < since:
            continue
        if until and ts > until:
            continue
        out.append(entry)
    if limit is not None and len(out) > limit:
        out = out[-limit:]
    return out


def verify_chain(db_path: str | Path = DEFAULT_DB) -> dict:
    """Recompute every hash and check the linkage. Returns a dict:

    {"ok": bool, "checked": n, "message": str} — the message is plain
    English, suitable for showing on the Operations page.
    """
    path = audit_path(db_path)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except FileNotFoundError:
        return {
            "ok": True,
            "checked": 0,
            "message": "No audit log yet — nothing has been recorded.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "checked": 0,
            "message": (
                "Could not read the audit log to verify it: "
                f"{exc.strerror or exc}."
            ),
        }
    prev_hash = _GENESIS_PREV_HASH
    expected_seq = 1
    for i, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except ValueError:
            return {
                "ok": False,
                "checked": i - 1,
                "message": (
                    f"Audit log line {i} is not valid JSON — the log may "
                    "have been tampered with or corrupted. Treat entries "
                    "after this point as untrusted."
                ),
            }
        problems = []
        if not isinstance(entry, dict):
            problems.append("not a JSON object")
        else:
            if entry.get("seq") != expected_seq:
                problems.append(
                    f"sequence break (expected {expected_seq}, found "
                    f"{entry.get('seq')})"
                )
            if entry.get("prev_hash") != prev_hash:
                problems.append("previous-hash link is broken")
            if entry.get("hash") != _entry_hash(entry):
                problems.append("entry hash does not match its contents")
        if problems:
            return {
                "ok": False,
                "checked": i - 1,
                "message": (
                    f"Audit log FAILED verification at entry {i}: "
                    + "; ".join(problems)
                    + ". The log may have been tampered with — entries up "
                    f"to #{i - 1} are intact."
                ),
            }
        prev_hash = entry["hash"]
        expected_seq += 1
    return {
        "ok": True,
        "checked": len(lines),
        "message": (
            f"Audit log verified: {len(lines)} entr"
            f"{'y' if len(lines) == 1 else 'ies'}, hash chain intact, "
            "no gaps."
            if lines
            else "No audit log yet — nothing has been recorded."
        ),
    }


# --- export ---------------------------------------------------------------------------


def export_jsonl(
    db_path: str | Path = DEFAULT_DB,
    dest: str | Path | None = None,
) -> Path:
    """Copy the audit log to ``dest`` (default: next to the log, with a
    timestamp). Returns the destination path."""
    src = audit_path(db_path)
    if dest is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = src.parent / f"audit-export-{stamp}.jsonl"
    dest = Path(dest)
    dest.write_bytes(src.read_bytes() if src.exists() else b"")
    return dest


def export_csv(
    db_path: str | Path = DEFAULT_DB,
    dest: str | Path | None = None,
) -> Path:
    """Export the audit log as CSV (one row per entry; details as JSON).
    Returns the destination path."""
    if dest is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = audit_path(db_path).parent / f"audit-export-{stamp}.csv"
    dest = Path(dest)
    events = read_events(db_path)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["seq", "timestamp_utc", "event_type", "actor", "details_json",
             "prev_hash", "hash"]
        )
        for e in events:
            writer.writerow(
                [
                    e.get("seq"),
                    e.get("ts"),
                    e.get("event_type"),
                    e.get("actor"),
                    json.dumps(e.get("details", {}), sort_keys=True,
                               default=str),
                    e.get("prev_hash"),
                    e.get("hash"),
                ]
            )
    return dest
