"""Background data-refresh job.

Runs the one-click "Refresh new data" (new daily prices + new SEC
filings) plus the pending-order fill pass on a daemon thread inside the
Streamlit server process, reporting progress through a small JSON status
file next to the database. Because the work lives in the server process
— not in a page's script run — navigating to another page, or closing
the browser entirely, never stops a run. Reopening the Market data page
later shows live progress or the finished summary.

Status states (all plain English):
  idle        — no run has happened yet (no status file)
  running     — a run is in progress (heartbeat is fresh)
  done        — finished; the summary and fill report are stored
  error       — the worker raised; a plain-English message is stored
  interrupted — a previous run's heartbeat went stale (>10 minutes with
                no sign of life); it is safe to start a new run

Only stdlib here (plus a deferred import of the UI helpers inside the
worker) so this module stays import-light.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_FILENAME = "refresh_status.json"
# A run with no heartbeat for this long is presumed dead (server
# restarted, thread killed, ...). Ten minutes is generous: even the
# slowest single ticker finishes far sooner.
STALE_AFTER_SECONDS = 10 * 60

_lock = threading.Lock()


def status_path(db_path: str | Path) -> Path:
    """Where this database's refresh status lives (gitignored)."""
    return Path(db_path).parent / STATUS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _heartbeat_age_s(status: dict) -> float | None:
    hb = status.get("heartbeat")
    if not hb:
        return None
    try:
        then = datetime.fromisoformat(hb)
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def _idle_status() -> dict:
    return {
        "state": "idle",
        "run_id": None,
        "done": 0,
        "total": 0,
        "phase": "",
        "step": "",
        "heartbeat": None,
        "started_at": None,
        "summary": None,
        "fill_report": None,
        "error": None,
    }


def _read_status_file(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_status_file(path: Path, status: dict) -> None:
    """Atomically replace the status file (readers never see a half-write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh)
    os.replace(tmp, path)


def _mark_interrupted(path: Path, status: dict) -> dict:
    """Record that a stale run died; returns the new status dict."""
    note = (
        "The previous refresh seems to have stopped partway through "
        "(no sign of life for over 10 minutes) — perhaps the app "
        "restarted. Nothing was half-saved: the refresh is safe to run "
        "again, and it only ever adds what's missing."
    )
    status = dict(status)
    status.update(
        {
            "state": "interrupted",
            "step": "",
            "error": note,
        }
    )
    _write_status_file(path, status)
    return status


def get_status(db_path: str | Path) -> dict:
    """Current refresh status. Never raises for a missing/corrupt file.

    A "running" run whose heartbeat is stale is reported (and recorded)
    as "interrupted" so the UI never shows a stuck progress bar.
    """
    path = status_path(db_path)
    status = _read_status_file(path)
    if status is None:
        return _idle_status()
    status = {**_idle_status(), **status}
    if status["state"] == "running":
        age = _heartbeat_age_s(status)
        if age is None or age > STALE_AFTER_SECONDS:
            return _mark_interrupted(path, status)
    return status


def start_refresh(db_path: str | Path) -> dict:
    """Start a background refresh, unless one is already running.

    Returns immediately with the current status dict. Starting while a
    run is active (fresh heartbeat) is a no-op returning that run's
    status; a stale run is marked interrupted first. The work happens on
    a daemon thread in the server process.
    """
    path = status_path(db_path)
    with _lock:
        status = get_status(db_path)
        if status["state"] == "running":
            return status  # already going — no double start
        run_id = uuid.uuid4().hex
        now = _now_iso()
        status = {
            **_idle_status(),
            "state": "running",
            "run_id": run_id,
            "phase": "Starting",
            "step": "Starting the refresh…",
            "heartbeat": now,
            "started_at": now,
        }
        _write_status_file(path, status)
        thread = threading.Thread(
            target=_run_refresh,
            args=(str(db_path), run_id),
            name="refresh-job",
            daemon=True,
        )
        thread.start()
        return status


def _write_if_current(path: Path, run_id: str, updates: dict) -> bool:
    """Merge updates into the status file only if our run is still current.

    Returns False when another run superseded us (or the run was marked
    interrupted) — the worker then stops quietly instead of clobbering
    the newer status.
    """
    with _lock:
        status = _read_status_file(path)
        if (
            status is None
            or status.get("run_id") != run_id
            or status.get("state") != "running"
        ):
            return False
        status = {**status, **updates, "heartbeat": _now_iso()}
        _write_status_file(path, status)
        return True


def _plain_error(exc: BaseException, context: str) -> str:
    msg = " ".join(str(exc).split())[:200] or exc.__class__.__name__
    return f"{context}: {msg}"


def _json_safe(obj):
    """Round-trip through JSON so tuples etc. become plain JSON values."""
    return json.loads(json.dumps(obj, default=str))


def _run_refresh(db_path: str, run_id: str) -> None:
    """Worker body: refresh data, then the pending-order fill pass."""
    # Deferred import: qrp.ui.helpers pulls in Streamlit, which we don't
    # need at module import time (and the worker may run in tests).
    from qrp.ui import helpers as H

    path = status_path(db_path)

    def _progress(done: int, total: int) -> None:
        # First half of the steps is prices, second half is filings
        # (refresh_new_data budgets 2 steps per ticker).
        if total and done * 2 <= total:
            phase = "Checking for new daily prices"
        else:
            phase = "Checking for new SEC filings"
        _write_if_current(
            path,
            run_id,
            {
                "done": done,
                "total": total,
                "phase": phase,
                "step": f"{phase} — {done} of {total} steps…",
            },
        )

    try:
        summary = H.refresh_new_data(db_path, progress_cb=_progress)
    except Exception as exc:  # never raise to the UI; record it instead
        logger.exception("background refresh failed")
        with _lock:
            status = _read_status_file(path) or _idle_status()
            if status.get("run_id") == run_id and status.get("state") == "running":
                status = {
                    **status,
                    "state": "error",
                    "step": "",
                    "error": _plain_error(exc, "The refresh stopped early"),
                    "heartbeat": _now_iso(),
                }
                _write_status_file(path, status)
        return

    # New price data may unlock pending paper orders: fill them now (the
    # same idempotent logic as the Paper trading page), so the account is
    # current before the user even opens that page.
    fill_report: dict | None = None
    try:
        broker = H.paper_broker(db_path)
        fill_report = _json_safe(broker.fill_pending_orders())
    except Exception as exc:
        logger.exception("background fill pass failed")
        fill_report = {
            "filled": [],
            "skipped": [],
            "pending_remaining": None,
            "dividends_credited": 0.0,
            "error": _plain_error(
                exc, "New prices were saved, but pending paper orders "
                "could not be checked"
            ),
        }

    with _lock:
        status = _read_status_file(path) or _idle_status()
        if status.get("run_id") == run_id and status.get("state") == "running":
            status = {
                **status,
                "state": "done",
                "done": status.get("total", 0),
                "phase": "Finished",
                "step": "Finished.",
                "summary": _json_safe(summary),
                "fill_report": fill_report,
                "heartbeat": _now_iso(),
            }
            _write_status_file(path, status)


# Re-exported for tests that want to poke the staleness constant.
__all__ = [
    "STATUS_FILENAME",
    "STALE_AFTER_SECONDS",
    "status_path",
    "get_status",
    "start_refresh",
]
