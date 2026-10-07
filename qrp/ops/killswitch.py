"""Kill switch (P6: ops hardening) — the big red button.

What it does, in plain English:
1. Cancels ALL open orders at the broker (paper: Alpaca paper or the local
   paper broker).
2. Sets a persistent "trading is stopped" flag that the order pipeline
   checks FIRST — staging and submission refuse while it is engaged.
3. Writes an audit event so the whole story reads back later.

What it does NOT do: it NEVER liquidates (sells off) your positions.
Canceling open orders and selling everything are different risks — a panic
sell-all is its own decision, made deliberately by a person, never by an
automatic guard. The kill switch leaves your holdings exactly as they are.

Triggers:
- Manual: the dashboard's "STOP trading" button (with confirmation), the
  CLI (``python -m qrp.ops.killswitch --engage``), or simply creating the
  ``STOP`` file next to the market database (``touch data/STOP``) — for when
  the dashboard isn't reachable.
- Automatic (paper now, live later): the pipeline guards in
  ``qrp.ops.pipeline`` engage it when a batch sees too many broker
  rejections, when API authentication fails, or when the submit path raises
  an unexpected exception.

Re-enabling is ALWAYS manual — a dashboard action with explicit
confirmation, or ``--release`` on the CLI. Nothing auto-resets it, ever.

Fail-closed: if the flag file exists but cannot be read or parsed, the
pipeline treats trading as STOPPED (a broken flag must never mean "go").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from qrp.data.store import DEFAULT_DB
from qrp.execution.broker import Broker, BrokerError
from qrp.ops import ops_files

# --- Documented constants -----------------------------------------------------

#: Broker rejections above this count inside ONE order batch engage the kill
#: switch automatically. Three is deliberately low: a healthy batch has
#: zero rejections, so several in a row means something is systematically
#: wrong (bad keys, a broken mapping, a broker outage) — not bad luck.
AUTO_TRIGGER_MAX_REJECTIONS = 3


class KillSwitchEngaged(Exception):
    """Trading is stopped (or the flag could not be read — fail-closed).

    The message is always plain English, safe to show the user.
    """


def killswitch_path(db_path: str | Path = DEFAULT_DB) -> Path:
    """Where the persistent kill-switch flag lives (gitignored)."""
    return ops_files(db_path).killswitch


def stop_flag_path(db_path: str | Path = DEFAULT_DB) -> Path:
    """The dead-simple file flag: when this file exists, trading is stopped.

    ``touch`` it when the dashboard isn't reachable; delete it (or use the
    dashboard/CLI release) to stand down.
    """
    return ops_files(db_path).stop_flag


def stop_file_present(db_path: str | Path = DEFAULT_DB) -> bool:
    """True when the manual STOP file flag exists."""
    return stop_flag_path(db_path).exists()


def read_state(db_path: str | Path = DEFAULT_DB) -> dict:
    """The kill-switch flag as a dict.

    Missing file => {"engaged": False, ...} (never engaged by default).
    A file that exists but cannot be read or parsed raises
    KillSwitchEngaged — fail-closed: a broken flag must never read as "go".
    """
    path = killswitch_path(db_path)
    state = {
        "engaged": False,
        "engaged_at": None,
        "reason": "",
        "actor": "",
        "released_at": None,
    }
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return state
    except OSError as exc:
        raise KillSwitchEngaged(
            "Trading is STOPPED: the kill-switch flag file exists but could "
            f"not be read ({exc.strerror or exc}). Until it can be read, "
            "nothing will be staged or submitted — delete or fix "
            f"{path} to stand down."
        )
    try:
        data = json.loads(raw)
    except ValueError:
        raise KillSwitchEngaged(
            "Trading is STOPPED: the kill-switch flag file is corrupted "
            f"(not valid JSON: {path}). Until it is fixed or deleted, "
            "nothing will be staged or submitted."
        )
    if not isinstance(data, dict):
        raise KillSwitchEngaged(
            "Trading is STOPPED: the kill-switch flag file has an unexpected "
            f"format ({path}). Until it is fixed or deleted, nothing will "
            "be staged or submitted."
        )
    state.update(
        {
            "engaged": bool(data.get("engaged", False)),
            "engaged_at": data.get("engaged_at"),
            "reason": str(data.get("reason") or ""),
            "actor": str(data.get("actor") or ""),
            "released_at": data.get("released_at"),
        }
    )
    return state


def is_engaged(db_path: str | Path = DEFAULT_DB) -> bool:
    """True when trading is stopped: flag engaged OR the STOP file exists.

    Raises KillSwitchEngaged when the flag file is unreadable (fail-closed).
    """
    if stop_file_present(db_path):
        return True
    return bool(read_state(db_path).get("engaged"))


def _engaged_message(db_path: str | Path = DEFAULT_DB) -> str:
    """Plain-English explanation of why trading is stopped right now."""
    if stop_file_present(db_path):
        return (
            "Trading is STOPPED: the STOP file flag is present "
            f"({stop_flag_path(db_path)}). Delete the file — or re-enable "
            "trading on the Operations page / with "
            "'python -m qrp.ops.killswitch --release' — to stand down."
        )
    state = read_state(db_path)
    reason = state.get("reason") or "no reason recorded"
    when = state.get("engaged_at") or "an unknown time"
    return (
        "Trading is STOPPED by the kill switch "
        f"(engaged {when} — {reason}). Re-enable it on the Operations page "
        "or with 'python -m qrp.ops.killswitch --release'."
    )


def ensure_trading_allowed(db_path: str | Path = DEFAULT_DB) -> None:
    """The order pipeline calls this FIRST, before anything else.

    Raises KillSwitchEngaged with a plain-English message when the kill
    switch is engaged, the STOP file is present, or the flag file cannot
    be read (fail-closed). Returns None when trading may proceed.
    """
    if stop_file_present(db_path):
        raise KillSwitchEngaged(_engaged_message(db_path))
    # read_state raises KillSwitchEngaged itself when the file is broken.
    if read_state(db_path).get("engaged"):
        raise KillSwitchEngaged(_engaged_message(db_path))


def _write_state(state: dict, db_path: str | Path = DEFAULT_DB) -> None:
    path = killswitch_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def engage(
    reason: str,
    *,
    actor: str = "you",
    db_path: str | Path = DEFAULT_DB,
) -> dict:
    """Engage the kill switch (manual). Writes the flag + an audit event."""
    from qrp.ops import audit as _audit

    state = read_state(db_path)
    state.update(
        {
            "engaged": True,
            "engaged_at": _now_iso(),
            "reason": reason or "manual stop",
            "actor": actor,
            "released_at": None,
        }
    )
    _write_state(state, db_path)
    entry = _audit.log_event(
        "kill_switch_engaged",
        {
            "reason": state["reason"],
            "engaged_at": state["engaged_at"],
            "stop_file_present": stop_file_present(db_path),
        },
        actor=actor,
        db_path=db_path,
    )
    return {"state": state, "audit_seq": entry["seq"]}


def release(
    *,
    actor: str = "you",
    db_path: str | Path = DEFAULT_DB,
) -> dict:
    """Release the kill switch. ALWAYS an explicit, deliberate action —
    nothing in the platform calls this on its own.

    Clears the flag and removes the STOP file flag (deleting that file by
    hand is itself an explicit manual action, so both paths count as
    deliberate). Writes an audit event either way.
    """
    from qrp.ops import audit as _audit

    state = read_state(db_path)
    was_engaged = bool(state.get("engaged")) or stop_file_present(db_path)
    state.update(
        {
            "engaged": False,
            "released_at": _now_iso(),
            "release_actor": actor,
        }
    )
    _write_state(state, db_path)
    stop_removed = False
    try:
        stop_flag_path(db_path).unlink()
        stop_removed = True
    except FileNotFoundError:
        pass
    except OSError:
        pass
    entry = _audit.log_event(
        "kill_switch_released",
        {
            "was_engaged": was_engaged,
            "released_at": state["released_at"],
            "stop_file_removed": stop_removed,
        },
        actor=actor,
        db_path=db_path,
    )
    return {"state": state, "audit_seq": entry["seq"]}


def cancel_all_open_orders(broker: Broker) -> dict:
    """Cancel every open order at the broker. Returns a plain-English report.

    Works with any ``Broker`` implementation (Alpaca paper, local paper
    broker). It ONLY cancels orders — positions are never touched. (Selling
    everything in a panic is its own risk; liquidation is a separate,
    deliberate, manual decision — never an automatic guard's job.)

    Returns {"canceled": [ids...], "failed": [{"id":..., "error":...}]}.
    Broker errors on individual orders are collected, not raised, so one
    stubborn order can't block the rest.
    """
    report: dict = {"canceled": [], "failed": []}
    try:
        open_orders = broker.list_orders("open")
    except BrokerError as exc:
        report["failed"].append(
            {"id": "<listing open orders>", "error": str(exc)}
        )
        return report
    for order in open_orders:
        try:
            if broker.cancel_order(order.id):
                report["canceled"].append(order.id)
            else:
                report["failed"].append(
                    {
                        "id": order.id,
                        "error": "the broker said there was no such open order",
                    }
                )
        except BrokerError as exc:
            report["failed"].append({"id": order.id, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - keep canceling the rest
            report["failed"].append(
                {"id": order.id, "error": f"unexpected problem: {exc}"}
            )
    return report


def trigger_kill_switch(
    broker: Broker,
    reason: str,
    *,
    actor: str = "you",
    db_path: str | Path = DEFAULT_DB,
) -> dict:
    """The full kill-switch sequence: cancel all open orders, then engage.

    1. Cancels ALL open orders at ``broker`` (positions untouched — see
       ``cancel_all_open_orders``).
    2. Sets the persistent disabled flag.
    3. Writes an audit event with the cancel report.

    Returns {"canceled": [...], "failed": [...], "state": ...}.
    """
    from qrp.ops import audit as _audit

    cancel_report = cancel_all_open_orders(broker)
    engaged = engage(reason, actor=actor, db_path=db_path)
    _audit.log_event(
        "kill_switch_cancel_report",
        {
            "reason": reason,
            "canceled_order_ids": cancel_report["canceled"],
            "failed_cancels": cancel_report["failed"],
            "positions_untouched": True,
        },
        actor=actor,
        db_path=db_path,
    )
    return {
        "canceled": cancel_report["canceled"],
        "failed": cancel_report["failed"],
        "state": engaged["state"],
    }


def describe_state(db_path: str | Path = DEFAULT_DB) -> str:
    """One plain-English line about the kill-switch state (for UIs/CLIs)."""
    try:
        engaged = is_engaged(db_path)
    except KillSwitchEngaged as exc:
        return str(exc)
    if not engaged:
        return "Kill switch: OFF — trading is allowed."
    state = read_state(db_path)
    reason = state.get("reason") or "no reason recorded"
    when = state.get("engaged_at") or "an unknown time"
    via = "the STOP file" if stop_file_present(db_path) else "the flag file"
    return (
        f"Kill switch: ENGAGED since {when} ({reason}; via {via}). "
        "Staging and submission are blocked until it is manually released."
    )


# ---------------------------------------------------------------------------
# CLI: python -m qrp.ops.killswitch --engage/--release/--status
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI. Never places or cancels orders on --status; --engage cancels
    all open orders at the configured paper broker first."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Kill switch for paper trading: --status shows the state, "
            "--engage stops all trading (cancels open orders first), "
            "--release re-enables it. Re-enabling is always manual."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true",
                       help="Show whether the kill switch is engaged.")
    group.add_argument("--engage", action="store_true",
                       help="Engage: cancel all open paper orders, stop trading.")
    group.add_argument("--release", action="store_true",
                       help="Release: re-enable trading (manual only).")
    parser.add_argument("--reason", default="",
                        help="Why the kill switch is being engaged.")
    parser.add_argument("--db", default=None,
                        help="Market database path (default: data/market.db).")
    args = parser.parse_args(argv)
    db_path = args.db if args.db else DEFAULT_DB

    if args.status:
        print(describe_state(db_path))
        return 0
    if args.release:
        release(actor="cli", db_path=db_path)
        print("Kill switch released — trading is allowed again.")
        print(describe_state(db_path))
        return 0
    # --engage: cancel open orders at the paper broker, then engage.
    broker = _paper_broker_for_cli(db_path)
    result = trigger_kill_switch(
        broker,
        args.reason or "manual stop from the command line",
        actor="cli",
        db_path=db_path,
    )
    n = len(result["canceled"])
    print(f"Kill switch ENGAGED. Canceled {n} open order(s).")
    for failure in result["failed"]:
        print(f"  Could not cancel {failure['id']}: {failure['error']}")
    print("Positions were NOT sold — only open orders were canceled.")
    return 0


def _paper_broker_for_cli(db_path: str | Path):
    """The broker the CLI cancels orders at: Alpaca paper when keys are
    saved, else the local paper broker. Read-only until --engage."""
    from qrp.execution.alpaca import load_paper_keys, default_keys_path
    from qrp.execution.paper import PaperBroker, PaperBrokerAdapter

    keys_path = default_keys_path(db_path)
    keys = load_paper_keys(keys_path)
    if keys is not None:
        from qrp.execution.alpaca import AlpacaBroker

        key_id, secret = keys
        return AlpacaBroker(
            key_id, secret, db_path=db_path, keys_path=keys_path
        )
    return PaperBrokerAdapter(PaperBroker(db_path))


if __name__ == "__main__":
    raise SystemExit(main())
