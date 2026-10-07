"""Order-pipeline guards (P6: ops hardening).

Two things every order path (staging and submission, Alpaca paper and
local paper broker) gets:

1. **Kill-switch-first** (``check_trading_allowed``): the flag is checked
   before anything else, and the path refuses with a plain-English message
   when trading is stopped. Fail-closed: an unreadable flag file also
   refuses.

2. **Automatic trigger hooks** (``submit_with_auto_guards``): while a
   batch is being submitted, the pipeline watches for the failure shapes
   that mean "something is systematically wrong" and engages the kill
   switch by itself — canceling all open orders first, exactly like a
   manual stop:
   - the broker rejects more than ``AUTO_TRIGGER_MAX_REJECTIONS`` orders
     in one batch (documented in ``qrp.ops.killswitch``),
   - API authentication fails (possible key problem — never keep trading
     blind),
   - any unexpected exception escapes the submit path (fail-safe: stop
     first, explain after).

   No autopilot or scheduling machinery lives here — these are just the
   hooks in the submit path, designed for paper now and reusable for live
   later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qrp.data.store import DEFAULT_DB
from qrp.execution.broker import Broker, BrokerError
from qrp.ops import killswitch as _ks
from qrp.ops.pretrade import client_order_id


def check_trading_allowed(db_path: str | Path = DEFAULT_DB) -> None:
    """Refuse when the kill switch is engaged (or unreadable — fail-closed).

    Raises ``KillSwitchEngaged`` with a plain-English message. The order
    pipeline calls this FIRST, before computing or touching anything.
    """
    _ks.ensure_trading_allowed(db_path)


def _is_auth_failure(exc: BaseException) -> bool:
    """True when the error means the API keys were rejected.

    Alpaca signals this as HTTP 401; the message check is a backstop for
    adapters that translate it.
    """
    if getattr(exc, "status_code", None) == 401:
        return True
    msg = str(exc).lower()
    return "didn't recognize these api keys" in msg


def submit_with_auto_guards(
    broker: Broker,
    orders: list[dict],
    submit_one: Callable[[dict], Any],
    *,
    db_path: str | Path = DEFAULT_DB,
    actor: str = "dashboard",
    on_rejected: Callable[[dict, str], None] | None = None,
    decision_date: str | None = None,
) -> dict:
    """Submit a batch with the automatic kill-switch hooks watching.

    Args:
        broker: the Broker (used for cancel-all if a hook fires).
        orders: normalized order dicts (ticker/side/shares/price/reason).
        submit_one: sends ONE order; returns whatever the broker reports,
            raises ``BrokerError`` when the broker refuses it.
        on_rejected: optional callback per rejected order
            ``(order, plain_english_error)`` — e.g. keep it staged for the
            next valid open instead of dropping it.
        decision_date: YYYY-MM-DD for the deterministic order IDs recorded
            in the audit log.

    The kill switch is checked FIRST (fail-closed). Every submit, accept,
    and rejection is audit-logged. Returns a plain-English report::

        {"submitted": [...], "rejected": [(order, error)...],
         "auto_engaged": bool, "engaged_reason": str, "notice": str}

    ``notice`` is "" on a clean run, else the loud plain-English
    explanation of what happened (including any auto-engage).
    """
    from qrp.ops import audit as _audit

    report: dict = {
        "submitted": [],
        "rejected": [],
        "auto_engaged": False,
        "engaged_reason": "",
        "notice": "",
    }
    try:
        check_trading_allowed(db_path)
    except _ks.KillSwitchEngaged as exc:
        report["notice"] = str(exc)
        _audit.log_event(
            "staging_blocked",
            {
                "reason": "kill_switch",
                "detail": str(exc),
                "n_orders": len(orders),
            },
            actor=actor,
            db_path=db_path,
        )
        return report

    rejections = 0
    day = (decision_date or "")[:10]

    def auto_engage(reason: str) -> None:
        result = _ks.trigger_kill_switch(
            broker, reason, actor="auto-guard", db_path=db_path
        )
        report["auto_engaged"] = True
        report["engaged_reason"] = reason
        n = len(result["canceled"])
        report["notice"] = (
            f"{reason} The kill switch engaged itself: {n} open order(s) "
            "canceled and all trading is now stopped until you manually "
            "re-enable it on the Operations page. Your positions were not "
            "sold."
        )

    for order in orders:
        cid = client_order_id(
            str(order.get("ticker", "")),
            str(order.get("side", "")),
            day,
        )
        try:
            result = submit_one(order)
        except BrokerError as exc:
            rejections += 1
            error = str(exc)
            report["rejected"].append((order, error))
            _audit.log_event(
                "order_rejected",
                {
                    "ticker": order.get("ticker"),
                    "side": order.get("side"),
                    "shares": order.get("shares"),
                    "client_order_id": cid,
                    "error": error,
                    "rejections_in_batch": rejections,
                },
                actor=actor,
                db_path=db_path,
            )
            if on_rejected is not None:
                try:
                    on_rejected(order, error)
                except Exception:  # noqa: BLE001 - never break the batch
                    pass
            if _is_auth_failure(exc):
                auto_engage(
                    "The broker rejected the API keys (authentication "
                    "failure) during order submission."
                )
                break
            if rejections > _ks.AUTO_TRIGGER_MAX_REJECTIONS:
                auto_engage(
                    f"The broker rejected {rejections} orders in this one "
                    "batch — something is systematically wrong, not just "
                    "one bad order."
                )
                break
            continue
        except Exception as exc:  # noqa: BLE001 - the fail-safe hook
            _audit.log_event(
                "order_rejected",
                {
                    "ticker": order.get("ticker"),
                    "side": order.get("side"),
                    "shares": order.get("shares"),
                    "client_order_id": cid,
                    "error": f"unexpected problem in the submit path: {exc}",
                },
                actor=actor,
                db_path=db_path,
            )
            auto_engage(
                "An unexpected error escaped the order-submit path "
                f"({exc})."
            )
            break
        report["submitted"].append(result)
        _audit.log_event(
            "order_submitted",
            {
                "ticker": order.get("ticker"),
                "side": order.get("side"),
                "shares": order.get("shares"),
                "client_order_id": cid,
                "broker_order_id": str(
                    getattr(result, "id", "") or ""
                ),
            },
            actor=actor,
            db_path=db_path,
        )
    return report
