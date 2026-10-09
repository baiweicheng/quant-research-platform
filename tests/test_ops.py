"""Tests for roadmap P6 (ops hardening): kill switch, audit log, pre-trade
checklist, pipeline auto-guards, and their wiring into the order path.

Covers (with fake brokers — no network, no real keys, no real orders):
- kill switch cancels ALL open orders and never touches positions;
  release requires an explicit action; fail-closed on an unreadable flag;
  the STOP file flag; the order pipeline refuses while engaged;
- the audit log is append-only, hash-chained, tamper/gap-detecting, and
  scrubs secrets;
- EACH pre-trade check trips on synthetic bad input and passes on good
  input (kill switch, stale data, buying power, position limits,
  fat-finger median/absolute/drift, duplicates, no-short);
- the automatic hooks engage the kill switch on mass rejections, auth
  failure, and unexpected submit-path exceptions;
- checklist integration blocks staging end-to-end;
- deterministic client_order_id passthrough on both broker adapters.
"""

import json
from datetime import date

import pandas as pd
import pytest

from qrp.data.quality import DataHealthReport, Finding
from qrp.data.store import upsert_bars
from qrp.execution.alpaca import AlpacaBroker, AlpacaError
from qrp.execution.broker import AccountInfo, Broker, BrokerError, OrderInfo
from qrp.execution.paper import PaperBroker, PaperBrokerAdapter
from qrp.ops import audit as OPS_AUDIT
from qrp.ops import killswitch as OPS_KS
from qrp.ops import pipeline as OPS_PIPE
from qrp.ops import pretrade as OPS_PRE
from qrp.ops.killswitch import KillSwitchEngaged

DAY = "2026-10-07"


# ---------------------------------------------------------------------------
# Fake broker
# ---------------------------------------------------------------------------


class FakeBroker(Broker):
    """In-memory Broker for ops tests (no network, no database)."""

    def __init__(
        self,
        *,
        cash: float = 100_000.0,
        positions: dict | None = None,
        prices: dict | None = None,
        open_orders: list | None = None,
        history: list | None = None,
        submit_behavior: str | None = None,
        cancel_raises_for: set | None = None,
    ) -> None:
        self._cash = float(cash)
        self._positions = dict(positions or {})
        self._prices = dict(prices or {})
        self._open = list(open_orders or [])
        self._history = list(history or [])
        self.submit_behavior = submit_behavior
        self.cancel_raises_for = set(cancel_raises_for or set())
        self.submitted: list[OrderInfo] = []
        self.submit_calls = 0
        self._next_id = 1000

    # -- Broker interface ------------------------------------------------

    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            q * self._prices.get(t, 0.0) for t, q in self._positions.items()
        )
        return AccountInfo(
            equity=equity, cash=self._cash, buying_power=self._cash
        )

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def last_price(self, ticker: str, date=None) -> float:
        if ticker not in self._prices:
            raise KeyError(ticker)
        return float(self._prices[ticker])

    def place_order(
        self, ticker, side, shares, reason="", *, client_order_id=None
    ) -> OrderInfo:
        self.submit_calls += 1
        if self.submit_behavior == "auth":
            raise AlpacaError(
                "Alpaca didn't recognize these API keys.", 401
            )
        if self.submit_behavior == "boom":
            raise RuntimeError("simulated bug in the submit path")
        if isinstance(self.submit_behavior, str) and \
                self.submit_behavior.startswith("reject:"):
            raise BrokerError(
                self.submit_behavior[len("reject:"):]
            )
        self._next_id += 1
        info = OrderInfo(
            id=str(self._next_id),
            ticker=ticker,
            side=side,
            shares=float(shares),
            status="open",
            client_order_id=client_order_id or "",
            reason=reason,
        )
        self._open.append(info)
        self.submitted.append(info)
        return info

    def cancel_order(self, order_id: str) -> bool:
        if str(order_id) in self.cancel_raises_for:
            raise BrokerError("the broker refused this cancel")
        for i, o in enumerate(self._open):
            if o.id == str(order_id):
                self._open.pop(i)
                return True
        return False

    def list_orders(self, status: str = "open") -> list[OrderInfo]:
        status = (status or "open").lower()
        if status == "open":
            return list(self._open)
        if status == "all":
            return list(self._open) + list(self._history)
        return []

    def get_order(self, order_id: str) -> OrderInfo | None:
        for o in self._open + self._history:
            if o.id == str(order_id):
                return o
        return None


def _open_order(oid, ticker, side, shares=10.0, cid="") -> OrderInfo:
    return OrderInfo(
        id=str(oid), ticker=ticker, side=side, shares=shares,
        status="open", client_order_id=cid,
    )


def _history_order(oid, ticker, side, shares) -> OrderInfo:
    return OrderInfo(
        id=str(oid), ticker=ticker, side=side, shares=shares,
        status="filled",
    )


def _fresh_report() -> DataHealthReport:
    return DataHealthReport(
        checked_at="2026-10-07T00:00:00Z", tickers_checked=2, findings=[]
    )


def _stale_report() -> DataHealthReport:
    return DataHealthReport(
        checked_at="2026-10-07T00:00:00Z",
        tickers_checked=2,
        findings=[
            Finding(
                check="stale",
                severity="warning",
                ticker="AAA",
                summary="AAA's newest price is from 2026-09-20 — 12 trading "
                        "days behind 2026-10-06.",
                detail="Prices more than two trading days old ...",
                action="Re-run the download on the Market data page.",
            )
        ],
    )


def _healthy_broker(**over) -> FakeBroker:
    """Fake broker where a small BUY passes every check."""
    kw = dict(
        cash=100_000.0,
        positions={"AAA": 100.0},
        prices={"AAA": 50.0, "BBB": 20.0},
        history=[
            _history_order(i, "BBB", "BUY", 100.0) for i in range(5)
        ],
    )
    kw.update(over)
    return FakeBroker(**kw)


def _good_orders() -> list[dict]:
    return [{"ticker": "BBB", "side": "BUY", "shares": 100.0,
             "price": 20.0, "reason": "test"}]


def _check(report, name) -> str:
    return next(c.status for c in report.checks if c.name == name)


def _submit(broker, order):
    """submit_one for the pipeline: unpack the normalized order dict."""
    return broker.place_order(
        order["ticker"], order["side"], order["shares"],
        reason=order.get("reason", ""),
        client_order_id=OPS_PRE.client_order_id(
            order["ticker"], order["side"], DAY),
    )


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_trigger_cancels_all_open_orders_and_never_touches_positions(
    tmp_path,
):
    db = tmp_path / "m.db"
    broker = FakeBroker(
        positions={"AAA": 100.0, "BBB": 50.0},
        prices={"AAA": 50.0, "BBB": 20.0},
        open_orders=[
            _open_order(1, "AAA", "SELL", 10.0),
            _open_order(2, "BBB", "BUY", 5.0),
            _open_order(3, "CCC", "BUY", 7.0),
        ],
    )
    before = dict(broker.get_positions())
    result = OPS_KS.trigger_kill_switch(
        broker, "test stop", actor="test", db_path=db
    )
    assert len(result["canceled"]) == 3
    assert result["failed"] == []
    assert broker.list_orders("open") == []
    # Positions are NEVER touched by the kill switch.
    assert broker.get_positions() == before
    assert OPS_KS.is_engaged(db) is True
    state = OPS_KS.read_state(db)
    assert state["engaged"] is True
    assert state["reason"] == "test stop"
    assert state["engaged_at"]


def test_trigger_collects_cancel_failures_without_raising(tmp_path):
    db = tmp_path / "m.db"
    broker = FakeBroker(
        open_orders=[
            _open_order(1, "AAA", "SELL"),
            _open_order(2, "BBB", "BUY"),
        ],
        cancel_raises_for={"2"},
    )
    result = OPS_KS.trigger_kill_switch(
        broker, "test", actor="test", db_path=db
    )
    assert result["canceled"] == ["1"]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["id"] == "2"
    # Still engaged even though one cancel failed.
    assert OPS_KS.is_engaged(db) is True


def test_release_requires_explicit_action(tmp_path):
    db = tmp_path / "m.db"
    broker = FakeBroker()
    OPS_KS.engage("test", actor="test", db_path=db)
    assert OPS_KS.is_engaged(db) is True
    # Nothing auto-releases: canceling orders doesn't, re-engaging doesn't.
    OPS_KS.cancel_all_open_orders(broker)
    assert OPS_KS.is_engaged(db) is True
    OPS_KS.engage("again", actor="test", db_path=db)
    assert OPS_KS.is_engaged(db) is True
    # Only an explicit release clears it.
    OPS_KS.release(actor="test", db_path=db)
    assert OPS_KS.is_engaged(db) is False


def test_fail_closed_on_unreadable_flag(tmp_path):
    db = tmp_path / "m.db"
    OPS_KS.killswitch_path(db).parent.mkdir(parents=True, exist_ok=True)
    OPS_KS.killswitch_path(db).write_text("{not valid json", encoding="utf-8")
    with pytest.raises(KillSwitchEngaged):
        OPS_KS.is_engaged(db)
    with pytest.raises(KillSwitchEngaged):
        OPS_PIPE.check_trading_allowed(db)


def test_stop_file_flag_engages_and_release_clears_it(tmp_path):
    db = tmp_path / "m.db"
    assert OPS_KS.is_engaged(db) is False
    OPS_KS.stop_flag_path(db).parent.mkdir(parents=True, exist_ok=True)
    OPS_KS.stop_flag_path(db).write_text("", encoding="utf-8")
    assert OPS_KS.is_engaged(db) is True
    with pytest.raises(KillSwitchEngaged):
        OPS_PIPE.check_trading_allowed(db)
    OPS_KS.release(actor="test", db_path=db)
    assert OPS_KS.is_engaged(db) is False
    assert not OPS_KS.stop_flag_path(db).exists()


def test_pipeline_refuses_while_engaged(tmp_path):
    db = tmp_path / "m.db"
    broker = FakeBroker()
    OPS_KS.engage("test stop", actor="test", db_path=db)
    with pytest.raises(KillSwitchEngaged, match="STOPPED"):
        OPS_PIPE.check_trading_allowed(db)
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _good_orders(), lambda o: _submit(broker, o), db_path=db
    )
    assert broker.submit_calls == 0
    assert "STOPPED" in report["notice"]
    assert report["submitted"] == []


def test_killswitch_cli_status_engage_release(tmp_path, capsys):
    db = tmp_path / "m.db"
    assert OPS_KS.main(["--status", "--db", str(db)]) == 0
    assert "OFF" in capsys.readouterr().out
    # --engage with no Alpaca keys uses the local paper broker (no network).
    assert OPS_KS.main(["--engage", "--reason", "cli test",
                        "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "ENGAGED" in out
    assert "NOT sold" in out
    assert OPS_KS.is_engaged(db) is True
    assert OPS_KS.main(["--status", "--db", str(db)]) == 0
    assert "ENGAGED" in capsys.readouterr().out
    assert OPS_KS.main(["--release", "--db", str(db)]) == 0
    assert "released" in capsys.readouterr().out.lower()
    assert OPS_KS.is_engaged(db) is False


def test_engage_and_release_write_audit_events(tmp_path):
    db = tmp_path / "m.db"
    broker = FakeBroker()
    OPS_KS.trigger_kill_switch(broker, "audit test", actor="tester",
                               db_path=db)
    OPS_KS.release(actor="tester", db_path=db)
    types = [e["event_type"] for e in OPS_AUDIT.read_events(db)]
    assert "kill_switch_engaged" in types
    assert "kill_switch_cancel_report" in types
    assert "kill_switch_released" in types
    actors = {e["actor"] for e in OPS_AUDIT.read_events(db)}
    assert actors == {"tester"}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_append_only_and_chain_verifies(tmp_path):
    db = tmp_path / "m.db"
    e1 = OPS_AUDIT.log_event("signals_used", {"n": 1}, actor="t",
                             db_path=db)
    e2 = OPS_AUDIT.log_event("orders_staged", {"n": 2}, actor="t",
                             db_path=db)
    assert e1["seq"] == 1
    assert e2["seq"] == 2
    assert e2["prev_hash"] == e1["hash"]
    events = OPS_AUDIT.read_events(db)
    assert [e["seq"] for e in events] == [1, 2]
    assert [e["event_type"] for e in events] == [
        "signals_used", "orders_staged"]
    result = OPS_AUDIT.verify_chain(db)
    assert result["ok"] is True
    assert result["checked"] == 2
    assert "intact" in result["message"]


def test_audit_append_never_rewrites_history(tmp_path):
    db = tmp_path / "m.db"
    OPS_AUDIT.log_event("signals_used", {"n": 1}, db_path=db)
    before = OPS_AUDIT.audit_path(db).read_bytes()
    OPS_AUDIT.log_event("orders_staged", {"n": 2}, db_path=db)
    after = OPS_AUDIT.audit_path(db).read_bytes()
    assert after.startswith(before)
    assert after.count(b"\n") == 2


def test_audit_tamper_detected(tmp_path):
    db = tmp_path / "m.db"
    OPS_AUDIT.log_event("signals_used", {"n": 1}, db_path=db)
    OPS_AUDIT.log_event("orders_staged", {"n": 2}, db_path=db)
    OPS_AUDIT.log_event("pretrade_checks", {"ok": True}, db_path=db)
    path = OPS_AUDIT.audit_path(db)
    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["details"] = {"n": 999}  # tamper: change contents, keep hash
    lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = OPS_AUDIT.verify_chain(db)
    assert result["ok"] is False
    assert "entry 2" in result["message"]
    assert "hash does not match" in result["message"]


def test_audit_gap_detected(tmp_path):
    db = tmp_path / "m.db"
    OPS_AUDIT.log_event("signals_used", {"n": 1}, db_path=db)
    OPS_AUDIT.log_event("orders_staged", {"n": 2}, db_path=db)
    OPS_AUDIT.log_event("pretrade_checks", {"ok": True}, db_path=db)
    path = OPS_AUDIT.audit_path(db)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    result = OPS_AUDIT.verify_chain(db)
    assert result["ok"] is False
    assert "sequence break" in result["message"]


def test_audit_scrubs_secrets(tmp_path):
    db = tmp_path / "m.db"
    secret = "PK" + "A" * 40
    OPS_AUDIT.log_event(
        "keys_saved",
        {"api_key": secret, "note": f"secret={secret}"},
        db_path=db,
    )
    raw = OPS_AUDIT.audit_path(db).read_text(encoding="utf-8")
    assert secret not in raw
    assert "<redacted" in raw
    entry = OPS_AUDIT.read_events(db)[0]
    assert secret not in json.dumps(entry["details"])


def test_audit_export_csv_and_jsonl(tmp_path):
    db = tmp_path / "m.db"
    OPS_AUDIT.log_event("signals_used", {"tickers": ["AAA"]}, db_path=db)
    OPS_AUDIT.log_event("orders_staged", {"n": 3}, db_path=db)
    csv_path = OPS_AUDIT.export_csv(db, tmp_path / "out.csv")
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("seq,timestamp_utc,event_type")
    assert len(rows) == 3
    assert "orders_staged" in rows[2]
    jsonl_path = OPS_AUDIT.export_jsonl(db, tmp_path / "out.jsonl")
    entries = [
        json.loads(ln) for ln in
        jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 2
    assert entries[0]["hash"] == \
        OPS_AUDIT.read_events(db)[0]["hash"]


def test_audit_verify_empty_log(tmp_path):
    db = tmp_path / "m.db"
    result = OPS_AUDIT.verify_chain(db)
    assert result["ok"] is True
    assert result["checked"] == 0


def test_audit_read_filters(tmp_path):
    db = tmp_path / "m.db"
    for i in range(3):
        OPS_AUDIT.log_event("signals_used", {"i": i}, db_path=db)
    OPS_AUDIT.log_event("orders_staged", {"n": 1}, db_path=db)
    assert len(OPS_AUDIT.read_events(db, event_types=["orders_staged"])) \
        == 1
    assert len(OPS_AUDIT.read_events(db, limit=2)) == 2
    assert OPS_AUDIT.read_events(db, limit=2)[-1]["seq"] == 4


# ---------------------------------------------------------------------------
# Pre-trade checklist
# ---------------------------------------------------------------------------


def test_pretrade_all_checks_pass_on_good_input(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker()
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert report.ok is True
    assert {c.status for c in report.checks} == {"pass"}
    assert len(report.checks) == 7


def test_pretrade_kill_switch_blocks(tmp_path):
    db = tmp_path / "m.db"
    OPS_KS.engage("testing", actor="test", db_path=db)
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert report.ok is False
    assert _check(report, "Kill switch") == "fail"
    assert "STOPPED" in next(
        c.detail for c in report.checks if c.name == "Kill switch")


def test_pretrade_kill_switch_fail_closed(tmp_path):
    db = tmp_path / "m.db"
    OPS_KS.killswitch_path(db).parent.mkdir(parents=True, exist_ok=True)
    OPS_KS.killswitch_path(db).write_text("garbage", encoding="utf-8")
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Kill switch") == "fail"


def test_pretrade_stale_ticker_outside_batch_does_not_block(tmp_path):
    """Refined freshness gate (2026-10-08): a stale ticker that is NOT
    in the batch is noted, never blocking — it was left out of the
    target. (Was test_pretrade_stale_data_blocks before the refinement:
    under the first delisted fix ANY stale ticker failed the gate,
    which blocked staging for days after every delisting.)"""
    db = tmp_path / "m.db"
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_stale_report(), audit_log=False,
    )
    assert report.ok is True
    assert _check(report, "Data freshness") == "pass"
    detail = next(
        c.detail for c in report.checks if c.name == "Data freshness")
    assert "AAA" in detail and "left out of this batch" in detail


def test_pretrade_stale_ticker_in_batch_blocks(tmp_path):
    """Defense in depth: a stale name forced INTO the batch fails —
    its order can't be sized off an old price."""
    db = tmp_path / "m.db"
    orders = [{"ticker": "AAA", "side": "BUY", "shares": 10.0,
               "price": 50.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        orders, _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_stale_report(), audit_log=False,
    )
    assert report.ok is False
    assert _check(report, "Data freshness") == "fail"
    detail = next(
        c.detail for c in report.checks if c.name == "Data freshness")
    assert "AAA" in detail and "in this batch" in detail


def test_pretrade_no_database_blocks(tmp_path):
    db = tmp_path / "m.db"
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=DataHealthReport(
            checked_at="2026-10-07T00:00:00Z", tickers_checked=0,
            findings=[Finding(
                check="general", severity="info", ticker=None,
                summary="No database yet — nothing to check.")],
        ),
        audit_log=False,
    )
    assert _check(report, "Data freshness") == "fail"


def test_pretrade_buying_power_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker()
    orders = [{"ticker": "BBB", "side": "BUY", "shares": 10_000.0,
               "price": 20.0, "reason": ""}]  # $200k vs $100k cash
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Buying power") == "fail"
    detail = next(
        c.detail for c in report.checks if c.name == "Buying power")
    assert "98%" in detail


def test_pretrade_position_limit_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker()  # holds 100 AAA @ $50 = $5k of $105k
    orders = [{"ticker": "AAA", "side": "BUY", "shares": 200.0,
               "price": 50.0, "reason": ""}]  # -> ~14% weight
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Position limits") == "fail"
    assert "5%" in next(
        c.detail for c in report.checks if c.name == "Position limits")


def test_pretrade_duplicate_orders_in_batch_block(tmp_path):
    db = tmp_path / "m.db"
    orders = _good_orders() + _good_orders()
    report = OPS_PRE.run_pretrade_checks(
        orders, _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Position limits") == "fail"


def test_pretrade_fat_finger_median_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker()  # 5 past BUY BBB @ 100 shares
    orders = [{"ticker": "BBB", "side": "BUY", "shares": 400.0,
               "price": 20.0, "reason": ""}]  # 4x the median
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "fail"
    assert "3×" in next(
        c.detail for c in report.checks if c.name == "Fat-finger guard")


def test_pretrade_fat_finger_absolute_cap_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(
        cash=1_000_000.0,
        history=[_history_order(i, "AAA", "BUY", 10_000.0)
                 for i in range(5)],
    )
    orders = [{"ticker": "AAA", "side": "BUY", "shares": 1_000.0,
               "price": 50.0, "reason": ""}]  # $50k > $25k cap
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "fail"
    assert "$25,000" in next(
        c.detail for c in report.checks if c.name == "Fat-finger guard")


def test_pretrade_fat_finger_price_drift_blocks_buys(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(
        history=[_history_order(i, "BBB", "BUY", 100.0) for i in range(5)],
    )
    orders = [{"ticker": "BBB", "side": "BUY", "shares": 100.0,
               "price": 30.0, "reason": ""}]  # 50% above the $20 close
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "fail"


def test_pretrade_price_drift_exempts_sells(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(
        history=[_history_order(i, "AAA", "SELL", 100.0) for i in range(5)],
    )
    # A gap-down exit far below the last close must never be trapped.
    orders = [{"ticker": "AAA", "side": "SELL", "shares": 100.0,
               "price": 30.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "pass"


def test_pretrade_no_history_warns_not_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(history=[])
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "warn"
    assert report.ok is True


def test_pretrade_duplicate_open_order_id_blocks(tmp_path):
    db = tmp_path / "m.db"
    cid = OPS_PRE.client_order_id("BBB", "BUY", DAY)
    broker = _healthy_broker(
        open_orders=[_open_order(1, "BBB", "BUY", 100.0, cid=cid)])
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Duplicate protection") == "fail"
    assert cid in next(
        c.detail for c in report.checks
        if c.name == "Duplicate protection")


def test_pretrade_duplicate_open_order_ticker_side_fallback(tmp_path):
    db = tmp_path / "m.db"
    # Broker doesn't surface client_order_id ("") — the (ticker, side)
    # fallback still catches the duplicate.
    broker = _healthy_broker(
        open_orders=[_open_order(1, "BBB", "BUY", 50.0, cid="")])
    report = OPS_PRE.run_pretrade_checks(
        _good_orders(), broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Duplicate protection") == "fail"


def test_pretrade_no_short_blocks(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker()  # holds 100 AAA
    orders = [{"ticker": "AAA", "side": "SELL", "shares": 150.0,
               "price": 50.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "No shorting / no margin") == "fail"


def test_client_order_id_format():
    assert OPS_PRE.client_order_id("aapl", "buy", "2026-10-07") == \
        "rebalance-20261007-AAPL-BUY"


def test_pretrade_writes_audit_event(tmp_path):
    db = tmp_path / "m.db"
    OPS_PRE.run_pretrade_checks(
        _good_orders(), _healthy_broker(), db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=True, actor="tester",
    )
    events = OPS_AUDIT.read_events(db, event_types=["pretrade_checks"])
    assert len(events) == 1
    assert events[0]["details"]["ok"] is True
    assert events[0]["actor"] == "tester"
    assert len(events[0]["details"]["checks"]) == 7


# ---------------------------------------------------------------------------
# Pipeline auto-guards
# ---------------------------------------------------------------------------


def _five_orders() -> list[dict]:
    return [
        {"ticker": t, "side": "BUY", "shares": 10.0, "price": 20.0,
         "reason": ""}
        for t in ["AAA", "BBB", "CCC", "DDD", "EEE"]
    ]


def test_pipeline_happy_path_submits_and_logs(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(prices={"AAA": 20.0, "BBB": 20.0, "CCC": 20.0,
                                     "DDD": 20.0, "EEE": 20.0})
    orders = _five_orders()
    report = OPS_PIPE.submit_with_auto_guards(
        broker, orders, lambda o: _submit(broker, o), db_path=db,
        actor="tester", decision_date=DAY,
    )
    assert len(report["submitted"]) == 5
    assert report["rejected"] == []
    assert report["auto_engaged"] is False
    assert OPS_KS.is_engaged(db) is False
    submitted = OPS_AUDIT.read_events(db, event_types=["order_submitted"])
    assert len(submitted) == 5
    assert submitted[0]["details"]["client_order_id"] == \
        "rebalance-20261007-AAA-BUY"
    assert submitted[0]["actor"] == "tester"


def test_pipeline_auto_engages_on_mass_rejections(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(
        submit_behavior="reject:broker says no",
        open_orders=[_open_order(1, "ZZZ", "BUY", 1.0)],
    )
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _five_orders(), lambda o: _submit(broker, o), db_path=db,
        decision_date=DAY,
    )
    # > 3 rejections in one batch -> the kill switch engages itself.
    assert report["auto_engaged"] is True
    assert OPS_KS.is_engaged(db) is True
    assert "rejected" in report["notice"]
    assert "positions were not sold" in report["notice"]
    # The pre-existing open order was canceled by the auto-engage.
    assert broker.list_orders("open") == []
    rejected = OPS_AUDIT.read_events(db, event_types=["order_rejected"])
    assert len(rejected) == 4  # the 4th rejection trips the guard
    assert OPS_AUDIT.read_events(
        db, event_types=["kill_switch_engaged"])


def test_pipeline_auto_engages_on_auth_failure(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(submit_behavior="auth")
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _five_orders(), lambda o: _submit(broker, o), db_path=db,
        decision_date=DAY,
    )
    assert report["auto_engaged"] is True
    assert OPS_KS.is_engaged(db) is True
    assert broker.submit_calls == 1  # stops at the first auth failure
    assert "API keys" in report["engaged_reason"]


def test_pipeline_auto_engages_on_unexpected_exception(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(submit_behavior="boom")
    # Must not propagate: engage + loud plain-English notice instead.
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _five_orders(), lambda o: _submit(broker, o), db_path=db,
        decision_date=DAY,
    )
    assert report["auto_engaged"] is True
    assert OPS_KS.is_engaged(db) is True
    assert "unexpected" in report["notice"].lower()
    assert "Traceback" not in report["notice"]


def test_pipeline_kill_switch_checked_before_any_submit(tmp_path):
    db = tmp_path / "m.db"
    OPS_KS.engage("pre-engaged", actor="test", db_path=db)
    broker = _healthy_broker()
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _five_orders(), lambda o: _submit(broker, o), db_path=db,
        decision_date=DAY,
    )
    assert broker.submit_calls == 0
    assert report["auto_engaged"] is False  # was already engaged
    assert "STOPPED" in report["notice"]


def test_pipeline_on_rejected_callback_runs(tmp_path):
    db = tmp_path / "m.db"
    broker = _healthy_broker(submit_behavior="reject:bad timing")
    kept = []
    report = OPS_PIPE.submit_with_auto_guards(
        broker, _five_orders()[:2], lambda o: _submit(broker, o), db_path=db,
        on_rejected=lambda o, e: kept.append((o["ticker"], e)),
        decision_date=DAY,
    )
    assert len(kept) == 2
    assert kept[0] == ("AAA", "bad timing")
    # Two rejections stay under the auto-engage threshold.
    assert report["auto_engaged"] is False


# ---------------------------------------------------------------------------
# End-to-end: checklist gates staging; adapters carry the order id
# ---------------------------------------------------------------------------


def test_checklist_blocks_staging_end_to_end(tmp_path):
    """The staging gate: a failing checklist means nothing is submitted.

    The planted failure is a stale ticker inside the batch itself —
    since the 2026-10-08 refinement, a stale name OUTSIDE the batch is
    noted but no longer blocks (see
    test_pretrade_stale_ticker_outside_batch_does_not_block).
    """
    db = tmp_path / "m.db"
    broker = _healthy_broker()
    orders = [{"ticker": "AAA", "side": "BUY", "shares": 100.0,
               "price": 50.0, "reason": "test"}]
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date=DAY,
        health_report=_stale_report(),  # AAA's data is stale; AAA is buying
    )
    assert report.ok is False
    submitted = []
    if report.ok:  # the staging flow honors the gate
        for o in orders:
            submitted.append(broker.place_order(
                o["ticker"], o["side"], o["shares"], reason=o["reason"]))
    assert submitted == []
    assert broker.submit_calls == 0
    assert "stale" in report.plain_english_summary().lower()


def test_local_adapter_end_to_end(tmp_path):
    """Real PaperBroker + adapter: staging, deterministic ids, kill switch."""
    db = tmp_path / "paper.db"
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    upsert_bars(
        pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": 100.0, "High": 101.0, "Low": 99.0,
            "Close": 100.0, "Volume": 1_000,
        }),
        "AAA", db,
    )
    paper = PaperBroker(db)
    adapter = PaperBrokerAdapter(paper)
    positions_before = dict(adapter.get_positions())
    info = adapter.place_order("AAA", "BUY", 10.0, reason="test")
    assert info.status == "open"
    staged_date = adapter._broker.pending_orders()[0]["staged_date"]
    assert info.client_order_id == f"rebalance-{staged_date}-AAA-BUY"
    # The checklist sees the staged order as a duplicate.
    report = OPS_PRE.run_pretrade_checks(
        [{"ticker": "AAA", "side": "BUY", "shares": 5.0,
          "price": 100.0, "reason": ""}],
        adapter, db_path=db, decision_date=date.today().isoformat(),
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Duplicate protection") == "fail"
    # Kill switch cancels the pending order, positions untouched.
    result = OPS_KS.trigger_kill_switch(
        adapter, "test", actor="test", db_path=db)
    assert result["canceled"] == [info.id]
    assert adapter.list_orders("open") == []
    assert adapter.get_positions() == positions_before


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else b"{}"

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _mock_alpaca(monkeypatch, tmp_path, responses):
    calls = []
    pending = list(responses)

    def fake_request(self, method, url, headers=None, timeout=None,
                     **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})
        return pending.pop(0)

    monkeypatch.setattr("requests.Session.request", fake_request)
    keys_path = tmp_path / "keys.json"
    broker = AlpacaBroker(
        "PKTESTKEYID123456", "x" * 40, keys_path=keys_path,
        db_path=tmp_path / "m.db",
    )
    return broker, calls


def _order_payload(**over):
    body = {
        "id": "order-1", "client_order_id": "qrp-abc123",
        "symbol": "AAA", "qty": "10", "side": "buy",
        "type": "market", "time_in_force": "opg", "status": "accepted",
        "filled_avg_price": None, "submitted_at": "2026-10-07T00:00:00Z",
    }
    body.update(over)
    return body


def test_alpaca_place_order_uses_given_client_order_id(
    monkeypatch, tmp_path
):
    broker, calls = _mock_alpaca(
        monkeypatch, tmp_path, [_FakeResponse(200, _order_payload())])
    info = broker.place_order(
        "AAA", "BUY", 10.0, reason="test", time_in_force="opg",
        client_order_id="rebalance-20261007-AAA-BUY",
    )
    sent = calls[0]["kwargs"]["json"]
    assert sent["client_order_id"] == "rebalance-20261007-AAA-BUY"
    assert sent["time_in_force"] == "opg"


def test_alpaca_place_order_random_id_when_none_given(
    monkeypatch, tmp_path
):
    broker, calls = _mock_alpaca(
        monkeypatch, tmp_path, [_FakeResponse(200, _order_payload())])
    broker.place_order("AAA", "BUY", 10.0, time_in_force="opg")
    sent = calls[0]["kwargs"]["json"]
    assert sent["client_order_id"].startswith("qrp-")


def test_alpaca_surfaces_client_order_id(monkeypatch, tmp_path):
    broker, _ = _mock_alpaca(
        monkeypatch, tmp_path,
        [_FakeResponse(200, _order_payload(
            client_order_id="rebalance-20261007-AAA-BUY"))])
    info = broker.place_order(
        "AAA", "BUY", 10.0, time_in_force="opg",
        client_order_id="rebalance-20261007-AAA-BUY",
    )
    assert info.client_order_id == "rebalance-20261007-AAA-BUY"


def test_alpaca_list_orders_surface_client_order_ids(
    monkeypatch, tmp_path
):
    broker, _ = _mock_alpaca(
        monkeypatch, tmp_path,
        [_FakeResponse(200, [_order_payload(
            id="o1", client_order_id="rebalance-20261007-AAA-BUY"),
            _order_payload(id="o2", client_order_id="")])])
    orders = broker.list_orders("open")
    assert orders[0].client_order_id == "rebalance-20261007-AAA-BUY"
    assert orders[1].client_order_id == ""


def test_ops_files_live_next_to_db(tmp_path):
    from qrp.ops import ops_files

    files = ops_files(tmp_path / "data" / "market.db")
    assert files.killswitch.parent == tmp_path / "data"
    assert files.killswitch.name == "killswitch.json"
    assert files.stop_flag.name == "STOP"
    assert files.audit_log.name == "audit.log"


def test_pretrade_fat_finger_compares_notional_not_shares(tmp_path):
    """Audit 2.3: the median-size check compares dollar notional.

    History: 5 BUY AAA @ 100 shares filled at $10 ($1,000 notional each).
    A 100-share $50 order ($5,000 = 5x the usual notional) must block even
    though the share count matches history exactly; a 500-share $2 order
    ($1,000 = 1x) must pass even though the share count is 5x history.
    """
    db = tmp_path / "m.db"

    def hist(oid):
        o = _history_order(oid, "AAA", "BUY", 100.0)
        o.filled_avg_price = 10.0
        return o

    broker = _healthy_broker(
        cash=1_000_000.0,
        prices={"AAA": 10.0, "BBB": 50.0, "CCC": 2.0},
        history=[hist(i) for i in range(5)],
    )
    big_notional = [{"ticker": "BBB", "side": "BUY", "shares": 100.0,
                     "price": 50.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        big_notional, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report, "Fat-finger guard") == "fail"
    detail = next(
        c.detail for c in report.checks if c.name == "Fat-finger guard")
    assert "$5,000" in detail

    small_notional = [{"ticker": "CCC", "side": "BUY", "shares": 500.0,
                       "price": 2.0, "reason": ""}]
    report2 = OPS_PRE.run_pretrade_checks(
        small_notional, broker, db_path=db, decision_date=DAY,
        health_report=_fresh_report(), audit_log=False,
    )
    assert _check(report2, "Fat-finger guard") == "pass"
