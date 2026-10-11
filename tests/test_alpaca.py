"""Tests for the Alpaca paper-trading adapter (roadmap P2).

Covers, with the HTTP layer mocked:
- auth headers on every request,
- GET /v2/account parsing (equity / cash / buying power),
- GET /v2/positions parsing (empty, whole, fractional),
- POST /v2/orders payload (symbol, qty-as-string, side, type=market,
  time_in_force=opg for explicit market-on-open calls; the adapter's
  "auto" default maps whole shares to opg and fractions to day),
- staging sends plain day market orders with fractional quantities
  straight to Alpaca — filling right away when the market is open, or
  queued by Alpaca for the next open when it's closed (the clock is
  read only for the user-facing disclosure line, never as a gate),
- GET /v2/orders, DELETE /v2/orders/{id}, GET /v2/orders/{id},
- error translation into plain English (401 bad keys, 403 insufficient
  buying power, 404, 422 incl. the fractional hint, 429 rate limit, 5xx,
  connection errors) with keys never leaking into messages,
- the live-URL guard,
- broker-ABC conformance for both the Alpaca adapter and the local
  PaperBroker,
- key storage (0600, gitignored, never logged),
- the staging queue (only orders Alpaca refused at submission wait
  there, retried as day orders on page visits),
- the read-only --verify CLI (no order placement, ever),
plus AppTest coverage for the Brokers page (save, test-connection
success/failure) and the Paper trading page's Alpaca-vs-local switching.
"""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from qrp.data.store import upsert_bars
from qrp.execution.alpaca import (
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    QUEUE_MAX_WAIT_DAYS,
    AlpacaBroker,
    AlpacaError,
    _verify_command,
    drop_queued_order,
    flush_staging_queue,
    load_paper_keys,
    load_staging_queue,
    queue_staged_order,
    save_paper_keys,
    save_staging_queue,
)
from qrp.execution.broker import AccountInfo, Broker, OrderInfo
from qrp.execution.paper import PaperBroker, PaperBrokerAdapter
from qrp.ops import killswitch as OPS_KS
from qrp.risk.guards import RiskConfig
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H
from qrp.ui.app import _alpaca_dust_note, _alpaca_sendable_orders
from qrp.ui.helpers import _redact_for_log

KEY_ID = "PKTESTKEYID123456"
SECRET = "test-secret-not-real-123"


# ---------------------------------------------------------------------------
# Mocked HTTP plumbing
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else b"{}"

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _mock_http(monkeypatch, tmp_path, responses, keys_name="keys.json"):
    """AlpacaBroker whose transport replays canned responses in order.

    Returns (broker, calls). Each call records method, url, headers and
    kwargs. Responses may be _FakeResponse or an Exception to raise
    (e.g. requests.ConnectionError).
    """
    calls = []
    pending = list(responses)

    def fake_request(self, method, url, headers=None, timeout=None, **kwargs):
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "kwargs": kwargs,
            }
        )
        resp = pending.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr("requests.Session.request", fake_request)
    keys_path = tmp_path / keys_name
    broker = AlpacaBroker(
        KEY_ID, SECRET, keys_path=keys_path, db_path=tmp_path / "m.db"
    )
    return broker, calls, keys_path


def _account_payload(**over):
    body = {
        "id": "acct-1",
        "equity": "100000.50",
        "cash": "25000.25",
        "buying_power": "50000.00",
        "portfolio_value": "100000.50",
    }
    body.update(over)
    return body


def _order_payload(**over):
    body = {
        "id": "ord-1",
        "symbol": "AAPL",
        "qty": "10",
        "side": "buy",
        "type": "market",
        "time_in_force": "opg",
        "status": "accepted",
        "filled_avg_price": None,
        "submitted_at": "2024-01-08T14:00:00Z",
    }
    body.update(over)
    return body


def test_clock_helpers_classify_open_closed_unknown():
    """The clock helpers feed only the pre-staging disclosure line:
    True/False when readable, None (never a guess) when not."""
    from qrp.execution.alpaca import clock_market_open, fetch_market_clock

    assert clock_market_open({"is_open": True}) is True
    assert clock_market_open({"is_open": False}) is False
    assert clock_market_open({"is_open": "yes"}) is None
    assert clock_market_open({}) is None
    assert clock_market_open(None) is None

    class NoClock:
        pass

    class BrokenClock:
        def get_clock(self):
            raise ConnectionError("boom")

    class GoodClock:
        def get_clock(self):
            return {"is_open": True}

    assert fetch_market_clock(NoClock()) is None
    assert fetch_market_clock(BrokenClock()) is None
    assert fetch_market_clock(GoodClock()) == {"is_open": True}


# ---------------------------------------------------------------------------
# Auth headers on every request
# ---------------------------------------------------------------------------


def test_auth_headers_on_every_request(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _account_payload()),
            _FakeResponse(200, []),
            _FakeResponse(200, _order_payload()),
            _FakeResponse(200, []),
            _FakeResponse(204),
            _FakeResponse(200, _order_payload(status="filled")),
        ],
    )
    broker.get_account()
    broker.get_positions()
    broker.place_order("AAPL", "BUY", 10, time_in_force="opg")
    broker.list_orders("open")
    broker.cancel_order("ord-1")
    broker.get_order("ord-1")
    assert len(calls) == 6
    for call in calls:
        assert call["headers"]["APCA-API-KEY-ID"] == KEY_ID
        assert call["headers"]["APCA-API-SECRET-KEY"] == SECRET
        assert call["url"].startswith(PAPER_BASE_URL)
        assert "api.alpaca.markets" not in call["url"] or "paper-" in call["url"]


# ---------------------------------------------------------------------------
# Account / positions parsing
# ---------------------------------------------------------------------------


def test_get_account_parses_equity_cash_buying_power(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, _account_payload())]
    )
    account = broker.get_account()
    assert account.equity == pytest.approx(100000.50)
    assert account.cash == pytest.approx(25000.25)
    assert account.buying_power == pytest.approx(50000.00)
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == PAPER_BASE_URL + "/v2/account"


def test_get_account_malformed_shape_is_plain_english_not_assert(
    monkeypatch, tmp_path
):
    """A non-dict /v2/account body must raise AlpacaError, never a bare
    AssertionError traceback."""
    broker, _, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, ["not", "a", "dict"])]
    )
    with pytest.raises(AlpacaError, match="didn't expect"):
        broker.get_account()


def test_get_account_malformed_numbers_are_plain_english(
    monkeypatch, tmp_path
):
    """Unparseable cash/buying_power values must raise AlpacaError."""
    broker, _, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(200, _account_payload(cash="nonsense", buying_power=None))],
    )
    with pytest.raises(AlpacaError, match="couldn't read"):
        broker.get_account()


def test_place_order_malformed_confirmation_is_plain_english(
    monkeypatch, tmp_path
):
    broker, _, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, "ok-string-not-dict")]
    )
    with pytest.raises(AlpacaError, match="didn't expect"):
        broker.place_order("AAPL", "BUY", 10.0, time_in_force="opg")


def test_get_positions_empty(monkeypatch, tmp_path):
    broker, _, _ = _mock_http(monkeypatch, tmp_path, [_FakeResponse(200, [])])
    assert broker.get_positions() == {}


def test_get_positions_parses_whole_and_fractional(monkeypatch, tmp_path):
    broker, _, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                200,
                [
                    {"symbol": "aapl", "qty": "10"},
                    {"symbol": "MSFT", "qty": "7.905"},
                    {"symbol": "ZERO", "qty": "0"},
                ],
            )
        ],
    )
    positions = broker.get_positions()
    assert positions == {"AAPL": 10.0, "MSFT": 7.905}


def test_get_clock(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                200,
                {
                    "is_open": False,
                    "next_open": "2024-01-08T14:30:00Z",
                    "next_close": "2024-01-08T21:00:00Z",
                },
            )
        ],
    )
    clock = broker.get_clock()
    assert clock["is_open"] is False
    assert calls[0]["url"] == PAPER_BASE_URL + "/v2/clock"


# ---------------------------------------------------------------------------
# Order placement payload: the market-on-open mapping
# ---------------------------------------------------------------------------


def test_place_order_opg_payload(monkeypatch, tmp_path):
    """Explicit market-on-open placement: type=market, time_in_force=opg."""
    broker, calls, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, _order_payload())]
    )
    info = broker.place_order("aapl", "buy", 10, time_in_force="opg")
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == PAPER_BASE_URL + "/v2/orders"
    payload = call["kwargs"]["json"]
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == "10"  # Alpaca wants qty as a decimal string
    assert payload["side"] == "buy"
    assert payload["type"] == "market"
    assert payload["time_in_force"] == "opg"
    assert info.ticker == "AAPL"
    assert info.side == "BUY"
    assert info.shares == pytest.approx(10.0)


def test_place_order_auto_tif_whole_shares_is_opg(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, _order_payload())]
    )
    broker.place_order("AAPL", "BUY", 25)
    assert calls[0]["kwargs"]["json"]["time_in_force"] == "opg"


def test_place_order_auto_tif_fractional_is_day(monkeypatch, tmp_path):
    """Alpaca only accepts fractions on day orders — the adapter knows."""
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(200, _order_payload(qty="7.905", time_in_force="day"))],
    )
    broker.place_order("AAPL", "BUY", 7.905)
    payload = calls[0]["kwargs"]["json"]
    assert payload["time_in_force"] == "day"
    assert payload["qty"] == "7.905"


def test_place_order_sell_side(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(200, _order_payload(side="sell"))],
    )
    info = broker.place_order("AAPL", "SELL", 5, time_in_force="opg")
    assert calls[0]["kwargs"]["json"]["side"] == "sell"
    assert info.side == "SELL"


def test_place_order_records_reason_locally(monkeypatch, tmp_path):
    broker, calls, keys_path = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(200, _order_payload())]
    )
    info = broker.place_order(
        "AAPL", "BUY", 10, reason="unit test reason", time_in_force="opg"
    )
    assert info.reason == "unit test reason"
    sidecar = keys_path.parent / "alpaca_order_reasons.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["ord-1"] == "unit test reason"


def test_place_order_rejects_bad_side_and_qty():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        broker = AlpacaBroker(
            KEY_ID, SECRET, keys_path=Path(td) / "k.json"
        )
        with pytest.raises(AlpacaError, match="BUY or SELL"):
            broker.place_order("AAPL", "HOLD", 10)
        with pytest.raises(AlpacaError, match="positive"):
            broker.place_order("AAPL", "BUY", 0)


# ---------------------------------------------------------------------------
# list / cancel / get order
# ---------------------------------------------------------------------------


def test_list_orders_open(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                200,
                [
                    _order_payload(
                        id="o1", status="accepted", symbol="AAPL", qty="10"
                    ),
                    _order_payload(
                        id="o2",
                        status="filled",
                        symbol="MSFT",
                        qty="5",
                        filled_avg_price="150.5",
                    ),
                ],
            )
        ],
    )
    orders = broker.list_orders("open")
    assert calls[0]["kwargs"]["params"]["status"] == "open"
    assert [o.id for o in orders] == ["o1"]
    assert orders[0].status == "open"


def test_list_orders_all_includes_everything(monkeypatch, tmp_path):
    broker, _, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                200,
                [
                    _order_payload(id="o1", status="accepted"),
                    _order_payload(id="o2", status="filled"),
                ],
            )
        ],
    )
    assert [o.id for o in broker.list_orders("all")] == ["o1", "o2"]


def test_cancel_order_true_on_204_false_on_404(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(204), _FakeResponse(404, {"message": "not found"})],
    )
    assert broker.cancel_order("o1") is True
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["url"] == PAPER_BASE_URL + "/v2/orders/o1"
    assert broker.cancel_order("missing") is False


def test_get_order_roundtrip_and_missing(monkeypatch, tmp_path):
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(
                200,
                _order_payload(
                    status="filled", filled_avg_price="150.5", qty="7.905"
                ),
            ),
            _FakeResponse(404, {"message": "not found"}),
        ],
    )
    info = broker.get_order("ord-1")
    assert isinstance(info, OrderInfo)
    assert info.status == "filled"
    assert info.filled_avg_price == pytest.approx(150.5)
    assert info.shares == pytest.approx(7.905)
    assert calls[0]["url"] == PAPER_BASE_URL + "/v2/orders/ord-1"
    assert broker.get_order("missing") is None


# ---------------------------------------------------------------------------
# Error translation: plain English, no keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,payload,fragment",
    [
        (
            401,
            {"message": "unauthorized."},
            "didn't recognize these API keys",
        ),
        (
            403,
            {"message": "insufficient buying power"},
            "not enough buying power",
        ),
        (
            422,
            {"message": "fractional qty not allowed for opg"},
            "fractional shares on plain day orders",
        ),
        (
            422,
            {"message": "order qty is invalid"},
            "rejected the order",
        ),
        (429, {"message": "rate limit"}, "rate-limiting"),
        (500, {"message": "boom"}, "having trouble"),
        (400, {"message": "symbol not found"}, "symbol not found"),
    ],
)
def test_error_translation_plain_english(
    monkeypatch, tmp_path, status, payload, fragment
):
    broker, _, _ = _mock_http(
        monkeypatch, tmp_path, [_FakeResponse(status, payload)]
    )
    with pytest.raises(AlpacaError) as excinfo:
        broker.get_account()
    message = str(excinfo.value)
    assert fragment in message
    assert KEY_ID not in message
    assert SECRET not in message
    assert "traceback" not in message.lower()


def test_connection_error_plain_english(monkeypatch, tmp_path):
    import requests

    broker, _, _ = _mock_http(
        monkeypatch, tmp_path, [requests.ConnectionError("dns")]
    )
    with pytest.raises(AlpacaError, match="Could not reach Alpaca"):
        broker.get_account()


def test_timeout_plain_english(monkeypatch, tmp_path):
    import requests

    broker, _, _ = _mock_http(
        monkeypatch, tmp_path, [requests.Timeout("slow")]
    )
    with pytest.raises(AlpacaError, match="took too long"):
        broker.get_account()


# ---------------------------------------------------------------------------
# Live guard
# ---------------------------------------------------------------------------


def test_live_base_url_refused_by_default(tmp_path):
    with pytest.raises(AlpacaError, match="paper trading only"):
        AlpacaBroker(
            KEY_ID,
            SECRET,
            base_url=LIVE_BASE_URL,
            keys_path=tmp_path / "k.json",
        )


def test_live_base_url_allowed_with_explicit_opt_in(tmp_path):
    broker = AlpacaBroker(
        KEY_ID,
        SECRET,
        base_url=LIVE_BASE_URL,
        allow_live=True,
        keys_path=tmp_path / "k.json",
    )
    assert broker._base_url == LIVE_BASE_URL


def test_default_base_url_is_paper(tmp_path):
    broker = AlpacaBroker(KEY_ID, SECRET, keys_path=tmp_path / "k.json")
    assert broker._base_url == PAPER_BASE_URL


# ---------------------------------------------------------------------------
# Broker ABC conformance — both brokers, identical surface
# ---------------------------------------------------------------------------


_INTERFACE = (
    "get_account",
    "get_positions",
    "place_order",
    "cancel_order",
    "list_orders",
    "get_order",
)


def test_both_brokers_satisfy_the_abc():
    assert issubclass(AlpacaBroker, Broker)
    assert issubclass(PaperBrokerAdapter, Broker)
    for name in _INTERFACE:
        assert callable(getattr(AlpacaBroker, name)), name
        assert callable(getattr(PaperBrokerAdapter, name)), name
    # The ABC itself cannot be instantiated without the interface.
    with pytest.raises(TypeError):
        Broker()  # type: ignore[abstract]


def test_paper_broker_adapter_matches_interface_semantics(tmp_path):
    """The local broker behind the same interface: account, positions,
    place/cancel/list/get order all behave like the Alpaca side."""
    db = tmp_path / "paper.db"
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 1_000,
            }
        ),
        "AAA",
        db,
    )
    inner = PaperBroker(db)
    adapter = PaperBrokerAdapter(inner)
    account = adapter.get_account()
    assert isinstance(account, AccountInfo)
    assert account.equity == pytest.approx(100_000.0)
    assert adapter.get_positions() == {}
    info = adapter.place_order("AAA", "BUY", 10, reason="test")
    assert isinstance(info, OrderInfo)
    assert info.ticker == "AAA" and info.side == "BUY"
    assert info.shares == pytest.approx(10.0)
    # Staged for the next open — not filled yet, so no position; the order
    # is open, exactly like an accepted Alpaca market-on-open order.
    assert info.status == "open"
    assert adapter.get_positions() == {}
    assert adapter.get_order(info.id) is not None
    assert [o.id for o in adapter.list_orders("open")] == [info.id]
    assert adapter.cancel_order(info.id) is True
    assert adapter.cancel_order("nope") is False


# ---------------------------------------------------------------------------
# Key storage: 0600, gitignored, never logged
# ---------------------------------------------------------------------------


def test_keys_saved_with_owner_only_permissions(tmp_path):
    path = tmp_path / "data" / "alpaca_paper_keys.json"
    save_paper_keys(KEY_ID, SECRET, path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_paper_keys(path) == (KEY_ID, SECRET)


def test_keys_file_is_gitignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text()
    assert "alpaca_paper_keys.json" in text
    assert "alpaca_order_reasons.json" in text
    assert "alpaca_staging_queue.json" in text


def test_keys_never_appear_in_error_text(monkeypatch, tmp_path, caplog):
    import logging

    broker, _, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(401, {"message": "unauthorized."})],
    )
    with caplog.at_level(logging.INFO):
        with pytest.raises(AlpacaError) as excinfo:
            broker.get_account()
    assert KEY_ID not in str(excinfo.value)
    assert SECRET not in str(excinfo.value)
    assert KEY_ID not in caplog.text
    assert SECRET not in caplog.text


def test_missing_keys_give_plain_english_guidance(tmp_path):
    with pytest.raises(AlpacaError, match="Brokers page"):
        AlpacaBroker("", "", keys_path=tmp_path / "k.json")


# ---------------------------------------------------------------------------
# Staging queue: rejected orders stay staged for the next valid open
# ---------------------------------------------------------------------------


def test_queue_roundtrip(tmp_path):
    keys_path = tmp_path / "keys.json"
    assert load_staging_queue(keys_path) == []
    queue_staged_order("aapl", "buy", 10.0, reason="r1", keys_path=keys_path)
    queue_staged_order("MSFT", "SELL", 5.0, keys_path=keys_path)
    entries = load_staging_queue(keys_path)
    assert len(entries) == 2
    assert entries[0]["ticker"] == "AAPL"
    assert entries[0]["side"] == "BUY"
    assert entries[0]["shares"] == pytest.approx(10.0)
    assert entries[0]["reason"] == "r1"
    assert entries[0]["attempts"] == 0
    assert drop_queued_order(0, keys_path) is True
    assert [e["ticker"] for e in load_staging_queue(keys_path)] == ["MSFT"]
    assert drop_queued_order(99, keys_path) is False


def test_flush_sends_queued_orders_as_day(monkeypatch, tmp_path):
    """Queue flush sends plain day market orders (fractional-capable)
    straight to Alpaca, without consulting the clock — the clock never
    gates a send."""
    broker, calls, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),  # no open orders
            _FakeResponse(200, _order_payload()),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.5, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 1
    assert result["waiting"] == 0
    assert result["notes"] == []
    assert not any("/v2/clock" in c["url"] for c in calls), (
        "the flush must not gate on the market clock"
    )
    payload = calls[1]["kwargs"]["json"]
    assert payload["time_in_force"] == "day"
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == "10.5"  # fractional quantities pass through
    assert load_staging_queue(keys_path) == []


def test_queue_entries_carry_no_hold_metadata(tmp_path):
    """The staging queue is a plain retry list for rejected submissions:
    no planned-open / hold fields (mid-day staging never holds orders
    back, so there is nothing for such fields to mean)."""
    keys_path = tmp_path / "keys.json"
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    (entry,) = load_staging_queue(keys_path)
    assert "planned_open" not in entry
    assert set(entry) == {
        "ticker", "side", "shares", "reason", "staged_at",
        "attempts", "last_error", "client_order_id",
    }


def test_is_fractionable_reads_asset_flag_and_caches(monkeypatch, tmp_path):
    """One GET /v2/assets/{symbol} per ticker; the flag drives the answer."""
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, {"symbol": "AAPL", "fractionable": True}),
            _FakeResponse(200, {"symbol": "MSFT", "fractionable": False}),
        ],
    )
    assert broker.is_fractionable("aapl") is True
    assert broker.is_fractionable("AAPL") is True  # cached — no 2nd call
    assert broker.is_fractionable("MSFT") is False
    assert len(calls) == 2
    assert calls[0]["url"] == PAPER_BASE_URL + "/v2/assets/AAPL"
    assert calls[1]["url"] == PAPER_BASE_URL + "/v2/assets/MSFT"


def test_is_fractionable_none_when_check_fails(monkeypatch, tmp_path):
    broker, _, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(404, {"message": "asset not found"})],
    )
    assert broker.is_fractionable("ZZZZ") is None

    broker, calls, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, [_order_payload(status="accepted")]),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 0
    assert len(calls) == 1  # open-orders check only; no POST, no clock
    assert load_staging_queue(keys_path) == []
    assert any(
        "already exists at Alpaca" in note for note in result["notes"]
    ), "the silent skip must now say what it skipped"


def test_flush_keeps_rejected_orders_with_plain_english_note(
    monkeypatch, tmp_path
):
    broker, _, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),
            _FakeResponse(403, {"message": "insufficient buying power"}),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 0
    assert result["waiting"] == 1
    entries = load_staging_queue(keys_path)
    assert len(entries) == 1
    assert entries[0]["attempts"] == 1
    assert "buying power" in entries[0]["last_error"]
    assert KEY_ID not in entries[0]["last_error"]


def test_flush_drops_orders_after_wait_days_not_visits(monkeypatch, tmp_path):
    """The drop policy is time-based: a fresh entry survives many failed
    flushes (casual daytime page visits); an entry that has waited
    QUEUE_MAX_WAIT_DAYS days is dropped with a plain-English note."""
    from datetime import datetime, timedelta, timezone

    broker, _, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),
            _FakeResponse(403, {"message": "insufficient buying power"}),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    entries = load_staging_queue(keys_path)
    # Simulate many failed attempts on a FRESH entry — it must survive.
    entries[0]["attempts"] = 50
    save_staging_queue(entries, keys_path)
    result = flush_staging_queue(broker)
    assert result["waiting"] == 1
    assert load_staging_queue(keys_path)[0]["attempts"] == 51


def test_flush_drops_entry_older_than_max_wait_days(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    broker, _, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),
            _FakeResponse(403, {"message": "insufficient buying power"}),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    entries = load_staging_queue(keys_path)
    old = datetime.now(timezone.utc) - timedelta(
        days=QUEUE_MAX_WAIT_DAYS + 1
    )
    entries[0]["staged_at"] = old.strftime("%Y-%m-%dT%H:%M:%SZ")
    save_staging_queue(entries, keys_path)
    result = flush_staging_queue(broker)
    assert result["waiting"] == 0
    assert load_staging_queue(keys_path) == []
    assert any("Gave up" in note for note in result["notes"])
    assert any(
        str(QUEUE_MAX_WAIT_DAYS) in note for note in result["notes"]
    )


def test_flush_backfills_missing_staged_at_as_new(monkeypatch, tmp_path):
    """Entries written before timestamps existed are treated as new,
    not dropped."""
    broker, _, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),
            _FakeResponse(403, {"message": "insufficient buying power"}),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    entries = load_staging_queue(keys_path)
    del entries[0]["staged_at"]
    save_staging_queue(entries, keys_path)
    result = flush_staging_queue(broker)
    assert result["waiting"] == 1
    assert load_staging_queue(keys_path)[0]["staged_at"]  # backfilled


# ---------------------------------------------------------------------------
# Read-only --verify CLI: never places an order
# ---------------------------------------------------------------------------


def test_verify_without_keys_is_a_clean_ask(tmp_path, capsys):
    rc = _verify_command(tmp_path / "missing.json")
    assert rc == 2
    out = capsys.readouterr().out
    assert "No Alpaca paper keys saved yet" in out
    assert "Brokers page" in out


def test_verify_all_checks_pass_and_never_trades(
    monkeypatch, tmp_path, capsys
):
    keys_path = tmp_path / "keys.json"
    save_paper_keys(KEY_ID, SECRET, keys_path)
    broker, calls, _ = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _account_payload()),
            _FakeResponse(
                200,
                {"is_open": False, "next_open": "2024-01-08T14:30:00Z"},
            ),
            _FakeResponse(200, [{"symbol": "AAPL", "qty": "10"}]),
        ],
        keys_name="keys.json",
    )
    assert broker._keys_path == keys_path
    rc = _verify_command(keys_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Authentication: OK" in out
    assert "Market clock: OK" in out
    assert "Positions: OK" in out
    assert "No orders were placed" in out
    # Read-only: only GETs, never a POST/DELETE.
    assert calls and all(c["method"] == "GET" for c in calls)


def test_verify_reports_bad_keys_plainly(monkeypatch, tmp_path, capsys):
    keys_path = tmp_path / "keys.json"
    save_paper_keys("WRONG", "WRONG", keys_path)
    _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(401, {"message": "unauthorized."})],
        keys_name="keys.json",
    )
    rc = _verify_command(keys_path)
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "didn't recognize these API keys" in out
    assert "WRONG" not in out


# ---------------------------------------------------------------------------
# AppTest: Brokers page
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _seed_db(path: Path) -> None:
    """Five trading days of flat prices for AAA ($100) and BBB ($50)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": base,
                    "High": base * 1.01,
                    "Low": base * 0.99,
                    "Close": base,
                    "Volume": 1_000,
                }
            ),
            ticker,
            path,
        )
    sig_dates = pd.date_range("2024-01-02", periods=5, freq="B")
    upsert_signals(
        pd.Series([0.5] * 5, index=sig_dates), "AAA", "momentum", path
    )
    upsert_signals(
        pd.Series([0.3] * 5, index=sig_dates), "BBB", "momentum", path
    )


def _seed_db_fresh(path: Path) -> None:
    """Recent trading days of flat prices for AAA ($100) and BBB ($50).

    Fresh enough that the P6 data-freshness gate passes — staging tests
    need current data, since staging on stale data is now blocked.
    """
    dates = pd.date_range(end=pd.Timestamp.today().normalize(),
                          periods=5, freq="B")
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": base,
                    "High": base * 1.01,
                    "Low": base * 0.99,
                    "Close": base,
                    "Volume": 1_000,
                }
            ),
            ticker,
            path,
        )


@pytest.fixture
def clean_alpaca_files():
    """The repo data dir must not keep Alpaca key/queue files — or P6 ops
    files (audit log, kill-switch flag) — between tests."""
    names = (
        "alpaca_paper_keys.json",
        "alpaca_staging_queue.json",
        "alpaca_order_reasons.json",
        "audit.log",
        "killswitch.json",
        "STOP",
    )
    data = _repo_root() / "data"
    for name in names:
        p = data / name
        if p.exists():
            p.unlink()
    yield
    for name in names:
        p = data / name
        if p.exists():
            p.unlink()


@pytest.fixture
def clean_repo_db():
    p = _repo_root() / "data" / "market.db"
    if p.exists():
        p.unlink()
    yield p
    if p.exists():
        p.unlink()


def _app():
    return AppTest.from_file(str(_repo_root() / "qrp" / "ui" / "app.py"))


def test_brokers_page_without_keys_shows_not_configured(clean_alpaca_files):
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    assert not any(
        "Paper API keys are saved" in str(s.value) for s in at.success
    )
    assert any(
        "alpaca.markets" in str(c.value).lower() for c in at.caption
    ), "expected the where-to-get-keys note"
    # TextInput proto Type enum: DEFAULT=0, PASSWORD=1.
    assert at.text_input(key="brokers_key_id").proto.type == 1
    assert at.text_input(key="brokers_secret").proto.type == 1
    assert any(
        "intentionally not available" in str(i.value) for i in at.info
    ), "expected the no-live-trading note"


def test_brokers_page_save_keys_writes_0600_file(clean_alpaca_files):
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    at.text_input(key="brokers_key_id").set_value(KEY_ID)
    at.text_input(key="brokers_secret").set_value(SECRET)
    at.button(key="brokers_save").click().run()
    assert not at.exception
    path = _repo_root() / "data" / "alpaca_paper_keys.json"
    assert path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_paper_keys(path) == (KEY_ID, SECRET)
    # The save reruns the page; it now reflects the saved state.
    assert any(
        "Paper API keys are saved" in str(s.value) for s in at.success
    )


def test_brokers_page_save_requires_both_fields(clean_alpaca_files):
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    at.text_input(key="brokers_key_id").set_value(KEY_ID)
    at.button(key="brokers_save").click().run()
    assert not at.exception
    assert any("both the key ID" in str(e.value) for e in at.error)
    assert not (_repo_root() / "data" / "alpaca_paper_keys.json").exists()


def test_brokers_page_test_connection_success(
    clean_alpaca_files, monkeypatch
):
    monkeypatch.setattr(
        H,
        "test_alpaca_key_pair",
        lambda kid, sec, timeout=15.0: (
            True,
            AccountInfo(equity=100000.0, cash=60000.0, buying_power=120000.0),
        ),
    )
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    at.text_input(key="brokers_key_id").set_value("K")
    at.text_input(key="brokers_secret").set_value("S")
    at.button(key="brokers_test").click().run()
    assert not at.exception
    assert any(
        "Connected" in str(s.value) and "buying power" in str(s.value).lower()
        for s in at.success
    ), "expected the plain-English account status"


def test_brokers_page_test_connection_failure(
    clean_alpaca_files, monkeypatch
):
    monkeypatch.setattr(
        H,
        "test_alpaca_key_pair",
        lambda kid, sec, timeout=15.0: (
            False,
            "Alpaca didn't recognize these API keys.",
        ),
    )
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    at.text_input(key="brokers_key_id").set_value("K")
    at.text_input(key="brokers_secret").set_value("S")
    at.button(key="brokers_test").click().run()
    assert not at.exception
    assert any(
        "Couldn't connect" in str(e.value) for e in at.error
    ), "expected the plain-English failure"


def test_brokers_page_remove_keys(clean_alpaca_files):
    path = _repo_root() / "data" / "alpaca_paper_keys.json"
    save_paper_keys(KEY_ID, SECRET, path)
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    assert any(
        "Paper API keys are saved" in str(s.value) for s in at.success
    )
    at.button(key="brokers_remove").click().run()
    assert not at.exception
    assert not path.exists()


# ---------------------------------------------------------------------------
# AppTest: Paper trading page — Alpaca vs local switching
# ---------------------------------------------------------------------------


def test_paper_page_without_keys_shows_local_broker(
    clean_alpaca_files, clean_repo_db
):
    _seed_db(clean_repo_db)
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Paper trading"
    at.run()
    assert not at.exception
    # No Alpaca: the local broker is the visible default, not tucked away.
    assert not any(
        "Local simulated broker" in str(e.label) for e in at.expander
    )
    assert any(
        "Brokers page" in str(i.value) for i in at.info
    ), "expected the tip pointing at the Brokers page"


def test_paper_page_with_alpaca_hides_local_behind_expander(
    clean_alpaca_files, clean_repo_db, tmp_path
):
    _seed_db(clean_repo_db)
    save_paper_keys(KEY_ID, SECRET, _repo_root() / "data" / "alpaca_paper_keys.json")
    fake_broker = SimpleNamespace(
        get_account=lambda: AccountInfo(
            equity=100000.0, cash=60000.0, buying_power=120000.0
        ),
        get_positions=lambda: {},
        get_position_details=lambda: [],
        list_orders=lambda status="open": [],
        _keys_path=tmp_path / "queue.json",
    )
    at = _app()
    at.session_state["alpaca_conn"] = {
        "configured": True,
        "ok": True,
        "broker": fake_broker,
        "account": AccountInfo(
            equity=100000.0, cash=60000.0, buying_power=120000.0
        ),
        "error": "",
    }
    at.session_state[H.NAV_PAGE_KEY] = "Paper trading"
    at.run()
    assert not at.exception
    assert any(
        "Paper trading through Alpaca" in str(m.value) for m in at.markdown
    ), "expected the Alpaca-first header"
    assert any(
        "Local simulated broker (offline testing)" in str(e.label)
        for e in at.expander
    ), "expected the local broker tucked behind the labeled expander"
    assert any(
        "Alpaca PAPER account" in str(c.value) for c in at.caption
    ), "expected the Alpaca account snapshot"


def test_paper_page_with_broken_alpaca_falls_back_to_local(
    clean_alpaca_files, clean_repo_db
):
    _seed_db(clean_repo_db)
    save_paper_keys(KEY_ID, SECRET, _repo_root() / "data" / "alpaca_paper_keys.json")
    at = _app()
    at.session_state["alpaca_conn"] = {
        "configured": True,
        "ok": False,
        "broker": None,
        "account": None,
        "error": "Alpaca didn't recognize these API keys.",
    }
    at.session_state[H.NAV_PAGE_KEY] = "Paper trading"
    at.run()
    assert not at.exception
    assert any(
        "connection failed" in str(w.value).lower() for w in at.warning
    ), "expected the plain-English fallback warning"
    assert not any(
        "Local simulated broker" in str(e.label) for e in at.expander
    ), "local broker should be the visible default on fallback"


# ---------------------------------------------------------------------------
# Unit: _alpaca_sendable_orders (fractional quantities, $1 dust,
# non-fractionable fallback, exact full closes)
# ---------------------------------------------------------------------------


def _mk_order(ticker, action, shares, price=100.0, reason="why"):
    return SimpleNamespace(
        ticker=ticker, action=action, shares=shares, price=price,
        reason=reason,
    )


def test_sendable_orders_fractional_exact_no_round_down():
    """Fractional quantities pass through (4 decimal places) — the 10.7
    order no longer loses its 0.7 to whole-share rounding."""
    orders = [
        _mk_order("AAA", "BUY", 10.7, price=100.0),
        _mk_order("BBB", "SELL", 3.25, price=50.0),
        _mk_order("CCC", "BUY", 10.123456, price=10.0),
    ]
    sendable, notes = _alpaca_sendable_orders(orders)
    assert [(s["ticker"], s["side"], s["shares"]) for s in sendable] == [
        ("AAA", "BUY", 10.7),
        ("BBB", "SELL", 3.25),
        ("CCC", "BUY", 10.1235),  # rounded to 4 decimal places
    ]
    assert notes == []


def test_sendable_orders_dust_under_one_dollar():
    """Alpaca's minimum order is $1 — anything smaller is skipped."""
    orders = [
        _mk_order("AAA", "BUY", 0.005, price=100.0),  # $0.50
        _mk_order("BBB", "SELL", 0.009, price=50.0),  # $0.45
        _mk_order("CCC", "BUY", 2.0, price=100.0),
    ]
    sendable, notes = _alpaca_sendable_orders(orders)
    assert [(s["ticker"], s["shares"]) for s in sendable] == [("CCC", 2.0)]
    assert len(notes) == 1
    assert "under $1" in notes[0]
    assert "minimum order size" in notes[0]


def test_sendable_orders_empty_and_all_dust():
    assert _alpaca_sendable_orders([]) == ([], [])
    sendable, notes = _alpaca_sendable_orders(
        [_mk_order("AAA", "BUY", 0.001, price=100.0)]
    )
    assert sendable == []
    assert len(notes) == 1 and "under $1" in notes[0]


def test_sendable_orders_non_fractionable_floors_with_note():
    orders = [
        _mk_order("AAA", "BUY", 10.7, price=100.0),
        _mk_order("BBB", "BUY", 10.0, price=50.0),
    ]
    sendable, notes = _alpaca_sendable_orders(
        orders, is_fractionable=lambda t: t == "BBB"
    )
    assert [(s["ticker"], s["shares"]) for s in sendable] == [
        ("AAA", 10),  # floored to whole shares
        ("BBB", 10.0),
    ]
    assert len(notes) == 1
    assert "AAA" in notes[0]
    assert "doesn't support fractional shares" in notes[0]
    assert "rounded down to whole shares" in notes[0]


def test_sendable_orders_non_fractionable_sub_share_skipped():
    sendable, notes = _alpaca_sendable_orders(
        [_mk_order("AAA", "BUY", 0.5, price=100.0)],
        is_fractionable=lambda t: False,
    )
    assert sendable == []
    assert len(notes) == 1
    assert "under one whole share" in notes[0]


def test_sendable_orders_unknown_fractionable_falls_back_safely():
    """Check failed (None) => whole shares, note says it couldn't check."""
    sendable, notes = _alpaca_sendable_orders(
        [_mk_order("AAA", "BUY", 10.7, price=100.0)],
        is_fractionable=lambda t: None,
    )
    assert [(s["ticker"], s["shares"]) for s in sendable] == [("AAA", 10)]
    assert "Couldn't check" in notes[0]


def test_sendable_orders_exact_full_close_leaves_nothing():
    """Closing a position sends the EXACT held quantity — a 4-dp rounding
    would strand a fractional residue forever."""
    held = 3.14159265
    sendable, notes = _alpaca_sendable_orders(
        [_mk_order("AAA", "SELL", held, price=10.0)],
        positions={"AAA": held},
    )
    assert len(sendable) == 1
    assert sendable[0]["shares"] == held  # exact, not 3.1416
    assert notes == []


def test_sendable_orders_full_close_below_one_dollar_still_sends():
    """A sub-$1 leftover position can still be closed from here."""
    sendable, _ = _alpaca_sendable_orders(
        [_mk_order("AAA", "SELL", 0.004, price=100.0)],
        positions={"AAA": 0.004},
    )
    assert [(s["ticker"], s["shares"]) for s in sendable] == [
        ("AAA", 0.004)
    ]


def test_sendable_orders_partial_sell_stays_fractional():
    sendable, _ = _alpaca_sendable_orders(
        [_mk_order("AAA", "SELL", 2.25, price=10.0)],
        positions={"AAA": 5.5},
    )
    assert [(s["ticker"], s["shares"]) for s in sendable] == [
        ("AAA", 2.25)
    ]


def test_sendable_orders_sell_all_exact_quantity_zero_residual():
    """Sell-all spec case (2026-10-09): held 7.9054321 must go out as
    exactly 7.9054321 — NOT rounded to 4 dp (7.9054 would strand
    0.0000321 shares forever). The residual after the sell is zero."""
    held = 7.9054321
    sendable, notes = _alpaca_sendable_orders(
        [_mk_order("AAA", "SELL", held, price=10.0)],
        positions={"AAA": held},
    )
    assert len(sendable) == 1
    assert sendable[0]["shares"] == held  # verbatim, no rounding
    assert held - sendable[0]["shares"] == 0
    assert notes == []
    # A partial sell of the same holding keeps the 4-dp rounding.
    partial, _ = _alpaca_sendable_orders(
        [_mk_order("AAA", "SELL", 2.718281828, price=10.0)],
        positions={"AAA": held},
    )
    assert partial[0]["shares"] == 2.7183


def test_dust_note_names_the_one_dollar_minimum():
    note = _alpaca_dust_note(2)
    assert "under $1" in note
    assert "minimum order size" in note
    # Under fractional trading there is no "can never be sold from
    # here" leftover anymore — the old dashboard wording is gone.
    assert "paper dashboard" not in note
    assert "can't be sold" not in note


# ---------------------------------------------------------------------------
# Unit: data_staleness_note + _redact_for_log
# ---------------------------------------------------------------------------


def _seed_db_dates(path: Path, dates: list) -> None:
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Open": base,
                    "High": base * 1.01,
                    "Low": base * 0.99,
                    "Close": base,
                    "Volume": 1_000,
                }
            ),
            ticker,
            path,
        )


def test_data_staleness_note_fresh_db(tmp_path):
    from datetime import date as real_date

    today = real_date.today().isoformat()
    db = tmp_path / "m.db"
    _seed_db_dates(db, [today])
    assert H.data_staleness_note(db) == ""


def test_data_staleness_note_old_db_warns(tmp_path):
    from datetime import date as real_date, timedelta

    old = (real_date.today() - timedelta(days=10)).isoformat()
    db = tmp_path / "m.db"
    _seed_db_dates(db, [old])
    note = H.data_staleness_note(db)
    assert "days old" in note
    assert "old closes" in note
    assert "Market data page" in note


def test_data_staleness_note_empty_db(tmp_path):
    assert H.data_staleness_note(tmp_path / "empty.db") == ""


def test_data_staleness_note_weekend_grace(tmp_path, monkeypatch):
    """Friday data checked on Monday is 1 trading day behind — no warning."""
    from datetime import date as real_date

    db = tmp_path / "m.db"
    _seed_db_dates(db, ["2024-01-05"])  # a Friday

    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2024, 1, 8)  # the following Monday

    monkeypatch.setattr(H, "date", FrozenDate)
    assert H.data_staleness_note(db) == ""


def test_data_staleness_note_counts_trading_days(tmp_path, monkeypatch):
    """Wednesday data checked the following Monday is 3 trading days
    behind — warn."""
    from datetime import date as real_date

    db = tmp_path / "m.db"
    _seed_db_dates(db, ["2024-01-03"])  # a Wednesday

    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2024, 1, 8)  # the following Monday

    monkeypatch.setattr(H, "date", FrozenDate)
    assert "days old" in H.data_staleness_note(db)


def test_redact_for_log_scrubs_labeled_secrets():
    assert _redact_for_log("secret=MYSECRETVALUE") == "secret=<redacted>"
    assert (
        _redact_for_log('{"secret": "MYSECRETVALUE"}')
        == '{"secret": "<redacted>"}'
    )
    assert (
        _redact_for_log("api_key=PKTESTKEYID123456")
        == "api_key=<redacted>"
    )


def test_redact_for_log_scrubs_bare_tokens():
    token = "A" * 40
    out = _redact_for_log(f"request failed with {token} embedded")
    assert token not in out
    assert "<redacted-token>" in out


def test_redact_for_log_leaves_plain_messages_alone():
    msg = "Connection timed out after 15 seconds."
    assert _redact_for_log(msg) == msg


# ---------------------------------------------------------------------------
# AppTest: Brokers save-failure path + secret clearing
# ---------------------------------------------------------------------------


def test_brokers_page_save_failure_is_plain_english(
    clean_alpaca_files, monkeypatch
):
    def boom(key_id, secret, db_path):
        raise AlpacaError("Could not save the API keys to /tmp/x: disk full.")

    monkeypatch.setattr(H, "save_alpaca_paper_keys_for_ui", boom)
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    at.text_input(key="brokers_key_id").set_value(KEY_ID)
    at.text_input(key="brokers_secret").set_value(SECRET)
    at.button(key="brokers_save").click().run()
    assert not at.exception, "a save failure must never raise a traceback"
    assert any(
        "Could not save" in str(e.value) for e in at.error
    ), "expected the plain-English save error"


def test_brokers_page_save_clears_typed_secrets(clean_alpaca_files):
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Brokers"
    at.run()
    assert not at.exception
    at.text_input(key="brokers_key_id").set_value(KEY_ID)
    at.text_input(key="brokers_secret").set_value(SECRET)
    at.button(key="brokers_save").click().run()
    assert not at.exception
    assert not at.session_state.get("brokers_key_id"), (
        "the typed key ID must not linger in session state after save"
    )
    assert not at.session_state.get("brokers_secret"), (
        "the typed secret must not linger in session state after save"
    )


# ---------------------------------------------------------------------------
# AppTest: Alpaca staging flow (_alpaca_preview / _alpaca_stage), the
# waiting queue UI, double-staging, and flush-on-visit — all with a fake
# broker, no network.
# ---------------------------------------------------------------------------


def _repo_queue_path() -> Path:
    return _repo_root() / "data" / "alpaca_staging_queue.json"


def _fake_alpaca_broker(
    place_behavior="ok", positions=None, market_open=False,
    clock_error=False, fractionable=None,
):
    """SimpleNamespace broker standing in for AlpacaBroker.

    place_behavior="ok" records the call and returns an open OrderInfo;
    "reject" raises AlpacaError like an order refused at submission.
    positions: {ticker: shares} the fake account holds (the P6 no-short
    check reads it). market_open drives the fake clock (2024-01-05 is
    the staged "today"; next open is Monday 2024-01-08 either way).
    fractionable maps ticker -> bool for the asset check (default all
    fractionable).
    """
    calls = []
    held = dict(positions or {})
    frac = dict(fractionable or {})

    def place_order(
        ticker, side, shares, reason="", time_in_force=None,
        client_order_id=None,
    ):
        calls.append(
            {
                "ticker": ticker,
                "side": side,
                "shares": shares,
                "reason": reason,
                "time_in_force": time_in_force,
                "client_order_id": client_order_id,
            }
        )
        if place_behavior == "reject":
            raise AlpacaError(
                "Alpaca rejected the order: not enough buying power in "
                "the paper account for it."
            )
        return OrderInfo(
            id="order-1", ticker=ticker, side=side, shares=shares,
            status="open", filled_avg_price=None, submitted_at=None,
            reason=reason, client_order_id=client_order_id or "",
        )

    def get_clock():
        if clock_error:
            raise AlpacaError(
                "Could not reach Alpaca's paper trading server."
            )
        return {
            "timestamp": "2024-01-05T15:00:00-05:00",
            "is_open": market_open,
            "next_open": "2024-01-08T09:30:00-05:00",
            "next_close": (
                "2024-01-05T16:00:00-05:00"
                if market_open
                else "2024-01-08T16:00:00-05:00"
            ),
        }

    broker = SimpleNamespace(
        get_account=lambda: AccountInfo(
            equity=100000.0, cash=60000.0, buying_power=120000.0
        ),
        get_positions=lambda: dict(held),
        get_position_details=lambda: [],
        list_orders=lambda status="open": [],
        place_order=place_order,
        # Conflict-aware submit (BUG-09): the fake never 422s, so this is
        # just place_order with the (info, note) contract.
        place_order_resolving_conflict=(
            lambda ticker, side, shares, reason="", time_in_force="day",
            client_order_id=None, max_revisions=5: (
                place_order(
                    ticker, side, shares, reason=reason,
                    time_in_force=time_in_force,
                    client_order_id=client_order_id,
                ),
                "",
            )
        ),
        cancel_order=lambda order_id: True,
        get_clock=get_clock,
        is_fractionable=lambda t: frac.get(str(t).upper(), True),
        _keys_path=_repo_queue_path(),
        cash=60000.0,
        positions=dict(held),
        risk_config=RiskConfig(),
        last_price=lambda t, date=None: {"AAA": 100.0, "BBB": 50.0}[
            t.upper()
        ],
    )
    broker._calls = calls
    return broker


def _paper_page_with_fake_broker(
    clean_repo_db, monkeypatch, broker, orders
):
    """Paper trading page driven by a fake Alpaca broker.

    compute_rebalance_orders is stubbed to return `orders`; the queue file
    lives in the repo data dir (cleaned by clean_alpaca_files).
    """
    _seed_db(clean_repo_db)
    save_paper_keys(
        KEY_ID, SECRET, _repo_root() / "data" / "alpaca_paper_keys.json"
    )
    monkeypatch.setattr(
        H,
        "resolve_staging_timing",
        lambda db_path=None: ("2024-01-05", "2024-01-05"),
    )
    monkeypatch.setattr(
        H,
        "build_target_for_ui",
        lambda *a, **k: (
            SimpleNamespace(weights={"AAA": 0.6, "BBB": 0.4}),
            {},
            "2024-01-05",
        ),
    )
    monkeypatch.setattr(
        H, "compute_rebalance_orders", lambda *a, **k: list(orders)
    )
    at = _app()
    at.session_state["alpaca_conn"] = {
        "configured": True,
        "ok": True,
        "broker": broker,
        "account": AccountInfo(
            equity=100000.0, cash=60000.0, buying_power=120000.0
        ),
        "error": "",
    }
    at.session_state[H.NAV_PAGE_KEY] = "Paper trading"
    at.run()
    assert not at.exception
    return at


def _stage_orders():
    return [
        SimpleNamespace(
            ticker="AAA", action="BUY", shares=10.7, price=100.0,
            reason="Buy AAA: new position",
        ),
        SimpleNamespace(
            ticker="BBB", action="SELL", shares=3.0, price=50.0,
            reason="Sell BBB: trim",
        ),
        SimpleNamespace(
            ticker="CCC", action="BUY", shares=0.02, price=25.0,
            reason="Buy CCC: tiny",
        ),
    ]


def test_alpaca_preview_shows_fractional_shares_and_dust_note(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    broker = _fake_alpaca_broker()
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    at.button(key="paper_alpaca_preview").click().run()
    assert not at.exception
    assert broker._calls == [], "preview must never send orders"
    tables = [df.value for df in at.dataframe]
    assert tables, "expected the preview table"
    preview = tables[0]
    shares_by_ticker = dict(zip(preview["Ticker"], preview["Shares"]))
    assert shares_by_ticker["AAA"] == "10.7", (
        "fractional shares are shown as-is — no whole-share round-down"
    )
    assert shares_by_ticker["BBB"] == "3"
    assert "CCC" not in shares_by_ticker, "a $0.50 order is under $1 dust"
    value_by_ticker = dict(zip(preview["Ticker"], preview["Est. value"]))
    assert value_by_ticker["AAA"] == "$1,070.00", (
        "estimated value matches the fractional target exactly"
    )
    assert any(
        "minimum order size" in str(i.value) for i in at.info
    ), "expected the under-$1 dust note in plain English"
    # The seeded DB is old (2024 dates) — the stale-data nudge must show.
    assert any(
        "days old" in str(w.value) for w in at.warning
    ), "expected the stale-data nudge near staging"


def test_alpaca_stage_sends_day_fractional_and_queues_rejections(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    # P6: staging now passes the pre-trade checklist first — the fake
    # account holds the BBB shares being sold (no-short check) and the
    # prices are fresh (data-freshness check). The fake clock says the
    # market is closed, so orders go out right away.
    broker = _fake_alpaca_broker(
        place_behavior="reject", positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    # Fractional shares as plain day market orders — never rounded to
    # whole shares, never anything but "day".
    # (The trailing rerun flushes the queue once more, retrying the two
    # rejected orders — hence 4 calls for 2 staged orders.)
    staged_calls = broker._calls[:2]
    assert [(c["ticker"], c["side"]) for c in staged_calls] == [
        ("AAA", "BUY"),
        ("BBB", "SELL"),
    ]
    assert all(c["time_in_force"] == "day" for c in broker._calls)
    assert staged_calls[0]["shares"] == pytest.approx(10.7), (
        "the fractional 0.7 survives — no whole-share round-down"
    )
    # P6: every order carries its deterministic client_order_id.
    assert [c["client_order_id"] for c in staged_calls] == [
        "rebalance-20240105-AAA-BUY",
        "rebalance-20240105-BBB-SELL",
    ]
    # Both rejections are kept in the local retry queue.
    queued = H.alpaca_queued_staging_orders()
    assert [(e["ticker"], e["side"]) for e in queued] == [
        ("AAA", "BUY"),
        ("BBB", "SELL"),
    ]
    assert any(
        "Waiting to retry to Alpaca" in str(s.value) for s in at.subheader
    ), "expected the visible retry queue"
    # The rejection reason is recorded durably on the queued entries in
    # plain English (it renders in the queue display on the next visit).
    assert all(
        "buying power" in e["last_error"] for e in queued
    ), "expected the plain-English rejection reason on the queued entries"


def test_alpaca_stage_open_market_sends_immediately(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Mid-day staging (his 2026-10-09 'allow mid-day fills' decision):
    orders go straight to Alpaca as day market orders and fill right
    away. Nothing is held back locally; only rejections would wait."""
    broker = _fake_alpaca_broker(market_open=True, positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    staged_calls = broker._calls[:2]
    assert [(c["ticker"], c["side"]) for c in staged_calls] == [
        ("AAA", "BUY"),
        ("BBB", "SELL"),
    ]
    assert all(c["time_in_force"] == "day" for c in broker._calls)
    assert staged_calls[0]["shares"] == pytest.approx(10.7), (
        "fractional quantities go out mid-day too"
    )
    assert H.alpaca_queued_staging_orders() == [], (
        "nothing is held back locally while the market is open"
    )


def test_alpaca_stage_sends_even_when_clock_unreadable(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """The clock only feeds the disclosure line: when it can't be read,
    staging still sends immediately (the wording covers both cases)."""
    broker = _fake_alpaca_broker(clock_error=True, positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    assert len(broker._calls) == 2, "an unreadable clock must not hold orders"
    assert all(c["time_in_force"] == "day" for c in broker._calls)
    assert H.alpaca_queued_staging_orders() == []
    assert any(
        "Couldn't check whether the market is open" in str(i.value)
        for i in at.info
    ), "expected the conditional clock-unreadable disclosure"


def test_alpaca_double_staging_skips_with_note(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Staging twice while an order waits in the queue must not create a
    duplicate queue entry — it skips with a plain-English note."""
    # P6: the pre-trade checklist runs first — fresh prices and held
    # shares so the checklist passes and the queue path is exercised.
    broker = _fake_alpaca_broker(
        place_behavior="reject", positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    assert len(H.alpaca_queued_staging_orders()) == 2
    # Stage the exact same orders again.
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    assert len(H.alpaca_queued_staging_orders()) == 2, (
        "no duplicate queue entries on double-staging"
    )
    assert any(
        "already waiting to be retried" in str(i.value) for i in at.info
    ), "expected the plain-English double-stage note"


def test_alpaca_stage_blocked_while_kill_switch_engaged(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """P6: with the kill switch engaged, staging refuses before anything
    is computed or sent — plain-English message, zero broker calls."""
    from qrp.data.store import DEFAULT_DB

    broker = _fake_alpaca_broker(
        place_behavior="ok", positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    OPS_KS.engage("test stop", actor="test", db_path=DEFAULT_DB)
    try:
        at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
        at.button(key="paper_alpaca_stage").click().run()
        assert not at.exception
        assert broker._calls == [], "no order may reach the broker"
        assert any(
            "STOPPED" in str(e.value) for e in at.error
        ), "expected the plain-English kill-switch refusal"
        assert H.alpaca_queued_staging_orders() == []
    finally:
        OPS_KS.release(actor="test", db_path=DEFAULT_DB)


def test_operations_kill_switch_round_trip(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Operations page: STOP trading (two-step confirm) engages the kill
    switch; the release path re-enables it. No real broker involved."""
    from qrp.data.store import DEFAULT_DB

    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Operations"
    at.run()
    assert not at.exception
    assert not OPS_KS.is_engaged(DEFAULT_DB)
    # Two-step: arm, then confirm.
    at.button(key="ops_stop").click().run()
    assert not at.exception
    assert not OPS_KS.is_engaged(DEFAULT_DB), \
        "arming must not engage by itself"
    at.button(key="ops_stop_confirm").click().run()
    assert not at.exception
    assert OPS_KS.is_engaged(DEFAULT_DB)
    # Release requires the explicit confirmation checkbox.
    at.checkbox(key="ops_release_confirm").set_value(True).run()
    at.button(key="ops_release").click().run()
    assert not at.exception
    assert not OPS_KS.is_engaged(DEFAULT_DB)


def test_alpaca_queue_discard_button(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    broker = _fake_alpaca_broker(place_behavior="reject")
    H.queue_alpaca_staged_order("AAA", "BUY", 10.0, reason="test reason")
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    # The entry survives the flush-on-visit (rejected) and is displayed.
    assert any(
        "Waiting to retry to Alpaca" in str(s.value) for s in at.subheader
    )
    at.button(key="paper_alpaca_drop_0").click().run()
    assert not at.exception
    assert H.alpaca_queued_staging_orders() == [], (
        "the Discard button must remove the queued order"
    )


def test_alpaca_flush_on_visit_sends_waiting_orders(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """A queued order is sent automatically on the next page visit."""
    broker = _fake_alpaca_broker(place_behavior="ok")
    H.queue_alpaca_staged_order("AAA", "BUY", 10.0, reason="test reason")
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    assert not at.exception
    assert H.alpaca_queued_staging_orders() == []
    assert broker._calls, "the waiting order should have been sent"
    assert broker._calls[0]["ticker"] == "AAA"
    assert broker._calls[0]["time_in_force"] == "day"


def test_alpaca_retry_queue_section_shows_waiting_orders(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Orders Alpaca refused wait under 'Waiting to retry to Alpaca',
    with fractional share counts shown (no hold banner, no target
    morning — mid-day staging never holds orders back)."""
    broker = _fake_alpaca_broker(place_behavior="reject")
    H.queue_alpaca_staged_order("AAA", "BUY", 10.5, reason="test reason")
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    assert not at.exception
    assert any(
        "Waiting to retry to Alpaca" in str(s.value) for s in at.subheader
    )
    body = " ".join(str(m.value) for m in at.markdown)
    assert "10.5 sh" in body
    assert "waiting to send" not in body.lower()
    assert len(H.alpaca_queued_staging_orders()) == 1


def test_alpaca_market_status_line_branches_on_clock(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Before confirming, the staging area says plainly what will
    happen: fill right away (market open), queued at Alpaca for the
    next open (closed), or conditional wording when the clock can't
    be read. Disclosure only — it never blocks the send."""
    broker = _fake_alpaca_broker(market_open=True)
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    assert any(
        "market is open right now" in str(i.value)
        and "fill right away" in str(i.value)
        for i in at.info
    ), "expected the market-open disclosure"

    broker2 = _fake_alpaca_broker(market_open=False)
    at2 = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker2, _stage_orders()
    )
    assert any(
        "market is closed right now" in str(i.value)
        and "next market open" in str(i.value)
        for i in at2.info
    ), "expected the market-closed disclosure"

    broker3 = _fake_alpaca_broker(clock_error=True)
    at3 = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker3, _stage_orders()
    )
    assert any(
        "Couldn't check whether the market is open" in str(i.value)
        for i in at3.info
    ), "expected the clock-unreadable disclosure"


def test_alpaca_flush_dedup_note_when_already_open(
    clean_alpaca_files, clean_repo_db, monkeypatch, tmp_path
):
    """Flush skips a queued order that is already open at Alpaca — and now
    says so instead of dropping it silently."""
    broker = _fake_alpaca_broker(place_behavior="ok")
    broker.list_orders = lambda status="open": [
        OrderInfo(
            id="o9", ticker="AAA", side="BUY", shares=10.0, status="open",
            filled_avg_price=None, submitted_at=None, reason="",
        )
    ]
    keys_path = tmp_path / "q.json"
    broker._keys_path = keys_path
    queue_staged_order("AAA", "BUY", 10.0, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 0
    assert broker._calls == []
    assert any(
        "already exists at Alpaca" in n for n in result["notes"]
    )

def test_alpaca_preview_survives_confirm_checkbox_rerun(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """Backlog item 26 / user report 2026-10-10: ticking the 'I understand'
    checkbox reruns the page, which used to wipe the preview at exactly the
    moment of confirmation. The preview must persist; it clears only once
    orders are staged."""
    broker = _fake_alpaca_broker(positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.button(key="paper_alpaca_preview").click().run()
    assert not at.exception

    def preview_tables():
        return [
            df.value for df in at.dataframe
            if "Ticker" in list(df.value.columns)
            and "Est. value" in list(df.value.columns)
        ]

    assert preview_tables(), "preview table should show after clicking Preview"
    # Tick the confirm checkbox: the rerun must NOT wipe the preview.
    at.checkbox(key="paper_alpaca_confirm").set_value(True).run()
    assert not at.exception
    assert preview_tables(), (
        "preview must survive the confirm-checkbox rerun"
    )
    assert broker._calls == [], "preview must never send orders"
    # Staging clears the preview (the account state has moved on).
    at.button(key="paper_alpaca_stage").click().run()
    assert not at.exception
    assert not preview_tables(), "preview must clear once orders are staged"


def test_alpaca_preview_dropped_when_inputs_change(
    clean_alpaca_files, clean_repo_db, monkeypatch
):
    """A stored preview is fingerprinted: changing the strategy after
    previewing must not show the old preview."""
    broker = _fake_alpaca_broker(positions={"BBB": 10.0})
    at = _paper_page_with_fake_broker(
        clean_repo_db, monkeypatch, broker, _stage_orders()
    )
    _seed_db_fresh(clean_repo_db)
    at.button(key="paper_alpaca_preview").click().run()
    assert not at.exception
    tables = [
        df.value for df in at.dataframe
        if "Ticker" in list(df.value.columns)
        and "Est. value" in list(df.value.columns)
    ]
    assert tables, "preview table should show after clicking Preview"
    # Switch strategy -> the stored preview no longer matches -> gone.
    box = next(s for s in at.selectbox if s.key == "paper_alpaca_strategy")
    other = next(v for v in ("momentum", "trend") if v != box.value)
    box.set_value(other).run()
    assert not at.exception
    tables = [
        df.value for df in at.dataframe
        if "Ticker" in list(df.value.columns)
        and "Est. value" in list(df.value.columns)
    ]
    assert not tables, "stale preview must not show after inputs change"
