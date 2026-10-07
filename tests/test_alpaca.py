"""Tests for the Alpaca paper-trading adapter (roadmap P2).

Covers, with the HTTP layer mocked:
- auth headers on every request,
- GET /v2/account parsing (equity / cash / buying power),
- GET /v2/positions parsing (empty, whole, fractional),
- POST /v2/orders payload for the market-on-open mapping
  (symbol, qty-as-string, side, type=market, time_in_force=opg),
- GET /v2/orders, DELETE /v2/orders/{id}, GET /v2/orders/{id},
- error translation into plain English (401 bad keys, 403 insufficient
  buying power, 404, 422 incl. the fractional hint, 429 rate limit, 5xx,
  connection errors) with keys never leaking into messages,
- the live-URL guard,
- broker-ABC conformance for both the Alpaca adapter and the local
  PaperBroker,
- key storage (0600, gitignored, never logged),
- the staging queue (keep rejected orders staged for the next valid open),
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
    QUEUE_MAX_ATTEMPTS,
)
from qrp.execution.broker import AccountInfo, Broker, OrderInfo
from qrp.execution.paper import PaperBroker, PaperBrokerAdapter
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H

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
    """Staging maps to market-on-open: type=market, time_in_force=opg."""
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


def test_flush_sends_queued_orders_as_opg(monkeypatch, tmp_path):
    broker, calls, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, []),  # no open orders
            _FakeResponse(200, _order_payload()),
        ],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 1
    assert result["waiting"] == 0
    assert result["notes"] == []
    payload = calls[1]["kwargs"]["json"]
    assert payload["time_in_force"] == "opg"
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == "10"
    assert load_staging_queue(keys_path) == []


def test_flush_skips_orders_already_open_at_alpaca(monkeypatch, tmp_path):
    broker, calls, keys_path = _mock_http(
        monkeypatch,
        tmp_path,
        [_FakeResponse(200, [_order_payload(status="accepted")])],
    )
    queue_staged_order("AAPL", "BUY", 10.0, keys_path=keys_path)
    result = flush_staging_queue(broker)
    assert result["sent"] == 0
    assert len(calls) == 1  # only the open-orders check; no POST
    assert load_staging_queue(keys_path) == []


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


def test_flush_drops_orders_after_max_attempts(monkeypatch, tmp_path):
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
    entries[0]["attempts"] = QUEUE_MAX_ATTEMPTS - 1
    save_staging_queue(entries, keys_path)
    result = flush_staging_queue(broker)
    assert result["waiting"] == 0
    assert load_staging_queue(keys_path) == []
    assert any("Gave up" in note for note in result["notes"])


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


@pytest.fixture
def clean_alpaca_files():
    """The repo data dir must not keep Alpaca key/queue files between tests."""
    names = (
        "alpaca_paper_keys.json",
        "alpaca_staging_queue.json",
        "alpaca_order_reasons.json",
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
