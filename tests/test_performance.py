"""Tests for the Performance page plumbing (Alpaca portfolio history).

Covers:
- AlpacaBroker.get_portfolio_history: request shape (endpoint, period /
  timeframe params) and response parsing (null equity points skipped,
  chronological order),
- helpers.portfolio_performance_for_ui: alignment of Alpaca equity with
  the local SPY benchmark on shared dates, metrics sanity (portfolio
  +10% vs SPY +4.5% on the seeded window), the YTD trim, the
  missing-SPY note, and the plain-English failure states (broker error,
  too little history).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from qrp.data.store import upsert_bars
from qrp.execution.alpaca import AlpacaBroker, AlpacaError
from qrp.ui import helpers as H

KEY_ID = "PKTESTKEYID123456"
SECRET = "test-secret-0123456789abcdef"
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Broker-level: mocked HTTP
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else b"{}"

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _mock_http(monkeypatch, tmp_path, responses):
    calls = []
    pending = list(responses)

    def fake_request(self, method, url, headers=None, timeout=None, **kwargs):
        calls.append(
            {"method": method, "url": url, "kwargs": kwargs}
        )
        resp = pending.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr("requests.Session.request", fake_request)
    broker = AlpacaBroker(
        KEY_ID, SECRET, keys_path=tmp_path / "keys.json",
        db_path=tmp_path / "m.db",
    )
    return broker, calls


def _ts(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=ET).timestamp())


def test_portfolio_history_request_and_parsing(monkeypatch, tmp_path):
    payload = {
        "timestamp": [_ts("2026-10-07"), _ts("2026-10-05"), _ts("2026-10-06")],
        "equity": [102.0, 100.0, None],
        "profit_loss": [2.0, 0.0, None],
        "base_value": 100.0,
    }
    broker, calls = _mock_http(monkeypatch, tmp_path, [_FakeResponse(200, payload)])
    points = broker.get_portfolio_history(period="3M", timeframe="1D")
    assert len(calls) == 1
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"].endswith("/v2/account/portfolio/history")
    assert calls[0]["kwargs"]["params"] == {
        "period": "3M",
        "timeframe": "1D",
    }
    # Null-equity day skipped; the rest sorted chronologically.
    assert points == [
        {"timestamp": _ts("2026-10-05"), "equity": 100.0},
        {"timestamp": _ts("2026-10-07"), "equity": 102.0},
    ]


def test_portfolio_history_empty_account(monkeypatch, tmp_path):
    broker, _ = _mock_http(
        monkeypatch, tmp_path,
        [_FakeResponse(200, {"timestamp": [], "equity": []})],
    )
    assert broker.get_portfolio_history() == []


# ---------------------------------------------------------------------------
# Helper-level: stub broker + seeded SPY database
# ---------------------------------------------------------------------------


def _seed_spy(db_path, days, first_close, last_close):
    closes = [
        first_close + (last_close - first_close) * i / (len(days) - 1)
        for i in range(len(days))
    ]
    upsert_bars(
        pd.DataFrame(
            {
                "Date": list(days),
                "Open": closes,
                "High": [c * 1.01 for c in closes],
                "Low": [c * 0.99 for c in closes],
                "Close": closes,
                "Volume": 1_000_000,
            }
        ),
        "SPY",
        db_path,
    )


def _points(days, first_value, last_value):
    values = [
        first_value + (last_value - first_value) * i / (len(days) - 1)
        for i in range(len(days))
    ]
    return [
        {"timestamp": _ts(d), "equity": v} for d, v in zip(days, values)
    ]


class _StubBroker:
    def __init__(self, points):
        self._points = points
        self.calls = []

    def get_portfolio_history(self, period="1M", timeframe="1D"):
        self.calls.append((period, timeframe))
        return list(self._points)


class _BrokenBroker:
    def get_portfolio_history(self, period="1M", timeframe="1D"):
        raise AlpacaError("Alpaca didn't recognize these API keys.", 401)


DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-10-05", "2026-10-16")]


def test_performance_aligns_and_computes_metrics(tmp_path):
    db = tmp_path / "market.db"
    _seed_spy(db, DAYS, 500.0, 522.50)  # SPY +4.5% over the window
    broker = _StubBroker(_points(DAYS, 100_000.0, 110_000.0))  # +10%
    result = H.portfolio_performance_for_ui(broker, "3M", db)
    assert result["ok"]
    assert broker.calls == [("3M", "1D")]
    assert result["start"] == DAYS[0]
    assert result["end"] == DAYS[-1]
    assert result["change_usd"] == pytest.approx(10_000.0)
    assert result["metrics"]["total_return"]["value"] == pytest.approx(0.10)
    assert result["spy_metrics"]["total_return"]["value"] == pytest.approx(
        0.045, abs=0.002
    )
    assert result["relative"]["beta"]["value"] == result["relative"]["beta"]["value"]
    pct = result["chart_pct"]
    assert list(pct.columns) == ["date", "Your portfolio", "SPY (S&P 500)"]
    assert pct["Your portfolio"].iloc[-1] == pytest.approx(10.0)
    assert pct["SPY (S&P 500)"].iloc[-1] == pytest.approx(4.5, abs=0.2)
    usd = result["chart_usd"]
    assert usd["Your portfolio"].iloc[-1] == pytest.approx(110_000.0)
    assert usd["SPY (S&P 500)"].iloc[-1] == pytest.approx(104_500.0, abs=250)


def test_performance_ytd_trims_to_january(tmp_path):
    db = tmp_path / "market.db"
    days = [
        d.strftime("%Y-%m-%d")
        for d in pd.bdate_range("2025-12-29", "2026-01-09")
    ]
    _seed_spy(db, days, 500.0, 510.0)
    broker = _StubBroker(_points(days, 100_000.0, 105_000.0))
    result = H.portfolio_performance_for_ui(broker, "YTD", db)
    assert result["ok"]
    assert broker.calls == [("1A", "1D")]
    assert result["start"] >= "2026-01-01"


def test_performance_without_spy_data_still_shows_portfolio(tmp_path):
    db = tmp_path / "market.db"  # no SPY seeded
    broker = _StubBroker(_points(DAYS, 100_000.0, 110_000.0))
    result = H.portfolio_performance_for_ui(broker, "All", db)
    assert result["ok"]
    assert result["spy_equity"] is None
    assert result["spy_metrics"] == {}
    assert any("SPY" in note for note in result["notes"])
    assert list(result["chart_pct"].columns) == ["date", "Your portfolio"]
    assert result["metrics"]["total_return"]["value"] == pytest.approx(0.10)


def test_performance_trims_zero_padded_history(tmp_path):
    """Alpaca pads pre-account days with $0 equity (an "All" request on
    a 2026 account returns five years of zeros). The window must start
    at the first day the account held value — a $0 start made the
    change read as the whole balance and every return n/a."""
    db = tmp_path / "market.db"
    _seed_spy(db, DAYS, 500.0, 522.50)
    padding = [
        {"timestamp": _ts(d), "equity": 0.0}
        for d in ("2021-10-22", "2022-06-15", "2025-01-02")
    ]
    broker = _StubBroker(padding + _points(DAYS, 100_000.0, 110_000.0))
    result = H.portfolio_performance_for_ui(broker, "All", db)
    assert broker.calls == [("5A", "1D")]
    assert result["ok"]
    assert result["start"] == DAYS[0]
    assert result["start_value"] == pytest.approx(100_000.0)
    assert result["change_usd"] == pytest.approx(10_000.0)
    assert result["metrics"]["total_return"]["value"] == pytest.approx(0.10)
    assert result["spy_metrics"]["total_return"]["value"] == pytest.approx(
        0.045, abs=0.002
    )
    assert result["chart_pct"]["date"].iloc[0] == DAYS[0]


def test_performance_all_zero_window(tmp_path):
    broker = _StubBroker(
        [{"timestamp": _ts(d), "equity": 0.0} for d in DAYS]
    )
    result = H.portfolio_performance_for_ui(broker, "All", tmp_path / "m.db")
    assert not result["ok"]
    assert "$0 balance" in result["error"]


def test_performance_single_valued_day_after_padding(tmp_path):
    points = [
        {"timestamp": _ts("2025-01-02"), "equity": 0.0},
        {"timestamp": _ts(DAYS[0]), "equity": 100_000.0},
    ]
    result = H.portfolio_performance_for_ui(
        _StubBroker(points), "All", tmp_path / "m.db"
    )
    assert not result["ok"]
    assert "at least" in result["error"]


def test_performance_broker_error_is_plain_english(tmp_path):
    result = H.portfolio_performance_for_ui(
        _BrokenBroker(), "1M", tmp_path / "market.db"
    )
    assert not result["ok"]
    assert "didn't recognize" in result["error"]


def test_performance_too_little_history(tmp_path):
    broker = _StubBroker(
        [{"timestamp": _ts(DAYS[0]), "equity": 100_000.0}]
    )
    result = H.portfolio_performance_for_ui(broker, "1M", tmp_path / "m.db")
    assert not result["ok"]
    assert "at least" in result["error"]
