"""Tests for delisted/inactive-ticker handling (2026-10-08 fix).

The problem: a stock that merged away or delisted (e.g. AVB, whose
prices legitimately end 2026-08-14) was treated exactly like a stale
download — the P5 staleness check flagged it, the P6 pre-trade
freshness gate blocked EVERY batch while it sat in the database, and
forward-filled signal scores could still rank it into a target
portfolio (sized off its last-ever close).

The rule under test (qrp.data.quality._classify_freshness):
- >10 trading days behind, rest of the database current -> "inactive"
  (info): left out of target portfolios, never blocks the gate.
- 3-10 trading days behind -> "stale" (warning): blocks, as before.
- >10% of tickers with data are >10 behind -> global stall: nothing is
  inactive, everything stays stale, gate still blocks.

Plus the UI seam: build_target_for_ui / build_strategy_target_for_ui
drop inactive names before sizing and say so in the portfolio notes.
"""

from datetime import date

import pandas as pd
import pytest

from qrp.data import quality as Q
from qrp.data.store import connect_db, upsert_bars
from qrp.execution.broker import AccountInfo, Broker, OrderInfo
from qrp.ops import pretrade as OPS_PRE
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H

ASOF = date(2026, 10, 8)  # a Thursday
FRESH_END = "2026-10-08"
DEAD_END = "2026-08-14"  # 39 trading days before ASOF
GAP_END = "2026-10-01"  # 5 trading days before ASOF


def _frame(dates, start=100.0):
    closes = [start + i * 0.05 for i in range(len(dates))]
    return pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(dates),
        }
    )


def _add(path, ticker, end, start=100.0):
    dates = pd.bdate_range("2026-07-01", end)
    upsert_bars(_frame(dates, start), ticker, path)


def _db_with_dead(tmp_path):
    """20 fresh tickers + DEAD (prices end 2026-08-14)."""
    path = tmp_path / "m.db"
    for i in range(20):
        _add(path, f"F{i:02d}", FRESH_END)
    _add(path, "DEAD", DEAD_END)
    return path


def _db_with_gap(tmp_path):
    """20 fresh tickers + GAP (prices end 2026-10-01: a download gap)."""
    path = tmp_path / "m.db"
    for i in range(20):
        _add(path, f"F{i:02d}", FRESH_END)
    _add(path, "GAP", GAP_END)
    return path


def _db_stalled(tmp_path):
    """Every ticker's prices end 2026-08-14 (nobody refreshed)."""
    path = tmp_path / "m.db"
    for i in range(5):
        _add(path, f"S{i}", DEAD_END)
    return path


class FakeBroker(Broker):
    """Minimal in-memory Broker (same shape as tests/test_ops.py)."""

    def __init__(self, cash=100_000.0, prices=None):
        self._cash = float(cash)
        self._prices = dict(prices or {})

    def get_account(self):
        return AccountInfo(
            equity=self._cash, cash=self._cash, buying_power=self._cash
        )

    def get_positions(self):
        return {}

    def last_price(self, ticker, date=None):
        if ticker not in self._prices:
            raise KeyError(ticker)
        return float(self._prices[ticker])

    def place_order(self, ticker, side, shares, reason="", *,
                    client_order_id=None):
        return OrderInfo(
            id="1", ticker=ticker, side=side, shares=float(shares),
            status="open", client_order_id=client_order_id or "",
            reason=reason,
        )

    def cancel_order(self, order_id):
        return True

    def list_orders(self, status="open"):
        return []

    def get_order(self, order_id):
        return None


def _gate(db_path):
    """Run the pre-trade checklist the way staging does (fresh local
    health report, no network), with one small planned buy."""
    report = Q.run_data_health_checks(db_path, network_sample=0)
    broker = FakeBroker(prices={"F00": 100.0})
    orders = [{"ticker": "F00", "side": "BUY", "shares": 10.0,
               "price": 100.0, "reason": "test"}]
    return OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db_path, decision_date="2026-10-08",
        health_report=report, audit_log=False,
    )


def _freshness(report):
    return next(c for c in report.checks if c.name == "Data freshness")


# --- classification ------------------------------------------------------


def test_dead_ticker_is_inactive_not_stale(tmp_path):
    db = _db_with_dead(tmp_path)
    with connect_db(db) as conn:
        findings = Q.check_staleness(conn, asof=ASOF)
    dead = [f for f in findings if f.ticker == "DEAD"]
    assert len(dead) == 1
    assert dead[0].check == "inactive"
    assert dead[0].severity == "info"
    assert "delisted" in dead[0].summary
    # Nothing else flagged at all: the other 20 are fresh.
    assert [f for f in findings if f.ticker != "DEAD"] == []
    assert Q.inactive_tickers(db, asof=ASOF) == {"DEAD"}


def test_recent_gap_stays_stale(tmp_path):
    db = _db_with_gap(tmp_path)
    with connect_db(db) as conn:
        findings = Q.check_staleness(conn, asof=ASOF)
    gap = [f for f in findings if f.ticker == "GAP"]
    assert len(gap) == 1
    assert gap[0].check == "stale" and gap[0].severity == "warning"
    assert "5 trading days behind" in gap[0].summary
    assert Q.inactive_tickers(db, asof=ASOF) == set()


def test_global_stall_nothing_is_inactive(tmp_path):
    db = _db_stalled(tmp_path)
    with connect_db(db) as conn:
        findings = Q.check_staleness(conn, asof=ASOF)
    assert len(findings) == 5
    assert all(
        f.check == "stale" and f.severity == "warning" for f in findings
    )
    assert Q.inactive_tickers(db, asof=ASOF) == set()


def test_inactive_is_point_in_time(tmp_path):
    """Before DEAD stopped trading, it must not count as inactive."""
    db = _db_with_dead(tmp_path)
    assert Q.inactive_tickers(db, asof=date(2026, 8, 14)) == set()
    assert "DEAD" in Q.inactive_tickers(db, asof=ASOF)


# --- the pre-trade gate ----------------------------------------------------


def test_gate_passes_with_dead_ticker(tmp_path):
    db = _db_with_dead(tmp_path)
    report = _gate(db)
    check = _freshness(report)
    assert check.status == "pass"
    assert "delisted" in check.detail and "DEAD" in check.detail
    assert report.ok is True


def test_gate_still_blocks_on_recent_gap(tmp_path):
    db = _db_with_gap(tmp_path)
    report = _gate(db)
    assert _freshness(report).status == "fail"
    assert report.ok is False


def test_gate_still_blocks_on_global_stall(tmp_path):
    db = _db_stalled(tmp_path)
    report = _gate(db)
    assert _freshness(report).status == "fail"
    assert report.ok is False


# --- target building -------------------------------------------------------


def _target_db(tmp_path):
    """11 live tickers + DEAD; DEAD scores #1 on every synthetic signal.

    Signal VALUES for DEAD exist right through 2026-10-08 (as the real
    store would hold after a fresh compute) — the forward-fill bug this
    fix removes is precisely that those scores kept it selectable.
    """
    path = tmp_path / "m.db"
    live = [f"L{i:02d}" for i in range(11)]
    for t in live:
        _add(path, t, FRESH_END)
    _add(path, "DEAD", DEAD_END)
    tickers = live + ["DEAD"]
    day = pd.Timestamp(FRESH_END)
    for sig in ("momentum", "value_composite"):
        for i, t in enumerate(live):
            upsert_signals(pd.Series({day: 0.50 + 0.01 * i}), t, sig, path)
        upsert_signals(pd.Series({day: 0.99}), "DEAD", sig, path)
    return path, tickers


def test_custom_target_excludes_delisted(tmp_path):
    db, tickers = _target_db(tmp_path)
    portfolio, scores, _day = H.build_target_for_ui(
        "momentum", "signal", FRESH_END, tickers, db
    )
    assert "DEAD" not in portfolio.weights
    assert "DEAD" not in scores
    assert portfolio.weights  # the live names still got sized
    assert any(
        "delisted" in n and "DEAD" in n for n in portfolio.notes
    )


def test_strategy_target_excludes_delisted(tmp_path):
    db, tickers = _target_db(tmp_path)
    portfolio, scores, _day, _legs = H.build_strategy_target_for_ui(
        "value_momentum", FRESH_END, tickers, db
    )
    assert "DEAD" not in portfolio.weights
    assert "DEAD" not in scores
    assert portfolio.weights
    assert any(
        "delisted" in n and "DEAD" in n for n in portfolio.notes
    )
