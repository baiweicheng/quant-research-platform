"""Tests for delisted/stale-ticker handling (2026-10-08 fix, refined).

The problem: a stock that merged away or delisted (e.g. AVB, whose
prices legitimately end 2026-08-14) was treated exactly like a stale
download — the P5 staleness check flagged it, the P6 pre-trade
freshness gate blocked EVERY batch while it sat in the database, and
forward-filled signal scores could still rank it into a target
portfolio (sized off its last-ever close).

The refined rules under test:
- Health tiers (qrp.data.quality): >2 trading days behind = "stale"
  (warning); >10 behind with the rest of the DB current = "inactive"
  (info, likely delisted); >10% of tickers deep behind = global stall
  (nothing is inactive; everything stays stale).
- Pre-trade gate (qrp.ops.pretrade): fails ONLY when the database as a
  whole is stale, or a stock in the order batch itself has no current
  price. Lagging names outside the batch are noted, never blocking —
  from day 3 after a delisting, not just day 11.
- Target building (qrp.ui.helpers): names whose newest price is >2
  trading days behind the decision date are excluded before sizing,
  with a plain-English note.
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
GAP3_END = "2026-10-05"  # 3 trading days before ASOF


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


def _db_mixed(tmp_path):
    """20 fresh + DEAD (39 days behind) + GAP3 (3 days behind)."""
    path = tmp_path / "m.db"
    for i in range(20):
        _add(path, f"F{i:02d}", FRESH_END)
    _add(path, "DEAD", DEAD_END)
    _add(path, "GAP3", GAP3_END)
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


def _gate(db_path, orders=None):
    """Run the pre-trade checklist the way staging does (fresh local
    health report, no network), with one small planned buy."""
    report = Q.run_data_health_checks(db_path, network_sample=0)
    broker = FakeBroker(prices={"F00": 100.0, "GAP": 100.0, "DEAD": 100.0})
    if orders is None:
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
    assert Q.database_globally_stale(db, asof=ASOF) is True


def test_inactive_is_point_in_time(tmp_path):
    """Before DEAD stopped trading, it must not count as inactive."""
    db = _db_with_dead(tmp_path)
    assert Q.inactive_tickers(db, asof=date(2026, 8, 14)) == set()
    assert "DEAD" in Q.inactive_tickers(db, asof=ASOF)


def test_stale_tickers_uses_the_two_day_line(tmp_path):
    """The exclusion rule catches BOTH the 3-day and the 39-day name,
    judged as of the decision date — and neither before their prices
    actually stopped."""
    db = _db_mixed(tmp_path)
    assert Q.stale_tickers(db, asof=ASOF) == {"DEAD", "GAP3"}
    assert Q.stale_tickers(db, asof=date(2026, 8, 14)) == set()
    assert Q.database_globally_stale(db, asof=ASOF) is False


# --- the pre-trade gate ----------------------------------------------------


def test_gate_passes_with_dead_ticker(tmp_path):
    db = _db_with_dead(tmp_path)
    report = _gate(db)
    check = _freshness(report)
    assert check.status == "pass"
    assert "DEAD" in check.detail and "left out of this batch" in check.detail
    assert report.ok is True


def test_gate_passes_with_3day_and_39day_laggards_outside_batch(tmp_path):
    """The refinement's core: a stock 3 days past its last price (the
    first days after a delisting) must not block staging either."""
    db = _db_mixed(tmp_path)
    report = _gate(db)
    check = _freshness(report)
    assert check.status == "pass"
    assert "DEAD" in check.detail and "GAP3" in check.detail
    assert "left out of this batch" in check.detail
    assert report.ok is True


def test_gate_passes_with_recent_gap_outside_batch(tmp_path):
    db = _db_with_gap(tmp_path)
    report = _gate(db)
    check = _freshness(report)
    assert check.status == "pass"
    assert "GAP" in check.detail
    assert report.ok is True


def test_gate_blocks_when_batch_ticker_has_no_price(tmp_path):
    """Defense in depth: a stale name forced into the batch fails."""
    db = _db_with_gap(tmp_path)
    orders = [{"ticker": "GAP", "side": "BUY", "shares": 10.0,
               "price": 100.0, "reason": "fabricated"}]
    report = _gate(db, orders=orders)
    check = _freshness(report)
    assert check.status == "fail"
    assert "GAP" in check.detail and "in this batch" in check.detail
    assert report.ok is False


def test_gate_blocks_when_delisted_ticker_in_batch(tmp_path):
    db = _db_with_dead(tmp_path)
    orders = [{"ticker": "DEAD", "side": "BUY", "shares": 10.0,
               "price": 100.0, "reason": "fabricated"}]
    report = _gate(db, orders=orders)
    assert _freshness(report).status == "fail"
    assert report.ok is False


def test_gate_still_blocks_on_global_stall(tmp_path):
    db = _db_stalled(tmp_path)
    report = _gate(db)
    assert _freshness(report).status == "fail"
    assert report.ok is False


# --- target building -------------------------------------------------------


def _target_db(tmp_path):
    """11 live tickers + DEAD (39d behind) + GAP3 (3d behind).

    DEAD and GAP3 hold the TOP stored signal values right through
    2026-10-08 (as the real store would after a fresh compute) — the
    forward-fill trap: without the exclusion they would still rank,
    and be bought, off their last-ever closes.
    """
    path = tmp_path / "m.db"
    live = [f"L{i:02d}" for i in range(11)]
    for t in live:
        _add(path, t, FRESH_END)
    _add(path, "DEAD", DEAD_END)
    _add(path, "GAP3", GAP3_END)
    tickers = live + ["DEAD", "GAP3"]
    day = pd.Timestamp(FRESH_END)
    for sig in ("momentum", "value_composite"):
        for i, t in enumerate(live):
            upsert_signals(pd.Series({day: 0.50 + 0.01 * i}), t, sig, path)
        upsert_signals(pd.Series({day: 0.99}), "DEAD", sig, path)
        upsert_signals(pd.Series({day: 0.98}), "GAP3", sig, path)
    return path, tickers


def test_custom_target_excludes_names_with_no_current_price(tmp_path):
    db, tickers = _target_db(tmp_path)
    portfolio, scores, _day = H.build_target_for_ui(
        "momentum", "signal", FRESH_END, tickers, db
    )
    for gone in ("DEAD", "GAP3"):
        assert gone not in portfolio.weights
        assert gone not in scores
    assert portfolio.weights  # the live names still got sized
    assert any(
        "no current price" in n and "DEAD" in n and "GAP3" in n
        for n in portfolio.notes
    )


def test_strategy_target_excludes_names_with_no_current_price(tmp_path):
    db, tickers = _target_db(tmp_path)
    portfolio, scores, _day, _legs = H.build_strategy_target_for_ui(
        "value_momentum", FRESH_END, tickers, db
    )
    for gone in ("DEAD", "GAP3"):
        assert gone not in portfolio.weights
        assert gone not in scores
    assert portfolio.weights
    assert any(
        "no current price" in n and "DEAD" in n and "GAP3" in n
        for n in portfolio.notes
    )
