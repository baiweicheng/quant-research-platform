"""Tests for the third-audit fix batch (BUG-01/02/03/04/05/07/09/11/13/15).

Each test pins the corrected behavior found during the audit review, so a
future change that reintroduces the bug fails loudly. BUG-15 (drift slider
hidden for named strategies) is a Streamlit rendering change, covered by
the scripted UAT rather than a unit test.
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.engine import run_backtest
from qrp.data.quality import DataHealthReport, Finding
from qrp.data.store import connect_db, upsert_splits
from qrp.execution.paper import PaperBroker, PaperBrokerAdapter
from qrp.execution.rebalance import orders_to_target
from qrp.ops import pretrade as P
from qrp.portfolio import optimizer as O
from qrp.portfolio.covariance import guard_estimation_window
from qrp.risk.factor_model import fit_factor_model
from qrp.risk.guards import Order


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bars_db(tickers_days: dict[str, list[tuple[str, float]]]) -> Path:
    """Temp DB with daily_bars for {ticker: [(date, close), ...]}."""
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE daily_bars (ticker TEXT, date TEXT, open REAL,"
        " high REAL, low REAL, close REAL, volume REAL)"
    )
    for ticker, days in tickers_days.items():
        for date, close in days:
            conn.execute(
                "INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)",
                (ticker, date, close, close * 1.01, close * 0.99, close, 1000),
            )
    conn.commit()
    conn.close()
    return db


def _health() -> DataHealthReport:
    return DataHealthReport(
        checked_at="2025-04-01T00:00:00Z", tickers_checked=1,
        findings=[Finding(check="x", severity="info", ticker=None, summary="ok")],
    )


def _check(name: str, report) -> str:
    return next(c for c in report.checks if c.name == name).status


# ---------------------------------------------------------------------------
# BUG-01: optimizer turnover must not double-count exited positions
# ---------------------------------------------------------------------------

def test_turnover_one_way_matches_independent_computation():
    rng = np.random.default_rng(0)
    tickers = ["A", "B", "C"]
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    days = {
        t: [
            (d.strftime("%Y-%m-%d"), float(p))
            for d, p in zip(
                dates, 100 + np.cumsum(rng.normal(0, 1, 60))
            )
        ]
        for t in tickers
    }
    db = _bars_db(days)
    w_prev = {"A": 0.4, "B": 0.3, "C": 0.3}
    port, stats = O.optimize_weights(
        {"A": 1.0, "B": 0.5, "C": 0.1}, asof="2025-04-01", db_path=db,
        w_prev=w_prev, max_weight=0.6, min_weight=0.0,
        turnover_penalty_bps=0.0,
    )
    universe = set(w_prev) | set(port.weights)
    expected = 0.5 * sum(
        abs(port.weights.get(t, 0.0) - w_prev.get(t, 0.0)) for t in universe
    )
    assert stats.turnover_one_way == pytest.approx(expected)


# ---------------------------------------------------------------------------
# BUG-02: thin-history tickers must not collapse the estimation window
# ---------------------------------------------------------------------------

def test_guard_estimation_window_drops_stub_ticker():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {f"S{i:02d}": rng.normal(0, 0.01, 252) for i in range(5)}
    )
    frame["NEWB"] = np.nan
    frame.loc[frame.index[-40:], "NEWB"] = rng.normal(0, 0.01, 40)
    kept, thin = guard_estimation_window(frame, list(frame.columns))
    assert thin == ["NEWB"]
    assert kept == [f"S{i:02d}" for i in range(5)]


def test_guard_estimation_window_leaves_uniform_panels_alone():
    rng = np.random.default_rng(2)
    cols = [f"S{i:02d}" for i in range(5)]
    frame = pd.DataFrame({c: rng.normal(0, 0.01, 60) for c in cols})
    kept, thin = guard_estimation_window(frame, cols)
    assert thin == [] and kept == cols


def test_guard_estimation_window_never_strands_below_two():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "OLD": rng.normal(0, 0.01, 252),
            "NEW": np.concatenate(
                [np.full(222, np.nan), rng.normal(0, 0.01, 30)]
            ),
        }
    )
    kept, thin = guard_estimation_window(frame, ["OLD", "NEW"])
    assert kept == ["OLD", "NEW"] and thin == []


# ---------------------------------------------------------------------------
# BUG-03: one degenerate residual column must not floor every stock
# ---------------------------------------------------------------------------

def test_specific_risk_fallback_is_per_stock():
    rng = np.random.default_rng(4)
    # Identical twins (large variance) dominate the single factor, so
    # their residual variance is exactly zero and the shrinkage helper
    # raises; the third ticker is constructed exactly orthogonal to them,
    # so it keeps real residual risk through the fallback.
    base = rng.normal(0, 0.05, 120)
    raw = rng.normal(0, 0.01, 120)
    # Exact orthogonalization (demeaned): independent but ⊥ base.
    ortho = raw - raw.mean() - base * np.dot(raw - raw.mean(), base) / np.dot(base, base)
    frame = pd.DataFrame({"TWIN1": base, "TWIN2": base, "NOISY": ortho})
    fit = fit_factor_model(frame, max_factors=1)
    assert any("tiny floor" in n for n in fit.notes)
    # NOISY keeps roughly its own residual variance, not the 1e-8 floor.
    assert fit.specific_var["NOISY"] > 1e-4
    # ...while the degenerate twins take the floor.
    assert fit.specific_var["TWIN1"] == pytest.approx(1e-8)


# ---------------------------------------------------------------------------
# BUG-04: buying-power check is self-funding (sells count; no 98% trap)
# ---------------------------------------------------------------------------

def test_buying_power_passes_fresh_full_size_batch():
    days = {
        f"STK{i:02d}": [("2026-10-09", 50.0)] for i in range(20)
    }
    db = _bars_db(days)
    broker = PaperBroker(db, starting_capital=100_000.0)
    orders = orders_to_target(
        broker, {f"STK{i:02d}": 0.05 for i in range(20)},
        "2026-10-09", 100_000.0, cost_bps=10.0, price_source="close",
    )
    planned = sum(o.shares * o.price for o in orders if o.action == "BUY")
    assert planned > 98_000  # the old 98%-of-buying-power rule failed this
    ords = [
        {"ticker": o.ticker, "side": o.action, "shares": o.shares,
         "price": o.price, "reason": ""}
        for o in orders
    ]
    rep = P.run_pretrade_checks(
        ords, PaperBrokerAdapter(broker), db_path=db,
        decision_date="2026-10-09", health_report=_health(), audit_log=False,
    )
    assert _check("Buying power", rep) == "pass"
    assert rep.ok


def test_buying_power_counts_sell_proceeds():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(db, starting_capital=100_000.0)
    # $50k of sells fund $50k of buys on a nearly-empty cash balance.
    ords = [
        {"ticker": "AAA", "side": "SELL", "shares": 1000.0, "price": 50.0,
         "reason": ""},
        {"ticker": "AAA", "side": "BUY", "shares": 1000.0, "price": 50.0,
         "reason": ""},
    ]
    rep = P.run_pretrade_checks(
        ords, PaperBrokerAdapter(broker), db_path=db,
        decision_date="2026-10-09", health_report=_health(), audit_log=False,
    )
    assert _check("Buying power", rep) == "pass"


def test_buying_power_still_blocks_unaffordable_batches():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(db, starting_capital=100_000.0)
    ords = [
        {"ticker": "AAA", "side": "BUY", "shares": 10000.0, "price": 50.0,
         "reason": ""},
    ]
    rep = P.run_pretrade_checks(
        ords, PaperBrokerAdapter(broker), db_path=db,
        decision_date="2026-10-09", health_report=_health(), audit_log=False,
    )
    assert _check("Buying power", rep) == "fail"
    assert not rep.ok


# ---------------------------------------------------------------------------
# BUG-05: the checklist enforces the cap the portfolio was built with
# ---------------------------------------------------------------------------

def test_position_limits_use_threaded_cap():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(db, starting_capital=100_000.0)
    # A 20% position: fine under a 20% cap, blocked under the 5% default.
    ords = [
        {"ticker": "AAA", "side": "BUY", "shares": 400.0, "price": 50.0,
         "reason": ""},
    ]
    rep20 = P.run_pretrade_checks(
        ords, PaperBrokerAdapter(broker), db_path=db,
        decision_date="2026-10-09", health_report=_health(), audit_log=False,
        max_position_weight=0.20,
    )
    rep05 = P.run_pretrade_checks(
        ords, PaperBrokerAdapter(broker), db_path=db,
        decision_date="2026-10-09", health_report=_health(), audit_log=False,
    )
    assert _check("Position limits", rep20) == "pass"
    assert _check("Position limits", rep05) == "fail"


def test_position_limits_reject_nonsense_cap():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(db, starting_capital=100_000.0)
    with pytest.raises(ValueError):
        P.run_pretrade_checks(
            [], PaperBrokerAdapter(broker), db_path=db,
            max_position_weight=1.5,
        )


# ---------------------------------------------------------------------------
# BUG-07: cash-fit scaling runs even at zero commission
# ---------------------------------------------------------------------------

def test_cash_scaling_applies_at_zero_commission():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(
        db, starting_capital=1_000.0, execution="open",
        cost_model="flat", cost_bps=0.0,
    )
    # Target implies a $50k buy against $1k of cash: the old code skipped
    # scaling entirely when cost_bps == 0 and emitted an unfillable order.
    orders = orders_to_target(
        broker, {"AAA": 0.5}, "2026-10-09", 100_000.0,
        cost_bps=0.0, price_source="close",
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert buys, "expected a (scaled) buy order"
    assert sum(o.shares * o.price for o in buys) <= 1_000.0 + 1e-6


# ---------------------------------------------------------------------------
# BUG-09: 422 id-conflict resolves instead of spinning forever
# ---------------------------------------------------------------------------

def _order_payload(cid: str, status: str) -> dict:
    return {
        "id": "ord-1", "symbol": "AAPL", "side": "buy", "qty": "10",
        "status": status, "filled_avg_price": None,
        "submitted_at": "2026-10-10T14:00:00Z", "client_order_id": cid,
    }


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _conflict_broker(tmp_path, responses):
    from qrp.execution.alpaca import AlpacaBroker
    from test_alpaca import KEY_ID, SECRET

    calls: list = []
    pending = list(responses)

    class FakeSession:
        def request(self, method, url, headers=None, timeout=None, **kwargs):
            payload = kwargs.get("json") or {}
            calls.append((method, payload.get("client_order_id")))
            resp = pending.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

    broker = AlpacaBroker(
        KEY_ID, SECRET, keys_path=tmp_path / "k.json",
        db_path=tmp_path / "m.db",
    )
    broker._session = FakeSession()
    return broker, calls


def test_id_conflict_with_canceled_order_resubmits_revision(tmp_path):
    cid = "rebalance-20261010-AAPL-BUY"
    broker, calls = _conflict_broker(
        tmp_path,
        [
            _FakeResponse(422, {"message": "client_order_id must be unique"}),
            _FakeResponse(200, _order_payload(cid, "canceled")),
            _FakeResponse(200, _order_payload(cid + "-r2", "accepted")),
        ],
    )
    info, note = broker.place_order_resolving_conflict(
        "AAPL", "BUY", 10.0, client_order_id=cid
    )
    assert info.client_order_id == cid + "-r2"
    assert ("POST", cid + "-r2") in calls
    assert "follow-up" in note


def test_id_conflict_with_live_order_sends_no_duplicate(tmp_path):
    cid = "rebalance-20261010-AAPL-BUY"
    broker, calls = _conflict_broker(
        tmp_path,
        [
            _FakeResponse(422, {"message": "client_order_id must be unique"}),
            _FakeResponse(200, _order_payload(cid, "accepted")),
        ],
    )
    info, note = broker.place_order_resolving_conflict(
        "AAPL", "BUY", 10.0, client_order_id=cid
    )
    assert info.status == "open"
    assert len([c for c in calls if c[0] == "POST"]) == 1  # only the attempt
    assert "no duplicate" in note


def test_id_conflict_with_filled_order_sends_no_duplicate(tmp_path):
    cid = "rebalance-20261010-AAPL-BUY"
    broker, calls = _conflict_broker(
        tmp_path,
        [
            _FakeResponse(422, {"message": "client_order_id must be unique"}),
            _FakeResponse(200, _order_payload(cid, "filled")),
        ],
    )
    info, _note = broker.place_order_resolving_conflict(
        "AAPL", "BUY", 10.0, client_order_id=cid
    )
    assert info.status == "filled"
    assert len([c for c in calls if c[0] == "POST"]) == 1


def test_non_conflict_422_still_raises(tmp_path):
    from qrp.execution.alpaca import AlpacaError

    broker, _calls = _conflict_broker(
        tmp_path,
        [_FakeResponse(422, {"message": "fractional qty not allowed"})],
    )
    with pytest.raises(AlpacaError):
        broker.place_order_resolving_conflict(
            "AAPL", "BUY", 10.0, client_order_id="rebalance-20261010-AAPL-BUY"
        )


# ---------------------------------------------------------------------------
# BUG-11: stock splits keep share counts on the adjusted-price basis
# ---------------------------------------------------------------------------

def test_paper_broker_adjusts_positions_on_split():
    db = _bars_db(
        {
            "NVDA": [
                ("2024-06-06", 1200.0), ("2024-06-07", 1200.0),
                ("2024-06-10", 120.0), ("2024-06-11", 120.0),
            ]
        }
    )
    upsert_splits(
        pd.DataFrame([{"date": "2024-06-10", "ratio": 10.0}]), "NVDA", db
    )
    broker = PaperBroker(
        db, starting_capital=100_000.0, execution="open",
        cost_model="flat", cost_bps=0.0,
    )
    res = broker.place_order("NVDA", "BUY", 4.0, "2024-06-06")
    assert res.filled
    assert broker.positions["NVDA"] == 4.0
    report = broker.fill_pending_orders()  # applies the split
    assert any("split" in n for n in report["notes"])
    # 4 pre-split shares are now 40 — the $4,800 position value survives
    # the $1,200 -> $120 price rewrite.
    assert broker.positions["NVDA"] == pytest.approx(40.0)
    assert broker.positions["NVDA"] * 120.0 == pytest.approx(4800.0)
    # Idempotent: a second pass changes nothing.
    broker.fill_pending_orders()
    assert broker.positions["NVDA"] == pytest.approx(40.0)


def test_backtest_engine_applies_splits():
    dates = pd.to_datetime(
        ["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"]
    )
    prices = pd.DataFrame(
        {"NVDA": [1200.0, 1200.0, 120.0, 120.0, 120.0]}, index=dates
    )
    signals = pd.DataFrame({"NVDA": [1.0] * 5}, index=dates)

    def strat(sig, day):
        return {"NVDA": 0.05}

    splits = pd.DataFrame(
        [{"ticker": "NVDA", "date": pd.Timestamp("2024-06-10"), "ratio": 10.0}]
    )
    res = run_backtest(
        prices, signals, strat, frequency="monthly", cost_bps=0.0,
        initial_capital=100_000.0, execution="close", splits=splits,
    )
    vals = res.equity["portfolio_value"].to_numpy()
    # Flat prices, 5% position: no phantom 90% loss on the split date.
    assert np.all(vals > 99_000.0)


# ---------------------------------------------------------------------------
# BUG-13: transient fill failures stay pending, retry at the next open
# ---------------------------------------------------------------------------

def test_cash_short_buy_stays_pending_and_retries():
    db = _bars_db(
        {
            # BBB (buy) tradeable from 10-09; AAA (sell) only from 10-10,
            # so the buy is attempted before its funding arrives.
            "BBB": [("2026-10-09", 50.0), ("2026-10-10", 50.0)],
            "AAA": [("2026-10-10", 50.0)],
        }
    )
    broker = PaperBroker(
        db, starting_capital=10_000.0, execution="open",
        cost_model="flat", cost_bps=0.0,
    )
    with connect_db(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO paper_positions (ticker, shares)"
            " VALUES ('AAA', 192.0)"
        )
        conn.execute("UPDATE paper_cash SET cash = 400.0 WHERE id = 1")
        for _ in range(24):
            conn.execute(
                "INSERT INTO paper_trades"
                " (date,ticker,side,shares,price,cost,reason)"
                " VALUES ('2026-10-08','AAA','BUY',8.0,50.0,0.0,'')"
            )
    broker.stage_orders(
        [
            Order(ticker="AAA", action="SELL", shares=192.0, price=50.0,
                  reason=""),
            Order(ticker="BBB", action="BUY", shares=9.0, price=50.0,
                  reason=""),
        ],
        "2026-10-08",
    )
    rep = broker.fill_pending_orders()
    # The sell fills on its day; the cash-short buy is KEPT, not deleted.
    assert [f["ticker"] for f in rep["filled"]] == ["AAA"]
    assert rep["skipped"] == []
    assert [r["ticker"] for r in rep["pending_retry"]] == ["BBB"]

    rep2 = broker.fill_pending_orders()
    filled_bbb = [f for f in rep2["filled"] if f["ticker"] == "BBB"]
    assert len(filled_bbb) == 1
    # Retried at the open AFTER the failed day, not the stale one.
    assert filled_bbb[0]["date"] == "2026-10-10"
    assert broker.positions.get("BBB", 0.0) == 9.0


def test_permanent_rejection_still_drops_the_order():
    db = _bars_db({"AAA": [("2026-10-09", 50.0)]})
    broker = PaperBroker(
        db, starting_capital=10_000.0, execution="open",
        cost_model="flat", cost_bps=0.0,
    )
    # Zero-share order: permanently invalid, must not linger.
    res = broker.place_order("AAA", "BUY", 0.0, "2026-10-08")
    assert not res.filled and not res.retryable


def test_staging_cap_uses_effective_optimizer_cap():
    """BUG-05 follow-up: when the optimizer relaxes the requested 5% cap
    (19 survivors -> ~5.26%), the checklist must enforce the EFFECTIVE
    cap, not the requested one — otherwise a legitimate auto-relaxed
    batch is blocked."""
    pytest.importorskip("streamlit")
    from qrp.ui.app import _staging_position_cap

    settings = {"method_key": "optimized", "optimizer": {"max_weight": 0.05}}
    # 19-stock portfolio at the relaxed 1/19 cap.
    relaxed = {f"S{i:02d}": 1 / 19 for i in range(19)}
    assert _staging_position_cap(None, settings, relaxed) == pytest.approx(
        1 / 19
    )
    # No relaxation needed -> the configured cap stands.
    tight = {"AAA": 0.04, "BBB": 0.04}
    assert _staging_position_cap(None, settings, tight) == pytest.approx(0.05)
    # A hand-edited overweight batch is NOT rescued: the effective cap is
    # the max of configured and actual, so 8% actual -> 8% cap, and the
    # checklist still judges each order against it (an 8% order passes a
    # portfolio whose max IS 8%, but a 9% order in a 5%-capped batch fails).
    assert _staging_position_cap(None, settings, {"AAA": 0.09}) == pytest.approx(
        0.09
    )


def test_effective_cap_batch_passes_pretrade():
    """End-to-end: a 19-stock batch at the relaxed ~5.26% cap passes the
    position-limit check when given the effective cap."""
    from qrp.execution.paper import PaperBrokerAdapter
    from qrp.risk.guards import RiskConfig

    db = _bars_db({"AAA": [("2026-10-09", 100.0)]})
    broker = PaperBroker(
        db, starting_capital=400_000.0, execution="open",
        cost_model="flat", cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0),
    )
    adapter = PaperBrokerAdapter(broker)
    hr = DataHealthReport(
        checked_at="2026-10-09T00:00:00Z", tickers_checked=1,
        findings=[Finding(check="x", severity="info", ticker=None,
                          summary="ok")],
    )
    # 5.263% of $400k = $21,052 — under the $25k fat-finger per-order cap.
    orders = [
        {"ticker": "AAA", "side": "BUY", "shares": 210.52, "price": 100.0,
         "reason": ""},
    ]
    rep = P.run_pretrade_checks(
        orders, adapter, db_path=db, decision_date="2026-10-09",
        health_report=hr, audit_log=False, max_position_weight=1 / 19,
    )
    assert [c.name for c in rep.checks if c.status == "fail"] == []
    # ...while the un-relaxed 5% cap would still block it.
    rep2 = P.run_pretrade_checks(
        orders, adapter, db_path=db, decision_date="2026-10-09",
        health_report=hr, audit_log=False, max_position_weight=0.05,
    )
    assert any(
        c.name == "Position limits" and c.status == "fail"
        for c in rep2.checks
    )
