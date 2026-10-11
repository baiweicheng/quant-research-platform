"""Tests for the dual-momentum market gate overlay (backlog #3).

Synthetic SPY history: a rising year (risk-on) followed by a falling year
(risk-off). Gate expectations are hand-computed from the synthetic closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrp.data.edgar import init_db as init_edgar_db
from qrp.data.store import upsert_bars, upsert_dividends
from qrp.portfolio.market_gate import (
    MarketGateError,
    apply_gate,
    gate_state,
)
from qrp.strategies.registry import (
    Strategy,
    strategy_target_portfolio,
)


def _seed_spy(db, closes: np.ndarray, start: str = "2022-01-03") -> None:
    init_edgar_db(db)
    dates = pd.bdate_range(start, periods=len(closes))
    upsert_bars(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": closes,
                "High": closes * 1.01,
                "Low": closes * 0.99,
                "Close": closes,
                "Volume": 10_000_000,
            }
        ),
        "SPY",
        db,
    )


def _up_db(tmp_path) -> "Path":
    """SPY: +30% in year 1 (252 days), then -30% in year 2."""
    db = tmp_path / "gate.db"
    up = 400.0 * (1.001 ** np.arange(252))          # ~+28.6%
    down = up[-1] * (0.9986 ** np.arange(1, 253))   # ~-29.7%
    _seed_spy(db, np.concatenate([up, down]))
    return db


def test_gate_risk_on_after_rising_year(tmp_path):
    db = _up_db(tmp_path)
    # ~day 300: trailing 252d window is mostly the rising leg.
    state = gate_state("2023-03-01", db)
    assert state["risk_on"] is True
    assert state["spy_12m_return"] > 0
    assert "positive" in state["reason"]


def test_gate_risk_off_after_falling_year(tmp_path):
    db = _up_db(tmp_path)
    # Last bar: trailing 252d window is the falling leg.
    state = gate_state("2023-12-29", db)
    assert state["risk_on"] is False
    assert state["spy_12m_return"] < 0
    assert "negative" in state["reason"]


def test_gate_is_point_in_time(tmp_path):
    db = _up_db(tmp_path)
    # The same calendar date must give the same answer regardless of how
    # much later history exists in the DB (no peeking at the future).
    early = gate_state("2023-03-01", db)
    assert early["asof"] <= "2023-03-01"


def test_gate_needs_spy_data(tmp_path):
    db = tmp_path / "empty.db"
    init_edgar_db(db)
    with pytest.raises(MarketGateError, match="SPY"):
        gate_state("2023-06-01", db)


def test_gate_needs_12m_history(tmp_path):
    db = tmp_path / "short.db"
    _seed_spy(db, 400.0 * (1.001 ** np.arange(100)))
    with pytest.raises(MarketGateError, match="12 months"):
        gate_state("2022-06-01", db)


def test_apply_gate_scales_to_cash_when_risk_off(tmp_path):
    db = _up_db(tmp_path)
    weights = {"AAA": 0.05, "BBB": 0.03}
    new_w, note = apply_gate(weights, "2023-12-29", db)
    assert new_w == {}
    assert "cash" in note


def test_apply_gate_leaves_weights_when_risk_on(tmp_path):
    db = _up_db(tmp_path)
    weights = {"AAA": 0.05, "BBB": 0.03}
    new_w, note = apply_gate(weights, "2023-03-01", db)
    assert new_w == weights
    assert "positive" in note


def test_gate_includes_dividends_in_total_return(tmp_path):
    db = tmp_path / "div.db"
    closes = np.full(504, 400.0)  # flat price: price return = 0
    _seed_spy(db, closes)
    # A 10% dividend halfway through the window -> total return positive.
    with __import__("sqlite3").connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dividends"
            " (ticker TEXT, ex_date TEXT, amount REAL)"
        )
        conn.execute(
            "INSERT INTO dividends VALUES ('SPY', '2023-06-15', 40.0)"
        )
    state = gate_state("2023-12-29", db)
    assert state["risk_on"] is True, (
        "flat price + 10% dividend = positive total return"
    )


def test_strategy_market_gate_flag_zeros_portfolio(tmp_path):
    db = _up_db(tmp_path)
    dates = pd.bdate_range("2023-01-03", periods=100)
    matrices = {
        "momentum": pd.DataFrame(
            {"AAA": np.linspace(0.1, 0.9, 100), "BBB": np.linspace(0.9, 0.1, 100)},
            index=dates,
        )
    }
    strat = Strategy(
        name="gated_test",
        description="test",
        signals=[("momentum", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.5,
        market_gate=True,
    )
    # Risk-on date: weights survive.
    p_on, _ = strategy_target_portfolio(
        strat, matrices, pd.Timestamp("2023-03-01"), db_path=db
    )
    assert sum(p_on.weights.values()) > 0
    assert any("Market gate" in n for n in p_on.notes)
    # Risk-off date: everything to cash, with a plain-English note.
    p_off, _ = strategy_target_portfolio(
        strat, matrices, pd.Timestamp("2023-12-29"), db_path=db
    )
    assert p_off.weights == {}
    assert p_off.cash_weight == 1.0
    assert any("cash" in n for n in p_off.notes)


def test_strategy_without_gate_unaffected(tmp_path):
    db = _up_db(tmp_path)
    dates = pd.bdate_range("2023-01-03", periods=100)
    matrices = {
        "momentum": pd.DataFrame({"AAA": np.full(100, 0.8)}, index=dates),
    }
    strat = Strategy(
        name="ungated_test",
        description="test",
        signals=[("momentum", 1.0)],
        weighting="signal",
        rebalance="monthly",
        max_weight=0.5,
    )
    p, _ = strategy_target_portfolio(
        strat, matrices, pd.Timestamp("2023-12-29"), db_path=db
    )
    assert p.weights, "gate defaults off: risk-off date changes nothing"
