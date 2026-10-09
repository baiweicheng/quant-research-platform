"""Tests for the Backtest page's "Optimized (mean-variance)" sizing.

Covers qrp.backtest.strategies.optimized_strategy (re-solves the Phase P3
optimizer at every rebalance date), its wiring through
H.run_backtest_for_ui(weighting_key="optimized"), and the Backtest page UI
(the Position sizing dropdown gains "Optimized (mean-variance)" with the
same Optimizer settings expander as the Portfolio/Paper/Risk pages).

All data is SYNTHETIC (no network). The properties pinned down:

1. An optimized backtest runs end-to-end: target weights appear, respect
   the optimizer's cap contract, trades happen, equity stays finite.
2. NO LOOKAHEAD: changing signal values — or DB prices — dated after a
   rebalance date never changes the weights solved for that date.
3. Warm-up: before the risk model has enough price history, the strategy
   returns {} (stay in cash) instead of raising; it recovers once
   history exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qrp.ui.helpers as H
from qrp.backtest.strategies import optimized_strategy
from qrp.data.edgar import upsert_facts
from qrp.data.store import init_db as init_market_db
from qrp.data.store import upsert_bars
from qrp.signals.store import init_signals_table, upsert_signals

N_TICKERS = 30
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]
MAX_W = 0.05


def _populate_db(path: Path, tickers: list[str] = TICKERS, n_days: int = 400) -> pd.DatetimeIndex:
    """Prices (n_days trading days) + a momentum signal + shares outstanding."""
    rng = np.random.default_rng(42)
    days = pd.bdate_range(end="2024-12-31", periods=n_days)
    init_market_db(path)
    init_signals_table(path)
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0004 + i * 0.00002, 0.015, size=len(days))
        closes = 100.0 * np.cumprod(1 + rets)
        df = pd.DataFrame(
            {
                "Date": days.strftime("%Y-%m-%d"),
                "Open": closes * 0.999,
                "High": closes * 1.005,
                "Low": closes * 0.995,
                "Close": closes,
                "Volume": 1_000_000,
            }
        )
        upsert_bars(df, t, path)
        sig = pd.Series(rng.normal(size=len(days)), index=pd.DatetimeIndex(days))
        upsert_signals(sig, t, "momentum", path)
        facts = pd.DataFrame(
            [
                {
                    "tag": "shares_outstanding",
                    "period_end": "2024-06-30",
                    "filed_date": "2024-07-28",
                    "form": "10-Q",
                    "fp": "Q2",
                    "unit": "shares",
                    "value": 1_000_000_000.0 * (i + 1),
                }
            ]
        )
        upsert_facts(facts, t, path)
    return days


def _signal_matrix(path: Path) -> pd.DataFrame:
    """Momentum signal matrix exactly as stored (dates x tickers)."""
    from qrp.backtest.engine import load_signal_matrix

    return load_signal_matrix("momentum", TICKERS, path=path)


def _assert_weights_contract(weights: dict[str, float], cap: float) -> None:
    """Engine invariant + the optimizer's documented cap contract.

    Weights are long-only and sum to <= 1. Each weight respects the cap,
    except when the optimizer's own documented relaxation applies:
    fewer than 1/cap holdings remain, so the cap becomes 1/k (noted in
    the optimizer's notes) — the only case a weight may exceed `cap`.
    """
    assert weights, "expected a non-empty target portfolio"
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) <= 1.0 + 1e-9
    if len(weights) * cap >= 1.0:
        assert all(w <= cap + 1e-6 for w in weights.values()), (
            f"cap {cap} broken with {len(weights)} holdings: {weights}"
        )


# ---------------------------------------------------------------------------
# strategy function, direct
# ---------------------------------------------------------------------------


def test_optimized_strategy_solves_with_valid_weights(tmp_path):
    db = tmp_path / "m.db"
    days = _populate_db(db)
    matrix = _signal_matrix(db)
    fn = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)
    weights = fn(matrix, days[-1])
    _assert_weights_contract(weights, MAX_W)
    # Fully invested (the optimizer's contract): no deliberate cash.
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert fn.weighting_method == "optimized"


def test_optimized_strategy_no_lookahead(tmp_path):
    """Mutating future signals or future DB prices never changes a solve."""
    db = tmp_path / "m.db"
    days = _populate_db(db)
    matrix = _signal_matrix(db)
    asof = days[300]

    before = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)(
        matrix, asof
    )
    assert before

    # 1) Future signal values: wildly different after `asof`.
    future = matrix.copy()
    future.loc[future.index > asof] = (
        future.loc[future.index > asof] * -100.0 + 5.0
    )
    after_signals = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)(
        future, asof
    )
    assert after_signals.keys() == before.keys()
    for t, w in before.items():
        assert after_signals[t] == pytest.approx(w, abs=1e-9)

    # 2) Future prices in the DB: triple every close after `asof`.
    for t in TICKERS:
        rows = matrix.index > asof
        future_days = matrix.index[rows]
        closes = np.full(len(future_days), 300.0)
        df = pd.DataFrame(
            {
                "Date": future_days.strftime("%Y-%m-%d"),
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": 1_000_000,
            }
        )
        upsert_bars(df, t, db)
    after_prices = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)(
        matrix, asof
    )
    assert after_prices.keys() == before.keys()
    for t, w in before.items():
        assert after_prices[t] == pytest.approx(w, abs=1e-9)


def test_optimized_strategy_warmup_stays_in_cash_then_recovers(tmp_path):
    db = tmp_path / "m.db"
    days = _populate_db(db)
    matrix = _signal_matrix(db)
    fn = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)
    # Second trading day: the risk model has far too little price history
    # (it needs 30+ observations) — stay in cash, never raise.
    assert fn(matrix, days[1]) == {}
    # No signal history at all on or before the date -> cash.
    assert fn(matrix, days[0] - pd.Timedelta(days=30)) == {}
    # Once history exists the same strategy solves again.
    weights = fn(matrix, days[-1])
    _assert_weights_contract(weights, MAX_W)


def test_optimized_strategy_empty_db_stays_in_cash(tmp_path):
    db = tmp_path / "empty.db"
    init_market_db(db)
    init_signals_table(db)
    matrix = pd.DataFrame(
        {t: [1.0, 2.0] for t in TICKERS[:5]},
        index=pd.to_datetime(["2024-06-03", "2024-06-04"]),
    )
    fn = optimized_strategy("momentum", db_path=db, max_weight=MAX_W)
    assert fn(matrix, pd.Timestamp("2024-06-04")) == {}


class _StubCovariance:
    """Fixed identity-ish covariance: proves the injection seam works."""

    def __init__(self, tickers: list[str]):
        self.sigma = pd.DataFrame(
            np.eye(len(tickers)) * 0.04, index=tickers, columns=tickers
        )
        self.notes: list[str] = []

    def covariance(self, tickers: list[str], asof: str) -> pd.DataFrame:
        return self.sigma.loc[tickers, tickers]


def test_optimized_strategy_covariance_seam_and_settings(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    matrix = _signal_matrix(db)
    asof = matrix.index[-1]
    fn = optimized_strategy(
        "momentum",
        db_path=db,
        optimizer_settings={"max_weight": 0.2, "risk_aversion": 4.0},
        covariance_model=_StubCovariance(TICKERS),
    )
    weights = fn(matrix, asof)
    assert weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    # 30 names with a 20% cap can never trigger the 1/k relaxation.
    assert all(w <= 0.2 + 1e-6 for w in weights.values())


# ---------------------------------------------------------------------------
# through run_backtest_for_ui
# ---------------------------------------------------------------------------


def test_run_backtest_for_ui_optimized_end_to_end(tmp_path):
    db = tmp_path / "m.db"
    days = _populate_db(db)
    result, metrics = H.run_backtest_for_ui(
        "momentum", TICKERS, "optimized", MAX_W, "monthly",
        10.0, 100_000.0,
        days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"),
        str(db),
        optimizer_settings={"max_weight": MAX_W},
    )
    non_empty = {d: w for d, w in result.target_weights.items() if w}
    assert len(non_empty) >= 3, "expected several solved rebalances"
    for weights in non_empty.values():
        _assert_weights_contract(weights, MAX_W)
    assert not result.trades.empty
    assert np.isfinite(result.equity["portfolio_value"]).all()
    assert metrics["final_value"]["value"] > 0


def test_run_backtest_for_ui_optimized_factor_model(tmp_path):
    """The 'factor' Risk model choice solves in backtests too."""
    db = tmp_path / "m.db"
    days = _populate_db(db)
    result, _metrics = H.run_backtest_for_ui(
        "momentum", TICKERS, "optimized", MAX_W, "monthly",
        10.0, 100_000.0,
        days[-120].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"),
        str(db),
        optimizer_settings={"max_weight": MAX_W, "covariance": "factor"},
    )
    non_empty = [w for w in result.target_weights.values() if w]
    assert non_empty, "expected at least one solved rebalance"
    for weights in non_empty:
        _assert_weights_contract(weights, MAX_W)


def test_run_backtest_for_ui_rule_based_unchanged(tmp_path):
    """The refactor left the rule-based path behaving exactly as before."""
    db = tmp_path / "m.db"
    days = _populate_db(db)
    result, _metrics = H.run_backtest_for_ui(
        "momentum", TICKERS, "equal", 0.05, "monthly",
        10.0, 100_000.0,
        days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"),
        str(db),
        optimizer_settings=None,  # ignored for rule-based sizing
    )
    non_empty = [w for w in result.target_weights.values() if w]
    assert non_empty
    for weights in non_empty:
        # Equal weight: every position the same size.
        assert len(set(round(w, 9) for w in weights.values())) == 1


# ---------------------------------------------------------------------------
# Backtest page UI (AppTest)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def seeded_default_db():
    """Give the real app a data/market.db with prices + signals, then clean up."""
    db = _repo_root() / "data" / "market.db"
    backup = None
    if db.exists():
        backup = db.read_bytes()
    db.parent.mkdir(parents=True, exist_ok=True)
    _populate_db(db, tickers=["AAA", "BBB", "CCC"], n_days=60)
    try:
        yield db
    finally:
        if backup is not None:
            db.write_bytes(backup)
        elif db.exists():
            db.unlink()
        for suffix in ("-wal", "-shm"):
            extra = db.with_name(db.name + suffix)
            if extra.exists():
                extra.unlink()


def test_backtest_page_offers_optimized_sizing(seeded_default_db):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_repo_root() / "qrp" / "ui" / "app.py"))
    at.session_state[H.NAV_PAGE_KEY] = "Backtest"
    at.run()
    assert not at.exception

    sizing = next(s for s in at.selectbox if s.label == "Position sizing")
    assert sizing.options == [
        "Equal weight",
        "Signal weight",
        "Inverse volatility",
        "Optimized (mean-variance)",
    ]
    slider_labels = [s.label for s in at.slider]
    assert "Max weight per stock" in slider_labels

    sizing.set_value("Optimized (mean-variance)").run()
    assert not at.exception
    slider_labels = [s.label for s in at.slider]
    # The optimizer's own controls appear; the rule-based standalone
    # max-weight slider is gone (the optimizer's % slider governs).
    assert "Risk aversion" in slider_labels
    assert "Turnover penalty (basis points)" in slider_labels
    assert "Max weight per stock (%)" in slider_labels
    assert "Max weight per stock" not in slider_labels
    risk_model = next(s for s in at.selectbox if s.label == "Risk model")
    assert risk_model.options == [
        "Shrinkage sample (simple)",
        "Statistical factor model",
    ]
