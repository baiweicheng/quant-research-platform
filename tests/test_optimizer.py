"""Tests for the Phase P3 constrained mean-variance optimizer.

Covers qrp/portfolio/optimizer.py (objective, constraints, expected-return
mapping, benchmark weights, graceful degradation) and
qrp/portfolio/covariance.py (Ledoit-Wolf shrinkage, the CovarianceModel
seam, thin-data handling), plus the dashboard helpers that wire the
optimizer into the Portfolio page and paper-trading staging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qrp.ui.helpers as H
from qrp.data.store import init_db as init_market_db
from qrp.data.store import upsert_bars
from qrp.data.edgar import upsert_facts
from qrp.portfolio.covariance import (
    CovarianceModel,
    LedoitWolfCovariance,
    TRADING_DAYS_PER_YEAR,
    ledoit_wolf_shrinkage,
)
from qrp.portfolio.optimizer import (
    DEFAULT_EXPECTED_RETURN_SCALE,
    OptimizationStats,
    benchmark_weights,
    expected_returns_from_scores,
    optimize_weights,
)
from qrp.signals.store import init_signals_table, upsert_signals


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

N_TICKERS = 12
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]


def _synthetic_returns(n: int = 30, t: int = 300, seed: int = 7) -> pd.DataFrame:
    """Correlated daily returns (3 latent factors + idiosyncratic noise)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(t, 3))
    B = rng.normal(size=(3, n))
    R = F @ B * 0.01 + rng.normal(size=(t, n)) * 0.008
    return pd.DataFrame(
        R, columns=[f"S{i:02d}" for i in range(n)]
    )


def _synthetic_cov(n: int = 30, seed: int = 7) -> pd.DataFrame:
    return ledoit_wolf_shrinkage(_synthetic_returns(n, 300, seed))


class StubCovariance:
    """Minimal CovarianceModel: proves P4 can slot in behind the interface."""

    def __init__(self, sigma: pd.DataFrame):
        self.sigma = sigma
        self.notes: list[str] = []

    def covariance(self, tickers: list[str], asof: str) -> pd.DataFrame:
        return self.sigma.loc[tickers, tickers]


def _scores(n: int = 30, seed: int = 11) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {f"S{i:02d}": float(rng.normal()) for i in range(n)}


def _populate_db(path: Path, tickers: list[str] = TICKERS) -> None:
    """Prices (300 trading days) + a momentum signal + shares outstanding."""
    rng = np.random.default_rng(42)
    days = pd.bdate_range(end="2026-10-06", periods=300)
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
        sig = pd.Series(
            rng.normal(size=len(days)), index=pd.DatetimeIndex(days)
        )
        upsert_signals(sig, t, "momentum", path)
        facts = pd.DataFrame(
            [
                {
                    "tag": "shares_outstanding",
                    "period_end": "2026-06-30",
                    "filed_date": "2026-07-28",
                    "form": "10-Q",
                    "fp": "Q2",
                    "unit": "shares",
                    "value": 1_000_000_000.0 * (i + 1),
                }
            ]
        )
        upsert_facts(facts, t, path)


# ---------------------------------------------------------------------------
# expected-return mapping
# ---------------------------------------------------------------------------

def test_expected_returns_zscore_math():
    mu = expected_returns_from_scores({"A": 1.0, "B": 2.0, "C": 3.0})
    vals = np.array([1.0, 2.0, 3.0])
    z = (vals - vals.mean()) / vals.std()
    assert mu["A"] == pytest.approx(z[0] * DEFAULT_EXPECTED_RETURN_SCALE)
    assert mu["B"] == pytest.approx(0.0, abs=1e-12)
    assert mu["C"] == pytest.approx(z[2] * DEFAULT_EXPECTED_RETURN_SCALE)


def test_expected_returns_constant_scores_give_no_view():
    mu = expected_returns_from_scores({"A": 5.0, "B": 5.0})
    assert mu == {"A": 0.0, "B": 0.0}


def test_expected_returns_empty_and_nonfinite():
    assert expected_returns_from_scores({}) == {}
    mu = expected_returns_from_scores({"A": float("nan"), "B": 1.0})
    assert set(mu) == {"B"}


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------

def test_ledoit_wolf_psd_and_annualized():
    rets = _synthetic_returns()
    cov = ledoit_wolf_shrinkage(rets)
    eigs = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigs >= -1e-8).all(), "shrinkage covariance must be PSD"
    daily_var = rets.var().mean()
    assert np.diag(cov.to_numpy()).mean() == pytest.approx(
        daily_var * TRADING_DAYS_PER_YEAR, rel=0.5
    )


def test_ledoit_wolf_rejects_zero_variance():
    rets = _synthetic_returns()
    rets["S00"] = 0.001  # constant returns -> zero variance
    with pytest.raises(ValueError, match="[Pp]ositive variance"):
        ledoit_wolf_shrinkage(rets)


def test_ledoit_wolf_needs_data():
    with pytest.raises(ValueError, match="at least 2 observations"):
        ledoit_wolf_shrinkage(pd.DataFrame({"A": [0.01]}))


def test_ledoit_wolf_delta_matches_reference():
    """Shrinkage intensity matches the identity-target reference values.

    Expected deltas were validated against sklearn's LedoitWolf (the
    reference implementation of Ledoit & Wolf 2004 "Honey, I shrunk the
    sample covariance matrix") on fixed seeds. The pre-fix code mixed in
    the constant-correlation paper's off-diagonal asymptotics and
    systematically under-shrank (e.g. 0.0084 instead of 0.0164 on the
    factor panel below).
    """
    rng = np.random.default_rng(7)
    # Spherical: iid noise -> heavy shrinkage toward the identity.
    _, d_spherical = ledoit_wolf_shrinkage(
        pd.DataFrame(rng.normal(0, 1, (252, 40))), return_delta=True
    )
    assert float(d_spherical) == pytest.approx(0.9164, abs=0.005)
    # Factor-structured: 3 factors + idiosyncratic noise -> light shrinkage.
    t, n, k = 252, 40, 3
    f = rng.normal(0, 0.01, (t, k))
    b = rng.normal(0, 1, (n, k))
    e = rng.normal(0, 0.005, (t, n))
    _, d_factor = ledoit_wolf_shrinkage(
        pd.DataFrame(f @ b.T + e), return_delta=True
    )
    assert float(d_factor) == pytest.approx(0.0164, abs=0.005)


def test_ledoit_wolf_shrunk_closer_than_sample():
    """On a known factor DGP, shrinkage beats the raw sample covariance.

    With the true covariance known, the shrunk estimate must be closer
    to it (Frobenius norm) than the unshrunk sample covariance — the
    whole point of shrinkage. A mis-specified delta (e.g. the old
    hybrid rho_hat) can fail this.
    """
    rng = np.random.default_rng(11)
    t, n, k = 500, 20, 2
    b = rng.normal(0, 1, (n, k))
    f_var, e_var = 0.01**2, 0.005**2
    true_cov = b @ b.T * f_var + np.eye(n) * e_var
    f = rng.normal(0, np.sqrt(f_var), (t, k))
    e = rng.normal(0, np.sqrt(e_var), (t, n))
    rets = pd.DataFrame(f @ b.T + e)
    shrunk = ledoit_wolf_shrinkage(rets).to_numpy() / TRADING_DAYS_PER_YEAR
    sample = np.cov(rets.to_numpy(), rowvar=False, bias=True)
    err_shrunk = np.linalg.norm(shrunk - true_cov, "fro")
    err_sample = np.linalg.norm(sample - true_cov, "fro")
    assert err_shrunk < err_sample, (
        f"shrunk error {err_shrunk:.6f} should beat sample error "
        f"{err_sample:.6f}"
    )


# ---------------------------------------------------------------------------
# CovarianceModel seam + thin-data handling
# ---------------------------------------------------------------------------

def test_stub_covariance_satisfies_protocol():
    stub = StubCovariance(_synthetic_cov())
    assert isinstance(stub, CovarianceModel)  # runtime-checkable Protocol
    out = stub.covariance(["S00", "S01"], "2026-10-06")
    assert list(out.columns) == ["S00", "S01"]


def test_covariance_drops_thin_history_with_note(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db, TICKERS[:3])
    # one more ticker with only 10 days of history
    rng = np.random.default_rng(3)
    days = pd.bdate_range(end="2026-10-06", periods=10)
    closes = 50.0 * np.cumprod(1 + rng.normal(0, 0.01, size=10))
    df = pd.DataFrame(
        {
            "Date": days.strftime("%Y-%m-%d"),
            "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": 1000,
        }
    )
    upsert_bars(df, "THIN", db)
    model = LedoitWolfCovariance(db, min_observations=30)
    cov = model.covariance(["T00", "T01", "T02", "THIN"], "2026-10-06")
    assert "THIN" not in cov.columns
    assert any("THIN" in n and "little price history" in n for n in model.notes)


def test_covariance_no_data_raises_plain_english(tmp_path):
    db = tmp_path / "empty.db"
    init_market_db(db)
    model = LedoitWolfCovariance(db)
    with pytest.raises(ValueError, match="[Dd]ownload prices"):
        model.covariance(["AAA"], "2026-10-06")


def test_covariance_single_ticker_diagonal_fallback(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db, ["SOLO"])
    model = LedoitWolfCovariance(db)
    cov = model.covariance(["SOLO"], "2026-10-06")
    assert cov.shape == (1, 1)
    assert cov.iloc[0, 0] > 0
    assert any("independently" in n for n in model.notes)


# ---------------------------------------------------------------------------
# optimizer: constraints, determinism, turnover penalty
# ---------------------------------------------------------------------------

def _optimize(**kw):
    kw.setdefault("asof", "2026-10-06")
    kw.setdefault("covariance_model", StubCovariance(_synthetic_cov()))
    return optimize_weights(_scores(), **kw)


def test_optimizer_respects_all_constraints():
    port, stats = _optimize(tracking_error=0.05)
    w = port.weights
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= -1e-9 for v in w.values()), "long-only"
    assert all(v <= 0.05 + 1e-6 for v in w.values()), "5% cap"
    assert stats.tracking_error_ann <= 0.05 + 1e-4
    assert isinstance(stats, OptimizationStats)
    assert port.cash_weight == 0.0  # fully invested


def test_optimizer_is_deterministic():
    p1, _ = _optimize()
    p2, _ = _optimize()
    assert set(p1.weights) == set(p2.weights)
    assert all(
        abs(p1.weights[t] - p2.weights[t]) < 1e-9 for t in p1.weights
    )


def test_turnover_penalty_reduces_trading():
    # w_prev respects the 5% cap, so the penalty is free to hold still.
    w_prev = {f"S{i:02d}": 0.05 for i in range(20)}
    _, s_free = _optimize(w_prev=w_prev, turnover_penalty_bps=0.0)
    _, s_penalized = _optimize(w_prev=w_prev, turnover_penalty_bps=500.0)
    assert s_penalized.turnover_one_way <= s_free.turnover_one_way + 1e-9
    # and with a huge penalty it barely moves from the starting portfolio
    _, s_huge = _optimize(w_prev=w_prev, turnover_penalty_bps=100_000.0)
    assert s_huge.turnover_one_way < 0.05


def test_small_universe_relaxes_cap_with_note():
    scores = {f"S{i:02d}": float(i) for i in range(5)}
    port, _ = optimize_weights(
        scores, asof="2026-10-06",
        covariance_model=StubCovariance(_synthetic_cov(5, seed=9)),
        max_weight=0.05,
    )
    assert all(v <= 0.20 + 1e-6 for v in port.weights.values())
    assert abs(sum(port.weights.values()) - 1.0) < 1e-6
    assert any("relaxed" in n for n in port.notes)


def test_infeasible_tracking_error_falls_back_with_note():
    # tracking_error=0 forces w == w_b exactly; the equal-weight benchmark
    # over 30 names (3.3% each) respects the cap, so instead force the
    # conflict via a benchmark that breaks the cap.
    sigma = _synthetic_cov()
    stub = StubCovariance(sigma)

    import qrp.portfolio.optimizer as opt

    real_bench = opt.benchmark_weights
    try:
        # benchmark: one name at 50% -> violates the 5% cap -> TE=0 infeasible
        opt.benchmark_weights = lambda tickers, asof, db_path=None: (
            {t: (0.5 if i == 0 else 0.5 / 29) for i, t in enumerate(tickers)},
            "test benchmark",
        )
        port, stats = optimize_weights(
            _scores(), asof="2026-10-06", covariance_model=stub,
            tracking_error=0.0,
        )
    finally:
        opt.benchmark_weights = real_bench
    assert any("dropped" in n for n in port.notes)
    w = port.weights
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v <= 0.05 + 1e-6 for v in w.values())


def test_optimize_validates_inputs():
    with pytest.raises(ValueError, match="risk_aversion"):
        _optimize(risk_aversion=0.0)
    with pytest.raises(ValueError, match="turnover_penalty_bps"):
        _optimize(turnover_penalty_bps=-1.0)
    with pytest.raises(ValueError, match="tracking_error"):
        _optimize(tracking_error=-0.01)
    with pytest.raises(ValueError, match="[Nn]o usable signal scores"):
        optimize_weights(
            {}, asof="2026-10-06",
            covariance_model=StubCovariance(_synthetic_cov()),
        )


# ---------------------------------------------------------------------------
# benchmark weights
# ---------------------------------------------------------------------------

def test_benchmark_cap_weighted(tmp_path):
    db = tmp_path / "m.db"
    tickers = TICKERS[:4]
    _populate_db(db, tickers)
    weights, note = benchmark_weights(tickers, "2026-10-06", db)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # expected: shares_outstanding ((i+1) * 1e9) x latest close, normalized
    from qrp.signals.store import load_bars

    expected = {}
    for i, t in enumerate(tickers):
        bars = load_bars(t, end="2026-10-06", path=db)
        expected[t] = 1_000_000_000.0 * (i + 1) * float(bars["close"].iloc[-1])
    total = sum(expected.values())
    for t in tickers:
        assert weights[t] == pytest.approx(expected[t] / total, rel=1e-9)
    assert "company size" in note


def test_benchmark_equal_weight_fallback(tmp_path):
    db = tmp_path / "m.db"
    init_market_db(db)  # prices table exists but no fundamentals at all
    weights, note = benchmark_weights(["AAA", "BBB"], "2026-10-06", db)
    assert weights == {"AAA": 0.5, "BBB": 0.5}
    assert "equal-weighted" in note


# ---------------------------------------------------------------------------
# dashboard helpers
# ---------------------------------------------------------------------------

def test_weighting_labels_include_optimized():
    assert H.WEIGHTING_LABELS["optimized"] == "Optimized (mean-variance)"
    desc = H.weighting_description("optimized")
    assert "optimizer" in desc.lower()
    # rule-based descriptions still work
    assert "Equal weight" in H.weighting_description("equal")


def test_optimizer_defaults_are_sane():
    d = H.OPTIMIZER_DEFAULTS
    assert d["risk_aversion"] > 0
    assert d["turnover_penalty_bps"] >= 0
    assert 0 < d["tracking_error"] <= 1
    assert 0 < d["max_weight"] <= 1


def test_build_optimized_target_for_ui_end_to_end(tmp_path):
    # 25 tickers so the 5% per-stock cap can actually hold (>= 20 names).
    tickers = [f"U{i:02d}" for i in range(25)]
    db = tmp_path / "m.db"
    _populate_db(db, tickers)
    portfolio, scores, day, stats = H.build_optimized_target_for_ui(
        "momentum", "2026-10-06", tickers, db_path=db
    )
    assert day == "2026-10-06"
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-6
    assert all(v <= 0.05 + 1e-6 for v in portfolio.weights.values())
    assert stats.predicted_vol_ann > 0
    assert stats.tracking_error_ann >= 0
    assert stats.turnover_one_way == pytest.approx(0.5, abs=1e-6)  # from cash
    assert portfolio.method == "optimized"


def test_build_target_for_ui_optimized_branch(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    portfolio, scores, day = H.build_target_for_ui(
        "momentum", "optimized", "2026-10-06", TICKERS, db_path=db
    )
    assert portfolio.method == "optimized"
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-6


def test_build_target_for_ui_rule_based_ignores_optimizer_kwargs(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    p1, _, _ = H.build_target_for_ui(
        "momentum", "equal", "2026-10-06", TICKERS, db_path=db,
        max_weight=0.20,
    )
    p2, _, _ = H.build_target_for_ui(
        "momentum", "equal", "2026-10-06", TICKERS, db_path=db,
        max_weight=0.20, w_prev={"T00": 0.9},
        optimizer_settings={"risk_aversion": 9.9},
    )
    assert p1.weights == p2.weights  # rule-based path untouched


def test_current_weights():
    class FakeBroker:
        positions = {"AAA": 10.0, "BBB": 5.0, "CCC": 100.0}

        def last_price(self, ticker, date=None):
            if ticker == "CCC":
                raise KeyError("no price")
            return {"AAA": 100.0, "BBB": 200.0}[ticker]

    w = H.current_weights(FakeBroker(), 3000.0)
    # AAA: 10*100=1000 -> 1/3; BBB: 5*200=1000 -> 1/3; CCC unpriceable
    assert w == pytest.approx({"AAA": 1 / 3, "BBB": 1 / 3})
    assert H.current_weights(FakeBroker(), 0.0) == {}


def test_execution_cli_optimized_dry_run(tmp_path, capsys):
    """The paper-trading CLI accepts --method optimized (dry run)."""
    from qrp.execution import cli as exec_cli

    tickers = [f"U{i:02d}" for i in range(25)]
    db = tmp_path / "m.db"
    _populate_db(db, tickers)
    rc = exec_cli.main(
        ["--signal", "momentum", "--method", "optimized",
         "--date", "2026-10-06", "--tickers", ",".join(tickers),
         "--db", str(db), "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "optimized (mean-variance)" in out
    assert "Expected annual return" in out
    assert "WOULD BUY" in out or "WOULD SELL" in out


# ---------------------------------------------------------------------------
# UI path (Streamlit AppTest): Portfolio page with the optimizer
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_db_path() -> Path:
    return _repo_root() / "data" / "market.db"


@pytest.fixture
def populated_db():
    """Repo-root market.db with synthetic prices/signals (cleaned after)."""
    p = _default_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    _populate_db(p, [f"U{i:02d}" for i in range(25)])
    yield p
    if p.exists():
        p.unlink()


def test_portfolio_page_optimizer_end_to_end(populated_db):
    """The real dashboard: pick Optimized, build, see plain-English output."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_repo_root() / "qrp" / "ui" / "app.py"))
    at.session_state[H.NAV_PAGE_KEY] = "Portfolio"
    at.run()
    assert not at.exception

    at.selectbox(key="portfolio_method").set_value("optimized").run()
    assert not at.exception
    assert any(
        "optimizer" in str(c.value).lower() for c in at.caption
    ), "expected the optimizer's plain-English description"

    build_buttons = [b for b in at.button if b.label == "Build portfolio"]
    assert build_buttons, "Build portfolio button missing"
    build_buttons[0].click().run()
    assert not at.exception, f"page raised: {at.exception}"

    texts = " ".join(
        str(getattr(el, "label", "")) for el in at.metric
    ).lower()
    assert "expected annual return" in texts
    assert "predicted annual volatility" in texts
    assert "tracking error" in texts
    assert "turnover" in texts
    captions = " ".join(str(c.value) for c in at.caption).lower()
    assert "optimizer" in captions or "signal" in captions
    frames = [d.value for d in at.dataframe]
    assert frames, "expected a weight table"
    cols = list(frames[0].columns)
    assert "Ticker" in cols and "After" in cols and "Before" in cols


def test_infeasible_25_ticker_five_percent_cap_produces_feasible_portfolio():
    """Audit 2.2: 25 tickers x 5% cap — no cvxpy exception.

    25 names cannot each hold 5% (that sums to 125%), so the optimizer
    holds a feasible subset at the cap: fully invested, every weight
    within the cap, no solver exception, plain-English notes.
    """
    n = 25
    scores = {f"S{i:02d}": float(i) for i in range(n)}
    port, _ = optimize_weights(
        scores, asof="2026-10-06",
        covariance_model=StubCovariance(_synthetic_cov(n, seed=9)),
        max_weight=0.05,
    )
    assert abs(sum(port.weights.values()) - 1.0) < 1e-6
    assert all(v <= 0.05 + 1e-6 for v in port.weights.values())
    assert port.notes, "expected plain-English notes about the sizing"


def test_tiny_universe_relaxes_cap_instead_of_crashing():
    """Audit 2.2: when n * max_weight < 1 the cap relaxes to 1/n.

    With 3 tickers and a 5% cap the weights could never sum to 100% —
    the pre-check relaxes the cap (noted) instead of letting cvxpy fail.
    """
    n = 3
    scores = {f"S{i:02d}": float(i) for i in range(n)}
    port, _ = optimize_weights(
        scores, asof="2026-10-06",
        covariance_model=StubCovariance(_synthetic_cov(n, seed=9)),
        max_weight=0.05,
    )
    assert abs(sum(port.weights.values()) - 1.0) < 1e-6
    assert any("relaxed" in note for note in port.notes)


def test_benchmark_weights_many_matches_per_date(tmp_path):
    """Audit 2.7: batched benchmark weights equal the per-date calls.

    Same point-in-time semantics (filed_date <= asof applied in memory),
    one DB round trip instead of N+1 per date.
    """
    from qrp.portfolio.optimizer import benchmark_weights, benchmark_weights_many

    db = tmp_path / "b.db"
    tickers = [f"S{i:02d}" for i in range(10)]
    _populate_db(db, tickers)
    asofs = ["2026-08-15", "2026-09-15", "2026-10-06"]
    batched = benchmark_weights_many(tickers, asofs, db)
    assert set(batched) == set(asofs)
    for asof in asofs:
        w_single, note_single = benchmark_weights(tickers, asof, db)
        w_batch, note_batch = batched[asof]
        assert set(w_batch) == set(w_single)
        for t in w_single:
            assert w_batch[t] == pytest.approx(w_single[t], rel=1e-12)
        assert note_batch == note_single
