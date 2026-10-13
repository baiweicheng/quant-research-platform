"""Tests for the Phase P4 statistical factor risk model.

Covers qrp/risk/factor_model.py:

- On synthetic returns with a KNOWN 3-factor structure, the model
  recovers ~3 factors, the covariance is PSD, and the tracking-error /
  volatility forecasts match the analytic values within tolerance.
- The K-selection rule behaves (variance threshold, max-factor cap).
- Thin data degrades gracefully: dropped tickers are noted, a single
  stock falls back to a diagonal matrix, and empty history raises a
  plain-English ValueError (never a traceback).
- The optimizer runs end-to-end with the factor model as its covariance
  source (the P3 seam).
- The realized-vs-predicted calibration check runs on synthetic history
  and renders a plain-English verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qrp.ui.helpers as H
from qrp.data.edgar import upsert_facts
from qrp.data.store import init_db as init_market_db
from qrp.data.store import upsert_bars
from qrp.portfolio.covariance import (
    CovarianceModel,
    LedoitWolfCovariance,
    ledoit_wolf_shrinkage,
)
from qrp.portfolio.optimizer import optimize_weights
from qrp.risk.factor_model import (
    CalibrationCheck,
    StatisticalFactorCovariance,
    describe_factors,
    factor_covariance_matrix,
    fit_factor_model,
    portfolio_volatility,
    realized_vs_predicted,
    tracking_error,
)
from qrp.signals.store import init_signals_table, upsert_signals


# ---------------------------------------------------------------------------
# synthetic data with a KNOWN factor structure
# ---------------------------------------------------------------------------

N_STOCKS = 25
T_DAYS = 400
TRUE_FACTOR_VARS = [5e-4, 4e-4, 3e-4]  # daily factor variances
NOISE_VAR = 5e-5  # daily idiosyncratic variance
TICKERS = [f"S{i:02d}" for i in range(N_STOCKS)]


def _factor_dgp(
    n: int = N_STOCKS,
    t: int = T_DAYS,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (returns_df, loadings (N x 3), factor_vars).

    Three true statistical factors with random-sign loadings (no single
    dominant market factor, so the 80%-variance rule lands on K=3), plus
    small idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    fvars = np.array(TRUE_FACTOR_VARS)
    F = rng.normal(size=(t, 3)) * np.sqrt(fvars)
    B = rng.uniform(-1, 1, size=(n, 3))
    R = F @ B.T + rng.normal(size=(t, n)) * np.sqrt(NOISE_VAR)
    rets = pd.DataFrame(R, columns=[f"S{i:02d}" for i in range(n)])
    return rets, B, fvars


def _analytic_cov(B: np.ndarray, fvars: np.ndarray) -> np.ndarray:
    """Annualized true covariance of the DGP."""
    n = B.shape[0]
    return (B @ np.diag(fvars) @ B.T + np.diag(np.full(n, NOISE_VAR))) * 252


def _populate_db(path: Path, n_days: int = 400, seed: int = 0) -> list[str]:
    """Prices from the factor DGP + momentum signal + shares outstanding."""
    rng = np.random.default_rng(seed)
    rets, _B, _fvars = _factor_dgp(t=n_days, seed=seed)
    days = pd.bdate_range(end="2026-10-06", periods=n_days)
    assert len(days) == n_days
    init_market_db(path)
    init_signals_table(path)
    for i, t in enumerate(TICKERS):
        closes = 100.0 * np.cumprod(1 + rets[t].to_numpy())
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
            rng.normal(size=n_days), index=pd.DatetimeIndex(days)
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
    return TICKERS


# ---------------------------------------------------------------------------
# K selection: recover the known structure
# ---------------------------------------------------------------------------

def test_k_selection_recovers_true_factors():
    rets, _B, _f = _factor_dgp(seed=0)
    fit = fit_factor_model(rets)
    assert fit.n_factors == 3
    # Robust across seeds: approximately 3, never degenerate.
    for seed in (1, 2, 3):
        assert fit_factor_model(_factor_dgp(seed=seed)[0]).n_factors in (2, 3, 4)


def test_k_selection_rule_threshold_and_cap():
    rets, _B, _f = _factor_dgp(seed=0)
    assert fit_factor_model(rets, max_factors=2).n_factors == 2
    assert fit_factor_model(rets, variance_threshold=0.40).n_factors <= 2
    capped = fit_factor_model(rets, variance_threshold=0.999, max_factors=7)
    assert capped.n_factors == 7
    assert capped.total_variance_explained == pytest.approx(
        sum(capped.variance_explained)
    )


def test_fit_rejects_degenerate_panels():
    with pytest.raises(ValueError, match="[Aa]t least 2"):
        fit_factor_model(pd.DataFrame({"A": [0.01, 0.02, 0.03]}))
    flat = pd.DataFrame(
        np.full((50, 3), 0.001), columns=["A", "B", "C"]
    )
    with pytest.raises(ValueError, match="[Nn]o price movement"):
        fit_factor_model(flat)


# ---------------------------------------------------------------------------
# covariance math: PSD, TE/volatility vs analytic truth
# ---------------------------------------------------------------------------

def test_factor_covariance_is_psd():
    rets, _B, _f = _factor_dgp(seed=0)
    sigma = factor_covariance_matrix(fit_factor_model(rets))
    assert list(sigma.index) == TICKERS
    eigs = np.linalg.eigvalsh(sigma.to_numpy())
    assert (eigs >= -1e-8).all(), "factor covariance must be PSD"
    assert (np.diag(sigma.to_numpy()) > 0).all()


def test_te_forecast_matches_analytic_value():
    rets, B, fvars = _factor_dgp(seed=0)
    fit = fit_factor_model(rets)
    sigma = factor_covariance_matrix(fit)
    true_ann = _analytic_cov(B, fvars)
    w = {t: 0.2 for t in TICKERS[:5]}
    bench = {t: 1.0 / N_STOCKS for t in TICKERS}
    w_arr = np.array([w.get(t, 0.0) for t in TICKERS])
    b_arr = np.array([bench[t] for t in TICKERS])
    active = w_arr - b_arr
    te_true = float(np.sqrt(active @ true_ann @ active))
    te_model = float(np.sqrt(active @ sigma.to_numpy() @ active))
    assert te_model == pytest.approx(te_true, rel=0.30)
    # And the public function agrees with the direct computation.
    model = StatisticalFactorCovariance.__new__(StatisticalFactorCovariance)
    model.notes = []
    assert tracking_error(w, bench, "2026-10-06", _StubSigma(sigma)) == (
        pytest.approx(te_model)
    )


def test_portfolio_volatility_matches_analytic_value():
    rets, B, fvars = _factor_dgp(seed=0)
    sigma = factor_covariance_matrix(fit_factor_model(rets))
    true_ann = _analytic_cov(B, fvars)
    w = {t: 0.2 for t in TICKERS[:5]}
    w_arr = np.array([w.get(t, 0.0) for t in TICKERS])
    vol_true = float(np.sqrt(w_arr @ true_ann @ w_arr))
    vol_model = portfolio_volatility(w, "2026-10-06", _StubSigma(sigma))
    assert vol_model == pytest.approx(vol_true, rel=0.30)
    assert vol_model > 0


def test_tracking_error_zero_for_benchmark_itself():
    rets, _B, _f = _factor_dgp(seed=0)
    sigma = factor_covariance_matrix(fit_factor_model(rets))
    bench = {t: 1.0 / N_STOCKS for t in TICKERS}
    te = tracking_error(bench, bench, "2026-10-06", _StubSigma(sigma))
    assert te == pytest.approx(0.0, abs=1e-9)


class _StubSigma:
    """Minimal CovarianceModel wrapping a fixed matrix (seam check)."""

    def __init__(self, sigma: pd.DataFrame):
        self.sigma = sigma
        self.notes: list[str] = []

    def covariance(self, tickers: list[str], asof: str) -> pd.DataFrame:
        return self.sigma.loc[tickers, tickers]


def test_stub_satisfies_protocol():
    rets, _B, _f = _factor_dgp(seed=0)
    stub = _StubSigma(factor_covariance_matrix(fit_factor_model(rets)))
    assert isinstance(stub, CovarianceModel)


def test_describe_factors_is_plain_english():
    rets, _B, _f = _factor_dgp(seed=0)
    fit = fit_factor_model(rets)
    lines = describe_factors(fit, top_n=3)
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("Factor ")
        assert "explains" in line and "past movement" in line
        assert "%" in line
    # With a dominant market-like first factor, it says so.
    rng = np.random.default_rng(0)
    F = rng.normal(size=(400, 2)) * np.sqrt([8e-4, 2e-4])
    B = rng.uniform(-1, 1, size=(10, 2))
    B[:, 0] = np.abs(B[:, 0]) + 0.5
    R = F @ B.T + rng.normal(size=(400, 10)) * np.sqrt(5e-5)
    fit2 = fit_factor_model(
        pd.DataFrame(R, columns=[f"M{i}" for i in range(10)])
    )
    assert "broad market" in describe_factors(fit2, top_n=1)[0]


# ---------------------------------------------------------------------------
# DB-backed model: thin-data handling, notes, the P3 seam
# ---------------------------------------------------------------------------

def test_statistical_model_satisfies_protocol(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    model = StatisticalFactorCovariance(db)
    assert isinstance(model, CovarianceModel)
    sigma = model.covariance(TICKERS, "2026-10-06")
    assert sigma.shape == (N_STOCKS, N_STOCKS)
    eigs = np.linalg.eigvalsh(sigma.to_numpy())
    assert (eigs >= -1e-8).all()
    assert any("factor model" in n for n in model.notes)
    assert any("shared pattern" in n for n in model.notes)


def test_fit_report_counts_factors_and_variance(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    fit = StatisticalFactorCovariance(db).fit_report(TICKERS, "2026-10-06")
    assert 2 <= fit.n_factors <= 4
    assert fit.total_variance_explained >= 0.80
    assert fit.trading_days_used > 200
    assert list(fit.loadings.columns) == [
        f"Factor {i + 1}" for i in range(fit.n_factors)
    ]


def test_thin_history_drops_ticker_with_note(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    # One ticker with only a week of prices: too thin to keep.
    days = pd.bdate_range(end="2026-10-06", periods=7)
    closes = 50.0 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, 7))
    upsert_bars(
        pd.DataFrame(
            {
                "Date": days.strftime("%Y-%m-%d"),
                "Open": closes, "High": closes,
                "Low": closes, "Close": closes, "Volume": 1000,
            }
        ),
        "THIN",
        db,
    )
    model = StatisticalFactorCovariance(db)
    sigma = model.covariance([*TICKERS, "THIN"], "2026-10-06")
    assert "THIN" not in sigma.index
    assert any("THIN" in n and "too little" in n for n in model.notes)


def test_no_data_raises_plain_english(tmp_path):
    db = tmp_path / "m.db"
    init_market_db(db)  # tables exist, no prices
    model = StatisticalFactorCovariance(db)
    with pytest.raises(ValueError, match="[Dd]ownload prices"):
        model.covariance(TICKERS, "2026-10-06")


def test_single_ticker_diagonal_fallback(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    model = StatisticalFactorCovariance(db)
    sigma = model.covariance(["S00"], "2026-10-06")
    assert sigma.shape == (1, 1)
    assert sigma.iloc[0, 0] > 0
    assert any("independently" in n for n in model.notes)


def test_factor_model_accepts_timestamp_asof(tmp_path):
    """Same Timestamp-asof regression as the Ledoit-Wolf model: the Risk
    page passes Timestamps, and sqlite3 cannot bind them."""
    db = tmp_path / "m.db"
    _populate_db(db)
    model = StatisticalFactorCovariance(db)
    cov_ts = model.covariance(TICKERS[:6], pd.Timestamp("2026-10-06"))
    cov_str = model.covariance(TICKERS[:6], "2026-10-06")
    pd.testing.assert_frame_equal(cov_ts, cov_str)


def test_trading_dates_accepts_timestamp_asof(tmp_path):
    """_trading_dates binds asof straight into SQL — Timestamps must be
    normalized first (used by realized_vs_predicted on the Risk page)."""
    from qrp.risk.factor_model import _trading_dates

    db = tmp_path / "m.db"
    _populate_db(db)
    dates_ts = _trading_dates(TICKERS[:6], pd.Timestamp("2026-10-06"), db)
    dates_str = _trading_dates(TICKERS[:6], "2026-10-06", db)
    assert dates_ts == dates_str
    assert len(dates_ts) > 0


# ---------------------------------------------------------------------------
# optimizer end-to-end with the factor model as its covariance source
# ---------------------------------------------------------------------------

def test_optimizer_runs_with_factor_model(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    rng = np.random.default_rng(11)
    scores = {t: float(rng.normal()) for t in TICKERS}
    model = StatisticalFactorCovariance(db)
    portfolio, stats = optimize_weights(
        scores, asof="2026-10-06", db_path=db, covariance_model=model
    )
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-6
    assert all(w <= 0.05 + 1e-6 for w in portfolio.weights.values())
    assert all(w >= 0 for w in portfolio.weights.values())
    assert stats.tracking_error_ann >= 0
    assert stats.predicted_vol_ann > 0
    assert any("factor model" in n for n in portfolio.notes)


def test_tracking_error_works_with_shrinkage_model_too(tmp_path):
    """The forecast functions accept any CovarianceModel (the P3 seam)."""
    db = tmp_path / "m.db"
    _populate_db(db)
    w = {t: 0.2 for t in TICKERS[:5]}
    bench = {t: 1.0 / N_STOCKS for t in TICKERS}
    te = tracking_error(w, bench, "2026-10-06", LedoitWolfCovariance(db))
    assert te > 0
    vol = portfolio_volatility(w, "2026-10-06", LedoitWolfCovariance(db))
    assert vol > 0


# ---------------------------------------------------------------------------
# realized-vs-predicted calibration check
# ---------------------------------------------------------------------------

def test_calibration_runs_on_synthetic_history(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db, n_days=600, seed=1)
    model = StatisticalFactorCovariance(db)
    w = {t: 0.2 for t in TICKERS[:5]}
    bench = {t: 1.0 / N_STOCKS for t in TICKERS}
    check = realized_vs_predicted(
        w, bench, "2026-10-06", model, db_path=db,
        n_checkpoints=4, horizon_days=63,
    )
    assert isinstance(check, CalibrationCheck)
    assert len(check.rows) >= 1
    for row in check.rows:
        assert row["forecast_te"] > 0
        assert row["realized_te"] >= 0
    assert check.verdict, "the verdict must be a non-empty sentence"
    assert any(
        phrase in check.verdict
        for phrase in ("in line", "running hot", "running cold", "couldn't")
    )


def test_calibration_needs_history(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db, n_days=120, seed=1)
    model = StatisticalFactorCovariance(db)
    w = {t: 0.2 for t in TICKERS[:5]}
    bench = {t: 1.0 / N_STOCKS for t in TICKERS}
    with pytest.raises(ValueError, match="[Hh]istory"):
        realized_vs_predicted(
            w, bench, "2026-10-06", model, db_path=db,
            n_checkpoints=4, horizon_days=63,
        )


# ---------------------------------------------------------------------------
# dashboard wiring: covariance choice
# ---------------------------------------------------------------------------

def test_make_covariance_model_wiring():
    assert isinstance(
        H.make_covariance_model("shrinkage", "/nonexistent.db"),
        LedoitWolfCovariance,
    )
    assert isinstance(
        H.make_covariance_model("factor", "/nonexistent.db"),
        StatisticalFactorCovariance,
    )
    # Shrinkage stays the default — no behavior surprise.
    assert isinstance(
        H.make_covariance_model(None, "/nonexistent.db"),
        LedoitWolfCovariance,
    )
    with pytest.raises(ValueError, match="[Uu]nknown"):
        H.make_covariance_model("quantum", "/nonexistent.db")
    assert H.COVARIANCE_LABELS["shrinkage"] == "Shrinkage sample (simple)"
    assert H.COVARIANCE_LABELS["factor"] == "Statistical factor model"


def test_build_optimized_target_uses_factor_model(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    settings = {
        **H.OPTIMIZER_DEFAULTS,
        "covariance": "factor",
        "max_weight": 0.20,
    }
    portfolio, scores, day, stats = H.build_optimized_target_for_ui(
        "momentum", "2026-10-06", TICKERS, db, optimizer_settings=settings
    )
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-6
    assert any("factor model" in n for n in portfolio.notes)
    assert stats.tracking_error_ann >= 0


def test_build_optimized_target_defaults_to_shrinkage(tmp_path):
    db = tmp_path / "m.db"
    _populate_db(db)
    portfolio, _scores, _day, _stats = H.build_optimized_target_for_ui(
        "momentum", "2026-10-06", TICKERS, db,
        optimizer_settings={**H.OPTIMIZER_DEFAULTS, "max_weight": 0.20},
    )
    assert abs(sum(portfolio.weights.values()) - 1.0) < 1e-6
    assert not any("factor model" in n for n in portfolio.notes)
