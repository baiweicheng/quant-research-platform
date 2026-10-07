"""Tests for roadmap P1 — risk reporting (VaR/CVaR, factor attribution,
French parser, stress windows).

No live downloads here: the French parser is tested against a saved
fixture, and the stress suite runs on synthetic price data.
"""

import numpy as np
import pandas as pd
import pytest

from qrp.data.store import upsert_bars
from qrp.risk.attribution import (
    REGRESSORS,
    attribution_short_line,
    attribution_summary,
    factor_attribution,
    significant_factors,
)
from qrp.risk.metrics import (
    compute_risk_metrics,
    daily_returns,
    historical_cvar,
    historical_var,
)
from qrp.signals.store import upsert_signals
from qrp.ui import helpers as H


# ---------------------------------------------------------------------------
# VaR / CVaR vs hand-computed values
# ---------------------------------------------------------------------------

def _equity_from_returns(returns: list[float]) -> pd.DataFrame:
    # Build the value path so the equity curve's own daily returns are
    # exactly `returns` (an equity curve's first value already reflects
    # any move before it, so start one step earlier at 100).
    values = [100.0]
    for r in returns:
        values.append(values[-1] * (1.0 + r))
    return pd.DataFrame({"portfolio_value": values})


def test_historical_var_hand_computed():
    # 21 returns: the two worst are -10% and -5%, the rest flat.
    # 5th percentile: index 0.05 * (21 - 1) = 1.0 -> exactly the -5% day.
    returns = pd.Series([-0.10, -0.05] + [0.0] * 19)
    assert historical_var(returns, 0.95) == pytest.approx(0.05)
    # 1st percentile: index 0.01 * 20 = 0.2 -> interpolates between the
    # -10% and -5% days: -(-0.10 + 0.2 * 0.05) = 0.09.
    assert historical_var(returns, 0.99) == pytest.approx(0.09)


def test_historical_cvar_hand_computed():
    returns = pd.Series([-0.10, -0.05] + [0.0] * 19)
    # Worst 5%: the -10% and -5% days -> average loss 7.5%.
    assert historical_cvar(returns, 0.95) == pytest.approx(0.075)
    # Worst 1%: only the -10% day is at/below the -9% cutoff.
    assert historical_cvar(returns, 0.99) == pytest.approx(0.10)
    # CVaR is always at least as big as the matching VaR.
    assert historical_cvar(returns, 0.95) >= historical_var(returns, 0.95)


def test_compute_risk_metrics_from_equity():
    returns = [-0.10, -0.05] + [0.0] * 19
    metrics = compute_risk_metrics(_equity_from_returns(returns))
    assert metrics["var_95"]["value"] == pytest.approx(0.05)
    assert metrics["cvar_95"]["value"] == pytest.approx(0.075)
    assert metrics["var_99"]["value"] == pytest.approx(0.09)
    assert metrics["daily_volatility"]["value"] == pytest.approx(
        pd.Series(returns).std(ddof=1)
    )
    # Every metric carries its plain-English explanation.
    for key, entry in metrics.items():
        for field in ("value", "label", "what", "why", "good"):
            assert field in entry, f"{key} missing {field}"
        assert entry["what"] and entry["why"] and entry["good"]


def test_compute_risk_metrics_empty_equity():
    metrics = compute_risk_metrics(
        pd.DataFrame({"portfolio_value": [100.0]})
    )
    assert metrics["var_95"]["value"] != metrics["var_95"]["value"]  # NaN
    assert H.metric_display("var_95", float("nan")) == "n/a"


def test_daily_returns_drops_bad_values():
    equity = pd.DataFrame({"portfolio_value": [100.0, 110.0, 0.0, 50.0]})
    rets = daily_returns(equity)
    # 100->110 = +10%; 110->0 = -100%; 0->50 is undefined -> dropped.
    assert list(rets.round(6)) == [0.10, -1.00]


# ---------------------------------------------------------------------------
# Factor attribution recovers known betas / alpha
# ---------------------------------------------------------------------------

def _synthetic_attribution_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(n, 6)), index=dates, columns=REGRESSORS
    )
    rf = pd.Series(0.0001, index=dates)
    true_betas = {
        "Mkt-RF": 1.2, "SMB": -0.3, "HML": 0.8,
        "RMW": 0.0, "CMA": 0.0, "UMD": 0.5,
    }
    alpha_daily = 0.0004
    noise = rng.normal(0, 0.002, n)
    strat = (
        alpha_daily
        + (factors * pd.Series(true_betas)).sum(axis=1)
        + rf.values
        + noise
    )
    strat = pd.Series(strat, index=dates)
    full = factors.copy()
    full["RF"] = rf
    return strat, full, true_betas, alpha_daily


def test_attribution_recovers_known_betas_and_alpha():
    strat, factors, true_betas, alpha_daily = _synthetic_attribution_data()
    res = factor_attribution(strat, factors)
    assert res["alpha_ann"] == pytest.approx(alpha_daily * 252, abs=0.02)
    for factor, beta in true_betas.items():
        assert res["betas"][factor] == pytest.approx(beta, abs=0.1), factor
    assert res["r_squared"] > 0.8
    assert res["n_obs"] == 500


def test_attribution_significance_flags_real_tilts():
    strat, factors, _, _ = _synthetic_attribution_data()
    res = factor_attribution(strat, factors)
    sig = significant_factors(res)
    for factor in ("Mkt-RF", "SMB", "HML", "UMD"):
        assert factor in sig, factor
    # True-zero betas should not look convincing (noise-only t-stats).
    for factor in ("RMW", "CMA"):
        assert factor not in sig, factor


def test_attribution_summary_plain_english():
    strat, factors, _, _ = _synthetic_attribution_data()
    res = factor_attribution(strat, factors)
    text = attribution_summary(res)
    assert "value" in text  # HML tilt described as value
    assert "momentum" in text  # UMD tilt
    assert "added about" in text  # positive alpha survives
    assert "R²" in text
    line = attribution_short_line(res)
    assert "value" in line and "momentum" in line


def test_attribution_negative_alpha_wording_has_no_plus_sign():
    # UAT catch: "trailed by about +0.7% a year" contradicted itself — the
    # negative-alpha wording must not carry a "+" sign.
    res = {
        "alpha_ann": -0.007,
        "betas": {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0,
                  "RMW": 0.0, "CMA": 0.0, "UMD": 0.0},
        "t_stats": {"Mkt-RF": 5.0, "SMB": 0.1, "HML": 0.1,
                    "RMW": 0.1, "CMA": 0.1, "UMD": 0.1},
        "r_squared": 0.8,
    }
    text = attribution_summary(res)
    assert "trailed by about 0.7% a year" in text
    assert "+0.7%" not in text
    line = attribution_short_line(res)
    assert "0.7%/yr behind tilts" in line
    assert "+0.7%" not in line


def test_attribution_needs_enough_overlap():
    strat, factors, _, _ = _synthetic_attribution_data()
    short = strat.iloc[:30]
    with pytest.raises(ValueError, match="60"):
        factor_attribution(short, factors)


def test_attribution_empty_factors_raises_plain_english():
    strat, _, _, _ = _synthetic_attribution_data()
    with pytest.raises(ValueError, match="[Nn]o factor data"):
        factor_attribution(strat, pd.DataFrame())


# ---------------------------------------------------------------------------
# French CSV parser on a saved fixture (no live download)
# ---------------------------------------------------------------------------

FIVE_FACTOR_FIXTURE = """\
Fama-French 5 factors, daily returns in percent
Some descriptive preamble line

 ,Mkt-RF,SMB,HML,RMW,CMA,RF
20200102, 0.50, -0.10, 0.20, 0.05, -0.03, 0.00
20200103, -99.99, 0.15, -999, 0.02, 0.01, 0.00
20200106, 0.30, 0.05, 0.10, 0.01, 0.02, 0.00

Copyright 2026 by Kenneth R. French
Trailing footer line
"""

MOMENTUM_FIXTURE = """\
Momentum factor, daily

 ,Mom
20200102, 0.40
20200103, -0.20

Copyright 2026 by Kenneth R. French
"""


def test_parse_french_csv_fixture():
    from qrp.data.french import parse_french_csv

    frame = parse_french_csv(FIVE_FACTOR_FIXTURE)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert frame.index[0] == pd.Timestamp("2020-01-02")
    assert len(frame) == 3  # footer lines are not data
    # Percent -> decimal.
    assert frame.loc["2020-01-02", "Mkt-RF"] == pytest.approx(0.005)
    assert frame.loc["2020-01-06", "CMA"] == pytest.approx(0.0002)
    # Missing codes -> NaN.
    assert np.isnan(frame.loc["2020-01-03", "Mkt-RF"])
    assert np.isnan(frame.loc["2020-01-03", "HML"])


def test_parse_french_csv_momentum_renamed_to_umd():
    from qrp.data.french import parse_french_csv

    frame = parse_french_csv(MOMENTUM_FIXTURE)
    assert "UMD" in frame.columns
    assert "Mom" not in frame.columns
    assert frame.loc["2020-01-02", "UMD"] == pytest.approx(0.004)


def test_parse_french_csv_no_header_raises_plain_english():
    from qrp.data.french import parse_french_csv

    with pytest.raises(ValueError, match="header"):
        parse_french_csv("just some text\nno table here\n")


def test_french_cache_roundtrip(tmp_path):
    from qrp.data.french import (
        default_factors_path,
        french_factors_status,
        load_french_factors,
    )

    db = tmp_path / "m.db"
    assert not french_factors_status(db)["downloaded"]
    assert load_french_factors(db).empty
    # Simulate a cached file (as download_french_factors would write).
    path = default_factors_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "Mkt-RF": [0.005, -0.002],
            "SMB": [0.001, 0.001],
            "HML": [0.002, 0.002],
            "RMW": [0.0, 0.0],
            "CMA": [0.0, 0.0],
            "UMD": [0.004, -0.002],
            "RF": [0.0, 0.0],
        }
    )
    frame.to_csv(path, index=False)
    status = french_factors_status(db)
    assert status["downloaded"]
    assert status["rows"] == 2
    assert status["first"] == "2020-01-02"
    assert status["last"] == "2020-01-03"
    loaded = load_french_factors(db)
    assert loaded.loc[pd.Timestamp("2020-01-02"), "Mkt-RF"] == pytest.approx(
        0.005
    )


# ---------------------------------------------------------------------------
# Stress windows end-to-end on synthetic data
# ---------------------------------------------------------------------------

def _seed_long_db(path, periods=900) -> None:
    """~3.5 years of synthetic bars + momentum signal from 2019."""
    dates = pd.bdate_range("2019-01-01", periods=periods)
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
    parity = (dates.month % 2).to_numpy().astype(float)
    upsert_signals(pd.Series(parity, index=dates), "AAA", "momentum", path)
    upsert_signals(pd.Series(1.0 - parity, index=dates), "BBB", "momentum", path)


def _run_one(db, tickers):
    def run(start, end):
        return H.run_backtest_for_ui(
            "momentum", tickers, "equal", 0.5, "monthly", 10.0, 100_000.0,
            start, end, db, cost_model="flat",
        )
    return run


def test_stress_suite_runs_all_three_windows(tmp_path):
    db = str(tmp_path / "s.db")
    tickers = ["AAA", "BBB"]
    _seed_long_db(db)
    rows = H.run_stress_suite_for_ui(_run_one(db, tickers))
    assert [r["Stress window"] for r in rows] == [
        "COVID crash (2020)",
        "2022 bear market",
        "Full history",
    ]
    for row in rows:
        for col in (
            "Total return", "Yearly growth (CAGR)", "Sharpe",
            "Worst fall", "Bad-day loss (VaR 95%)",
        ):
            assert col in row, f"{row['Stress window']} missing {col}"
        # No "Not enough data" — the synthetic db covers every window.
        assert "Note" not in row


def test_stress_suite_degrades_gracefully_on_missing_window(tmp_path):
    db = str(tmp_path / "s2.db")
    tickers = ["AAA", "BBB"]
    _seed_long_db(db, periods=60)  # only ~3 months in 2019

    def run(start, end):
        if start == "2020-01-01":
            raise ValueError("No price data in that date range.")
        return _run_one(db, tickers)(start, end)

    rows = H.run_stress_suite_for_ui(run)
    covid = rows[0]
    assert covid["Stress window"] == "COVID crash (2020)"
    assert "Not enough data" in covid["Note"]
    assert "Total return" not in covid


def test_risk_metrics_for_ui_reuses_existing_metrics(tmp_path):
    db = str(tmp_path / "s3.db")
    tickers = ["AAA", "BBB"]
    _seed_long_db(db, periods=120)
    result, metrics = _run_one(db, tickers)("2019-01-01", "2019-06-30")
    risk = H.risk_metrics_for_ui(result.equity, metrics)
    # New metrics come from qrp.risk.metrics; max_drawdown/volatility are
    # the very same entries the Backtest page already computes — reused,
    # not recomputed or duplicated.
    assert risk["max_drawdown"] is metrics["max_drawdown"]
    assert risk["volatility"] is metrics["volatility"]
    assert risk["var_95"]["value"] == risk["var_95"]["value"]  # not NaN


def test_compare_attribution_needs_factor_data(tmp_path):
    db = str(tmp_path / "s4.db")
    with pytest.raises(ValueError, match="[Ff]actor data"):
        H.compare_attribution_for_ui(
            {"Some strategy": pd.DataFrame()}, db_path=db
        )


def test_metric_display_formats_new_risk_keys():
    assert H.metric_display("var_95", 0.0234) == "+2.34%"
    assert H.metric_display("cvar_95", 0.031) == "+3.10%"
    assert H.metric_display("daily_volatility", 0.012) == "+1.20%"
    assert H.metric_display("var_99", float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# Dashboard regression tests (Streamlit AppTest)
# ---------------------------------------------------------------------------

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app(timeout: float = 30.0):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(
        str(root / "qrp" / "ui" / "app.py"), default_timeout=timeout
    )


@pytest.fixture
def seeded_db():
    """Default DB seeded with ~3.5y of synthetic bars + momentum signal."""
    from pathlib import Path

    from qrp.data.french import default_factors_path

    p = Path(__file__).resolve().parents[1] / "data" / "market.db"
    if p.exists():
        p.unlink()
    factor_csv = default_factors_path(p)
    if factor_csv.exists():
        factor_csv.unlink()
    _seed_long_db(str(p))
    yield p
    if p.exists():
        p.unlink()
    if factor_csv.exists():
        factor_csv.unlink()


def test_backtest_stress_preset_sets_dates(seeded_db):
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Backtest"
    at.run()
    assert not at.exception
    at.button(key="preset_covid").click().run()
    assert not at.exception
    assert at.session_state["backtest_start"] == "2020-01-01"
    assert at.session_state["backtest_end"] == "2020-12-31"
    at.button(key="preset_full").click().run()
    assert not at.exception
    assert at.session_state["backtest_start"] == "2019-01-01"


def test_backtest_stress_suite_renders_table(seeded_db):
    at = _app(timeout=120.0)
    at.session_state[H.NAV_PAGE_KEY] = "Backtest"
    at.run()
    assert not at.exception
    at.button(key="run_stress_suite").click().run()
    assert not at.exception, at.exception
    assert any(
        "Stress windows, side by side" in str(s.value)
        for s in at.subheader
    )
    tables = [df.value for df in at.dataframe]
    assert tables, "expected the side-by-side table to render"
    windows = set()
    for df in tables:
        if "Stress window" in df.columns:
            windows.update(df["Stress window"].tolist())
    assert {
        "COVID crash (2020)", "2022 bear market", "Full history"
    } <= windows


def test_backtest_risk_section_renders(seeded_db):
    at = _app(timeout=120.0)
    at.session_state[H.NAV_PAGE_KEY] = "Backtest"
    at.run()
    assert not at.exception
    at.button(key="run_backtest").click().run()
    assert not at.exception, at.exception
    assert any(
        "how bumpy was the ride" in str(s.value).lower()
        for s in at.subheader
    )
    assert any(
        "Bad-day loss threshold" in str(e.label) for e in at.expander
    ), "expected the VaR explainer"


def test_market_data_factor_download_button(seeded_db, monkeypatch):
    from qrp.data.french import default_factors_path

    def fake_download(db_path):
        path = default_factors_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": ["2020-01-02"],
                "Mkt-RF": [0.005],
                "SMB": [0.001],
                "HML": [0.002],
                "RMW": [0.0],
                "CMA": [0.0],
                "UMD": [0.004],
                "RF": [0.0],
            }
        ).to_csv(path, index=False)
        return {
            "rows": 1,
            "first": "2020-01-02",
            "last": "2020-01-02",
            "path": str(path),
        }

    monkeypatch.setattr(H, "download_french_factors_for_ui", fake_download)
    at = _app()
    at.session_state[H.NAV_PAGE_KEY] = "Market data"
    at.run()
    assert not at.exception
    assert any(b.label == "Download factor data" for b in at.button)
    # No factor data cached in this scratch db -> the friendly note shows.
    assert any(
        "not downloaded yet" in str(i.value).lower() for i in at.info
    )
    dl = [b for b in at.button if b.label == "Download factor data"][0]
    dl.click().run()
    assert not at.exception, at.exception
    # After the download + rerun, the status line shows the stored range.
    assert any("2020-01-02" in str(s.value) for s in at.success)


def test_market_beta_wording_follows_size_not_just_significance():
    from qrp.risk.attribution import _market_tilt_words

    assert _market_tilt_words(1.1) == "moves closely with the market"
    assert _market_tilt_words(0.5) == "moves partly with the market"
    assert _market_tilt_words(0.01) == "barely moves with the market"
    assert _market_tilt_words(-0.5) == "moves against the market"
