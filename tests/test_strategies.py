"""Tests for Phase 8: fundamental signals, strategy registry, drift buffers.

All tests use SYNTHETIC data (no network). The most important one is
test_fundamentals_are_point_in_time: a fact filed AFTER the signal date
must never influence the signal — the no-peeking rule for fundamentals.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from qrp.backtest.engine import run_backtest
from qrp.backtest.strategies import top_n_by_signal
from qrp.data.edgar import facts_as_of, init_db
from qrp.data.store import upsert_bars
from qrp.signals.fundamental import (
    ValueComposite,
    latest_known_asof,
    _load_fact_frames,
)
from qrp.strategies import (
    composite_scores,
    get_strategy,
    list_strategies,
    strategy_target_weights,
)

INITIAL = 100_000.0


def _bars(tickers: list[str], closes: dict[str, float], periods: int = 400):
    """Flat-price bars DataFrame per ticker (Date/Open/.../Volume columns)."""
    dates = pd.bdate_range("2023-01-02", periods=periods)
    out = {}
    for t in tickers:
        c = closes[t]
        out[t] = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": c,
                "High": c * 1.01,
                "Low": c * 0.99,
                "Close": c,
                "Volume": 1_000_000,
            }
        )
    return out


def _fact(
    db,
    ticker: str,
    tag: str,
    period_end: str,
    filed_date: str,
    value: float,
    form: str = "10-K",
):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, ?, ?, ?, ?, 'FY', 'USD', ?)",
            (ticker, tag, period_end, filed_date, form, value),
        )


def _setup_two_stock_db(db, periods: int = 400):
    """A and B: flat $100 prices; identical filings except where noted."""
    init_db(db)
    for t, df in _bars(["A", "B"], {"A": 100.0, "B": 100.0}, periods).items():
        upsert_bars(df, t, db)
    for t in ("A", "B"):
        _fact(db, t, "shares_outstanding", "2022-12-31", "2023-02-01", 1e9)
        _fact(db, t, "net_income", "2022-12-31", "2023-02-01", 1e8)
        _fact(db, t, "cashflow_ops", "2022-12-31", "2023-02-01", 1e8)
        _fact(db, t, "equity", "2022-12-31", "2023-02-01", 1e9)
    return db


# --- point-in-time correctness (the most important test) --------------------

def test_fundamentals_are_point_in_time(tmp_path):
    """A fact filed AFTER the signal date must never influence the signal.

    A files a restated equity number (1000x bigger) whose filing date is
    after the as-of date. Storage is append-only, so the original filing
    stays knowable until the restatement lands — the future number must
    never leak into an earlier signal (neither directly nor via the
    composite), and the restated number must take over afterwards.
    """
    from qrp.signals.fundamental import fundamental_matrix

    db = _setup_two_stock_db(tmp_path / "t.db")
    _fact(db, "A", "equity", "2022-12-31", "2024-06-01", 1e12)  # restatement

    dates = pd.bdate_range("2023-01-02", periods=400)
    eq = fundamental_matrix(["A", "B"], "equity", dates, db)
    early, late = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-07-01")
    # Before the restatement filing, A's ORIGINAL number is visible (not
    # erased); B's normally-filed number is visible.
    assert eq.loc[early, "A"] == pytest.approx(1e9)
    assert eq.loc[early, "B"] == pytest.approx(1e9)
    # After the filing date, the restated number is knowable.
    assert eq.loc[late, "A"] == pytest.approx(1e12)

    # The composite signal built on it stays honest too: at the early date
    # A and B tie on all three components (0.75 each) — a leak of the 1e12
    # filing would push A to ~0.83.
    matrix = ValueComposite().compute_all(["A", "B"], db)
    assert matrix.loc[early, "A"] == pytest.approx(0.75)
    assert matrix.loc[early, "B"] == pytest.approx(0.75)
    assert matrix.loc[late, "A"] > matrix.loc[late, "B"]


def test_bulk_loader_matches_facts_as_of(tmp_path):
    """The fast bulk loader agrees with facts_as_of on every sampled date."""
    db = _setup_two_stock_db(tmp_path / "t.db")
    _fact(db, "A", "net_income", "2023-12-31", "2024-02-15", 2e8)
    _fact(db, "A", "net_income", "2022-12-31", "2024-05-01", 1.5e8)  # restatement
    frames = _load_fact_frames(["A"], ["net_income"], db)
    facts = frames[("A", "net_income")]
    dates = pd.bdate_range("2023-01-02", periods=400)
    bulk = latest_known_asof(dates, facts)
    # Sample trading days across filing regimes (all are weekdays).
    for day in ["2023-06-01", "2024-01-10", "2024-03-01", "2024-06-03"]:
        ref = facts_as_of("A", "net_income", day, db)
        expected = (
            ref.loc[ref["period_end"].idxmax(), "value"] if not ref.empty else np.nan
        )
        got = bulk.loc[pd.Timestamp(day)]
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected)


def test_value_percentiles_matches_reference_loop(tmp_path):
    """The vectorized _value_percentiles equals the old per-day loop.

    The loop is kept here as a test oracle: rank each ratio across
    tickers one day at a time, average over the components available.
    """
    from qrp.signals.fundamental import (
        MIN_VALUE_COMPONENTS,
        TAG_CASHFLOW_OPS,
        TAG_EQUITY,
        TAG_NET_INCOME,
        _market_cap_matrix,
        _value_percentiles,
        fundamental_matrix,
    )

    db = tmp_path / "v.db"
    init_db(db)
    for t, df in _bars(
        ["A", "B", "C"], {"A": 50.0, "B": 100.0, "C": 200.0}, 60
    ).items():
        upsert_bars(df, t, db)
    # Varying fundamentals; C files only equity (1 component < 2 -> NaN).
    _fact(db, "A", "shares_outstanding", "2022-12-31", "2023-02-01", 1e9)
    _fact(db, "B", "shares_outstanding", "2022-12-31", "2023-02-01", 2e9)
    _fact(db, "C", "shares_outstanding", "2022-12-31", "2023-02-01", 1e9)
    _fact(db, "A", "equity", "2022-12-31", "2023-02-01", 1e9)
    _fact(db, "B", "equity", "2022-12-31", "2023-02-01", 3e9)
    _fact(db, "C", "equity", "2022-12-31", "2023-02-01", 2e9)
    _fact(db, "A", "net_income", "2022-12-31", "2023-03-01", 5e8)
    _fact(db, "B", "net_income", "2022-12-31", "2023-03-01", 1e8)
    _fact(db, "A", "cashflow_ops", "2022-12-31", "2023-03-01", 4e8)
    _fact(db, "B", "cashflow_ops", "2022-12-31", "2023-03-01", 6e8)

    dates = pd.bdate_range("2023-01-02", periods=60)
    tickers = ["A", "B", "C"]
    got = _value_percentiles(tickers, dates, db)

    # Reference: the exact per-day loop the old implementation used.
    mcap, _ = _market_cap_matrix(tickers, dates, db)
    ratios = {}
    for tag in (TAG_EQUITY, TAG_NET_INCOME, TAG_CASHFLOW_OPS):
        vals = fundamental_matrix(tickers, tag, dates, db)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[tag] = vals / mcap.replace(0, np.nan)
    expected = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for day in dates:
        row = pd.DataFrame({k: v.loc[day] for k, v in ratios.items()})
        enough = row.notna().sum(axis=1) >= MIN_VALUE_COMPONENTS
        score = row.rank(pct=True).mean(axis=1, skipna=True)
        score[~enough] = np.nan
        expected.loc[day] = score

    pd.testing.assert_frame_equal(got, expected)
    # Sanity: A is cheapest on earnings, B on book; C is always NaN.
    last = dates[-1]  # all filings knowable by the final date
    assert got.loc[last, "A"] > got.loc[last, "B"]
    assert got["C"].isna().all()


def test_shared_inputs_match_unshared(tmp_path):
    """compute_all(..., shared=...) equals the standalone computation."""
    from qrp.signals.fundamental import (
        Profitability,
        TrendingValue,
        _SharedInputs,
    )

    db = _setup_two_stock_db(tmp_path / "t.db")
    tickers = ["A", "B"]
    shared = _SharedInputs(tickers, db)
    for sig in (ValueComposite(), Profitability(), TrendingValue()):
        pd.testing.assert_frame_equal(
            sig.compute_all(tickers, db, shared=shared),
            sig.compute_all(tickers, db),
        )


def test_cheaper_stock_scores_higher(tmp_path):
    """Higher book/earnings/cash-flow per dollar of price => higher score."""
    db = tmp_path / "t.db"
    init_db(db)
    for t, df in _bars(["A", "B"], {"A": 100.0, "B": 100.0}).items():
        upsert_bars(df, t, db)
    for t in ("A", "B"):
        _fact(db, t, "shares_outstanding", "2022-12-31", "2023-02-01", 1e9)
    # A is 5x cheaper on every component; same $100 price => same market cap.
    for tag, aval, bval in [
        ("equity", 5e9, 1e9),
        ("net_income", 5e8, 1e8),
        ("cashflow_ops", 5e8, 1e8),
    ]:
        _fact(db, "A", tag, "2022-12-31", "2023-02-01", aval)
        _fact(db, "B", tag, "2022-12-31", "2023-02-01", bval)
    matrix = ValueComposite().compute_all(["A", "B"], db)
    day = pd.Timestamp("2024-03-01")
    assert matrix.loc[day, "A"] == pytest.approx(1.0)
    assert matrix.loc[day, "B"] == pytest.approx(0.5)
    assert matrix.loc[day, "A"] > matrix.loc[day, "B"]


def test_value_composite_skips_data_starved_stocks(tmp_path):
    """A stock missing 2+ of the 3 value components gets NaN, not a guess."""
    db = tmp_path / "t.db"
    init_db(db)
    for t, df in _bars(["A", "B"], {"A": 100.0, "B": 100.0}).items():
        upsert_bars(df, t, db)
    for t in ("A", "B"):
        _fact(db, t, "shares_outstanding", "2022-12-31", "2023-02-01", 1e9)
        _fact(db, t, "equity", "2022-12-31", "2023-02-01", 1e9)
    # B has only 1 of 3 components (equity); A has all three.
    _fact(db, "A", "net_income", "2022-12-31", "2023-02-01", 1e8)
    _fact(db, "A", "cashflow_ops", "2022-12-31", "2023-02-01", 1e8)
    matrix = ValueComposite().compute_all(["A", "B"], db)
    day = pd.Timestamp("2024-03-01")
    assert pd.notna(matrix.loc[day, "A"])
    assert pd.isna(matrix.loc[day, "B"])


# --- strategy registry -------------------------------------------------------

def test_registry_has_all_six_strategies():
    names = {n for n, _ in list_strategies()}
    assert {
        "trending_value",
        "value_momentum",
        "quality_lowvol",
        "momentum",
        "trend",
        "low_volatility",
    } <= names


def test_every_strategy_is_explained_in_plain_english():
    for name, _ in list_strategies():
        s = get_strategy(name)
        assert s.description.strip(), name
        assert s.good_at.strip(), name
        assert s.main_risk.strip(), name
        assert s.data_used.strip(), name
        assert s.signals, name
        assert s.rebalance in ("daily", "weekly", "monthly"), name


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_composite_scores_blend_ranks():
    dates = pd.bdate_range("2024-01-02", periods=10)
    m1 = pd.DataFrame({"A": [3.0] * 10, "B": [1.0] * 10}, index=dates)
    m2 = pd.DataFrame({"A": [1.0] * 10, "B": [3.0] * 10}, index=dates)
    s = get_strategy("value_momentum")  # 0.5 / 0.5 blend
    scores = composite_scores(s, {"value_composite": m1, "momentum": m2}, dates[-1])
    # A: (1.0 + 0.5)/2 = 0.75 ; B: (0.5 + 1.0)/2 = 0.75
    assert scores["A"] == pytest.approx(0.75)
    assert scores["B"] == pytest.approx(0.75)


def test_strategy_target_weights_are_long_only():
    dates = pd.bdate_range("2024-01-02", periods=10)
    mats = {
        "trending_value": pd.DataFrame(
            {"A": [0.9] * 10, "B": [0.1] * 10}, index=dates
        )
    }
    s = get_strategy("trending_value")
    weights, scores = strategy_target_weights(s, mats, dates[-1])
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) <= 1.0 + 1e-9
    assert set(weights) <= {"A", "B"}


# --- drift buffers -----------------------------------------------------------

def _alt_targets(_matrix: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
    """Alternate the 60/40 split every month (forces real turnover)."""
    return {"A": 0.6, "B": 0.4} if day.month % 2 == 0 else {"A": 0.4, "B": 0.6}


def test_drift_tolerance_skips_small_trades():
    dates = pd.bdate_range("2024-01-02", periods=120)
    prices = pd.DataFrame(
        {"A": np.full(120, 100.0), "B": np.full(120, 100.0)}, index=dates
    )
    shape = pd.DataFrame(
        {"A": np.ones(120), "B": np.ones(120)}, index=dates
    )
    plain = run_backtest(
        prices, shape, _alt_targets, frequency="monthly",
        cost_bps=0.0, initial_capital=INITIAL, drift_tolerance=0.0,
    )
    buffered = run_backtest(
        prices, shape, _alt_targets, frequency="monthly",
        cost_bps=0.0, initial_capital=INITIAL, drift_tolerance=0.25,
    )
    # Each monthly flip moves a position by 0.2 of the portfolio < 0.25, so
    # after the initial build no further trades happen.
    assert len(buffered.trades) < len(plain.trades)
    first_day = buffered.trades["date"].min()
    assert (buffered.trades["date"] == first_day).all()


def test_drift_tolerance_zero_preserves_old_behavior():
    dates = pd.bdate_range("2024-01-02", periods=60)
    prices = pd.DataFrame(
        {"A": 100.0 * (1.001 ** np.arange(60)),
         "B": np.full(60, 100.0)}, index=dates
    )
    sig = pd.DataFrame(
        {"A": np.ones(60), "B": np.zeros(60)}, index=dates
    )
    res = run_backtest(
        prices, sig, top_n_by_signal("s", 1), frequency="monthly",
        cost_bps=0.0, initial_capital=INITIAL, drift_tolerance=0.0,
    )
    assert res.equity["portfolio_value"].iloc[-1] > INITIAL


# --- weekly vs monthly --------------------------------------------------------

def test_weekly_and_monthly_rebalance_both_run():
    dates = pd.bdate_range("2024-01-02", periods=120)
    prices = pd.DataFrame(
        {"A": 100.0 * (1.002 ** np.arange(120)),
         "B": np.full(120, 100.0)}, index=dates
    )
    sig = pd.DataFrame(
        {"A": np.linspace(0, 1, 120), "B": np.linspace(1, 0, 120)}, index=dates
    )
    weekly = run_backtest(
        prices, sig, top_n_by_signal("s", 1), frequency="weekly",
        cost_bps=0.0, initial_capital=INITIAL,
    )
    monthly = run_backtest(
        prices, sig, top_n_by_signal("s", 1), frequency="monthly",
        cost_bps=0.0, initial_capital=INITIAL,
    )
    assert len(weekly.equity) == len(monthly.equity) == 120
    assert len(weekly.target_weights) >= len(monthly.target_weights)
    assert weekly.equity["portfolio_value"].notna().all()


def test_registry_presets_size_by_max_weight_only():
    """The old top_n presets converted 1:1 (20 stocks -> 5% cap)."""
    from qrp.portfolio.construction import implied_stock_count

    assert not hasattr(get_strategy("trending_value"), "top_n")
    assert implied_stock_count(get_strategy("trending_value").max_weight) == 20
    assert implied_stock_count(get_strategy("value_momentum").max_weight) == 20
    assert implied_stock_count(get_strategy("quality_lowvol").max_weight) == 20
    assert implied_stock_count(get_strategy("momentum").max_weight) == 10
    assert implied_stock_count(get_strategy("trend").max_weight) == 10
    assert implied_stock_count(get_strategy("low_volatility").max_weight) == 10


# --- TTM normalization of flow metrics ---------------------------------------

def _qfact(db, ticker, tag, period_end, filed_date, value, form, fp):
    """One fundamental filing with an explicit form/fp (for TTM tests)."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, ?, ?, ?, ?, ?, 'USD', ?)",
            (ticker, tag, period_end, filed_date, form, fp, value),
        )


def _ttm_db(db):
    """Ticker A: Q1/Q2/Q3/K of FY2023 then Q1 FY2024, all for net_income.

    YTD values: Q1=10, Q2=25, Q3=45, FY=60  -> quarterly increments
    10, 15, 20, 15. Q1'24 YTD=12 -> increment 12.
    """
    init_db(db)
    for t, df in _bars(["A"], {"A": 100.0}, periods=400).items():
        upsert_bars(df, t, db)
    _qfact(db, "A", "net_income", "2023-03-31", "2023-04-28", 10.0, "10-Q", "Q1")
    _qfact(db, "A", "net_income", "2023-06-30", "2023-07-28", 25.0, "10-Q", "Q2")
    _qfact(db, "A", "net_income", "2023-09-30", "2023-10-27", 45.0, "10-Q", "Q3")
    _qfact(db, "A", "net_income", "2023-12-31", "2024-02-15", 60.0, "10-K", "FY")
    _qfact(db, "A", "net_income", "2024-03-31", "2024-04-26", 12.0, "10-Q", "Q1")
    return db


def _ttm_on(db, ticker, day):
    from qrp.signals.fundamental import ttm_matrix

    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    m = ttm_matrix([ticker], "net_income", dates, db)
    return float(m.loc[pd.Timestamp(day), ticker])


def test_ttm_equals_hand_computed_values(tmp_path):
    """Q1=10, Q2 YTD=25, Q3 YTD=45, FY=60 -> increments 10/15/20/15."""
    db = _ttm_db(tmp_path / "t.db")
    # After the 10-K is filed: full FY2023 = 10+15+20+15 = 60.
    assert _ttm_on(db, "A", "2024-02-16") == pytest.approx(60.0)
    # After Q1'24: trailing four quarters Q2'23..Q1'24 = 15+20+15+12 = 62.
    assert _ttm_on(db, "A", "2024-04-26") == pytest.approx(62.0)
    # Before 12 months are covered: NaN, not a partial-year guess.
    assert np.isnan(_ttm_on(db, "A", "2023-05-01"))
    assert np.isnan(_ttm_on(db, "A", "2023-08-01"))


def test_ttm_restatement_does_not_leak(tmp_path):
    """A 10-K/A restatement filed later must not affect earlier dates."""
    db = _ttm_db(tmp_path / "t.db")
    before = _ttm_on(db, "A", "2024-05-01")
    assert before == pytest.approx(62.0)
    # Restate FY2023 annual 60 -> 70, filed 2024-06-01 (after the as-of date).
    _qfact(db, "A", "net_income", "2023-12-31", "2024-06-01", 70.0, "10-K/A", "FY")
    # Earlier date: unchanged — the restatement had not been filed yet.
    assert _ttm_on(db, "A", "2024-05-01") == pytest.approx(before)
    # After the restatement: Q4 increment becomes 70-45=25, TTM = 72.
    assert _ttm_on(db, "A", "2024-06-03") == pytest.approx(72.0)


def test_ttm_annual_only_filer(tmp_path):
    """A company filing only 10-Ks gets its annual value, not 4x annual."""
    db = tmp_path / "t.db"
    init_db(db)
    for t, df in _bars(["C"], {"C": 50.0}, periods=400).items():
        upsert_bars(df, t, db)
    _qfact(db, "C", "net_income", "2022-12-31", "2023-02-01", 100.0, "10-K", "FY")
    _qfact(db, "C", "net_income", "2023-12-31", "2024-02-01", 120.0, "10-K", "FY")
    assert _ttm_on(db, "C", "2024-03-01") == pytest.approx(120.0)
    assert _ttm_on(db, "C", "2023-03-01") == pytest.approx(100.0)


def test_value_composite_ranks_ttm_not_raw_quarters(tmp_path):
    """The earnings component must compare TTM vs TTM.

    A files a huge 10-Q quarter (YTD 100); B's latest filing is a 10-K year
    of 120. Raw-latest ranking would call B richer in earnings; TTM ranking
    (A=170, B=120) correctly calls A richer. Equity and cash flow tie so
    only the earnings component decides.
    """
    from qrp.signals.fundamental import _value_percentiles

    db = tmp_path / "t.db"
    init_db(db)
    for t, df in _bars(["A", "B"], {"A": 100.0, "B": 100.0}, periods=400).items():
        upsert_bars(df, t, db)
    for t in ("A", "B"):
        _qfact(db, t, "shares_outstanding", "2023-12-31", "2024-02-01", 1e6, "10-K", "FY")
        _qfact(db, t, "equity", "2023-12-31", "2024-02-01", 1e9, "10-K", "FY")
        _qfact(db, t, "cashflow_ops", "2023-12-31", "2024-02-01", 5e7, "10-K", "FY")
    # A: full FY2023 quarterly history + a big Q1'24.
    _qfact(db, "A", "net_income", "2023-03-31", "2023-04-28", 10.0, "10-Q", "Q1")
    _qfact(db, "A", "net_income", "2023-06-30", "2023-07-28", 25.0, "10-Q", "Q2")
    _qfact(db, "A", "net_income", "2023-09-30", "2023-10-27", 45.0, "10-Q", "Q3")
    _qfact(db, "A", "net_income", "2023-12-31", "2024-02-15", 80.0, "10-K", "FY")
    _qfact(db, "A", "net_income", "2024-03-31", "2024-04-26", 100.0, "10-Q", "Q1")
    # B: annual-only filer, FY2023 = 120.
    _qfact(db, "B", "net_income", "2023-12-31", "2024-02-01", 120.0, "10-K", "FY")

    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    scores = _value_percentiles(["A", "B"], dates, db)
    day = pd.Timestamp("2024-05-01")
    # TTM check first: A = 15+20+35+100 = 170, B = 120.
    from qrp.signals.fundamental import ttm_matrix

    tm = ttm_matrix(["A", "B"], "net_income", dates, db)
    assert float(tm.loc[day, "A"]) == pytest.approx(170.0)
    assert float(tm.loc[day, "B"]) == pytest.approx(120.0)
    # ...so the composite ranks A (richer TTM earnings) above B.
    assert float(scores.loc[day, "A"]) > float(scores.loc[day, "B"])
