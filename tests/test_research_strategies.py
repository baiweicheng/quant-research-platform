"""Tests for the research batch (era 3): continuous momentum (Frog-in-the-Pan),
overnight-intraday divergence, twin momentum (price + earnings surprise),
and chronological strategy ordering.

All tests use SYNTHETIC data (no network). Signal expectations are
hand-verifiable small cases, not golden values from the implementation.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from qrp.data.edgar import init_db as init_edgar_db
from qrp.data.store import upsert_bars, upsert_dividends
from qrp.signals.research import (
    ContinuousMomentum,
    OvernightIntraday,
    TwinMomentum,
    _session_legs,
    _sue_from_quarters,
)
from qrp.strategies import (
    ERA_LABELS,
    era_label,
    get_strategy,
    list_strategies,
    strategy_target_weights,
)


def _bars_df(closes: np.ndarray, start: str) -> pd.DataFrame:
    """OHLCV DataFrame with DatetimeIndex for Signal.compute().

    Column case matters: compute() takes lowercase open/high/low/close
    (the in-memory bar format); upsert_bars takes the capitalized
    Date/Open/.../Volume database format — see _seed_prices.
    """
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1_000_000,
        },
        index=dates,
    )


def _seed_prices(db, ticker_closes: dict[str, np.ndarray],
                 start: str = "2022-01-03") -> None:
    init_edgar_db(db)  # creates the fundamentals table too
    for t, closes in ticker_closes.items():
        closes = np.asarray(closes, dtype=float)
        dates = pd.bdate_range(start, periods=len(closes))
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": closes,
                    "High": closes * 1.01,
                    "Low": closes * 0.99,
                    "Close": closes,
                    "Volume": 1_000_000,
                }
            ),
            t,
            db,
        )


def _eps_fact(db, ticker, period_end, filed_date, value, fp, form="10-Q"):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, 'eps_diluted', ?, ?, ?, ?, 'USD', ?)",
            (ticker, period_end, filed_date, form, fp, value),
        )


def _quarters_2021_2023(values):
    """(period_end, filed_date, value, fp) for 2021Q1..2023Q3 (9 quarters)."""
    out = []
    i = 0
    for year in (2021, 2022, 2023):
        for q, (me, mf) in (("Q1", (3, 5)), ("Q2", (6, 8)), ("Q3", (9, 11))):
            out.append((f"{year}-{me:02d}-30", f"{year}-{mf:02d}-15",
                        values[i], q))
            i += 1
    return out


def _seed_eps(db, ticker, quarterly_values):
    """Store 10-Q diluted-EPS facts for 2021Q1..2023Q3.

    Stored 10-Q income-statement values are year-to-date (the platform's
    convention for flow tags — see qrp.signals.fundamental), so quarterly
    inputs are cumulated within each fiscal year before storing.
    """
    assert len(quarterly_values) == 9
    ytd = []
    for yr in range(3):
        run = 0.0
        for q in range(3):
            run += quarterly_values[yr * 3 + q]
            ytd.append(run)
    for pe, fd, v, fp in _quarters_2021_2023(ytd):
        _eps_fact(db, ticker, pe, fd, v, fp)


# --- Strategy A: continuous momentum ----------------------------------------

def test_smooth_winner_outranks_choppy_winner():
    """Same ~12% formation gain; the smooth climber must rank first."""
    n = 400
    s = 100.0 * (1.0005 ** np.arange(n))          # steady +0.05%/day
    c = np.full(n, 100.0)
    for j in (200, 250, 300):                     # three +3.92% jumps
        c[j:] *= 1.0392                          # ~= same 12.2% total
    sig = ContinuousMomentum()
    last_s = sig.compute(_bars_df(s, "2022-01-03")).iloc[-1]
    last_c = sig.compute(_bars_df(c, "2022-01-03")).iloc[-1]
    assert pd.notna(last_s) and pd.notna(last_c)  # both are winners
    # ID_S ~= -1 -> score ~= +1; ID_C ~= -0.03 -> score ~= +0.03.
    assert last_s == pytest.approx(1.0, abs=0.01)
    assert last_s > last_c


def test_continuous_momentum_loser_gets_no_score():
    n = 400
    up = 100.0 * (1.0005 ** np.arange(n))
    down = 100.0 * (0.9995 ** np.arange(n))
    sig = ContinuousMomentum()
    vu = sig.compute(_bars_df(up, "2022-01-03"))
    vd = sig.compute(_bars_df(down, "2022-01-03"))
    assert pd.notna(vu.iloc[-1])   # winner keeps its score
    assert pd.isna(vd.iloc[-1])    # loser gets no score — never a candidate
    assert vu.iloc[:252].isna().all()  # too little history stays NaN


# --- Strategy B: overnight-intraday divergence -------------------------------

def _seed_flow_db(db, n=200):
    """F: +0.1%/day intraday, flat overnight. R: the exact reverse."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    init_edgar_db(db)
    px = 100.0
    f_o, f_c = [], []
    for _ in range(n):
        f_o.append(px)
        px *= 1.001
        f_c.append(px)
    px = 100.0
    r_o, r_c = [], []
    for _ in range(n):
        r_o.append(px * 1.001)
        px *= 1.001
        r_c.append(px)
    for t, o, c in (("F", f_o, f_c), ("R", r_o, r_c)):
        upsert_bars(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Open": o,
                    "High": np.maximum(o, c) * 1.005,
                    "Low": np.minimum(o, c) * 0.995,
                    "Close": c,
                    "Volume": 1_000_000,
                }
            ),
            t,
            db,
        )


def test_intraday_strong_outranks_overnight_strong(tmp_path):
    db = str(tmp_path / "flow.db")
    _seed_flow_db(db)
    mat = OvernightIntraday().compute_all(["F", "R"], db)
    last = mat.iloc[-1]
    assert pd.notna(last["F"]) and pd.notna(last["R"])
    # F: positive intraday Z, negative overnight Z -> positive score;
    # R is the mirror image -> negative score.
    assert last["F"] > 0 > last["R"]


def test_ex_date_dividend_does_not_corrupt_overnight():
    """The dividend is added back into the overnight leg.

    Raw prices print the open $1 lower on the ex-date; without the
    add-back that reads as a -1% overnight return. With it, the leg is 0.
    """
    dates = pd.bdate_range("2023-07-28", periods=5)
    opens = pd.DataFrame({"X": [100.0, 100.0, 99.0, 100.0, 100.0]},
                         index=dates)
    closes = pd.DataFrame({"X": [100.0] * 5}, index=dates)
    divs = pd.DataFrame({"X": [0.0, 0.0, 1.0, 0.0, 0.0]}, index=dates)
    on, _ = _session_legs(opens, closes, divs)
    assert on["X"].iloc[2] == pytest.approx(0.0, abs=1e-12)
    # Without the dividend the same bars read as a -1% overnight drop.
    on_raw, _ = _session_legs(opens, closes, divs * 0.0)
    assert on_raw["X"].iloc[2] == pytest.approx(-0.01)


def test_overnight_signal_moves_with_dividend_table(tmp_path):
    """End-to-end: the stored dividend actually changes the signal.

    Same dip bars in two databases; only one stores the $1 dividend.
    The compounded signals must differ (the unadjusted one bakes in a
    fake -1% overnight). Three tickers so Z-scores carry magnitudes.
    """
    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)
    ex = dates[150].strftime("%Y-%m-%d")
    base = 100.0 * (1.0002 ** np.arange(n))
    y = 50.0 * (1.0001 ** np.arange(n))
    z = 75.0 * (1.0003 ** np.arange(n))
    dip = base.copy()
    dip[150] = base[149] - 1.0
    tickers = ["X", "Y", "Z"]

    def build(db, with_dividend):
        init_edgar_db(db)
        specs = (("X", dip, base), ("Y", y, y * 1.0005), ("Z", z, z * 0.9998))
        for t, opens, closes in specs:
            upsert_bars(
                pd.DataFrame(
                    {
                        "Date": dates.strftime("%Y-%m-%d"),
                        "Open": opens,
                        "High": np.maximum(opens, closes) * 1.005,
                        "Low": np.minimum(opens, closes) * 0.995,
                        "Close": closes,
                        "Volume": 1_000_000,
                    }
                ),
                t,
                db,
            )
        if with_dividend:
            upsert_dividends(
                pd.DataFrame({"ex_date": [ex], "amount": [1.0]}), "X", db
            )

    db_a, db_c = str(tmp_path / "a.db"), str(tmp_path / "c.db")
    build(db_a, with_dividend=True)
    build(db_c, with_dividend=False)
    sig = OvernightIntraday()
    a = sig.compute_all(tickers, db_a)["X"].dropna()
    c = sig.compute_all(tickers, db_c)["X"].dropna()
    common = a.index.intersection(c.index)
    assert len(common) > 0
    assert not np.allclose(a[common], c[common], rtol=1e-6, atol=1e-6)


# --- Strategy C: twin momentum (SUE) -----------------------------------------

def test_twin_momentum_requires_both_legs(tmp_path):
    """Great price + great earnings qualifies; great price + awful
    earnings is disqualified (NaN) even though its price leg is best."""
    db = str(tmp_path / "twin.db")
    n = 400
    g = 100.0 * (1.0018 ** np.arange(n))   # strong rise
    h = 100.0 * (1.0020 ** np.arange(n))   # even stronger rise
    m = np.full(n, 100.0)                   # flat (price-leg anchor)
    _seed_prices(db, {"G": g, "H": h, "M": m}, start="2022-10-03")
    _seed_eps(db, "G",
              [1.00, 1.03, 1.02, 1.07, 1.09, 1.08, 1.13, 1.16, 1.32])
    _seed_eps(db, "H",
              [1.32, 1.29, 1.30, 1.25, 1.22, 1.23, 1.18, 1.14, 0.88])
    _seed_eps(db, "M", [1.00] * 9)
    mat = TwinMomentum().compute_all(["G", "H", "M"], db)
    last = mat.iloc[-1]
    assert pd.notna(last["G"]), "G should qualify on both legs"
    assert pd.isna(last["H"]), "H must be disqualified by its earnings leg"
    assert mat["H"].isna().all()


def test_sue_needs_eight_quarters():
    quarters = {
        pd.Timestamp(f"2022-{m:02d}-28"): (1.00 + 0.02 * i, q)
        for i, (m, q) in enumerate(
            [(3, 1), (6, 2), (9, 3), (3, 1), (6, 2), (9, 3), (3, 1)]
        )
    }
    assert pd.isna(_sue_from_quarters(quarters))


def test_sue_is_point_in_time(tmp_path):
    """A restatement filed AFTER the signal date must not change it —
    but it must move later dates."""
    db = str(tmp_path / "pit.db")
    n = 400
    g = 100.0 * (1.0018 ** np.arange(n))
    h = 100.0 * (1.0020 ** np.arange(n))
    m = np.full(n, 100.0)
    _seed_prices(db, {"G": g, "H": h, "M": m}, start="2022-10-03")
    _seed_eps(db, "G",
              [1.00, 1.03, 1.02, 1.07, 1.09, 1.08, 1.13, 1.16, 1.32])
    _seed_eps(db, "H",
              [1.32, 1.29, 1.30, 1.25, 1.22, 1.23, 1.18, 1.14, 0.88])
    _seed_eps(db, "M", [1.00] * 9)
    mat1 = TwinMomentum().compute_all(["G", "H", "M"], db)
    # 2023-11-01: latest known filing is 2023Q2 (filed 2023-08-15) —
    # the 2023Q3 10-Q (filed 2023-11-15) is still in the future.
    before = mat1.loc[:"2023-11-01", "G"].iloc[-1]
    assert pd.notna(before)
    # Restate 2023Q3's EPS down to 0.10, filed 2023-12-20 (after as-of).
    _eps_fact(db, "G", "2023-09-30", "2023-12-20", 0.10, "Q3", form="10-Q/A")
    mat2 = TwinMomentum().compute_all(["G", "H", "M"], db)
    after = mat2.loc[:"2023-11-01", "G"].iloc[-1]
    assert before == pytest.approx(after), "restatement leaked into the past"
    # ...but the restatement is real information after its filing date:
    # G's collapsed surprise now disqualifies it where it used to qualify.
    assert pd.notna(mat1.loc["2024-01-02", "G"])
    assert pd.isna(mat2.loc["2024-01-02", "G"])


# --- Chronological ordering ---------------------------------------------------

def test_strategies_listed_chronologically():
    names = [n for n, _ in list_strategies()]
    assert names == [
        "momentum", "trend", "low_volatility",                    # era 1
        "trending_value", "value_momentum", "quality_lowvol",      # era 2
        "continuous_momentum", "overnight_intraday", "twin_momentum",  # era 3
    ]


def test_era_labels():
    assert era_label("momentum") == ERA_LABELS[1]
    assert ERA_LABELS[1] == "Original technical strategies"
    assert era_label("value_momentum") == ERA_LABELS[2]
    assert ERA_LABELS[2] == "Fundamental strategies"
    assert era_label("twin_momentum") == ERA_LABELS[3]
    assert ERA_LABELS[3] == "New research strategies"
    assert era_label("nope") == ""


def test_new_strategy_presets():
    a = get_strategy("continuous_momentum")
    assert a.formal_name == "Continuous Momentum (Frog-in-the-Pan)"
    assert (a.rebalance, a.weighting, a.max_weight) == (
        "monthly", "inverse_vol", 0.02)
    b = get_strategy("overnight_intraday")
    assert b.formal_name == "Intraday Institutional Flow (Overnight Divergence)"
    assert (b.rebalance, b.weighting, b.max_weight) == ("monthly", "equal", 0.02)
    c = get_strategy("twin_momentum")
    assert c.formal_name == "Twin Momentum (Price + Earnings Surprise)"
    assert (c.rebalance, c.weighting, c.max_weight) == ("monthly", "equal", 0.025)
    for key in ("continuous_momentum", "overnight_intraday", "twin_momentum"):
        s = get_strategy(key)
        assert s.era == 3
        assert "1." in s.calculation and "2." in s.calculation
        assert s.description.strip() and s.good_at.strip()
        assert s.main_risk.strip()


def test_continuous_momentum_target_weights():
    """Strategy-level smoke: weights respect the cap and sum <= 1."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"S{i}" for i in range(60)]
    sig = pd.DataFrame(
        np.tile(np.linspace(0.1, 1.0, 60), (10, 1)),
        index=dates, columns=tickers,
    )
    vols = pd.DataFrame(
        np.tile(np.linspace(0.01, 0.05, 60), (10, 1)),
        index=dates, columns=tickers,
    )
    matrices = {"continuous_momentum": sig, "__vol__": vols}
    weights, _ = strategy_target_weights(
        get_strategy("continuous_momentum"), matrices, dates[-1]
    )
    assert weights
    assert sum(weights.values()) <= 1.0 + 1e-9
    assert max(weights.values()) <= 0.02 + 1e-9
    assert len(weights) <= 50


def test_continuous_momentum_bear_market_holds_only_winners():
    """Bear market: 20 winners, 280 losers (NaN scores). The strategy must
    hold exactly the 20 winners — never pad the ~50 slots with losers —
    and the unfilled slice sits in cash."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"S{i}" for i in range(300)]
    sig = pd.DataFrame(np.nan, index=dates, columns=tickers)
    # 20 winners with real scores; the other 280 are NaN (non-winners).
    sig.iloc[:, :20] = np.tile(np.linspace(0.1, 1.0, 20), (10, 1))
    vols = pd.DataFrame(0.02, index=dates, columns=tickers)
    matrices = {"continuous_momentum": sig, "__vol__": vols}
    weights, scores = strategy_target_weights(
        get_strategy("continuous_momentum"), matrices, dates[-1]
    )
    # Exactly the 20 winners, each at the 2% cap; nothing else.
    assert set(weights) == set(tickers[:20])
    assert sum(weights.values()) == pytest.approx(0.4)
    # The unfilled 60% sits in cash — no loser sneaks in via scores either.
    assert set(scores) == set(tickers[:20])
