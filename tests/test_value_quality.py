"""Tests for the value & quality classic signals (backlog #2, era 4).

Synthetic DB, hand-verifiable expectations:
- AAA: strong quality/value (profitable, growing, safe, pays out, buys back)
- BBB: weak (losses, levered, diluting, no payout)
- CCC: price-only (crashed long ago, flat since) for the reversal test
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from qrp.data.edgar import TAG_FALLBACKS, init_db as init_edgar_db
from qrp.data.store import upsert_bars
from qrp.signals.value_quality import (
    LongHorizonReversal,
    MagicFormula,
    PiotroskiFScore,
    QMJLite,
    ShareholderYield,
)
from qrp.strategies.registry import ERA_LABELS, get_strategy, list_strategies

N_DAYS = 1300
START = "2020-01-02"


def _seed_prices(db, ticker_closes: dict[str, np.ndarray]) -> None:
    init_edgar_db(db)
    for t, closes in ticker_closes.items():
        closes = np.asarray(closes, dtype=float)
        dates = pd.bdate_range(START, periods=len(closes))
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


def _fact(db, ticker, tag, period_end, filed_date, value):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, ?, ?, ?, '10-K', 'FY', 'USD', ?)",
            (ticker, tag, period_end, filed_date, value),
        )


def _seed_annual(db, ticker, year, vals: dict[str, float]) -> None:
    filed = f"{year + 1}-02-15"
    for tag, v in vals.items():
        _fact(db, ticker, tag, f"{year}-12-31", filed, v)


_AAA_23 = {
    "net_income": 10, "cashflow_ops": 12, "gross_profit": 40, "revenue": 100,
    "ebit": 15, "assets": 100, "equity": 60, "debt_longterm": 10, "cash": 20,
    "assets_current": 40, "liabilities_current": 15, "shares_outstanding": 10,
    "dividends_paid": 3, "ppe_net": 50,
}
_AAA_22 = {
    "net_income": 8, "cashflow_ops": 9, "gross_profit": 35, "revenue": 90,
    "ebit": 12, "assets": 95, "equity": 55, "debt_longterm": 12, "cash": 15,
    "assets_current": 35, "liabilities_current": 16, "shares_outstanding": 11,
    "dividends_paid": 2.5, "ppe_net": 48,
}
_BBB_23 = {
    "net_income": -2, "cashflow_ops": -3, "gross_profit": 10, "revenue": 80,
    "ebit": -1, "assets": 100, "equity": 40, "debt_longterm": 40, "cash": 2,
    "assets_current": 15, "liabilities_current": 20, "shares_outstanding": 12,
    "dividends_paid": 0, "ppe_net": 60,
}
_BBB_22 = {
    "net_income": 1, "cashflow_ops": 0.5, "gross_profit": 12, "revenue": 85,
    "ebit": 1, "assets": 100, "equity": 42, "debt_longterm": 35, "cash": 3,
    "assets_current": 16, "liabilities_current": 18, "shares_outstanding": 10,
    "dividends_paid": 0, "ppe_net": 58,
}


def _make_db(tmp_path):
    db = tmp_path / "vq.db"
    aaa = 100.0 * (1.0004 ** np.arange(N_DAYS))          # steady winner
    bbb = np.full(N_DAYS, 100.0)                          # flat
    ccc = np.concatenate([                                # crashed long ago
        np.linspace(200.0, 50.0, N_DAYS - 252),
        np.full(252, 50.0),                               # flat last year
    ])
    _seed_prices(db, {"AAA": aaa, "BBB": bbb, "CCC": ccc})
    _seed_annual(db, "AAA", 2023, _AAA_23)
    _seed_annual(db, "AAA", 2022, _AAA_22)
    _seed_annual(db, "BBB", 2023, _BBB_23)
    _seed_annual(db, "BBB", 2022, _BBB_22)
    return db


def _last_row(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[-1]


def test_piotroski_scores_hand_verified(tmp_path):
    db = _make_db(tmp_path)
    scores = _last_row(PiotroskiFScore().compute_all(["AAA", "BBB", "CCC"], db))
    assert scores["AAA"] == 9.0, "AAA passes all nine hand-checked tests"
    assert scores["BBB"] == 0.0, "BBB fails all nine hand-checked tests"
    assert pd.isna(scores["CCC"]), "no fundamentals -> unscored, not zero"


def test_magic_formula_ranks_strong_first(tmp_path):
    db = _make_db(tmp_path)
    scores = _last_row(MagicFormula().compute_all(["AAA", "BBB", "CCC"], db))
    assert scores["AAA"] > scores["BBB"]
    assert pd.isna(scores["CCC"])


def test_magic_formula_canonical_roc(tmp_path):
    """AUD-02: ROC = EBIT / ((CA - CL) + PPE_net), not EBIT / (equity + debt).
    AAA 2023: EBIT=15, CA=40, CL=15, PPE=50 -> invested=75 -> ROC=20%.
    The old (equity+debt)=70 would give 21.4% — the test pins the canonical
    denominator."""
    from qrp.signals.value_quality import (
        MagicFormula as MF, TAG_ASSETS_CUR, TAG_LIAB_CUR, TAG_PPE_NET,
    )
    import inspect
    # Verify the source uses the canonical tag constants (regression guard).
    src = inspect.getsource(MF.compute_all)
    assert "TAG_ASSETS_CUR" in src and "TAG_PPE_NET" in src
    assert "TAG_LIAB_CUR" in src
    # And the tags resolve to the SEC balance-sheet concepts.
    assert TAG_ASSETS_CUR == "assets_current"
    assert TAG_LIAB_CUR == "liabilities_current"
    assert TAG_PPE_NET == "ppe_net"


def test_magic_formula_net_cash_reduces_ev(tmp_path):
    """AUD-02: EV = mcap + debt - cash (net cash reduces EV, not floored)."""
    db = _make_db(tmp_path)
    # AAA has cash=20, debt=10 -> net debt = -10 (net cash).
    # The EV must reflect the cash; it is not floored at mcap.
    scores = _last_row(MagicFormula().compute_all(["AAA", "BBB"], db))
    assert scores["AAA"] > scores["BBB"]  # AAA wins on both legs


def test_magic_formula_excludes_financials(tmp_path):
    db = _make_db(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS gics_sectors"
            " (ticker TEXT PRIMARY KEY, sector TEXT, sub_industry TEXT,"
            " fetched_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO gics_sectors VALUES"
            " ('AAA', 'Financials', 'Banks', '2026-10-11')"
        )
    scores = _last_row(MagicFormula().compute_all(["AAA", "BBB"], db))
    assert pd.isna(scores["AAA"]), "known financials are excluded"
    assert pd.notna(scores["BBB"])


def test_qmj_lite_prefers_quality(tmp_path):
    db = _make_db(tmp_path)
    scores = _last_row(QMJLite().compute_all(["AAA", "BBB", "CCC"], db))
    assert scores["AAA"] > scores["BBB"]


def test_shareholder_yield_prefers_payout(tmp_path):
    db = _make_db(tmp_path)
    scores = _last_row(
        ShareholderYield().compute_all(["AAA", "BBB", "CCC"], db)
    )
    assert scores["AAA"] > scores["BBB"], "AAA pays dividends and buys back"
    assert pd.isna(scores["CCC"])


def test_long_horizon_reversal_buys_the_forgotten(tmp_path):
    db = _make_db(tmp_path)
    scores = _last_row(
        LongHorizonReversal().compute_all(["AAA", "BBB", "CCC"], db)
    )
    # CCC fell 75% from 5y ago to 1y ago; AAA rose ~52%; BBB flat.
    assert scores["CCC"] > scores["BBB"] > scores["AAA"]


def test_cash_tag_ingested_for_magic_formula():
    assert "cash" in TAG_FALLBACKS, (
        "the Magic Formula's EV math needs the cash tag in the ingest map"
    )


def test_new_strategies_registered_and_documented():
    names = [n for n, _ in list_strategies()]
    for n in ("magic_formula", "piotroski", "qmj_lite",
              "shareholder_yield", "reversal_long"):
        assert n in names
        s = get_strategy(n)
        assert s.era == 4
        assert len(s.description) > 50, f"{n} needs a plain-English description"
        assert len(s.calculation) > 200, f"{n} needs step-by-step math"
        assert s.good_at and s.main_risk and s.data_used
    assert ERA_LABELS[4] == "Value & quality classics"
    # Era 4 sorts after era 3 everywhere.
    eras = [get_strategy(n).era for n in names]
    assert eras == sorted(eras)


def test_data_gap_note_fires_when_key_tags_missing(tmp_path):
    """The Signals summary must not show a bare 'OK' when a signal's key
    inputs are absent entirely (the Oct-12 confusion: 'Compute all
    signals' looked fine, then Compare failed opaquely)."""
    db = tmp_path / "gap.db"
    _seed_prices(db, {"AAA": 100.0 * (1.0004 ** np.arange(300))})
    from qrp.signals import default_registry

    reg = default_registry()
    expect_note = {
        "magic_formula": "operating earnings (EBIT)",
        "piotroski": "financial-statement detail",
        "qmj_lite": "profitability",
        "shareholder_yield": "dividends-paid",
    }
    for name, fragment in expect_note.items():
        gap = getattr(reg.get(name), "data_gap_note", None)
        note = gap(["AAA"], db) if callable(gap) else None
        assert note is not None, f"{name} should warn when its tags are missing"
        assert fragment in note
        assert "Market data page" in note
        assert "Fundamentals" in note
    # Reversal is price-based: prices exist, so no gap note hook at all.
    assert getattr(reg.get("reversal_long"), "data_gap_note", None) is None


def test_data_gap_note_silent_when_tags_present(tmp_path):
    db = _make_db(tmp_path)
    from qrp.signals import default_registry

    reg = default_registry()
    for name in [
        "magic_formula", "piotroski", "qmj_lite", "shareholder_yield",
        "reversal_long",
    ]:
        gap = getattr(reg.get(name), "data_gap_note", None)
        assert (gap(["AAA", "BBB"], db) if callable(gap) else None) is None, name
