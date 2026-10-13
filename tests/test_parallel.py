"""Tests for backlog #12: parallel signal computation and strategy compare.

The contract: parallel runs must produce results identical to the
sequential runs — same stored rows, same comparison rows, same equity
curves. Speed is a bonus; correctness is the requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import qrp.ui.helpers as H
from qrp.data.edgar import init_db as init_edgar_db
from qrp.data.store import upsert_bars
from qrp.signals import default_registry


def _seed_prices(db, n_days: int = 400) -> list[str]:
    init_edgar_db(db)
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rng = np.random.default_rng(42)
    for t in tickers:
        rets = rng.normal(0.0005, 0.012, n_days)
        closes = 100.0 * (1 + rets).cumprod()
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
    return tickers


def _summary_key(summary: list[dict]) -> list[tuple]:
    return [
        (r["Signal"], r["Tickers OK"], r["Rows stored"], r["Status"])
        for r in summary
    ]


def test_parallel_signals_match_sequential(tmp_path):
    db = str(tmp_path / "par.db")
    tickers = _seed_prices(db)
    seq = H.compute_all_signals(tickers, db, parallel=False)
    par = H.compute_all_signals(tickers, db, parallel=True)
    assert _summary_key(par) == _summary_key(seq), (
        "parallel compute-all must match the sequential run exactly"
    )


def test_parallel_signals_progress_callback(tmp_path):
    db = str(tmp_path / "par2.db")
    tickers = _seed_prices(db, n_days=300)
    seen: list[tuple[int, int]] = []
    H.compute_all_signals(
        tickers, db, parallel=True,
        progress=lambda done, total, label: seen.append((done, total)),
    )
    n = len(list(default_registry().list_signals()))
    assert seen, "progress must fire in parallel mode"
    assert seen[-1] == (n, n), "final call signals completion"


def test_parallel_compare_matches_sequential(tmp_path, monkeypatch):
    db = str(tmp_path / "cmp.db")
    tickers = _seed_prices(db)
    # Populate signal tables first (sequential — the compare reads them).
    H.compute_all_signals(tickers, db, parallel=False)
    # This scratch DB has prices but no company fundamentals, so restrict
    # the race to strategies whose signals actually have stored values
    # here (momentum, trend, low_volatility, continuous_momentum).
    price_only = {
        "momentum", "trend", "low_volatility", "continuous_momentum",
    }
    real_list = H.list_strategies
    monkeypatch.setattr(
        H, "list_strategies",
        lambda: [(n, d) for n, d in real_list() if n in price_only],
    )
    kwargs = dict(
        tickers=tuple(tickers),
        start="2023-06-01",
        end="2024-01-31",
        capital=100_000.0,
        max_weight=0.5,
        cost_bps=0.0,
        db=db,
        mtime=0.0,
        frequency="monthly",
        cost_model="spread",
    )
    rows_seq, eq_seq = H._cached_compare_full(**kwargs, parallel=False)
    rows_par, eq_par = H._cached_compare_full(**kwargs, parallel=True)
    assert rows_par == rows_seq, "comparison rows must be identical"
    assert set(eq_par) == set(eq_seq)
    for key in eq_seq:
        pd.testing.assert_frame_equal(eq_par[key], eq_seq[key])


def test_worker_count_caps_and_floors():
    assert H._worker_count(1) == 1
    assert H._worker_count(100) <= 4
    assert H._worker_count(2) <= 2


def test_compare_failure_note_names_missing_signal():
    note = H._compare_failure_note(
        ValueError(
            "No stored values for signal 'magic_formula' in range — "
            "compute signals first (Signals page)."
        )
    )
    assert "Magic Formula" in note, "names the signal in plain English"
    assert "Compute all signals" in note, "tells the user the one-step fix"
    assert "Signals page" in note


def test_compare_failure_note_generic_error():
    note = H._compare_failure_note(RuntimeError("boom"))
    assert "unexpected error" in note
    assert "RuntimeError" in note
    assert "boom" in note


def test_parallel_compare_degraded_row_for_missing_signal(tmp_path, monkeypatch):
    """A strategy whose signal was never computed degrades to an
    explanatory failed row — and the row keeps the numeric keys the
    results page needs so the page cannot KeyError (the Oct-12 bug)."""
    import math

    db = str(tmp_path / "cmpfail.db")
    tickers = _seed_prices(db)
    H.compute_all_signals(tickers, db, parallel=False)
    # magic_formula's signal was NOT computed above (its signal needs
    # fundamentals, absent from this scratch DB) — its worker must die
    # gracefully like it did on the Oct-12 report.
    price_only = {
        "momentum", "trend", "low_volatility", "continuous_momentum",
        "magic_formula",
    }
    real_list = H.list_strategies
    monkeypatch.setattr(
        H, "list_strategies",
        lambda: [(n, d) for n, d in real_list() if n in price_only],
    )
    kwargs = dict(
        tickers=tuple(tickers),
        start="2023-06-01",
        end="2024-01-31",
        capital=100_000.0,
        max_weight=0.5,
        cost_bps=0.0,
        db=db,
        mtime=0.0,
        frequency="monthly",
        cost_model="spread",
    )
    rows, _eq = H._cached_compare_full(**kwargs, parallel=True)
    failed = [r for r in rows if r["Total return"] == "worker failed"]
    assert len(failed) == 1, f"only magic_formula should fail, got {rows}"
    row = failed[0]
    assert row["Strategy"] == "Magic Formula (Greenblatt)"
    assert math.isnan(row["total_return"]), "NaN keeps chart code crash-free"
    assert math.isnan(row["sharpe"])
    assert "Compute all signals" in row["failure_reason"]
    ok_rows = [r for r in rows if r["Total return"] != "worker failed"]
    assert len(ok_rows) == 4
    for r in ok_rows:
        assert isinstance(r["total_return"], float)
        assert "failure_reason" not in r
