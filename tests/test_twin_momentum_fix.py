"""Twin Momentum missing-data fixes.

Regression tests for the October 2026 bug where computing the
twin_momentum signal on a database without the (then new) eps_diluted
tag crashed with KeyError: 'form' and the Signals page reported
"Failed — saved data unchanged".

Covers:
(a) the original crash: tag-less DB must not raise (failed before fix);
(b) partial tag coverage: tickers without the tag don't crash, tickers
    with it are scored normally;
(c) the incremental fundamentals download backfills a newly-added tag
    (the reason the user's re-download didn't help);
(d) the plain-English "no earnings data" note appears in the
    compute-all summary instead of a bare "OK / 0 rows".
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from qrp.data.edgar import (
    _incremental_new_facts,
    init_db as init_edgar_db,
    set_last_fetch,
)
from qrp.data.store import upsert_bars
from qrp.signals.research import (
    TAG_EPS_DILUTED,
    TwinMomentum,
    _quarterly_eps_events,
)
from qrp.ui.helpers import (
    compute_all_signals,
    compute_signal_rows,
    signal_data_note,
)


def _bars_df(closes: np.ndarray, start: str) -> pd.DataFrame:
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


def _seed_prices(db, ticker_closes: dict, start: str = "2022-01-03") -> None:
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
            out.append(
                (f"{year}-{me:02d}-30", f"{year}-{mf:02d}-15", values[i], q)
            )
            i += 1
    return out


def _seed_eps(db, ticker, quarterly_values):
    """Store 10-Q diluted-EPS facts for 2021Q1..2023Q3 (YTD convention)."""
    assert len(quarterly_values) == 9
    ytd = []
    for yr in range(3):
        run = 0.0
        for q in range(3):
            run += quarterly_values[yr * 3 + q]
            ytd.append(run)
    for pe, fd, v, fp in _quarters_2021_2023(ytd):
        _eps_fact(db, ticker, pe, fd, v, fp)


def _trend(n, drift=0.0018, start=100.0):
    return start * (1.0 + drift) ** np.arange(n)


# --- (a) the original crash -------------------------------------------------

def test_quarterly_eps_events_tolerates_columnless_frame():
    """The exact KeyError: 'form' from the bug report."""
    events = _quarterly_eps_events(pd.DataFrame())
    assert events == []


def test_quarterly_eps_events_tolerates_missing_columns():
    events = _quarterly_eps_events(pd.DataFrame({"a": [1]}))
    assert events == []


def test_twin_momentum_no_eps_tag_no_crash(tmp_path):
    """Tag-less DB: no exception, all-NaN signal, 0 rows stored."""
    db = str(tmp_path / "notag.db")
    _seed_prices(db, {"AAA": _trend(300)})
    mat = TwinMomentum().compute_all(["AAA"], db)
    assert mat.shape == (300, 1)
    assert mat.isna().all().all()
    rows = compute_signal_rows("twin_momentum", ["AAA"], db)
    assert rows[0]["Status"] == "OK"
    assert rows[0]["Values stored"] == 0


# --- (b) partial coverage ---------------------------------------------------

def test_twin_momentum_partial_tag_coverage(tmp_path):
    """Tickers without the tag don't crash; covered tickers score."""
    db = str(tmp_path / "partial.db")
    n = 400
    g = 100.0 * (1.0018 ** np.arange(n))  # strong rise, rising EPS
    h = 100.0 * (1.0020 ** np.arange(n))  # stronger rise, falling EPS
    m = np.full(n, 100.0)  # flat, and NO eps_diluted facts at all
    _seed_prices(db, {"G": g, "H": h, "M": m}, start="2022-10-03")
    _seed_eps(
        db, "G", [1.00, 1.03, 1.02, 1.07, 1.09, 1.08, 1.13, 1.16, 1.32]
    )
    _seed_eps(
        db, "H", [1.32, 1.29, 1.30, 1.25, 1.22, 1.23, 1.18, 1.14, 0.88]
    )
    mat = TwinMomentum().compute_all(["G", "H", "M"], db)
    assert mat.shape[1] == 3
    assert pd.notna(mat["G"].iloc[-1]), "covered ticker should score"
    assert mat["M"].isna().all(), "tag-less ticker stays NaN"
    rows = compute_signal_rows("twin_momentum", ["G", "H", "M"], db)
    assert all(r["Status"] == "OK" for r in rows)
    # No data-gap note: some tickers DO have the tag.
    assert signal_data_note("twin_momentum", ["G", "H", "M"], db) is None


# --- (c) incremental backfill of a newly-added tag --------------------------

def _fundamentals_fact(db, ticker, tag, period_end, filed_date, value):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals"
            " (ticker, tag, period_end, filed_date, form, fp, unit, value)"
            " VALUES (?, ?, ?, ?, '10-Q', 'Q3', 'USD', ?)",
            (ticker, tag, period_end, filed_date, value),
        )


def test_incremental_backfills_never_fetched_tag(tmp_path):
    """A ticker downloaded before eps_diluted existed gets the tag's full
    history on the next incremental run — this is what the user's
    re-download was missing."""
    db = str(tmp_path / "backfill.db")
    init_edgar_db(db)
    _fundamentals_fact(db, "AAA", "net_income", "2023-09-30", "2023-11-01", 5.0)
    set_last_fetch("AAA", db)  # last download = now
    extracted = pd.DataFrame(
        [
            {
                "tag": "net_income",
                "period_end": "2023-09-30",
                "filed_date": "2023-11-01",
                "form": "10-Q",
                "fp": "Q3",
                "unit": "USD",
                "value": 5.0,
            },
            {
                "tag": TAG_EPS_DILUTED,
                "period_end": "2023-09-30",
                "filed_date": "2023-11-01",
                "form": "10-Q",
                "fp": "Q3",
                "unit": "USD",
                "value": 1.2,
            },
            {
                "tag": TAG_EPS_DILUTED,
                "period_end": "2023-06-30",
                "filed_date": "2023-08-01",
                "form": "10-Q",
                "fp": "Q2",
                "unit": "USD",
                "value": 1.1,
            },
            {
                "tag": "net_income",
                "period_end": "2023-12-31",
                "filed_date": "2099-01-01",
                "form": "10-K",
                "fp": "FY",
                "unit": "USD",
                "value": 22.0,
            },
        ]
    )
    out = _incremental_new_facts(extracted, "AAA", db)
    eps = out[out["tag"] == TAG_EPS_DILUTED]
    assert len(eps) == 2, "eps_diluted history must be backfilled in full"
    ni = out[out["tag"] == "net_income"]
    assert len(ni) == 1, "already-seen filings stay filtered"
    assert ni.iloc[0]["filed_date"] == "2099-01-01"


def test_incremental_unchanged_when_tag_already_present(tmp_path):
    """Normal incremental behavior is untouched for tags already stored."""
    db = str(tmp_path / "incr.db")
    init_edgar_db(db)
    _fundamentals_fact(db, "AAA", "net_income", "2023-09-30", "2023-11-01", 5.0)
    _fundamentals_fact(
        db, "AAA", TAG_EPS_DILUTED, "2023-09-30", "2023-11-01", 1.2
    )
    set_last_fetch("AAA", db)
    extracted = pd.DataFrame(
        [
            {
                "tag": TAG_EPS_DILUTED,
                "period_end": "2023-09-30",
                "filed_date": "2023-11-01",
                "form": "10-Q",
                "fp": "Q3",
                "unit": "USD",
                "value": 1.2,
            },
        ]
    )
    out = _incremental_new_facts(extracted, "AAA", db)
    assert out.empty, "nothing new: old filings still filtered"


# --- (d) the plain-English note ---------------------------------------------

def test_data_gap_note_when_tag_entirely_absent(tmp_path):
    db = str(tmp_path / "gap.db")
    _seed_prices(db, {"AAA": _trend(300)})
    note = signal_data_note("twin_momentum", ["AAA"], db)
    assert note is not None
    assert "No earnings data yet" in note
    assert "fundamentals" in note.lower()


def test_no_data_gap_note_when_tag_present(tmp_path):
    db = str(tmp_path / "nogap.db")
    _seed_prices(db, {"AAA": _trend(300)})
    _eps_fact(db, "AAA", "2023-09-30", "2023-11-15", 1.2, "Q3")
    assert signal_data_note("twin_momentum", ["AAA"], db) is None


def test_no_data_gap_note_for_other_signals(tmp_path):
    db = str(tmp_path / "other.db")
    _seed_prices(db, {"AAA": _trend(300)})
    assert signal_data_note("momentum", ["AAA"], db) is None


def test_compute_all_summary_shows_note_not_failed(tmp_path):
    """The summary Status carries the plain-English note — not 'Failed',
    and not a bare 'OK' with 0 rows."""
    db = str(tmp_path / "summ.db")
    _seed_prices(db, {"AAA": _trend(300)})
    summary = compute_all_signals(["AAA"], db)
    twin = next(s for s in summary if s["Signal"] == "Twin Momentum")
    assert twin["Tickers OK"] == "1/1"
    assert twin["Rows stored"] == 0
    assert "No earnings data yet" in twin["Status"]
    assert "Failed" not in twin["Status"]


def test_compute_all_still_reports_real_failures(tmp_path, monkeypatch):
    """The exception path still says 'Failed' for genuine crashes, and
    the real error is logged (not swallowed silently)."""
    import logging
    import traceback

    db = str(tmp_path / "fail.db")
    _seed_prices(db, {"AAA": _trend(300)})

    def boom(name, tickers, db_path):
        raise RuntimeError("synthetic engine explosion")

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    root = logging.getLogger("qrp.ui.helpers")
    handler = _H()
    root.addHandler(handler)
    try:
        summary = compute_all_signals(
            ["AAA"], db, compute_rows=boom
        )
    finally:
        root.removeHandler(handler)
    failed = [s for s in summary if "Failed" in s["Status"]]
    assert failed, "genuine crashes must still report Failed"
    logged = []
    for r in records:
        logged.append(r.getMessage())
        if r.exc_info and r.exc_info[0] is not None:
            logged.append("".join(traceback.format_exception(*r.exc_info)))
    assert any(
        "synthetic engine explosion" in t for t in logged
    ), "the real error must reach the log"
