"""Tests for the background data-refresh jobs (qrp/data/refresh_job.py).

Two independent scopes — "prices" and "filings" — each with its own
status file, single-run guard, and heartbeat. All synthetic: the
workers' refresh_prices / refresh_filings / paper_broker are
monkeypatched, so no network and no real database work happens. Each
test uses a throwaway directory so status files never collide.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from qrp.data import refresh_job as RJ


@pytest.fixture
def dbdir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d / "t.db"


def _write_raw_status(db, scope, **overrides):
    st = {**RJ._idle_status(), **overrides}
    RJ._write_status_file(RJ.status_path(db, scope), st)
    return st


def _wait_for(db, scope, want_states, timeout=10.0):
    """Poll get_status until the state is one of want_states."""
    deadline = time.time() + timeout
    st = RJ.get_status(db, scope)
    while st["state"] not in want_states and time.time() < deadline:
        time.sleep(0.05)
        st = RJ.get_status(db, scope)
    return st


@pytest.fixture
def quick_worker(monkeypatch):
    """Fast fake workers: a few progress ticks, then a summary, per scope."""
    import qrp.ui.helpers as H

    seen = {"progress": [], "prices_calls": 0, "filings_calls": 0}

    def fake_prices(db_path, progress_cb=None):
        seen["prices_calls"] += 1
        for i in range(1, 5):
            time.sleep(0.01)
            progress_cb(i, 4)
            seen["progress"].append(i)
        return {
            "tickers": 2,
            "new_bars": 3,
            "new_facts": 0,
            "lines": ["Added 3 new daily prices for 2 stock(s)."],
            "up_to_date": False,
            # tuples on purpose: the status file must JSON-round-trip them
            "price_failures": [("ZZZ", "boom")],
            "fund_failures": [],
        }

    def fake_filings(db_path, progress_cb=None):
        seen["filings_calls"] += 1
        for i in range(1, 3):
            time.sleep(0.01)
            progress_cb(i, 2)
        return {
            "tickers": 2,
            "new_bars": 0,
            "new_facts": 9,
            "lines": ["Added 9 new reported numbers from SEC filings."],
            "up_to_date": False,
            "price_failures": [],
            "fund_failures": [],
        }

    class FakeBroker:
        def fill_pending_orders(self):
            return {
                "filled": [{"ticker": "AAA"}],
                "skipped": [],
                "pending_remaining": 0,
                "dividends_credited": 0.0,
            }

    monkeypatch.setattr(H, "refresh_prices", fake_prices)
    monkeypatch.setattr(H, "refresh_filings", fake_filings)
    monkeypatch.setattr(H, "paper_broker", lambda db_path: FakeBroker())
    return seen


def test_get_status_idle_without_file(dbdir):
    for scope in RJ.SCOPES:
        st = RJ.get_status(dbdir, scope)
        assert st["state"] == "idle"
        assert st["summary"] is None


def test_get_status_survives_corrupt_file(dbdir):
    RJ.status_path(dbdir, "prices").write_text("not json{{")
    st = RJ.get_status(dbdir, "prices")
    assert st["state"] == "idle"


def test_start_refresh_rejects_unknown_scope(dbdir):
    with pytest.raises(ValueError):
        RJ.start_refresh(dbdir, "everything")
    with pytest.raises(ValueError):
        RJ.get_status(dbdir, "everything")


def test_start_runs_to_done(dbdir, quick_worker):
    st0 = RJ.start_refresh(dbdir, "prices")
    assert st0["state"] == "running"
    assert st0["run_id"]
    assert st0["scope"] == "prices"

    st = _wait_for(dbdir, "prices", {"done"})
    assert st["state"] == "done"
    assert st["summary"]["new_bars"] == 3
    assert st["fill_report"]["filled"] == [{"ticker": "AAA"}]
    # tuples survived the JSON round-trip as lists
    assert st["summary"]["price_failures"] == [["ZZZ", "boom"]]
    assert any("3 new daily prices" in line for line in st["summary"]["lines"])
    # progress ticks were recorded along the way
    assert quick_worker["progress"] == [1, 2, 3, 4]
    assert quick_worker["prices_calls"] == 1
    assert quick_worker["filings_calls"] == 0


def test_filings_job_skips_fill_pass(dbdir, monkeypatch):
    """The filings job must never touch the paper broker."""
    import qrp.ui.helpers as H

    broker_calls = {"n": 0}

    def fake_filings(db_path, progress_cb=None):
        return {"lines": ["up to date"], "up_to_date": True}

    def boom_broker(db_path):
        broker_calls["n"] += 1
        raise AssertionError("filings job must not call paper_broker")

    monkeypatch.setattr(H, "refresh_filings", fake_filings)
    monkeypatch.setattr(H, "paper_broker", boom_broker)
    RJ.start_refresh(dbdir, "filings")
    st = _wait_for(dbdir, "filings", {"done"})
    assert st["state"] == "done"
    assert st["fill_report"] is None
    assert broker_calls["n"] == 0


def test_second_start_while_running_is_noop(dbdir, monkeypatch):
    """A second start while a run is live returns it instead of doubling."""
    import qrp.ui.helpers as H

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def blocking_refresh(db_path, progress_cb=None):
        calls["n"] += 1
        entered.set()
        release.wait(10)
        return {"lines": [], "up_to_date": True}

    class FakeBroker:
        def fill_pending_orders(self):
            return {
                "filled": [],
                "skipped": [],
                "pending_remaining": 0,
                "dividends_credited": 0.0,
            }

    monkeypatch.setattr(H, "refresh_prices", blocking_refresh)
    monkeypatch.setattr(H, "paper_broker", lambda db_path: FakeBroker())

    s1 = RJ.start_refresh(dbdir, "prices")
    assert entered.wait(10)
    s2 = RJ.start_refresh(dbdir, "prices")
    assert s2["run_id"] == s1["run_id"]  # same run, not a new one
    release.set()

    st = _wait_for(dbdir, "prices", {"done"})
    assert st["state"] == "done"
    assert calls["n"] == 1  # the worker ran exactly once


def test_stale_heartbeat_marked_interrupted(dbdir):
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    _write_raw_status(
        dbdir, "prices", state="running", run_id="abc", heartbeat=old,
        started_at=old,
    )
    st = RJ.get_status(dbdir, "prices")
    assert st["state"] == "interrupted"
    assert "10 minutes" in st["error"]
    # ... and the marking was persisted, not just computed
    st2 = RJ.get_status(dbdir, "prices")
    assert st2["state"] == "interrupted"


def test_start_after_stale_begins_new_run(dbdir, quick_worker):
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    _write_raw_status(
        dbdir, "prices", state="running", run_id="stale-run", heartbeat=old,
        started_at=old,
    )
    s = RJ.start_refresh(dbdir, "prices")
    assert s["state"] == "running"
    assert s["run_id"] != "stale-run"
    st = _wait_for(dbdir, "prices", {"done"})
    assert st["state"] == "done"


def test_worker_error_captured_as_plain_english(dbdir, monkeypatch):
    import qrp.ui.helpers as H

    def boom(db_path, progress_cb=None):
        raise RuntimeError("kaput\nwith\ttabs")

    monkeypatch.setattr(H, "refresh_prices", boom)
    RJ.start_refresh(dbdir, "prices")
    st = _wait_for(dbdir, "prices", {"error"})
    assert st["state"] == "error"
    assert "kaput" in st["error"]
    assert "\n" not in st["error"]
    assert "\t" not in st["error"]


def test_fill_pass_failure_keeps_done_with_note(dbdir, monkeypatch):
    """If only the fill pass fails, the refresh still counts as done."""
    import qrp.ui.helpers as H

    monkeypatch.setattr(
        H,
        "refresh_prices",
        lambda db_path, progress_cb=None: {
            "lines": ["Added 1 new daily price for 1 stock(s)."],
            "up_to_date": False,
        },
    )

    class BadBroker:
        def fill_pending_orders(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(H, "paper_broker", lambda db_path: BadBroker())
    RJ.start_refresh(dbdir, "prices")
    st = _wait_for(dbdir, "prices", {"done"})
    assert st["state"] == "done"
    assert "nope" in st["fill_report"]["error"]


def test_superseded_worker_does_not_clobber(dbdir, quick_worker, monkeypatch):
    """An old worker finishing late must not overwrite a newer run."""
    import qrp.ui.helpers as H

    fake = H.refresh_prices  # the quick_worker's fake (already patched)
    started_second = threading.Event()

    def slow_then_fast(db_path, progress_cb=None):
        # First call: pretend to be slow; the test will start a second
        # run mid-way, then let this one finish.
        if not started_second.is_set():
            started_second.set()
            time.sleep(0.3)
        return fake(db_path, progress_cb=progress_cb)

    monkeypatch.setattr(H, "refresh_prices", slow_then_fast)

    s1 = RJ.start_refresh(dbdir, "prices")
    assert started_second.wait(10)
    # Force-start a second run by faking staleness of the first.
    path = RJ.status_path(dbdir, "prices")
    cur = RJ._read_status_file(path)
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cur["heartbeat"] = old
    RJ._write_status_file(path, cur)
    s2 = RJ.start_refresh(dbdir, "prices")
    assert s2["run_id"] != s1["run_id"]
    st = _wait_for(dbdir, "prices", {"done"}, timeout=15)
    # The winner is the second run; the first run's late "done"
    # write must not have resurrected its own run_id.
    assert st["run_id"] == s2["run_id"]


def test_scopes_have_independent_status_files(dbdir, quick_worker):
    """A prices run must not touch the filings status (and vice versa)."""
    assert RJ.status_path(dbdir, "prices") != RJ.status_path(dbdir, "filings")
    RJ.start_refresh(dbdir, "prices")
    st = _wait_for(dbdir, "prices", {"done"})
    assert st["state"] == "done"
    assert RJ.get_status(dbdir, "filings")["state"] == "idle"


def test_concurrent_scopes_both_complete(dbdir, quick_worker):
    """Prices and filings may run at the same time without clobbering."""
    RJ.start_refresh(dbdir, "prices")
    RJ.start_refresh(dbdir, "filings")
    st_p = _wait_for(dbdir, "prices", {"done"})
    st_f = _wait_for(dbdir, "filings", {"done"})
    assert st_p["state"] == "done"
    assert st_f["state"] == "done"
    assert st_p["summary"]["new_bars"] == 3
    assert st_f["summary"]["new_facts"] == 9
    assert st_f["fill_report"] is None
    assert quick_worker["prices_calls"] == 1
    assert quick_worker["filings_calls"] == 1


def test_filings_job_records_last_completed_at(dbdir, quick_worker):
    assert RJ.filings_last_checked(dbdir) is None  # never checked yet
    RJ.start_refresh(dbdir, "filings")
    st = _wait_for(dbdir, "filings", {"done"})
    assert st["state"] == "done"
    assert st["last_completed_at"]
    checked = RJ.filings_last_checked(dbdir)
    assert checked is not None
    # Plain-English date, e.g. "October 7, 2026".
    assert checked[0].isupper() and ", 20" in checked


def test_last_completed_at_preserved_during_new_run(dbdir, monkeypatch):
    """The 'Filings last checked' line must not go blank mid-run."""
    import qrp.ui.helpers as H

    stamp = "2026-10-01T12:00:00+00:00"
    _write_raw_status(
        dbdir, "filings", state="done", last_completed_at=stamp,
        summary={"lines": [], "up_to_date": True},
    )
    assert RJ.filings_last_checked(dbdir) == "October 1, 2026"

    entered = threading.Event()
    release = threading.Event()

    def blocking(db_path, progress_cb=None):
        entered.set()
        release.wait(10)
        return {"lines": [], "up_to_date": True}

    monkeypatch.setattr(H, "refresh_filings", blocking)
    RJ.start_refresh(dbdir, "filings")
    assert entered.wait(10)
    mid = RJ.get_status(dbdir, "filings")
    assert mid["state"] == "running"
    assert mid["last_completed_at"] == stamp  # preserved, not cleared
    assert RJ.filings_last_checked(dbdir) == "October 1, 2026"
    release.set()
    done = _wait_for(dbdir, "filings", {"done"})
    assert done["last_completed_at"] != stamp  # refreshed on completion


def test_filings_last_checked_survives_restart_and_corruption(dbdir):
    """The stamp lives in the file: restarts keep it, corruption hides it."""
    stamp = "2026-09-15T08:30:00+00:00"
    _write_raw_status(dbdir, "filings", state="done",
                      last_completed_at=stamp)
    # Simulate an app restart: fresh import state, same files.
    assert RJ.filings_last_checked(dbdir) == "September 15, 2026"
    RJ.status_path(dbdir, "filings").write_text("not json{{")
    assert RJ.filings_last_checked(dbdir) is None
    assert RJ.get_status(dbdir, "filings")["state"] == "idle"


# -- Market data page states -------------------------------------------------

def _market_data_app(monkeypatch, statuses):
    """AppTest on the Market data page with both refresh jobs mocked.

    `statuses` maps "prices"/"filings" to status dicts and may carry
    "last_checked" for the filings caption.
    """
    from pathlib import Path

    import streamlit as st_mod

    from qrp.ui import helpers as H
    from streamlit.testing.v1 import AppTest

    calls = {"start": 0, "prices": 0, "filings": 0}
    states = dict(statuses)

    def fake_status(db_path, scope="prices"):
        return states[scope]

    def fake_start(db_path, scope="prices"):
        calls["start"] += 1
        calls[scope] += 1
        return states[scope]

    monkeypatch.setattr(H, "refresh_job_status", fake_status)
    monkeypatch.setattr(H, "refresh_job_start", fake_start)
    monkeypatch.setattr(
        H, "refresh_filings_last_checked", lambda db_path: states.get(
            "last_checked"
        ),
    )
    # st.rerun() inside AppTest is erratic (it can discard the run's
    # deltas), so no-op it: the tests drive re-runs manually.
    monkeypatch.setattr(st_mod, "rerun", lambda *a, **k: None)

    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "qrp" / "ui" / "app.py"))
    at.session_state[H.NAV_PAGE_KEY] = "Market data"
    return at, calls, states


def _idle_statuses(**overrides):
    base = {
        "prices": {**RJ._idle_status(), "state": "idle"},
        "filings": {**RJ._idle_status(), "state": "idle"},
        "last_checked": None,
    }
    base.update(overrides)
    return base


def _button(at, label):
    return [b for b in at.button if b.label == label]


def test_page_idle_shows_both_buttons_and_starts_scoped_jobs(monkeypatch):
    at, calls, states = _market_data_app(monkeypatch, _idle_statuses())
    at.run()
    assert not at.exception
    assert len(_button(at, "Refresh prices")) == 1
    assert len(_button(at, "Refresh SEC filings")) == 1
    # The old combined button is gone.
    assert _button(at, "Refresh new data") == []

    _button(at, "Refresh prices")[0].click().run()
    _button(at, "Refresh SEC filings")[0].click().run()
    assert not at.exception
    assert calls["prices"] == 1
    assert calls["filings"] == 1


def test_filings_section_shows_last_checked(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch, _idle_statuses(last_checked="October 7, 2026")
    )
    at.run()
    assert not at.exception
    assert any(
        "Filings last checked: October 7, 2026." in c.value
        for c in at.caption
    )


def test_filings_section_shows_not_checked_yet(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch, _idle_statuses(last_checked=None)
    )
    at.run()
    assert not at.exception
    assert any("not checked yet" in c.value for c in at.caption)


def test_page_running_shows_progress_not_button(monkeypatch):
    statuses = _idle_statuses()
    statuses["prices"] = {
        **RJ._idle_status(),
        "state": "running",
        "done": 3,
        "total": 10,
        "step": "Checking for new daily prices — 3 of 10 steps…",
    }
    at, calls, states = _market_data_app(monkeypatch, statuses)
    at.run()
    assert not at.exception
    # AppTest 1.64 has no progress accessor; the running branch's caption
    # text is unique to it, and st.progress itself rendered without error.
    assert any(
        "keeps running even if you switch pages" in c.value
        for c in at.caption
    )
    # No double-start while a run is live; the filings side is unaffected.
    assert _button(at, "Refresh prices") == []
    assert _button(at, "Try the prices refresh again") == []
    assert len(_button(at, "Refresh SEC filings")) == 1


def test_page_done_prices_shows_summary_and_fill_report(monkeypatch):
    statuses = _idle_statuses()
    statuses["prices"] = {
        **RJ._idle_status(),
        "state": "done",
        "summary": {
            "up_to_date": False,
            "lines": [
                "Added 5 new daily prices for 2 stock(s).",
                "Note: something worth knowing.",
            ],
        },
        "fill_report": {
            "filled": [{"ticker": "AAA"}],
            "skipped": [],
        },
    }
    at, calls, states = _market_data_app(monkeypatch, statuses)
    at.run()
    assert not at.exception
    assert any("5 new daily prices" in s.value for s in at.success)
    assert any("Note:" in i.value for i in at.info)
    assert any("Filled 1 pending paper order" in s.value for s in at.success)
    # A finished run can be refreshed again.
    assert len(_button(at, "Refresh prices")) == 1


def test_page_done_filings_shows_summary_without_fill_report(monkeypatch):
    statuses = _idle_statuses()
    statuses["filings"] = {
        **RJ._idle_status(),
        "state": "done",
        "summary": {
            "up_to_date": False,
            "lines": ["Added 9 new reported numbers from SEC filings."],
        },
        # Even if a fill report were present, the filings scope never
        # shows one.
        "fill_report": {"filled": [{"ticker": "AAA"}], "skipped": []},
    }
    at, calls, states = _market_data_app(monkeypatch, statuses)
    at.run()
    assert not at.exception
    assert any("9 new reported numbers" in s.value for s in at.success)
    assert not any("pending paper order" in s.value for s in at.success)
    assert len(_button(at, "Refresh SEC filings")) == 1


def test_page_error_shows_try_again(monkeypatch):
    statuses = _idle_statuses()
    statuses["filings"] = {
        **RJ._idle_status(),
        "state": "error",
        "error": "The refresh stopped early: kaput",
    }
    at, calls, states = _market_data_app(monkeypatch, statuses)
    at.run()
    assert not at.exception
    assert any("kaput" in e.value for e in at.error)
    assert len(_button(at, "Try the filings refresh again")) == 1

    _button(at, "Try the filings refresh again")[0].click().run()
    assert not at.exception
    assert calls["filings"] == 1
    # The prices side was never touched.
    assert calls["prices"] == 0


def test_page_navigate_away_and_back_keeps_running(monkeypatch):
    """Leaving the page mid-run must not stop or restart the run."""
    from qrp.ui import helpers as H

    statuses = _idle_statuses()
    statuses["prices"] = {
        **RJ._idle_status(),
        "state": "running",
        "done": 1,
        "total": 10,
        "step": "Checking for new daily prices — 1 of 10 steps…",
    }
    at, calls, states = _market_data_app(monkeypatch, statuses)
    at.run()
    assert len(at.caption) >= 1

    # "Navigate away": run a different page, then come back.
    at.session_state[H.NAV_PAGE_KEY] = "Signals"
    at.run()
    at.session_state[H.NAV_PAGE_KEY] = "Market data"
    at.run()
    assert not at.exception
    # Still the same running run — progress shown, never a start button.
    assert any(
        "keeps running even if you switch pages" in c.value
        for c in at.caption
    )
    assert _button(at, "Refresh prices") == []
    assert calls["start"] == 0
