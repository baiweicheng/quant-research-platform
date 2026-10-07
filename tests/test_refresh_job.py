"""Tests for the background data-refresh job (qrp/data/refresh_job.py).

All synthetic — the worker's refresh_new_data / paper_broker are
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


def _write_raw_status(db, **overrides):
    st = {**RJ._idle_status(), **overrides}
    RJ._write_status_file(RJ.status_path(db), st)
    return st


def _wait_for(db, want_states, timeout=10.0):
    """Poll get_status until the state is one of want_states."""
    deadline = time.time() + timeout
    st = RJ.get_status(db)
    while st["state"] not in want_states and time.time() < deadline:
        time.sleep(0.05)
        st = RJ.get_status(db)
    return st


@pytest.fixture
def quick_worker(monkeypatch):
    """A fast fake worker: a few progress ticks, then a summary."""
    import qrp.ui.helpers as H

    seen = {"progress": []}

    def fake_refresh(db_path, progress_cb=None):
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

    class FakeBroker:
        def fill_pending_orders(self):
            return {
                "filled": [{"ticker": "AAA"}],
                "skipped": [],
                "pending_remaining": 0,
                "dividends_credited": 0.0,
            }

    monkeypatch.setattr(H, "refresh_new_data", fake_refresh)
    monkeypatch.setattr(H, "paper_broker", lambda db_path: FakeBroker())
    return seen


def test_get_status_idle_without_file(dbdir):
    st = RJ.get_status(dbdir)
    assert st["state"] == "idle"
    assert st["summary"] is None


def test_get_status_survives_corrupt_file(dbdir):
    RJ.status_path(dbdir).write_text("not json{{")
    st = RJ.get_status(dbdir)
    assert st["state"] == "idle"


def test_start_runs_to_done(dbdir, quick_worker):
    st0 = RJ.start_refresh(dbdir)
    assert st0["state"] == "running"
    assert st0["run_id"]

    st = _wait_for(dbdir, {"done"})
    assert st["state"] == "done"
    assert st["summary"]["new_bars"] == 3
    assert st["fill_report"]["filled"] == [{"ticker": "AAA"}]
    # tuples survived the JSON round-trip as lists
    assert st["summary"]["price_failures"] == [["ZZZ", "boom"]]
    assert any("3 new daily prices" in line for line in st["summary"]["lines"])
    # progress ticks were recorded along the way
    assert quick_worker["progress"] == [1, 2, 3, 4]


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

    monkeypatch.setattr(H, "refresh_new_data", blocking_refresh)
    monkeypatch.setattr(H, "paper_broker", lambda db_path: FakeBroker())

    s1 = RJ.start_refresh(dbdir)
    assert entered.wait(10)
    s2 = RJ.start_refresh(dbdir)
    assert s2["run_id"] == s1["run_id"]  # same run, not a new one
    release.set()

    st = _wait_for(dbdir, {"done"})
    assert st["state"] == "done"
    assert calls["n"] == 1  # the worker ran exactly once


def test_stale_heartbeat_marked_interrupted(dbdir):
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    _write_raw_status(
        dbdir, state="running", run_id="abc", heartbeat=old, started_at=old
    )
    st = RJ.get_status(dbdir)
    assert st["state"] == "interrupted"
    assert "10 minutes" in st["error"]
    # ... and the marking was persisted, not just computed
    st2 = RJ.get_status(dbdir)
    assert st2["state"] == "interrupted"


def test_start_after_stale_begins_new_run(dbdir, quick_worker):
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    _write_raw_status(
        dbdir, state="running", run_id="stale-run", heartbeat=old,
        started_at=old,
    )
    s = RJ.start_refresh(dbdir)
    assert s["state"] == "running"
    assert s["run_id"] != "stale-run"
    st = _wait_for(dbdir, {"done"})
    assert st["state"] == "done"


def test_worker_error_captured_as_plain_english(dbdir, monkeypatch):
    import qrp.ui.helpers as H

    def boom(db_path, progress_cb=None):
        raise RuntimeError("kaput\nwith\ttabs")

    monkeypatch.setattr(H, "refresh_new_data", boom)
    RJ.start_refresh(dbdir)
    st = _wait_for(dbdir, {"error"})
    assert st["state"] == "error"
    assert "kaput" in st["error"]
    assert "\n" not in st["error"]
    assert "\t" not in st["error"]


def test_fill_pass_failure_keeps_done_with_note(dbdir, monkeypatch):
    """If only the fill pass fails, the refresh still counts as done."""
    import qrp.ui.helpers as H

    monkeypatch.setattr(
        H,
        "refresh_new_data",
        lambda db_path, progress_cb=None: {
            "lines": ["Added 1 new daily price for 1 stock(s)."],
            "up_to_date": False,
        },
    )

    class BadBroker:
        def fill_pending_orders(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(H, "paper_broker", lambda db_path: BadBroker())
    RJ.start_refresh(dbdir)
    st = _wait_for(dbdir, {"done"})
    assert st["state"] == "done"
    assert "nope" in st["fill_report"]["error"]


def test_superseded_worker_does_not_clobber(dbdir, quick_worker, monkeypatch):
    """An old worker finishing late must not overwrite a newer run."""
    import qrp.ui.helpers as H

    fake = H.refresh_new_data  # the quick_worker's fake (already patched)
    started_second = threading.Event()

    def slow_then_fast(db_path, progress_cb=None):
        # First call: pretend to be slow; the test will start a second
        # run mid-way, then let this one finish.
        if not started_second.is_set():
            started_second.set()
            time.sleep(0.3)
        return fake(db_path, progress_cb=progress_cb)

    monkeypatch.setattr(H, "refresh_new_data", slow_then_fast)

    s1 = RJ.start_refresh(dbdir)
    assert started_second.wait(10)
    # Force-start a second run by faking staleness of the first.
    path = RJ.status_path(dbdir)
    cur = RJ._read_status_file(path)
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cur["heartbeat"] = old
    RJ._write_status_file(path, cur)
    s2 = RJ.start_refresh(dbdir)
    assert s2["run_id"] != s1["run_id"]
    st = _wait_for(dbdir, {"done"}, timeout=15)
    # The winner is the second run; the first run's late "done"
    # write must not have resurrected its own run_id.
    assert st["run_id"] == s2["run_id"]


# -- Market data page states -------------------------------------------------

def _market_data_app(monkeypatch, status):
    """AppTest on the Market data page with the refresh job mocked."""
    from pathlib import Path

    import streamlit as st_mod

    from qrp.ui import helpers as H
    from streamlit.testing.v1 import AppTest

    calls = {"start": 0}
    states = {"current": status}

    def fake_status(db_path):
        return states["current"]

    def fake_start(db_path):
        calls["start"] += 1
        return states["current"]

    monkeypatch.setattr(H, "refresh_job_status", fake_status)
    monkeypatch.setattr(H, "refresh_job_start", fake_start)
    # st.rerun() inside AppTest is erratic (it can discard the run's
    # deltas), so no-op it: the tests drive re-runs manually.
    monkeypatch.setattr(st_mod, "rerun", lambda *a, **k: None)

    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "qrp" / "ui" / "app.py"))
    at.session_state[H.NAV_PAGE_KEY] = "Market data"
    return at, calls, states


def _button(at, label):
    return [b for b in at.button if b.label == label]


def test_page_idle_shows_start_button_and_starts_job(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch, {**RJ._idle_status(), "state": "idle"}
    )
    at.run()
    assert not at.exception
    assert len(_button(at, "Refresh new data")) == 1

    _button(at, "Refresh new data")[0].click().run()
    assert not at.exception
    assert calls["start"] == 1


def test_page_running_shows_progress_not_button(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch,
        {
            **RJ._idle_status(),
            "state": "running",
            "done": 3,
            "total": 10,
            "step": "Checking for new daily prices — 3 of 10 steps…",
        },
    )
    at.run()
    assert not at.exception
    # AppTest 1.64 has no progress accessor; the running branch's caption
    # text is unique to it, and st.progress itself rendered without error.
    assert any(
        "keeps running even if you switch pages" in c.value
        for c in at.caption
    )
    # No double-start while a run is live.
    assert _button(at, "Refresh new data") == []
    assert _button(at, "Try the refresh again") == []


def test_page_done_shows_summary_and_button_again(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch,
        {
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
        },
    )
    at.run()
    assert not at.exception
    assert any("5 new daily prices" in s.value for s in at.success)
    assert any("Note:" in i.value for i in at.info)
    assert any("Filled 1 pending paper order" in s.value for s in at.success)
    # A finished run can be refreshed again.
    assert len(_button(at, "Refresh new data")) == 1


def test_page_error_shows_try_again(monkeypatch):
    at, calls, states = _market_data_app(
        monkeypatch,
        {
            **RJ._idle_status(),
            "state": "error",
            "error": "The refresh stopped early: kaput",
        },
    )
    at.run()
    assert not at.exception
    assert any("kaput" in e.value for e in at.error)
    assert len(_button(at, "Try the refresh again")) == 1

    _button(at, "Try the refresh again")[0].click().run()
    assert not at.exception
    assert calls["start"] == 1


def test_page_navigate_away_and_back_keeps_running(monkeypatch):
    """Leaving the page mid-run must not stop or restart the run."""
    from qrp.ui import helpers as H

    at, calls, states = _market_data_app(
        monkeypatch,
        {
            **RJ._idle_status(),
            "state": "running",
            "done": 1,
            "total": 10,
            "step": "Checking for new daily prices — 1 of 10 steps…",
        },
    )
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
    assert _button(at, "Refresh new data") == []
    assert calls["start"] == 0
