"""UI tests for the S&P 500 membership updater (Market data page)."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qrp.ui import helpers as H  # noqa: E402
from qrp.data import sp500_update as U  # noqa: E402


def _canned_diff():
    return {
        "checked_on": "2026-04-01",
        "up_to_date": False,
        "dated": [
            {
                "date": "2026-03-20",
                "added": "HHH",
                "removed": "FFF",
                "added_name": "H Corp",
                "removed_name": "F Corp",
            }
        ],
        "renames": [],
        "undated_adds": [],
        "undated_removes": [],
        "corrections": [],
        "already_recorded": [],
    }


def _up_to_date_diff():
    return {
        "checked_on": "2026-04-01",
        "up_to_date": True,
        "dated": [],
        "renames": [],
        "undated_adds": [],
        "undated_removes": [],
        "corrections": [],
        "already_recorded": [],
    }


@pytest.fixture()
def market_data_app(tmp_path, monkeypatch):
    """AppTest on the Market data page with the check-status file redirected."""
    status_path = tmp_path / "sp500_membership_check.json"
    monkeypatch.setattr(
        H, "sp500_status_path", lambda db_path=H.DEFAULT_DB: status_path
    )
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "qrp" / "ui" / "app.py"))
    at.session_state[H.NAV_PAGE_KEY] = "Market data"
    at.run()
    assert not at.exception
    return at


def test_membership_section_renders(market_data_app):
    at = market_data_app
    assert any("S&P 500 membership" in str(s.value) for s in at.subheader)
    assert any(b.label == "Check for S&P 500 changes" for b in at.button)
    assert any(
        "not checked yet" in str(c.value) for c in at.caption
    ), "expected the last-checked caption"


def test_check_up_to_date_shows_success(market_data_app):
    at = market_data_app
    with mock.patch.object(
        U, "check_for_changes", return_value=_up_to_date_diff()
    ):
        at.button(key="sp500_check").click().run()
    assert not at.exception
    assert any(
        "list is current" in str(s.value) for s in at.success
    ), "expected the up-to-date success message"
    # Apply must not appear when there is nothing to apply
    assert not any(b.label == "Apply changes" for b in at.button)


def test_check_found_shows_report_and_apply(market_data_app):
    at = market_data_app
    with mock.patch.object(U, "check_for_changes", return_value=_canned_diff()):
        at.button(key="sp500_check").click().run()
    assert not at.exception
    infos = "\n".join(str(i.value) for i in at.info)
    assert "Added: HHH (H Corp)" in infos
    assert "Removed: FFF (F Corp)" in infos
    assert "effective 2026-03-20" in infos
    assert any(b.label == "Apply changes" for b in at.button)


def test_check_fetch_failure_is_plain_english(market_data_app):
    at = market_data_app
    with mock.patch.object(
        U,
        "check_for_changes",
        side_effect=U.MembershipCheckError("Couldn't reach Wikipedia right now."),
    ):
        at.button(key="sp500_check").click().run()
    assert not at.exception
    assert any(
        "Couldn't reach Wikipedia" in str(e.value) for e in at.error
    ), "expected the plain-English fetch failure"
    assert not any(b.label == "Apply changes" for b in at.button)


def test_apply_flow_shows_summary_and_next_steps(market_data_app):
    at = market_data_app
    summary = {
        "applied": True,
        "checked_on": "2026-04-01",
        "summary_lines": ["Added: HHH (effective 2026-03-20)."],
        "new_tickers": ["HHH"],
        "next_steps": "Next: download HHH via Custom list.",
        "git_note": "Note: local repo files changed.",
    }
    with mock.patch.object(U, "check_for_changes", return_value=_canned_diff()):
        at.button(key="sp500_check").click().run()
    assert any(b.label == "Apply changes" for b in at.button)
    with mock.patch.object(U, "apply_updates", return_value=summary):
        at.button(key="sp500_apply").click().run()
    assert not at.exception
    assert any("applied" in str(s.value).lower() for s in at.success)
    assert any("Custom list" in str(w.value) for w in at.markdown) or any(
        "Custom list" in str(w.value) for w in at.write
    )


def test_apply_failure_shows_plain_error(market_data_app):
    at = market_data_app
    with mock.patch.object(U, "check_for_changes", return_value=_canned_diff()):
        at.button(key="sp500_check").click().run()
    with mock.patch.object(
        U,
        "apply_updates",
        side_effect=U.MembershipCheckError("Refusing to write: boom."),
    ):
        at.button(key="sp500_apply").click().run()
    assert not at.exception
    assert any("Refusing to write" in str(e.value) for e in at.error)


def test_last_checked_persists(market_data_app, tmp_path):
    at = market_data_app
    with mock.patch.object(
        U, "check_for_changes", return_value=_up_to_date_diff()
    ):
        at.button(key="sp500_check").click().run()
    membership_captions = [
        str(c.value) for c in at.caption if "Membership last checked:" in str(c.value)
    ]
    assert len(membership_captions) == 1
    assert "not checked yet" not in membership_captions[0]
