"""Tests for sticky Compare inputs and the 2015-01-01 default from-date.

Root cause of the reported bug: Streamlit wipes keyed widget values when a
page is left (verified empirically on streamlit 1.64/1.65), while the plain
results-cache dict survives. On return the boxes rebuilt from ``value=``
defaults, so the cache key never matched and nothing restored. The Compare
page now keeps a plain (non-widget) ``compare_inputs`` snapshot and re-seeds
the widget keys from it before the widgets are created.
"""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from qrp.ui import helpers as H
import qrp.ui.app as app

COMPARE_WIDGET_KEYS = [
    "compare_start",
    "compare_end",
    "compare_capital",
    "compare_max_weight",
    "compare_frequency",
]


@pytest.fixture
def compare_app(monkeypatch):
    """Compare page with stubbed data gates (no database needed)."""
    monkeypatch.setattr(
        H, "require_data", lambda *needs, **kw: {"tickers": ["AAA", "BBB"]}
    )
    monkeypatch.setattr(
        H, "data_date_range", lambda *a, **k: ("2013-01-02", "2024-01-05")
    )
    at = AppTest.from_file(str(Path(app.__file__)))
    at.session_state[H.NAV_PAGE_KEY] = "Compare strategies"
    return at


def _go_to(at, page):
    at.session_state[H.NAV_PAGE_KEY] = page
    at.run()
    assert not at.exception


def _fake_rows():
    return [
        {
            "Strategy": "Momentum",
            "total_return": 0.5,
            "sharpe": 1.0,
            "Total return": "+50.00%",
            "Yearly growth (CAGR)": "+10.00%",
            "Worst fall": "-5.00%",
        }
    ]


def test_compare_default_from_date_is_2015(compare_app):
    """Fix 1: the From box defaults to 2015-01-01, not the 2013 data start."""
    at = compare_app
    at.run()
    assert not at.exception
    assert at.text_input(key="compare_start").value == "2015-01-01"
    assert at.text_input(key="compare_end").value == "2024-01-05"


def test_compare_default_from_date_follows_late_data_start(monkeypatch):
    """If the data itself starts after 2015, the default follows the data."""
    import qrp.ui.app as app_module

    monkeypatch.setattr(
        H, "require_data", lambda *needs, **kw: {"tickers": ["AAA"]}
    )
    monkeypatch.setattr(
        H, "data_date_range", lambda *a, **k: ("2020-06-01", "2024-01-05")
    )
    at = AppTest.from_file(str(Path(app_module.__file__)))
    at.session_state[H.NAV_PAGE_KEY] = "Compare strategies"
    at.run()
    assert not at.exception
    assert at.text_input(key="compare_start").value == "2020-06-01"


def test_compare_inputs_survive_navigation_and_restore_results(compare_app):
    """Fix 2a: custom inputs survive a page switch; cached results restore."""
    at = compare_app
    at.run()
    assert not at.exception

    # User customizes the inputs (weekly differs from the monthly default).
    at.text_input(key="compare_start").set_value("2015-01-01").run()
    at.selectbox(key="compare_frequency").set_value("weekly").run()
    at.slider(key="compare_max_weight").set_value(0.03).run()

    # Pretend a comparison was run with exactly those inputs.
    cache = at.session_state["compare_results_cache"]
    key = H.compare_cache_key(
        [n for n, _ in H.strategy_options()],
        ["AAA", "BBB"],
        "2015-01-01",
        "2024-01-05",
        100000.0,
        0.03,
        10.0,
        "open",
        cost_model="spread",
        frequency="weekly",
    )
    H.store_cached_compare(
        cache,
        key,
        {
            "time": "2026-01-01 00:00",
            "rows": _fake_rows(),
            "bench_row": None,
            "spy_row": None,
        },
    )

    # Navigate away and back. (AppTest, like production Streamlit, wipes
    # keyed widget state when the page is left; plain session-state entries
    # such as compare_inputs and the results cache survive.)
    _go_to(at, "Home")

    # Navigate back: boxes must show the saved inputs, results must restore.
    _go_to(at, "Compare strategies")
    assert at.text_input(key="compare_start").value == "2015-01-01"
    assert at.text_input(key="compare_end").value == "2024-01-05"
    assert at.selectbox(key="compare_frequency").value == "weekly"
    assert at.slider(key="compare_max_weight").value == 0.03
    assert any(
        "Results from" in str(c.value) for c in at.caption
    ), "cached comparison should have been restored"
    assert any(b.label == "Recompute" for b in at.button)


def test_compare_changing_input_after_restore_invalidates(compare_app):
    """Fix 2b: changing any input after a restore drops the stale results."""
    at = compare_app
    at.run()
    assert not at.exception

    at.selectbox(key="compare_frequency").set_value("weekly").run()
    cache = at.session_state["compare_results_cache"]
    key = H.compare_cache_key(
        [n for n, _ in H.strategy_options()],
        ["AAA", "BBB"],
        "2015-01-01",
        "2024-01-05",
        100000.0,
        0.05,
        10.0,
        "open",
        cost_model="spread",
        frequency="weekly",
    )
    H.store_cached_compare(
        cache,
        key,
        {
            "time": "2026-01-01 00:00",
            "rows": _fake_rows(),
            "bench_row": None,
            "spy_row": None,
        },
    )
    _go_to(at, "Home")
    _go_to(at, "Compare strategies")
    assert any("Results from" in str(c.value) for c in at.caption)

    # Change one input: the restore must vanish, offering a fresh run.
    at.selectbox(key="compare_frequency").set_value("daily").run()
    assert not any("Results from" in str(c.value) for c in at.caption)
    assert any(b.label == "Compare strategies" for b in at.button)


def test_compare_input_defaults_pure():
    """The default-input helper needs no Streamlit runtime."""
    d = app._compare_input_defaults("2013-01-02", "2024-01-05")
    assert d["compare_start"] == "2015-01-01"
    assert d["compare_end"] == "2024-01-05"
    assert d["compare_capital"] == 100_000.0
    assert d["compare_max_weight"] == 0.05
    assert d["compare_frequency"] == "monthly"
    d2 = app._compare_input_defaults("2020-06-01", "2024-01-05")
    assert d2["compare_start"] == "2020-06-01"
