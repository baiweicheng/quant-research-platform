"""Tests for the three UX features: persistent comparison results,
"Compute all signals", and formal strategy names + calculation details.

All tests use SYNTHETIC data and in-memory stand-ins (no network, no
Streamlit runtime) — they exercise the pure helper functions the pages
rely on, plus the registry data contract the Strategies page renders.
"""

from __future__ import annotations

import inspect

import pytest

streamlit = pytest.importorskip("streamlit")

from qrp.signals import default_registry
from qrp.strategies.registry import STRATEGIES, display_name, get_strategy
from qrp.ui import helpers as H


EXPECTED_KEYS = {
    "trending_value",
    "value_momentum",
    "quality_lowvol",
    "momentum",
    "trend",
    "low_volatility",
    "continuous_momentum",
    "overnight_intraday",
    "twin_momentum",
}

EXPECTED_FORMAL_NAMES = {
    "trending_value": "Trending Value (O'Shaughnessy)",
    "value_momentum": "Value + Momentum Composite (AQR-Style)",
    "quality_lowvol": "Quality + Low Volatility Composite",
    "momentum": "12-1 Month Cross-Sectional Momentum",
    "trend": "200-Day Moving-Average Trend Filter",
    "low_volatility": "Low Volatility (3-Month)",
    "continuous_momentum": "Continuous Momentum (Frog-in-the-Pan)",
    "overnight_intraday": "Intraday Institutional Flow (Overnight Divergence)",
    "twin_momentum": "Twin Momentum (Price + Earnings Surprise)",
}


# -- Feature 3a/3b: formal names + calculation text -------------------------


def test_internal_registry_keys_unchanged():
    """The machine-readable keys (used by CLI/tests/saved state) are stable."""
    assert set(STRATEGIES) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        # get_strategy still resolves every old key.
        assert get_strategy(key).name == key


def test_every_strategy_has_formal_display_name():
    for key, expected in EXPECTED_FORMAL_NAMES.items():
        formal = STRATEGIES[key].formal_name
        assert formal, f"{key} is missing its formal name"
        assert display_name(key) == formal
        assert display_name(key) == expected


def test_every_strategy_has_nonempty_calculation():
    for key, s in STRATEGIES.items():
        assert s.calculation.strip(), f"{key} is missing its calculation text"
        # The calculation must describe real steps, not just echo the blurb.
        assert "1." in s.calculation and "2." in s.calculation, (
            f"{key}: calculation text should be numbered steps"
        )


def test_display_name_falls_back_for_unknown_names():
    """Signal names reused as labels keep the old title-cased behavior."""
    assert display_name("value_composite") == "Value Composite"
    assert display_name("not_registered") == "Not Registered"


def test_calculation_names_match_real_math():
    """Spot-checks that formal names honestly describe the signal code."""
    # Low volatility is computed over 63 trading days (~3 months), not 12.
    assert "3-Month" in STRATEGIES["low_volatility"].formal_name
    # Trend really is a 200-day moving-average filter.
    assert "200-Day" in STRATEGIES["trend"].formal_name
    assert "200-day average" in STRATEGIES["trend"].calculation
    # Value composite really uses book-to-market, earnings yield and
    # cash-flow yield (not P/E, P/B, EV/EBITDA).
    calc = STRATEGIES["value_momentum"].calculation
    assert "book-to-market" in calc
    assert "earnings yield" in calc
    assert "cash-flow yield" in calc
    # Trending Value really blends value with a 6-month momentum half.
    assert "126 trading days" in STRATEGIES["trending_value"].calculation


# -- Feature 1: persistent comparison results --------------------------------


def test_compare_cache_key_is_stable_for_same_inputs():
    k1 = H.compare_cache_key(
        ["a", "b"], ["X", "Y"], "2020-01-01", "2021-01-01",
        100_000.0, 0.10, 10.0, "open",
    )
    k2 = H.compare_cache_key(
        # Order of names/tickers must not matter.
        ["b", "a"], ["Y", "X"], "2020-01-01", "2021-01-01",
        100_000.0, 0.10, 10.0, "open",
    )
    assert k1 == k2


def test_compare_cache_key_changes_with_any_input():
    base = dict(
        strategy_names=["a", "b"], tickers=["X"], start="2020-01-01",
        end="2021-01-01", capital=100_000.0, max_weight=0.10,
        cost_bps=10.0, execution="open",
    )
    key0 = H.compare_cache_key(**base)
    variants = [
        {**base, "end": "2022-01-01"},
        {**base, "capital": 200_000.0},
        {**base, "max_weight": 0.20},
        {**base, "tickers": ["X", "Y"]},
        {**base, "strategy_names": ["a"]},
        {**base, "execution": "close"},
        {**base, "start": None},
    ]
    for v in variants:
        assert H.compare_cache_key(**v) != key0


def test_compare_cache_round_trip_same_inputs_restored():
    """Simulates leaving the page and coming back with the same inputs."""
    cache: dict = {}
    key = H.compare_cache_key(
        ["momentum"], ["AAA"], "2020-01-01", "2021-01-01",
        100_000.0, 0.10, 10.0, "open",
    )
    payload = {
        "time": "2026-10-03 12:00",
        "rows": [{"Strategy": "12-1 Month Cross-Sectional Momentum"}],
        "bench_row": {"Strategy": "Market benchmark"},
        "spy_row": None,
    }
    H.store_cached_compare(cache, key, payload)
    restored = H.get_cached_compare(cache, key)
    assert restored is not None
    assert restored["rows"][0]["Strategy"] == "12-1 Month Cross-Sectional Momentum"
    assert restored["time"] == "2026-10-03 12:00"


def test_compare_cache_changed_inputs_not_restored():
    """A changed input must never surface stale results."""
    cache: dict = {}
    key_old = H.compare_cache_key(
        ["momentum"], ["AAA"], "2020-01-01", "2021-01-01",
        100_000.0, 0.10, 10.0, "open",
    )
    H.store_cached_compare(cache, key_old, {"rows": ["old"]})
    key_new = H.compare_cache_key(
        ["momentum"], ["AAA"], "2020-01-01", "2021-01-01",
        50_000.0, 0.10, 10.0, "open",  # different capital
    )
    assert H.get_cached_compare(cache, key_new) is None


# -- Feature 2: compute all signals ------------------------------------------


def test_compute_all_signals_iterates_every_registered_signal():
    """Every registered signal is computed exactly once (mocked compute)."""
    called: list[str] = []

    def fake_compute(name, tickers, db_path):
        called.append(name)
        return [
            {
                "Ticker": t,
                "Values stored": 5,
                "Total in database": 5,
                "Status": "OK",
            }
            for t in tickers
        ]

    summary = H.compute_all_signals(
        ["AAA", "BBB"], "/tmp/never.db", compute_rows=fake_compute
    )
    expected = [name for name, _ in default_registry().list_signals()]
    assert called == expected  # every signal, in registry order
    assert len(summary) == len(expected)
    for row in summary:
        assert row["Status"] == "OK"
        assert row["Tickers OK"] == "2/2"
        assert row["Rows stored"] == 10


def test_compute_all_signals_progress_callback():
    seen: list[tuple[int, int, str]] = []
    H.compute_all_signals(
        ["AAA"],
        "/tmp/never.db",
        compute_rows=lambda n, t, d: [],
        progress=lambda done, total, label: seen.append((done, total, label)),
    )
    n = len(default_registry().list_signals())
    # One call per signal plus one final "done" call.
    assert len(seen) == n + 1
    assert seen[-1] == (n, n, "")


def test_compute_all_signals_failure_does_not_raise():
    def boom(name, tickers, db_path):
        raise RuntimeError("disk exploded")

    summary = H.compute_all_signals(
        ["AAA"], "/tmp/never.db", compute_rows=boom
    )
    assert len(summary) == len(default_registry().list_signals())
    assert all("Failed" in r["Status"] for r in summary)


# -- Feature 3c/3e: UI surfaces the new content ------------------------------


def test_strategy_comparison_rows_carry_calculation():
    """The data contract the Strategies page renders."""
    rows = H.strategy_comparison_rows()
    assert len(rows) == len(EXPECTED_KEYS)
    by_name = {r["name"]: r for r in rows}
    for key in EXPECTED_KEYS:
        assert by_name[key]["Calculation"].strip()
        # The Strategies page table hides the long text column...
        assert by_name[key]["Strategy"] == EXPECTED_FORMAL_NAMES[key]


def test_strategy_calculation_helper():
    assert H.strategy_calculation("momentum") == STRATEGIES["momentum"].calculation
    with pytest.raises(KeyError):
        H.strategy_calculation("nope")


def test_strategies_page_and_picker_show_calculation_sections():
    """Smoke: the new 'How it is calculated' sections exist where the
    strategy descriptions appear (Strategies page + strategy picker)."""
    import qrp.ui.app as app

    assert "How it is calculated" in inspect.getsource(app.page_strategies)
    assert "How it is calculated" in inspect.getsource(app._strategy_picker)


def test_compare_page_remembers_results():
    """Smoke: the Compare page uses the session-state cache."""
    import qrp.ui.app as app

    src = inspect.getsource(app.page_compare)
    assert "compare_cache_key" in src
    assert "Recompute" in src
    assert "_render_compare_results" in src


def test_signals_page_has_compute_all():
    """Smoke: the Signals page offers the bulk compute button."""
    import qrp.ui.app as app

    src = inspect.getsource(app.page_signals)
    assert "Compute all signals" in src
    assert "compute_all_signals" in src
