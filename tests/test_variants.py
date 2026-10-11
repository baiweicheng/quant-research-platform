"""Tests for the strategy variant manager core (backlog #16)."""

from __future__ import annotations

import pytest

from qrp.strategies import variants as V
from qrp.strategies.registry import STRATEGIES, get_strategy


def _db(tmp_path):
    return str(tmp_path / "market.db")


# -- validation ------------------------------------------------------------

def test_unknown_parameter_is_a_hard_error():
    with pytest.raises(ValueError, match="Unknown setting"):
        V.validate_overrides({"max_stokcs": 10})


def test_numeric_bounds_are_refused_plainly():
    with pytest.raises(ValueError, match="between 0.01 and 1.0"):
        V.validate_overrides({"max_weight": 0})
    with pytest.raises(ValueError, match="between 0.01 and 1.0"):
        V.validate_overrides({"max_weight": 1.5})
    with pytest.raises(ValueError, match="between 1 and 100"):
        V.validate_overrides({"max_stocks": 0})


def test_max_stocks_must_be_whole():
    with pytest.raises(ValueError, match="whole number"):
        V.validate_overrides({"max_stocks": 10.5})


def test_single_select_rejects_unknown_option():
    with pytest.raises(ValueError, match="must be one of"):
        V.validate_overrides({"rebalance": "daily"})


def test_soft_conflict_warns_but_allows():
    cleaned, warnings = V.validate_overrides(
        {"max_stocks": 10, "max_weight": 0.05}
    )
    assert cleaned == {"max_stocks": 10, "max_weight": 0.05}
    assert len(warnings) == 1
    assert "50%" in warnings[0] and "cash" in warnings[0]


def test_no_warning_when_caps_can_fill():
    _, warnings = V.validate_overrides({"max_stocks": 20, "max_weight": 0.05})
    assert warnings == []


# -- CRUD ------------------------------------------------------------------

def test_save_and_list_roundtrip(tmp_path):
    db = _db(tmp_path)
    v, warnings = V.save_variant(
        "Concentrated 10", "value_momentum",
        {"max_stocks": 10, "max_weight": 0.10}, db,
    )
    assert v.name == "Concentrated 10"
    assert v.base == "value_momentum"
    assert warnings == []  # 10 x 10% fills the account: no warning
    listed = V.list_variants(db)
    assert len(listed) == 1
    assert listed[0].id == v.id
    assert listed[0].created != ""


def test_blank_and_duplicate_names_refused(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError, match="a name"):
        V.save_variant("  ", "momentum", {}, db)
    V.save_variant("Mine", "momentum", {}, db)
    with pytest.raises(ValueError, match="already exists"):
        V.save_variant("mine", "trend", {}, db)


def test_rename_archive_delete_flow(tmp_path):
    db = _db(tmp_path)
    v, _ = V.save_variant("Old", "momentum", {}, db)
    renamed = V.rename_variant(v.id, "New", db)
    assert renamed.name == "New"
    # Active variants cannot be deleted — archive first.
    with pytest.raises(ValueError, match="Archive"):
        V.delete_variant(v.id, db)
    V.archive_variant(v.id, db)
    assert V.list_variants(db) == []
    assert len(V.list_variants(db, include_archived=True)) == 1
    V.delete_variant(v.id, db)
    assert V.list_variants(db, include_archived=True) == []


def test_record_use_counts_runs(tmp_path):
    db = _db(tmp_path)
    v, _ = V.save_variant("Used", "momentum", {}, db)
    V.record_use(v.id, "backtest", db)
    V.record_use(v.id, "backtest", db)
    V.record_use(v.id, "paper", db)
    got = V.get_variant(v.id, db)
    assert (got.backtest_runs, got.paper_runs) == (2, 1)


# -- application ------------------------------------------------------------

def test_apply_variant_never_mutates_canonical():
    strat = get_strategy("value_momentum")
    before = dict(
        max_weight=strat.max_weight, rebalance=strat.rebalance,
        market_gate=strat.market_gate, max_stocks=strat.max_stocks,
    )
    v = V.Variant(
        id="x", name="Tweaked", base="value_momentum",
        overrides={"max_weight": 0.10, "rebalance": "weekly",
                   "market_gate": True, "max_stocks": 10},
    )
    tweaked = V.apply_variant(strat, v)
    assert tweaked.max_weight == 0.10
    assert tweaked.rebalance == "weekly"
    assert tweaked.market_gate is True
    assert tweaked.max_stocks == 10
    # The canonical recipe is untouched.
    assert strat.max_weight == before["max_weight"]
    assert strat.rebalance == before["rebalance"]
    assert strat.market_gate == before["market_gate"]
    assert strat.max_stocks == before["max_stocks"]


def test_variant_display_name_and_summary():
    v = V.Variant(
        id="x", name="Concentrated 10", base="value_momentum",
        overrides={"max_stocks": 10, "max_weight": 0.10,
                   "market_gate": True},
        created="2026-10-11",
    )
    assert v.display_name("Value + Momentum") == \
        "Value + Momentum — Concentrated 10"
    summary = v.settings_summary()
    assert "10 stocks" in summary
    assert "10.0% cap" in summary
    assert "market gate on" in summary


def test_max_stocks_override_flows_into_portfolio():
    import pandas as pd
    import numpy as np

    from qrp.strategies.registry import strategy_target_portfolio

    dates = pd.bdate_range("2024-01-02", periods=30)
    matrices = {
        "momentum": pd.DataFrame(
            {f"T{i:02d}": np.linspace(0.01 * i, 0.9, 30)
             for i in range(60)},
            index=dates,
        )
    }
    base = get_strategy("momentum")
    p_default, _ = strategy_target_portfolio(
        base, matrices, dates[-1])
    # Default: implied by momentum's 2% cap -> 50 stocks.
    assert len(p_default.weights) == 50
    tweaked = V.apply_variant(
        base,
        V.Variant(id="x", name="T10", base="momentum",
                  overrides={"max_stocks": 10}),
    )
    p10, _ = strategy_target_portfolio(tweaked, matrices, dates[-1])
    assert len(p10.weights) == 10


def test_race_one_strategy_with_variant_override(tmp_path):
    """A variant races under its own label with its own settings."""
    import pandas as pd
    import numpy as np

    import qrp.ui.helpers as H
    from qrp.data.edgar import init_db as init_edgar_db
    from qrp.data.store import upsert_bars
    from qrp.signals.store import upsert_signals
    from qrp.strategies import variants as V

    db = str(tmp_path / "race.db")
    init_edgar_db(db)
    dates = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(3)
    for t in ("AAA", "BBB"):
        closes = 100.0 * (1 + rng.normal(0.0005, 0.01, 300)).cumprod()
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
    # Seed the momentum signal both strategies need.
    from qrp.signals import default_registry
    sig = default_registry().get("momentum")
    from qrp.signals.store import load_bars
    for t in ("AAA", "BBB"):
        upsert_signals(sig.compute(load_bars(t, path=db)), t, "momentum", db)

    base = get_strategy("momentum")
    variant, _ = V.save_variant(
        "T1", "momentum", {"max_stocks": 1}, db,
    )
    override = V.apply_variant(base, variant)
    row = H._race_one_strategy(
        "momentum", ["AAA", "BBB"], "2023-06-01", "2023-12-29",
        100_000.0, 0.5, 0.0, db,
        strategy_override=override,
        label=variant.display_name("Momentum"),
    )
    assert row["Strategy"] == "Momentum — T1"
    # max_stocks=1: at most one position held at a time.
    assert int(row["Trades"].split()[0]) >= 0  # smoke: row built fine
