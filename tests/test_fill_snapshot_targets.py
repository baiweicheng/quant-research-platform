"""Fill-check snapshot targets (2026-10-26): the snapshot must record
the EFFECTIVE targets the staged orders aimed at, not the raw strategy
weights.

The Oct 25 case: DELL sat just over the 5% cap, so cap-aware
generation trimmed it to 4.9% — but the snapshot recorded the strategy
weight (5.0%), and the fill check read the deliberate trim as a
shortfall and wanted 0.2214 shares back. Left alone, every week would
ping-pong: trim, top-up, trim again.

Note on the numbers: a 5.0% strategy weight trimmed to 4.9% leaves a
gap of exactly 2% of the target value — precisely the fill check's
drift boundary — so whether the old snapshot fired depended on float
noise. The robust reproducer below uses a strategy weight below the
cap (4.5% held at 5.2%, trimmed to 4.9%): against the old target the
fill check wants a 0.4pp sell, against the effective target it wants
nothing. The DELL-exact case is covered separately.

All synthetic — no network, no broker, no real money.
"""

from qrp.execution.alpaca import compute_fill_gaps, effective_target_weights

CAP = 0.05
TRIM = 0.049  # cap minus 0.1pp, matching _apply_position_cap


def snapshot(weights, equity, prices):
    return {
        "target_weights": weights,
        "reference_prices": prices,
        "equity": equity,
    }


# --- effective_target_weights unit behavior ---


def test_over_cap_name_gets_trim_landing():
    out = effective_target_weights(
        {"DELL": 0.045, "AAPL": 0.05},
        {"DELL": 52.0, "AAPL": 50.0},
        {"DELL": 100.0, "AAPL": 100.0},
        100_000.0, CAP,
    )
    assert out == {"DELL": TRIM, "AAPL": 0.05}


def test_at_cap_unchanged():
    out = effective_target_weights(
        {"DELL": 0.05}, {"DELL": 50.0}, {"DELL": 100.0}, 100_000.0, CAP,
    )
    assert out == {"DELL": 0.05}


def test_under_cap_unchanged():
    out = effective_target_weights(
        {"DELL": 0.05}, {"DELL": 48.0}, {"DELL": 100.0}, 100_000.0, CAP,
    )
    assert out == {"DELL": 0.05}


def test_missing_price_unchanged():
    out = effective_target_weights(
        {"DELL": 0.045}, {"DELL": 52.0}, {}, 100_000.0, CAP,
    )
    assert out == {"DELL": 0.045}


def test_no_positions_falls_back_to_strategy_weights():
    out = effective_target_weights(
        {"DELL": 0.045}, {}, {"DELL": 100.0}, 100_000.0, CAP,
    )
    assert out == {"DELL": 0.045}


def test_cap_none_falls_back():
    out = effective_target_weights(
        {"DELL": 0.045}, {"DELL": 52.0}, {"DELL": 100.0}, 100_000.0, None,
    )
    assert out == {"DELL": 0.045}


def test_held_name_with_no_strategy_weight_gains_none():
    # A full exit stays a full exit: no target is invented for it.
    out = effective_target_weights(
        {"AAPL": 0.05}, {"XYZ": 60.0}, {"XYZ": 100.0}, 100_000.0, CAP,
    )
    assert out == {"AAPL": 0.05}


def test_trim_target_floor_for_tiny_cap():
    out = effective_target_weights(
        {"DELL": 0.0004}, {"DELL": 60.0}, {"DELL": 100.0},
        100_000.0, 0.0005,
    )
    assert out == {"DELL": 0.0005}


# --- end-to-end through compute_fill_gaps ---


def test_oct25_style_trim_produces_no_buyback():
    # Strategy 4.5%, held 5.2% at staging, trimmed to 4.9% and filled.
    effective = effective_target_weights(
        {"DELL": 0.045}, {"DELL": 52.0}, {"DELL": 100.0}, 100_000.0, CAP,
    )
    assert effective == {"DELL": TRIM}
    gaps, _ = compute_fill_gaps(
        snapshot(effective, 100_000.0, {"DELL": 100.0}),
        {"DELL": 49.0},
    )
    assert gaps == []


def test_old_strategy_target_would_have_sold_the_trim():
    # The same account against the OLD (strategy-weight) snapshot:
    # the fill check wants to sell the trim back down to 4.5%.
    gaps, _ = compute_fill_gaps(
        snapshot({"DELL": 0.045}, 100_000.0, {"DELL": 100.0}),
        {"DELL": 49.0},
    )
    assert len(gaps) == 1
    assert gaps[0]["ticker"] == "DELL"
    assert gaps[0]["side"] == "SELL"
    assert gaps[0]["shares"] == 4.0


def test_dell_exact_case_no_gap():
    # Strategy 5.0%, trimmed to 4.9%, filled exactly at the landing.
    effective = effective_target_weights(
        {"DELL": 0.05}, {"DELL": 50.4}, {"DELL": 100.0}, 100_000.0, CAP,
    )
    assert effective == {"DELL": TRIM}
    gaps, _ = compute_fill_gaps(
        snapshot(effective, 100_000.0, {"DELL": 100.0}),
        {"DELL": 49.0},
    )
    assert gaps == []


def test_genuine_partial_fill_still_tops_up():
    # Held 4.0% against an effective 4.9% target: a real shortfall,
    # so the top-up fires with the "partly filled" reason intact.
    gaps, _ = compute_fill_gaps(
        snapshot({"DELL": TRIM}, 100_000.0, {"DELL": 100.0}),
        {"DELL": 40.0},
    )
    assert len(gaps) == 1
    assert gaps[0]["ticker"] == "DELL"
    assert gaps[0]["side"] == "BUY"
    assert gaps[0]["shares"] == 9.0
    assert gaps[0]["reason"] == "staged order only partly filled — topping up"


def test_in_cap_rebalance_identical_to_before():
    # Nothing over the cap: effective targets equal the strategy
    # weights, and the fill check behaves exactly as before.
    weights = {"AAPL": 0.05, "MSFT": 0.04}
    effective = effective_target_weights(
        weights, {"AAPL": 50.0, "MSFT": 40.0},
        {"AAPL": 100.0, "MSFT": 100.0}, 100_000.0, CAP,
    )
    assert effective == weights
    old_gaps, _ = compute_fill_gaps(
        snapshot(weights, 100_000.0, {"AAPL": 100.0, "MSFT": 100.0}),
        {"AAPL": 48.0, "MSFT": 40.0},
    )
    new_gaps, _ = compute_fill_gaps(
        snapshot(effective, 100_000.0, {"AAPL": 100.0, "MSFT": 100.0}),
        {"AAPL": 48.0, "MSFT": 40.0},
    )
    assert new_gaps == old_gaps
    assert len(new_gaps) == 1
    assert new_gaps[0]["ticker"] == "AAPL"
    assert new_gaps[0]["side"] == "BUY"
