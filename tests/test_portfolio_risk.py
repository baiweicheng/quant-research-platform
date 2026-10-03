"""Tests for Phase 4: portfolio construction + risk safeguards.

All synthetic — no network, no database. The invariants under test:

- every weighting method: weights >= 0, sum <= 1, no position above
  max_weight, at most max_positions positions;
- check_order refuses dangerous orders with a plain-English reason;
- the drawdown guard trips exactly when it should.
"""

import pandas as pd
import pytest

from qrp.portfolio.construction import build_target_weights
from qrp.risk.guards import (
    Order,
    PortfolioState,
    RiskConfig,
    check_order,
    check_portfolio,
    drawdown_guard_triggered,
)


def _scores(n=30, negative=False):
    """Deterministic fake scores: 'AAA' highest, descending."""
    scores = {}
    for i in range(n):
        ticker = f"S{i:02d}"
        scores[ticker] = (n - i) * 0.01 * (-1 if negative else 1)
    return scores


@pytest.mark.parametrize("method", ["equal_weight", "signal_weight"])
def test_weighting_invariants_equal_and_signal(method):
    tp = build_target_weights(
        _scores(30), method=method, max_weight=0.2, max_positions=20
    )
    assert all(w >= 0 for w in tp.weights.values())
    assert sum(tp.weights.values()) <= 1.0 + 1e-9
    assert all(w <= 0.2 + 1e-9 for w in tp.weights.values())
    assert len(tp.weights) <= 20
    assert tp.cash_weight >= -1e-9
    assert abs(sum(tp.weights.values()) + tp.cash_weight - 1.0) < 1e-9


def test_inverse_vol_invariants():
    scores = _scores(30)
    vols = {t: 0.01 + 0.001 * i for i, t in enumerate(scores)}
    tp = build_target_weights(
        scores, method="inverse_volatility", max_weight=0.2, max_positions=20,
        volatilities=vols,
    )
    assert all(w >= 0 for w in tp.weights.values())
    assert sum(tp.weights.values()) <= 1.0 + 1e-9
    assert all(w <= 0.2 + 1e-9 for w in tp.weights.values())
    # Calmer stock (lower vol) gets the bigger weight.
    assert tp.weights["S00"] > tp.weights["S19"]


def test_inverse_vol_needs_volatilities():
    with pytest.raises(ValueError, match="volatilities"):
        build_target_weights(_scores(5), method="inverse_volatility")


def test_single_extreme_signal_still_capped():
    scores = {"AAA": 100.0, "BBB": 0.1, "CCC": 0.05, "DDD": 0.01}
    tp = build_target_weights(scores, method="signal_weight", max_weight=0.2)
    assert tp.weights["AAA"] <= 0.2 + 1e-9
    assert sum(tp.weights.values()) <= 1.0 + 1e-9


def test_all_negative_scores_stays_in_cash():
    tp = build_target_weights(_scores(10, negative=True), method="signal_weight")
    assert tp.weights == {}
    assert tp.cash_weight == 1.0


def test_empty_scores_stays_in_cash():
    tp = build_target_weights({})
    assert tp.weights == {}
    assert tp.cash_weight == 1.0


def test_max_positions_respected():
    tp = build_target_weights(
        _scores(50), method="equal_weight", max_weight=0.20, max_positions=7
    )
    assert len(tp.weights) == 7
    for w in tp.weights.values():
        assert abs(w - 1 / 7) < 1e-9


def test_dust_positions_dropped():
    # One dominant score + a long tail of tiny ones: after capping the big
    # one, the tail stocks are still below the dust threshold.
    scores = {"AAA": 10.0}
    scores.update({f"S{i:03d}": 0.0001 for i in range(200)})
    tp = build_target_weights(
        scores, method="signal_weight", max_weight=0.9, max_positions=201,
        min_weight=0.01,
    )
    assert all(w >= 0.01 - 1e-9 for w in tp.weights.values())
    assert any("dust" in n for n in tp.notes)


def _state():
    return PortfolioState(
        cash=50_000.0,
        positions={"AAPL": 100.0},
        prices={"AAPL": 200.0, "MSFT": 300.0},
    )


def test_check_order_rejects_bigger_than_cash():
    ok, reason = check_order(
        Order("MSFT", "BUY", 1000, 300.0), _state(), RiskConfig()
    )
    assert not ok
    assert reason  # non-empty, plain-English
    assert "cash" in reason.lower()


def test_check_order_rejects_outside_collar():
    ok, reason = check_order(
        Order("MSFT", "BUY", 10, 400.0), _state(), RiskConfig(price_collar=0.05)
    )
    assert not ok
    assert "5%" in reason or "stale" in reason.lower() or "typo" in reason.lower()


def test_check_order_rejects_oversize_position():
    # $200k of AAPL at $200 would be ~83% of the $240k portfolio.
    ok, reason = check_order(
        Order("AAPL", "BUY", 1000, 200.0),
        PortfolioState(cash=220_000.0, positions={}, prices={"AAPL": 200.0}),
        RiskConfig(max_position_weight=0.2),
    )
    assert not ok
    assert "20%" in reason


def test_check_order_rejects_sell_more_than_held():
    ok, reason = check_order(Order("AAPL", "SELL", 500, 200.0), _state(), RiskConfig())
    assert not ok
    assert "only hold" in reason.lower()


def test_check_order_rejects_too_many_positions():
    state = PortfolioState(
        cash=100_000.0,
        positions={f"S{i:02d}": 10.0 for i in range(20)},
        prices={f"S{i:02d}": 100.0 for i in range(20)} | {"NEW": 100.0},
    )
    ok, reason = check_order(
        Order("NEW", "BUY", 10, 100.0), state, RiskConfig(max_positions=20)
    )
    assert not ok
    assert "20" in reason


def test_check_order_accepts_sane_order():
    ok, reason = check_order(Order("MSFT", "BUY", 10, 300.0), _state(), RiskConfig())
    assert ok
    assert reason == ""


def test_check_order_halt_blocks_buys_not_sells():
    cfg = RiskConfig()
    buy_ok, _ = check_order(
        Order("MSFT", "BUY", 10, 300.0), _state(), cfg, halt_new_buys=True
    )
    sell_ok, _ = check_order(
        Order("AAPL", "SELL", 10, 200.0), _state(), cfg, halt_new_buys=True
    )
    assert not buy_ok
    assert sell_ok


def test_check_portfolio_flags_violations():
    ok, reasons = check_portfolio({"AAPL": 0.5, "MSFT": 0.6}, RiskConfig())
    assert not ok
    assert any("AAPL" in r for r in reasons)
    assert any("100%" in r for r in reasons)  # weights sum to 110%


def test_check_portfolio_accepts_good_portfolio():
    tp = build_target_weights(_scores(30), method="signal_weight")
    ok, reasons = check_portfolio(tp.weights, RiskConfig())
    assert ok
    assert reasons == []


def test_drawdown_guard_triggers_on_big_fall():
    # Rises to 120, then falls 20% to 96.
    values = pd.Series([100.0, 110, 120, 110, 100, 96.0])
    assert drawdown_guard_triggered(values, RiskConfig(max_drawdown_stop=0.15))
    assert not drawdown_guard_triggered(values, RiskConfig(max_drawdown_stop=0.25))


def test_drawdown_guard_ignores_mild_dip():
    values = pd.Series([100.0, 105, 103.0])  # ~1.9% dip
    assert not drawdown_guard_triggered(values, RiskConfig(max_drawdown_stop=0.15))


def test_drawdown_guard_disabled_with_none():
    values = pd.Series([100.0, 50.0])  # 50% crash
    assert not drawdown_guard_triggered(values, RiskConfig(max_drawdown_stop=None))


def test_drawdown_guard_accepts_backtest_frame():
    df = pd.DataFrame({"portfolio_value": [100.0, 120, 84.0]})  # 30% drawdown
    assert drawdown_guard_triggered(df, RiskConfig(max_drawdown_stop=0.15))


def test_implied_stock_count():
    from qrp.portfolio.construction import implied_stock_count

    assert implied_stock_count(0.20) == 5
    assert implied_stock_count(0.10) == 10
    assert implied_stock_count(0.05) == 20
    assert implied_stock_count(0.50) == 2
    assert implied_stock_count(1.0) == 1
    with pytest.raises(ValueError):
        implied_stock_count(0.0)
    with pytest.raises(ValueError):
        implied_stock_count(1.5)


def test_signal_weight_differentiates_under_cap():
    """Regression: with n == round(1/max_weight) the old normalize-then-cap
    flattened every position to exactly the cap. Scores must stay visible."""
    scores = {f"S{i:02d}": float(30 - i) for i in range(10)}
    tp = build_target_weights(
        scores, method="signal_weight", max_weight=0.10, max_positions=10
    )
    ws = tp.weights
    assert len(ws) == 10
    assert all(w <= 0.10 + 1e-9 for w in ws.values())
    assert sum(ws.values()) <= 1.0 + 1e-9
    # Strictly decreasing with score — not flattened.
    ordered = [ws[f"S{i:02d}"] for i in range(10)]
    assert all(a > b for a, b in zip(ordered, ordered[1:]))
    assert abs(ordered[0] - 0.10) < 1e-9, "best scorer gets the full cap"
    assert tp.cash_weight >= -1e-9
    assert abs(sum(ws.values()) + tp.cash_weight - 1.0) < 1e-9


def test_signal_weight_cash_note_when_concentrated():
    """Wildly skewed scores can't deploy everything under the cap — the
    leftover must stay in cash and say so."""
    tp = build_target_weights(
        {"AAA": 100.0, "BBB": 10.0}, method="signal_weight",
        max_weight=0.10, max_positions=10,
    )
    assert tp.weights["AAA"] == pytest.approx(0.10)
    assert tp.weights["BBB"] < tp.weights["AAA"]
    assert tp.cash_weight > 0.5
    assert any("cash" in n for n in tp.notes)


def test_collar_exempts_sells_gap_down_crash_exit():
    """A 10% gap-down SELL (crash exit) passes the collar — you can always
    get out, even far below the last close."""
    ok, reason = check_order(
        Order("AAPL", "SELL", 100, 180.0),  # last close $200, selling at $180
        _state(),
        RiskConfig(price_collar=0.05),
    )
    assert ok, f"sell should be exempt from the collar: {reason}"


def test_collar_still_blocks_gap_buys():
    """A 10% gap-up BUY is still blocked — the typo/stale-price guard stays
    for money going in."""
    ok, reason = check_order(
        Order("MSFT", "BUY", 10, 330.0),  # last close $300, buying at $330
        _state(),
        RiskConfig(price_collar=0.05),
    )
    assert not ok
    assert "5%" in reason or "stale" in reason.lower() or "typo" in reason.lower()


def test_collar_sell_at_last_close_still_checked_for_other_rules():
    """Exempting sells from the collar doesn't exempt them from everything:
    selling more than held is still blocked."""
    ok, reason = check_order(
        Order("AAPL", "SELL", 500, 100.0),  # far outside collar, but also too many
        _state(),
        RiskConfig(price_collar=0.05),
    )
    assert not ok
    assert "only hold" in reason
