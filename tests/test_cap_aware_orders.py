"""Cap-aware order generation (2026-10-25): a rebalance staging was
blocked by the Position limits checklist because DELL sat just over
the 5% single-stock cap, even though no order in the batch involved
DELL. The target diff had nothing to say about it, so the batch the
checklist saw still breached the cap.

The fix: order generation itself enforces the cap when a caller
passes max_position_weight (the staging paths pass the same effective
cap the checklist enforces):

- a holding already over the cap gets a trim sell down to just under
  it (cap minus 0.1pp — 4.9% for a 5% cap), labeled as a cap trim, even
  when the regular target diff is tiny or absent;
- a buy that would land over the cap is shrunk to land exactly at it;
- the checklist message names the trim a breacher that is NOT in the
  batch needs, instead of telling the user to "shrink the orders".

All synthetic — no network, no real money. Order-generation tests use
a throwaway PaperBroker database; checklist tests use an in-memory
fake broker, mirroring tests/test_ops.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qrp.data.quality import DataHealthReport
from qrp.data.store import upsert_bars
from qrp.execution.broker import AccountInfo, Broker
from qrp.execution.paper import PaperBroker
from qrp.execution.rebalance import orders_to_target
from qrp.ops import pretrade as OPS_PRE
from qrp.risk.guards import RiskConfig

DAY = "2024-01-08"  # a Monday
PRICES = {"AAA": 100.0, "BBB": 50.0, "DELL": 150.0}


def seed_bars(path) -> None:
    """Five trading days of flat prices for AAA, BBB, and DELL."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    for ticker, base in PRICES.items():
        df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": base,
                "High": base * 1.01,
                "Low": base * 0.99,
                "Close": base,
                "Volume": 1_000,
            }
        )
        upsert_bars(df, ticker, path)


def make_broker(path, holdings: dict[str, float]) -> PaperBroker:
    """PaperBroker holding `holdings` (ticker -> shares) at flat prices.

    Permissive risk caps on purpose: establishing an over-cap holding
    is the scenario under test, and the per-order guard would
    otherwise refuse to build it.
    """
    broker = PaperBroker(
        path,
        starting_capital=100_000.0,
        cost_bps=0.0,
        risk_config=RiskConfig(max_position_weight=1.0, max_positions=10),
    )
    for ticker, shares in holdings.items():
        result = broker.place_order(ticker, "BUY", shares, DAY)
        assert result.filled, result.reason
    return broker


def capital_of(broker: PaperBroker) -> float:
    return broker.cash + sum(
        shares * PRICES[t] for t, shares in broker.positions.items()
    )


def post_weight(broker, orders, ticker: str, capital: float) -> float:
    """Weight of `ticker` after `orders`, valued at the flat price."""
    shares = broker.positions.get(ticker, 0.0)
    for o in orders:
        if o.ticker == ticker:
            shares += o.shares if o.action == "BUY" else -o.shares
    return shares * PRICES[ticker] / capital


def test_small_overcap_drift_gets_trim_to_just_under_cap(tmp_path):
    """The reported DELL shape: held just over the cap, target at cap.

    Held 5.02% against a 5% target — the regular diff is a $20 sell
    (0.13 shares). Without cap awareness that is all the batch would
    carry, landing exactly on the cap with no headroom. With the cap
    passed, the sell is topped up to land at 4.9% and labeled.
    """
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = make_broker(db, {"DELL": 5_020.0 / 150.0})
    capital = capital_of(broker)
    assert capital == pytest.approx(100_000.0)

    orders = orders_to_target(
        broker, {"DELL": 0.05}, DAY, capital, cost_bps=0.0,
        max_position_weight=0.05,
    )
    sells = [o for o in orders if o.ticker == "DELL"]
    assert len(sells) == 1
    assert sells[0].action == "SELL"
    # (5,020 - 4,900) / 150 shares lands the position at 4.9%.
    assert sells[0].shares == pytest.approx(0.8)
    assert "cap trim" in sells[0].reason.lower()
    assert post_weight(broker, orders, "DELL", capital) == pytest.approx(
        0.049
    )

    # Old behavior preserved when no cap is passed: just the $20 diff.
    plain = orders_to_target(
        broker, {"DELL": 0.05}, DAY, capital, cost_bps=0.0,
    )
    assert len(plain) == 1
    assert plain[0].shares == pytest.approx(5_020.0 / 150.0 - 5_000.0 / 150.0)


def test_untouched_overcap_holding_gets_trim_alongside_other_sells(tmp_path):
    """Sells elsewhere in the batch do not excuse an over-cap holding.

    DELL sits at 5.3% with its target kept at its current weight (no
    regular diff at all) while BBB exits completely. The batch must
    still carry a DELL trim — an untouched holding is exactly what the
    checklist used to block on.
    """
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = make_broker(db, {"DELL": 5_300.0 / 150.0, "BBB": 100.0})
    capital = capital_of(broker)

    orders = orders_to_target(
        broker, {"DELL": 0.053}, DAY, capital, cost_bps=0.0,
        max_position_weight=0.05,
    )
    by_key = {(o.ticker, o.action): o for o in orders}
    assert ("BBB", "SELL") in by_key  # the full exit, as before
    assert ("DELL", "SELL") in by_key  # the new cap trim
    assert "cap trim" in by_key[("DELL", "SELL")].reason.lower()
    assert post_weight(broker, orders, "DELL", capital) == pytest.approx(
        0.049
    )


def test_buy_shrunk_to_land_exactly_at_cap(tmp_path):
    """A buy that would land over the cap is sized down to the cap."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = make_broker(db, {})
    capital = capital_of(broker)

    orders = orders_to_target(
        broker, {"AAA": 0.06}, DAY, capital, cost_bps=0.0,
        max_position_weight=0.05,
    )
    assert len(orders) == 1
    assert orders[0].action == "BUY"
    # 5% of $100,000 at $100/share — never a buy AND a trim sell.
    assert orders[0].shares == pytest.approx(50.0)
    assert "single-stock cap" in orders[0].reason
    assert post_weight(broker, orders, "AAA", capital) == pytest.approx(
        0.05
    )


def test_buy_of_overcap_holding_becomes_trim_not_more_buying(tmp_path):
    """Buying more of an already over-cap holding is never the answer."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = make_broker(db, {"DELL": 5_300.0 / 150.0})
    capital = capital_of(broker)

    orders = orders_to_target(
        broker, {"DELL": 0.06}, DAY, capital, cost_bps=0.0,
        max_position_weight=0.05,
    )
    assert all(o.action == "SELL" for o in orders)
    assert len([o for o in orders if o.ticker == "DELL"]) == 1
    assert post_weight(broker, orders, "DELL", capital) == pytest.approx(
        0.049
    )


def test_in_cap_rebalance_unchanged_by_cap_parameter(tmp_path):
    """Passing the cap must not disturb a batch that already fits."""
    db = tmp_path / "t.db"
    seed_bars(db)
    broker = make_broker(db, {"AAA": 40.0, "BBB": 40.0})
    capital = capital_of(broker)
    target = {"AAA": 0.045, "BBB": 0.025}

    plain = orders_to_target(broker, target, DAY, capital, cost_bps=0.0)
    capped = orders_to_target(
        broker, target, DAY, capital, cost_bps=0.0,
        max_position_weight=0.05,
    )
    assert [(o.ticker, o.action, o.shares, o.reason) for o in capped] == [
        (o.ticker, o.action, o.shares, o.reason) for o in plain
    ]


# ---------------------------------------------------------------------------
# Sendable layer: a cap trim must survive whole-share rounding
# ---------------------------------------------------------------------------


def test_sendable_cap_trim_survives_whole_share_rounding():
    """A sub-share cap-trim sell for a non-fractionable name rounds UP
    to one whole share instead of vanishing — vanishing would leave
    the position over the cap and the checklist blocked again. A
    plain sub-share sell still skips, as before."""
    from types import SimpleNamespace

    from qrp.ui.app import _alpaca_sendable_orders

    trim = SimpleNamespace(
        ticker="DELL", action="SELL", shares=0.2, price=150.0,
        reason="Cap trim: sold DELL down from 5.3% to 4.9% of the "
        "portfolio — it was over the 5% single-stock cap.",
    )
    plain = SimpleNamespace(
        ticker="AAA", action="SELL", shares=0.2, price=100.0,
        reason="Trimmed AAA: reduced from 4.0% to 3.9% of the portfolio",
    )
    sendable, notes = _alpaca_sendable_orders(
        [trim, plain],
        is_fractionable=lambda t: False,
        positions={"DELL": 10.0, "AAA": 10.0},
    )
    assert [(s["ticker"], s["shares"]) for s in sendable] == [
        ("DELL", 1.0)
    ]
    assert any("cap-trim" in n for n in notes)
    assert any("AAA" in n and "skipped" in n for n in notes)


# ---------------------------------------------------------------------------
# Checklist message wording
# ---------------------------------------------------------------------------


class FakeBroker(Broker):
    """Minimal in-memory Broker for the checklist wording tests."""

    def __init__(self, *, cash, positions, prices):
        self._cash = float(cash)
        self._positions = dict(positions)
        self._prices = dict(prices)

    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            q * self._prices.get(t, 0.0) for t, q in self._positions.items()
        )
        return AccountInfo(equity=equity, cash=self._cash,
                           buying_power=self._cash)

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def list_orders(self, status: str = "open"):
        return []

    def last_price(self, ticker: str, date=None) -> float:
        if ticker not in self._prices:
            raise KeyError(ticker)
        return float(self._prices[ticker])

    def place_order(self, *args, **kwargs):
        raise NotImplementedError

    def cancel_order(self, order_id):
        raise NotImplementedError

    def get_order(self, order_id):
        raise NotImplementedError


def _fresh_report() -> DataHealthReport:
    return DataHealthReport(
        checked_at="2026-10-07T00:00:00Z", tickers_checked=2, findings=[]
    )


def _position_detail(report, name="Position limits") -> str:
    return next(c.detail for c in report.checks if c.name == name)


def test_checklist_names_trim_for_breacher_not_in_batch(tmp_path):
    """The misleading wording, fixed: DELL over cap, no DELL order."""
    db = tmp_path / "m.db"
    broker = FakeBroker(
        cash=94_700.0,
        positions={"DELL": 5_300.0 / 150.0},
        prices={"DELL": 150.0, "BBB": 50.0},
    )
    orders = [{"ticker": "BBB", "side": "BUY", "shares": 10.0,
               "price": 50.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date="2026-10-07",
        health_report=_fresh_report(), audit_log=False,
    )
    detail = _position_detail(report)
    assert "DELL" in detail
    assert "not in this batch" in detail
    assert "4.9%" in detail
    assert "Shrink the orders" not in detail


def test_checklist_keeps_old_wording_for_breacher_in_batch(tmp_path):
    """A breacher that IS being bought keeps the shrink-the-order text."""
    db = tmp_path / "m.db"
    broker = FakeBroker(
        cash=94_700.0,
        positions={"DELL": 5_300.0 / 150.0},
        prices={"DELL": 150.0},
    )
    orders = [{"ticker": "DELL", "side": "BUY", "shares": 10.0,
               "price": 150.0, "reason": ""}]
    report = OPS_PRE.run_pretrade_checks(
        orders, broker, db_path=db, decision_date="2026-10-07",
        health_report=_fresh_report(), audit_log=False,
    )
    detail = _position_detail(report)
    assert "Shrink the orders" in detail
    assert "not in this batch" not in detail
