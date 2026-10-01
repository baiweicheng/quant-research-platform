"""Turn a target portfolio into a concrete order list (Phase 5).

A "target portfolio" says what fraction of your money each stock should be
(e.g. AAPL 20%). This module answers the follow-up: "what do I actually
buy and sell today to get there from what I currently hold?"

Rules, in plain terms:

- For each stock, compare what you hold to what the target says you should
  hold. The difference becomes one BUY or one SELL order.
- Stocks you hold but that are NOT in the target are sold completely
  (the target says 0%, so you move to 0%).
- SELLS always come before BUYS in the returned list. This matters because
  sell proceeds land in cash first, so the buys that follow can afford to
  use that cash — cash never goes negative mid-rebalance.
- Tiny differences under $1 ("dust") are skipped, so the account does not
  churn on meaningless $0.30 adjustments.
- Buys are shrunk proportionally when trading fees would push them over the
  available cash, so every returned order is actually executable.
- Prices used are the day's closing prices from the local database.
"""

from __future__ import annotations

from qrp.execution.broker import BrokerClient
from qrp.risk.guards import Order

# Differences smaller than this (in dollars) are ignored to avoid churn.
DUST_DOLLARS = 1.0


def orders_to_target(
    broker: BrokerClient,
    target_weights: dict[str, float],
    date: str,
    capital: float,
    cost_bps: float = 0.0,
) -> list[Order]:
    """Diff current holdings against a target portfolio.

    Args:
        broker: the account to read current holdings from.
        target_weights: ticker -> target weight as a fraction of capital
            (from qrp.portfolio.construction.build_target_weights).
        date: YYYY-MM-DD trading day whose closing prices set order prices.
        capital: total dollars the weights are fractions of (usually the
            account's current total value: cash + positions).
        cost_bps: trading fee in basis points, used to shrink buys so the
            final order list fits the cash actually available once fees are
            included (a target of exactly 100% invested would otherwise
            overshoot by the fee amount).

    Returns:
        List of Orders, sells first then buys. Each order carries the
        day's closing price. An empty list means "already at target".
        Buys are scaled down proportionally when fees would push them over
        the available cash — every returned order is executable.
    """
    target = {t.strip().upper(): float(w) for t, w in target_weights.items()}
    positions = broker.positions
    closes = {
        t: broker.last_price(t, date) for t in set(target) | set(positions)
    }

    # Size positions off capital net of expected trading fees. Fees shrink
    # the account slightly, so a target sitting exactly at a risk cap (e.g.
    # 20.0%) would otherwise land a hair over it and get blocked by the
    # guard. This overestimates fees a touch — the safe direction; any
    # leftover simply stays as cash.
    fee_rate = cost_bps / 10_000.0
    effective_capital = capital * (1 - fee_rate) if cost_bps > 0 else capital

    sells: list[Order] = []
    for ticker, held in positions.items():
        target_value = target.get(ticker, 0.0) * effective_capital
        diff = target_value - held * closes[ticker]
        if diff < -DUST_DOLLARS:
            shares = min(held, -diff / closes[ticker])
            if shares > 0:
                sells.append(
                    Order(ticker=ticker, action="SELL", shares=shares, price=closes[ticker])
                )
    buys: list[Order] = []
    for ticker, weight in target.items():
        held = positions.get(ticker, 0.0)
        diff = weight * effective_capital - held * closes[ticker]
        if diff > DUST_DOLLARS:
            buys.append(
                Order(
                    ticker=ticker,
                    action="BUY",
                    shares=diff / closes[ticker],
                    price=closes[ticker],
                )
            )

    # Shrink buys (keeping their relative sizes) so they fit the cash that
    # will actually be there after the sells, fees included.
    if buys and cost_bps > 0:
        available = broker.cash + sum(
            o.shares * o.price * (1 - fee_rate) for o in sells
        )
        needed = sum(o.shares * o.price * (1 + fee_rate) for o in buys)
        if needed > available + 1e-9 and needed > 0:
            scale = available / needed
            buys = [
                Order(ticker=o.ticker, action=o.action,
                      shares=o.shares * scale, price=o.price)
                for o in buys
                if o.shares * scale * o.price > DUST_DOLLARS
            ]
    return sells + buys
