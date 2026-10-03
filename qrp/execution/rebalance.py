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
from qrp.data.yahoo import normalize_ticker
from qrp.risk.guards import Order

# Differences smaller than this (in dollars) are ignored to avoid churn.
DUST_DOLLARS = 1.0


def _trade_reason(
    *,
    ticker: str,
    action: str,
    held: float,
    price: float,
    capital: float,
    target_weight: float,
    ranks: dict[str, int] | None,
) -> str:
    """Plain-English "why" for one rebalance order (Feature 5).

    held: shares currently owned; price: today's close; capital: total
    dollars the weights are fractions of; target_weight: the target's
    fraction of capital for this ticker (0.0 for a full exit); ranks:
    optional normalized ticker -> 1-based rank (1 = best), used only to
    name the stock's rank when known.
    """
    w = target_weight
    cw = held * price / capital if capital > 0 else 0.0
    rank = (ranks or {}).get(ticker)
    rank_bit = f", ranked #{rank}" if rank is not None else ""
    if action == "SELL":
        if target_weight <= 0:
            tail = f" — rank fell to #{rank}" if rank is not None else ""
            return f"Sold {ticker}: it left the target portfolio{tail}"
        return (
            f"Trimmed {ticker}: reduced from {cw:.1%} to {w:.1%} "
            f"of the portfolio{rank_bit}"
        )
    if held <= 0:
        return (
            f"Bought {ticker}: new position at {w:.1%} of the portfolio"
            f"{rank_bit}"
        )
    return (
        f"Bought more {ticker}: raised from {cw:.1%} to {w:.1%} "
        f"of the portfolio{rank_bit}"
    )


def priceable_closes(
    broker: BrokerClient, tickers: set[str], date: str
) -> tuple[dict[str, float], list[str]]:
    """Closing prices for the tickers that have a bar on `date`.

    Returns (closes, missing): ticker -> close for every ticker with a
    stored bar, plus the tickers that could not be priced (delisted,
    halted, or simply missing that day). Callers should skip missing
    tickers with a plain-English note instead of crashing.
    """
    closes: dict[str, float] = {}
    missing: list[str] = []
    for ticker in tickers:
        t = normalize_ticker(ticker)
        try:
            closes[t] = broker.last_price(t, date)
        except KeyError:
            missing.append(t)
    return closes, sorted(missing)


def orders_to_target(
    broker: BrokerClient,
    target_weights: dict[str, float],
    date: str,
    capital: float,
    cost_bps: float = 0.0,
    ranks: dict[str, int] | None = None,
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
        ranks: optional normalized ticker -> 1-based rank (1 = best).
            When given, each order's plain-English reason names the
            stock's rank; without it, reasons carry no rank mention.

    Returns:
        List of Orders, sells first then buys. Each order carries the
        day's closing price and a plain-English ``reason`` explaining why
        it was made (new position, adding, trimming, or full exit — see
        _trade_reason). An empty list means "already at target".
        Buys are scaled down proportionally when fees would push them over
        the available cash — every returned order is executable.
        Tickers with no stored bar on `date` (delisted, halted) are
        skipped: a position you cannot price is left untouched rather
        than guessed at.
    """
    target = {normalize_ticker(t): float(w) for t, w in target_weights.items()}
    positions = {normalize_ticker(t): float(s) for t, s in broker.positions.items()}
    closes, _missing = priceable_closes(
        broker, set(target) | set(positions), date
    )
    # Only trade what can be priced; unpriceable holdings stay as they are.
    target = {t: w for t, w in target.items() if t in closes}
    positions = {t: s for t, s in positions.items() if t in closes}

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
                    Order(
                        ticker=ticker,
                        action="SELL",
                        shares=shares,
                        price=closes[ticker],
                        reason=_trade_reason(
                            ticker=ticker,
                            action="SELL",
                            held=held,
                            price=closes[ticker],
                            capital=capital,
                            target_weight=target.get(ticker, 0.0),
                            ranks=ranks,
                        ),
                    )
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
                    reason=_trade_reason(
                        ticker=ticker,
                        action="BUY",
                        held=held,
                        price=closes[ticker],
                        capital=capital,
                        target_weight=weight,
                        ranks=ranks,
                    ),
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
                      shares=o.shares * scale, price=o.price,
                      reason=o.reason)
                for o in buys
                if o.shares * scale * o.price > DUST_DOLLARS
            ]
    return sells + buys
