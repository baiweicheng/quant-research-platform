"""Risk management safeguards (Phase 4).

Think of these as the seatbelts and airbags of the platform: automatic checks
that refuse to let a mistake — a typo'd price, an order bigger than your cash,
one stock taking over the whole portfolio — turn into real damage.

Three layers:

1. ``check_order`` — inspects a single BUY/SELL order *before* it goes out
   and rejects it with a plain-English reason if anything looks wrong.
2. ``check_portfolio`` — validates a whole target portfolio (the output of
   qrp.portfolio.construction) against the same caps.
3. ``drawdown_guard_triggered`` — watches the portfolio's value over time;
   if it has fallen more than ``max_drawdown_stop`` from its peak, new buying
   is halted until a human looks at it. (Selling still works — the guard
   never traps you in a falling position.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from qrp.data.yahoo import normalize_ticker


@dataclass
class RiskConfig:
    """The knobs for every safety check. Plain-English meaning of each:

    max_position_weight: the biggest slice any single stock may take of the
        whole portfolio (0.05 = 5%). Protects against one bad pick sinking you.
    max_positions: the most stocks you may hold at once. Keeps the portfolio
        focused instead of a hundred tiny bets.
    max_drawdown_stop: if the portfolio's value falls this far below its own
        peak (0.15 = 15%), halt *new buying* until a human reviews. Set to
        None to turn the guard off. Note what it does NOT do: it never sells
        anything by itself — it just stops new money from going in while
        things are falling.
    max_order_value: the most dollars any single order may cost. Catches
        "fat finger" orders (e.g. 1000 shares instead of 100). None = no cap.
    price_collar: a BUY order is rejected if its price is further than this
        from the stock's last known close price (0.05 = 5%). Catches typos
        and stale prices. SELLs are exempt on purpose — in a crash you must
        always be able to get out, even far below the last close.
    """

    max_position_weight: float = 0.05
    max_positions: int = 20
    max_drawdown_stop: float | None = 0.15
    max_order_value: float | None = None
    price_collar: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be between 0 and 1")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.max_drawdown_stop is not None and not 0 < self.max_drawdown_stop < 1:
            raise ValueError("max_drawdown_stop must be between 0 and 1 (or None)")
        if self.max_order_value is not None and self.max_order_value <= 0:
            raise ValueError("max_order_value must be positive (or None)")
        if not 0 <= self.price_collar < 1:
            raise ValueError("price_collar must be between 0 and 1")


@dataclass
class Order:
    """One BUY or SELL order waiting for a safety check.

    ticker: US stock symbol, e.g. "AAPL".
    action: "BUY" or "SELL" (case-insensitive).
    shares: how many shares (must be positive).
    price: the price per share you expect to trade at (must be positive).
    reason: plain-English explanation of WHY this trade was made (attached
        by orders_to_target when it knows; empty unless a caller set one).
    """

    ticker: str
    action: str
    shares: float
    price: float
    reason: str = ""


@dataclass
class PortfolioState:
    """What you currently own, needed to judge an order.

    cash: dollars sitting uninvested.
    positions: ticker -> shares currently held.
    prices: ticker -> last known close price per share.
    """

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    def total_value(self) -> float:
        """Cash plus the market value of every position."""
        return self.cash + sum(
            shares * self.prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )

    def position_count(self) -> int:
        """How many stocks are currently held (ignoring zero-share entries)."""
        return sum(1 for s in self.positions.values() if s > 0)


def check_order(
    order: Order,
    state: PortfolioState,
    config: RiskConfig,
    *,
    halt_new_buys: bool = False,
) -> tuple[bool, str]:
    """Inspect one order before it goes out.

    Args:
        order: the BUY/SELL order to check.
        state: what you currently own (cash, positions, last prices).
        config: the safety knobs.
        halt_new_buys: set True when the drawdown guard has tripped
            (see drawdown_guard_triggered) — new BUYs are then refused
            while SELLs still go through.

    Returns:
        (True, "") if the order passes every check, else (False, reason)
        with a plain-English explanation of what was wrong.
    """
    ticker = normalize_ticker(order.ticker)
    action = order.action.strip().upper()
    if action not in ("BUY", "SELL"):
        return False, (
            f"Blocked: unknown order action {order.action!r} — "
            "only BUY or SELL are allowed."
        )
    if not order.shares > 0:
        return False, "Blocked: an order must be for a positive number of shares."
    if not order.price > 0:
        return False, "Blocked: an order must have a positive price per share."

    held = state.positions.get(ticker, 0.0)
    if action == "SELL" and order.shares > held + 1e-9:
        return False, (
            f"Blocked: you are trying to sell {order.shares:g} shares of {ticker} "
            f"but you only hold {held:g}. This platform never lets you sell "
            "stock you do not own (no shorting)."
        )

    # Stale-price / typo collar — BUYs only. A sell must always be able to
    # get out: in a gap-down crash the exit price is far below the last
    # close, and blocking it would trap the position. (Buys keep the
    # collar because an overpriced buy just wastes cash on a typo or a
    # stale quote.)
    last = state.prices.get(ticker, 0.0)
    if action == "BUY" and last > 0:
        drift = abs(order.price - last) / last
        if drift > config.price_collar:
            return False, (
                f"Blocked: the {ticker} order price ${order.price:,.2f} is "
                f"{drift:.1%} away from its last close of ${last:,.2f} "
                f"(the limit is {config.price_collar:.0%}). This usually means "
                "a typo or a stale price, so the order was stopped."
            )

    if action == "BUY":
        if halt_new_buys:
            stop = config.max_drawdown_stop
            return False, (
                "Blocked: new buying is paused because the portfolio has fallen "
                f"more than {stop:.0%} from its peak (drawdown stop). You can "
                "still sell; buying resumes once you review and lift the halt."
            )
        value = order.shares * order.price
        if config.max_order_value is not None and value > config.max_order_value:
            return False, (
                f"Blocked: this buy would cost ${value:,.2f}, which is above "
                f"your per-order cap of ${config.max_order_value:,.2f}. Split "
                "it into smaller orders, or raise the cap if you really mean it."
            )
        if value > state.cash + 1e-9:
            return False, (
                f"Blocked: this buy costs ${value:,.2f} but you only have "
                f"${state.cash:,.2f} in cash. The order was stopped so you "
                "never spend money you do not have."
            )
        total = state.total_value()
        if total > 0:
            # Value the whole post-trade position at the order price; the
            # price collar above guarantees it is close to the last close.
            new_weight = (held + order.shares) * order.price / total
            if new_weight > config.max_position_weight + 1e-9:
                return False, (
                    f"Blocked: after this buy, {ticker} would be "
                    f"{new_weight:.1%} of your portfolio, above your "
                    f"{config.max_position_weight:.0%} single-stock cap. "
                    "Buy fewer shares to stay diversified."
                )
        if held <= 0 and state.position_count() >= config.max_positions:
            return False, (
                f"Blocked: you already hold {config.max_positions} stocks "
                "(your maximum). Sell something first if you want to add "
                f"{ticker}."
            )

    return True, ""


def check_portfolio(
    weights: dict[str, float], config: RiskConfig
) -> tuple[bool, list[str]]:
    """Validate a whole target portfolio against the safety caps.

    Args:
        weights: ticker -> target weight as a fraction of capital.
        config: the safety knobs.

    Returns:
        (True, []) if everything is fine, else (False, [reasons]) with one
        plain-English reason per problem found.
    """
    reasons: list[str] = []
    for ticker, w in weights.items():
        t = normalize_ticker(ticker)
        if w < -1e-9:
            reasons.append(
                f"{t}: weight is {w:.2%} (negative). Long-only portfolios "
                "cannot hold negative weights — that would be a short bet."
            )
        if w > config.max_position_weight + 1e-9:
            reasons.append(
                f"{t}: weight is {w:.1%}, above the "
                f"{config.max_position_weight:.0%} single-stock cap."
            )
    if len(weights) > config.max_positions:
        reasons.append(
            f"The portfolio holds {len(weights)} stocks, above the maximum "
            f"of {config.max_positions}."
        )
    total = sum(weights.values())
    if total > 1.0 + 1e-9:
        reasons.append(
            f"Weights add up to {total:.1%} — more than 100% of your money. "
            "That would mean borrowing to invest (leverage), which is not allowed."
        )
    return (len(reasons) == 0, reasons)


def drawdown_guard_triggered(
    equity: pd.Series | pd.DataFrame, config: RiskConfig
) -> bool:
    """Has the portfolio fallen far enough from its peak to halt new buying?

    A "drawdown" is how far the portfolio's value has dropped from its own
    highest point so far. If the current drawdown is bigger than
    ``max_drawdown_stop`` (e.g. the portfolio is down 20% from its peak and
    the stop is 15%), this returns True — a signal to pause new buying until
    a human takes a look.

    Args:
        equity: portfolio values over time — either a Series of values or a
            DataFrame with a "portfolio_value" column (like the backtest
            engine produces).
        config: the safety knobs (uses max_drawdown_stop; None = guard off).

    Returns:
        True when the guard trips, False otherwise.
    """
    stop = config.max_drawdown_stop
    if stop is None:
        return False
    values = (
        equity["portfolio_value"] if isinstance(equity, pd.DataFrame) else equity
    )
    values = pd.Series(values).dropna()
    if values.empty:
        return False
    peak = values.cummax()
    # Avoid divide-by-zero where the running peak is 0 (flat zero equity).
    drawdown = (peak - values) / peak.replace(0, float("nan"))
    current = drawdown.fillna(0.0).iloc[-1]
    return bool(current > stop)
