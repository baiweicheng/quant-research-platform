"""Broker interface (Phase 5).

A ``BrokerClient`` is the seam between the platform and any brokerage — the
place where a real broker (e.g. Robinhood) would plug in later. The ONLY
implementation in this repo is the paper-trading broker
(``qrp.execution.paper``): simulated money, real math, zero connection to
any real brokerage.

By construction there is NO live-broker implementation here and no code path
that can place a real order. Wiring a real broker is a future phase and
requires the user's explicit approval first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrderResult:
    """What happened when you tried to place one order.

    filled: True if the order executed, False if it was rejected.
    ticker / action: what was attempted ("BUY" or "SELL").
    shares: how many shares were attempted (and filled, when filled).
    price: price per share — the execution price when filled.
    cost: trading fees paid in dollars (0 when the order was rejected).
    reason: plain-English explanation — "" when the order filled, otherwise
        why it was stopped (not enough cash, broke a risk rule, ...).
    """

    ticker: str
    action: str
    shares: float
    price: float
    cost: float
    filled: bool
    reason: str = ""


class BrokerClient(ABC):
    """Abstract connection to a brokerage — real or simulated."""

    @property
    @abstractmethod
    def cash(self) -> float:
        """Dollars sitting uninvested in the account."""
        ...

    @property
    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Ticker -> shares currently held."""
        ...

    @abstractmethod
    def last_price(self, ticker: str, date: str | None = None) -> float:
        """Last known close price for a ticker.

        Args:
            ticker: US stock symbol.
            date: YYYY-MM-DD trading day, or None for the latest day in the DB.

        Raises:
            KeyError: if no price is stored for that ticker/date.
        """
        ...

    def exec_price(self, ticker: str, date: str | None = None) -> float:
        """Price this broker would fill an order at on `date`.

        The default is the close (same as last_price). A broker with an
        execution setting — e.g. next-day-open — overrides this to return
        that day's open instead. Kept separate from last_price (which stays
        the close) because valuation and risk checks still mark to the
        close, while fills follow the execution convention.
        """
        return self.last_price(ticker, date)

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        date: str | None = None,
    ) -> OrderResult:
        """Attempt one market order.

        Args:
            ticker: US stock symbol.
            side: "BUY" or "SELL".
            quantity: number of shares (must be positive).
            date: YYYY-MM-DD trading day to execute on, or None for the
                latest day in the DB.

        Returns:
            OrderResult — filled with execution details, or rejected with a
            plain-English reason. A rejected order changes nothing.
        """
        ...


# ---------------------------------------------------------------------------
# Order-lifecycle interface (Phase P2)
# ---------------------------------------------------------------------------


class BrokerError(Exception):
    """A broker operation failed, explained in plain English.

    Never carries API keys, raw tracebacks, or blobs of broker JSON —
    only something safe to show the user.
    """


@dataclass
class AccountInfo:
    """A snapshot of a brokerage account, in plain numbers.

    equity: total account value (cash + positions), in dollars.
    cash: dollars sitting uninvested.
    buying_power: dollars available for new buy orders right now.
    """

    equity: float
    cash: float
    buying_power: float


@dataclass
class OrderInfo:
    """One order, however the broker underneath tracks it.

    id: the broker's order id (a string — Alpaca ids are UUIDs, the local
        broker uses its database row ids as strings).
    ticker / side: what is being traded ("BUY" or "SELL").
    shares: how many shares (full precision — Alpaca accepts fractions).
    status: simplified lifecycle — "open" (working at the broker), "filled",
        "canceled", "rejected", or "expired".
    filled_avg_price: average fill price once filled, else None.
    submitted_at: when the broker accepted it (broker's timestamp string),
        else None.
    reason: the plain-English note recorded when the order was staged
        (may be "" when the broker doesn't carry one).
    client_order_id: the deterministic idempotency id the platform
        submitted with ("" when the broker doesn't surface one, e.g. an
        order placed before P6). P6's duplicate protection matches on
        this; Alpaca dedupes server-side on it too.
    """

    id: str
    ticker: str
    side: str
    shares: float
    status: str
    filled_avg_price: float | None = None
    submitted_at: str | None = None
    reason: str = ""
    client_order_id: str = ""


class Broker(ABC):
    """Order-lifecycle interface — the shape real brokerages work in.

    Where ``BrokerClient`` (above) is the simulation-oriented interface
    (immediate fills, mark-to-market math, the local staging machinery),
    ``Broker`` mirrors how an actual brokerage API behaves: you submit an
    order, it sits open at the broker, and later it is filled, canceled,
    or rejected. Both the Alpaca adapter (``qrp.execution.alpaca``) and a
    thin wrapper around the local paper broker implement this, so the
    dashboard can drive either one the same way.

    All methods raise ``BrokerError`` with a plain-English message when the
    broker cannot answer — never a raw traceback or a blob of JSON.
    """

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Current account snapshot: equity, cash, buying power."""
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Ticker -> shares currently held (only nonzero positions)."""
        ...

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        side: str,
        shares: float,
        reason: str = "",
        *,
        client_order_id: str | None = None,
    ) -> OrderInfo:
        """Submit one order; it becomes "open" at the broker.

        For the Alpaca adapter this is a market order queued for the next
        market open (submitted at staging time, Alpaca holds it overnight).
        For the local wrapper it stages a pending order, exactly like the
        dashboard's "Stage orders" button.

        Args:
            ticker: US stock symbol.
            side: "BUY" or "SELL".
            shares: number of shares, must be positive (fractions allowed).
            reason: plain-English note for why this order exists.
            client_order_id: deterministic idempotency id (P6: e.g.
                "rebalance-20261007-AAPL-BUY"). When given, the broker
                submits with it so a retried batch cannot double-trade;
                when None, the adapter picks one as before.

        Returns:
            OrderInfo with status "open". (Orders the broker later refuses
            show up with status "rejected" via list_orders/get_order.)

        Raises:
            BrokerError with a plain-English explanation when the input is
            invalid or the broker refuses the order outright.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel one open order. Returns True when it existed and is now
        canceled; False when there was no such open order."""
        ...

    @abstractmethod
    def list_orders(self, status: str = "open") -> list[OrderInfo]:
        """Orders at the broker.

        Args:
            status: "open" (working orders), "filled", "canceled",
                "rejected", or "all".
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderInfo | None:
        """One order by id, or None when the broker has no such order."""
        ...
