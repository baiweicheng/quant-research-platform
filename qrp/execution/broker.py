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
