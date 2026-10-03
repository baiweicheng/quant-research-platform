"""Paper-trading broker (Phase 5).

A simulated brokerage account: real math, fake money. It behaves like a
broker — you hold cash and share positions, you place market orders — but
nothing leaves your computer and no real dollars move.

How it works, in plain terms:

- Orders execute at the trading day's closing price (a simplification: real
  markets move between the moment you decide and the moment you trade).
- THE TIMING RULE (no peeking): a paper rebalance that trades at day D's
  close may only use signal values dated strictly before D. A signal dated
  D is only known after D's closing bell — too late to trade at D's close.
  This is exactly the rule the backtest engine follows (it decides on date
  T and executes at the next day's close), so paper-trading results are
  directly comparable to backtest results. Use resolve_paper_timing() to
  get the (trading_day, signal_cutoff) pair; it raises a plain-English
  error when no honest trade is possible.
- EVERY order first passes through ``qrp.risk.guards.check_order``. A
  rejected order returns a failed ``OrderResult`` with a plain-English
  reason and changes nothing — the same seatbelts as everywhere else.
- Each trade pays ``cost_bps`` basis points in fees (default 10 = 0.10%),
  so costs are never ignored.
- Cash can never go negative: sells always execute before buys, and a buy
  is rejected if the shares plus fees would cost more than the cash on hand.
- The account is stored in SQLite (tables ``paper_cash``, ``paper_positions``,
  ``paper_trades``) so it survives across runs. A fresh account starts with
  $100,000 of play money (configurable).

This is the ONLY broker in the repo. There is no live-broker code here.
"""

from __future__ import annotations

from pathlib import Path

import sqlite3

from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.execution.broker import BrokerClient, OrderResult
from qrp.risk.guards import Order, PortfolioState, RiskConfig, check_order

SCHEMA_PAPER = """
CREATE TABLE IF NOT EXISTS paper_cash (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_positions (
    ticker TEXT PRIMARY KEY,
    shares REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    date   TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side   TEXT NOT NULL,
    shares REAL NOT NULL,
    price  REAL NOT NULL,
    cost   REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
"""


def price_trading_days(db_path: str | Path = DEFAULT_DB) -> list[str]:
    """All YYYY-MM-DD dates with stored price data, oldest first."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM daily_bars ORDER BY date"
        ).fetchall()
    return [str(r[0]) for r in rows]


def resolve_paper_timing(
    db_path: str | Path = DEFAULT_DB,
    asof: str | None = None,
    *,
    asof_is_trading_day: bool = False,
) -> tuple[str, str]:
    """Decide WHEN a paper rebalance trades and WHICH signals it may use.

    Returns (trading_day, signal_cutoff):

    - trading_day: the day whose closing prices the orders execute at.
    - signal_cutoff: only signal values dated on or before this day are
      used — always strictly before trading_day.

    Why strictly before? A signal dated day D is only known after D's
    closing bell, when it is too late to trade at D's close. So a paper
    trade at D's close may only use information from before D — the same
    no-peeking rule the backtest engine enforces (it decides on date T and
    executes at the next day's close). This keeps paper-trading results
    directly comparable to backtest results.

    Args:
        db_path: SQLite file with daily bars.
        asof: optional YYYY-MM-DD. With asof_is_trading_day=False (the
            default, used by the dashboard) it is a *signal cutoff* date
            and trading_day becomes the first price date after it — handy
            for "what would I have done back then" replays. With
            asof_is_trading_day=True (used by the CLI's --date) it is the
            *trading day* itself and the cutoff becomes the trading day
            before it.
        asof_is_trading_day: how to interpret asof (see above).

    With asof=None: trading_day is the latest price date in the database
    and signal_cutoff is the trading day before it.

    Raises ValueError with a plain-English explanation when no honest trade
    is possible — e.g. the database holds fewer than two trading days, or
    the requested date leaves no later (or earlier) trading day available.
    """
    dates = price_trading_days(db_path)
    if not dates:
        raise ValueError(
            "No price data — run the data CLI first to ingest some tickers."
        )
    if asof is None:
        if len(dates) < 2:
            raise ValueError(
                "Only one trading day of prices is stored — a paper trade "
                "needs at least two, because it trades at the latest close "
                "using only earlier signals. Ingest more history first."
            )
        return dates[-1], dates[-2]
    day = asof.strip()
    if asof_is_trading_day:
        if day not in dates:
            raise ValueError(
                f"No price data stored for {day} — pick a trading day "
                f"between {dates[0]} and {dates[-1]}."
            )
        earlier = [d for d in dates if d < day]
        if not earlier:
            raise ValueError(
                f"{day} is the earliest stored trading day — there is no "
                "earlier day for the signals to come from. Ingest more "
                "history first."
            )
        return day, earlier[-1]
    later = [d for d in dates if d > day]
    if not later:
        raise ValueError(
            f"To trade on signals as of {day}, the next trading day's "
            f"prices are needed — but the database only has prices through "
            f"{dates[-1]}. Ingest fresh prices first, or clear the date to "
            "use the latest available trading day."
        )
    return later[0], day


class PaperBroker(BrokerClient):
    """Simulated brokerage account backed by SQLite daily bars."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        starting_capital: float = 100_000.0,
        cost_bps: float = 10.0,
        risk_config: RiskConfig | None = None,
    ) -> None:
        """Args:
        db_path: SQLite file (defaults to the project's data/market.db).
        starting_capital: play-money cash for a brand-new account.
        cost_bps: trading fee per trade in basis points (10 = 0.10%).
        risk_config: safety knobs; defaults to the standard RiskConfig.
        """
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        self._db = Path(db_path)
        self._starting_capital = float(starting_capital)
        self.cost_bps = float(cost_bps)
        self.risk_config = risk_config or RiskConfig()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db) as conn:
            conn.executescript(SCHEMA_PAPER)
            # Migration: older databases were created before the `reason`
            # column existed. Add it without touching any stored trades —
            # old rows simply get the empty default.
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(paper_trades)")
            }
            if "reason" not in cols:
                conn.execute(
                    "ALTER TABLE paper_trades ADD COLUMN reason"
                    " TEXT NOT NULL DEFAULT ''"
                )
            (n,) = conn.execute("SELECT COUNT(*) FROM paper_cash").fetchone()
            if n == 0:
                conn.execute(
                    "INSERT INTO paper_cash (id, cash) VALUES (1, ?)",
                    (self._starting_capital,),
                )

    # -- account state ----------------------------------------------------

    @property
    def cash(self) -> float:
        with sqlite3.connect(self._db) as conn:
            (cash,) = conn.execute(
                "SELECT cash FROM paper_cash WHERE id = 1"
            ).fetchone()
        return float(cash)

    @property
    def positions(self) -> dict[str, float]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT ticker, shares FROM paper_positions WHERE shares > 0"
            ).fetchall()
        return {t: float(s) for t, s in rows}

    def reset(self, capital: float | None = None) -> float:
        """Wipe the paper account and start over with fresh play money.

        Args:
            capital: new starting cash (default: the starting_capital from
                construction).

        Returns:
            The new cash balance.
        """
        amount = self._starting_capital if capital is None else float(capital)
        if amount <= 0:
            raise ValueError("reset capital must be positive")
        with sqlite3.connect(self._db) as conn:
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_trades")
            conn.execute("UPDATE paper_cash SET cash = ? WHERE id = 1", (amount,))
        return amount

    def trades(self, limit: int = 50) -> list[dict]:
        """Most recent paper trades, newest first (for review/audit)."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT date, ticker, side, shares, price, cost, reason"
                " FROM paper_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "date": d,
                "ticker": t,
                "side": s,
                "shares": float(sh),
                "price": float(p),
                "cost": float(c),
                "reason": r or "",
            }
            for d, t, s, sh, p, c, r in rows
        ]

    # -- prices ------------------------------------------------------------

    def _latest_trading_day(self) -> str:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
        if not row or not row[0]:
            raise KeyError("no price data in daily_bars — run the data CLI first")
        return str(row[0])

    def last_price(self, ticker: str, date: str | None = None) -> float:
        ticker = normalize_ticker(ticker)
        day = date or self._latest_trading_day()
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT close FROM daily_bars WHERE ticker = ? AND date = ?",
                (ticker, day),
            ).fetchone()
        if not row:
            raise KeyError(f"no stored price for {ticker} on {day}")
        return float(row[0])

    # -- orders ------------------------------------------------------------

    def _fee(self, notional: float) -> float:
        """Trading fee in dollars for a trade worth `notional` dollars."""
        return notional * self.cost_bps / 10_000.0

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        date: str | None = None,
        *,
        reason: str | None = None,
    ) -> OrderResult:
        ticker = normalize_ticker(ticker)
        action = side.strip().upper()
        day = date or self._latest_trading_day()
        # The plain-English "why" stored alongside the trade. Callers that
        # know the reason (e.g. orders_to_target orders carry one on the
        # Order) pass it in; a manual one-off order gets a clear default.
        stored_reason = (
            reason if reason is not None
            else "Manual order — no reason was recorded."
        )

        def rejected(reason: str) -> OrderResult:
            return OrderResult(
                ticker=ticker,
                action=action,
                shares=quantity,
                price=0.0,
                cost=0.0,
                filled=False,
                reason=reason,
            )

        if action not in ("BUY", "SELL"):
            return rejected(
                f"Blocked: unknown order side {side!r} — only BUY or SELL are allowed."
            )
        if not quantity > 0:
            return rejected("Blocked: an order must be for a positive number of shares.")
        try:
            price = self.last_price(ticker, day)
        except KeyError as exc:
            return rejected(f"Blocked: {exc}.")

        positions = self.positions
        # Held positions with no price on `day` (delisted/halted) are simply
        # left out of the valuation — an order for a different ticker must
        # not crash because of them.
        prices = {}
        for t in set(positions) | {ticker}:
            try:
                prices[t] = self.last_price(t, day)
            except KeyError:
                continue
        state = PortfolioState(cash=self.cash, positions=positions, prices=prices)
        ok, reason = check_order(
            Order(ticker=ticker, action=action, shares=quantity, price=price),
            state,
            self.risk_config,
        )
        if not ok:
            return rejected(reason)

        notional = quantity * price
        fee = self._fee(notional)
        if action == "BUY":
            # check_order ignores fees; enforce them here so cash never dips
            # below zero once the real (fee-inclusive) total is charged.
            if notional + fee > state.cash + 1e-9:
                return rejected(
                    f"Blocked: this buy costs ${notional:,.2f} plus "
                    f"${fee:,.2f} in fees, but you only have "
                    f"${state.cash:,.2f} in cash. The order was stopped so "
                    "you never spend money you do not have."
                )
            new_cash = state.cash - notional - fee
            new_shares = positions.get(ticker, 0.0) + quantity
        else:
            new_cash = state.cash + notional - fee
            new_shares = positions.get(ticker, 0.0) - quantity

        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "UPDATE paper_cash SET cash = ? WHERE id = 1", (new_cash,)
            )
            if new_shares > 1e-9:
                conn.execute(
                    "INSERT OR REPLACE INTO paper_positions (ticker, shares)"
                    " VALUES (?, ?)",
                    (ticker, new_shares),
                )
            else:
                conn.execute(
                    "DELETE FROM paper_positions WHERE ticker = ?", (ticker,)
                )
            conn.execute(
                "INSERT INTO paper_trades (date, ticker, side, shares, price, cost, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (day, ticker, action, quantity, price, fee, stored_reason),
            )
        return OrderResult(
            ticker=ticker,
            action=action,
            shares=quantity,
            price=price,
            cost=fee,
            filled=True,
        )
