"""Paper-trading broker (Phase 5).

A simulated brokerage account: real math, fake money. It behaves like a
broker — you hold cash and share positions, you place market orders — but
nothing leaves your computer and no real dollars move.

How it works, in plain terms:

- Orders fill at the next trading day's OPENING price by default (like
  real market-on-open orders: you run the numbers at night, the trades
  fill when the market opens). "close" execution — filling at the day's
  close — is kept as an option for the older convention.
- THE NIGHTLY WORKFLOW (no time travel): the dashboard stages a rebalance
  as PENDING orders — the decisions are made from data through the latest
  close, but nothing trades yet. The orders fill automatically at the next
  market open once new price data arrives (on page load, or via the
  "Refresh new data" button, or the explicit "Fill pending orders"
  button). Filling is idempotent: a refresh never fills the same order
  twice, and orders whose fill day never arrives simply stay pending.
- THE TIMING RULE (no peeking): a paper rebalance that trades at day D may
  only use signal values dated strictly before D. A signal dated D is only
  known after D's closing bell — too late to trade at D's open OR close.
  This is exactly the rule the backtest engine follows (it decides on date
  T and executes on the next trading day), so paper-trading results are
  directly comparable to backtest results. Use resolve_paper_timing() to
  get the (trading_day, signal_cutoff) pair; it raises a plain-English
  error when no honest trade is possible. Staging uses
  resolve_staging_timing(): the decision date is the latest close, and the
  fill day is always strictly after it.
- Trading costs follow the same approach as the backtest: the "spread"
  cost model estimates each trade's cost from the stock's recent price
  ranges, priced off the day BEFORE the fill for open execution (the fill
  day's range isn't known yet — the same no-peeking rule the backtest
  engine uses). "flat" keeps the fixed fee per trade (the default, so
  existing behavior never changes silently); the dashboard's paper trading
  uses "spread", matching the backtest defaults.
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

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qrp.backtest.costs import SpreadCostModel
from qrp.data.store import DEFAULT_DB, connect_db, load_dividends
from qrp.data.yahoo import normalize_ticker
from qrp.execution.broker import (
    AccountInfo,
    Broker,
    BrokerClient,
    BrokerError,
    OrderInfo,
    OrderResult,
)
from qrp.risk.guards import Order, PortfolioState, RiskConfig, check_order
from qrp.signals.store import load_bars

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
CREATE TABLE IF NOT EXISTS paper_dividends (
    ticker           TEXT NOT NULL,
    ex_date          TEXT NOT NULL,
    shares_held      REAL NOT NULL,
    amount_per_share REAL NOT NULL,
    amount_paid      REAL NOT NULL,
    credited_on      TEXT NOT NULL,
    PRIMARY KEY (ticker, ex_date)
);
CREATE TABLE IF NOT EXISTS paper_pending_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    staged_date TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,
    shares      REAL NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
"""


def _shares_held_on(conn, ticker: str, ex_date: str) -> float:
    """Shares of ticker held at the start of ex_date, by trade replay.

    Replays the paper trade log for the ticker up to (not including)
    ex_date: a stock bought ON the ex-date, or sold the day before,
    counts exactly like a real brokerage would — only whoever owned it
    at the prior close gets paid.
    """
    rows = conn.execute(
        "SELECT side, shares FROM paper_trades"
        " WHERE ticker = ? AND date < ? ORDER BY id",
        (ticker, ex_date),
    ).fetchall()
    held = 0.0
    for side, shares in rows:
        if side == "BUY":
            held += float(shares)
        elif side == "SELL":
            held -= float(shares)
    return max(held, 0.0)


def price_trading_days(db_path: str | Path = DEFAULT_DB) -> list[str]:
    """All YYYY-MM-DD dates with stored price data, oldest first.

    Returns [] when the database (or just the table) does not exist yet —
    a fresh install is a normal state, not an error.
    """
    path = Path(db_path)
    if not path.exists():
        return []
    with connect_db(path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='daily_bars'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return []
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

    - trading_day: the day the orders fill at — the day's OPENING price
      under the default "open" execution (like real market-on-open orders),
      or the day's close under "close" execution.
    - signal_cutoff: only signal values dated on or before this day are
      used — always strictly before trading_day.

    Why strictly before? A signal dated day D is only known after D's
    closing bell, when it is too late to trade at D's open or close. So a
    paper trade on D may only use information from before D — the same
    no-peeking rule the backtest engine enforces (it decides on date T and
    executes on the next trading day). This keeps paper-trading results
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


def resolve_staging_timing(
    db_path: str | Path = DEFAULT_DB,
) -> tuple[str, str]:
    """Decide WHEN a paper rebalance is decided for the nightly workflow.

    Returns (decision_date, signal_cutoff) — both the latest trading day
    with stored price data.

    The nightly workflow: you run the numbers after the closing bell, so
    the decision may use everything known through today's close (signals
    dated today included). The staged orders do NOT fill today — they wait
    as pending orders and fill at the next trading day's open once that
    day's prices exist, which is always strictly after the decision date,
    so the no-peeking rule holds exactly like the backtest (decide on T,
    execute on T+1).

    Raises ValueError with a plain-English explanation when the database
    holds no price data at all.
    """
    dates = price_trading_days(db_path)
    if not dates:
        raise ValueError(
            "No price data — run the data CLI first to ingest some tickers."
        )
    return dates[-1], dates[-1]


class PaperBroker(BrokerClient):
    """Simulated brokerage account backed by SQLite daily bars."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        starting_capital: float = 100_000.0,
        cost_bps: float = 10.0,
        risk_config: RiskConfig | None = None,
        execution: str = "open",
        cost_model: str = "flat",
    ) -> None:
        """Args:
        db_path: SQLite file (defaults to the project's data/market.db).
        starting_capital: play-money cash for a brand-new account.
        cost_bps: trading fee per trade in basis points (10 = 0.10%) —
            used only when cost_model is "flat".
        risk_config: safety knobs; defaults to the standard RiskConfig.
        execution: when orders fill — "open" (default: the trading day's
            opening price, like real market-on-open orders staged the
            night before) or "close" (the day's closing price, the older
            convention).
        cost_model: how each fill's cost is priced — "flat" (the default:
            the fixed cost_bps fee per trade) or "spread" (the realistic
            estimate from each stock's recent high-low price ranges, the
            same model and default the backtests use). With "spread", an
            open-execution fill is priced off the day BEFORE the fill —
            the fill day's range isn't known yet, exactly like the
            backtest engine.
        """
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if execution not in ("open", "close"):
            raise ValueError(
                f"unknown execution {execution!r} — use 'open' or 'close'"
            )
        if cost_model not in ("flat", "spread"):
            raise ValueError(
                f"unknown cost_model {cost_model!r} — use 'flat' or 'spread'"
            )
        self._db = Path(db_path)
        self._starting_capital = float(starting_capital)
        self.cost_bps = float(cost_bps)
        self.execution = execution
        self.cost_model = cost_model
        self.risk_config = risk_config or RiskConfig()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        with connect_db(self._db) as conn:
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
        with connect_db(self._db) as conn:
            (cash,) = conn.execute(
                "SELECT cash FROM paper_cash WHERE id = 1"
            ).fetchone()
        return float(cash)

    @property
    def positions(self) -> dict[str, float]:
        with connect_db(self._db) as conn:
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
        with connect_db(self._db) as conn:
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_trades")
            conn.execute("DELETE FROM paper_dividends")
            conn.execute("DELETE FROM paper_pending_orders")
            conn.execute("UPDATE paper_cash SET cash = ? WHERE id = 1", (amount,))
        return amount

    def trades(self, limit: int = 50) -> list[dict]:
        """Most recent paper trades, newest first (for review/audit)."""
        with connect_db(self._db) as conn:
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

    def dividends(self, limit: int = 50) -> list[dict]:
        """Dividend payouts credited to the account, newest first."""
        with connect_db(self._db) as conn:
            rows = conn.execute(
                "SELECT ex_date, ticker, shares_held, amount_per_share,"
                " amount_paid, credited_on FROM paper_dividends"
                " ORDER BY ex_date DESC, ticker LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ex_date": e,
                "ticker": t,
                "shares_held": float(sh),
                "amount_per_share": float(a),
                "amount_paid": float(p),
                "credited_on": c,
            }
            for e, t, sh, a, p, c in rows
        ]

    # -- pending orders (the nightly workflow) -------------------------------

    def stage_orders(
        self, orders: list[Order], staged_date: str
    ) -> dict[str, int]:
        """Stage rebalance orders as PENDING — decided now, filled later.

        In plain English: this is the "run the numbers at night" step.
        Nothing trades, cash and positions do not change. Each order waits
        until fill_pending_orders() finds a trading day after `staged_date`
        with a real opening price, then fills there at the open.

        Args:
            orders: the orders to stage (e.g. from
                qrp.execution.rebalance.orders_to_target), with full-precision
                share counts and plain-English reasons.
            staged_date: YYYY-MM-DD decision date — data through this
                day's close was used. Fills always happen strictly after it.

        Returns:
            {"staged": n, "skipped_duplicates": m} — an order for the same
            ticker, side and staged date that is already pending is skipped,
            so double-clicking "Stage orders" can never double-trade.

        Raises:
            ValueError with a plain-English explanation for an empty order
            list or an order with an unknown side / non-positive shares.
        """
        if not orders:
            raise ValueError(
                "Nothing to stage — the order list is empty (the account "
                "is already at its target)."
            )
        staged = 0
        skipped = 0
        with connect_db(self._db) as conn:
            for o in orders:
                ticker = normalize_ticker(o.ticker)
                action = o.action.strip().upper()
                shares = float(o.shares)
                if action not in ("BUY", "SELL"):
                    raise ValueError(
                        f"Cannot stage order for {ticker}: unknown side "
                        f"{o.action!r} — only BUY or SELL are allowed."
                    )
                if not shares > 0:
                    raise ValueError(
                        f"Cannot stage order for {ticker}: shares must be "
                        "positive."
                    )
                dup = conn.execute(
                    "SELECT 1 FROM paper_pending_orders"
                    " WHERE staged_date = ? AND ticker = ? AND side = ?",
                    (staged_date, ticker, action),
                ).fetchone()
                if dup is not None:
                    skipped += 1
                    continue
                conn.execute(
                    "INSERT INTO paper_pending_orders"
                    " (staged_date, ticker, side, shares, reason, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        staged_date,
                        ticker,
                        action,
                        shares,
                        o.reason or "",
                        datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    ),
                )
                staged += 1
        return {"staged": staged, "skipped_duplicates": skipped}

    def pending_orders(self) -> list[dict]:
        """Pending (staged, not yet filled) orders, oldest first."""
        with connect_db(self._db) as conn:
            rows = conn.execute(
                "SELECT id, staged_date, ticker, side, shares, reason,"
                " created_at FROM paper_pending_orders ORDER BY id"
            ).fetchall()
        return [
            {
                "id": i,
                "staged_date": d,
                "ticker": t,
                "side": s,
                "shares": float(sh),
                "reason": r or "",
                "created_at": c,
            }
            for i, d, t, s, sh, r, c in rows
        ]

    def cancel_pending_order(self, order_id: int) -> bool:
        """Cancel one pending order. Returns True when it existed.

        In plain English: "never mind that one" — the order is dropped
        before it ever trades, and cash/positions are untouched.
        """
        with connect_db(self._db) as conn:
            cur = conn.execute(
                "DELETE FROM paper_pending_orders WHERE id = ?",
                (int(order_id),),
            )
        return cur.rowcount > 0

    def _fill_date_for(self, ticker: str, staged_date: str) -> str | None:
        """First trading date after `staged_date` with an opening price.

        Returns None when no such date is stored yet (weekend, night, or
        fresh data not downloaded) — the order simply stays pending.
        """
        with connect_db(self._db) as conn:
            row = conn.execute(
                "SELECT MIN(date) FROM daily_bars"
                " WHERE ticker = ? AND date > ?"
                " AND open IS NOT NULL AND close IS NOT NULL",
                (normalize_ticker(ticker), staged_date),
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def fill_pending_orders(self) -> dict:
        """Fill every pending order whose next open has arrived.

        In plain English: the morning-after step. For each pending order,
        the first trading day after its staged date that has a real
        opening price becomes the fill day, and the order fills there at
        the open (or the close under "close" execution) — exactly like a
        real market-on-open order. Dividend cash for the fill day is
        credited BEFORE the fills, using only already-filled trades, so a
        buyer filled at the open ON an ex-date is not paid the dividend —
        the same overnight-holder rule real brokerages use.

        This is idempotent: each pending order is processed exactly once
        (filled or reported as unfillable, then removed), so refreshing
        twice can never double-fill. Orders with no fill day yet stay
        pending, untouched.

        Returns a dict with:
          - "filled": list of dicts (date, ticker, side, shares, price,
            cost) for orders that filled,
          - "skipped": list of dicts (ticker, side, reason) for orders
            that could not fill (e.g. no price that day, safety check),
          - "pending_remaining": how many orders are still pending,
          - "dividends_credited": dollars of dividend cash collected
            during the fill pass.
        """
        pendings = self.pending_orders()
        report: dict = {
            "filled": [],
            "skipped": [],
            "pending_remaining": 0,
            "dividends_credited": 0.0,
        }
        if not pendings:
            return report
        # Group fillable orders by their fill day, earliest first.
        by_day: dict[str, list[dict]] = {}
        for p in pendings:
            fill_day = self._fill_date_for(p["ticker"], p["staged_date"])
            if fill_day is None:
                continue
            by_day.setdefault(fill_day, []).append(p)
        for fill_day in sorted(by_day):
            # Dividends first: only overnight holders (already-filled
            # trades) are entitled — pending buyers are not in the trade
            # log yet, so the replay counts them out, like a real broker.
            try:
                credited = self.credit_dividends(asof=fill_day)
                report["dividends_credited"] += sum(
                    c["amount_paid"] for c in credited
                )
            except Exception:
                # Dividend data problems must never block the fills.
                pass
            # Sells before buys inside a fill day, like orders_to_target,
            # so sell proceeds fund the buys that follow.
            day_orders = sorted(
                by_day[fill_day],
                key=lambda p: (0 if p["side"] == "SELL" else 1, p["id"]),
            )
            for p in day_orders:
                try:
                    result = self.place_order(
                        p["ticker"],
                        p["side"],
                        p["shares"],
                        fill_day,
                        reason=p["reason"],
                    )
                except Exception as exc:  # never let one bad order stop the pass
                    result = None
                    fail_reason = (
                        f"Could not fill: {exc}. Nothing was traded."
                    )
                if result is not None and result.filled:
                    self.cancel_pending_order(p["id"])
                    report["filled"].append(
                        {
                            "date": fill_day,
                            "ticker": result.ticker,
                            "side": result.action,
                            "shares": result.shares,
                            "price": result.price,
                            "cost": result.cost,
                        }
                    )
                else:
                    self.cancel_pending_order(p["id"])
                    report["skipped"].append(
                        {
                            "ticker": p["ticker"],
                            "side": p["side"],
                            "reason": (
                                result.reason
                                if result is not None
                                else fail_reason
                            ),
                        }
                    )
        # Sweep any newer dividends the fill-day credits did not cover.
        try:
            credited = self.credit_dividends()
            report["dividends_credited"] += sum(
                c["amount_paid"] for c in credited
            )
        except Exception:
            pass
        report["pending_remaining"] = len(self.pending_orders())
        return report

    def credit_dividends(
        self, asof: str | None = None, tickers: list[str] | None = None
    ) -> list[dict]:
        """Pay the account cash for dividends whose ex-date has arrived.

        In plain English: when a company whose shares this account holds
        pays a dividend, the cash lands here — call this after downloading
        fresh market data so newly announced payouts get credited. Each
        (ticker, ex-date) is credited at most once; calling it again finds
        nothing new and changes nothing.

        How the share count is worked out: the trade log is replayed up to
        (not including) the ex-date, so a stock bought ON the ex-date — or
        sold the day before — is handled exactly like a real brokerage:
        only whoever owned it at the prior close gets paid.

        Args:
            asof: credit dividends with ex_date on or before this
                YYYY-MM-DD (default: the latest trading day with price
                data).
            tickers: only these tickers (default: every ticker with a
                stored dividend on or before `asof`).

        Returns:
            The payouts credited on this call (newest ex-date first), each
            a dict with ex_date, ticker, shares_held, amount_per_share and
            amount_paid.
        """
        if asof is None:
            asof = self._latest_trading_day()
        if tickers is None:
            # Default: every ticker that has a stored dividend at all.
            with connect_db(self._db) as conn:
                has = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table'"
                        " AND name='dividends'"
                    ).fetchone()
                    is not None
                )
                tickers = (
                    [r[0] for r in conn.execute(
                        "SELECT DISTINCT ticker FROM dividends"
                    ).fetchall()]
                    if has
                    else []
                )
            divs = load_dividends(tickers, end=asof, path=self._db)
        else:
            divs = load_dividends(
                [normalize_ticker(t) for t in tickers], end=asof, path=self._db
            )
        if divs.empty:
            return []
        credited: list[dict] = []
        today = self._latest_trading_day()
        with connect_db(self._db) as conn:
            for row in divs.itertuples():
                ticker = normalize_ticker(row.ticker)
                ex_date = row.ex_date.strftime("%Y-%m-%d")
                already = conn.execute(
                    "SELECT 1 FROM paper_dividends WHERE ticker = ?"
                    " AND ex_date = ?",
                    (ticker, ex_date),
                ).fetchone()
                if already is not None:
                    continue
                held = _shares_held_on(conn, ticker, ex_date)
                paid = held * float(row.amount)
                conn.execute(
                    "INSERT INTO paper_dividends"
                    " (ticker, ex_date, shares_held, amount_per_share,"
                    " amount_paid, credited_on)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        ticker,
                        ex_date,
                        held,
                        float(row.amount),
                        paid,
                        today,
                    ),
                )
                if paid > 0:
                    conn.execute(
                        "UPDATE paper_cash SET cash = cash + ? WHERE id = 1",
                        (paid,),
                    )
                    credited.append(
                        {
                            "ex_date": ex_date,
                            "ticker": ticker,
                            "shares_held": held,
                            "amount_per_share": float(row.amount),
                            "amount_paid": paid,
                        }
                    )
        return credited

    # -- prices ------------------------------------------------------------

    def _latest_trading_day(self) -> str:
        with connect_db(self._db) as conn:
            row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
        if not row or not row[0]:
            raise KeyError("no price data in daily_bars — run the data CLI first")
        return str(row[0])

    def last_price(self, ticker: str, date: str | None = None) -> float:
        ticker = normalize_ticker(ticker)
        day = date or self._latest_trading_day()
        with connect_db(self._db) as conn:
            row = conn.execute(
                "SELECT close FROM daily_bars WHERE ticker = ? AND date = ?",
                (ticker, day),
            ).fetchone()
        if not row:
            raise KeyError(f"no stored price for {ticker} on {day}")
        return float(row[0])

    def exec_price(self, ticker: str, date: str | None = None) -> float:
        """Price an order would fill at on `date` under this account's
        execution setting: the day's open for "open" execution (the
        default — like real market-on-open orders), the day's close for
        "close" execution. Valuation (last_price) always stays on the
        close; only fills follow the execution convention."""
        ticker = normalize_ticker(ticker)
        day = date or self._latest_trading_day()
        column = "open" if self.execution == "open" else "close"
        with connect_db(self._db) as conn:
            row = conn.execute(
                f"SELECT {column} FROM daily_bars WHERE ticker = ? AND date = ?",
                (ticker, day),
            ).fetchone()
        if not row or row[0] is None:
            raise KeyError(f"no stored {column} price for {ticker} on {day}")
        return float(row[0])

    # -- orders ------------------------------------------------------------

    def _fee(self, notional: float) -> float:
        """Trading fee in dollars for a trade worth `notional` dollars."""
        return notional * self.cost_bps / 10_000.0

    def _trading_day_before(self, day: str) -> str | None:
        """Latest stored trading day strictly before `day`, if any."""
        with connect_db(self._db) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE date < ?", (day,)
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def _trade_cost(self, ticker: str, day: str, notional: float) -> float:
        """Cost in dollars for one fill of `notional` dollars on `day`.

        "flat" charges the fixed cost_bps fee. "spread" estimates the
        realistic cost from the stock's recent high-low ranges via
        SpreadCostModel — priced off the day BEFORE the fill for open
        execution, because the fill day's range isn't known when the
        order is placed (the same no-peeking rule the backtest engine
        uses); for close execution the fill day itself is used, also
        exactly like the backtest.
        """
        if self.cost_model == "flat" or notional <= 0:
            return self._fee(notional)
        if self.execution == "open":
            prior = self._trading_day_before(day)
            cost_asof = prior if prior is not None else day
        else:
            cost_asof = day
        bars = load_bars(ticker, end=cost_asof, path=self._db)
        if bars.empty:
            return notional * SpreadCostModel.FALLBACK_RATE
        model = SpreadCostModel({normalize_ticker(ticker): bars})
        rate = model.rate(
            normalize_ticker(ticker), pd.Timestamp(cost_asof), notional
        )
        return notional * rate

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
            price = self.exec_price(ticker, day)
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
        fee = self._trade_cost(ticker, day, notional)
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

        with connect_db(self._db) as conn:
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


class PaperBrokerAdapter(Broker):
    """The local simulated broker, speaking the order-lifecycle interface.

    Why a wrapper instead of new methods on PaperBroker: PaperBroker
    already defines ``place_order()`` with immediate-fill semantics (the
    ``BrokerClient`` contract used by the backtest/rebalance machinery).
    The new ``Broker.place_order()`` means "queue for the next open" — a
    different contract — so the wrapper translates without changing any
    existing behavior.
    """

    def __init__(self, broker: PaperBroker) -> None:
        self._broker = broker

    # -- Broker interface ------------------------------------------------

    def get_account(self) -> AccountInfo:
        cash = self._broker.cash
        equity = cash
        for ticker, shares in self._broker.positions.items():
            try:
                equity += shares * self._broker.last_price(ticker)
            except KeyError:
                # No stored price (delisted with no history): value it at 0,
                # same as the dashboard's account table.
                pass
        # No margin in the simulation: buying power is just cash.
        return AccountInfo(equity=equity, cash=cash, buying_power=cash)

    def get_positions(self) -> dict[str, float]:
        return dict(self._broker.positions)

    def place_order(
        self,
        ticker: str,
        side: str,
        shares: float,
        reason: str = "",
    ) -> OrderInfo:
        """Stage one order as pending (the nightly workflow).

        The order waits like any staged order and fills at the next market
        open once new price data arrives — use ``fill_pending_orders()`` on
        the wrapped broker (the dashboard does this on every visit).
        """
        t = normalize_ticker(ticker)
        s = (side or "").strip().upper()
        if s not in ("BUY", "SELL"):
            raise BrokerError(
                f"Cannot stage an order for {t}: side must be BUY or SELL, "
                f"not {side!r}."
            )
        try:
            qty = float(shares)
        except (TypeError, ValueError):
            qty = 0.0
        if not qty > 0:
            raise BrokerError(
                f"Cannot stage an order for {t}: shares must be positive."
            )
        try:
            price = self._broker.last_price(t)
        except KeyError:
            raise BrokerError(
                f"Cannot stage an order for {t}: no stored price data for "
                "it — download prices first (Market data page)."
            )
        days = price_trading_days(self._broker._db)
        staged_date = days[-1] if days else datetime.now(timezone.utc).strftime(
            "%Y-%m-%d"
        )
        try:
            result = self._broker.stage_orders(
                [Order(ticker=t, action=s, shares=qty, price=price,
                       reason=reason or "")],
                staged_date=staged_date,
            )
        except ValueError as exc:
            raise BrokerError(str(exc))
        if result["staged"] == 0:
            raise BrokerError(
                f"An identical {s} order for {t} is already pending — "
                "skipped so nothing is double-staged."
            )
        pending = self._broker.pending_orders()
        match = next(
            (
                p for p in pending
                if p["ticker"] == t and p["side"] == s
                and p["staged_date"] == staged_date
            ),
            None,
        )
        if match is None:  # pragma: no cover - defensive
            raise BrokerError(
                f"The {s} order for {t} was staged but could not be found "
                "afterwards — please try again."
            )
        return self._pending_to_order_info(match)

    def cancel_order(self, order_id: str) -> bool:
        try:
            oid = int(order_id)
        except (TypeError, ValueError):
            return False
        return self._broker.cancel_pending_order(oid)

    def list_orders(self, status: str = "open") -> list[OrderInfo]:
        status = (status or "open").lower()
        if status == "open":
            return [self._pending_to_order_info(p)
                    for p in self._broker.pending_orders()]
        if status == "all":
            return self.list_orders("open") + self.list_orders("filled")
        if status == "filled":
            return [self._trade_to_order_info(t)
                    for t in self._recent_trades_with_ids(200)]
        if status in ("canceled", "rejected", "expired"):
            # The local broker deletes canceled pendings outright and never
            # rejects — nothing to list.
            return []
        raise BrokerError(
            f"Unknown order status {status!r} — use open, filled, canceled, "
            "rejected, expired, or all."
        )

    def get_order(self, order_id: str) -> OrderInfo | None:
        for info in self.list_orders("open"):
            if info.id == str(order_id):
                return info
        for info in self.list_orders("filled"):
            if info.id == str(order_id):
                return info
        return None

    # -- translation ------------------------------------------------------

    @staticmethod
    def _pending_to_order_info(p: dict) -> OrderInfo:
        return OrderInfo(
            id=str(p["id"]),
            ticker=p["ticker"],
            side=p["side"],
            shares=float(p["shares"]),
            status="open",
            submitted_at=p.get("created_at"),
            reason=p.get("reason", "") or "",
        )

    @staticmethod
    def _trade_to_order_info(t: dict) -> OrderInfo:
        return OrderInfo(
            id=str(t["id"]),
            ticker=t["ticker"],
            side=t["side"],
            shares=float(t["shares"]),
            status="filled",
            filled_avg_price=float(t["price"]),
            submitted_at=t["date"],
            reason=t.get("reason", "") or "",
        )

    def _recent_trades_with_ids(self, limit: int) -> list[dict]:
        with connect_db(self._broker._db) as conn:
            rows = conn.execute(
                "SELECT id, date, ticker, side, shares, price, cost, reason"
                " FROM paper_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": i, "date": d, "ticker": t, "side": s,
                "shares": float(sh), "price": float(p),
                "cost": float(c), "reason": r or "",
            }
            for i, d, t, s, sh, p, c, r in rows
        ]
