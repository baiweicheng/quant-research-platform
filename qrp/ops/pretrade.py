"""Pre-trade checklist (P6: ops hardening).

Runs BEFORE any order batch is staged or submitted — for both the Alpaca
paper flow and the local paper broker. Every check returns pass/fail/warn
with a plain-English reason; any failure blocks the batch (the orders
never reach the broker) and the reason is shown, never a traceback.

The seven checks:
1. Kill-switch state (fail-closed: an unreadable flag file blocks).
2. Data freshness — consumes P5's ``DataHealthReport``; stale data blocks
   (a rebalance on old data is worse than none).
3. Buying power: planned buys <= buying power x 0.98.
4. Position limits: 5% per-stock cap, no duplicate orders beyond target.
5. Fat-finger: one order > 3x its historical median size, or over an
   absolute dollar cap, or (buys only) priced > 10% away from the last
   close — any of these blocks.
6. Duplicate protection: deterministic ``client_order_id``
   (``rebalance-YYYYMMDD-TICKER-side``); no open order may carry that ID.
7. No-short / no-margin guardrails: long-only; a sell may never exceed
   what is held.

"Historical median size" (check 5) is the paper account's OWN order
history: ``broker.list_orders("all")`` — for Alpaca, the last 500 orders
(open and closed); for the local broker, pending plus filled orders.
Per (ticker, side) when at least 3 such orders exist, else the
account-wide median when at least 3 orders exist at all, else the check
warns that there is not enough history yet (it passes — blocking every
first-ever trade would make the platform unusable on day one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any

from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.execution.broker import Broker, BrokerError

# --- Documented constants (the checklist's knobs) ------------------------------

#: Planned buys may use at most this fraction of buying power — the 2%
#: buffer absorbs price moves between staging at night and the fill.
BUYING_POWER_FACTOR = 0.98

#: No single stock may exceed this weight of the account after the batch.
#: 5% is the platform's standing single-stock cap (set 2026-10-04) — a
#: per-stock ceiling, not a headcount.
MAX_POSITION_WEIGHT = 0.05

#: A single order bigger than this multiple of its historical median dollar
#: size (notional) is treated as a possible fat-finger and blocked. 3x is
#: deliberately tight: legitimate rebalances move a position by fractions,
#: never by multiples. Notional, not share count: 400 shares of a $3,000
#: stock and 400 shares of a $50 stock are worlds apart.
FAT_FINGER_MEDIAN_MULTIPLE = 3.0

#: Absolute dollar cap per order, whatever the history says. $25,000 is
#: about five times a typical full 5% position in the default $100k paper
#: account — far above any legitimate single order, far below "oops".
FAT_FINGER_MAX_ORDER_VALUE = 25_000.0

#: A BUY priced further than this from the last close is blocked (typo or
#: stale price). Sells are EXEMPT on purpose — same rationale as
#: qrp.risk.guards: in a gap-down you must always be able to get out, and
#: blocking an exit would trap the position.
FAT_FINGER_MAX_PRICE_DRIFT = 0.10

#: Minimum past orders needed before the median-size check engages.
MIN_HISTORY_FOR_MEDIAN = 3


# --- Result types -----------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One checklist verdict.

    status: "pass" | "fail" | "warn". detail is always plain English.
    """

    name: str
    status: str
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail}


@dataclass
class PretradeReport:
    """Everything the checklist run found."""

    checks: list[CheckResult] = field(default_factory=list)
    decision_date: str = ""

    @property
    def ok(self) -> bool:
        """True when the batch may proceed (no failures; warnings are OK)."""
        return all(c.status != "fail" for c in self.checks)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    def plain_english_summary(self) -> str:
        fails = self.failures()
        warns = self.warnings()
        if fails:
            return (
                f"Pre-trade checklist BLOCKED the batch: "
                f"{len(fails)} check(s) failed — "
                + "; ".join(f"{c.name}: {c.detail}" for c in fails)
            )
        if warns:
            return (
                f"Pre-trade checklist passed with {len(warns)} warning(s) — "
                + "; ".join(f"{c.name}: {c.detail}" for c in warns)
            )
        return (
            f"Pre-trade checklist passed: all {len(self.checks)} checks "
            "cleared."
        )

    def as_dict(self) -> dict:
        return {
            "decision_date": self.decision_date,
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
        }


# --- Order normalization -------------------------------------------------------------


def normalize_orders(orders: list[Any]) -> list[dict]:
    """Accept Order dataclasses or plain dicts; return uniform dicts with
    ticker/side/shares/price/reason."""
    out = []
    for o in orders:
        if isinstance(o, dict):
            ticker = o.get("ticker", "")
            side = o.get("side", o.get("action", ""))
            shares = o.get("shares", 0.0)
            price = o.get("price", 0.0)
            reason = o.get("reason", "")
        else:
            ticker = getattr(o, "ticker", "")
            side = getattr(o, "action", getattr(o, "side", ""))
            shares = getattr(o, "shares", 0.0)
            price = getattr(o, "price", 0.0)
            reason = getattr(o, "reason", "")
        out.append(
            {
                "ticker": normalize_ticker(str(ticker or "")),
                "side": str(side or "").strip().upper(),
                "shares": float(shares or 0.0),
                "price": float(price or 0.0),
                "reason": str(reason or ""),
            }
        )
    return out


def client_order_id(ticker: str, side: str, decision_date: str) -> str:
    """Deterministic order ID: ``rebalance-YYYYMMDD-TICKER-side``.

    Same inputs => same ID, so a retried staging run can never double-trade:
    the duplicate check (and Alpaca itself) recognizes the ID.
    """
    day = str(decision_date or "")[:10].replace("-", "")
    return (
        f"rebalance-{day}-{normalize_ticker(ticker)}-"
        f"{str(side).strip().upper()}"
    )


# --- The run ------------------------------------------------------------------------------


def run_pretrade_checks(
    orders: list[Any],
    broker: Broker,
    *,
    db_path: str | Path = DEFAULT_DB,
    decision_date: str | None = None,
    health_report: Any | None = None,
    audit_log: bool = True,
    actor: str = "dashboard",
) -> PretradeReport:
    """Run all seven checks against a batch of planned orders.

    Args:
        orders: Order dataclasses or dicts (ticker/side/shares/price).
        broker: the Broker the orders would go to (Alpaca paper or local).
        decision_date: YYYY-MM-DD the rebalance was decided (used for the
            deterministic order IDs); defaults to today.
        health_report: a P5 ``DataHealthReport``; when None, the local
            checks (stale + outlier, no network) are run fresh.
        audit_log: write a "pretrade_checks" audit event with the verdicts.
        actor: recorded in the audit event.

    Returns a PretradeReport. ``report.ok`` False => the batch must NOT be
    staged or submitted; the failures explain why in plain English.
    """
    from qrp.ops import audit as _audit

    norm = normalize_orders(orders)
    day = (decision_date or date.today().isoformat())[:10]
    report = PretradeReport(decision_date=day)

    ctx = _CheckContext(broker, db_path, day, norm)
    report.checks.append(_check_kill_switch(ctx))
    report.checks.append(_check_data_freshness(ctx, health_report))
    report.checks.append(_check_buying_power(ctx))
    report.checks.append(_check_position_limits(ctx))
    report.checks.append(_check_fat_finger(ctx))
    report.checks.append(_check_duplicates(ctx))
    report.checks.append(_check_no_short(ctx))

    if audit_log:
        _audit.log_event(
            "pretrade_checks",
            {
                "decision_date": day,
                "ok": report.ok,
                "orders": [
                    {
                        "ticker": o["ticker"],
                        "side": o["side"],
                        "shares": o["shares"],
                        "price": o["price"],
                        "client_order_id": client_order_id(
                            o["ticker"], o["side"], day
                        ),
                    }
                    for o in norm
                ],
                "checks": [c.as_dict() for c in report.checks],
            },
            actor=actor,
            db_path=db_path,
        )
    return report


class _CheckContext:
    """Broker/account snapshot shared by all checks (fetched once)."""

    def __init__(
        self, broker: Broker, db_path: str | Path, day: str,
        orders: list[dict],
    ) -> None:
        self.broker = broker
        self.db_path = db_path
        self.day = day
        self.orders = orders
        self.account = None
        self.account_error = ""
        self.positions: dict[str, float] = {}
        self.open_orders: list = []
        self.open_orders_error = ""
        self.history: list = []
        self.history_error = ""
        try:
            self.account = broker.get_account()
        except BrokerError as exc:
            self.account_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - never let this raise
            self.account_error = f"unexpected problem: {exc}"
        try:
            self.positions = dict(broker.get_positions() or {})
        except (BrokerError, Exception):  # noqa: BLE001 - best effort
            self.positions = {}
        try:
            self.open_orders = list(broker.list_orders("open") or [])
        except BrokerError as exc:
            self.open_orders_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - best effort
            self.open_orders_error = f"unexpected problem: {exc}"
        try:
            self.history = list(broker.list_orders("all") or [])
        except (BrokerError, Exception):  # noqa: BLE001 - best effort
            self.history = []
            self.history_error = "order history unavailable"
        self._prices: dict[str, float] = {}

    def last_price(self, ticker: str) -> float | None:
        if ticker not in self._prices:
            try:
                self._prices[ticker] = float(
                    self.broker.last_price(ticker)
                )
            except (KeyError, BrokerError, Exception):  # noqa: BLE001
                self._prices[ticker] = None  # type: ignore[assignment]
        return self._prices[ticker]


# --- Individual checks -------------------------------------------------------------------


def _check_kill_switch(ctx: _CheckContext) -> CheckResult:
    from qrp.ops import killswitch as _ks

    name = "Kill switch"
    try:
        engaged = _ks.is_engaged(ctx.db_path)
    except _ks.KillSwitchEngaged as exc:
        # Fail-closed: an unreadable flag file reads as STOPPED.
        return CheckResult(name, "fail", str(exc))
    if engaged:
        try:
            _ks.ensure_trading_allowed(ctx.db_path)
            why = "the kill switch is engaged"  # pragma: no cover
        except _ks.KillSwitchEngaged as exc:
            why = str(exc)
        return CheckResult(
            name, "fail",
            f"{why} Release it on the Operations page (or with "
            "'python -m qrp.ops.killswitch --release') before staging.",
        )
    return CheckResult(name, "pass", "Kill switch is off.")


def _check_data_freshness(
    ctx: _CheckContext, health_report: Any | None
) -> CheckResult:
    from qrp.data import quality as _quality

    name = "Data freshness"
    report = health_report
    if report is None:
        try:
            # Local checks only (stale + outlier) — no network, fast enough
            # to run on every staging. The full suite with network
            # spot-checks lives on the Data health page.
            report = _quality.run_data_health_checks(
                ctx.db_path, network_sample=0
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed
            return CheckResult(
                name, "fail",
                "Could not run the data-health checks "
                f"({exc}). Staging is blocked until the data can be "
                "verified — a rebalance on unknown data is worse than none.",
            )
    stale = [
        f for f in (report.findings or [])
        if f.check == "stale" and f.severity in ("warning", "error")
    ]
    if stale:
        first = stale[0]
        extra = f" (+{len(stale) - 1} more)" if len(stale) > 1 else ""
        action = f" {first.action}" if first.action else ""
        return CheckResult(
            name, "fail",
            f"Price data is stale: {first.summary}{extra}.{action} "
            "Staging is blocked — a rebalance on old data is worse than "
            "no rebalance.",
        )
    no_db = any(
        f.check == "general" and "No database yet" in f.summary
        for f in (report.findings or [])
    )
    if no_db:
        return CheckResult(
            name, "fail",
            "There is no market data stored yet — download prices on the "
            "Market data page first.",
        )
    errors = [
        f for f in (report.findings or [])
        if f.severity == "error"
    ]
    if errors:
        return CheckResult(
            name, "warn",
            f"The data-health check found {len(errors)} problem(s) that "
            "need attention (see the Data health page), but prices are "
            "fresh enough to stage.",
        )
    return CheckResult(name, "pass", "Price data is fresh.")


def _check_buying_power(ctx: _CheckContext) -> CheckResult:
    name = "Buying power"
    if ctx.account is None:
        return CheckResult(
            name, "fail",
            "Could not read the account, so buying power cannot be "
            f"verified ({ctx.account_error or 'unknown error'}). Staging "
            "is blocked until the broker answers.",
        )
    planned = sum(
        o["shares"] * o["price"] for o in ctx.orders if o["side"] == "BUY"
    )
    limit = ctx.account.buying_power * BUYING_POWER_FACTOR
    if planned > limit + 1e-9:
        return CheckResult(
            name, "fail",
            f"Planned buys cost ${planned:,.2f} but only "
            f"${limit:,.2f} may be used (buying power "
            f"${ctx.account.buying_power:,.2f} x {BUYING_POWER_FACTOR:.0%} "
            "safety factor). The batch was stopped so you never spend "
            "money you do not have.",
        )
    if not any(o["side"] == "BUY" for o in ctx.orders):
        return CheckResult(name, "pass", "No buys planned — nothing to fund.")
    return CheckResult(
        name, "pass",
        f"Planned buys (${planned:,.2f}) fit within "
        f"${limit:,.2f} of buying power.",
    )


def _check_position_limits(ctx: _CheckContext) -> CheckResult:
    name = "Position limits"
    if ctx.account is None:
        return CheckResult(
            name, "fail",
            "Could not read the account, so post-trade weights cannot be "
            "checked. Staging is blocked until the broker answers.",
        )
    equity = ctx.account.equity
    if not equity > 0:
        return CheckResult(
            name, "fail",
            "The account has no value to weight positions against — "
            "staging is blocked.",
        )
    seen: set[tuple[str, str]] = set()
    dupes: list[str] = []
    for o in ctx.orders:
        key = (o["ticker"], o["side"])
        if key in seen:
            dupes.append(f"{o['side']} {o['ticker']}")
        seen.add(key)
    if dupes:
        return CheckResult(
            name, "fail",
            "Duplicate orders in this batch (beyond target): "
            + ", ".join(sorted(set(dupes)))
            + ". One order per stock per side — remove the extras.",
        )
    # Post-trade shares per ticker.
    shares_after = dict(ctx.positions)
    for o in ctx.orders:
        delta = o["shares"] if o["side"] == "BUY" else -o["shares"]
        shares_after[o["ticker"]] = shares_after.get(o["ticker"], 0.0) + delta
    breaches = []
    for ticker, shares in shares_after.items():
        if shares <= 0:
            continue
        price = None
        for o in ctx.orders:
            if o["ticker"] == ticker and o["price"] > 0:
                price = o["price"]
                break
        if price is None:
            price = ctx.last_price(ticker)
        if price is None or price <= 0:
            continue  # unpriceable holding: valued at 0 elsewhere; skip
        weight = shares * price / equity
        if weight > MAX_POSITION_WEIGHT + 1e-9:
            breaches.append(f"{ticker} ({weight:.1%})")
    if breaches:
        return CheckResult(
            name, "fail",
            "These stocks would exceed the "
            f"{MAX_POSITION_WEIGHT:.0%} single-stock cap after this batch: "
            + ", ".join(sorted(breaches))
            + ". Shrink the orders to stay diversified.",
        )
    return CheckResult(
        name, "pass",
        f"Every stock stays at or under {MAX_POSITION_WEIGHT:.0%} of the "
        "account; no duplicates.",
    )


def _order_history_notionals(ctx: _CheckContext) -> tuple[dict, list[float]]:
    """(notionals per (ticker, side), all notionals) from order history.

    Dollar notional (shares x reference price), NOT raw share counts: 400
    shares of a $3,000 stock and 400 shares of a $50 stock are worlds
    apart, so comparing share counts across tickers is meaningless. The
    reference price is the order's fill price when known, else the stock's
    current last price. Orders that cannot be priced are skipped.
    """
    per: dict[tuple[str, str], list[float]] = {}
    all_sizes: list[float] = []
    for o in ctx.history:
        ticker = normalize_ticker(str(getattr(o, "ticker", "") or ""))
        side = str(getattr(o, "side", "") or "").upper()
        try:
            shares = float(getattr(o, "shares", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not ticker or side not in ("BUY", "SELL") or not shares > 0:
            continue
        price = getattr(o, "filled_avg_price", None)
        try:
            price = float(price) if price is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        if not price > 0:
            price = ctx.last_price(ticker) or 0.0
        if not price > 0:
            continue
        notional = shares * price
        per.setdefault((ticker, side), []).append(notional)
        all_sizes.append(notional)
    return per, all_sizes


def _check_fat_finger(ctx: _CheckContext) -> CheckResult:
    name = "Fat-finger guard"
    problems: list[str] = []
    per, all_sizes = _order_history_notionals(ctx)
    for o in ctx.orders:
        ticker, side, shares = o["ticker"], o["side"], o["shares"]
        value = shares * o["price"]
        if value > FAT_FINGER_MAX_ORDER_VALUE:
            problems.append(
                f"{side} {shares:g} {ticker} = ${value:,.2f}, above the "
                f"${FAT_FINGER_MAX_ORDER_VALUE:,.0f} per-order cap"
            )
            continue
        sizes = per.get((ticker, side), [])
        med = (
            median(sizes)
            if len(sizes) >= MIN_HISTORY_FOR_MEDIAN
            else (
                median(all_sizes)
                if len(all_sizes) >= MIN_HISTORY_FOR_MEDIAN
                else None
            )
        )
        if med is not None and med > 0 and value > med * FAT_FINGER_MEDIAN_MULTIPLE:
            basis = (
                f"its usual {side.lower()} order value for {ticker}"
                if len(sizes) >= MIN_HISTORY_FOR_MEDIAN
                else "the account's usual order value"
            )
            problems.append(
                f"{side} {shares:g} {ticker} = ${value:,.2f} is more than "
                f"{FAT_FINGER_MEDIAN_MULTIPLE:g}× {basis} "
                f"(${med:,.2f})"
            )
        if side == "BUY":
            # Sells are exempt on purpose: never trap an exit (see the
            # constant's comment and qrp.risk.guards).
            last = ctx.last_price(ticker)
            if last is not None and last > 0 and o["price"] > 0:
                drift = abs(o["price"] - last) / last
                if drift > FAT_FINGER_MAX_PRICE_DRIFT:
                    problems.append(
                        f"BUY {ticker} at ${o['price']:,.2f} is "
                        f"{drift:.1%} away from its last close of "
                        f"${last:,.2f} (limit "
                        f"{FAT_FINGER_MAX_PRICE_DRIFT:.0%}) — likely a "
                        "typo or a stale price"
                    )
    if problems:
        return CheckResult(
            name, "fail",
            "Blocked as a possible fat-finger: " + "; ".join(problems) + ".",
        )
    if not all_sizes:
        return CheckResult(
            name, "warn",
            "No order history yet, so the usual-size comparison is skipped "
            "this time — the dollar cap and price-drift checks still ran.",
        )
    return CheckResult(name, "pass", "Order sizes look normal.")


def _check_duplicates(ctx: _CheckContext) -> CheckResult:
    name = "Duplicate protection"
    if ctx.open_orders_error:
        return CheckResult(
            name, "fail",
            "Could not read the broker's open orders, so duplicates cannot "
            f"be ruled out ({ctx.open_orders_error}). Staging is blocked "
            "until the broker answers.",
        )
    problems = []
    for o in ctx.orders:
        cid = client_order_id(o["ticker"], o["side"], ctx.day)
        for open_o in ctx.open_orders:
            open_cid = str(getattr(open_o, "client_order_id", "") or "")
            same_id = bool(open_cid) and open_cid == cid
            same_ticket = (
                normalize_ticker(str(getattr(open_o, "ticker", "") or ""))
                == o["ticker"]
                and str(getattr(open_o, "side", "") or "").upper()
                == o["side"]
            )
            if same_id or same_ticket:
                problems.append(
                    f"{o['side']} {o['ticker']} (order {cid} already open)"
                )
                break
    if problems:
        return CheckResult(
            name, "fail",
            "These orders look like duplicates of orders already open at "
            "the broker: " + "; ".join(problems) + ".",
        )
    return CheckResult(
        name, "pass",
        "No open order carries any of this batch's order IDs.",
    )


def _check_no_short(ctx: _CheckContext) -> CheckResult:
    name = "No shorting / no margin"
    problems = []
    for o in ctx.orders:
        if o["side"] not in ("BUY", "SELL"):
            problems.append(
                f"{o['ticker']}: unknown side {o['side']!r} — only BUY or "
                "SELL are allowed (this platform is long-only)"
            )
            continue
        if not o["shares"] > 0:
            problems.append(
                f"{o['side']} {o['ticker']}: shares must be positive"
            )
            continue
        if o["side"] == "SELL":
            held = ctx.positions.get(o["ticker"], 0.0)
            if o["shares"] > held + 1e-9:
                problems.append(
                    f"SELL {o['shares']:g} {o['ticker']} but only "
                    f"{held:g} held — selling what you do not own "
                    "(shorting) is never allowed"
                )
    if problems:
        return CheckResult(
            name, "fail", "Blocked: " + "; ".join(problems) + "."
        )
    return CheckResult(name, "pass", "Long-only: every sell is covered.")
