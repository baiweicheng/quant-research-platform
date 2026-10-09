"""Automated data-health checks for the local market database (P5: data quality).

Everything downstream — signals, backtests, paper trading — trusts the
prices in ``data/market.db``. These checks verify that trust instead of
assuming it. They **never change or "fix" data**; they only flag things in
plain English so a person can decide what to do.

Checks (each returns :class:`Finding` items with a severity):
- **stale** — a ticker whose newest price is older than expected
  (trading-day aware: weekends don't count, ~2 trading days of grace).
- **inactive** — a ticker whose prices stopped far longer ago (>10
  trading days) while everything else is current: it almost certainly
  delisted, merged, or was renamed. Info only — it is left out of
  target portfolios and never blocks trading. If a large share of the
  database is that old (the download as a whole is stale), the delisted
  explanation doesn't hold, so those tickers stay "stale" and keep
  blocking instead (the global-stall guard in _classify_freshness).
- **outlier** — a single-day move bigger than 25% (documented below), a
  move far beyond the day's cross-sectional median, or an impossible bar
  (high below low, close outside high/low).
- **corporate_action** — Yahoo's declared splits/dividends cross-checked
  against the price history and the dividends already stored. Dividends are
  verified price-implied (the adjusted/unadjusted close ratio steps by a
  mechanically exact amount at each ex-date); splits can't be verified from
  split-adjusted prices, so they get plausibility checks (sane factor,
  ex-date near real data) instead. Mismatches say "verify against the SEC
  filing" because SEC EDGAR filings are the authoritative record; Yahoo
  stays the primary source and is only ever questioned, never overwritten.
- **cross_vendor** — a few stored closes compared against a fresh
  re-download, plus an honest note: true cross-vendor checking (a second,
  independent provider) is not available on free data tiers, so this is a
  local-corruption check, not a Yahoo-vs-someone-else check.

Results are structured dataclasses (:class:`DataHealthReport`), not just
UI strings, so P6's pre-trade checklist can consume them programmatically.
Run from the "Data health" dashboard page or via
``run_data_health_checks(...)`` directly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from qrp.data.store import connect_db

# --- Tunables (documented, not magic) ---------------------------------------

#: A ticker is "stale" when its newest bar is more than this many trading
#: days behind the latest expected trading day. Two days of grace covers a
#: long weekend plus a slow download; more than that means the download
#: probably stopped part-way. Mirrors the convention in
#: qrp.ui.helpers.data_staleness_note.
STALE_TRADING_DAYS = 2

#: A ticker whose newest bar is more than this many trading days behind
#: is no longer merely "stale" — it almost certainly stopped trading
#: (delisted, merged, or renamed; e.g. AVB after its August 2026
#: merger). It gets an "inactive" info finding instead of a stale
#: warning: it can't be bought, so target building leaves it out, and
#: it never blocks the pre-trade freshness gate. Two weeks is far
#: beyond any download hiccup or holiday stretch.
INACTIVE_TRADING_DAYS = 10

#: Global-stall guard for the inactive rule: if MORE than this share of
#: the tickers with data are deeper than INACTIVE_TRADING_DAYS behind,
#: the whole database is stale (the user simply hasn't refreshed in a
#: while) — no ticker is classified inactive, everything stays "stale",
#: and the freshness gate keeps blocking. A delisting is a per-ticker
#: story; it can't explain a tenth of the market going quiet at once.
INACTIVE_MAX_SHARE = 0.10

#: One-day simple return beyond this (in either direction) is flagged as a
#: suspicious outlier. 25% is deliberately high: real single-day moves that
#: big are rare and memorable (takeovers, earnings blowups), while Yahoo
#: pre-adjusts for splits, so ordinary corporate actions never trip it.
#: Anything flagged is for *review* — the check cannot tell a real event
#: from a bad row.
OUTLIER_RETURN_THRESHOLD = 0.25

#: Cross-sectional leg of the outlier check: a move is also flagged (as
#: info, not warning) when it exceeds this multiple of the day's median
#: absolute move *and* clears MIN_CROSS_SECTIONAL_RETURN, so a calm market
#: doesn't flag every 2% wiggle.
CROSS_SECTIONAL_MULTIPLE = 5.0
MIN_CROSS_SECTIONAL_RETURN = 0.10

#: NOTE (corrected 2026-10-07 against real data): Yahoo's chart API serves
#: *split-adjusted* closes (verified: AAPL's 2020-08-31 4:1 split shows
#: $124.81, not ~$500, and the AdjClose/Close ratio sits flat at 0.969 on
#: both sides of the ex-date). A genuine split is therefore INVISIBLE in
#: the adjusted series by construction — the ratio only steps for
#: DIVIDENDS. So splits get plausibility checks (sane factor, ex-date near
#: real price data) plus the SEC-filing note, while the ratio-step math
#: verifies dividends, where it is mechanically exact.

#: Overnight AdjClose/Close steps at or above this size, with no declared
#: dividend nearby, are flagged as possibly-missing dividends. 3% is well
#: above any rounding noise (the ratio is deterministic per bar) and well
#: below any plausible unlisted payout worth chasing.
DIVIDEND_STEP_DETECTION = 0.03

#: A declared dividend is "confirmed" when the ratio steps by the expected
#: amount within this absolute tolerance (2 percentage points — generous,
#: since the adjustment is mechanically exact).
DIVIDEND_STEP_TOLERANCE = 0.02

#: Split factors outside this range are implausible (real splits top out
#: around 100:1; reverse splits around 1:100) and get flagged for a manual
#: look at the SEC filing.
SPLIT_FACTOR_MAX = 100.0
SPLIT_FACTOR_MIN = 0.01

#: Fresh re-download vs stored close must agree within this relative
#: tolerance to pass the cross-vendor spot check. Closes are exact to the
#: cent, so 0.5% is generous — anything beyond it is corruption or a
#: genuine vendor revision.
CROSSCHECK_TOLERANCE = 0.005

#: How many tickers the cross-vendor check spot-checks per run. Small on
#: purpose: it is a smoke test, not a second pipeline.
CROSSCHECK_SAMPLE = 5

#: Corporate-action cross-check looks back this far from ``asof``.
ACTION_LOOKBACK_DAYS = 365

#: A stored dividend matches a Yahoo-listed dividend when the ex-dates are
#: within this many calendar days (ex-date conventions differ by a day or
#: two between sources).
DIVIDEND_MATCH_DAYS = 5

#: A price-ratio step matches a declared split when within this many
#: calendar days of the split's ex-date.
SPLIT_MATCH_DAYS = 3

SEC_GROUND_TRUTH_NOTE = (
    "SEC EDGAR filings are the authoritative record for splits and "
    "dividends — if the numbers look wrong, check the company's filing "
    "before trusting either copy."
)

CROSS_VENDOR_LIMITATION = (
    "True cross-vendor checking — comparing prices against a second, "
    "independent data provider — is not available on free data plans: "
    "Alpha Vantage's free tier allows only about 25 spot checks a day, "
    "Tiingo's free tier forbids storing data locally, and Finnhub blocks "
    "historical prices on free keys. So this check re-downloads a few "
    "tickers fresh from Yahoo and compares against what's stored: it "
    "catches local corruption, not Yahoo-wide errors."
)


# --- Result types ------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One data-health observation, in plain English.

    ``check`` is one of "stale", "inactive", "outlier",
    "corporate_action", "cross_vendor" or "general". ``severity`` is
    "info", "warning" or
    "error". ``summary`` is the one-liner for the UI; ``detail`` explains
    what it means; ``action`` says what the user can do ("" when nothing
    is needed).
    """

    check: str
    severity: str
    ticker: str | None
    summary: str
    detail: str = ""
    action: str = ""

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "ticker": self.ticker,
            "summary": self.summary,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class DataHealthReport:
    """Everything one run of the checks found."""

    checked_at: str  # ISO timestamp (UTC)
    tickers_checked: int
    findings: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {"info": 0, "warning": 0, "error": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)

    def worst_severity(self) -> str:
        """'ok' | 'info' | 'warning' | 'error' — for P6's checklist."""
        if self.has_errors():
            return "error"
        if self.has_warnings():
            return "warning"
        if self.findings:
            return "info"
        return "ok"

    def findings_for(self, ticker: str) -> list[Finding]:
        return [f for f in self.findings if f.ticker == ticker]

    def plain_english_summary(self) -> str:
        c = self.counts()
        if not self.findings:
            return "All checks passed — nothing looks off in the stored data."
        parts = []
        if c["error"]:
            parts.append(f"{c['error']} problem(s) that need attention")
        if c["warning"]:
            parts.append(f"{c['warning']} thing(s) worth a look")
        if c["info"]:
            parts.append(f"{c['info']} note(s)")
        return "Data-health check found " + ", ".join(parts) + "."

    def as_dict(self) -> dict:
        """Structured form for programmatic consumers (P6 pre-trade)."""
        return {
            "checked_at": self.checked_at,
            "tickers_checked": self.tickers_checked,
            "worst_severity": self.worst_severity(),
            "counts": self.counts(),
            "findings": [f.as_dict() for f in self.findings],
        }


# --- Shared date logic --------------------------------------------------------

def trading_days_between(start: date, end: date) -> int:
    """Mon–Fri days d with ``start < d <= end``.

    Weekends are skipped; US market holidays are NOT excluded (same
    convention as the dashboard's staleness nudge — a holiday can add one
    phantom "trading day" to the count, which the 2-day grace absorbs).
    """
    n = 0
    probe = start
    while True:
        probe = probe + timedelta(days=1)
        if probe > end:
            break
        if probe.weekday() < 5:
            n += 1
    return n


def expected_latest_trading_day(asof: date) -> date:
    """The latest weekday on or before ``asof`` (weekends roll back)."""
    d = asof
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def _parse(d: str | date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# --- DB access -----------------------------------------------------------------

def _tickers(conn: sqlite3.Connection, tickers: list[str] | None) -> list[str]:
    if tickers is not None:
        return list(tickers)
    rows = conn.execute("SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker")
    return [r[0] for r in rows]


def _bars(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM daily_bars"
        " WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )
    return df


# --- Check 1: stale prices ------------------------------------------------------

def _last_bar_dates(
    conn: sqlite3.Connection, ticker_list: list[str], asof: date
) -> dict[str, str | None]:
    """ticker -> newest stored bar date (YYYY-MM-DD) on or before asof.

    Point-in-time on purpose: bars dated after ``asof`` don't count, so
    asking about a past date answers as of that date (used by the
    target-building exclusion, where a stock that died in August must
    still look alive in a January replay).
    """
    out: dict[str, str | None] = {}
    cutoff = asof.isoformat()
    for ticker in ticker_list:
        row = conn.execute(
            "SELECT MAX(date) FROM daily_bars WHERE ticker = ? AND date <= ?",
            (ticker, cutoff),
        ).fetchone()
        out[ticker] = row[0] if row else None
    return out


def _classify_freshness(
    last_by_ticker: dict[str, str | None], asof: date
) -> tuple[dict[str, int], set[str]]:
    """(behind_by_ticker, inactive_set) — the shared staleness rule.

    ``behind`` counts trading days from each ticker's newest bar to the
    latest expected trading day (tickers with no bar on or before
    ``asof`` aren't in it). Tickers deeper than INACTIVE_TRADING_DAYS
    count as inactive (delisted-looking) — UNLESS more than
    INACTIVE_MAX_SHARE of the tickers with data are that deep, in which
    case the whole database is simply stale (global stall) and nothing
    is inactive: a delisting can't explain a tenth of the market going
    quiet at once. Deterministic — same database, same answer.
    """
    expected = expected_latest_trading_day(asof)
    behind = {
        t: trading_days_between(_parse(last), expected)
        for t, last in last_by_ticker.items()
        if last
    }
    deep = {t for t, b in behind.items() if b > INACTIVE_TRADING_DAYS}
    if behind and len(deep) / len(behind) <= INACTIVE_MAX_SHARE:
        return behind, deep
    return behind, set()


def check_staleness(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    asof: date | None = None,
) -> list[Finding]:
    """Flag tickers whose newest bar is older than expected.

    Three outcomes per ticker: fresh enough (no finding); "stale"
    (warning — the download probably stopped part-way, and it blocks
    the pre-trade freshness gate); or "inactive" (info — prices stopped
    more than INACTIVE_TRADING_DAYS ago while the rest of the database
    is current, so the ticker almost certainly delisted, merged, or was
    renamed; it is left out of target portfolios and never blocks).
    The global-stall guard keeps a long-unrefreshed database "stale"
    rather than suddenly declaring a tenth of the market delisted.
    """
    asof = _parse(asof) if asof is not None else date.today()
    resolved = _tickers(conn, tickers)
    last_by = _last_bar_dates(conn, resolved, asof)
    behind, inactive = _classify_freshness(last_by, asof)
    expected = expected_latest_trading_day(asof)
    out: list[Finding] = []
    for ticker in resolved:
        last = last_by[ticker]
        if not last:
            out.append(
                Finding(
                    check="stale",
                    severity="info",
                    ticker=ticker,
                    summary=f"{ticker} has no price data stored.",
                    detail=(
                        f"{ticker} was skipped during the download (or has "
                        "since been delisted). It simply won't appear in "
                        "signals or backtests."
                    ),
                    action=(
                        f"If you expected {ticker} to be there, re-run the "
                        "download for it on the Market data page."
                    ),
                )
            )
            continue
        if ticker in inactive:
            out.append(
                Finding(
                    check="inactive",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"{ticker}'s prices end {last} and it hasn't "
                        "traded since — it was likely delisted, merged, "
                        "or renamed, so it's left out of picks and "
                        "doesn't block trading."
                    ),
                    detail=(
                        "A stock that stops trading for weeks while the "
                        "rest of the database is current has almost always "
                        "been delisted, bought out, or renamed. There are "
                        "no newer prices to download, so this one never "
                        "holds up a rebalance."
                    ),
                    action=(
                        f"If you believe {ticker} still trades, re-run "
                        "the download for it on the Market data page — if "
                        "prices come back, this note clears itself."
                    ),
                )
            )
            continue
        if behind[ticker] > STALE_TRADING_DAYS:
            out.append(
                Finding(
                    check="stale",
                    severity="warning",
                    ticker=ticker,
                    summary=(
                        f"{ticker}'s newest price is from {last} — "
                        f"{behind[ticker]} trading days behind {expected}."
                    ),
                    detail=(
                        "Prices more than two trading days old usually mean "
                        "the download stopped part-way (or this ticker "
                        "stopped trading). Signals and backtests built on "
                        "stale prices quietly answer the wrong question."
                    ),
                    action=(
                        "Re-run the download on the Market data page — "
                        "incremental, so it only fetches what's missing."
                    ),
                )
            )
    return out


def _last_bars_from_db(
    db_path: str | Path, day: date
) -> dict[str, str | None] | None:
    """ticker -> newest bar date on/before ``day`` for the whole database.

    None when there is no database or no price table yet (callers treat
    that as "nothing known", never as an error — these helpers sit on
    build/check paths that must not crash on a fresh install).
    """
    path = Path(db_path)
    if not path.exists():
        return None
    with connect_db(path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='daily_bars'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return None
        return _last_bar_dates(conn, _tickers(conn, None), day)


def inactive_tickers(
    db_path: str | Path, asof: date | str | None = None
) -> set[str]:
    """Tickers that look delisted as of ``asof`` (default: today).

    The exact rule from check_staleness (see _classify_freshness):
    newest stored bar more than INACTIVE_TRADING_DAYS trading days
    behind the latest expected trading day, with the global-stall guard
    (if a large share of the database is that far behind, nothing counts
    as inactive — the database itself is stale). Point-in-time: only
    bars dated on or before ``asof`` are considered. A missing database
    or price table yields an empty set.
    """
    day = _parse(asof) if asof is not None else date.today()
    last_by = _last_bars_from_db(db_path, day)
    if not last_by:
        return set()
    _, inactive = _classify_freshness(last_by, day)
    return inactive


def stale_tickers(
    db_path: str | Path, asof: date | str | None = None
) -> set[str]:
    """Tickers with prices but no CURRENT price as of ``asof``.

    Newest stored bar (on or before ``asof``) more than
    STALE_TRADING_DAYS trading days behind the latest expected trading
    day — whether it stopped updating 3 days ago (a download gap) or 3
    months ago (delisted). This is the target-building exclusion rule:
    a name without a current price can't be bought anyway, so it never
    enters a target portfolio. Deliberately per-ticker with NO
    global-stall guard — judging is relative to ``asof`` (the decision
    date, normally the data's own latest date), so a generally old
    database viewed as of its own latest date excludes nothing. The
    pre-trade freshness gate separately refuses to stage when the
    database as a whole is stale (see database_globally_stale).
    Missing database/table -> empty set.
    """
    day = _parse(asof) if asof is not None else date.today()
    last_by = _last_bars_from_db(db_path, day)
    if not last_by:
        return set()
    expected = expected_latest_trading_day(day)
    return {
        t
        for t, last in last_by.items()
        if last
        and trading_days_between(_parse(last), expected) > STALE_TRADING_DAYS
    }


def database_globally_stale(
    db_path: str | Path, asof: date | str | None = None
) -> bool:
    """True when the database AS A WHOLE is too old to trade on.

    Either the freshest stored bar is more than STALE_TRADING_DAYS
    trading days behind the expected latest trading day (nobody has
    refreshed lately), or the global-stall condition holds (more than
    INACTIVE_MAX_SHARE of the tickers with data are deeper than
    INACTIVE_TRADING_DAYS behind — see _classify_freshness). A handful
    of lagging or delisted names among otherwise current data does NOT
    make the database globally stale; the pre-trade freshness gate
    blocks only on this whole-database condition (or on a stale name
    inside the order batch itself). Missing database/table -> False
    (the "no database" finding reports that case separately).
    """
    day = _parse(asof) if asof is not None else date.today()
    last_by = _last_bars_from_db(db_path, day)
    if not last_by:
        return False
    lasts = [_parse(v) for v in last_by.values() if v]
    if not lasts:
        return False
    expected = expected_latest_trading_day(day)
    if trading_days_between(max(lasts), expected) > STALE_TRADING_DAYS:
        return True
    behind, inactive = _classify_freshness(last_by, day)
    deep = {t for t, b in behind.items() if b > INACTIVE_TRADING_DAYS}
    return bool(deep) and not inactive


# --- Check 2: outliers + impossible bars -----------------------------------------

def check_outliers(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    threshold: float = OUTLIER_RETURN_THRESHOLD,
) -> list[Finding]:
    """Flag huge one-day moves, off-median moves, and impossible bars."""
    tickers = _tickers(conn, tickers)
    series: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _bars(conn, ticker)
        if len(df) >= 2:
            series[ticker] = df

    # Per-date median absolute return across tickers (cross-sectional leg).
    date_rets: dict[str, list[float]] = {}
    rets: dict[str, list[tuple[str, float]]] = {}
    for ticker, df in series.items():
        closes = pd.to_numeric(df["close"], errors="coerce")
        prev = closes.shift(1)
        r = closes / prev - 1
        rows = []
        for d, rv, pv in zip(df["date"], r, prev):
            if pd.isna(rv) or pd.isna(pv) or pv <= 0:
                continue  # non-positive prices are flagged as errors below
            rows.append((str(d), float(rv)))
            date_rets.setdefault(str(d), []).append(abs(float(rv)))
        rets[ticker] = rows
    medians = {
        d: float(pd.Series(v).median()) for d, v in date_rets.items() if v
    }

    out: list[Finding] = []
    flagged_days: set[tuple[str, str]] = set()
    for ticker, rows in rets.items():
        for d, rv in rows:
            if abs(rv) > threshold:
                flagged_days.add((ticker, d))
                out.append(
                    Finding(
                        check="outlier",
                        severity="warning",
                        ticker=ticker,
                        summary=(
                            f"{ticker} moved {rv:+.1%} on {d} — an unusually "
                            "large one-day move."
                        ),
                        detail=(
                            f"Moves beyond ±{threshold:.0%} in a day are rare "
                            "enough to double-check: it could be real (a "
                            "takeover, an earnings surprise) or a bad row in "
                            "the download. Yahoo prices are already "
                            "split-adjusted, so ordinary splits never trip "
                            "this. This check never changes the price — it "
                            "only asks you to look."
                        ),
                        action=(
                            f"Check the news for {ticker} around {d}. If "
                            "nothing explains it, re-download the ticker on "
                            "the Market data page."
                        ),
                    )
                )
                continue
            med = medians.get(d, 0.0)
            if (
                med > 0
                and abs(rv) > MIN_CROSS_SECTIONAL_RETURN
                and abs(rv) > CROSS_SECTIONAL_MULTIPLE * med
            ):
                flagged_days.add((ticker, d))
                out.append(
                    Finding(
                        check="outlier",
                        severity="info",
                        ticker=ticker,
                        summary=(
                            f"{ticker} moved {rv:+.1%} on {d}, far more than "
                            "the typical stock that day "
                            f"(median move {med:.1%})."
                        ),
                        detail=(
                            "On a calm day this stands out; on a wild day "
                            "the median rises with it and nothing fires. "
                            "Usually harmless — sometimes the first hint of "
                            "a data problem."
                        ),
                    )
                )

    # Impossible bars: internal contradictions no real market can produce.
    for ticker, df in series.items():
        o = pd.to_numeric(df["open"], errors="coerce")
        h = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        v = pd.to_numeric(df["volume"], errors="coerce")
        for d, oo, hh, ll, cc, vv in zip(
            df["date"], o, h, lo, c, v
        ):
            d = str(d)
            if pd.isna(hh) or pd.isna(ll) or pd.isna(cc):
                continue
            if cc <= 0:
                out.append(
                    Finding(
                        check="outlier",
                        severity="error",
                        ticker=ticker,
                        summary=(
                            f"{ticker} has a non-positive close (${cc:,.2f}) "
                            f"on {d}."
                        ),
                        detail=(
                            "Stock prices can't be zero or negative — this "
                            "row is corrupted (a failed download or a "
                            "vendor glitch)."
                        ),
                        action=(
                            f"Re-download {ticker} on the Market data page; "
                            "the fresh copy overwrites the bad row."
                        ),
                    )
                )
                continue
            if hh < ll:
                out.append(
                    Finding(
                        check="outlier",
                        severity="error",
                        ticker=ticker,
                        summary=(
                            f"{ticker} has an impossible price bar on {d}: "
                            f"the day's high (${hh:,.2f}) is below its low "
                            f"(${ll:,.2f})."
                        ),
                        detail=(
                            "High can never be below low — this row is "
                            "corrupted. Signals computed from it are "
                            "unreliable until it's replaced."
                        ),
                        action=(
                            f"Re-download {ticker} on the Market data page; "
                            "the fresh copy overwrites the bad row."
                        ),
                    )
                )
                continue
            if cc < ll - 1e-9 or cc > hh + 1e-9:
                out.append(
                    Finding(
                        check="outlier",
                        severity="error",
                        ticker=ticker,
                        summary=(
                            f"{ticker}'s close on {d} (${cc:,.2f}) sits "
                            f"outside the day's range "
                            f"(${ll:,.2f}–${hh:,.2f})."
                        ),
                        detail=(
                            "The close must fall between the day's low and "
                            "high — this row contradicts itself."
                        ),
                        action=(
                            f"Re-download {ticker} on the Market data page; "
                            "the fresh copy overwrites the bad row."
                        ),
                    )
                )
            if not pd.isna(vv) and vv < 0:
                out.append(
                    Finding(
                        check="outlier",
                        severity="error",
                        ticker=ticker,
                        summary=(
                            f"{ticker} shows negative trading volume on {d}."
                        ),
                        detail="Volume can't be negative — corrupted row.",
                        action=(
                            f"Re-download {ticker} on the Market data page."
                        ),
                    )
                )
    return out


# --- Check 3: corporate actions ---------------------------------------------------

#: Injected fetcher contract: fetch_actions(ticker, start, end) -> dict with
#: "splits" (DataFrame: date, numerator, denominator),
#: "dividends" (DataFrame: ex_date, amount) and
#: "prices" (DataFrame: Date, Close, AdjClose).
FetchActions = Callable[[str, str, str], dict]


def check_corporate_actions(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    fetch_actions: FetchActions | None = None,
    window_days: int = ACTION_LOOKBACK_DAYS,
    asof: date | None = None,
) -> list[Finding]:
    """Cross-check declared splits/dividends against the price history.

    ``fetch_actions`` is injected so tests can use canned data; the
    dashboard passes the real Yahoo fetcher. Any network failure becomes an
    "info" finding for that ticker — a failed check is skipped, never fatal.
    """
    asof = _parse(asof) if asof is not None else date.today()
    out: list[Finding] = []
    if fetch_actions is None:
        out.append(
            Finding(
                check="corporate_action",
                severity="info",
                ticker=None,
                summary=(
                    "Corporate-action cross-check skipped (no data fetcher)."
                ),
                detail=(
                    "This check needs Yahoo's current splits/dividends "
                    "listing; it runs from the Data health page, which "
                    "supplies the fetcher."
                ),
            )
        )
        return out

    start = (asof - timedelta(days=window_days)).isoformat()
    end = asof.isoformat()
    for ticker in _tickers(conn, tickers):
        try:
            actions = fetch_actions(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            out.append(
                Finding(
                    check="corporate_action",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"Couldn't reach Yahoo to cross-check {ticker}'s "
                        "splits and dividends — skipped, not a failure."
                    ),
                    detail=str(exc)[:200],
                )
            )
            continue

        splits = actions.get("splits")
        dividends = actions.get("dividends")
        prices = actions.get("prices")
        if prices is None or getattr(prices, "empty", True):
            out.append(
                Finding(
                    check="corporate_action",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"No price reference returned for {ticker} — the "
                        "split cross-check was skipped."
                    ),
                    detail="",
                )
            )
            continue

        adj = pd.to_numeric(prices.get("AdjClose"), errors="coerce")
        if adj.isna().all():
            out.append(
                Finding(
                    check="corporate_action",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"Yahoo didn't return adjusted prices for {ticker} — "
                        "the split cross-check was skipped."
                    ),
                    detail="",
                )
            )
            continue

        # (a) Declared splits: plausibility only. Yahoo serves
        # split-adjusted closes, so a genuine split leaves NO trace in the
        # price series (verified against AAPL's 2020 4:1 split) — splits can
        # only be truly verified against the SEC filing, never from prices.
        declared_splits: list[str] = []
        if splits is not None and not splits.empty:
            price_days = {
                str(x)[:10] for x in prices["Date"].astype(str)
            }
            for row in splits.itertuples():
                d = str(row.date)[:10]
                try:
                    factor = float(row.numerator) / float(row.denominator)
                except (ZeroDivisionError, TypeError, ValueError):
                    continue
                declared_splits.append(d)
                num = (
                    int(row.numerator)
                    if float(row.numerator).is_integer()
                    else row.numerator
                )
                den = (
                    int(row.denominator)
                    if float(row.denominator).is_integer()
                    else row.denominator
                )
                if not (SPLIT_FACTOR_MIN <= factor <= SPLIT_FACTOR_MAX):
                    out.append(
                        Finding(
                            check="corporate_action",
                            severity="warning",
                            ticker=ticker,
                            summary=(
                                f"Yahoo lists an unusual {num}-for-{den} "
                                f"split for {ticker} on {d}."
                            ),
                            detail=(
                                "Real splits top out around 100:1 and "
                                "reverse splits around 1:100 — this factor "
                                "is outside that range, which smells like a "
                                "bad listing row. " + SEC_GROUND_TRUTH_NOTE
                            ),
                            action=(
                                "Check the company's SEC filing to confirm "
                                "the split is real."
                            ),
                        )
                    )
                if not any(
                    abs((_parse(d) - _parse(pd_)).days) <= SPLIT_MATCH_DAYS
                    for pd_ in price_days
                ):
                    out.append(
                        Finding(
                            check="corporate_action",
                            severity="info",
                            ticker=ticker,
                            summary=(
                                f"Yahoo lists a {num}-for-{den} split for "
                                f"{ticker} on {d}, but there's no price "
                                "data near that date."
                            ),
                            detail=(
                                "The listing may be mis-dated, or the "
                                "ticker's history doesn't cover the split. "
                                + SEC_GROUND_TRUTH_NOTE
                            ),
                        )
                    )

        # (b) Dividend-implied steps: the AdjClose/Close ratio steps up by
        # ~D/C at each dividend ex-date (mechanically exact). Verify each
        # declared dividend shows its step, and flag steps nobody declared.
        ratio = (adj / pd.to_numeric(prices["Close"], errors="coerce")).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        r_dates = prices["Date"].astype(str).str[:10]
        declared_divs: list[tuple[str, float]] = []

        def _ratio_step(day: str) -> float | None:
            """Actual AdjClose/Close step across a dividend ex-date."""
            idx = r_dates[r_dates < day].index
            jdx = r_dates[r_dates >= day].index
            before = ratio.loc[idx].tail(3).median()
            after = ratio.loc[jdx].head(3).median()
            if pd.isna(before) or pd.isna(after) or before == 0:
                return None
            return float(after / before)

        if dividends is not None and not dividends.empty:
            closes = pd.to_numeric(prices["Close"], errors="coerce")
            for row in dividends.itertuples():
                try:
                    d = str(row.ex_date)[:10]
                    amt = float(row.amount)
                except (TypeError, ValueError):
                    continue
                if amt <= 0:
                    continue
                declared_divs.append((d, amt))
                step = _ratio_step(d)
                if step is None:
                    continue
                idx = r_dates[r_dates < d].index
                if len(idx) == 0:
                    continue
                close_before = float(closes.loc[idx].iloc[-1])
                if close_before <= 0 or amt >= close_before:
                    continue
                expected = close_before / (close_before - amt)
                if abs(step - expected) > DIVIDEND_STEP_TOLERANCE:
                    out.append(
                        Finding(
                            check="corporate_action",
                            severity="warning",
                            ticker=ticker,
                            summary=(
                                f"Yahoo lists a ${amt:.2f} dividend for "
                                f"{ticker} (ex-date {d}), but the price "
                                "history doesn't show the expected step."
                            ),
                            detail=(
                                "A real dividend steps the adjusted-price "
                                f"ratio by ~{expected:.3f}x on the ex-date; "
                                f"here it moved ~{step:.3f}x. Either the "
                                "listing is wrong or the prices around that "
                                "date are. " + SEC_GROUND_TRUTH_NOTE
                            ),
                            action=(
                                "Check the company's SEC filing to confirm "
                                "the dividend."
                            ),
                        )
                    )

        # Reverse: dividend-sized steps with no declared dividend nearby.
        for i in range(1, len(prices)):
            r0, r1 = ratio.iloc[i - 1], ratio.iloc[i]
            if pd.isna(r0) or pd.isna(r1) or r0 == 0:
                continue
            step = r1 / r0
            if abs(step - 1) < DIVIDEND_STEP_DETECTION:
                continue
            d = str(r_dates.iloc[i])
            if any(
                abs((_parse(d) - _parse(dd)).days) <= SPLIT_MATCH_DAYS
                for dd, _ in declared_divs
            ):
                continue
            pct = (step - 1) * 100
            out.append(
                Finding(
                    check="corporate_action",
                    severity="warning",
                    ticker=ticker,
                    summary=(
                        f"{ticker}'s prices show a dividend-sized step "
                        f"({pct:+.1f}%) on {d}, but no dividend is listed."
                    ),
                    detail=(
                        "The adjusted-price ratio jumped overnight with no "
                        "matching dividend record — a missing or mis-dated "
                        "payout in the listing. " + SEC_GROUND_TRUTH_NOTE
                    ),
                    action=(
                        "Verify against the company's SEC filing; "
                        "re-downloading won't help if the listing itself is "
                        "missing the payout."
                    ),
                )
            )

        if declared_splits:
            out.append(
                Finding(
                    check="corporate_action",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"{ticker} has {len(declared_splits)} listed "
                        f"split(s) ({', '.join(declared_splits)})."
                    ),
                    detail=(
                        "Splits can't be verified from split-adjusted "
                        "prices — only the listing's plausibility was "
                        "checked. " + SEC_GROUND_TRUTH_NOTE
                    ),
                )
            )

        # (c) Dividends: stored vs Yahoo-listed.
        yahoo_divs: list[tuple[str, float]] = []
        if dividends is not None and not dividends.empty:
            for row in dividends.itertuples():
                try:
                    yahoo_divs.append(
                        (str(row.ex_date)[:10], float(row.amount))
                    )
                except (TypeError, ValueError):
                    continue
        has_div_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='dividends'"
            ).fetchone()
            is not None
        )
        stored_divs: list[tuple[str, float]] = []
        if has_div_table:
            stored = conn.execute(
                "SELECT ex_date, amount FROM dividends WHERE ticker = ?"
                " AND ex_date >= ? AND ex_date <= ?",
                (ticker, start, end),
            ).fetchall()
            stored_divs = [
                (str(ex)[:10], float(amt))
                for ex, amt in stored
                if float(amt) > 0
            ]

        def _near(target: str, cands: list[tuple[str, float]]) -> bool:
            t = _parse(target)
            return any(
                abs((_parse(c[0]) - t).days) <= DIVIDEND_MATCH_DAYS
                for c in cands
            )

        for ex, amt in yahoo_divs:
            if not _near(ex, stored_divs):
                # Only warn when the dividend falls inside the ticker's
                # stored history: a download starting in 2024 can't be
                # expected to hold 2019's payouts.
                (db_first,) = conn.execute(
                    "SELECT MIN(date) FROM daily_bars WHERE ticker = ?",
                    (ticker,),
                ).fetchone()
                (db_last,) = conn.execute(
                    "SELECT MAX(date) FROM daily_bars WHERE ticker = ?",
                    (ticker,),
                ).fetchone()
                if (
                    db_first
                    and db_last
                    and not (db_first[:10] <= ex <= db_last[:10])
                ):
                    continue
                out.append(
                    Finding(
                        check="corporate_action",
                        severity="warning",
                        ticker=ticker,
                        summary=(
                            f"Yahoo lists a ${amt:.2f} dividend for {ticker} "
                            f"(ex-date {ex}) that isn't in the database."
                        ),
                        detail=(
                            "The dividend download probably missed it — "
                            "backtests would under-credit this payout."
                        ),
                        action=(
                            f"Re-download {ticker} on the Market data page; "
                            "the dividend is stored idempotently."
                        ),
                    )
                )
        for ex, amt in stored_divs:
            if not _near(ex, yahoo_divs):
                out.append(
                    Finding(
                        check="corporate_action",
                        severity="info",
                        ticker=ticker,
                        summary=(
                            f"The database has a ${amt:.2f} dividend for "
                            f"{ticker} on {ex} that Yahoo no longer lists."
                        ),
                        detail=(
                            "Listings get revised (or a payout is "
                            "withdrawn). Kept as-is — harmless either way — "
                            "but re-downloading syncs the records."
                        ),
                    )
                )
    return out


# --- Check 4: cross-vendor spot checks ------------------------------------------

#: Injected fetcher contract: fetch_bars(ticker, start, end) -> DataFrame
#: with at least Date and Close columns (same shape as
#: qrp.data.yahoo.fetch_daily_bars).
FetchBars = Callable[[str, str, str], pd.DataFrame]


def check_cross_vendor(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
    fetch_bars: FetchBars | None = None,
    sample: int = CROSSCHECK_SAMPLE,
) -> list[Finding]:
    """Spot-check stored closes against a fresh re-download.

    The honest limitation (documented in the finding itself): free data
    tiers don't allow true cross-vendor checks, so this compares the
    database against a *fresh pull from the same vendor* — it catches local
    corruption, not Yahoo-wide errors.
    """
    out = [
        Finding(
            check="cross_vendor",
            severity="info",
            ticker=None,
            summary=(
                "Cross-vendor comparison is limited on free data plans — "
                "this check re-downloads a few tickers fresh instead."
            ),
            detail=CROSS_VENDOR_LIMITATION,
        )
    ]
    if fetch_bars is None:
        out.append(
            Finding(
                check="cross_vendor",
                severity="info",
                ticker=None,
                summary="Spot-check skipped (no data fetcher provided).",
                detail="Runs from the Data health page, which supplies one.",
            )
        )
        return out

    for ticker in sorted(_tickers(conn, tickers))[:sample]:
        row = conn.execute(
            "SELECT date, close FROM daily_bars WHERE ticker = ?"
            " ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if not row or row[1] is None:
            continue
        day, stored_close = str(row[0])[:10], float(row[1])
        try:
            fresh = fetch_bars(ticker, day, day)
        except Exception as exc:  # noqa: BLE001 - best-effort
            out.append(
                Finding(
                    check="cross_vendor",
                    severity="info",
                    ticker=ticker,
                    summary=(
                        f"Couldn't re-download {ticker} for the spot check "
                        "— skipped, not a failure."
                    ),
                    detail=str(exc)[:200],
                )
            )
            continue
        if fresh is None or getattr(fresh, "empty", True):
            continue
        match = fresh[fresh["Date"].astype(str).str[:10] == day]
        if match.empty:
            continue
        fresh_close = float(match["Close"].iloc[0])
        if stored_close == 0:
            continue
        drift = abs(fresh_close - stored_close) / abs(stored_close)
        if drift > CROSSCHECK_TOLERANCE:
            out.append(
                Finding(
                    check="cross_vendor",
                    severity="warning",
                    ticker=ticker,
                    summary=(
                        f"{ticker}'s stored close on {day} "
                        f"(${stored_close:,.2f}) doesn't match a fresh "
                        f"download (${fresh_close:,.2f})."
                    ),
                    detail=(
                        "The local row disagrees with a brand-new pull — "
                        "either the row got corrupted or the vendor revised "
                        "the bar."
                    ),
                    action=(
                        f"Re-download {ticker} on the Market data page; the "
                        "fresh copy overwrites the row."
                    ),
                )
            )
    return out


# --- Orchestration ----------------------------------------------------------------

def run_data_health_checks(
    db_path: str | Path,
    tickers: list[str] | None = None,
    asof: date | None = None,
    fetch_actions: FetchActions | None = None,
    fetch_bars: FetchBars | None = None,
    network_sample: int | None = 50,
) -> DataHealthReport:
    """Run every check against the local database and return a report.

    ``fetch_actions`` / ``fetch_bars`` are injected (tests pass fakes; the
    dashboard passes the real Yahoo fetchers). Network failures degrade to
    "info" findings per ticker — the report always completes.

    The local checks (stale, outlier) always cover every ticker. The
    network checks (corporate actions, cross-vendor) cover at most
    ``network_sample`` tickers — the ones the local checks flagged first,
    then a deterministic slice of the rest — because each one costs an
    HTTP request. ``None`` means "check every ticker over the network"
    (slow on a full universe); ``0`` skips the network checks. Whenever
    sampling kicks in, an info finding says so plainly.
    """
    db_path = Path(db_path)
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not db_path.exists():
        return DataHealthReport(
            checked_at=checked_at,
            tickers_checked=0,
            findings=[
                Finding(
                    check="general",
                    severity="info",
                    ticker=None,
                    summary="No database yet — nothing to check.",
                    detail="",
                    action=(
                        "Download data on the Market data page first, then "
                        "run the checks again."
                    ),
                )
            ],
        )
    with connect_db(db_path) as conn:
        has_table = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table'"
                " AND name='daily_bars'"
            ).fetchone()
            is not None
        )
        if not has_table:
            return DataHealthReport(
                checked_at=checked_at,
                tickers_checked=0,
                findings=[
                    Finding(
                        check="general",
                        severity="info",
                        ticker=None,
                        summary="The database has no price table yet.",
                        detail="",
                        action="Download data on the Market data page first.",
                    )
                ],
            )
        resolved = _tickers(conn, tickers)
        findings: list[Finding] = []
        findings += check_staleness(conn, resolved, asof=asof)
        findings += check_outliers(conn, resolved)

        # Network checks are sampled: flagged tickers first, then the rest.
        flagged = sorted(
            {
                f.ticker
                for f in findings
                if f.ticker and f.severity in ("warning", "error")
            }
        )
        if network_sample is None:
            net_tickers = resolved
        elif network_sample <= 0:
            net_tickers = []
        else:
            flagged_set = set(flagged)
            rest = [t for t in resolved if t not in flagged_set]
            net_tickers = flagged + rest[: max(0, network_sample - len(flagged))]
        if len(net_tickers) < len(resolved):
            what = (
                "were skipped (network_sample=0)"
                if not net_tickers
                else (
                    f"ran on {len(net_tickers)} of {len(resolved)} tickers "
                    "(the ones the local checks flagged, plus a slice of "
                    "the rest)"
                )
            )
            findings.append(
                Finding(
                    check="general",
                    severity="info",
                    ticker=None,
                    summary=(
                        "Network cross-checks (splits/dividends, fresh "
                        f"re-downloads) {what}."
                    ),
                    detail=(
                        "Each of those checks costs one web request per "
                        "ticker, so the dashboard samples instead of "
                        "hammering Yahoo's free endpoint. Call "
                        "run_data_health_checks(..., network_sample=None) "
                        "to check every ticker over the network."
                    ),
                )
            )
        findings += check_corporate_actions(
            conn, net_tickers, fetch_actions=fetch_actions, asof=asof
        )
        findings += check_cross_vendor(
            conn, net_tickers, fetch_bars=fetch_bars
        )
    return DataHealthReport(
        checked_at=checked_at,
        tickers_checked=len(resolved),
        findings=findings,
    )
