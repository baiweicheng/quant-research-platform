"""Research strategies, batch 3 (user-approved paper review).

Three long-only strategies from a research paper the user shared, implemented
from the CANONICAL academic definitions — the paper's own formulas were
garbled in transmission, so each signal below cites the original source and
notes where this implementation had to choose.

- continuous_momentum: Frog-in-the-Pan (Da, Gurun & Warachka 2014) — among
  past winners, prefer the smooth climbers over the choppy jumpers.
- overnight_intraday: intraday vs overnight return divergence — quiet
  institutional daytime accumulation without retail overnight hype.
- twin_momentum: price momentum AND earnings surprise (SUE) must both
  confirm — dual-engine momentum, after Novy-Marx / Bernard-Thomas.

All three are long-only adaptations of published long-short anomalies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qrp.data.store import DEFAULT_DB, connect_db, load_dividends
from qrp.data.yahoo import normalize_ticker
from qrp.signals.base import Signal
from qrp.signals.fundamental import (
    FundamentalSignal,
    _load_fact_frames,
    _SharedInputs,
)

# NOTE: qrp.backtest.engine imports qrp.signals.store at module level, so
# importing load_open_matrix here at module level would be circular. The
# helper below imports it lazily (always safe: by call time both modules
# are fully initialized) — the same pattern as _load_prices in
# qrp.signals.fundamental.
def _load_opens(tickers: list[str], db_path: str | Path) -> pd.DataFrame:
    from qrp.backtest.engine import load_open_matrix

    return load_open_matrix(tickers, path=db_path)


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional Z-scores, one row (date) at a time.

    Each day's values are centered and scaled across tickers, so a score
    of +1 means "one standard deviation above the average stock that
    day". Rows with zero spread (every stock identical) become NaN —
    there is nothing to rank.
    """
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mu, axis=0).div(sd, axis=0)


class ContinuousMomentum(Signal):
    """Frog-in-the-Pan: smooth winners beat choppy winners.

    Canonical definition (Da, Gurun & Warachka 2014, Review of Financial
    Studies): over the 12-month formation window skipping the most recent
    month, let PRET be the cumulative return and %pos / %neg the fraction
    of days with positive / negative returns. Information discreteness is

        ID = sign(PRET) x (%neg - %pos)

    A winner that climbed on many small up days has %pos >> %neg, so
    ID is near -1 (continuous); a winner driven by a few big jumps has
    ID near +1 (discrete). Low ID = "the frog didn't notice the water
    heating" = investors underreacted = the trend persists.

    NOTE on the formula: the task brief wrote ID with (%pos - %neg), but
    that is backwards relative to both the canonical paper and the
    brief's own "rank past winners by LOW ID first" instruction (it would
    rank choppy winners first). This implements the canonical
    sign(PRET) x (%neg - %pos), verified against the published paper.

    Direction: higher = smoother past winner = more attractive long.
    Stocks that are not winners (PRET <= 0) score below every winner —
    this is a winners-only idea, and losers are never ranked above them.
    """

    name = "continuous_momentum"
    description = (
        "How smoothly the stock climbed over the past year (ignoring the "
        "most recent month). Winners that rose on many small up days "
        "score high; winners that spiked on a few big jumps score low — "
        "smooth climbs tend to keep going while spikes tend to snap back."
    )
    lookback_days = 253  # need closes t-252 .. t inclusive

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        # Formation window: closes t-252 .. t-21 (skips the latest month).
        cumret = close.shift(21) / close.shift(252) - 1.0
        rets = close.pct_change()
        # Daily returns inside the window: [t-251, t-21] = 231 returns.
        # rolling(231) at t covers [t-230, t]; shifting by 21 moves it to
        # [t-251, t-21].
        up = (rets > 0).rolling(231, min_periods=231).sum().shift(21)
        down = (rets < 0).rolling(231, min_periods=231).sum().shift(21)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_up = up / 231.0
            pct_down = down / 231.0
            id_metric = np.sign(cumret) * (pct_down - pct_up)
        score = -id_metric
        # Winners only: a non-winner can never outrank a winner. ID lives
        # in [-1, 1], so -10.0 sits safely below every real score. Dates
        # with too little history stay NaN (never stored).
        score = score.where((cumret > 0) | cumret.isna(), -10.0)
        return score.rename(self.name)


class OvernightIntraday(FundamentalSignal):
    """Intraday strength minus overnight strength (institutional flow).

    Each day's return is split in two: the overnight leg (previous close
    to today's open — dominated by retail orders and headlines) and the
    intraday leg (open to close — dominated by institutions working big
    orders through the day). Both legs are compounded over ~126 trading
    days, skipping the most recent 5 sessions to dodge short-term noise,
    then Z-scored across stocks each day. The signal is

        Z(intraday) - Z(overnight)

    A high score means steady daytime buying that the overnight crowd
    hasn't chased yet — quiet institutional accumulation.

    Dividend honesty: prices here are raw (unadjusted), so on an ex-date
    the open naturally prints lower by the dividend amount. The dividend
    is added back into the overnight leg, otherwise every payer would
    show a fake negative overnight return on its ex-date. Uses the
    platform's dividends table (filled by the price download).

    It subclasses FundamentalSignal only for plumbing: like the
    fundamental signals it needs the database (dividends) and the whole
    cross-section at once, so it goes through compute_all.

    Direction: higher = stronger daytime vs overnight drift = more
    attractive long.
    """

    name = "overnight_intraday"
    description = (
        "Whether the stock has been bought steadily during the trading day "
        "without the overnight crowd piling in. Strong daytime buying "
        "with quiet overnights hints at patient institutional buying — "
        "a high score favors those stocks."
    )
    lookback_days = 132  # 126 formation sessions + 5 skipped + 1

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = shared if shared is not None else _SharedInputs(tickers, db_path)
        closes = sh.prices
        if closes.empty:
            return pd.DataFrame()
        opens = _load_opens(tickers, db_path).reindex(
            index=closes.index, columns=closes.columns
        )
        divs = _dividend_matrix(tickers, closes.index, db_path)
        overnight, intraday = _session_legs(opens, closes, divs)
        # Compound each leg over 126 sessions, skipping the latest 5:
        # rolling(126) at t-5 covers [t-130, t-5]; shift(5) aligns it to t.
        # log-sum is the fast vectorized equivalent of prod(1+r)-1.
        on_cum = _compound(overnight, 126).shift(5)
        in_cum = _compound(intraday, 126).shift(5)
        return _zscore(in_cum) - _zscore(on_cum)


def _session_legs(
    opens: pd.DataFrame, closes: pd.DataFrame, divs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily returns into overnight and intraday legs.

    overnight_t = (open_t + div_t) / close_{t-1} - 1
    intraday_t  = close_t / open_t - 1

    Prices here are raw (unadjusted), so on an ex-date the open naturally
    prints lower by the dividend amount. The dividend is added back into
    the overnight leg — otherwise every payer would show a fake negative
    overnight return on its ex-date.
    """
    prev_close = closes.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = (opens + divs) / prev_close.replace(0, np.nan) - 1.0
        intraday = closes / opens.replace(0, np.nan) - 1.0
    return overnight, intraday


def _compound(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compounded return over a rolling window: prod(1+r) - 1.

    Computed as exp(sum(log1p(r))) - 1 — identical math, fully
    vectorized (rolling.apply with a Python prod is far slower on a
    500-stock universe). NaN anywhere in the window -> NaN out.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log1p(returns)
    summed = logs.rolling(window, min_periods=window).sum()
    with np.errstate(over="ignore", invalid="ignore"):
        return np.expm1(summed)


def _dividend_matrix(
    tickers: list[str], dates: pd.DatetimeIndex, db_path: str | Path
) -> pd.DataFrame:
    """Per-share cash dividends as dates x tickers (0 where none)."""
    divs = load_dividends(tickers, path=db_path)
    mat = pd.DataFrame(0.0, index=dates, columns=tickers)
    if divs.empty:
        return mat
    piv = divs.pivot_table(
        index="ex_date", columns="ticker", values="amount", aggfunc="sum"
    )
    piv.index = pd.to_datetime(piv.index)
    common = mat.index.intersection(piv.index)
    cols = [c for c in piv.columns if c in mat.columns]
    mat.loc[common, cols] = piv.loc[common, cols].fillna(0.0)
    return mat


# --- Twin momentum: SUE machinery -------------------------------------------

# Diluted EPS lives in us-gaap (not dei). No basic-EPS fallback: mixing
# diluted and basic across periods would inject fake earnings surprises
# when the source changes. Tickers reporting only basic EPS simply get
# no SUE score.
TAG_EPS_DILUTED = "eps_diluted"


def _quarterly_eps_events(facts: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """(filed_date, SUE) events from 10-Q diluted-EPS facts.

    Only 10-Q/10-Q/A filings are used (the spec's conservative choice —
    Q4 is reported in the 10-K, so this signal only fires on Q1-Q3).
    Stored 10-Q income-statement values are year-to-date, exactly like
    the platform's other flow tags (net_income etc.), so each quarter's
    own EPS is its YTD minus the previous quarter's YTD within the same
    fiscal year (detected by the quarter number rolling back to Q1).

    SUE (standardized unexpected EPS, Bernard & Thomas / Foster, Olsen
    & Shevlin): for the latest known quarter, UE = EPS_q - EPS_sameQ_ly
    (the same quarter of the prior fiscal year — the seasonal lag, since
    only Q1-Q3 exist the "4 quarters back" of the textbooks would land
    on the wrong season). drift and scale are the mean and sample
    standard deviation of the up-to-8 most recent quarterly UEs.
    SUE = (UE_latest - drift) / scale; NaN when fewer than 8 quarterly
    observations or fewer than 2 valid UEs. When past surprises are
    (near-)identical the standardized surprise is defined as zero
    ("no surprise"), not an exploding ratio — see _sue_from_quarters.

    Missing data is not an error: a ticker with no diluted-EPS facts
    (or a facts frame without the expected columns) simply yields no
    events, so its SUE stays NaN and it is skipped by the strategy.
    """
    required = {"form", "fp", "filed_date", "period_end", "value"}
    if facts.empty or not required.issubset(set(facts.columns)):
        return []
    f = facts[
        facts["form"].isin(["10-Q", "10-Q/A"])
        & facts["fp"].isin(["Q1", "Q2", "Q3"])
    ].copy()
    if f.empty:
        return []
    f["filed_date"] = pd.to_datetime(f["filed_date"])
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f.sort_values("filed_date").reset_index(drop=True)
    qn_of = {"Q1": 1, "Q2": 2, "Q3": 3}
    # quarter_end -> (ytd_value, quarter_number); latest filing wins.
    quarters: dict[pd.Timestamp, tuple[float, int]] = {}
    events: list[tuple[pd.Timestamp, float]] = []
    for filed, grp in f.groupby("filed_date", sort=True):
        for _, r in grp.iterrows():
            fp = str(r.get("fp", "")).strip().upper()
            if fp not in qn_of:
                continue
            quarters[pd.Timestamp(r["period_end"])] = (
                float(r["value"]),
                qn_of[fp],
            )
        events.append((pd.Timestamp(filed), _sue_from_quarters(quarters)))
    return events


def _sue_from_quarters(
    quarters: dict[pd.Timestamp, tuple[float, int]],
) -> float:
    """SUE from known quarter-ends -> (ytd EPS, quarter number)."""
    items = sorted(quarters.items())  # chronological by quarter end
    # De-YTD into quarterly increments, resetting at each fiscal year.
    # A quarter whose increment spans more than one quarter (a skipped
    # filing) is marked NaN rather than compared across seasons.
    qeps: list[tuple[pd.Timestamp, int, float]] = []  # (end, qn, eps)
    prev_ytd = 0.0
    prev_qn: int | None = None
    for end, (ytd, qn) in items:
        if prev_qn is None or qn <= prev_qn:
            prev_ytd = 0.0  # new fiscal year: Q1's YTD is just Q1
            expected = 1
        else:
            expected = prev_qn + 1
        inc = ytd - prev_ytd if qn == expected else float("nan")
        qeps.append((end, qn, inc))
        prev_ytd = ytd
        prev_qn = qn
    # Need at least 8 quarterly observations to judge a surprise.
    obs = [(e, q, v) for e, q, v in qeps if not pd.isna(v)]
    if len(obs) < 8:
        return float("nan")
    # Seasonal surprises: each quarter vs the same quarter last year.
    ues: list[float] = []
    for i in range(len(obs)):
        _end_i, q_i, v_i = obs[i]
        for j in range(i - 1, -1, -1):
            if obs[j][1] == q_i:
                ues.append(v_i - obs[j][2])
                break
    ues = [u for u in ues[-8:] if not pd.isna(u)]
    if len(ues) < 2:
        return float("nan")
    drift = float(np.mean(ues))
    scale = float(np.std(ues, ddof=1))
    if not np.isfinite(scale):
        return float("nan")
    # Numerical guard: when past surprises are (near-)identical, their
    # spread is floating-point noise, not information. The latest surprise
    # then equals its own history exactly -> standardized surprise is
    # zero ("no surprise"), not an exploding ratio.
    tiny = 1e-9 * max(1.0, abs(drift))
    if scale <= tiny:
        return 0.0 if abs(ues[-1] - drift) <= tiny else float("nan")
    return (ues[-1] - drift) / scale


def _sue_matrix(
    tickers: list[str], dates: pd.DatetimeIndex, db_path: str | Path
) -> pd.DataFrame:
    """Standardized unexpected EPS: dates x tickers, point-in-time.

    Each ticker's filings are walked chronologically; a filing dated
    after a signal date never influences that date's SUE (merge_asof
    backward from filing dates onto the trading grid).
    """
    frames = _load_fact_frames(tickers, [TAG_EPS_DILUTED], db_path)
    cols = {}
    for t in tickers:
        tn = normalize_ticker(t)
        facts = frames.get((tn, TAG_EPS_DILUTED), pd.DataFrame())
        events = _quarterly_eps_events(facts)
        if not events:
            cols[t] = pd.Series(np.nan, index=dates)
            continue
        ev = pd.DataFrame(events, columns=["filed_date", "sue"]).sort_values(
            "filed_date"
        )
        grid = pd.DataFrame({"date": dates}).sort_values("date")
        merged = pd.merge_asof(
            grid, ev, left_on="date", right_on="filed_date", direction="backward"
        )
        cols[t] = pd.Series(merged["sue"].to_numpy(), index=dates)
    return pd.DataFrame(cols, index=dates)


def _eps_diluted_present(tickers: list[str], db_path: str | Path) -> bool:
    """True when any of the tickers has any eps_diluted facts stored.

    Used to tell "the signal ran but its earnings data is missing"
    apart from "the signal genuinely scored nothing".
    """
    path = Path(db_path)
    if not path.exists():
        return False
    tickers = [normalize_ticker(t) for t in tickers]
    if not tickers:
        return False
    qmarks = ",".join("?" for _ in tickers)
    try:
        with connect_db(path) as conn:
            row = conn.execute(
                "SELECT 1 FROM fundamentals WHERE tag = ?"
                f" AND ticker IN ({qmarks}) LIMIT 1",
                [TAG_EPS_DILUTED, *tickers],
            ).fetchone()
    except Exception:
        return False  # e.g. the fundamentals table is not created yet
    return row is not None


class TwinMomentum(FundamentalSignal):
    """Price trend AND earnings surprise must both confirm.

    Two engines, both required (the "twin momentum" idea):
    1. Price leg: the 12-month return skipping the most recent month
       (the platform's standard momentum), Z-scored across stocks.
    2. Earnings leg: standardized unexpected diluted EPS (SUE) from
       10-Q filings — how big this quarter's earnings surprise was
       versus the stock's own history of surprises — Z-scored across
       stocks.

    A stock qualifies only if BOTH legs sit above their cross-sectional
    mean that day; otherwise it is disqualified (NaN — never ranked,
    never bought). Qualifiers are ranked by the average of the two
    Z-scores. The dual requirement filters out price runs with no
    earnings behind them — the speculative multiple expansion that
    crashes hardest.

    It subclasses FundamentalSignal for plumbing (SEC data +
    cross-section), reusing the no-peeking machinery.

    Direction: higher = stronger confirmed momentum = more attractive
    long. Disqualified stocks simply have no score.

    Data note: the diluted-EPS tag is new — after pulling this version,
    re-run the fundamentals download so the tag is populated; until
    then this signal stores nothing and the strategy reports that
    plainly instead of failing.
    """

    name = "twin_momentum"
    description = (
        "Buys stocks where the price trend AND the earnings surprise both "
        "point up. A rising price backed by a real earnings beat is much "
        "sturdier than a price spike with nothing behind it — stocks "
        "failing either test are skipped entirely."
    )
    lookback_days = 253  # price leg needs t-252

    def compute_all(
        self,
        tickers: list[str],
        db_path: str | Path = DEFAULT_DB,
        *,
        shared: "_SharedInputs | None" = None,
    ) -> pd.DataFrame:
        tickers = [normalize_ticker(t) for t in tickers]
        sh = shared if shared is not None else _SharedInputs(tickers, db_path)
        prices = sh.prices
        if prices.empty:
            return pd.DataFrame()
        mom = prices.shift(21) / prices.shift(252) - 1.0
        z_mom = _zscore(mom)
        sue = _sue_matrix(tickers, prices.index, db_path)
        z_sue = _zscore(sue)
        mean_mom = z_mom.mean(axis=1)
        mean_sue = z_sue.mean(axis=1)
        both_ok = z_mom.gt(mean_mom, axis=0) & z_sue.gt(mean_sue, axis=0)
        return ((z_mom + z_sue) / 2).where(both_ok)

    def data_gap_note(
        self, tickers: list[str], db_path: str | Path = DEFAULT_DB
    ) -> str | None:
        """Plain-English note when the earnings leg has no data at all.

        Optional convention: any signal may define
        ``data_gap_note(tickers, db_path)``; the UI's signal summaries
        show the returned text instead of a bare "OK" when the signal
        ran but its key input data is missing entirely. Returns None
        when there is no gap.
        """
        if _eps_diluted_present(tickers, db_path):
            return None
        return (
            "No earnings data yet — on the Market data page, tick "
            "'Fundamentals (SEC filings)' and press Download (or 'Check "
            "for new filings') so the diluted-EPS figures are fetched, "
            "then compute this signal again."
        )


__all__ = [
    "ContinuousMomentum",
    "OvernightIntraday",
    "TwinMomentum",
    "_zscore",
    "_session_legs",
]
