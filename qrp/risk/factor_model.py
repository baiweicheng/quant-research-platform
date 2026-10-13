"""Statistical factor risk model (Phase P4).

Where Phase P3's shrinkage covariance treats every stock pair on its own,
this module *learns the handful of shared market patterns that move groups
of stocks together* — the broad market rising or falling, tech stocks
moving as a pack, and so on — so the optimizer knows which bets are really
the *same* bet in disguise.

How it works, in plain English:

1. Take the trailing ~252 trading days of daily returns for the requested
   stocks (only prices known on or before ``asof`` — the same no-peeking
   rule as everywhere else).
2. Run PCA (principal component analysis) on those returns: it finds the
   shared patterns, ordered by how much of the stocks' movement each one
   explains. Keep the smallest number of patterns that together explain
   ~80% of past movement, capped at 20 (the K-selection rule, documented
   below). These are *statistical* factors — anonymous patterns the math
   found, not named economic sectors like "technology".
3. Shrink the factors' covariance with Ledoit-Wolf (the same shrinkage
   P3 uses) and shrink each stock's leftover (idiosyncratic) variance
   toward the average leftover variance.
4. Assemble the model covariance as
   ``Sigma = loadings x factor_cov x loadings' + diag(specific risk)`` —
   the classic factor-model form. By construction it is symmetric and
   positive definite.

Why the optimizer needs it: a naive optimizer without a risk model is an
"error maximizer" — it loads up on whatever the raw estimates got wrong.
The factor structure tells it that ten stocks riding the same pattern are
one big bet, not ten small ones, so it sizes positions against the true
shared risk.

Honest limits (also shown in the UI):

- Statistical factors are anonymous: the model knows "pattern 2 moves
  these stocks together" but not *why* (no sector or style labels).
- It learns from the past year, so a brand-new regime (a crash, a sector
  rotation) is invisible until it shows up in the trailing window.
- Research benchmarks put DIY statistical models at ~80-90% of a licensed
  risk model's (Barra/Axioma) forecasting value — most of the benefit,
  free, without the curated economic exposures.

K-selection rule (documented, deterministic): the smallest K whose
cumulative explained variance reaches ``variance_threshold`` (default
80%), but at least 1, at most ``max_factors`` (default 20), and never
more than the data's effective rank (components with economically zero
variance are dropped).

Plain-English data rules (never crash, never peek) mirror P3:

- Only prices dated on or before ``asof`` are used.
- A stock needs at least ``min_observations`` daily returns (default 30)
  and non-zero variance to be included; thinner histories are dropped
  with a plain-English note.
- With fewer than 2 usable stocks, no shared pattern can be learned, so
  the module falls back to a diagonal matrix (each stock's own variance,
  independence assumed) and says so plainly — the same fallback P3 uses.
- With no usable price history at all, it raises a ValueError with a
  plain-English explanation ("download prices first") instead of a
  traceback.

No new heavy dependencies: PCA is implemented with numpy's SVD (no
sklearn), and shrinkage reuses P3's ``ledoit_wolf_shrinkage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qrp.data.store import DEFAULT_DB, connect_db
from qrp.data.yahoo import normalize_ticker
from qrp.portfolio.covariance import (
    MIN_DAILY_VARIANCE,
    TRADING_DAYS_PER_YEAR,
    CovarianceModel,
    asof_day,
    guard_estimation_window,
    ledoit_wolf_shrinkage,
)
from qrp.portfolio.optimizer import benchmark_weights_many

DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MIN_OBSERVATIONS = 30
#: Keep the smallest number of factors covering this share of past variance.
DEFAULT_VARIANCE_THRESHOLD = 0.80
#: Never keep more factors than this, however the threshold falls.
DEFAULT_MAX_FACTORS = 20
#: Annualized specific-variance floor: a stock's idiosyncratic risk is
#: never estimated as exactly zero (that would tell the optimizer the
#: stock is risk-free once the shared patterns are removed).
SPECIFIC_VAR_FLOOR_ANN = 1e-8


@dataclass
class FactorFit:
    """What the factor model learned, as of a date.

    tickers: normalized tickers the fit covers (index/columns order).
    asof: YYYY-MM-DD; only data on or before this date was used.
    n_factors: K, the number of shared patterns kept.
    variance_explained: per-factor share of past return variance
        (length K, ordered largest first).
    total_variance_explained: their sum — "how much of past movement the
        shared patterns account for".
    loadings: N x K DataFrame — each stock's exposure to each pattern.
        Columns are "Factor 1".."Factor K".
    factor_cov: K x K DataFrame, *annualized* shrunk covariance of the
        factor returns.
    specific_var: per-stock *annualized* idiosyncratic variance (the
        movement the shared patterns don't explain), shrunk toward the
        average.
    trading_days_used: how many daily returns went into the fit.
    notes: plain-English strings describing any degradation.
    """

    tickers: list[str]
    asof: str
    n_factors: int
    variance_explained: list[float]
    total_variance_explained: float
    loadings: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    trading_days_used: int
    notes: list[str] = field(default_factory=list)


@dataclass
class CalibrationCheck:
    """Did past tracking-error forecasts match what actually happened?

    rows: one dict per check — {"asof": fit date, "forecast_te": ...,
        "realized_te": ...}, both annualized.
    mean_forecast / mean_realized: averages across the checks.
    verdict: one plain-English sentence ("roughly in line" / "running
        hot" / "running cold").
    notes: plain-English strings (e.g. checks skipped for thin history).
    """

    rows: list[dict] = field(default_factory=list)
    mean_forecast: float = 0.0
    mean_realized: float = 0.0
    verdict: str = ""
    notes: list[str] = field(default_factory=list)


def fit_factor_model(
    returns: pd.DataFrame,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    max_factors: int = DEFAULT_MAX_FACTORS,
    asof: str = "",
) -> FactorFit:
    """Fit the statistical factor model on a T x N daily-returns panel.

    Pure function (no database): PCA via numpy SVD, K chosen by the
    documented K-selection rule, Ledoit-Wolf shrinkage on the factor
    covariance and on the residual (specific) variances.

    Args:
        returns: T x N DataFrame of daily simple returns (columns are
            tickers). Rows with any NaN are dropped; every column needs
            non-zero variance.
        variance_threshold: keep the smallest K covering this share of
            variance (0.80 = "the patterns behind 80% of past movement").
        max_factors: hard cap on K.
        asof: recorded on the fit (no effect on the math).

    Returns:
        FactorFit with annualized factor covariance and specific risk.

    Raises:
        ValueError: plain-English, when fewer than 2 usable return
            series remain or nothing moves at all.
    """
    clean = returns.dropna(axis=0, how="any")
    t, n = clean.shape
    tickers = list(clean.columns)
    if n < 2 or t < 2:
        raise ValueError(
            "The risk model needs at least 2 stocks with at least 2 daily "
            f"returns each to learn shared patterns from (got {n} "
            f"stock(s), {t} day(s))."
        )
    X = clean.to_numpy(dtype=float)
    X = X - X.mean(axis=0)

    # SVD of the demeaned panel: X = U diag(s) V'. Columns of V are the
    # principal directions; s_k^2 / T is the variance each explains.
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    comp_var = s**2 / t
    total_var = float(comp_var.sum())
    if not np.isfinite(total_var) or total_var <= MIN_DAILY_VARIANCE:
        raise ValueError(
            "These stocks show no price movement in the lookback window, "
            "so there is nothing for the risk model to learn from."
        )
    ratios = comp_var / total_var
    # K-selection rule: smallest K reaching the variance threshold...
    k = int(np.searchsorted(np.cumsum(ratios), variance_threshold)) + 1
    # ...capped by the max, and by the effective rank (components with
    # economically zero variance carry no information).
    rank = int((comp_var > MIN_DAILY_VARIANCE).sum())
    k = max(1, min(k, max_factors, rank, n, t))
    k = min(k, len(s))

    B = vt[:k].T  # N x K loadings (unit eigenvectors)
    F = X @ B  # T x K factor returns; sample cov is diag(s[:k]^2 / T)
    factor_names = [f"Factor {i + 1}" for i in range(k)]
    factor_rets = pd.DataFrame(F, columns=factor_names)
    # Ledoit-Wolf on the factor returns (annualized by the helper).
    factor_cov = ledoit_wolf_shrinkage(factor_rets)

    # Specific risk: what the K patterns leave unexplained, per stock.
    E = X - F @ B.T
    resid = pd.DataFrame(E, columns=tickers)
    notes: list[str] = []
    try:
        resid_cov = ledoit_wolf_shrinkage(resid)
        specific = np.diag(resid_cov.to_numpy(dtype=float))
    except ValueError:
        # Shrinkage fails when (at least) one stock's residual variance is
        # zero — e.g. two tickers with identical return series. Flooring
        # EVERY stock then would pretend no stock has idiosyncratic risk.
        # Fall back per stock instead: each keeps its own sample variance,
        # and only the degenerate ones take the floor.
        sample_var = np.var(E, axis=0, ddof=1) * TRADING_DAYS_PER_YEAR
        sample_var = np.where(np.isfinite(sample_var), sample_var, 0.0)
        specific = np.maximum(sample_var, SPECIFIC_VAR_FLOOR_ANN)
        notes.append(
            "The shared patterns explain (nearly) all past movement for "
            "at least one stock, so that stock's leftover risk was set to "
            "a tiny floor rather than zero — every other stock keeps its "
            "own measured leftover risk."
        )
    specific = np.maximum(specific, SPECIFIC_VAR_FLOOR_ANN)

    explained = [float(r) for r in ratios[:k]]
    return FactorFit(
        tickers=tickers,
        asof=asof,
        n_factors=k,
        variance_explained=explained,
        total_variance_explained=float(sum(explained)),
        loadings=pd.DataFrame(B, index=tickers, columns=factor_names),
        factor_cov=factor_cov,
        specific_var=pd.Series(specific, index=tickers),
        trading_days_used=t,
        notes=notes,
    )


def factor_covariance_matrix(fit: FactorFit) -> pd.DataFrame:
    """Assemble the annualized model covariance: B F B' + D.

    loadings (N x K) x factor covariance (K x K) x loadings' plus the
    diagonal specific-risk matrix. Positive definite by construction
    (PSD factor term + strictly positive diagonal), symmetrized against
    floating-point noise.
    """
    B = fit.loadings.to_numpy(dtype=float)
    Fc = fit.factor_cov.to_numpy(dtype=float)
    D = np.diag(fit.specific_var.to_numpy(dtype=float))
    sigma = B @ Fc @ B.T + D
    sigma = 0.5 * (sigma + sigma.T)
    return pd.DataFrame(sigma, index=fit.tickers, columns=fit.tickers)


def describe_factors(fit: FactorFit, top_n: int = 5) -> list[str]:
    """One plain-English sentence per top factor.

    Each line says how much past movement the pattern explains and which
    stocks it pushes together (or in opposite directions), naming the
    three strongest loadings. Factor 1 is flagged as the broad-market
    pattern when most stocks load on it with the same sign — the usual
    shape of the first statistical factor.
    """
    lines: list[str] = []
    for k in range(min(top_n, fit.n_factors)):
        col = fit.loadings.iloc[:, k]
        share = fit.variance_explained[k]
        top = col.abs().sort_values(ascending=False).head(3)
        pos = [t for t in top.index if col[t] > 0]
        neg = [t for t in top.index if col[t] <= 0]
        if k == 0 and max(float((col > 0).mean()), float((col < 0).mean())) >= 0.8:
            kind = (
                "moves with the broad market — most stocks rise and fall "
                "with it"
            )
        elif pos and neg:
            kind = (
                f"pushes {', '.join(pos)} one way and "
                f"{', '.join(neg)} the other"
            )
        else:
            kind = f"moves {', '.join(top.index)} in the same direction"
        lines.append(
            f"Factor {k + 1}: explains {share:.0%} of past movement; {kind}."
        )
    return lines


class StatisticalFactorCovariance:
    """Factor-model covariance from trailing daily returns in the market DB.

    Implements P3's ``CovarianceModel`` Protocol, so it slots into the
    optimizer as a drop-in alternative to ``LedoitWolfCovariance``.
    Reads up to ``lookback_days`` trading days of closes on or before
    ``asof`` for each ticker in one SQL query (the same query shape P3
    uses — duplicated here on purpose so P3's module stays untouched),
    converts to daily simple returns, and fits the factor model.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        max_factors: int = DEFAULT_MAX_FACTORS,
    ) -> None:
        self.db_path = Path(db_path)
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.variance_threshold = variance_threshold
        self.max_factors = max_factors
        self.notes: list[str] = []

    def _load_return_frame(
        self, tickers: list[str], asof: str
    ) -> pd.DataFrame:
        """T x N DataFrame of daily simple returns, dates on or before asof.

        One SQL query: the trailing `lookback_days` trading days of closes
        per ticker, pivoted wide. Columns are normalized tickers.
        """
        tickers = [normalize_ticker(t) for t in tickers]
        if not tickers or not self.db_path.exists():
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in tickers)
        query = (
            "SELECT ticker, date, close FROM daily_bars"
            f" WHERE ticker IN ({placeholders}) AND date <= ?"
            " ORDER BY date"
        )
        with connect_db(self.db_path) as conn:
            has_table = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table'"
                    " AND name='daily_bars'"
                ).fetchone()
                is not None
            )
            if not has_table:
                return pd.DataFrame()
            df = pd.read_sql_query(query, conn, params=[*tickers, asof_day(asof)])
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        )
        wide = wide.sort_index().tail(self.lookback_days)
        returns = wide.pct_change().iloc[1:]
        returns.index = pd.to_datetime(returns.index)
        return returns.sort_index()

    def _usable_returns(
        self, tickers: list[str], asof: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return frame restricted to tickers with enough history.

        Sets ``self.notes`` (dropped tickers are reported plainly) and
        raises a plain-English ValueError when nothing is usable.
        """
        self.notes = []
        tickers = [normalize_ticker(t) for t in tickers]
        if not tickers:
            raise ValueError(
                "No tickers to estimate a covariance for — the optimizer "
                "needs at least one scored stock with price history."
            )
        rets = self._load_return_frame(tickers, asof)
        usable = [
            c
            for c in rets.columns
            if int(rets[c].notna().sum()) >= self.min_observations
            and float(rets[c].var()) > MIN_DAILY_VARIANCE
        ]
        dropped = [c for c in rets.columns if c not in usable]
        never_seen = [t for t in tickers if t not in rets.columns]
        if dropped or never_seen:
            missing = sorted(set(dropped) | set(never_seen))
            shown = ", ".join(missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            self.notes.append(
                f"{len(missing)} stock(s) had too little price history to "
                f"estimate risk from ({shown}{more}) and were left out of "
                "the optimization."
            )
        if not usable:
            raise ValueError(
                "No usable price history for any of these stocks — "
                "download prices first (Market data page), then try again."
            )
        # Window-collapse guard (same as the shrinkage path): a
        # stub-history ticker would drag the shared estimation window down
        # for every stock. Left-out names are excluded, with a note.
        usable, thin = guard_estimation_window(rets, usable)
        if thin:
            shown = ", ".join(sorted(thin)[:5])
            more = f" and {len(thin) - 5} more" if len(thin) > 5 else ""
            self.notes.append(
                f"{len(thin)} stock(s) had much shorter price history "
                f"than the rest ({shown}{more}) — including them would "
                "shrink the risk estimate for every stock down to their "
                "short window, so they were left out of the optimization."
            )
        median_days = float(
            np.median([rets[c].notna().sum() for c in usable])
        )
        if median_days < self.lookback_days:
            self.notes.append(
                f"Only about {median_days:.0f} trading days of history "
                "were available (fewer than the usual "
                f"{self.lookback_days}) — fewer days means a noisier read "
                "of how stocks move together."
            )
        return rets, usable

    def fit_report(self, tickers: list[str], asof: str) -> FactorFit:
        """Fit details for the UI: factor count, loadings, variance shares.

        Raises a plain-English ValueError when fewer than 2 stocks have
        enough history to learn shared patterns from.
        """
        rets, usable = self._usable_returns(tickers, asof)
        if len(usable) < 2:
            raise ValueError(
                "The risk model needs at least 2 stocks with enough price "
                "history to learn shared patterns from — only "
                f"{len(usable)} qualified."
            )
        fit = fit_factor_model(
            rets[usable],
            variance_threshold=self.variance_threshold,
            max_factors=self.max_factors,
            asof=asof,
        )
        fit.notes = [*self.notes, *fit.notes]
        self.notes = list(fit.notes)
        self.notes.append(
            f"Risk model: statistical factor model — {fit.n_factors} "
            f"shared pattern(s) covering "
            f"{fit.total_variance_explained:.0%} of past price movement."
        )
        return fit

    def covariance(
        self, tickers: list[str], asof: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Annualized factor-model covariance for `tickers` as of `asof`.

        Never crashes on thin data: drops under-observed tickers (noted),
        falls back to a diagonal matrix with < 2 survivors (noted, the
        same fallback P3 uses), and raises a plain-English ValueError only
        when nothing is usable.
        """
        rets, usable = self._usable_returns(tickers, asof)
        if len(usable) < 2:
            t = usable[0]
            var = float(rets[t].var())
            var_ann = var * TRADING_DAYS_PER_YEAR if var > 0 else 0.09
            self.notes.append(
                "Only one stock had enough history, so stocks are treated "
                "as moving independently of each other (no shared pattern "
                "could be learned)."
            )
            return pd.DataFrame([[var_ann]], index=usable, columns=usable)
        fit = fit_factor_model(
            rets[usable],
            variance_threshold=self.variance_threshold,
            max_factors=self.max_factors,
            asof=asof,
        )
        self.notes.extend(fit.notes)
        self.notes.append(
            f"Risk model: statistical factor model — {fit.n_factors} "
            f"shared pattern(s) covering "
            f"{fit.total_variance_explained:.0%} of past price movement."
        )
        return factor_covariance_matrix(fit)


def tracking_error(
    weights: dict[str, float],
    bench_weights: dict[str, float],
    asof: str,
    model: CovarianceModel,
) -> float:
    """Forecast annualized tracking error vs a benchmark, ex ante.

    Tracking error is the predicted yearly volatility of
    (portfolio return − benchmark return) — "how far your portfolio's
    returns are expected to wander from the benchmark's". A 5% tracking
    error means roughly ±5% a year vs the benchmark.

    Args:
        weights: ticker -> portfolio weight (fractions, need not sum to 1).
        bench_weights: ticker -> benchmark weight (P3's
            ``benchmark_weights`` — reused here, never duplicated).
        asof: YYYY-MM-DD; the covariance uses only data on or before it.
        model: any ``CovarianceModel`` (the factor model or P3's
            shrinkage model).

    Returns:
        Annualized tracking error as a fraction (0.05 = 5%). Tickers the
        model could not estimate are left out of the active bet.
    """
    w = {normalize_ticker(t): float(v) for t, v in weights.items()}
    b = {normalize_ticker(t): float(v) for t, v in bench_weights.items()}
    tickers = sorted(set(w) | set(b))
    if not tickers:
        raise ValueError(
            "No weights to measure tracking error for — build a portfolio "
            "first."
        )
    sigma = model.covariance(tickers, asof)
    avail = [t for t in tickers if t in sigma.index]
    if not avail:
        raise ValueError(
            "None of these stocks had enough price history to estimate "
            "tracking error from — download prices first."
        )
    active = np.array([w.get(t, 0.0) - b.get(t, 0.0) for t in avail])
    S = sigma.loc[avail, avail].to_numpy(dtype=float)
    S = 0.5 * (S + S.T)
    return float(np.sqrt(max(active @ S @ active, 0.0)))


def portfolio_volatility(
    weights: dict[str, float],
    asof: str,
    model: CovarianceModel,
) -> float:
    """Forecast annualized portfolio volatility, ex ante.

    "How much the portfolio's value is expected to swing per year"
    (16% ≈ a typical stock-market year; 25% is a bumpy ride). Same
    mechanics as ``tracking_error`` without the benchmark leg.
    """
    w = {normalize_ticker(t): float(v) for t, v in weights.items()}
    tickers = sorted(w)
    if not tickers:
        raise ValueError(
            "No weights to measure volatility for — build a portfolio first."
        )
    sigma = model.covariance(tickers, asof)
    avail = [t for t in tickers if t in sigma.index]
    if not avail:
        raise ValueError(
            "None of these stocks had enough price history to estimate "
            "volatility from — download prices first."
        )
    vec = np.array([w[t] for t in avail])
    S = sigma.loc[avail, avail].to_numpy(dtype=float)
    S = 0.5 * (S + S.T)
    return float(np.sqrt(max(vec @ S @ vec, 0.0)))


def _trading_dates(
    tickers: list[str], asof: str, db_path: str | Path
) -> list[str]:
    """Sorted trading dates on or before `asof` with any bar present."""
    tickers = [normalize_ticker(t) for t in tickers]
    path = Path(db_path)
    if not tickers or not path.exists():
        return []
    placeholders = ",".join("?" for _ in tickers)
    query = (
        "SELECT DISTINCT date FROM daily_bars"
        f" WHERE ticker IN ({placeholders}) AND date <= ?"
        " ORDER BY date"
    )
    with connect_db(path) as conn:
        rows = conn.execute(query, [*tickers, asof_day(asof)]).fetchall()
    return [r[0] for r in rows]


def _realized_te(
    weights: dict[str, float],
    bench: dict[str, float],
    window_end: str,
    horizon_days: int,
    tickers: list[str],
    db_path: str | Path,
) -> float:
    """Annualized std of (portfolio − benchmark) daily returns.

    Uses the `horizon_days` trading days ending on `window_end`, from
    prices strictly after the forecast date — what *actually* happened.
    """
    tickers = [normalize_ticker(t) for t in tickers]
    placeholders = ",".join("?" for _ in tickers)
    query = (
        "SELECT ticker, date, close FROM daily_bars"
        f" WHERE ticker IN ({placeholders}) AND date <= ?"
        " ORDER BY date"
    )
    with connect_db(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[*tickers, asof_day(window_end)])
    if df.empty:
        return 0.0
    wide = df.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()
    # One extra day before the window so the first return is defined.
    rets = wide.pct_change().iloc[1:].tail(horizon_days)
    cols = [c for c in rets.columns if c in weights or c in bench]
    if len(rets) < 2 or not cols:
        return 0.0
    w = np.array([float(weights.get(c, 0.0)) for c in cols])
    b = np.array([float(bench.get(c, 0.0)) for c in cols])
    active = (rets[cols].to_numpy(dtype=float) @ (w - b))
    active = active[np.isfinite(active)]
    if len(active) < 2:
        return 0.0
    return float(np.std(active, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def realized_vs_predicted(
    weights: dict[str, float],
    bench_weights: dict[str, float],
    asof: str,
    model: CovarianceModel,
    db_path: str | Path = DEFAULT_DB,
    n_checkpoints: int = 4,
    horizon_days: int = 63,
) -> CalibrationCheck:
    """Honest check: did past tracking-error forecasts match reality?

    Stands at several past dates (default: 4 quarterly checkpoints over
    the trailing year), forecasts the tracking error using only data
    knowable *then* (no peeking), then measures what the portfolio-vs-
    benchmark actually did over the following ~3 months. Compares the
    averages and renders a plain-English verdict.

    Checkpoints without enough history are skipped and noted; with no
    checkable history at all it raises a plain-English ValueError.
    """
    w = {normalize_ticker(t): float(v) for t, v in weights.items()}
    b = {normalize_ticker(t): float(v) for t, v in bench_weights.items()}
    tickers = sorted(set(w) | set(b))
    if not tickers:
        raise ValueError(
            "No weights to check forecasts for — build a portfolio first."
        )
    dates = _trading_dates(tickers, asof, db_path)
    notes: list[str] = []
    rows: list[dict] = []
    need_before = int(getattr(model, "lookback_days", DEFAULT_LOOKBACK_DAYS))
    # Collect the checkpoint fit dates first so the benchmark weights for
    # all of them come from ONE batched query (audit 2.7) instead of one
    # per-ticker-per-date round trip.
    fit_dates: list[str] = []
    ends: list[int] = []
    for i in range(1, n_checkpoints + 1):
        # `end` is the fit date's index; the realized window is the
        # `horizon_days` trading days right after it.
        end = len(dates) - 1 - i * horizon_days
        if end < need_before:
            break  # not enough history left for earlier checkpoints
        if end + horizon_days >= len(dates):
            continue  # realized window would run past `asof`
        fit_dates.append(dates[end])
        ends.append(end)
    bench_by_date = benchmark_weights_many(tickers, fit_dates, db_path)
    tried = 0
    for fit_date, end in zip(fit_dates, ends):
        tried += 1
        try:
            bench, _note = bench_by_date[fit_date]
            forecast = tracking_error(w, bench, fit_date, model)
            realized = _realized_te(
                w, bench, dates[end + horizon_days], horizon_days,
                tickers, db_path,
            )
        except ValueError as exc:
            notes.append(f"Skipped the check as of {fit_date}: {exc}")
            continue
        rows.append(
            {
                "asof": fit_date,
                "forecast_te": forecast,
                "realized_te": realized,
            }
        )
    if tried > len(rows):
        notes.append(
            f"Only {len(rows)} of {tried} past check(s) had enough data — "
            "the rest were skipped."
        )
    if not rows:
        raise ValueError(
            "Not enough price history to check past forecasts against — "
            "about a year of daily prices is needed."
        )
    mean_f = float(np.mean([r["forecast_te"] for r in rows]))
    mean_r = float(np.mean([r["realized_te"] for r in rows]))
    if mean_r <= 1e-9:
        verdict = (
            "Past forecasts couldn't be checked — the portfolio barely "
            "moved in the test windows."
        )
    else:
        ratio = mean_f / mean_r
        if 0.80 <= ratio <= 1.25:
            plain = "roughly in line with what actually happened"
        elif ratio > 1.25:
            plain = "running hot — past forecasts overstated the risk"
        else:
            plain = "running cold — past forecasts understated the risk"
        verdict = (
            f"{plain} (average forecast {mean_f:.1%} a year vs "
            f"{mean_r:.1%} actually realized)."
        )
    return CalibrationCheck(
        rows=rows,
        mean_forecast=mean_f,
        mean_realized=mean_r,
        verdict=verdict,
        notes=notes,
    )
