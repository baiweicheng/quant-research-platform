"""Constrained mean-variance optimizer (Phase P3).

Where the rule-based methods in ``qrp.portfolio.construction`` size
positions by fixed recipes (even split, score-proportional, ...), this
module *solves* for the weights: it picks the long-only, fully-invested
portfolio that maximizes

    expected return  -  (risk_aversion / 2) * portfolio variance
                     -  turnover_penalty * one-way trading vs current holdings

subject to:

- fully invested (weights sum to 1),
- long-only (no weight below 0),
- per-stock cap (default 5% — the platform's existing cap),
- a tracking-error budget vs the S&P 500 benchmark (how far the
  portfolio's returns may stray from the market's, annualized).

In plain English: "get the most expected return for the risk taken,
while limiting trading and staying close to the market."

The honest assumptions, stated up front (also shown in the UI):

- Expected returns are NOT forecasts. Each stock's signal score is
  converted to a z-score across the scored universe and multiplied by a
  scale (default: 10% a year per unit of score spread). That is a rough,
  transparent guess about the signal's edge — enough to make the
  optimizer prefer higher-scoring stocks, not a prediction anyone should
  take literally. (The research warning applies: a naive optimizer is an
  "error maximizer", which is exactly why the risk penalty, the
  shrinkage covariance, and the constraints are all load-bearing here.)
- Risk is annualized portfolio variance from the covariance model
  (Ledoit-Wolf shrinkage by default; Phase P4 plugs a factor model into
  the same seam).
- The turnover penalty is priced like a trading cost: ``turnover_penalty``
  is the penalty per dollar traded (default 20 basis points = 0.2%), so
  the optimizer only trades when the improvement beats that hurdle.

Solved with cvxpy (CLARABEL, falling back to SCS). When the
tracking-error budget can't be met together with the other constraints
(e.g. the cap-weighted benchmark itself breaks the 5% per-stock cap), the
optimizer drops just that budget, says so plainly in the notes, and still
enforces everything else — a degraded but valid answer beats a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from qrp.data.edgar import facts_as_of
from qrp.data.sp500_history import history_available, members_on
from qrp.data.store import DEFAULT_DB
from qrp.data.yahoo import normalize_ticker
from qrp.portfolio.construction import TargetPortfolio
from qrp.portfolio.covariance import (
    CovarianceModel,
    LedoitWolfCovariance,
)

# -- defaults (also the dashboard's slider defaults) ------------------------

#: How strongly the optimizer dislikes risk (dimensionless; higher = calmer).
DEFAULT_RISK_AVERSION = 2.0
#: Penalty per dollar of one-way turnover, in basis points (20 = 0.20%).
DEFAULT_TURNOVER_PENALTY_BPS = 20.0
#: Annualized tracking-error budget vs the S&P 500 (0.05 = may stray ~5%/yr).
DEFAULT_TRACKING_ERROR = 0.05
#: No stock may exceed this fraction of the portfolio.
DEFAULT_MAX_WEIGHT = 0.05
#: Optimized positions below this are dropped as dust (same as rule-based).
DEFAULT_MIN_WEIGHT = 0.01
#: Annualized expected active return per unit of score z-score (see below).
DEFAULT_EXPECTED_RETURN_SCALE = 0.10

_OPTIMIZED = "optimized"

#: Plain-English description of the optimizer shown in the CLI and UI.
OPTIMIZER_DESCRIPTION = (
    "Optimized (mean-variance): a math optimizer picks the weights to get "
    "the most expected return for the risk taken. It stays fully invested, "
    "keeps every stock at or under the max weight, stays close to the "
    "S&P 500 (tracking-error budget), and avoids trading unless the "
    "improvement beats the turnover penalty. Expected returns come from "
    "the signal scores — a rough, honest guess about the signal's edge, "
    "not a measured forecast."
)


@dataclass
class OptimizationStats:
    """What the optimizer expects, in plain annualized numbers.

    expected_return_ann: the signal-implied expected return (a rough,
        score-based guess — see the module docstring), per year.
    predicted_vol_ann: predicted portfolio volatility, per year.
    tracking_error_ann: predicted volatility of (portfolio - benchmark)
        returns, per year — "how far we may stray from the market".
    turnover_one_way: estimated one-way turnover vs the starting
        portfolio, as a fraction (0.5 * sum |w_new - w_old|): "what share
        of the portfolio gets traded".
    benchmark_note: plain-English note saying which benchmark was used.
    """

    expected_return_ann: float
    predicted_vol_ann: float
    tracking_error_ann: float
    turnover_one_way: float
    benchmark_note: str


def expected_returns_from_scores(
    scores: dict[str, float],
    scale: float = DEFAULT_EXPECTED_RETURN_SCALE,
) -> dict[str, float]:
    """Map signal scores to annualized expected active returns.

    The mapping is deliberately simple and documented: each stock's score
    becomes a cross-sectional z-score (how many standard deviations above
    or below the average stock it scores), multiplied by ``scale``. A
    stock scoring one standard deviation above average is therefore
    expected to beat the average stock by ``scale`` per year.

    This is a *relative* view, not a forecast: it only says
    higher-scoring stocks should do better than lower-scoring ones, by a
    transparently rough amount. Tickers are normalized; non-finite scores
    are ignored; when every score is identical, every expected return is
    0 (no view) and the optimizer simply minimizes risk.
    """
    clean = {
        normalize_ticker(t): float(s)
        for t, s in scores.items()
        if isfinite(s)
    }
    if not clean:
        return {}
    vals = np.array(list(clean.values()))
    std = float(vals.std())
    if not isfinite(std) or std <= 0:
        return {t: 0.0 for t in clean}
    mean = float(vals.mean())
    return {t: ((s - mean) / std) * scale for t, s in clean.items()}


def benchmark_weights(
    tickers: list[str],
    asof: str,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[dict[str, float], str]:
    """S&P 500 benchmark weights for the tracking-error constraint.

    Point-in-time and honest about what it could find:

    - The benchmark set is the S&P 500 membership on ``asof`` (from the
      platform's membership table), intersected with ``tickers``. Tickers
      the table has never heard of are treated as members (same rule as
      the backtest engine).
    - When enough members have both a shares-outstanding figure (from the
      latest SEC filing knowable on or before ``asof`` — no peeking) and
      a price, weights are by company size (shares x price), normalized.
    - Otherwise it falls back to equal weight over the benchmark set,
      and the returned note says so plainly.

    Returns (weights, note). Weights sum to 1 over the tickers they cover;
    tickers outside the benchmark set get an implicit 0.
    """
    from qrp.signals.store import load_bars

    tickers = [normalize_ticker(t) for t in tickers]
    members = members_on(asof) if history_available() else set()
    bench_set = [t for t in tickers if t in members] or list(tickers)
    caps: dict[str, float] = {}
    for t in bench_set:
        facts = facts_as_of(t, "shares_outstanding", asof, path=db_path)
        if facts.empty:
            continue
        shares = float(facts.sort_values("period_end").iloc[-1]["value"])
        bars = load_bars(t, end=asof, path=db_path)
        if bars.empty:
            continue
        price = float(bars["close"].iloc[-1])
        if (
            isfinite(shares)
            and isfinite(price)
            and shares > 0
            and price > 0
        ):
            caps[t] = shares * price
    need = max(2, len(bench_set) // 2)
    if caps and len(caps) >= need and sum(caps.values()) > 0:
        total = sum(caps.values())
        note = (
            f"Benchmark: the S&P 500 weighted by company size "
            f"(shares outstanding x price) for {len(caps)} of "
            f"{len(bench_set)} benchmark stocks."
        )
        return {t: c / total for t, c in caps.items()}, note
    n = len(bench_set)
    note = (
        "Not enough company-size data was available, so the benchmark is "
        "an equal-weighted S&P 500 instead of a size-weighted one."
    )
    return ({t: 1.0 / n for t in bench_set} if n else {}), note


def _solve_cvxpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_b: np.ndarray,
    w_prev: np.ndarray,
    risk_aversion: float,
    turnover_penalty: float,
    tracking_error: float,
    max_weight: float,
) -> tuple[np.ndarray, bool]:
    """Solve the mean-variance problem; return (weights, te_enforced).

    Tries CLARABEL then SCS. If the tracking-error budget makes the
    problem infeasible, retries without it and reports te_enforced=False.
    Raises ValueError (plain English) when nothing solves.
    """
    import cvxpy as cp

    n = len(mu)

    def _problem(enforce_te: bool) -> cp.Problem:
        w = cp.Variable(n)
        variance = cp.quad_form(w, sigma)
        objective = cp.Maximize(
            mu @ w
            - (risk_aversion / 2.0) * variance
            - turnover_penalty * cp.norm(w - w_prev, 1)
        )
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
        if enforce_te:
            constraints.append(
                cp.quad_form(w - w_b, sigma) <= tracking_error**2
            )
        return cp.Problem(objective, constraints)

    last_err: Exception | None = None
    for enforce_te in (True, False):
        problem = _problem(enforce_te)
        for solver in ("CLARABEL", "SCS"):
            try:
                problem.solve(solver=solver, verbose=False)
            except Exception as exc:  # noqa: BLE001 - try next solver
                last_err = exc
                continue
            if problem.status in ("optimal", "optimal_inaccurate"):
                w_val = np.asarray(problem.variables()[0].value).flatten()
                if w_val.shape == (n,) and np.all(np.isfinite(w_val)):
                    return w_val, enforce_te
            last_err = RuntimeError(f"solver status: {problem.status}")
    raise ValueError(
        "The optimizer could not find a valid portfolio with these "
        "settings (even after relaxing the tracking-error budget). Try a "
        "larger per-stock cap or fewer constraints."
        + (f" Last solver message: {last_err}" if last_err else "")
    )


def _cap_redistribute(
    weights: dict[str, float], max_weight: float, notes: list[str]
) -> dict[str, float]:
    """Cap weights at max_weight, passing excess to uncapped holdings."""
    weights = dict(weights)
    capped_any = False
    for _ in range(100):
        over = {t: w for t, w in weights.items() if w > max_weight + 1e-12}
        if not over:
            break
        capped_any = True
        excess = sum(w - max_weight for w in over.values())
        for t in over:
            weights[t] = max_weight
        under = {t: w for t, w in weights.items() if w < max_weight - 1e-12}
        base = sum(under.values())
        if base <= 0:
            break
        for t in under:
            weights[t] += excess * (weights[t] / base)
    if capped_any:
        notes.append(
            f"Position cap applied: no stock holds more than {max_weight:.0%} "
            "of the portfolio."
        )
    return weights


def optimize_weights(
    scores: dict[str, float],
    *,
    asof: str,
    tickers: list[str] | None = None,
    db_path: str | Path = DEFAULT_DB,
    w_prev: dict[str, float] | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    turnover_penalty_bps: float = DEFAULT_TURNOVER_PENALTY_BPS,
    tracking_error: float = DEFAULT_TRACKING_ERROR,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    expected_return_scale: float = DEFAULT_EXPECTED_RETURN_SCALE,
    covariance_model: CovarianceModel | None = None,
) -> tuple[TargetPortfolio, OptimizationStats]:
    """Solve for the constrained mean-variance portfolio.

    Args:
        scores: ticker -> signal score (higher = more attractive).
        asof: YYYY-MM-DD; only data on or before this date is used.
        tickers: candidate universe (defaults to the scored tickers).
        db_path: SQLite market database.
        w_prev: ticker -> current weight as a fraction of capital (zeros
            for a fresh portfolio). Drives the turnover penalty and the
            reported turnover.
        risk_aversion: higher = the optimizer accepts less risk.
        turnover_penalty_bps: penalty per dollar of one-way turnover, in
            basis points — the "only trade when it's worth it" knob.
        tracking_error: annualized budget for straying from the S&P 500.
        max_weight: per-stock cap. When the universe is smaller than
            1/max_weight stocks, the cap is relaxed to 1/n (noted) rather
            than failing.
        min_weight: optimized positions below this are dropped as dust.
        expected_return_scale: annualized expected return per unit of
            score z-score (see expected_returns_from_scores).
        covariance_model: a CovarianceModel (default: Ledoit-Wolf from
            the market DB). Any P4 factor model implementing the interface
            slots in here.

    Returns:
        (TargetPortfolio, OptimizationStats). The portfolio is long-only,
        fully invested, and respects the cap; the stats carry the
        annualized expectations. Notes explain every degradation in plain
        English. Raises ValueError with a plain-English message (never a
        raw traceback) when no valid portfolio can be built.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    if turnover_penalty_bps < 0:
        raise ValueError("turnover_penalty_bps cannot be negative.")
    if tracking_error < 0:
        raise ValueError("tracking_error cannot be negative.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if not 0 <= min_weight < max_weight:
        raise ValueError("min_weight must be >= 0 and below max_weight.")

    notes: list[str] = []
    mu = expected_returns_from_scores(scores, expected_return_scale)
    if tickers is not None:
        wanted = {normalize_ticker(t) for t in tickers}
        mu = {t: m for t, m in mu.items() if t in wanted}
    if not mu:
        raise ValueError(
            "No usable signal scores — compute signals first, then try again."
        )

    model = covariance_model or LedoitWolfCovariance(db_path)
    sigma_df = model.covariance(list(mu), asof)
    notes.extend(getattr(model, "notes", []))
    universe = [t for t in mu if t in sigma_df.index]
    if not universe:
        raise ValueError(
            "None of the scored stocks had enough price history to "
            "estimate risk from — download prices first (Market data page)."
        )
    if len(universe) < len(mu):
        notes.append(
            f"{len(mu) - len(universe)} scored stock(s) were left out: no "
            "usable price history to estimate their risk."
        )
    mu = {t: mu[t] for t in universe}

    n = len(universe)
    if n * max_weight < 1 - 1e-12:
        old_cap = max_weight
        max_weight = 1.0 / n
        notes.append(
            f"Only {n} stocks qualified, so the per-stock cap was relaxed "
            f"to {max_weight:.1%} (with the {old_cap:.0%} cap "
            "the weights could not sum to 100%)."
        )

    # Annualized covariance, symmetrized with a hair of ridge for the
    # solver's PSD check.
    sigma = sigma_df.loc[universe, universe].to_numpy(dtype=float)
    sigma = 0.5 * (sigma + sigma.T) + 1e-10 * np.eye(n)

    bench, bench_note = benchmark_weights(universe, asof, db_path)
    w_b = np.array([bench.get(t, 0.0) for t in universe])
    if w_b.sum() > 0:
        w_b = w_b / w_b.sum()

    prev = {normalize_ticker(t): float(v) for t, v in (w_prev or {}).items()}
    w_prev_vec = np.array([prev.get(t, 0.0) for t in universe])
    mu_vec = np.array([mu[t] for t in universe])

    raw, te_enforced = _solve_cvxpy(
        mu_vec,
        sigma,
        w_b,
        w_prev_vec,
        risk_aversion,
        turnover_penalty_bps / 10_000.0,
        tracking_error,
        max_weight,
    )
    if not te_enforced:
        notes.append(
            f"The {tracking_error:.0%} tracking-error budget could not be "
            "met together with the other limits (the size-weighted "
            "benchmark itself breaks the per-stock cap), so it was dropped "
            "— every other limit still holds."
        )

    # Solver tolerance can leave 1e-9 dust below zero; clip it away.
    weights = {t: float(max(w, 0.0)) for t, w in zip(universe, raw)}
    weights = _cap_redistribute(weights, max_weight, notes)
    dust = [t for t, w in weights.items() if w < min_weight]
    if dust:
        for t in dust:
            del weights[t]
        shown = ", ".join(sorted(dust)[:5])
        more = f" and {len(dust) - 5} more" if len(dust) > 5 else ""
        notes.append(
            f"Dropped {len(dust)} dust position(s) below {min_weight:.1%} "
            f"({shown}{more}); their money went back to the other holdings."
        )
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(
            "The optimizer returned an empty portfolio — this should not "
            "happen; try different settings."
        )
    weights = {t: w / total for t, w in weights.items()}
    weights = _cap_redistribute(weights, max_weight, notes)
    total = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    # Verify every constraint before returning — belt and suspenders.
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Internal error: optimized weights do not sum to 1.")
    if any(w < -1e-9 or w > max_weight + 1e-6 for w in weights.values()):
        raise ValueError("Internal error: optimized weights break the caps.")

    w_vec = np.array([weights.get(t, 0.0) for t in universe])
    exp_ret = float(mu_vec @ w_vec)
    pred_vol = float(np.sqrt(max(w_vec @ sigma @ w_vec, 0.0)))
    active = w_vec - w_b
    te = float(np.sqrt(max(active @ sigma @ active, 0.0)))
    if te_enforced and te > tracking_error + 1e-4:
        notes.append(
            f"The measured tracking error ({te:.1%}) slightly exceeds the "
            f"{tracking_error:.0%} budget after rounding — the difference "
            "is solver tolerance, not a broken limit."
        )
    turnover = 0.5 * float(
        np.sum(np.abs(w_vec - w_prev_vec))
        + sum(v for t, v in prev.items() if t not in weights)
    )
    notes.append(
        "Expected returns are a rough mapping from signal scores "
        f"(about {expected_return_scale:.0%} a year per unit of score "
        "spread) — a guess about the signal's edge, not a measured "
        "forecast."
    )

    portfolio = TargetPortfolio(
        weights=weights,
        cash_weight=0.0,
        method=_OPTIMIZED,
        notes=notes,
    )
    stats = OptimizationStats(
        expected_return_ann=exp_ret,
        predicted_vol_ann=pred_vol,
        tracking_error_ann=te,
        turnover_one_way=turnover,
        benchmark_note=bench_note,
    )
    return portfolio, stats
