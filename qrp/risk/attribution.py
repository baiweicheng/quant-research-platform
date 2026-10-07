"""Factor attribution: what explains a strategy's returns? (roadmap P1).

Plain English: academic research found that most of what stocks do can be
explained by a handful of broad "tilts" — the market as a whole, small vs
big companies, cheap vs expensive stocks, profitable vs unprofitable
companies, careful vs aggressive spenders, and momentum (recent winners vs
losers). This module runs a regression of a strategy's daily returns
against those tilts (the Fama-French 5 factors plus momentum, downloaded
free from Ken French's data library — see qrp.data.french) and reports:

- **alpha**: the yearly return left over after the tilts are accounted
  for — "was there anything clever beyond riding known patterns?"
- **betas**: how strongly the strategy leans on each tilt, with t-stats
  (a t-stat beyond ±2 means the lean is statistically convincing, not
  just noise).
- **R-squared**: what share of the strategy's day-to-day moves the tilts
  explain together (0.90 = "mostly just the tilts").

The math is ordinary least squares, computed directly with numpy (no
extra dependencies). Everything is explained in plain English because the
person reading the report is not assumed to have a finance background.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# The six tilts, in regression order. "Mkt-RF" is the market's return minus
# the risk-free rate (what stocks earned over cash).
REGRESSORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]

# Plain-English name and "what it compares" for each tilt.
FACTOR_INFO: dict[str, tuple[str, str]] = {
    "Mkt-RF": ("the market", "the stock market as a whole vs holding cash"),
    "SMB": ("size", "small companies vs big companies"),
    "HML": ("value", "cheap stocks vs expensive stocks (by book value)"),
    "RMW": ("profitability", "profitable companies vs unprofitable ones"),
    "CMA": ("investment", "careful spenders vs aggressive expanders"),
    "UMD": ("momentum", "recent winners vs recent losers"),
}

# How to describe a significant lean, by tilt and direction.
_TILT_WORDS: dict[str, tuple[str, str]] = {
    "SMB": (
        "leans toward smaller companies",
        "leans toward bigger companies",
    ),
    "HML": (
        "leans toward cheaper (value) stocks",
        "leans toward expensive (growth) stocks",
    ),
    "RMW": (
        "leans toward more profitable companies",
        "leans toward less profitable companies",
    ),
    "CMA": (
        "leans toward careful spenders",
        "leans toward aggressive expanders",
    ),
    "UMD": (
        "leans toward recent winners (momentum)",
        "leans toward recent losers (against momentum)",
    ),
}


def _market_tilt_words(beta: float) -> str:
    """Plain-English for a significant Mkt-RF beta, sized honestly.

    A statistically significant market beta near zero means "precisely no
    market exposure" — not "moves with the market" — so the wording
    follows the beta's size, not just its t-stat.
    """
    if beta >= 0.8:
        return "moves closely with the market"
    if beta >= 0.2:
        return "moves partly with the market"
    if beta > -0.2:
        return "barely moves with the market"
    return "moves against the market"


def _tilt_words(factor: str, beta: float) -> str:
    if factor == "Mkt-RF":
        return _market_tilt_words(beta)
    return _TILT_WORDS[factor][0] if beta > 0 else _TILT_WORDS[factor][1]

# |t| beyond this counts as a convincing lean rather than noise.
T_SIGNIFICANCE = 2.0

MIN_OBS = 60  # about three months of trading days


def equity_to_returns(equity: pd.DataFrame) -> pd.Series:
    """Daily simple returns from a backtest equity curve, indexed by date.

    Args:
        equity: DataFrame with "date" and "portfolio_value" columns (the
            shape the backtest engine produces).

    Returns:
        Series of daily returns indexed by date, with undefined values
        (division by zero, infinities) dropped.
    """
    values = equity["portfolio_value"].to_numpy(dtype=float)
    dates = pd.to_datetime(equity["date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pd.Series(values[1:] / values[:-1] - 1.0, index=dates[1:])
    return rets.replace([np.inf, -np.inf], np.nan).dropna()


def factor_attribution(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    min_obs: int = MIN_OBS,
) -> dict[str, object]:
    """Regress strategy excess returns on the Fama-French factors.

    Args:
        strategy_returns: daily simple returns of the strategy, indexed by
            date (datetime-like).
        factors: DataFrame indexed by date with columns Mkt-RF, SMB, HML,
            RMW, CMA, UMD and RF (the risk-free rate), in plain decimals —
            the shape qrp.data.french.load_french_factors returns.
        min_obs: minimum overlapping days required for a trustworthy fit.

    Returns:
        {"alpha_ann": float (yearly alpha, plain decimal),
         "betas": {factor: float}, "t_stats": {factor: float},
         "r_squared": float, "n_obs": int,
         "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.

    Raises:
        ValueError with a plain-English message when there are too few
        overlapping days or the factor table is unusable.
    """
    if factors.empty or "RF" not in factors.columns:
        raise ValueError(
            "No factor data is available for those dates."
        )
    strat = pd.Series(strategy_returns, dtype=float)
    strat.index = pd.to_datetime(strat.index).normalize()
    fac = factors.copy()
    fac.index = pd.to_datetime(fac.index).normalize()

    common = strat.index.intersection(fac.index).sort_values()
    if len(common) < min_obs:
        raise ValueError(
            f"Only {len(common)} days overlap between the strategy and the "
            f"factor data — at least {min_obs} are needed for a trustworthy "
            "attribution."
        )
    y = (strat.loc[common] - fac.loc[common, "RF"]).dropna()
    x = fac.loc[y.index, REGRESSORS].dropna()
    y = y.loc[x.index]
    n = len(y)
    if n < min_obs:
        raise ValueError(
            f"Only {n} clean overlapping days — at least {min_obs} are "
            "needed for a trustworthy attribution."
        )

    # OLS: y = alpha + X @ betas + error. lstsq is the numerically stable
    # way to solve it; the covariance of the estimates then gives t-stats.
    x_mat = x.to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), x_mat])
    coef, residuals, _rank, _sv = np.linalg.lstsq(
        design, y.to_numpy(dtype=float), rcond=None
    )
    k = design.shape[1]
    resid = y.to_numpy(dtype=float) - design @ coef
    rss = float(resid @ resid)
    dof = n - k
    if dof <= 0:
        raise ValueError("Not enough data points for six factors.")
    s2 = rss / dof
    try:
        cov = s2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The factor regression could not be solved (the factor data "
            "is degenerate over this window)."
        ) from exc
    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = coef / std_err
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0

    betas = {f: float(coef[i + 1]) for i, f in enumerate(REGRESSORS)}
    t_map = {f: float(t_stats[i + 1]) for i, f in enumerate(REGRESSORS)}
    return {
        "alpha_ann": float(coef[0] * TRADING_DAYS_PER_YEAR),
        "alpha_t": float(t_stats[0]),
        "betas": betas,
        "t_stats": t_map,
        "r_squared": float(min(max(r_squared, 0.0), 1.0)),
        "n_obs": n,
        "start": common.min().strftime("%Y-%m-%d"),
        "end": common.max().strftime("%Y-%m-%d"),
    }


def significant_factors(result: dict[str, object]) -> list[str]:
    """Tilts with |t| beyond the significance bar, strongest first."""
    t_stats = result["t_stats"]
    assert isinstance(t_stats, dict)
    return sorted(
        [f for f in REGRESSORS if abs(t_stats[f]) > T_SIGNIFICANCE],
        key=lambda f: -abs(t_stats[f]),
    )


def _fmt_pct1(value: float) -> str:
    """+1.2% style with one decimal."""
    return f"{value:+.1%}"


def attribution_summary(result: dict[str, object]) -> str:
    """One-paragraph plain-English reading of an attribution result."""
    sig = significant_factors(result)
    r2 = float(result["r_squared"])
    alpha = float(result["alpha_ann"])
    betas = result["betas"]
    assert isinstance(betas, dict)

    if sig:
        tilts = "; ".join(
            f"{_tilt_words(f, betas[f])} (beta {betas[f]:+.2f})"
            for f in sig
        )
        tilt_sentence = (
            f"It leans on {len(sig)} classic tilt{'s' if len(sig) > 1 else ''}: "
            f"{tilts}."
        )
    else:
        tilt_sentence = (
            "It doesn't lean convincingly on any of the classic tilts — "
            "its moves aren't explained by the usual market patterns."
        )

    if alpha >= 0.005:
        alpha_sentence = (
            f"After accounting for those tilts, it still added about "
            f"{_fmt_pct1(alpha)} a year beyond what the tilts explain — "
            "there was something extra going on."
        )
    elif alpha <= -0.005:
        alpha_sentence = (
            f"After accounting for those tilts, it trailed by about "
            f"{-alpha:.1%} a year — the tilts explain the returns, "
            "and then some."
        )
    else:
        alpha_sentence = (
            "After accounting for those tilts, there's no extra return "
            "left over — the tilts explain it all."
        )

    if r2 >= 0.7:
        r2_sentence = (
            f"The classic tilts explain most of its day-to-day moves "
            f"(R² = {r2:.2f})."
        )
    elif r2 >= 0.3:
        r2_sentence = (
            f"The classic tilts explain some of its day-to-day moves "
            f"(R² = {r2:.2f}), but a good part is its own."
        )
    else:
        r2_sentence = (
            f"The classic tilts explain little of its day-to-day moves "
            f"(R² = {r2:.2f}) — it marches to its own drum."
        )
    return f"{tilt_sentence} {alpha_sentence} {r2_sentence}"


def attribution_short_line(result: dict[str, object]) -> str:
    """One-line label for tables/expanders, e.g. 'value + momentum tilt · +1.2%/yr beyond tilts'."""
    sig = significant_factors(result)
    alpha = float(result["alpha_ann"])
    if sig:
        tilt_bit = " + ".join(FACTOR_INFO[f][0] for f in sig)
    else:
        tilt_bit = "no clear tilt"
    if alpha >= 0.005:
        alpha_bit = f"{_fmt_pct1(alpha)}/yr beyond tilts"
    elif alpha <= -0.005:
        alpha_bit = f"{-alpha:.1%}/yr behind tilts"
    else:
        alpha_bit = "tilts explain it all"
    return f"{tilt_bit} · {alpha_bit}"
