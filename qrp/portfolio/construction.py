"""Target portfolio construction from signal scores (Phase 4).

Answers the question: "I have a score for every stock — how much of each
should I hold?"

All methods are long-only (you never bet against a stock): every weight is
>= 0 and weights sum to at most 1, with whatever is left over held as cash.
Hard safety rails apply to every method, no exceptions:

- ``max_weight``: no single stock may exceed this fraction of the portfolio
  (default 20%). Keeps one bad pick from sinking you.
- ``max_positions``: never hold more than this many stocks (default 20).
  Keeps the portfolio focused and tradable.
- ``min_weight``: positions smaller than this are "dust" — they are dropped
  and their money goes back to the other holdings (default 1%). Avoids
  cluttering the portfolio with meaningless $50 positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from qrp.data.yahoo import normalize_ticker

EQUAL_WEIGHT = "equal_weight"
SIGNAL_WEIGHT = "signal_weight"
INVERSE_VOLATILITY = "inverse_volatility"
METHODS = (EQUAL_WEIGHT, SIGNAL_WEIGHT, INVERSE_VOLATILITY)

# Plain-English descriptions shown in the CLI and (later) the UI.
METHOD_DESCRIPTIONS = {
    EQUAL_WEIGHT: (
        "Equal weight: split the money evenly across the highest-scoring stocks. "
        "Simple and hard to mess up — no single stock can dominate. "
        "A good default when you want a straightforward, diversified portfolio."
    ),
    SIGNAL_WEIGHT: (
        "Signal weight: the best-scoring stock gets the full max-weight "
        "allowance and the rest get proportionally less, so stronger ideas "
        "count more (stocks with negative scores get nothing). With the "
        "max-weight cap in place, some money may stay in cash when the "
        "scores are very concentrated."
    ),
    INVERSE_VOLATILITY: (
        "Inverse volatility: give more money to calmer stocks and less to jumpy "
        "ones, based on each stock's recent price swings. "
        "Pick this when you want the portfolio's ups and downs to be gentler — "
        "you are trading some upside potential for a smoother ride."
    ),
}


def implied_stock_count(max_weight: float) -> int:
    """Stocks implied by a max-weight cap: n = round(1 / max_weight).

    The single position-sizing knob. A 20% cap means ~5 stocks, 10% means
    ~10, 5% means ~20. Under equal weighting each stock gets 1/n, so the
    old separate "top N" control was just this number in disguise — and
    the two could contradict each other.
    """
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1")
    return max(1, round(1.0 / max_weight))


@dataclass
class TargetPortfolio:
    """A target portfolio: what fraction of your money goes where.

    weights: ticker -> target weight as a fraction of capital
        (e.g. 0.05 means "put 5% of the money in this stock").
        Every weight is >= 0, none exceeds max_weight, and they sum to <= 1.
    cash_weight: fraction of capital deliberately left as cash
        (1 - sum of weights). Cash shows up when position caps bind, when
        nothing qualifies, or when dust positions are dropped.
    method: which weighting method produced this portfolio.
    notes: human-readable log of what was adjusted along the way
        (caps applied, dust dropped, ...). The CLI prints these so you can
        see exactly what the safety rails did.
    """

    weights: dict[str, float]
    cash_weight: float
    method: str
    notes: list[str] = field(default_factory=list)


def _apply_max_weight_cap(
    weights: dict[str, float], max_weight: float, notes: list[str]
) -> dict[str, float]:
    """Cap every weight at max_weight, passing the excess to other holdings.

    The excess from capped positions is redistributed to the uncapped ones in
    proportion to their current weights (bigger positions absorb more). If
    every position ends up capped, whatever is left over simply stays as cash
    — the portfolio never forces money into stocks just to stay invested.
    """
    weights = dict(weights)
    capped_any = False
    for _ in range(100):  # redistribution can push others over; iterate to settle
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
            notes.append(
                f"Every position hit the {max_weight:.0%} cap, so "
                f"{excess:.1%} of the capital could not be placed and stays in cash."
            )
            break
        for t in under:
            weights[t] += excess * (weights[t] / base)
    if capped_any:
        notes.append(
            f"Position cap applied: no stock holds more than {max_weight:.0%} "
            "of the portfolio."
        )
    return weights


def _drop_dust(
    weights: dict[str, float], min_weight: float, notes: list[str]
) -> dict[str, float]:
    """Remove positions smaller than min_weight; their money is redeployed."""
    dust = [t for t, w in weights.items() if w < min_weight]
    for t in dust:
        del weights[t]
    if dust:
        shown = ", ".join(sorted(dust)[:5])
        more = f" and {len(dust) - 5} more" if len(dust) > 5 else ""
        notes.append(
            f"Dropped {len(dust)} dust position(s) below {min_weight:.1%} "
            f"({shown}{more}); their money went back to the other holdings."
        )
    return weights


def build_target_weights(
    scores: dict[str, float],
    method: str = EQUAL_WEIGHT,
    *,
    max_weight: float = 0.10,
    max_positions: int = 20,
    min_weight: float = 0.01,
    volatilities: dict[str, float] | None = None,
) -> TargetPortfolio:
    """Turn signal scores into a capped, long-only target portfolio.

    Args:
        scores: ticker -> signal score (higher = more attractive as a long
            position). NaN scores are ignored.
        method: one of "equal_weight", "signal_weight", "inverse_volatility".
        max_weight: no position may exceed this fraction of capital.
        max_positions: never hold more than this many stocks; only the
            highest-scoring ones are kept.
        min_weight: positions below this are dropped as dust.
        volatilities: required for "inverse_volatility": ticker -> recent
            price volatility as a positive number (e.g. the 63-day realized
            volatility). Stocks with missing or non-positive volatility are
            skipped for that method.

    Returns:
        TargetPortfolio with ticker -> weight, cash weight, and notes.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1")
    if max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    if not 0 <= min_weight < max_weight:
        raise ValueError("min_weight must be >= 0 and below max_weight")

    notes: list[str] = []
    clean = {normalize_ticker(t): float(s) for t, s in scores.items() if isfinite(s)}
    if not clean:
        notes.append("No usable scores — everything stays in cash.")
        return TargetPortfolio(weights={}, cash_weight=1.0, method=method, notes=notes)

    # Highest score first; alphabetical tiebreak keeps results deterministic.
    ranked = sorted(clean, key=lambda t: (-clean[t], t))

    if method == EQUAL_WEIGHT:
        picks = ranked[:max_positions]
        raw = {t: 1.0 / len(picks) for t in picks}
        if any(clean[t] < 0 for t in picks):
            notes.append(
                "Note: equal weighting holds the top scorers even if some "
                "scores are negative. If you only want positively-scored "
                "stocks, use signal weighting instead."
            )
    elif method == SIGNAL_WEIGHT:
        positives = [t for t in ranked if clean[t] > 0][:max_positions]
        if not positives:
            notes.append(
                "No positive scores — a long-only portfolio cannot hold "
                "negatively-scored stocks, so everything stays in cash."
            )
            return TargetPortfolio(
                weights={}, cash_weight=1.0, method=method, notes=notes
            )
        # Scale so the best scorer gets exactly the max-weight allowance and
        # every other stock gets a proportionally smaller slice. This keeps
        # the score order visible in the weights. (Simply normalizing scores
        # to sum to 1 and then capping would flatten every position to the
        # cap whenever n == round(1 / max_weight) — the old behavior.)
        top = clean[positives[0]]
        raw = {t: max_weight * clean[t] / top for t in positives}
        total = sum(raw.values())
        if total > 1.0:
            # Extremely concentrated scores: shrink everything proportionally
            # so the portfolio still fits in 100% of the capital. The best
            # stock keeps the biggest slice; the cap is still respected
            # because shrinking only makes weights smaller.
            raw = {t: w / total for t, w in raw.items()}
            notes.append(
                "Scores were extremely concentrated, so weights were shrunk "
                "proportionally to fit 100% of the capital — the ranking "
                "order is unchanged."
            )
        else:
            leftover = 1.0 - total
            if leftover > 1e-9:
                notes.append(
                    f"Signal weighting placed {total:.1%} and left "
                    f"{leftover:.1%} in cash: with the {max_weight:.0%} cap, "
                    "the scores only supported deploying that much "
                    "proportionally — the top scorer still got the full "
                    f"{max_weight:.0%}."
                )
    else:  # INVERSE_VOLATILITY
        if volatilities is None:
            raise ValueError(
                "inverse_volatility needs volatilities={ticker: volatility}; "
                "pass each stock's recent price volatility as a positive number."
            )
        vols = {
            normalize_ticker(t): float(v)
            for t, v in volatilities.items()
            if isfinite(v)
        }
        cands = [t for t in ranked if vols.get(t, 0.0) > 0][:max_positions]
        if not cands:
            notes.append(
                "No usable volatility data — everything stays in cash."
            )
            return TargetPortfolio(
                weights={}, cash_weight=1.0, method=method, notes=notes
            )
        inv = {t: 1.0 / vols[t] for t in cands}
        total = sum(inv.values())
        raw = {t: v / total for t, v in inv.items()}

    capped = _apply_max_weight_cap(raw, max_weight, notes)
    no_dust = _drop_dust(capped, min_weight, notes)

    # Put the dust money back to work proportionally — never by topping every
    # position back up to 100%, which is what used to flatten signal-weighted
    # positions back to the cap. Money that cannot be placed within the cap
    # simply stays in cash.
    weights = dict(no_dust)
    dust_money = sum(capped.values()) - sum(no_dust.values())
    if dust_money > 1e-12 and weights:
        base = sum(weights.values())
        for t in weights:
            weights[t] += dust_money * (weights[t] / base)
        weights = _apply_max_weight_cap(weights, max_weight, notes)
    if not weights:
        notes.append("Nothing survived the dust filter — everything stays in cash.")
    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    return TargetPortfolio(
        weights=weights, cash_weight=cash_weight, method=method, notes=notes
    )
