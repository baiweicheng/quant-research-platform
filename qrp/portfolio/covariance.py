"""Covariance estimation for the portfolio optimizer (Phase P3).

The optimizer needs to know how stocks move *together* — not just how
jumpy each one is on its own — so it can avoid loading up on stocks that
all fall at the same time. That "how they move together" table is the
covariance matrix.

This module defines the seam Phase P4 (risk model) will plug into:

- ``CovarianceModel``: the interface — "give me an annualized covariance
  matrix for these tickers using only data known on or before this date".
  Anything implementing it (a stub in tests, a factor model in P4) works
  with the optimizer unchanged.
- ``LedoitWolfCovariance``: the Phase P3 implementation. It estimates the
  covariance from the trailing ~252 trading days of daily returns and
  *shrinks* the raw estimate toward a simple, stable target (Ledoit &
  Wolf 2004). Shrinkage matters because the raw estimate is noisy — with
  500 stocks and 252 days there are more pairs than observations — and an
  unshrunk matrix would let the optimizer chase estimation noise. (That
  failure mode is why the research calls a naive optimizer an "error
  maximizer".)

Plain-English data rules (never crash, never peek):

- Only prices dated on or before ``asof`` are used — the same no-peeking
  rule as the backtest engine.
- A stock needs at least ``min_observations`` daily returns (default 30)
  to be included; thinner histories are dropped with a plain-English
  note.
- With fewer than 2 usable stocks, correlations can't be estimated, so
  the module falls back to a diagonal matrix (each stock's own variance,
  zero correlation assumed) and says so plainly.
- With no usable price history at all, it raises a ValueError with a
  plain-English explanation ("download prices first") instead of a
  traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from qrp.data.store import DEFAULT_DB, connect_db
from qrp.data.yahoo import normalize_ticker

TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MIN_OBSERVATIONS = 30
#: Daily return variance at or below this counts as "no movement" (a stock
#: whose price never moves carries no information and breaks the math).
#: 1e-12 is ~1e-6 daily volatility — economically zero, well above
#: floating-point noise (~1e-38 for a constant series).
MIN_DAILY_VARIANCE = 1e-12


@runtime_checkable
class CovarianceModel(Protocol):
    """How the optimizer learns how stocks move together.

    The Phase P3/P4 seam: P3 ships ``LedoitWolfCovariance``; Phase P4
    (risk model) will add a factor-based implementation behind this same
    interface, and the optimizer will accept it without any changes.

    Contract:

    - ``covariance(tickers, asof)`` returns a DataFrame indexed and
      columned by the (normalized) tickers it could estimate, holding
      *annualized* covariances of daily simple returns.
    - The returned matrix is symmetric and positive semi-definite.
    - Only data dated on or before ``asof`` (YYYY-MM-DD) may be used.
    - Implementations expose a ``notes`` list of plain-English strings
      describing any degradation (dropped tickers, fallbacks).
    - With no usable data at all, raise ValueError with a plain-English
      message — never a raw traceback.
    """

    notes: list[str]

    def covariance(self, tickers: list[str], asof: str) -> pd.DataFrame:
        """Annualized covariance matrix for `tickers` as of `asof`."""
        ...


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame, return_delta: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, float]:
    """Shrink a sample covariance toward a scaled identity (Ledoit-Wolf).

    Args:
        returns: T x N DataFrame of daily simple returns (columns are
            tickers). NaNs are dropped row-wise; every column needs at
            least 2 observations and non-zero variance.
        return_delta: when True, return ``(covariance, delta)`` instead
            of just the covariance — useful for tests and diagnostics.

    Returns:
        N x N DataFrame of *annualized* covariances (daily estimate x 252),
        symmetric and positive definite (plus the shrinkage intensity when
        ``return_delta`` is set).

    The math (Ledoit & Wolf 2004, "Honey, I shrunk the sample
    covariance", shrinkage toward mu*I):

    - S: sample covariance with 1/T normalization.
    - mu: average of the diagonal of S (the shrinkage target's level).
    - The optimal shrinkage intensity delta blends (1-delta)*S + delta*mu*I,
      where delta is estimated from the data (pi_hat, rho_hat, gamma_hat
      below). delta = 0 means "trust the sample fully"; delta = 1 means
      "assume every stock is independent with the same variance".
    - Annualizing multiplies by 252 trading days so the optimizer can mix
      these numbers with annualized expected returns.
    """
    clean = returns.dropna(axis=0, how="any")
    t, n = clean.shape
    if n < 1 or t < 2:
        raise ValueError(
            "Ledoit-Wolf needs at least 2 observations of at least 1 "
            f"return series (got {t} rows, {n} columns)."
        )
    tickers = list(clean.columns)
    X = clean.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    S = (X.T @ X) / t
    variances = np.diag(S)
    if bool((variances <= MIN_DAILY_VARIANCE).any()):
        bad = [
            tickers[i]
            for i in np.nonzero(variances <= MIN_DAILY_VARIANCE)[0][:5]
        ]
        raise ValueError(
            "Ledoit-Wolf needs positive variance for every series; "
            f"these have none: {', '.join(bad)}."
        )
    mu = float(np.trace(S) / n)

    # pi_hat = sum of asymptotic variances of the sample covariances.
    # Vectorized identity: (1/T) sum_t (||x_t||^2)^2 - ||S||_F^2.
    row_norm_sq = (X**2).sum(axis=1)
    pi_hat = float((row_norm_sq**2).mean() - (S**2).sum())

    # rho_hat: asymptotic covariance between the sample covariances and
    # the shrinkage target's entries. For the identity target F = mu*I,
    # only the diagonal target entries are random (f_ii = mu-hat, the
    # average sample variance); every off-diagonal target entry is the
    # constant zero, so its asymptotic covariance with any sample
    # covariance vanishes. Hence
    #   rho = sum_i AsyCov(sqrt(T) s_ii, sqrt(T) mu-hat)
    #       = (1/p) sum_{i,j} theta_{ij},
    # where theta_{ij} = AsyCov(sqrt(T) s_ii, sqrt(T) s_jj), estimated by
    # Q_{ij} - s_ii*s_jj with Q the covariance of squared (demeaned)
    # returns. (Do NOT use the constant-correlation target's off-diagonal
    # asymptotics here — that was a different paper's formula and it
    # biased delta.)
    Y = X**2
    Q = (Y.T @ Y) / t
    theta = Q - np.outer(variances, variances)
    rho_hat = float(theta.sum() / n)

    # gamma_hat: squared distance between S and the shrinkage target.
    gamma_hat = float(((S - mu * np.eye(n)) ** 2).sum())

    if gamma_hat <= 0:
        delta = 1.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(min(max(kappa / t, 0.0), 1.0))

    shrunk = (1.0 - delta) * S + delta * mu * np.eye(n)
    # Symmetrize against floating-point noise; the shrinkage target keeps
    # the result strictly positive definite (mu > 0 here).
    shrunk = 0.5 * (shrunk + shrunk.T)
    annualized = shrunk * TRADING_DAYS_PER_YEAR
    result = pd.DataFrame(annualized, index=tickers, columns=tickers)
    if return_delta:
        return result, delta
    return result


class LedoitWolfCovariance:
    """Shrinkage covariance from trailing daily returns in the market DB.

    Implements ``CovarianceModel``. Reads up to ``lookback_days`` trading
    days of closes on or before ``asof`` for each ticker in one SQL query,
    converts to daily simple returns, and applies
    ``ledoit_wolf_shrinkage``. Tickers with too little history are dropped
    (recorded in ``notes``); with fewer than 2 survivors it falls back to
    a diagonal matrix (independence assumption, noted plainly).
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    ) -> None:
        self.db_path = Path(db_path)
        self.lookback_days = lookback_days
        self.min_observations = min_observations
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
            df = pd.read_sql_query(query, conn, params=[*tickers, asof])
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(
            index="date", columns="ticker", values="close", aggfunc="last"
        )
        wide = wide.sort_index().tail(self.lookback_days)
        # Daily simple returns; a return needs two consecutive closes.
        returns = wide.pct_change().iloc[1:]
        returns.index = pd.to_datetime(returns.index)
        return returns.sort_index()

    def covariance(self, tickers: list[str], asof: str) -> pd.DataFrame:
        """Annualized Ledoit-Wolf covariance for `tickers` as of `asof`.

        Never crashes on thin data: drops under-observed tickers (noted),
        falls back to a diagonal matrix with < 2 survivors (noted), and
        raises a plain-English ValueError only when nothing is usable.
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
            shown = ", ".join(sorted(set(dropped) | set(never_seen))[:5])
            more = (
                f" and {len(set(dropped) | set(never_seen)) - 5} more"
                if len(set(dropped) | set(never_seen)) > 5
                else ""
            )
            self.notes.append(
                f"{len(set(dropped) | set(never_seen))} stock(s) had too "
                f"little price history to estimate risk from ({shown}{more}) "
                "and were left out of the optimization."
            )
        if not usable:
            raise ValueError(
                "No usable price history for any of these stocks — "
                "download prices first (Market data page), then try again."
            )
        if len(usable) < 2:
            # Diagonal fallback: assume stocks move independently. The
            # variance comes from whatever history exists.
            t = usable[0]
            var = float(rets[t].var())
            var_ann = var * TRADING_DAYS_PER_YEAR if var > 0 else 0.09
            self.notes.append(
                "Only one stock had enough history, so stocks are treated "
                "as moving independently of each other (no correlation "
                "estimated)."
            )
            return pd.DataFrame(
                [[var_ann]], index=usable, columns=usable
            )
        return ledoit_wolf_shrinkage(rets[usable])
