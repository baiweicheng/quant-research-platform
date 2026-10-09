"""Classic quantitative long-equity signals (Phase 2).

All three are well-understood, widely used factors. Each is documented with
its direction: higher score = more attractive long position.
"""

from __future__ import annotations

import pandas as pd

from qrp.signals.base import Signal


class Momentum(Signal):
    """12-month return skipping the most recent month ("12-2" momentum).

    The standard academic definition (Jegadeesh & Titman 1993, Fama-French
    UMD): total return from 252 trading days ago to 21 trading days ago,
    i.e. close[t-21] / close[t-252] - 1. Skipping the latest month avoids
    short-term reversal contaminating the trend signal.

    Direction: higher = stronger past winner = more attractive long.
    """

    name = "momentum"
    formal_name = "12-1 Month Cross-Sectional Momentum"
    era = 1  # original technicals (chronological display order)
    # Plain-English description for non-finance users (shown in the UI later).
    description = (
        "How much the stock has climbed over the past year, ignoring the "
        "most recent month. Stocks on a strong run tend to keep climbing "
        "for a while, so a high score means the stock has been a steady winner."
    )
    lookback_days = 253  # need rows t-252 .. t inclusive

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        return (close.shift(21) / close.shift(252) - 1.0).rename(self.name)


class Trend(Signal):
    """Distance of close above/below its 200-day moving average.

    Computed as (close - sma200) / sma200, so the value is a fraction of
    the long-run average price (e.g. 0.10 = 10% above the 200-day MA).

    Direction: higher = price further above its long-term average =
    stronger uptrend = more attractive long.
    """

    name = "trend"
    formal_name = "200-Day Moving-Average Trend Filter"
    era = 1  # original technicals (chronological display order)
    # Plain-English description for non-finance users (shown in the UI later).
    description = (
        "Whether today's price sits above or below its average over the "
        "past 10 months. Trading above that average usually means a healthy "
        "uptrend is intact, so a high score favors stocks in a solid uptrend."
    )
    lookback_days = 200

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        sma = close.rolling(200).mean()
        return ((close - sma) / sma.replace(0, float("nan"))).rename(self.name)


class LowVolatility(Signal):
    """Negative of 63-day realized volatility of daily returns.

    Signal = -(standard deviation of daily returns over the trailing 63
    trading days, ~3 months). Negation makes the direction consistent with
    the other signals: calm stocks score higher (less negative), choppy
    stocks score lower.

    Direction: higher = calmer stock = more attractive long (tilts toward
    the low-volatility anomaly). A perfectly flat price gives 0.0 -- the
    standard deviation of zeros is 0, never inf or NaN.
    """

    name = "low_volatility"
    formal_name = "Low Volatility (3-Month)"
    era = 1  # original technicals (chronological display order)
    # Plain-English description for non-finance users (shown in the UI later).
    description = (
        "How calm or jumpy the stock's price has been over the past 3 "
        "months. Calmer stocks have historically delivered steadier returns, "
        "so a high score favors steadier stocks over roller-coaster ones."
    )
    lookback_days = 64  # 63 return observations + 1 day for pct_change

    def compute(self, df: pd.DataFrame) -> pd.Series:
        rets = df["close"].astype(float).pct_change()
        vol = rets.rolling(63).std()
        return (-vol).rename(self.name)
