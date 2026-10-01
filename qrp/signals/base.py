"""Base class for cross-sectional long-equity signals.

A Signal maps one ticker's price history to a per-date score. Scores are
cross-sectionally comparable: a higher value means a more attractive long
position. Every signal documents its direction explicitly so there is no
ambiguity about what "high" means.
"""

from __future__ import annotations

import abc

import pandas as pd


class Signal(abc.ABC):
    """Abstract base class for a quantitative signal.

    Attributes:
        name: short machine-readable identifier, e.g. "momentum".
        description: one-line human description, including direction
            (what a high value means for a long position).
        lookback_days: minimum number of daily bars required before the
            first non-NaN value is produced.
    """

    name: str = ""
    description: str = ""
    lookback_days: int = 0

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the signal for every row of df.

        Args:
            df: DataFrame with a DatetimeIndex (oldest first) and OHLCV
                columns named open, high, low, close, volume.

        Returns:
            pd.Series indexed like df with float values; NaN wherever the
            available history is shorter than lookback_days.
        """
