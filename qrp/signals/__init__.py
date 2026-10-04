"""Signal framework and registry (Phase 2; fundamental signals added in Phase 8;
research batch 3 added later).

Use default_registry() to get a registry pre-loaded with the built-in
quantitative signals (momentum, trend, low_volatility), the
fundamental/blended signals (value_composite, profitability,
trending_value), and the research batch (continuous_momentum,
overnight_intraday, twin_momentum).
"""

from __future__ import annotations

from qrp.signals.fundamental import Profitability, TrendingValue, ValueComposite
from qrp.signals.quant import LowVolatility, Momentum, Trend
from qrp.signals.registry import SignalRegistry
from qrp.signals.research import (
    ContinuousMomentum,
    OvernightIntraday,
    TwinMomentum,
)


def default_registry() -> SignalRegistry:
    """Return a registry with all built-in signals registered."""
    reg = SignalRegistry()
    for sig in (
        Momentum(),
        Trend(),
        LowVolatility(),
        ValueComposite(),
        Profitability(),
        TrendingValue(),
        ContinuousMomentum(),
        OvernightIntraday(),
        TwinMomentum(),
    ):
        reg.register(sig)
    return reg


__all__ = [
    "SignalRegistry",
    "default_registry",
    "Momentum",
    "Trend",
    "LowVolatility",
    "ValueComposite",
    "Profitability",
    "TrendingValue",
    "ContinuousMomentum",
    "OvernightIntraday",
    "TwinMomentum",
]
