"""Signal framework and registry (Phase 2; fundamental signals added in Phase 8;
research batch 3 added later; value & quality classics added 2026-10-11).

Use default_registry() to get a registry pre-loaded with the built-in
quantitative signals (momentum, trend, low_volatility), the
fundamental/blended signals (value_composite, profitability,
trending_value), the research batch (continuous_momentum,
overnight_intraday, twin_momentum), and the value & quality classics
(magic_formula, piotroski, qmj_lite, shareholder_yield, reversal_long).
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
from qrp.signals.value_quality import (
    LongHorizonReversal,
    MagicFormula,
    PiotroskiFScore,
    QMJLite,
    ShareholderYield,
)


def default_registry() -> SignalRegistry:
    """Return a registry with all built-in signals registered."""
    reg = SignalRegistry()
    # Registration order pins the within-era sequence (see
    # SignalRegistry.list_signals): era 2 lists Trending Value first,
    # mirroring the strategy list, with its two building-block signals
    # (value, profitability) where their Composite strategies sit there.
    for sig in (
        Momentum(),
        Trend(),
        LowVolatility(),
        TrendingValue(),
        ValueComposite(),
        Profitability(),
        ContinuousMomentum(),
        OvernightIntraday(),
        TwinMomentum(),
        MagicFormula(),
        PiotroskiFScore(),
        QMJLite(),
        ShareholderYield(),
        LongHorizonReversal(),
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
    "MagicFormula",
    "PiotroskiFScore",
    "QMJLite",
    "ShareholderYield",
    "LongHorizonReversal",
]
