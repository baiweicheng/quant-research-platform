"""Signal framework and registry (Phase 2).

Use default_registry() to get a registry pre-loaded with the built-in
quantitative signals (momentum, trend, low_volatility).
"""

from __future__ import annotations

from qrp.signals.quant import LowVolatility, Momentum, Trend
from qrp.signals.registry import SignalRegistry


def default_registry() -> SignalRegistry:
    """Return a registry with the built-in Phase 2 signals registered."""
    reg = SignalRegistry()
    for sig in (Momentum(), Trend(), LowVolatility()):
        reg.register(sig)
    return reg


__all__ = [
    "SignalRegistry",
    "default_registry",
    "Momentum",
    "Trend",
    "LowVolatility",
]
