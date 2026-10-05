"""Dict-backed registry mapping signal names to Signal instances."""

from __future__ import annotations

from qrp.signals.base import Signal


class SignalRegistry:
    """Register, look up, and list available signals."""

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}

    def register(self, signal: Signal) -> None:
        """Add a signal; replaces any existing signal with the same name."""
        if not isinstance(signal, Signal):
            raise TypeError(f"expected a Signal instance, got {type(signal)!r}")
        if not signal.name:
            raise ValueError("signal must define a non-empty name")
        self._signals[signal.name] = signal

    def get(self, name: str) -> Signal:
        """Return the signal registered under name; KeyError if unknown."""
        try:
            return self._signals[name]
        except KeyError:
            raise KeyError(f"unknown signal: {name!r}") from None

    def list_signals(self) -> list[tuple[str, str]]:
        """Return (name, description) for every registered signal."""
        return [(name, sig.description) for name, sig in self._signals.items()]

    def display_name(self, name: str) -> str:
        """Formal user-facing name for a registered signal.

        Returns the signal's formal_name (e.g. "12-1 Month
        Cross-Sectional Momentum"). For names that are not registered
        signals, falls back to the title-cased key so callers never have
        to special-case.
        """
        sig = self._signals.get(name)
        if sig is not None and sig.formal_name:
            return sig.formal_name
        return name.replace("_", " ").title()
