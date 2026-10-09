"""Dict-backed registry mapping signal names to Signal instances."""

from __future__ import annotations

from qrp.signals.base import Signal


class SignalRegistry:
    """Register, look up, and list available signals."""

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}
        self._next_seq = 0

    def register(self, signal: Signal) -> None:
        """Add a signal; replaces any existing signal with the same name."""
        if not isinstance(signal, Signal):
            raise TypeError(f"expected a Signal instance, got {type(signal)!r}")
        if not signal.name:
            raise ValueError("signal must define a non-empty name")
        # Registration sequence, mirroring the strategies registry's _seq:
        # together with `era` it pins the chronological display order even
        # if registration order in the source is ever shuffled.
        signal._seq = self._next_seq  # type: ignore[attr-defined]
        self._next_seq += 1
        self._signals[signal.name] = signal

    def get(self, name: str) -> Signal:
        """Return the signal registered under name; KeyError if unknown."""
        try:
            return self._signals[name]
        except KeyError:
            raise KeyError(f"unknown signal: {name!r}") from None

    def list_signals(self) -> list[tuple[str, str]]:
        """(name, description) for every registered signal, chronological.

        Ordered by build era (original technicals, then fundamentals,
        then the research batch), oldest first — the same sequence the
        strategy list uses, so every picker, page and table in the UI
        shares this one sequence.
        """
        ordered = sorted(
            self._signals.values(),
            key=lambda s: (s.era, getattr(s, "_seq", 0)),
        )
        return [(sig.name, sig.description) for sig in ordered]

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
