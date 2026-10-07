"""Operations hardening (roadmap P6) — the safety gate before any live money.

Paper trading is the default and the only mode this package knows: the
kill switch, audit log, pre-trade checklist, and pipeline guards below all
operate on paper trading (the Alpaca paper adapter and the local paper
broker). They are designed so a future live path can reuse them, but no
live path exists here.

Modules:
- ``killswitch``: the big red button — cancels all open orders, disables
  trading until a human re-enables it. Never liquidates positions.
- ``audit``: append-only, tamper-evident (hash-chained) log of everything
  the platform decides and does.
- ``pretrade``: the checklist every order batch passes before anything is
  staged or submitted — each check returns pass/fail/warn in plain English.
- ``pipeline``: the order-pipeline guards — kill-switch-first, plus the
  automatic trigger hooks (mass rejections, auth failure, submit-path
  exceptions) that engage the kill switch by themselves.

All persistent state lives next to the market database (gitignored, local
only — see ``ops_files()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qrp.data.store import DEFAULT_DB


@dataclass(frozen=True)
class OpsFiles:
    """Where P6's persistent state lives (all gitignored, local only)."""

    killswitch: Path  #: {"engaged": bool, "engaged_at": ..., "reason": ...}
    stop_flag: Path  #: dead-simple file flag: exists => trading stopped
    audit_log: Path  #: append-only JSONL audit trail


def ops_files(db_path: str | Path = DEFAULT_DB) -> OpsFiles:
    """Locate the ops state files next to the market database."""
    data_dir = Path(db_path).parent
    return OpsFiles(
        killswitch=data_dir / "killswitch.json",
        stop_flag=data_dir / "STOP",
        audit_log=data_dir / "audit.log",
    )
