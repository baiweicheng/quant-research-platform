"""Customizable strategies: named variants with validated parameter overrides.

A variant is a named set of parameter overrides on top of one canonical
strategy. The canonical recipe is never mutated — overrides always
produce a clearly labeled variant ("Value + Momentum — Concentrated 10"),
so a track record is always attributable to the exact settings that
produced it.

Overridable parameters (all optional; anything not overridden keeps the
canonical value):
    max_weight:      per-stock cap, fraction of capital (0.01-1.0)
    max_stocks:      explicit stock count (1-100); when omitted the count
                     is implied by the cap (round(1 / max_weight))
    rebalance:       "weekly" | "monthly" | "quarterly"
    weighting:       "equal_weight" | "signal_weight" | "inverse_volatility"
    drift_tolerance: anti-churn filter, fraction (0.0-0.5)
    market_gate:     bool — the dual-momentum overlay (risk-off -> cash)

Conflict handling (the five rules):
  1. Mutually exclusive settings (rebalance, weighting) are single-select
     in the UI — the model only ever sees one value.
  2. Hard numerical conflicts are refused with a plain-English error
     (e.g. max_weight 0, max_stocks -3, unknown parameter name).
  3. Soft conflicts warn but are allowed when intentional: e.g. 10 stocks
     x 5% cap leaves 50% of the account in cash by construction.
  4. Deterministic precedence, safety wins: a risk-off market gate
     de-risks fully even if the drift buffer would otherwise skip the
     trades (de-risking is never blocked by the anti-churn filter).
  5. Nothing fails silently: every refusal and every warning is returned
     to the caller as data, and the UI shows it.

Storage: a JSON file next to the market database (strategy_variants.json),
so variants travel with the user's data and never touch the repo.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from qrp.data.store import DEFAULT_DB

# parameter -> (kind, low, high) for numeric validation
_NUMERIC_BOUNDS = {
    "max_weight": (0.01, 1.0),
    "max_stocks": (1, 100),
    "drift_tolerance": (0.0, 0.5),
}
_SINGLE_SELECT = {
    "rebalance": ("weekly", "monthly", "quarterly"),
    # Short names, matching Strategy.weighting ("equal", "signal",
    # "inverse_vol") — see WEIGHTING_METHODS in qrp/backtest/strategies.py.
    "weighting": ("equal", "signal", "inverse_vol"),
}
# Every parameter a variant is allowed to touch.
KNOWN_PARAMS = set(_NUMERIC_BOUNDS) | set(_SINGLE_SELECT) | {"market_gate"}


@dataclass
class Variant:
    """A named parameter override on a canonical strategy."""

    id: str
    name: str
    base: str  # canonical strategy key, e.g. "value_momentum"
    overrides: dict
    created: str = ""
    archived: bool = False
    backtest_runs: int = 0
    paper_runs: int = 0

    def display_name(self, base_display: str) -> str:
        return f"{base_display} — {self.name}"

    def settings_summary(self) -> str:
        bits = []
        o = self.overrides
        if "max_stocks" in o:
            bits.append(f"{int(o['max_stocks'])} stocks")
        if "max_weight" in o:
            bits.append(f"{float(o['max_weight']):.1%} cap")
        if "rebalance" in o:
            bits.append(str(o["rebalance"]))
        if "weighting" in o:
            bits.append(
                {"equal": "equal weight", "signal": "signal weight",
                 "inverse_vol": "inverse volatility"}.get(
                    str(o["weighting"]), str(o["weighting"]))
            )
        if "drift_tolerance" in o:
            bits.append(f"{float(o['drift_tolerance']):.0%} drift buffer")
        if o.get("market_gate"):
            bits.append("market gate on")
        return ", ".join(bits) or "canonical settings"


def validate_overrides(overrides: dict) -> tuple[dict, list[str]]:
    """Validate a raw overrides dict.

    Returns (cleaned, warnings). Raises ValueError with a plain-English
    message on any hard conflict. Unknown parameter names are a hard
    error — silently ignoring a typo'd knob would be the worst outcome.
    """
    cleaned: dict = {}
    warnings: list[str] = []
    for key, value in (overrides or {}).items():
        if key not in KNOWN_PARAMS:
            raise ValueError(
                f"Unknown setting {key!r} — the customizable settings are: "
                f"{', '.join(sorted(KNOWN_PARAMS))}."
            )
        if key in _NUMERIC_BOUNDS:
            low, high = _NUMERIC_BOUNDS[key]
            try:
                num = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{key} must be a number between {low} and {high}."
                )
            if key == "max_stocks":
                if int(num) != num:
                    raise ValueError("max_stocks must be a whole number.")
                num = int(num)
            if not (low <= num <= high):
                raise ValueError(
                    f"{key} must be between {low} and {high} "
                    f"(you gave {value})."
                )
            cleaned[key] = num
        elif key in _SINGLE_SELECT:
            options = _SINGLE_SELECT[key]
            if str(value) not in options:
                raise ValueError(
                    f"{key} must be one of {', '.join(options)} "
                    f"(you gave {value!r})."
                )
            cleaned[key] = str(value)
        else:  # market_gate
            cleaned[key] = bool(value)
    # Soft conflict: the caps structurally leave cash on the table.
    mw = cleaned.get("max_weight")
    ms = cleaned.get("max_stocks")
    if mw is not None and ms is not None and ms * mw < 1.0:
        cash_pct = (1.0 - ms * mw) * 100
        warnings.append(
            f"With {ms} stocks x {mw:.0%} cap, up to {cash_pct:.0f}% of the "
            "account sits in cash by construction. That's allowed — just "
            "making sure it's intentional."
        )
    return cleaned, warnings


def variants_path(db_path: str | Path = DEFAULT_DB) -> Path:
    return Path(db_path).parent / "strategy_variants.json"


def _load(db_path: str | Path = DEFAULT_DB) -> dict:
    path = variants_path(db_path)
    if not path.exists():
        return {"variants": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"variants": []}
    if not isinstance(data, dict) or "variants" not in data:
        return {"variants": []}
    return data


def _save(data: dict, db_path: str | Path = DEFAULT_DB) -> None:
    path = variants_path(db_path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _row_to_variant(row: dict) -> Variant:
    return Variant(
        id=str(row.get("id", "")),
        name=str(row.get("name", "")),
        base=str(row.get("base", "")),
        overrides=dict(row.get("overrides", {})),
        created=str(row.get("created", "")),
        archived=bool(row.get("archived", False)),
        backtest_runs=int(row.get("backtest_runs", 0) or 0),
        paper_runs=int(row.get("paper_runs", 0) or 0),
    )


def _variant_to_row(v: Variant) -> dict:
    return {
        "id": v.id,
        "name": v.name,
        "base": v.base,
        "overrides": v.overrides,
        "created": v.created,
        "archived": v.archived,
        "backtest_runs": v.backtest_runs,
        "paper_runs": v.paper_runs,
    }


def list_variants(
    db_path: str | Path = DEFAULT_DB, include_archived: bool = False
) -> list[Variant]:
    """All saved variants, newest first."""
    rows = _load(db_path)["variants"]
    out = [_row_to_variant(r) for r in rows]
    if not include_archived:
        out = [v for v in out if not v.archived]
    return sorted(out, key=lambda v: v.created, reverse=True)


def get_variant(
    variant_id: str, db_path: str | Path = DEFAULT_DB
) -> Variant | None:
    for v in list_variants(db_path, include_archived=True):
        if v.id == variant_id:
            return v
    return None


def save_variant(
    name: str,
    base: str,
    overrides: dict,
    db_path: str | Path = DEFAULT_DB,
) -> tuple[Variant, list[str]]:
    """Validate, then persist a new variant. Returns (variant, warnings).

    Raises ValueError on hard conflicts or a blank/duplicate name.
    """
    from qrp.strategies.registry import STRATEGIES  # local: import cycle

    name = (name or "").strip()
    if not name:
        raise ValueError("Give the variant a name — e.g. 'Concentrated 10'.")
    if base not in STRATEGIES:
        raise ValueError(f"Unknown strategy {base!r}.")
    cleaned, warnings = validate_overrides(overrides)
    data = _load(db_path)
    existing = {str(r.get("name", "")).lower() for r in data["variants"]}
    if name.lower() in existing:
        raise ValueError(
            f"A variant named {name!r} already exists — pick another name."
        )
    variant = Variant(
        id=uuid.uuid4().hex[:12],
        name=name,
        base=base,
        overrides=cleaned,
        created=date.today().isoformat(),
    )
    data["variants"].append(_variant_to_row(variant))
    _save(data, db_path)
    return variant, warnings


def rename_variant(
    variant_id: str, new_name: str, db_path: str | Path = DEFAULT_DB
) -> Variant:
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("The name can't be blank.")
    data = _load(db_path)
    for row in data["variants"]:
        if row.get("id") != variant_id and str(
            row.get("name", "")
        ).lower() == new_name.lower():
            raise ValueError(
                f"A variant named {new_name!r} already exists."
            )
    for row in data["variants"]:
        if row.get("id") == variant_id:
            row["name"] = new_name
            _save(data, db_path)
            return _row_to_variant(row)
    raise ValueError("Variant not found.")


def archive_variant(
    variant_id: str, db_path: str | Path = DEFAULT_DB
) -> None:
    """Archive (hide) a variant. Its track record stays attributable —
    archived variants are never deleted silently."""
    data = _load(db_path)
    for row in data["variants"]:
        if row.get("id") == variant_id:
            row["archived"] = True
            _save(data, db_path)
            return
    raise ValueError("Variant not found.")


def unarchive_variant(
    variant_id: str, db_path: str | Path = DEFAULT_DB
) -> None:
    """Restore an archived variant to active use."""
    data = _load(db_path)
    for row in data["variants"]:
        if row.get("id") == variant_id:
            row["archived"] = False
            _save(data, db_path)
            return
    raise ValueError("Variant not found.")


def delete_variant(
    variant_id: str, db_path: str | Path = DEFAULT_DB
) -> None:
    """Permanently delete an archived variant. Refuses to delete an
    active one — archive first, so deletion is always deliberate."""
    data = _load(db_path)
    for row in data["variants"]:
        if row.get("id") == variant_id:
            if not row.get("archived"):
                raise ValueError(
                    "Archive the variant first — deletion is permanent and "
                    "only allowed for archived variants."
                )
            data["variants"] = [
                r for r in data["variants"] if r.get("id") != variant_id
            ]
            _save(data, db_path)
            return
    raise ValueError("Variant not found.")


def record_use(
    variant_id: str, where: str, db_path: str | Path = DEFAULT_DB
) -> None:
    """Note that a variant was used in a backtest or paper run, so the
    manager can show where each variant is used. Never raises."""
    if where not in ("backtest", "paper"):
        return
    try:
        data = _load(db_path)
        for row in data["variants"]:
            if row.get("id") == variant_id:
                row[f"{where}_runs"] = int(row.get(f"{where}_runs", 0) or 0) + 1
        _save(data, db_path)
    except OSError:
        pass


def apply_variant(strategy, variant: Variant):
    """A copy of the canonical strategy with the variant's overrides.

    The original is never mutated. Raises ValueError on hard conflicts.
    """
    from dataclasses import replace  # local: keep imports light

    cleaned, _ = validate_overrides(variant.overrides)
    return replace(strategy, **cleaned)
