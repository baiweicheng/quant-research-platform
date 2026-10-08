"""In-repo S&P 500 membership updater.

qrp/data/sp500_history.csv (point-in-time membership: ticker, date_added,
date_removed) and qrp/data/sp500_tickers.txt (the current 503-member seed
list) are built by build_sp500_history.py from a hand-maintained
transcription of Wikipedia's S&P 500 change log. That transcription lags
real index changes by days or weeks (e.g. the October 2026 WBD->TWLO,
CTVA->VYLR and PSKY->SKYD changes had to be hand-patched from S&P Dow
Jones announcements because Wikipedia's log hadn't recorded them).

This module powers the dashboard's "Check for S&P 500 changes" button. In
plain English:

- CHECK fetches two Wikipedia pages (the live constituent table and the
  dated change log) and reports what's new versus the checked-in files:
  dated additions/removals, tickers that appeared/vanished with no dated
  log entry yet, suspected ticker renames, and date corrections for
  entries we previously recorded with an unconfirmed date.
- APPLY writes the reviewed changes into the CSV, the ticker list, and
  the builder's CHANGES log. REVIEW-BEFORE-APPLY is mandatory: this table
  feeds every backtest's point-in-time universe, so a bad parse must
  never silently corrupt it. Nothing is ever auto-applied.

Design notes (why it works this way):

- The updater never hand-edits the CSV. It appends to the builder's
  CHANGES log and regenerates the CSV with the builder's own reverse-walk
  (build_intervals), so the files it produces are exactly what the
  builder would produce. A self-check refuses the write if regeneration
  disagrees.
- Changes Wikipedia hasn't dated yet get the DETECTION date plus an
  "# UNCONFIRMED effective date" marker on the CHANGES entry. If
  Wikipedia's log later records the true date, the next check reports it
  as a date correction.
- Ticker renames (same company, new symbol — e.g. PSKY -> SKYD) are
  applied rename-in-place: the old symbol is swapped for the new one in
  the CHANGES entries and the seed list, keeping the original dates
  (FB->META / PCLN->BKNG / CDAY->DAY precedent), with no new dated entry.
- Rename detection: a ticker that vanishes from the live constituent
  table while exactly one new ticker appears (and Wikipedia's dated log
  says nothing) is flagged as a suspected rename. Dated log pairs are
  real composition changes (replacements), so they are applied as
  ordinary add/remove — but the report notes same-date pairs and the
  user can declare a rename explicitly if one really was a rename.
  (Evidence: Wikipedia's dated log never records pure renames — they
  appear only as free-text notes; the repo's rename precedents were all
  handled outside dated entries. Applying rename-in-place to a dated
  pair like EA -> FERG would wrongly give FERG EA's 1976 membership.)
- Honest limitation: this checker is only as fresh as Wikipedia, whose
  pages can lag S&P's announcements by days. The dashboard says so.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from qrp.data import build_sp500_history as _builder


class MembershipCheckError(Exception):
    """Plain-English failure of the membership check or apply step."""


WIKI_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_HIST_URL = (
    "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
)
USER_AGENT = "qrp-sp500-updater/1.0 (personal research project)"
EXPECTED_MEMBERS = 503

DATA_DIR = Path(__file__).resolve().parent
BUILDER_PATH = DATA_DIR / "build_sp500_history.py"
CSV_PATH = DATA_DIR / "sp500_history.csv"
TICKERS_PATH = DATA_DIR / "sp500_tickers.txt"
HISTORY_MODULE_PATH = DATA_DIR / "sp500_history.py"

UNCONFIRMED_MARK = "UNCONFIRMED"


# ---------------------------------------------------------------------------
# Wikipedia fetching + parsing (stdlib html.parser only — no new dependency)
# ---------------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Extract top-level HTML tables as lists of rows of cell text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._table = []
        elif self._depth == 1 and tag == "tr" and self._table is not None:
            self._row = []
        elif (
            self._depth == 1
            and tag in ("td", "th")
            and self._row is not None
            and self._cell is None
        ):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
            self._depth = max(0, self._depth - 1)
        elif tag == "tr" and self._depth == 1 and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif (
            tag in ("td", "th")
            and self._depth == 1
            and self._cell is not None
        ):
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            text = re.sub(r"\[\d+\]", "", text).strip()  # citation markers
            self._row.append(text)
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._depth == 1 and self._cell is not None:
            self._cell.append(data)


def _fetch_page(url: str, timeout: int = 30) -> str:
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        resp.raise_for_status()
    except requests.RequestException:
        raise MembershipCheckError(
            "Couldn't reach Wikipedia right now. Your membership list is "
            "unchanged — check your internet connection and try again later."
        )
    return resp.text


def parse_constituents(html: str) -> dict[str, dict[str, str]]:
    """Live constituent table -> {normalized ticker: {as_listed, name}}.

    Raises MembershipCheckError when the table can't be found or looks
    wrong (Wikipedia layout change) — never returns a partial guess.
    """
    parser = _TableParser()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        header = [c.lower() for c in table[0]]
        if "symbol" not in header:
            continue
        result: dict[str, dict[str, str]] = {}
        for row in table[1:]:
            if len(row) < 2:
                continue
            as_listed = row[0].strip()
            if not as_listed or as_listed.lower() == "symbol":
                continue
            norm = _builder.normalize(as_listed)
            result[norm] = {"as_listed": as_listed, "name": row[1].strip()}
        if not (495 <= len(result) <= 510):
            raise MembershipCheckError(
                "Wikipedia's S&P 500 table doesn't look the way we expected "
                f"(found {len(result)} tickers, expected about 503). The "
                "page layout may have changed — your list is unchanged."
            )
        return result
    raise MembershipCheckError(
        "Couldn't find the S&P 500 constituent table on Wikipedia's page — "
        "the page layout may have changed. Your list is unchanged."
    )


def _parse_wiki_date(text: str) -> str:
    """'September 21, 2026' -> '2026-09-21'. Raises ValueError if unknown."""
    text = text.strip()
    for fmt in ("%B %d, %Y", "%d %B %Y"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {text!r}")


def parse_change_log(html: str) -> list[dict]:
    """Dated change log -> [{date, added, removed, added_name, removed_name}].

    Tickers are normalized (BRK.B -> BRK-B); empty sides become None.
    Newest first. Raises MembershipCheckError when the table can't be
    found or yields nothing usable.
    """
    parser = _TableParser()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        header = [c.lower() for c in table[0]]
        if "effective date" not in header:
            continue
        changes: list[dict] = []
        for row in table[1:]:
            if len(row) < 5:
                continue
            try:
                iso = _parse_wiki_date(row[0])
            except ValueError:
                continue  # subheader or malformed row, not a change
            added = row[1].strip() or None
            removed = row[3].strip() or None
            if not added and not removed:
                continue
            changes.append(
                {
                    "date": iso,
                    "added": _builder.normalize(added) if added else None,
                    "removed": _builder.normalize(removed) if removed else None,
                    "added_name": row[2].strip(),
                    "removed_name": row[4].strip(),
                }
            )
        if not changes:
            raise MembershipCheckError(
                "Found Wikipedia's change log table but couldn't read any "
                "dated entries from it — the layout may have changed. Your "
                "list is unchanged."
            )
        changes.sort(key=lambda c: c["date"], reverse=True)
        return changes
    raise MembershipCheckError(
        "Couldn't find the dated change log on Wikipedia's historical "
        "components page — the layout may have changed. Your list is "
        "unchanged."
    )


def fetch_live_membership() -> tuple[dict[str, dict[str, str]], list[dict]]:
    """(constituents, dated_changes) from Wikipedia's live pages."""
    list_html = _fetch_page(WIKI_LIST_URL)
    hist_html = _fetch_page(WIKI_HIST_URL)
    return parse_constituents(list_html), parse_change_log(hist_html)


# ---------------------------------------------------------------------------
# CHANGES entry-line parsing (shared by diff + apply)
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r'^(?P<indent>\s*)\("(?P<date>\d{4}-\d{2}-\d{2})",\s*'
    r'(?P<added>None|"[^"]+"),\s*'
    r'(?P<removed>None|"[^"]+")\)(?P<tail>,?)\s*(?P<comment>#.*)?$'
)


def _parse_entry_line(line: str) -> dict | None:
    """A CHANGES source line -> {date, added, removed, comment} or None."""
    m = _ENTRY_RE.match(line)
    if not m:
        return None

    def _tick(s: str) -> str | None:
        return None if s == "None" else _builder.normalize(s[1:-1])

    return {
        "date": m.group("date"),
        "added": _tick(m.group("added")),
        "removed": _tick(m.group("removed")),
        "comment": m.group("comment") or "",
        "indent": m.group("indent"),
        "tail": m.group("tail"),
    }


def _render_entry_line(
    d: str, a: str | None, r: str | None, comment: str = ""
) -> str:
    a_s = f'"{a}"' if a else "None"
    r_s = f'"{r}"' if r else "None"
    line = f'    ("{d}", {a_s}, {r_s}),'
    if comment:
        line += f"  {comment}"
    return line


def _load_changes_with_flags(
    builder_path: Path = BUILDER_PATH,
) -> tuple[list[tuple[str, str | None, str | None]], set[tuple]]:
    """CHANGES parsed from the builder file + UNCONFIRMED-flagged entries.

    Parsed from the file (not the imported module) so the result always
    reflects what's on disk. The UNCONFIRMED marker lives in the source
    as a trailing comment (e.g. ``# UNCONFIRMED effective date (detected
    2026-10-07)``) so the builder itself treats the entry normally while
    later checks can spot it and correct the date once Wikipedia's log
    catches up.
    """
    changes: list[tuple[str, str | None, str | None]] = []
    unconfirmed: set[tuple] = set()
    for line in Path(builder_path).read_text(encoding="utf-8").splitlines():
        e = _parse_entry_line(line)
        if e is None:
            continue
        triple = (e["date"], e["added"], e["removed"])
        changes.append(triple)
        if UNCONFIRMED_MARK in e["comment"]:
            unconfirmed.add(triple)
    return changes, unconfirmed


def _recent_local_changes(
    builder_path: Path, wiki_max: str
) -> tuple[set[str], set[str]]:
    """Tickers added/removed by local CHANGES entries newer than wiki_max.

    These are changes already in the user's files (e.g. hand-patched from
    S&P announcements) that Wikipedia's log hasn't caught up with yet.
    """
    added: set[str] = set()
    removed: set[str] = set()
    for line in Path(builder_path).read_text(encoding="utf-8").splitlines():
        e = _parse_entry_line(line)
        if e is None or e["date"] <= wiki_max:
            continue
        if e["added"]:
            added.add(e["added"])
        if e["removed"]:
            removed.add(e["removed"])
    return added, removed


def _rename_already_recorded(
    builder_path: Path, old: str, new: str
) -> bool:
    """Does the builder's CHANGES list already document this rename?

    Only the CHANGES list region is searched (not the module docstring),
    so a passing mention elsewhere can't cause a false positive.
    """
    src = Path(builder_path).read_text(encoding="utf-8")
    start = src.find("CHANGES: list[tuple[str, str | None, str | None]] = [")
    region = src[start:] if start != -1 else src
    return bool(re.search(re.escape(old) + r"\s*->\s*" + re.escape(new), region))


def _local_tickers(tickers_path: Path = TICKERS_PATH) -> list[str]:
    """as-listed tickers from the seed list (comments/blanks skipped)."""
    out = []
    for line in Path(tickers_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Diff: what does Wikipedia know that the checked-in files don't?
# ---------------------------------------------------------------------------


def check_for_changes(
    list_html: str | None = None,
    hist_html: str | None = None,
    data_dir: Path = DATA_DIR,
    detection_date: str | None = None,
) -> dict:
    """Diff Wikipedia's live pages against the checked-in files.

    Returns a JSON-serializable diff dict (see module docstring). HTML may
    be injected (tests / UAT); otherwise it is fetched live.
    """
    detection_date = detection_date or date.today().isoformat()
    builder_path = Path(data_dir) / "build_sp500_history.py"
    tickers_path = Path(data_dir) / "sp500_tickers.txt"

    if list_html is None or hist_html is None:
        live_constituents, live_changes = fetch_live_membership()
    else:
        live_constituents = parse_constituents(list_html)
        live_changes = parse_change_log(hist_html)

    changes, unconfirmed = _load_changes_with_flags(builder_path)
    change_set = set(changes)
    confirmed = [c for c in changes if c not in unconfirmed]
    confirmed_end = max((c[0] for c in confirmed), default="0000-00-00")

    # 1. Date corrections for UNCONFIRMED entries Wikipedia has now dated.
    corrections: list[dict] = []
    corrected_pairs: set[tuple] = set()
    for d, a, r in sorted(unconfirmed):
        for e in live_changes:
            if e["added"] == a and e["removed"] == r:
                corrected_pairs.add((a, r))
                corrections.append(
                    {
                        "added": a,
                        "removed": r,
                        "old_date": d,
                        "new_date": e["date"],
                        "confirmed": e["date"] == d,
                    }
                )
                break

    # 2. New dated entries: newer than confirmed coverage (or same-day but
    #    not transcribed), excluding ones already handled as corrections.
    dated: list[dict] = []
    for e in live_changes:
        triple = (e["date"], e["added"], e["removed"])
        if triple in change_set:
            continue
        if (e["added"], e["removed"]) in corrected_pairs:
            continue
        if e["date"] > confirmed_end:
            e = dict(e)
            e["as_listed_added"] = (
                live_constituents[e["added"]]["as_listed"]
                if e["added"] in live_constituents
                else e["added"]
            )
            dated.append(e)

    # 3. Undated deltas: constituent table vs seed list, minus anything
    #    the dated entries already explain, minus changes already in the
    #    user's files that Wikipedia's log hasn't caught up with.
    live_norm = set(live_constituents)
    local_norm = {_builder.normalize(t) for t in _local_tickers(tickers_path)}
    dated_adds = {e["added"] for e in dated if e["added"]}
    dated_removes = {e["removed"] for e in dated if e["removed"]}
    raw_adds = {t for t in live_norm - local_norm if t not in dated_adds}
    raw_removes = {t for t in local_norm - live_norm if t not in dated_removes}

    wiki_max = max(e["date"] for e in live_changes)
    recent_added, recent_removed = _recent_local_changes(
        builder_path, wiki_max
    )
    already_recorded = sorted(
        {t for t in raw_adds if t in recent_removed}
        | {t for t in raw_removes if t in recent_added}
    )
    raw_adds = {t for t in raw_adds if t not in recent_removed}
    raw_removes = {t for t in raw_removes if t not in recent_added}

    # 4. Rename heuristic (undated only): exactly one ticker vanished while
    #    exactly one appeared and the dated log says nothing -> suspected
    #    ticker rename (the PSKY -> SKYD pattern). Applied rename-in-place.
    #    If the rename is already documented in the builder file, it's
    #    Wikipedia lagging, not a new change.
    renames: list[dict] = []
    if len(raw_adds) == 1 and len(raw_removes) == 1:
        live_only = next(iter(raw_adds))
        local_only = next(iter(raw_removes))
        # The swap could be a rename in either direction; the builder's
        # rename note (if any) tells us which way it actually went.
        if _rename_already_recorded(builder_path, live_only, local_only):
            # live_only is the old ticker, local_only the new: already done.
            old_t, new_t = live_only, local_only
        elif _rename_already_recorded(builder_path, local_only, live_only):
            old_t, new_t = local_only, live_only
        else:
            # New rename: the local list still has the old ticker, the
            # live list already shows the new one.
            old_t, new_t = local_only, live_only
        if _rename_already_recorded(builder_path, old_t, new_t):
            already_recorded = sorted(set(already_recorded) | {old_t, new_t})
        else:
            renames.append(
                {
                    "old": old_t,
                    "new": new_t,
                    "date": None,
                    "old_name": "",
                    "new_name": live_constituents[new_t]["name"],
                    "new_as_listed": live_constituents[new_t]["as_listed"],
                }
            )
        raw_adds, raw_removes = set(), set()

    undated_adds = [
        {
            "ticker": t,
            "name": live_constituents[t]["name"],
            "as_listed": live_constituents[t]["as_listed"],
        }
        for t in sorted(raw_adds)
    ]
    undated_removes = [{"ticker": t} for t in sorted(raw_removes)]

    diff = {
        "checked_on": detection_date,
        "up_to_date": not (
            dated or renames or undated_adds or undated_removes or corrections
        ),
        "dated": dated,
        "renames": renames,
        "undated_adds": undated_adds,
        "undated_removes": undated_removes,
        "corrections": corrections,
        "already_recorded": already_recorded,
    }
    return diff


def describe_diff(diff: dict) -> list[str]:
    """Plain-English lines describing a diff, for the dashboard report."""
    lag_note = ""
    if diff.get("already_recorded"):
        lag_note = (
            " (Wikipedia's pages are still catching up on changes already "
            f"in your files: {', '.join(diff['already_recorded'])}.)"
        )
    if diff["up_to_date"]:
        return [
            "Your S&P 500 list is current — last checked "
            f"{diff['checked_on']}.{lag_note}"
        ]
    lines: list[str] = []
    for e in diff["dated"]:
        bits = []
        if e["added"]:
            name = f" ({e['added_name']})" if e["added_name"] else ""
            bits.append(f"Added: {e['added']}{name}")
        if e["removed"]:
            name = f" ({e['removed_name']})" if e["removed_name"] else ""
            bits.append(f"Removed: {e['removed']}{name}")
        lines.append(f"{'; '.join(bits)} (effective {e['date']}).")
    for r in diff["renames"]:
        name = f" ({r['new_name']})" if r["new_name"] else ""
        lines.append(
            f"Suspected rename: {r['old']} -> {r['new']}{name} — the old "
            "ticker vanished from the live list while this one appeared, "
            "with no dated log entry. It will be applied as a rename (same "
            "company, new symbol)."
        )
    for a in diff["undated_adds"]:
        name = f" ({a['name']})" if a["name"] else ""
        lines.append(
            "Added (no dated log entry yet — will use today's date, marked "
            f"unconfirmed): {a['ticker']}{name}."
        )
    for rm in diff["undated_removes"]:
        lines.append(
            "Removed (no dated log entry yet — will use today's date, "
            f"marked unconfirmed): {rm['ticker']}."
        )
    for c in diff["corrections"]:
        pair = "/".join(p for p in (c["added"], c["removed"]) if p)
        if c["confirmed"]:
            lines.append(
                f"Confirmed: {pair} really was effective {c['new_date']} — "
                "dropping the unconfirmed mark."
            )
        else:
            lines.append(
                f"Date correction: {pair} moves from unconfirmed "
                f"{c['old_date']} to {c['new_date']}."
            )
    # Same-date pairs are usually replacements, but flag the possibility.
    by_date: dict[str, dict[str, list]] = {}
    for e in diff["dated"]:
        slot = by_date.setdefault(e["date"], {"added": [], "removed": []})
        if e["added"]:
            slot["added"].append(e["added"])
        if e["removed"]:
            slot["removed"].append(e["removed"])
    for d in sorted(by_date):
        adds, removes = by_date[d]["added"], by_date[d]["removed"]
        if len(adds) == 1 and len(removes) == 1:
            lines.append(
                f"Note: {adds[0]} and {removes[0]} share effective date {d}. "
                "If that was actually a ticker rename (same company, new "
                "symbol) rather than a replacement, declare it below as "
                f"'{removes[0]} -> {adds[0]}'."
            )
        elif adds and removes:
            lines.append(
                f"Note: on {d}, additions ({', '.join(adds)}) and removals "
                f"({', '.join(removes)}) share the same effective date. If "
                "any pair was actually a ticker rename (same company, new "
                "symbol), declare it below as 'OLD -> NEW'."
            )
    lines.append(
        "Heads-up: this check reads Wikipedia's pages, which can lag S&P's "
        "own announcements by days. If you know of a change Wikipedia "
        "hasn't recorded yet, wait for the next check."
    )
    if diff.get("already_recorded"):
        lines.append(
            "Already in your files (Wikipedia's pages haven't caught up "
            f"yet — nothing to do): {', '.join(diff['already_recorded'])}."
        )
    return lines


def parse_rename_declarations(text: str) -> list[tuple[str, str]]:
    """'PSKY -> SKYD' lines (one per line or comma-separated) -> [(old, new)]."""
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"[\n,]+", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "->" not in chunk:
            raise MembershipCheckError(
                f"Couldn't understand the rename entry {chunk!r} — use the "
                "form 'OLD -> NEW', e.g. 'PSKY -> SKYD'."
            )
        old, new = (p.strip() for p in chunk.split("->", 1))
        if not old or not new:
            raise MembershipCheckError(
                f"Couldn't understand the rename entry {chunk!r} — use the "
                "form 'OLD -> NEW'."
            )
        out.append((_builder.normalize(old), _builder.normalize(new)))
    return out


# ---------------------------------------------------------------------------
# Apply (review-before-apply: only called from the dashboard's Apply button)
# ---------------------------------------------------------------------------


def _build_new_builder_source(
    diff: dict,
    renames: list[dict],
    detection_date: str,
    builder_path: Path = BUILDER_PATH,
) -> str:
    """CHANGES with corrections fixed, renames swapped, new entries added."""
    src_lines = Path(builder_path).read_text(encoding="utf-8").splitlines()
    rename_map = {r["old"]: r["new"] for r in renames}
    rename_dates = {r["old"]: (r["date"] or detection_date) for r in renames}
    correction_map = {
        (c["old_date"], c["added"], c["removed"]): c for c in diff["corrections"]
    }

    out_lines: list[str] = []
    for line in src_lines:
        e = _parse_entry_line(line)
        if e is None:
            out_lines.append(line)
            continue
        key = (e["date"], e["added"], e["removed"])
        if key in correction_map:
            # Wikipedia has dated it now: fix the date, drop the marker.
            c = correction_map[key]
            out_lines.append(
                _render_entry_line(c["new_date"], c["added"], c["removed"])
            )
            continue
        new_added = rename_map.get(e["added"], e["added"]) if e["added"] else None
        new_removed = (
            rename_map.get(e["removed"], e["removed"]) if e["removed"] else None
        )
        if new_added != e["added"] or new_removed != e["removed"]:
            touched = [
                o for o in (e["added"], e["removed"]) if o in rename_map
            ]
            note = ", ".join(
                f"{o}->{rename_map[o]} {rename_dates[o]}" for o in touched
            )
            out_lines.append(
                _render_entry_line(
                    e["date"], new_added, new_removed, f"# {note}"
                )
            )
        else:
            out_lines.append(line)

    # New entries (dated + undated), newest first, after the CHANGES line.
    new_entries: list[tuple[str, str | None, str | None, str]] = []
    for e in diff["dated"]:
        new_entries.append((e["date"], e["added"], e["removed"], ""))
    for a in diff["undated_adds"]:
        new_entries.append(
            (
                detection_date,
                a["ticker"],
                None,
                f"# {UNCONFIRMED_MARK} effective date "
                f"(detected {detection_date}; not in Wikipedia log yet)",
            )
        )
    for rm in diff["undated_removes"]:
        new_entries.append(
            (
                detection_date,
                None,
                rm["ticker"],
                f"# {UNCONFIRMED_MARK} effective date "
                f"(detected {detection_date}; not in Wikipedia log yet)",
            )
        )
    # (renames need no new CHANGES entry: the symbol swap above encodes
    # rename-in-place, exactly like the FB->META / PSKY->SKYD precedents.)
    new_entries.sort(key=lambda t: t[0], reverse=True)

    if new_entries:
        anchor = "CHANGES: list[tuple[str, str | None, str | None]] = ["
        joined = "\n".join(out_lines)
        if anchor not in joined:
            raise MembershipCheckError(
                "The builder file's CHANGES list doesn't look the way we "
                "expected — refusing to edit it by hand-guess. Nothing "
                "was written."
            )
        block = "\n".join(
            _render_entry_line(d, a, r, c) for d, a, r, c in new_entries
        )
        joined = joined.replace(anchor, anchor + "\n" + block, 1)
        return joined
    return "\n".join(out_lines)


def _build_new_ticker_list(
    diff: dict,
    renames: list[dict],
    tickers_path: Path = TICKERS_PATH,
) -> str:
    """Seed list with removals dropped, additions appended, renames swapped.

    New tickers keep Wikipedia's as-listed spelling (e.g. BRK.B); the
    platform normalizes dots to dashes when it downloads.
    """
    rename_map = {r["old"]: r["new"] for r in renames}
    rename_listed = {r["old"]: r.get("new_as_listed") or r["new"] for r in renames}
    remove_norm: set[str] = set()
    for e in diff["dated"]:
        if e["removed"]:
            remove_norm.add(e["removed"])
    for rm in diff["undated_removes"]:
        remove_norm.add(rm["ticker"])

    add_forms: list[tuple[str, str]] = []  # (normalized, as_listed)
    for e in diff["dated"]:
        if e["added"]:
            add_forms.append((e["added"], e.get("as_listed_added") or e["added"]))
    for a in diff["undated_adds"]:
        add_forms.append((a["ticker"], a.get("as_listed") or a["ticker"]))

    out: list[str] = []
    for line in Path(tickers_path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue
        n = _builder.normalize(s)
        if n in remove_norm:
            continue
        if n in rename_map:
            out.append(rename_listed[n])
            continue
        out.append(line)

    present = {
        _builder.normalize(l.strip())
        for l in out
        if l.strip() and not l.strip().startswith("#")
    }
    for norm, listed in add_forms:
        if norm not in present:
            out.append(listed)
            present.add(norm)
    # Rename targets are already in place via the swap above.
    return "\n".join(out) + "\n"


def _render_csv(intervals: list[tuple[str, str, str]]) -> str:
    buf = ["ticker,date_added,date_removed"]
    buf.extend(f"{t},{a},{r}" for t, a, r in intervals)
    return "\n".join(buf) + "\n"


def _update_module_constants(
    new_dates: list[str], detection_date: str, data_dir: Path
) -> None:
    """Bump RETRIEVED / coverage constants so user-facing notes stay true."""
    coverage_end = max(new_dates) if new_dates else None

    hist_path = Path(data_dir) / "sp500_history.py"
    src = hist_path.read_text(encoding="utf-8")
    new_src = re.sub(
        r'^RETRIEVED = "\d{4}-\d{2}-\d{2}"$',
        f'RETRIEVED = "{detection_date}"',
        src,
        count=1,
        flags=re.MULTILINE,
    )
    if coverage_end:
        new_src = re.sub(
            r'^COVERAGE_END = "\d{4}-\d{2}-\d{2}"$',
            f'COVERAGE_END = "{coverage_end}"',
            new_src,
            count=1,
            flags=re.MULTILINE,
        )
    if new_src == src:
        raise MembershipCheckError(
            "The membership reader file's date constants don't look the "
            "way we expected — refusing to guess. Nothing was written."
        )
    hist_path.write_text(new_src, encoding="utf-8")

    builder_path = Path(data_dir) / "build_sp500_history.py"
    bsrc = builder_path.read_text(encoding="utf-8")
    new_bsrc = re.sub(
        r'^RETRIEVED = "\d{4}-\d{2}-\d{2}"$',
        f'RETRIEVED = "{detection_date}"',
        bsrc,
        count=1,
        flags=re.MULTILINE,
    )
    if coverage_end:
        new_bsrc = re.sub(
            r"Retrieved: \d{4}-\d{2}-\d{2}\. Change-log coverage: "
            r"1976-07-01 -> \d{4}-\d{2}-\d{2}\.",
            f"Retrieved: {detection_date}. Change-log coverage: "
            f"1976-07-01 -> {coverage_end}.",
            new_bsrc,
            count=1,
        )
    if new_bsrc == bsrc:
        raise MembershipCheckError(
            "The builder file's date constants don't look the way we "
            "expected — refusing to guess. Nothing was written."
        )
    builder_path.write_text(new_bsrc, encoding="utf-8")


def _count_members(tickers_text: str) -> int:
    return sum(
        1
        for line in tickers_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def apply_updates(
    diff: dict,
    rename_declarations: str = "",
    data_dir: Path = DATA_DIR,
    detection_date: str | None = None,
    expected_members: int = EXPECTED_MEMBERS,
) -> dict:
    """Apply a reviewed diff. Returns a plain-English summary dict.

    Steps: validate declared renames -> plan new file contents -> refuse
    unless the seed list still has exactly `expected_members` members ->
    regenerate the CSV with the builder's own reverse-walk in a scratch
    dir and refuse on new anomalies -> write everything -> bump date
    constants. Raises MembershipCheckError (plain English, nothing
    half-written) on any problem.
    """
    data_dir = Path(data_dir)
    builder_path = data_dir / "build_sp500_history.py"
    tickers_path = data_dir / "sp500_tickers.txt"
    csv_path = data_dir / "sp500_history.csv"
    detection_date = detection_date or date.today().isoformat()

    if diff.get("up_to_date"):
        raise MembershipCheckError(
            "Nothing to apply — the check found no changes."
        )

    # Declared renames must refer to pairs this diff actually found.
    declared = parse_rename_declarations(rename_declarations)
    removing = {
        _builder.normalize(e["removed"]) for e in diff["dated"] if e["removed"]
    }
    removing |= {rm["ticker"] for rm in diff["undated_removes"]}
    adding = {
        _builder.normalize(e["added"]) for e in diff["dated"] if e["added"]
    }
    adding |= {a["ticker"] for a in diff["undated_adds"]}
    renames: list[dict] = [dict(r) for r in diff["renames"]]
    auto_pairs = {(r["old"], r["new"]) for r in renames}
    for old, new in declared:
        if (old, new) in auto_pairs:
            continue  # already flagged as a rename; nothing more to do
        if old not in removing or new not in adding:
            raise MembershipCheckError(
                f"Couldn't apply the rename {old} -> {new}: that pair "
                "isn't among the changes found. Nothing was written."
            )
        pair_date = next(
            (
                e["date"]
                for e in diff["dated"]
                if e["added"] == new and e["removed"] == old
            ),
            detection_date,
        )
        new_as_listed = next(
            (
                a.get("as_listed") or new
                for a in diff["undated_adds"]
                if a["ticker"] == new
            ),
            next(
                (
                    e.get("as_listed_added") or new
                    for e in diff["dated"]
                    if e["added"] == new
                ),
                new,
            ),
        )
        renames.append(
            {
                "old": old,
                "new": new,
                "date": pair_date,
                "old_name": "",
                "new_name": "",
                "new_as_listed": new_as_listed,
            }
        )
        # A declared rename supersedes the plain add/remove entries.
        diff["dated"] = [
            e
            for e in diff["dated"]
            if not (e["added"] == new and e["removed"] == old)
        ]
        diff["undated_adds"] = [
            a for a in diff["undated_adds"] if a["ticker"] != new
        ]
        diff["undated_removes"] = [
            r for r in diff["undated_removes"] if r["ticker"] != old
        ]

    new_builder_source = _build_new_builder_source(
        diff, renames, detection_date, builder_path
    )
    new_tickers_text = _build_new_ticker_list(diff, renames, tickers_path)

    n_members = _count_members(new_tickers_text)
    if n_members != expected_members:
        raise MembershipCheckError(
            f"Refusing to write: the updated list would have {n_members} "
            f"members instead of {expected_members}. Nothing was written — "
            "please report this so it can be investigated."
        )

    # Validate by regenerating in a scratch dir with the builder itself:
    # baseline anomalies come from the files as they are now (exec'd, so
    # always fresh), and the new files must not introduce new ones.
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        current_source = builder_path.read_text(encoding="utf-8")
        current_tickers = tickers_path.read_text(encoding="utf-8")
        (tmp_path / "build_sp500_history.py").write_text(
            current_source, encoding="utf-8"
        )
        (tmp_path / "sp500_tickers.txt").write_text(
            current_tickers, encoding="utf-8"
        )
        ns0: dict = {"__file__": str(tmp_path / "build_sp500_history.py")}
        exec(compile(current_source, "<builder_baseline>", "exec"), ns0)
        _b_intervals, _b_anomalies, _b_dropped = ns0["build_intervals"]()
        baseline_anomalies = set(_b_anomalies)

        (tmp_path / "build_sp500_history.py").write_text(
            new_builder_source, encoding="utf-8"
        )
        (tmp_path / "sp500_tickers.txt").write_text(
            new_tickers_text, encoding="utf-8"
        )
        namespace: dict = {"__file__": str(tmp_path / "build_sp500_history.py")}
        try:
            exec(
                compile(new_builder_source, "<sp500_builder>", "exec"),
                namespace,
            )
        except SyntaxError as exc:
            raise MembershipCheckError(
                "The generated builder file didn't parse — nothing was "
                f"written (technical detail: {exc})."
            )
        intervals, anomalies, _dropped = namespace["build_intervals"]()
        fresh = set(anomalies) - baseline_anomalies
        if fresh:
            raise MembershipCheckError(
                "Refusing to write: regenerating the membership table "
                "surfaced new inconsistencies "
                f"({'; '.join(sorted(fresh)[:3])}). Nothing was written."
            )
        open_now = {t for t, _a, r in intervals if not r}
        want = {
            _builder.normalize(l)
            for l in new_tickers_text.splitlines()
            if l.strip() and not l.strip().startswith("#")
        }
        if open_now != want:
            raise MembershipCheckError(
                "Refusing to write: the regenerated table's current members "
                "don't match the updated ticker list. Nothing was written."
            )
        new_csv_text = _render_csv(intervals)

    # All checks passed — write for real.
    builder_path.write_text(new_builder_source, encoding="utf-8")
    tickers_path.write_text(new_tickers_text, encoding="utf-8")
    csv_path.write_text(new_csv_text, encoding="utf-8")

    applied_dates = (
        [e["date"] for e in diff["dated"]]
        + [c["new_date"] for c in diff["corrections"]]
        + (
            [detection_date]
            if (diff["undated_adds"] or diff["undated_removes"])
            else []
        )
    )
    _update_module_constants(applied_dates, detection_date, data_dir)

    added_tickers = sorted(
        {e["added"] for e in diff["dated"] if e["added"]}
        | {a["ticker"] for a in diff["undated_adds"]}
        | {r["new"] for r in renames}
    )
    summary_lines = describe_diff({**diff, "up_to_date": False})
    return {
        "applied": True,
        "checked_on": detection_date,
        "summary_lines": summary_lines,
        "new_tickers": added_tickers,
        "next_steps": (
            "Next: on the Market data page, use Download data with a Custom "
            f"list of {', '.join(added_tickers)} "
            "to download their price history (and fundamentals), then press "
            "'Compute all signals' on the Signals page."
            if added_tickers
            else "Next: recompute signals on the Signals page if anything "
            "about the membership changed."
        ),
        "git_note": (
            "Note: this updated the checked-in membership files in your "
            "local copy of the repo. If a future 'git pull' complains about "
            "these files, it's safe to take the repo version — it contains "
            "the same change."
        ),
    }


# ---------------------------------------------------------------------------
# Dashboard check-state persistence (survives page navigation)
# ---------------------------------------------------------------------------


def read_check_status(status_path: str | Path) -> dict:
    """Last check state, or a never-checked default. Never raises."""
    default = {
        "last_checked": None,
        "outcome": "never",
        "detail": "",
        "pending": None,
    }
    try:
        raw = Path(status_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return {**default, **data}
    except (OSError, ValueError):
        return default


def record_check(
    status_path: str | Path,
    outcome: str,
    detail: str = "",
    pending: dict | None = None,
) -> None:
    """Persist a check outcome. outcome: up_to_date|changes_found|applied|error."""
    payload = {
        "last_checked": date.today().isoformat(),
        "outcome": outcome,
        "detail": detail,
        "pending": pending,
    }
    tmp = Path(str(status_path) + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(status_path)