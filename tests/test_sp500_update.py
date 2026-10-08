"""Tests for qrp/data/sp500_update.py (S&P 500 membership updater)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qrp.data import sp500_update as U  # noqa: E402

FIXTURE_BUILDER = '''"""Minimal builder fixture for updater tests."""

from __future__ import annotations

import csv
import os
from pathlib import Path

RETRIEVED = "2026-01-01"

HERE = os.path.dirname(os.path.abspath(__file__))


def normalize(ticker):
    if ticker is None:
        return None
    return str(ticker).strip().upper().replace(".", "-")


def current_tickers():
    out = []
    with open(os.path.join(HERE, "sp500_tickers.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


CHANGES: list[tuple[str, str | None, str | None]] = [
    ("2026-03-01", "DDD", "CCC"),
    ("2026-01-15", "BBB", None),
    ("2026-01-10", None, "EEE"),
]


def build_intervals():
    changes = sorted(
        ((d, normalize(a), normalize(r)) for d, a, r in CHANGES),
        key=lambda r: r[0],
        reverse=True,
    )
    open_end = {normalize(t): None for t in current_tickers()}
    intervals = []
    anomalies = []
    for d, a, r in changes:
        if a is not None:
            if a in open_end:
                intervals.append((a, d, open_end.pop(a) or ""))
            else:
                anomalies.append(f"added {a} on {d} but not a current member")
                open_end[a] = None
        if r is not None:
            if r in open_end:
                intervals.append((r, "1976-07-01", d))
                del open_end[r]
            else:
                open_end[r] = d
    dropped = sorted(set(t for t, _, _ in changes) - set(current_tickers()))
    for t, end in sorted(open_end.items()):
        intervals.append((t, "1976-07-01", end or ""))
    intervals.sort()
    return intervals, anomalies, dropped
'''

FIXTURE_HISTORY_MODULE = '''"""Minimal sp500_history fixture."""

from __future__ import annotations

RETRIEVED = "2026-01-01"
COVERAGE_END = "2026-03-01"
'''

INTERESTING = ["AAA", "BBB", "DDD", "FFF", "GGG"]
FILLERS = [f"Z{i:03d}" for i in range(498)]  # 503 members total


def fixture_tickers():
    return INTERESTING + FILLERS


def make_constituent_html(tickers):
    """tickers: list of as-listed symbols (names derived)."""
    rows = ["<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>"]
    for i, sym in enumerate(tickers):
        rows.append(
            f'<tr><td><a href="https://en.wikipedia.org/wiki/X{i}">{sym}</a>'
            f"</td><td>{sym} Company</td><td>Industrials</td></tr>"
        )
    return (
        '<html><body><table class="wikitable sortable">'
        + "".join(rows)
        + "</table></body></html>"
    )


def make_log_html(entries):
    """entries: list of (date_str, added, added_name, removed, removed_name)."""
    rows = [
        "<tr><th>Effective Date</th><th>Ticker</th><th>Security</th>"
        "<th>Ticker</th><th>Security</th><th>Reason</th></tr>",
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>",
    ]
    for d, a, an, r, rn in entries:
        rows.append(
            f"<tr><td>{d}</td><td>{a or ''}</td><td>{an}</td>"
            f"<td>{r or ''}</td><td>{rn}</td><td>test</td><td></td></tr>"
        )
    return (
        '<html><body><table class="wikitable sortable">'
        + "".join(rows)
        + "</table></body></html>"
    )


@pytest.fixture()
def datadir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "build_sp500_history.py").write_text(FIXTURE_BUILDER)
    (d / "sp500_history.py").write_text(FIXTURE_HISTORY_MODULE)
    (d / "sp500_tickers.txt").write_text(
        "# fixture\n" + "\n".join(fixture_tickers()) + "\n"
    )
    (d / "sp500_history.csv").write_text("ticker,date_added,date_removed\n")
    return d


def _check(datadir, table_tickers, log_entries):
    return U.check_for_changes(
        list_html=make_constituent_html(table_tickers),
        hist_html=make_log_html(log_entries),
        data_dir=datadir,
        detection_date="2026-04-01",
    )


def _blank_diff(**over):
    d = {
        "checked_on": "2026-04-01",
        "up_to_date": False,
        "dated": [],
        "renames": [],
        "undated_adds": [],
        "undated_removes": [],
        "corrections": [],
        "already_recorded": [],
    }
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_constituents_ok():
    tickers = [f"TKR{i:03d}" for i in range(500)] + ["BRK.B"]
    out = U.parse_constituents(make_constituent_html(tickers))
    assert len(out) == 501
    assert out["BRK-B"]["as_listed"] == "BRK.B"
    assert out["BRK-B"]["name"] == "BRK.B Company"
    assert out["TKR000"]["as_listed"] == "TKR000"


def test_parse_constituents_wrong_count_raises():
    with pytest.raises(U.MembershipCheckError, match="doesn't look"):
        U.parse_constituents(make_constituent_html(["AAA", "BBB"]))


def test_parse_constituents_no_table_raises():
    with pytest.raises(U.MembershipCheckError, match="constituent table"):
        U.parse_constituents("<html><body><p>no tables</p></body></html>")


def test_parse_change_log_ok():
    out = U.parse_change_log(
        make_log_html(
            [
                ("March 1, 2026", "DDD", "D Corp", "CCC", "C Corp"),
                ("January 15, 2026", "BBB", "B Corp", "", ""),
                ("January 10, 2026", "", "", "EEE", "E Corp"),
            ]
        )
    )
    assert len(out) == 3
    assert out[0]["date"] == "2026-03-01"
    assert out[0]["added"] == "DDD" and out[0]["removed"] == "CCC"
    assert out[0]["added_name"] == "D Corp"
    assert out[1]["added"] == "BBB" and out[1]["removed"] is None
    assert out[2]["added"] is None and out[2]["removed"] == "EEE"


def test_parse_change_log_no_table_raises():
    with pytest.raises(U.MembershipCheckError, match="change log"):
        U.parse_change_log("<html><body><p>no tables</p></body></html>")


def test_parse_change_log_normalizes_dots():
    out = U.parse_change_log(make_log_html([("March 1, 2026", "BRK.B", "B", "", "")]))
    assert out[0]["added"] == "BRK-B"


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_up_to_date(datadir):
    diff = _check(
        datadir,
        fixture_tickers(),
        [("March 1, 2026", "DDD", "D", "CCC", "C")],
    )
    assert diff["up_to_date"] is True
    assert U.describe_diff(diff) == [
        "Your S&P 500 list is current — last checked 2026-04-01."
    ]


def test_diff_new_dated_entry(datadir):
    diff = _check(
        datadir,
        fixture_tickers(),
        [
            ("March 20, 2026", "HHH", "H Corp", "FFF", "F Corp"),
            ("March 1, 2026", "DDD", "D", "CCC", "C"),
        ],
    )
    assert diff["up_to_date"] is False
    assert len(diff["dated"]) == 1
    e = diff["dated"][0]
    assert (e["date"], e["added"], e["removed"]) == ("2026-03-20", "HHH", "FFF")
    # the dated entry explains the table delta -> no undated noise
    assert diff["undated_adds"] == [] and diff["undated_removes"] == []


def test_diff_ignores_entries_within_coverage(datadir):
    diff = _check(
        datadir,
        fixture_tickers(),
        [
            ("March 1, 2026", "DDD", "D Corp", "CCC", "C Corp"),
            ("January 15, 2026", "BBB", "B Corp", "", ""),
        ],
    )
    assert diff["dated"] == []
    assert diff["up_to_date"] is True


def test_diff_undated_1to1_is_suspected_rename(datadir):
    table = [t for t in fixture_tickers() if t != "FFF"] + ["HHH"]
    diff = _check(
        datadir, table, [("March 1, 2026", "DDD", "D", "CCC", "C")]
    )
    assert diff["up_to_date"] is False
    assert diff["dated"] == []
    assert len(diff["renames"]) == 1
    assert diff["renames"][0]["old"] == "FFF"
    assert diff["renames"][0]["new"] == "HHH"
    assert diff["undated_adds"] == [] and diff["undated_removes"] == []


def test_diff_no_rename_when_multiple(datadir):
    table = [t for t in fixture_tickers() if t not in ("FFF", "GGG")] + [
        "HHH",
        "III",
    ]
    diff = _check(
        datadir, table, [("March 1, 2026", "DDD", "D", "CCC", "C")]
    )
    assert diff["renames"] == []
    assert {a["ticker"] for a in diff["undated_adds"]} == {"HHH", "III"}
    assert {r["ticker"] for r in diff["undated_removes"]} == {"FFF", "GGG"}


def test_diff_correction_for_unconfirmed(datadir):
    src = (datadir / "build_sp500_history.py").read_text()
    anchor = "CHANGES: list[tuple[str, str | None, str | None]] = ["
    src = src.replace(
        anchor,
        anchor
        + '\n    ("2026-03-25", "HHH", None),  # UNCONFIRMED effective date '
        "(detected 2026-03-25; not in Wikipedia log yet)",
        1,
    )
    (datadir / "build_sp500_history.py").write_text(src)
    # HHH was added locally by that unconfirmed apply: swap a filler for it
    tickers = [t for t in fixture_tickers() if t != "Z000"] + ["HHH"]
    (datadir / "sp500_tickers.txt").write_text("# fixture\n" + "\n".join(tickers) + "\n")
    diff = _check(
        datadir,
        tickers,
        [
            ("March 28, 2026", "HHH", "H Corp", "", ""),
            ("March 1, 2026", "DDD", "D", "CCC", "C"),
        ],
    )
    assert len(diff["corrections"]) == 1
    c = diff["corrections"][0]
    assert (c["added"], c["old_date"], c["new_date"]) == (
        "HHH",
        "2026-03-25",
        "2026-03-28",
    )
    assert diff["dated"] == []  # not double-reported as new


def test_diff_already_recorded_when_local_ahead(datadir):
    src = (datadir / "build_sp500_history.py").read_text()
    anchor = "CHANGES: list[tuple[str, str | None, str | None]] = ["
    src = src.replace(
        anchor, anchor + '\n    ("2026-03-20", "HHH", "FFF"),', 1
    )
    (datadir / "build_sp500_history.py").write_text(src)
    tickers = [t for t in fixture_tickers() if t != "FFF"] + ["HHH"]
    (datadir / "sp500_tickers.txt").write_text("# fixture\n" + "\n".join(tickers) + "\n")
    # Wikipedia table still shows FFF (hasn't caught up); log max 2026-03-01
    diff = _check(
        datadir,
        fixture_tickers(),
        [("March 1, 2026", "DDD", "D", "CCC", "C")],
    )
    assert diff["up_to_date"] is True
    assert sorted(diff["already_recorded"]) == ["FFF", "HHH"]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _apply(datadir, diff, **kw):
    return U.apply_updates(
        diff, data_dir=datadir, detection_date="2026-04-01", **kw
    )


def _csv_rows(datadir):
    return (datadir / "sp500_history.csv").read_text().splitlines()


def _tickers(datadir):
    return U._local_tickers(datadir / "sp500_tickers.txt")


def test_apply_undated_add_remove(datadir):
    diff = _blank_diff(
        undated_adds=[{"ticker": "HHH", "name": "H Corp", "as_listed": "HHH"}],
        undated_removes=[{"ticker": "FFF"}],
    )
    summary = _apply(datadir, diff)
    assert summary["applied"] is True
    rows = _csv_rows(datadir)
    assert "HHH,2026-04-01," in rows
    assert "FFF,1976-07-01,2026-04-01" in rows
    src = (datadir / "build_sp500_history.py").read_text()
    assert "# UNCONFIRMED effective date (detected 2026-04-01" in src
    tickers = _tickers(datadir)
    assert len(tickers) == 503 and "HHH" in tickers and "FFF" not in tickers
    assert "HHH" in summary["new_tickers"]
    assert "Custom list" in summary["next_steps"]


def test_apply_503_guard_refuses(datadir):
    diff = _blank_diff(
        undated_adds=[
            {"ticker": "HHH", "name": "H", "as_listed": "HHH"},
            {"ticker": "III", "name": "I", "as_listed": "III"},
        ],
        undated_removes=[{"ticker": "FFF"}],
    )
    before = {
        p: (datadir / p).read_text()
        for p in (
            "build_sp500_history.py",
            "sp500_history.csv",
            "sp500_tickers.txt",
            "sp500_history.py",
        )
    }
    with pytest.raises(U.MembershipCheckError, match="instead of 503"):
        _apply(datadir, diff)
    for p, text in before.items():
        assert (datadir / p).read_text() == text  # nothing half-written


def test_apply_dated_entry_keeps_true_date(datadir):
    diff = _blank_diff(
        dated=[
            {
                "date": "2026-03-20",
                "added": "HHH",
                "removed": "FFF",
                "added_name": "H Corp",
                "removed_name": "F Corp",
                "as_listed_added": "HHH",
            }
        ]
    )
    _apply(datadir, diff)
    rows = _csv_rows(datadir)
    assert "HHH,2026-03-20," in rows
    assert "FFF,1976-07-01,2026-03-20" in rows
    src = (datadir / "build_sp500_history.py").read_text()
    assert '("2026-03-20", "HHH", "FFF"),' in src
    assert "UNCONFIRMED" not in src  # dated entries are confirmed


def test_apply_rename_in_place(datadir):
    # BBB was added 2026-01-15 via CHANGES; renaming keeps that date.
    diff = _blank_diff(
        renames=[
            {
                "old": "BBB",
                "new": "HHH",
                "date": None,
                "old_name": "",
                "new_name": "H Corp",
                "new_as_listed": "HHH",
            }
        ]
    )
    _apply(datadir, diff)
    rows = _csv_rows(datadir)
    assert "HHH,2026-01-15," in rows  # original start date kept
    assert not any(r.startswith("BBB,") for r in rows)
    src = (datadir / "build_sp500_history.py").read_text()
    assert '("2026-01-15", "HHH", None),' in src
    assert "BBB->HHH" in src  # rename note recorded
    tickers = _tickers(datadir)
    assert "HHH" in tickers and "BBB" not in tickers and len(tickers) == 503


def test_apply_declared_rename_overrides_dated_pair(datadir):
    diff = _blank_diff(
        dated=[
            {
                "date": "2026-03-20",
                "added": "HHH",
                "removed": "FFF",
                "added_name": "H Corp",
                "removed_name": "F Corp",
                "as_listed_added": "HHH",
            }
        ]
    )
    _apply(datadir, diff, rename_declarations="FFF -> HHH")
    rows = _csv_rows(datadir)
    # rename-in-place: HHH inherits FFF's 1976 start, no dated swap
    assert "HHH,1976-07-01," in rows
    assert not any(r.startswith("FFF,") for r in rows)
    src = (datadir / "build_sp500_history.py").read_text()
    assert '("2026-03-20", "HHH", "FFF")' not in src


def test_apply_invalid_rename_declaration(datadir):
    diff = _blank_diff(
        undated_adds=[{"ticker": "HHH", "name": "H", "as_listed": "HHH"}],
        undated_removes=[{"ticker": "FFF"}],
    )
    with pytest.raises(U.MembershipCheckError, match="isn't among"):
        _apply(datadir, diff, rename_declarations="AAA -> HHH")
    with pytest.raises(U.MembershipCheckError, match="OLD -> NEW"):
        _apply(datadir, diff, rename_declarations="gibberish")


def test_apply_correction(datadir):
    src = (datadir / "build_sp500_history.py").read_text()
    anchor = "CHANGES: list[tuple[str, str | None, str | None]] = ["
    src = src.replace(
        anchor,
        anchor
        + '\n    ("2026-03-25", "HHH", None),  # UNCONFIRMED effective date '
        "(detected 2026-03-25; not in Wikipedia log yet)",
        1,
    )
    (datadir / "build_sp500_history.py").write_text(src)
    tickers = [t for t in fixture_tickers() if t != "Z000"] + ["HHH"]
    (datadir / "sp500_tickers.txt").write_text("# fixture\n" + "\n".join(tickers) + "\n")
    diff = _blank_diff(
        corrections=[
            {
                "added": "HHH",
                "removed": None,
                "old_date": "2026-03-25",
                "new_date": "2026-03-28",
                "confirmed": False,
            }
        ]
    )
    _apply(datadir, diff)
    src = (datadir / "build_sp500_history.py").read_text()
    assert '("2026-03-28", "HHH", None),' in src
    assert "UNCONFIRMED" not in src
    assert "HHH,2026-03-28," in _csv_rows(datadir)


def test_apply_bumps_constants(datadir):
    diff = _blank_diff(
        dated=[
            {
                "date": "2026-03-20",
                "added": "HHH",
                "removed": "FFF",
                "added_name": "H",
                "removed_name": "F",
                "as_listed_added": "HHH",
            }
        ]
    )
    _apply(datadir, diff)
    hist = (datadir / "sp500_history.py").read_text()
    assert 'RETRIEVED = "2026-04-01"' in hist
    assert 'COVERAGE_END = "2026-03-20"' in hist


def test_apply_refuses_when_nothing_to_do(datadir):
    with pytest.raises(U.MembershipCheckError, match="no changes"):
        _apply(datadir, {"up_to_date": True})


def test_apply_keeps_as_listed_spelling(datadir):
    # BRK.B-style: the seed list keeps the dot; the CSV normalizes.
    diff = _blank_diff(
        undated_adds=[
            {"ticker": "BRK-B", "name": "Berkshire", "as_listed": "BRK.B"}
        ],
        undated_removes=[{"ticker": "FFF"}],
    )
    _apply(datadir, diff)
    tickers = _tickers(datadir)
    assert "BRK.B" in tickers
    assert "BRK-B,2026-04-01," in _csv_rows(datadir)


# ---------------------------------------------------------------------------
# Status persistence + plain-English report
# ---------------------------------------------------------------------------


def test_status_roundtrip(tmp_path):
    p = tmp_path / "status.json"
    assert U.read_check_status(p)["outcome"] == "never"
    U.record_check(p, "changes_found", "3 changes", pending={"a": 1})
    st = U.read_check_status(p)
    assert st["outcome"] == "changes_found"
    assert st["detail"] == "3 changes"
    assert st["pending"] == {"a": 1}
    assert st["last_checked"] is not None


def test_status_corrupt_file_returns_default(tmp_path):
    p = tmp_path / "status.json"
    p.write_text("not json{{")
    assert U.read_check_status(p)["outcome"] == "never"


def test_describe_diff_plain_english():
    diff = _blank_diff(
        dated=[
            {
                "date": "2026-03-20",
                "added": "HHH",
                "removed": "FFF",
                "added_name": "H Corp",
                "removed_name": "F Corp",
            }
        ],
        renames=[
            {
                "old": "GGG",
                "new": "III",
                "date": None,
                "old_name": "",
                "new_name": "I Corp",
            }
        ],
    )
    lines = U.describe_diff(diff)
    text = "\n".join(lines)
    assert (
        "Added: HHH (H Corp); Removed: FFF (F Corp) (effective 2026-03-20)."
        in text
    )
    assert "Suspected rename: GGG -> III (I Corp)" in text
    assert "share effective date 2026-03-20" in text  # rename-box hint
    assert "Wikipedia" in text
