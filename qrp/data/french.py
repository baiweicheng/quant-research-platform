"""Fama-French factor data from Ken French's data library (roadmap P1).

What this is, in plain English: professors Eugene Fama and Kenneth French
found that most of what stocks do can be explained by a handful of broad
"tilts" — the market as a whole, small vs big companies, cheap vs expensive
stocks, profitable vs unprofitable companies, careful vs aggressive spenders
— plus momentum (recent winners vs losers). Professor French publishes these
tilts as free daily data from Dartmouth:

    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

This module downloads the two daily files (the 5 factors and the momentum
factor), parses their quirky format, and caches one tidy CSV on your
computer. The attribution code in qrp.risk.attribution then uses it to
answer "is this strategy doing something clever, or just riding well-known
tilts?"

Real-world file quirks handled here:
- The downloads are ZIP archives, each holding one CSV.
- Each CSV starts with several lines of descriptive text before the real
  header row (whose first cell is blank, e.g. ",Mkt-RF,SMB,...").
- Dates are integers like 19630701 (YYYYMMDD), values are in percent
  (0.65 means 0.65%) and are converted to plain decimals here.
- Missing values are coded -99.99 (or -999) and become NaN.
- After the data there is a blank line and trailing copyright/footer lines,
  which are ignored.

Nothing here costs money and no key is needed. If the download ever fails
(network down, site moved), every caller gets a plain-English error — the
rest of the platform works fine without factor data.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from qrp.data.store import DEFAULT_DB

# The two daily files in Ken French's data library. "2x3" is just the name
# of the portfolio-sorting recipe used to build the factors.
FIVE_FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

# Canonical column names, in the order we store them. UMD is the academic
# name for the momentum factor (Up Minus Down); the downloaded file calls
# it "Mom", which we rename.
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]

_MISSING_CODES = {-99.99, -999.0, -999.99}
_DATE_RE = re.compile(r"^\s*(\d{8})\s*,")


def _find_header(lines: list[str]) -> tuple[int, list[str]]:
    """Locate the header row: the first line whose first cell is blank.

    Returns (line_index, column_names). Raises ValueError when no header
    is found (e.g. the site changed its format).
    """
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if cells and cells[0] == "" and any(c for c in cells[1:]):
            return i, cells
    raise ValueError(
        "Could not find the factor table header in the downloaded file — "
        "Ken French's site may have changed its format."
    )


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse one downloaded Ken French CSV into a tidy DataFrame.

    Args:
        text: the decoded CSV text (one file from inside the ZIP).

    Returns:
        DataFrame indexed by date (DatetimeIndex) with one column per
        factor plus RF, in plain decimals (0.0065 = 0.65%). Missing values
        (-99.99 / -999) become NaN. The momentum column is renamed to
        "UMD" when the file calls it "Mom".
    """
    lines = text.replace("\r\n", "\n").split("\n")
    header_idx, columns = _find_header(lines)
    # Data rows: start with an 8-digit date and a comma. Parsing stops at
    # the first line that doesn't (the blank line before the copyright
    # footer), so trailing junk is never read as data.
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        match = _DATE_RE.match(line)
        if not match:
            break
        cells = [c.strip() for c in line.split(",")]
        rows.append([match.group(1), *cells[1:]])
    if not rows:
        raise ValueError(
            "The downloaded factor file contained no data rows — the "
            "site's format may have changed."
        )
    frame = pd.DataFrame(rows, columns=["date", *columns[1:]])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    for col in frame.columns[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame[col].replace(list(_MISSING_CODES), float("nan"))
        # Values arrive in percent (0.65 = 0.65%); store plain decimals.
        frame[col] = frame[col] / 100.0
    frame = frame.rename(columns={"Mom": "UMD", "MOM": "UMD"})
    return frame.set_index("date").sort_index()


def _download_zip_text(url: str, timeout: int = 60) -> str:
    """Fetch a ZIP from Ken French's library; return its CSV as text."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(
            "Could not download factor data from Ken French's data library "
            f"({url}). Check your internet connection and try again. "
            f"Details: {exc}"
        ) from exc
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = [
                n for n in archive.namelist()
                if n.lower().endswith(".csv")
            ]
            if not names:
                raise ValueError("no CSV inside")
            with archive.open(names[0]) as handle:
                return handle.read().decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, ValueError) as exc:
        raise ValueError(
            "The factor download did not arrive as expected (not a ZIP "
            "with a CSV inside) — Ken French's site may have changed. "
            f"Details: {exc}"
        ) from exc


def default_factors_path(db_path: str | Path = DEFAULT_DB) -> Path:
    """Where the tidy factor CSV is cached (next to market.db)."""
    return Path(db_path).parent / "french_factors_daily.csv"


def download_french_factors(
    db_path: str | Path = DEFAULT_DB,
) -> dict[str, object]:
    """Download, parse, merge and cache the daily Fama-French factors.

    Fetches the 5-factor file and the momentum file, keeps the dates they
    share, and writes one CSV with columns date,Mkt-RF,SMB,HML,RMW,CMA,UMD,RF
    (plain decimals) next to the market database. Re-running simply
    overwrites the cache with the latest data.

    Returns:
        {"rows": int, "first": "YYYY-MM-DD", "last": "YYYY-MM-DD",
         "path": str} describing what was stored.
    """
    five = parse_french_csv(_download_zip_text(FIVE_FACTOR_URL))
    mom = parse_french_csv(_download_zip_text(MOMENTUM_URL))
    if "UMD" not in mom.columns:
        raise ValueError(
            "The momentum file no longer has a momentum column — Ken "
            "French's site may have changed its format."
        )
    merged = five.join(mom[["UMD"]], how="inner").sort_index()
    for col in FACTOR_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = merged[FACTOR_COLUMNS].dropna(how="all")
    if merged.empty:
        raise ValueError(
            "The factor files had no overlapping dates — nothing to store."
        )
    path = default_factors_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = merged.copy()
    out.index.name = "date"
    out.to_csv(path, date_format="%Y-%m-%d")
    return {
        "rows": int(len(out)),
        "first": out.index.min().strftime("%Y-%m-%d"),
        "last": out.index.max().strftime("%Y-%m-%d"),
        "path": str(path),
    }


def load_french_factors(
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Load the cached daily factors; empty DataFrame when not downloaded.

    Returns:
        DataFrame indexed by date with columns
        Mkt-RF,SMB,HML,RMW,CMA,UMD,RF in plain decimals. Empty (with those
        columns) when the cache file does not exist yet — callers treat
        "no factor data" as "not downloaded", never an error.
    """
    path = default_factors_path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=FACTOR_COLUMNS)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    for col in FACTOR_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan")
    return frame[FACTOR_COLUMNS]


def french_factors_status(db_path: str | Path = DEFAULT_DB) -> dict[str, object]:
    """One-line status of the factor cache for the UI.

    Returns:
        {"downloaded": bool, "rows": int, "first": str|None,
         "last": str|None, "path": str}.
    """
    path = default_factors_path(db_path)
    if not path.exists():
        return {
            "downloaded": False, "rows": 0, "first": None,
            "last": None, "path": str(path),
        }
    frame = load_french_factors(db_path)
    if frame.empty:
        return {
            "downloaded": False, "rows": 0, "first": None,
            "last": None, "path": str(path),
        }
    return {
        "downloaded": True,
        "rows": int(len(frame)),
        "first": frame.index.min().strftime("%Y-%m-%d"),
        "last": frame.index.max().strftime("%Y-%m-%d"),
        "path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m qrp.data.french [--db PATH]."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the free daily Fama-French factors (5 factors + "
            "momentum) from Ken French's data library at Dartmouth and "
            "cache them next to the market database."
        )
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB), help="Market database path"
    )
    args = parser.parse_args(argv)
    try:
        info = download_french_factors(args.db)
    except ValueError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(
        f"Stored {info['rows']} days of factor data "
        f"({info['first']}..{info['last']}) in {info['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
