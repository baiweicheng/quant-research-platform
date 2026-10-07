"""Timestamped backup and restore for the local market database (P5).

The database (``data/market.db``) holds everything regenerable-but-slow:
downloaded prices, fundamentals, signals, and the paper-trading account.
This module copies it to ``data/backups/`` (git-ignored), verifies each
copy with SQLite's own integrity check, restores from a chosen backup,
and prunes old backups so the folder doesn't grow forever.

Retention: the newest ``BACKUP_KEEP_DEFAULT`` (5) backups are kept.
Restoring never destroys data silently: the current database is copied
aside as ``pre-restore-<timestamp>.db`` first, and a corrupt or missing
backup refuses with a plain-English message instead of a traceback.

CLI:
    python -m qrp.data.backup --backup            # make a backup now
    python -m qrp.data.backup --list              # list existing backups
    python -m qrp.data.backup --verify FILE       # integrity-check a backup
    python -m qrp.data.backup --restore FILE      # restore (with safety copy)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qrp.data.store import DEFAULT_DB

#: How many of the newest backups to keep. Five is enough rope: a bad week
#: of downloads is recoverable, and each backup is one full copy of the
#: database, so the folder stays small.
BACKUP_KEEP_DEFAULT = 5

BACKUP_PREFIX = "market-"
SAFETY_PREFIX = "pre-restore-"


def backup_dir(db_path: str | Path = DEFAULT_DB) -> Path:
    """``data/backups/`` next to the database (created on demand)."""
    return Path(db_path).resolve().parent / "backups"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _fmt_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f} KB"
    return f"{n} bytes"


@dataclass
class BackupInfo:
    name: str
    path: Path
    size_bytes: int
    created_at: str  # ISO timestamp (UTC)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "size": _fmt_size(self.size_bytes),
            "created_at": self.created_at,
        }


def list_backups(db_path: str | Path = DEFAULT_DB) -> list[BackupInfo]:
    """Backups on disk, newest first. Never raises on a missing folder."""
    return list_backups_for_folder(backup_dir(db_path))


def verify_backup(path: str | Path) -> tuple[bool, str]:
    """SQLite's own integrity check plus a schema sanity look.

    Returns (ok, plain-English message). ``ok`` is False for missing
    files, non-database files, and databases that fail the check.
    """
    path = Path(path)
    if not path.exists():
        return False, (
            f"There's no backup file at {path.name} — it may have been "
            "deleted. Pick one from the list instead."
        )
    try:
        conn = sqlite3.connect(path)
        try:
            (result,) = conn.execute("PRAGMA integrity_check").fetchone()
            if str(result).lower() != "ok":
                return False, (
                    f"{path.name} failed SQLite's integrity check "
                    f"({result}) — the file looks corrupted, so it won't "
                    "be used. Pick another backup."
                )
            has_prices = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table'"
                    " AND name='daily_bars'"
                ).fetchone()
                is not None
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False, (
            f"{path.name} isn't a readable database file — it looks "
            "corrupted or truncated. Nothing was changed; pick another "
            "backup."
        )
    if not has_prices:
        return True, (
            f"{path.name} passed the integrity check, but it has no price "
            "data in it — fine as a safety copy, odd as a restore target."
        )
    return True, f"{path.name} passed the integrity check."


def _unique_name(folder: Path, stem: str) -> Path:
    """A non-colliding backup path (two backups in one second get -2, -3)."""
    candidate = folder / f"{stem}.db"
    i = 2
    while candidate.exists():
        candidate = folder / f"{stem}-{i}.db"
        i += 1
    return candidate


def _checkpoint(db_path: Path) -> None:
    """Flush WAL content into the main database file.

    The platform opens every connection in WAL mode, so recent writes may
    live only in the ``-wal`` sidecar. A file copy without a checkpoint
    would silently miss them — and a restore over a live database would
    see the stale WAL resurrect "deleted" rows. Checkpoint first, then the
    plain file copy is a complete snapshot.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        try:
            (db_path.parent / (db_path.name + suffix)).unlink()
        except OSError:
            pass


def _prune(folder: Path, keep: int) -> int:
    """Delete backups beyond the newest ``keep``. Returns number removed."""
    infos = list_backups_for_folder(folder)
    removed = 0
    for info in infos[keep:]:
        try:
            info.path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def list_backups_for_folder(folder: Path) -> list[BackupInfo]:
    infos = []
    if not folder.exists():
        return []
    # Sort by filename, not mtime: shutil.copy2 preserves the source's
    # mtime, so every backup would otherwise tie. The timestamp in the
    # name sorts lexicographically, newest last.
    for p in sorted(folder.glob("*.db"), key=lambda q: q.name):
        try:
            st = p.stat()
            infos.append(
                BackupInfo(
                    name=p.name,
                    path=p,
                    size_bytes=st.st_size,
                    created_at=datetime.fromtimestamp(
                        st.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                )
            )
        except OSError:
            continue
    return infos[::-1]


@dataclass
class BackupResult:
    ok: bool
    message: str
    path: Path | None = None
    pruned: int = 0


def create_backup(
    db_path: str | Path = DEFAULT_DB,
    keep: int = BACKUP_KEEP_DEFAULT,
) -> BackupResult:
    """Copy the database to a timestamped backup; verify; prune old ones."""
    db_path = Path(db_path)
    if not db_path.exists():
        return BackupResult(
            ok=False,
            message=(
                "There's no database to back up yet — download data on the "
                "Market data page first."
            ),
        )
    folder = backup_dir(db_path)
    folder.mkdir(parents=True, exist_ok=True)
    _checkpoint(db_path)  # WAL content must be in the file we copy
    dest = _unique_name(folder, f"{BACKUP_PREFIX}{_timestamp()}")
    try:
        shutil.copy2(db_path, dest)
    except OSError as exc:
        return BackupResult(
            ok=False,
            message=f"Couldn't write the backup ({exc}). Disk full?",
        )
    ok, verify_msg = verify_backup(dest)
    if not ok:
        try:
            dest.unlink()
        except OSError:
            pass
        return BackupResult(
            ok=False,
            message=(
                "The backup copy failed its own integrity check, so it was "
                "deleted rather than kept. Nothing was lost — your live "
                f"database is untouched. ({verify_msg})"
            ),
        )
    pruned = _prune(folder, keep)
    size = _fmt_size(dest.stat().st_size)
    message = f"Backed up to {dest.name} ({size}). "
    if pruned:
        message += (
            f"Keeping the newest {keep} backups — {pruned} older "
            f"{'one was' if pruned == 1 else 'ones were'} removed."
        )
    else:
        message += f"Keeping the newest {keep} backups."
    return BackupResult(ok=True, message=message, path=dest, pruned=pruned)


def restore_backup(
    backup_path: str | Path,
    db_path: str | Path = DEFAULT_DB,
) -> BackupResult:
    """Restore the database from a backup, with a safety copy first.

    The current database (if any) is copied aside as
    ``pre-restore-<timestamp>.db`` before anything is overwritten, so a
    mistaken restore is itself recoverable. A corrupt or missing backup
    refuses with a plain-English message and changes nothing.
    """
    db_path = Path(db_path)
    backup_path = Path(backup_path)
    if not backup_path.is_absolute():
        backup_path = backup_dir(db_path) / backup_path.name
    ok, verify_msg = verify_backup(backup_path)
    if not ok:
        return BackupResult(ok=False, message=verify_msg)

    folder = backup_dir(db_path)
    folder.mkdir(parents=True, exist_ok=True)
    safety_name = None
    if db_path.exists():
        _checkpoint(db_path)  # safety copy must hold the latest writes
        safety_name = f"{SAFETY_PREFIX}{_timestamp()}.db"
        try:
            shutil.copy2(db_path, folder / safety_name)
        except OSError as exc:
            return BackupResult(
                ok=False,
                message=(
                    "Couldn't save a safety copy of your current database "
                    f"({exc}), so the restore was aborted before touching "
                    "anything."
                ),
            )
    _checkpoint(backup_path)  # the backup itself may carry a WAL sidecar
    try:
        shutil.copy2(backup_path, db_path)
    except OSError as exc:
        return BackupResult(
            ok=False,
            message=f"Couldn't write the database file ({exc}).",
        )
    _checkpoint(db_path)  # drop any stale WAL from before the restore
    ok, verify_msg = verify_backup(db_path)
    if not ok:
        return BackupResult(
            ok=False,
            message=(
                "The restore wrote the file but it failed the integrity "
                "check afterwards — something is wrong with the disk or the "
                "copy. Your safety copy is "
                f"{safety_name or 'unavailable'} in the backups folder."
            ),
        )
    message = f"Restored from {backup_path.name}. "
    if safety_name:
        message += (
            f"Your previous database was saved first as {safety_name}, "
            "just in case."
        )
    else:
        message += "There was no previous database to save."
    return BackupResult(ok=True, message=message, path=db_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Back up and restore the local market database."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--backup", action="store_true", help="Make a backup now.")
    g.add_argument("--list", action="store_true", help="List backups.")
    g.add_argument(
        "--verify", metavar="FILE", help="Integrity-check a backup file."
    )
    g.add_argument(
        "--restore", metavar="FILE", help="Restore from a backup file."
    )
    p.add_argument(
        "--keep",
        type=int,
        default=BACKUP_KEEP_DEFAULT,
        help=f"Backups to keep (default: {BACKUP_KEEP_DEFAULT}).",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backup:
        result = create_backup(args.db, keep=args.keep)
        print(result.message)
        return 0 if result.ok else 1
    if args.list:
        infos = list_backups(args.db)
        if not infos:
            print("No backups yet — run with --backup to make one.")
            return 0
        for info in infos:
            print(f"{info.name}  {info.created_at}  {_fmt_size(info.size_bytes)}")
        return 0
    if args.verify:
        ok, message = verify_backup(args.verify)
        print(message)
        return 0 if ok else 1
    if args.restore:
        result = restore_backup(args.restore, args.db)
        print(result.message)
        return 0 if result.ok else 1
    return 2  # unreachable: argparse enforces one action


if __name__ == "__main__":
    raise SystemExit(main())
