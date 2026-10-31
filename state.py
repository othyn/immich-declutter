"""SQLite bookkeeping: which assets were evaluated, proposed, kept or deleted."""

from __future__ import annotations

import sqlite3
import time

# status:
#   none      evaluated, not a candidate
#   proposed  tagged
#   kept      was tagged but the user removed the tag -> never propose again
#   deleted   was tagged and got deleted (or trashed)
#   retry     evaluation failed (e.g. no thumbnail yet); try again next run
FINAL = ("none", "proposed", "kept", "deleted")


class State:
    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                category TEXT,
                reason TEXT,
                failures INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS assets_status ON assets(status);
            CREATE TABLE IF NOT EXISTS media (
                asset_id TEXT PRIMARY KEY,
                types TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        self.db.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.db.commit()

    def delete_meta(self, key: str) -> None:
        self.db.execute("DELETE FROM meta WHERE key = ?", (key,))
        self.db.commit()

    def known(self, max_failures: int, include_none: bool = True) -> set[str]:
        """Assets that need no (re-)evaluation: final status, or failed too often.
        include_none=False re-evaluates 'none' assets, e.g. after new categories were enabled."""
        final = FINAL if include_none else tuple(s for s in FINAL if s != "none")
        q = (
            f"SELECT asset_id FROM assets WHERE status IN ({','.join('?' * len(final))}) "
            "OR (status = 'retry' AND failures >= ?)"
        )
        return {r[0] for r in self.db.execute(q, (*final, max_failures))}

    def retry_ids(self, max_failures: int) -> list[str]:
        q = "SELECT asset_id FROM assets WHERE status = 'retry' AND failures < ?"
        return [r[0] for r in self.db.execute(q, (max_failures,))]

    def proposed(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT asset_id, category FROM assets WHERE status = 'proposed'"))

    def set(self, asset_id: str, status: str, category: str | None = None, reason: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO assets (asset_id, status, category, reason, failures, updated_at) VALUES (?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(asset_id) DO UPDATE SET status = excluded.status, "
            "category = COALESCE(excluded.category, assets.category), "
            "reason = COALESCE(excluded.reason, assets.reason), updated_at = excluded.updated_at",
            (asset_id, status, category, reason, time.time()),
        )

    def fail(self, asset_id: str, error: str) -> None:
        self.db.execute(
            "INSERT INTO assets (asset_id, status, reason, failures, updated_at) VALUES (?, 'retry', ?, 1, ?) "
            "ON CONFLICT(asset_id) DO UPDATE SET status = 'retry', reason = excluded.reason, "
            "failures = assets.failures + 1, updated_at = excluded.updated_at",
            (asset_id, error[:200], time.time()),
        )

    def media_known(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT asset_id FROM media WHERE types != '?'")}

    def media_retry_ids(self) -> list[str]:
        """Originals exiftool could not read last time ('?'), e.g. a file still being moved."""
        return [r[0] for r in self.db.execute("SELECT asset_id FROM media WHERE types = '?'")]

    def set_media(self, asset_id: str, types: set[str] | None) -> None:
        """Media tags are facts about the file: read once, never reconciled."""
        self.db.execute(
            "INSERT INTO media (asset_id, types, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(asset_id) DO UPDATE SET types = excluded.types, updated_at = excluded.updated_at",
            (asset_id, "?" if types is None else ",".join(sorted(types)), time.time()),
        )

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM assets GROUP BY status"))
