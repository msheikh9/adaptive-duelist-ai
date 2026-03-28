"""v002: Add statistical accumulator columns to player_profiles.

These columns support online (Welford) computation of reaction time stats
and exact win-rate tracking across sessions without lossy aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.db import Database

# SQLite only supports ADD COLUMN when a DEFAULT is provided for NOT NULL.
_ALTERATIONS = [
    "ALTER TABLE player_profiles ADD COLUMN match_count   INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE player_profiles ADD COLUMN win_count     INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE player_profiles ADD COLUMN reaction_count INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE player_profiles ADD COLUMN reaction_M2    REAL    NOT NULL DEFAULT 0.0;",
    "ALTER TABLE player_profiles ADD COLUMN duration_sum   REAL    NOT NULL DEFAULT 0.0;",
]


def apply(db: Database) -> None:
    for stmt in _ALTERATIONS:
        try:
            db.execute(stmt)
        except Exception:
            # Column already exists (e.g., test re-runs) — safe to ignore.
            pass
    db.connection.commit()
