"""v003: Add source column to matches table.

source='human' for live game matches (default).
source='self_play' for synthetic training data.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.db import Database

_ALTERATIONS = [
    "ALTER TABLE matches ADD COLUMN source TEXT NOT NULL DEFAULT 'human';",
]

def apply(db: Database) -> None:
    for stmt in _ALTERATIONS:
        try:
            db.execute(stmt)
        except Exception:
            pass  # Column already exists — idempotent
    db.connection.commit()
