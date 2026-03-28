"""Schema migration runner.

Applies pending migrations in order on startup. Each migration is
idempotent — tables use IF NOT EXISTS. The schema_version table
tracks which migrations have been applied.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data.migrations.v001_initial import apply as apply_v001
from data.migrations.v002_profile_accumulators import apply as apply_v002

if TYPE_CHECKING:
    from data.db import Database

logger = logging.getLogger(__name__)

# Ordered list of (version, apply_function) pairs.
MIGRATIONS: list[tuple[int, callable]] = [
    (1, apply_v001),
    (2, apply_v002),
]


def run_migrations(db: Database) -> None:
    """Apply all pending migrations to bring the database up to date."""
    current_version = db.get_schema_version()
    applied = 0

    for version, apply_fn in MIGRATIONS:
        if version <= current_version:
            continue
        logger.info("Applying migration v%03d...", version)
        apply_fn(db)
        db.execute(
            "INSERT INTO schema_version (version) VALUES (?);",
            (version,),
        )
        db.connection.commit()
        applied += 1
        logger.info("Migration v%03d applied.", version)

    if applied == 0:
        logger.info("Database schema is up to date (version %d).", current_version)
    else:
        logger.info("Applied %d migration(s). Now at version %d.",
                     applied, db.get_schema_version())
