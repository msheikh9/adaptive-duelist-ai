"""v001: Initial schema — all core tables and indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from data.schema import ALL_INDEXES, ALL_TABLES

if TYPE_CHECKING:
    from data.db import Database


def apply(db: Database) -> None:
    for ddl in ALL_TABLES:
        db.execute(ddl)
    for idx in ALL_INDEXES:
        db.execute(idx)
    db.connection.commit()
