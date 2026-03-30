"""FastAPI dependencies and shared utilities."""

from __future__ import annotations

import enum
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Generator

from fastapi import Depends

from data.db import DEFAULT_DB_PATH, Database
from data.migrations.migration_runner import run_migrations


def get_db_path() -> Path:
    """Return the DB path from env var or default."""
    env = os.environ.get("ADAPTIVE_DUELIST_DB")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def get_db(db_path: Path = Depends(get_db_path)) -> Generator[Database, None, None]:
    """Open, migrate, yield, and close a Database connection."""
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    try:
        yield db
    finally:
        db.close()


def get_configs():
    """Load and return (game_cfg, ai_cfg)."""
    from config.config_loader import load_config
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


def safe_dict(obj) -> object:
    """Recursively convert obj to a JSON-serializable form.

    Handles: dataclasses, Enums, Paths, dicts, lists/tuples, primitives.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: safe_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_dict(i) for i in obj]
    return obj
