"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations


@pytest.fixture
def config_dir():
    """Path to the real config directory."""
    return Path(__file__).parent.parent / "config"


@pytest.fixture
def game_config(config_dir):
    game_cfg, _, _ = load_config(config_dir)
    return game_cfg


@pytest.fixture
def ai_config(config_dir):
    _, ai_cfg, _ = load_config(config_dir)
    return ai_cfg


@pytest.fixture
def display_config(config_dir):
    _, _, display_cfg = load_config(config_dir)
    return display_cfg


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary SQLite database with migrations applied."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    yield db
    db.close()
