"""Adaptive Duelist — entry point.

Initializes configuration, database, and logging subsystems,
then launches the game engine.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import ConfigError, load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def initialize() -> tuple:
    """Load config, initialize database, run migrations.

    Returns (game_config, ai_config, display_config, database).
    """
    setup_logging()
    log = logging.getLogger("init")

    log.info("Loading configuration...")
    try:
        game_cfg, ai_cfg, display_cfg = load_config()
    except ConfigError as e:
        log.error("Configuration error: %s", e)
        sys.exit(1)
    log.info("Configuration loaded successfully.")

    log.info("Initializing database...")
    db = Database()
    db.connect()
    run_migrations(db)
    log.info("Database ready (version %d).", db.get_schema_version())

    return game_cfg, ai_cfg, display_cfg, db


def main() -> None:
    game_cfg, ai_cfg, display_cfg, db = initialize()

    log = logging.getLogger("main")
    log.info("System initialized successfully.")
    log.info(
        "Arena: %dx%d | Tick rate: %d Hz | Fighter HP: %d",
        game_cfg.arena.width,
        game_cfg.arena.height,
        game_cfg.simulation.tick_rate,
        game_cfg.fighter.max_hp,
    )

    from game.engine import Engine

    engine = Engine(game_cfg, ai_cfg, display_cfg, db)
    try:
        engine.run()
    finally:
        db.close()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
