#!/usr/bin/env python3
"""CLI script to train a sklearn prediction model from stored match data.

Usage:
    python -m scripts.train_model [--db PATH] [--version VERSION]

Reads all semantic_events from the database, builds a labeled dataset,
trains a RandomForest classifier, evaluates on a chronological holdout,
and registers the model in the model_registry table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config.config_loader import load_config
from data.db import Database, DEFAULT_DB_PATH
from data.migrations.migration_runner import run_migrations
from ai.profile.player_profile import PlayerProfile
from ai.training.dataset_builder import build_dataset
from ai.training.model_trainer import register_model, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prediction model")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="Path to SQLite database")
    parser.add_argument("--version", type=str, default=None,
                        help="Model version string (auto-generated if omitted)")
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()

    db = Database(args.db)
    db.connect()
    run_migrations(db)

    # Load existing profile for aggregate features
    from ai.profile.player_profile import PLAYER_ID
    row = db.fetchone(
        "SELECT * FROM player_profiles WHERE player_id = ? LIMIT 1;",
        (PLAYER_ID,),
    )
    profile = PlayerProfile.from_db_row(row) if row else PlayerProfile()

    # Build dataset
    X, y = build_dataset(
        db, profile,
        max_hp=game_cfg.fighter.max_hp,
        max_stamina=game_cfg.fighter.max_stamina,
        tick_rate=game_cfg.simulation.tick_rate,
        window_ticks=ai_cfg.prediction.window_ticks,
    )

    if len(X) < ai_cfg.training.min_samples_to_train:
        log.error(
            "Not enough samples: %d (need %d). Play more matches first.",
            len(X), ai_cfg.training.min_samples_to_train,
        )
        sys.exit(1)

    # Count matches in dataset
    match_count = db.fetchone("SELECT COUNT(DISTINCT match_id) as c FROM matches;")
    n_matches = match_count["c"] if match_count else 0
    if n_matches < ai_cfg.training.min_matches_to_train:
        log.error(
            "Not enough matches: %d (need %d).",
            n_matches, ai_cfg.training.min_matches_to_train,
        )
        sys.exit(1)

    # Train
    result = train_model(
        X, y,
        n_estimators=ai_cfg.training.random_forest_n_estimators,
        max_depth=ai_cfg.training.random_forest_max_depth,
        min_samples_leaf=ai_cfg.training.random_forest_min_samples_leaf,
        holdout_fraction=ai_cfg.training.holdout_fraction,
        version=args.version,
    )

    # Register
    register_model(db, result)

    # Summary
    print(f"\nTraining complete.")
    print(f"  Version:       {result.version}")
    print(f"  Model path:    {result.model_path}")
    print(f"  Dataset size:  {result.dataset_size}")
    print(f"  Train / Test:  {result.train_size} / {result.test_size}")
    print(f"  Accuracy:      {result.accuracy:.3f}")
    print(f"  Top-2 Acc:     {result.top2_accuracy:.3f}")
    print(f"  Elapsed:       {result.elapsed_seconds:.1f}s")
    print(f"  Labels:        {result.label_counts}")

    db.close()


if __name__ == "__main__":
    main()
