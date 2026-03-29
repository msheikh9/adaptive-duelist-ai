#!/usr/bin/env python3
"""Show model registry status and training data statistics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


def main() -> None:
    from config.config_loader import load_config
    from data.db import Database, DEFAULT_DB_PATH
    from data.migrations.migration_runner import run_migrations

    parser = argparse.ArgumentParser(description="Show model and training data status")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()
    db = Database(args.db)
    db.connect()
    run_migrations(db)

    # Active model
    active = db.fetchone(
        "SELECT version, eval_accuracy, eval_top2_acc, dataset_size, created_at "
        "FROM model_registry WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1;"
    )

    # All versions
    versions = db.fetchall(
        "SELECT version, eval_accuracy, is_active, created_at FROM model_registry "
        "ORDER BY created_at DESC;"
    )

    # Match counts by source
    total_row = db.fetchone("SELECT COUNT(*) as c FROM matches;")
    total = total_row["c"] if total_row else 0

    try:
        human_row = db.fetchone(
            "SELECT COUNT(*) as c FROM matches WHERE source = 'human';"
        )
        self_play_row = db.fetchone(
            "SELECT COUNT(*) as c FROM matches WHERE source = 'self_play';"
        )
        human_count = human_row["c"] if human_row else 0
        self_play_count = self_play_row["c"] if self_play_row else 0
    except Exception:
        human_count = total
        self_play_count = 0

    last_dataset_size = 0
    if versions:
        last_dataset_size = versions[0]["dataset_size"] or 0

    retrain_every = ai_cfg.training.retrain_after_every_n_matches
    delta = total - last_dataset_size

    db.close()

    print("\nModel Registry")
    print("-" * 50)
    if active:
        print(f"  Active version:    {active['version']}")
        print(f"  Holdout accuracy:  {active['eval_accuracy']:.3f}" if active['eval_accuracy'] else "  Holdout accuracy:  N/A")
        print(f"  Dataset size:      {active['dataset_size']}")
        print(f"  Created:           {active['created_at']}")
    else:
        print("  No active model.")

    print(f"\n  All versions ({len(versions)}):")
    for v in versions:
        marker = " *" if v["is_active"] else "  "
        acc = f"{v['eval_accuracy']:.3f}" if v["eval_accuracy"] else "N/A"
        print(f"   {marker} {v['version']}  acc={acc}  [{v['created_at']}]")

    print("\nTraining Data")
    print("-" * 50)
    print(f"  Total matches:     {total}")
    print(f"  Human matches:     {human_count}")
    print(f"  Self-play matches: {self_play_count}")
    print(f"  Since last train:  {delta} (retrain every {retrain_every})")


if __name__ == "__main__":
    main()
