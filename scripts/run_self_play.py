#!/usr/bin/env python3
"""Generate self-play training data against scripted opponents.

Usage:
    python -m scripts.run_self_play [--matches N] [--profiles P ...] [--seed N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    from ai.training.scripted_opponent import ScriptedProfile
    from config.config_loader import load_config
    from data.db import DEFAULT_DB_PATH

    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument("--matches", type=int, default=50,
                        help="Number of matches to generate (default: 50)")
    parser.add_argument("--profiles", nargs="+",
                        choices=[p.value for p in ScriptedProfile],
                        default=[p.value for p in ScriptedProfile],
                        help="Profiles to cycle (default: all 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed start (default: 0)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="Database path")
    parser.add_argument("--max-ticks", type=int, default=5000,
                        help="Tick cap per match (default: 5000)")
    args = parser.parse_args()

    from ai.training.self_play_runner import run_self_play

    game_cfg, ai_cfg, _ = load_config()
    profiles = [ScriptedProfile(p) for p in args.profiles]

    result = run_self_play(
        n_matches=args.matches,
        profiles=profiles,
        seed_start=args.seed,
        db_path=args.db,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
        max_ticks=args.max_ticks,
    )

    print(f"\nSelf-play complete.")
    print(f"  Matches run:             {result.matches_run}")
    print(f"  Profiles used:           {', '.join(set(result.profiles_used))}")
    print(f"  Semantic events inserted: {result.semantic_events_inserted}")


if __name__ == "__main__":
    main()
