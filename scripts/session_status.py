#!/usr/bin/env python3
"""Show current session adaptation state: player archetype and profile stats.

Read-only. Does not modify the database or player profile.

Usage:
    python3 scripts/session_status.py [--db PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    from config.config_loader import load_config
    from data.db import DEFAULT_DB_PATH, Database
    from data.migrations.migration_runner import run_migrations
    from ai.profile.player_profile import PLAYER_ID, PlayerProfile
    from ai.profile.archetype_classifier import classify_archetype

    parser = argparse.ArgumentParser(
        description="Show session adaptation state: archetype + profile summary"
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help="Database path (default: data/game.db)",
    )
    args = parser.parse_args()

    _, ai_cfg, _ = load_config()

    db = Database(args.db)
    db.connect()
    run_migrations(db)

    row = db.fetchone(
        "SELECT * FROM player_profiles WHERE player_id = ? LIMIT 1;",
        (PLAYER_ID,),
    )
    db.close()

    if row is None:
        print("No player profile found in database.")
        print(f"  DB path: {args.db}")
        print("  Play at least one match to generate a profile.")
        sys.exit(0)

    profile = PlayerProfile.from_db_row(row)
    min_m = ai_cfg.session_adaptation.min_matches_for_archetype

    print("Session Adaptation Status")
    print("=" * 40)
    print(f"  Player ID:        {profile.player_id}")
    print(f"  Matches played:   {profile.match_count}")
    print(f"  Sessions:         {profile.session_count}")
    print(f"  Win rate vs AI:   {profile.win_rate_vs_ai:.1%}")
    print()
    print("Behavioral Profile")
    print("-" * 40)
    print(f"  Aggression index:    {profile.aggression_index:.2f}")
    print(f"  Dodge frequency:     {profile.dodge_frequency:.2f}")
    print(f"  Initiative rate:     {profile.initiative_rate:.2f}")
    print(f"  Total commitments:   {profile.total_commitments}")

    if profile.action_frequencies:
        print()
        print("Action Frequencies")
        print("-" * 40)
        total = sum(profile.action_frequencies.values()) or 1
        for action, count in sorted(
            profile.action_frequencies.items(), key=lambda x: -x[1]
        ):
            pct = count / total
            print(f"  {action:20s}  {count:4d}  ({pct:.1%})")

    print()
    print("Archetype Classification")
    print("-" * 40)
    if profile.match_count < min_m:
        print(f"  Archetype:  BALANCED (insufficient data — need {min_m} matches,"
              f" have {profile.match_count})")
    else:
        label = classify_archetype(profile)
        print(f"  Archetype:  {label.value}")

    print()
    print("Config")
    print("-" * 40)
    sa = ai_cfg.session_adaptation
    print(f"  decay_factor:              {sa.decay_factor}")
    print(f"  min_matches_for_archetype: {sa.min_matches_for_archetype}")
    print(f"  min_session_samples:       {sa.min_session_samples}")


if __name__ == "__main__":
    main()
