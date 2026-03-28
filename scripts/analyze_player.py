"""CLI: Analyze player tendencies across recent matches.

Usage: python3 scripts/analyze_player.py [--last N] [--db PATH]

Aggregates player_profiles + semantic_events into a readable
terminal report showing behavioral tendencies and AI performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analytics.pattern_miner import mine_patterns  # noqa: E402
from analytics.planner_metrics import compute_planner_metrics  # noqa: E402
from data.db import Database  # noqa: E402
from data.migrations.migration_runner import run_migrations  # noqa: E402


def format_player_report(pattern, metrics, match_count: int) -> str:
    """Format a player analysis as a human-readable terminal string."""
    lines: list[str] = []

    lines.append(f"{'=' * 60}")
    lines.append(f"  PLAYER ANALYSIS ({match_count} match(es))")
    lines.append(f"{'=' * 60}")

    ps = pattern
    lines.append(f"\n  Player: {ps.player_id}")
    lines.append(f"  Total commitments observed: {ps.total_commitments}")

    # ---- Tendencies ----
    lines.append(f"\n  --- Behavioral Tendencies ---")
    lines.append(f"  Aggression index:   {ps.aggression_index:.2f}")
    lines.append(f"  Dodge frequency:    {ps.dodge_frequency:.2f}")
    lines.append(f"  Initiative rate:    {ps.initiative_rate:.2f}")
    lines.append(f"  Movement bias:      {ps.movement_bias:+.2f}")

    # ---- Top commitments ----
    if ps.top_commitments:
        lines.append(f"\n  --- Top Actions ---")
        for action, count in ps.top_commitments:
            lines.append(f"    {action:30s} {count:5d}")

    # ---- Spacing ----
    if ps.spacing_tendencies:
        lines.append(f"\n  --- Spacing Distribution ---")
        total_sp = sum(ps.spacing_tendencies.values())
        for zone in ("CLOSE", "MID", "FAR"):
            count = ps.spacing_tendencies.get(zone, 0)
            pct = count / total_sp if total_sp > 0 else 0.0
            lines.append(f"    {zone:10s} {count:5d}  ({pct:.0%})")

    # ---- Transitions ----
    if ps.top_bigrams:
        lines.append(f"\n  --- Top Transitions ---")
        for pattern_str, count in ps.top_bigrams:
            lines.append(f"    {pattern_str:40s} {count:5d}")

    if ps.top_trigrams:
        lines.append(f"\n  --- Top Trigrams ---")
        for pattern_str, count in ps.top_trigrams:
            lines.append(f"    {pattern_str:50s} {count:5d}")

    # ---- Exploitable habits ----
    if ps.exploitable_habits:
        lines.append(f"\n  --- Exploitable Habits ---")
        for habit in ps.exploitable_habits:
            lines.append(f"    \u2022 {habit}")

    # ---- AI planner performance ----
    pm = metrics
    if pm.total_decisions > 0:
        lines.append(
            f"\n  --- AI Planner Performance "
            f"(across {match_count} match(es)) ---"
        )
        lines.append(f"  Total decisions: {pm.total_decisions}")
        lines.append(
            f"  Predictions available: {pm.prediction_available_pct:.0%}"
        )
        if pm.mode_outcome_rates:
            lines.append(f"  Mode success rates:")
            for mode, rate in sorted(pm.mode_outcome_rates.items()):
                lines.append(f"    {mode:30s} {rate:.0%}")
    else:
        lines.append(f"\n  No AI planner data available (T0 baseline).")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze player tendencies")
    parser.add_argument(
        "--last", type=int, default=10, help="Number of recent matches")
    parser.add_argument("--db", default="data/game.db", help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}")
        sys.exit(1)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    rows = db.fetchall(
        "SELECT match_id FROM matches ORDER BY started_at DESC LIMIT ?;",
        (args.last,),
    )
    match_ids = [row["match_id"] for row in rows]

    pattern = mine_patterns(db)
    metrics = compute_planner_metrics(db, match_ids)

    print(format_player_report(pattern, metrics, len(match_ids)))

    db.close()


if __name__ == "__main__":
    main()
