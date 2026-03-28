"""CLI: Explain a completed match.

Usage: python3 scripts/explain_match.py <match_id> [--db PATH]

Prints a readable post-match report covering prediction accuracy,
planner mode breakdown, top player patterns, and notable AI decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analytics.report_builder import build_match_report  # noqa: E402
from data.db import Database  # noqa: E402
from data.migrations.migration_runner import run_migrations  # noqa: E402


def format_report(report) -> str:
    """Format an ExplainabilityReport as a human-readable terminal string."""
    lines: list[str] = []
    me = report.match_explanation

    if me is None:
        return "No match data found."

    # ---- Header ----
    lines.append(f"{'=' * 60}")
    lines.append(f"  MATCH EXPLANATION: {me.match_id}")
    lines.append(f"{'=' * 60}")

    winner = me.winner or "UNKNOWN"
    ticks = me.total_ticks or 0
    p_hp = me.player_hp_final if me.player_hp_final is not None else "?"
    a_hp = me.ai_hp_final if me.ai_hp_final is not None else "?"
    lines.append(
        f"\n  Winner: {winner}  |  Ticks: {ticks}"
        f"  |  HP: Player={p_hp} AI={a_hp}"
    )
    lines.append(f"  Tier: {me.tier_label}")
    lines.append(f"  AI decisions: {me.decision_count}")

    # ---- Prediction accuracy ----
    lines.append(f"\n  --- Prediction Performance ---")
    if me.prediction_accuracy is not None:
        lines.append(f"  Top-1 accuracy: {me.prediction_accuracy:.1%}")
    else:
        lines.append(f"  Top-1 accuracy: unavailable")
    if me.top2_accuracy is not None:
        lines.append(f"  Top-2 accuracy: {me.top2_accuracy:.1%}")

    # ---- Mode breakdown ----
    lines.append(f"\n  --- Tactical Mode Usage ---")
    for mode, count in sorted(me.mode_usage.items(), key=lambda x: -x[1]):
        s, t = me.mode_success.get(mode, (0, 0))
        rate_str = f" ({s}/{t} success)" if t > 0 else ""
        lines.append(f"    {mode:30s} {count:4d}{rate_str}")

    # ---- Player top commitments ----
    if me.player_top_commitments:
        lines.append(f"\n  --- Player Top Commitments ---")
        for action, count in me.player_top_commitments:
            lines.append(f"    {action:30s} {count:4d}")

    # ---- Exploit targets ----
    if me.exploit_targets:
        lines.append(f"\n  --- Exploit Targets Used ---")
        for target in me.exploit_targets:
            lines.append(f"    \u2022 {target}")

    # ---- Notable decisions ----
    if me.notable_decisions:
        lines.append(f"\n  --- Notable Decisions ---")
        for dec in me.notable_decisions:
            lines.append(f"  [tick {dec.tick_id:5d}] {dec.explanation}")

    # ---- Planner quality ----
    pm = report.planner_metrics
    if pm and pm.total_decisions > 0:
        lines.append(f"\n  --- Planner Quality ---")
        lines.append(f"  Avg commit delay: {pm.avg_commit_delay:.1f} ticks")
        lines.append(
            f"  Predictions available: {pm.prediction_available_pct:.0%}"
        )
        lines.append(f"  HOLD as top label: {pm.hold_top_label_pct:.0%}")

    # ---- Pattern summary ----
    ps = report.pattern_summary
    if ps and ps.total_commitments > 0:
        lines.append(f"\n  --- Player Pattern Summary ---")
        lines.append(
            f"  Aggression: {ps.aggression_index:.2f}  |  "
            f"Dodge: {ps.dodge_frequency:.2f}  |  "
            f"Initiative: {ps.initiative_rate:.2f}"
        )
        if ps.exploitable_habits:
            lines.append(f"  Exploitable habits:")
            for habit in ps.exploitable_habits:
                lines.append(f"    \u2022 {habit}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a completed match")
    parser.add_argument("match_id", help="Match ID to explain")
    parser.add_argument("--db", default="data/game.db", help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}")
        sys.exit(1)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    report = build_match_report(db, args.match_id)
    print(format_report(report))

    db.close()


if __name__ == "__main__":
    main()
