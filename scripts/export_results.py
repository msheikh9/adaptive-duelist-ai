"""Export artifacts: benchmark reports, simulation summaries, analytics summaries.

Produces deterministic, machine-readable JSON outputs from DB contents.

Usage:
  python3 scripts/export_results.py simulation-summary --db PATH -o FILE
  python3 scripts/export_results.py analytics-summary --db PATH -o FILE [--last N]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.db import Database
from data.migrations.migration_runner import run_migrations


def export_simulation_summary(db: Database) -> dict:
    """Export a summary of all matches in the DB."""
    matches = db.fetchall(
        "SELECT * FROM matches ORDER BY started_at;",
    )

    total = len(matches)
    wins = {"PLAYER": 0, "AI": 0}
    total_ticks = 0
    finished = 0

    for m in matches:
        if m["winner"]:
            wins[m["winner"]] = wins.get(m["winner"], 0) + 1
        if m["total_ticks"]:
            total_ticks += m["total_ticks"]
            finished += 1

    event_count = 0
    row = db.fetchone("SELECT COUNT(*) as cnt FROM semantic_events;")
    if row:
        event_count = row["cnt"]

    decision_count = 0
    if db.table_exists("ai_decisions"):
        row = db.fetchone("SELECT COUNT(*) as cnt FROM ai_decisions;")
        if row:
            decision_count = row["cnt"]

    return {
        "total_matches": total,
        "finished_matches": finished,
        "total_ticks": total_ticks,
        "avg_ticks_per_match": total_ticks / finished if finished > 0 else 0,
        "wins": wins,
        "total_events": event_count,
        "total_decisions": decision_count,
        "matches": [
            {
                "match_id": m["match_id"],
                "started_at": m["started_at"],
                "total_ticks": m["total_ticks"],
                "winner": m["winner"],
                "player_hp_final": m["player_hp_final"],
                "ai_hp_final": m["ai_hp_final"],
            }
            for m in matches
        ],
    }


def export_analytics_summary(db: Database, last_n: int) -> dict:
    """Export aggregate analytics for recent matches."""
    from analytics.report_builder import build_aggregate_report

    rows = db.fetchall(
        "SELECT match_id FROM matches ORDER BY started_at DESC LIMIT ?;",
        (last_n,),
    )
    match_ids = [r["match_id"] for r in rows]

    if not match_ids:
        return {"matches_included": [], "note": "no matches found"}

    report = build_aggregate_report(db, match_ids)
    return asdict(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export artifacts to JSON")
    sub = parser.add_subparsers(dest="command")

    p_sim = sub.add_parser("simulation-summary",
                           help="Export simulation summary")
    p_sim.add_argument("--db", default="data/game.db")
    p_sim.add_argument("--output", "-o", default=None)

    p_ana = sub.add_parser("analytics-summary",
                           help="Export analytics summary")
    p_ana.add_argument("--db", default="data/game.db")
    p_ana.add_argument("--last", type=int, default=10)
    p_ana.add_argument("--output", "-o", default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    if args.command == "simulation-summary":
        result = export_simulation_summary(db)
    elif args.command == "analytics-summary":
        result = export_analytics_summary(db, args.last)
    else:
        parser.print_help()
        sys.exit(1)

    output = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Exported to {args.output}", file=sys.stderr)
    else:
        print(output)

    db.close()


if __name__ == "__main__":
    main()
