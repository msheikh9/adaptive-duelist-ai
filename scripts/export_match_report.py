"""CLI: Export a match report to JSON.

Usage: python3 scripts/export_match_report.py <match_id> [--db PATH] [-o FILE]

Exports all report dataclasses serialized cleanly to JSON,
intended for future UI/HUD integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analytics.report_builder import build_match_report  # noqa: E402
from data.db import Database  # noqa: E402
from data.migrations.migration_runner import run_migrations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export match report to JSON")
    parser.add_argument("match_id", help="Match ID to export")
    parser.add_argument("--db", default="data/game.db", help="Database path")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    report = build_match_report(db, args.match_id)
    report_dict = asdict(report)
    output = json.dumps(report_dict, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report exported to {args.output}", file=sys.stderr)
    else:
        print(output)

    db.close()


if __name__ == "__main__":
    main()
