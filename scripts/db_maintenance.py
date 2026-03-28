"""Database maintenance tools.

Commands:
  stats       — table row counts, file size, schema version
  integrity   — SQLite integrity check
  prune       — delete old matches and associated data
  vacuum      — compact the database file

Usage:
  python3 scripts/db_maintenance.py stats [--db PATH]
  python3 scripts/db_maintenance.py integrity [--db PATH]
  python3 scripts/db_maintenance.py prune --before-date DATE [--db PATH] [--dry-run]
  python3 scripts/db_maintenance.py vacuum [--db PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.db import Database
from data.migrations.migration_runner import run_migrations


def cmd_stats(db: Database, db_path: Path) -> None:
    """Print database statistics."""
    print(f"Database: {db_path}")
    print(f"File size: {db_path.stat().st_size:,} bytes")
    print(f"Schema version: {db.get_schema_version()}")
    print()

    tables = [
        "matches", "semantic_events", "player_profiles",
        "ai_decisions", "model_registry",
    ]
    for table in tables:
        if db.table_exists(table):
            row = db.fetchone(f"SELECT COUNT(*) as cnt FROM {table};")
            count = row["cnt"] if row else 0
            print(f"  {table:25s} {count:>8,} rows")
        else:
            print(f"  {table:25s} (not found)")

    # DB-level stats
    row = db.fetchone("PRAGMA page_count;")
    pages = row[0] if row else 0
    row = db.fetchone("PRAGMA page_size;")
    page_size = row[0] if row else 0
    row = db.fetchone("PRAGMA freelist_count;")
    free = row[0] if row else 0
    print(f"\n  Pages: {pages:,}  Page size: {page_size:,}B  "
          f"Free pages: {free:,}")


def cmd_integrity(db: Database) -> None:
    """Run SQLite integrity check."""
    print("Running integrity check...")
    rows = db.fetchall("PRAGMA integrity_check;")
    for row in rows:
        print(f"  {row[0]}")
    if rows and rows[0][0] == "ok":
        print("Database integrity: OK")
    else:
        print("Database integrity: ISSUES DETECTED")
        sys.exit(1)


def cmd_prune(db: Database, before_date: str, dry_run: bool) -> None:
    """Delete matches and associated data before a given date."""
    matches = db.fetchall(
        "SELECT match_id, started_at FROM matches WHERE started_at < ?;",
        (before_date,),
    )
    if not matches:
        print(f"No matches found before {before_date}.")
        return

    match_ids = [m["match_id"] for m in matches]
    print(f"Found {len(match_ids)} match(es) before {before_date}.")

    if dry_run:
        print("DRY RUN — no changes made. Matches that would be pruned:")
        for m in matches:
            print(f"  {m['match_id']}  started={m['started_at']}")
        return

    # Count affected rows
    placeholders = ",".join("?" for _ in match_ids)
    for table, col in [
        ("semantic_events", "match_id"),
        ("ai_decisions", "match_id"),
    ]:
        if db.table_exists(table):
            row = db.fetchone(
                f"SELECT COUNT(*) as cnt FROM {table} "
                f"WHERE {col} IN ({placeholders});",
                tuple(match_ids),
            )
            count = row["cnt"] if row else 0
            print(f"  Deleting {count:,} rows from {table}")
            db.execute_safe(
                f"DELETE FROM {table} WHERE {col} IN ({placeholders});",
                tuple(match_ids),
            )

    # Delete matches last (FK constraints)
    db.execute_safe(
        f"DELETE FROM matches WHERE match_id IN ({placeholders});",
        tuple(match_ids),
    )
    print(f"  Deleted {len(match_ids)} match(es) from matches")
    print("Prune complete.")


def cmd_vacuum(db: Database) -> None:
    """Compact the database by running VACUUM."""
    print("Running VACUUM...")
    db.execute("VACUUM;")
    db.connection.commit()
    print("VACUUM complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database maintenance tools")
    sub = parser.add_subparsers(dest="command")

    p_stats = sub.add_parser("stats", help="Show DB statistics")
    p_stats.add_argument("--db", default="data/game.db")

    p_int = sub.add_parser("integrity", help="Run integrity check")
    p_int.add_argument("--db", default="data/game.db")

    p_prune = sub.add_parser("prune", help="Prune old match data")
    p_prune.add_argument("--before-date", required=True,
                         help="ISO date cutoff (e.g. 2025-01-01)")
    p_prune.add_argument("--dry-run", action="store_true",
                         help="Show what would be deleted")
    p_prune.add_argument("--db", default="data/game.db")

    p_vacuum = sub.add_parser("vacuum", help="Compact database")
    p_vacuum.add_argument("--db", default="data/game.db")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists() and args.command != "stats":
        print(f"Error: database not found at {db_path}")
        sys.exit(1)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    if args.command == "stats":
        cmd_stats(db, db_path)
    elif args.command == "integrity":
        cmd_integrity(db)
    elif args.command == "prune":
        cmd_prune(db, args.before_date, args.dry_run)
    elif args.command == "vacuum":
        cmd_vacuum(db)

    db.close()


if __name__ == "__main__":
    main()
