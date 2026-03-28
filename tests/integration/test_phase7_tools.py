"""Integration tests for Phase 7 tools: bulk runner, replay audit,
DB maintenance, CLI entrypoint, and export."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def ai_cfg():
    _, cfg, _ = load_config()
    return cfg


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    d.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, ended_at, "
        "total_ticks, winner, player_hp_final, ai_hp_final, rng_seed, config_hash) "
        "VALUES ('m1', 's1', '2025-01-01', '2025-01-01', 500, 'AI', 30, 180, 42, 'h');",
    )
    yield d
    d.close()


# ------------------------------------------------------------------ #
# Bulk simulate                                                        #
# ------------------------------------------------------------------ #

class TestBulkSimulate:
    def test_baseline_batch(self, game_cfg):
        from scripts.bulk_simulate import run_baseline_match

        result = run_baseline_match(game_cfg, seed=42, max_ticks=5000)
        assert result["ticks"] > 0
        assert result["winner"] in ("PLAYER", "AI", "DRAW")

    def test_planner_batch(self, game_cfg, ai_cfg, tmp_path):
        from scripts.bulk_simulate import run_planner_match
        from ai.layers.tactical_planner import AITier

        db = Database(tmp_path / "bulk.db")
        db.connect()
        run_migrations(db)
        result = run_planner_match(
            game_cfg, ai_cfg, db, seed=42,
            tier=AITier.T2_FULL_ADAPTIVE, max_ticks=500,
            match_id="bulk-test-0",
        )
        assert result["ticks"] > 0
        assert result["winner"] in ("PLAYER", "AI", "DRAW")
        db.close()

    def test_csv_export(self, game_cfg, tmp_path):
        from scripts.bulk_simulate import run_baseline_match, RESULT_FIELDS
        import csv

        results = []
        for i in range(3):
            r = run_baseline_match(game_cfg, seed=i, max_ticks=1000)
            r.update(match_index=i, seed=i, tier="T0_BASELINE")
            results.append(r)

        csv_path = tmp_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in RESULT_FIELDS})

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["tier"] == "T0_BASELINE"

    def test_jsonl_export(self, game_cfg, tmp_path):
        from scripts.bulk_simulate import run_baseline_match

        results = []
        for i in range(2):
            r = run_baseline_match(game_cfg, seed=i, max_ticks=1000)
            r.update(match_index=i, seed=i, tier="T0_BASELINE")
            results.append(r)

        jsonl_path = tmp_path / "results.jsonl"
        with open(jsonl_path, "w") as f:
            for row in results:
                f.write(json.dumps(row, default=str) + "\n")

        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2


# ------------------------------------------------------------------ #
# Replay audit                                                         #
# ------------------------------------------------------------------ #

class TestReplayAudit:
    def test_audit_empty_dir(self, tmp_path):
        from scripts.replay_audit import audit_replay

        replay_dir = tmp_path / "replays"
        replay_dir.mkdir()
        files = list(replay_dir.glob("*.replay"))
        assert len(files) == 0

    def test_audit_corrupt_file(self, tmp_path, game_cfg):
        from scripts.replay_audit import audit_replay

        bad_file = tmp_path / "corrupt.replay"
        bad_file.write_bytes(b"not a replay")
        result = audit_replay(bad_file, game_cfg)
        assert result["status"] == "CORRUPT"
        assert "error" in result


# ------------------------------------------------------------------ #
# DB maintenance                                                       #
# ------------------------------------------------------------------ #

class TestDBMaintenance:
    def test_stats(self, db, tmp_path):
        from scripts.db_maintenance import cmd_stats

        # Should not crash
        cmd_stats(db, tmp_path / "test.db")

    def test_integrity(self, db):
        from scripts.db_maintenance import cmd_integrity

        cmd_integrity(db)

    def test_prune_dry_run(self, db):
        from scripts.db_maintenance import cmd_prune

        cmd_prune(db, "2026-01-01", dry_run=True)
        # Match should still exist
        row = db.fetchone("SELECT * FROM matches WHERE match_id = 'm1';")
        assert row is not None

    def test_prune_actual(self, db):
        from scripts.db_maintenance import cmd_prune

        cmd_prune(db, "2026-01-01", dry_run=False)
        row = db.fetchone("SELECT * FROM matches WHERE match_id = 'm1';")
        assert row is None

    def test_vacuum(self, db):
        from scripts.db_maintenance import cmd_vacuum

        cmd_vacuum(db)


# ------------------------------------------------------------------ #
# CLI entrypoint                                                       #
# ------------------------------------------------------------------ #

class TestCLI:
    def test_help_output(self):
        from scripts.cli import COMMANDS, print_help

        # Just verify it runs without error
        assert len(COMMANDS) > 0
        # Verify all referenced scripts exist
        from pathlib import Path
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        for cmd, (script, _) in COMMANDS.items():
            path = scripts_dir / script
            assert path.exists(), f"Script not found for '{cmd}': {path}"

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "sys.argv = ['cli', 'nonexistent']; "
             "from scripts.cli import main; main()"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1


# ------------------------------------------------------------------ #
# Export results                                                       #
# ------------------------------------------------------------------ #

class TestExportResults:
    def test_simulation_summary_from_db(self, db):
        from scripts.export_results import export_simulation_summary

        result = export_simulation_summary(db)
        assert result["total_matches"] == 1
        output = json.dumps(result, default=str)
        assert "m1" in output

    def test_analytics_summary_from_db(self, db):
        from scripts.export_results import export_analytics_summary

        result = export_analytics_summary(db, last_n=10)
        # Should work with just a match record, even with no decisions
        output = json.dumps(result, default=str)
        assert len(output) > 0
