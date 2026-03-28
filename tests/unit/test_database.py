"""Tests for database initialization and migrations."""

from __future__ import annotations

from data.schema import SCHEMA_VERSION


class TestDatabaseInitializes:
    def test_connection_opens(self, tmp_db):
        assert tmp_db.connection is not None

    def test_schema_version_is_current(self, tmp_db):
        assert tmp_db.get_schema_version() == SCHEMA_VERSION

    def test_matches_table_exists(self, tmp_db):
        assert tmp_db.table_exists("matches")

    def test_semantic_events_table_exists(self, tmp_db):
        assert tmp_db.table_exists("semantic_events")

    def test_player_profiles_table_exists(self, tmp_db):
        assert tmp_db.table_exists("player_profiles")

    def test_ai_decisions_table_exists(self, tmp_db):
        assert tmp_db.table_exists("ai_decisions")

    def test_model_registry_table_exists(self, tmp_db):
        assert tmp_db.table_exists("model_registry")

    def test_rerunning_migrations_is_idempotent(self, tmp_db):
        from data.migrations.migration_runner import run_migrations
        run_migrations(tmp_db)
        assert tmp_db.get_schema_version() == SCHEMA_VERSION

    def test_safe_write_returns_none_on_bad_sql(self, tmp_db):
        result = tmp_db.execute_safe("INSERT INTO nonexistent VALUES (?)", (1,))
        assert result is None

    def test_safe_batch_returns_false_on_bad_sql(self, tmp_db):
        result = tmp_db.executemany_safe("INSERT INTO nonexistent VALUES (?)", [(1,)])
        assert result is False

    def test_insert_and_fetch_match(self, tmp_db):
        tmp_db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
            "VALUES (?, ?, datetime('now'), ?, ?);",
            ("m1", "s1", 42, "abc123"),
        )
        row = tmp_db.fetchone("SELECT * FROM matches WHERE match_id = ?;", ("m1",))
        assert row is not None
        assert row["match_id"] == "m1"
        assert row["rng_seed"] == 42
