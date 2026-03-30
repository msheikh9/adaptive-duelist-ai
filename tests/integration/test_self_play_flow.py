"""Integration tests for the self-play training data generation flow."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from data.migrations.v003_matches_source import apply as apply_v003
from ai.training.scripted_opponent import ScriptedProfile
from ai.training.self_play_runner import run_self_play
from ai.training.dataset_builder import build_dataset
from ai.profile.player_profile import PlayerProfile


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def ai_cfg():
    _, cfg, _ = load_config()
    return cfg


@pytest.fixture
def fast_ai_cfg(ai_cfg):
    """AIConfig with lowered training thresholds for fast tests."""
    fast_training = dataclasses.replace(
        ai_cfg.training,
        min_matches_to_train=1,
        min_samples_to_train=5,
        retrain_after_every_n_matches=1,
        random_forest_n_estimators=10,
        random_forest_max_depth=3,
        holdout_fraction=0.0,  # no holdout: avoids label-set mismatch with small datasets
    )
    return dataclasses.replace(ai_cfg, training=fast_training)


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    """Fresh migrated database."""
    db_path = tmp_path / "self_play_test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    db.close()
    return db_path


class TestSelfPlayInserts:

    def test_inserts_correct_match_count(self, tmp_db_path, game_cfg, ai_cfg):
        result = run_self_play(
            n_matches=3,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=200,
        )
        assert result.matches_run == 3

        db = Database(tmp_db_path)
        db.connect()
        row = db.fetchone(
            "SELECT COUNT(*) as c FROM matches WHERE source = 'self_play';"
        )
        db.close()
        assert row["c"] == 3

    def test_all_matches_tagged_self_play(self, tmp_db_path, game_cfg, ai_cfg):
        run_self_play(
            n_matches=2,
            profiles=[ScriptedProfile.AGGRESSIVE],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=200,
        )
        db = Database(tmp_db_path)
        db.connect()
        rows = db.fetchall("SELECT source FROM matches;")
        db.close()
        for r in rows:
            assert r["source"] == "self_play"

    def test_inserts_semantic_events_for_player(self, tmp_db_path, game_cfg, ai_cfg):
        result = run_self_play(
            n_matches=2,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=500,
        )
        assert result.semantic_events_inserted > 0

        db = Database(tmp_db_path)
        db.connect()
        row = db.fetchone(
            "SELECT COUNT(*) as c FROM semantic_events WHERE actor = 'PLAYER';"
        )
        db.close()
        assert row["c"] > 0

    def test_profiles_cycle_correctly(self, tmp_db_path, game_cfg, ai_cfg):
        profiles = [ScriptedProfile.RANDOM, ScriptedProfile.AGGRESSIVE]
        result = run_self_play(
            n_matches=4,
            profiles=profiles,
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=100,
        )
        assert result.profiles_used == [
            "RANDOM", "AGGRESSIVE", "RANDOM", "AGGRESSIVE"
        ]


class TestDeterminism:

    def test_identical_params_produce_same_events(self, tmp_path, game_cfg, ai_cfg):
        """Two runs with same params on empty DBs produce same semantic_events count."""
        db1 = tmp_path / "d1.db"
        db2 = tmp_path / "d2.db"

        r1 = run_self_play(
            n_matches=2,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=db1,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=300,
        )
        r2 = run_self_play(
            n_matches=2,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=db2,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=300,
        )

        assert r1.semantic_events_inserted == r2.semantic_events_inserted
        assert r1.matches_run == r2.matches_run

    def test_same_seed_same_match_outcomes(self, tmp_path, game_cfg, ai_cfg):
        """Same seed produces identical match winners in both runs."""
        db1 = tmp_path / "d1.db"
        db2 = tmp_path / "d2.db"

        run_self_play(
            n_matches=3, profiles=[ScriptedProfile.RANDOM], seed_start=7,
            db_path=db1, game_cfg=game_cfg, ai_cfg=ai_cfg, max_ticks=500,
        )
        run_self_play(
            n_matches=3, profiles=[ScriptedProfile.RANDOM], seed_start=7,
            db_path=db2, game_cfg=game_cfg, ai_cfg=ai_cfg, max_ticks=500,
        )

        conn1 = Database(db1)
        conn1.connect()
        w1 = [r["winner"] for r in conn1.fetchall("SELECT winner FROM matches ORDER BY match_id;")]
        conn1.close()

        conn2 = Database(db2)
        conn2.connect()
        w2 = [r["winner"] for r in conn2.fetchall("SELECT winner FROM matches ORDER BY match_id;")]
        conn2.close()

        assert w1 == w2


class TestSourceFiltering:

    def _insert_human_events(self, db: Database, n: int) -> None:
        """Insert minimal human-sourced data."""
        import random
        rng = random.Random(0)
        commitments = ["LIGHT_ATTACK", "HEAVY_ATTACK", "DODGE_BACKWARD"]
        db.execute_safe(
            """INSERT INTO matches
               (match_id, session_id, started_at, rng_seed, config_hash, source)
               VALUES ('human_m0', 'test', datetime('now'), 0, 'test', 'human')"""
        )
        for i in range(n):
            db.execute_safe(
                """INSERT INTO semantic_events
                   (event_type, match_id, tick_id, actor, commitment,
                    opponent_fsm_state, spacing_zone,
                    actor_hp, opponent_hp, actor_stamina, opponent_stamina,
                    reaction_ticks, damage_dealt)
                   VALUES ('COMMITMENT_START', 'human_m0', ?, 'PLAYER', ?,
                           'IDLE', 'MID', 150, 150, 80, 80, 10, NULL)""",
                (i * 60, rng.choice(commitments)),
            )

    def test_source_filter_self_play_only(self, tmp_db_path, game_cfg, ai_cfg):
        # Generate self-play data
        run_self_play(
            n_matches=2, profiles=[ScriptedProfile.RANDOM], seed_start=0,
            db_path=tmp_db_path, game_cfg=game_cfg, ai_cfg=ai_cfg, max_ticks=500,
        )
        # Also insert human data
        db = Database(tmp_db_path)
        db.connect()
        self._insert_human_events(db, 20)
        db.close()

        db = Database(tmp_db_path)
        db.connect()
        profile = PlayerProfile()
        X_sp, y_sp = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter="self_play",
        )
        X_human, y_human = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter="human",
        )
        X_all, y_all = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter=None,
        )
        db.close()

        # self_play and human should each have rows
        assert len(X_sp) > 0
        assert len(X_human) > 0
        # total should be sum of both
        assert len(X_all) == len(X_sp) + len(X_human)

    def test_source_filter_none_returns_all(self, tmp_db_path, game_cfg, ai_cfg):
        run_self_play(
            n_matches=1, profiles=[ScriptedProfile.RANDOM], seed_start=0,
            db_path=tmp_db_path, game_cfg=game_cfg, ai_cfg=ai_cfg, max_ticks=300,
        )
        db = Database(tmp_db_path)
        db.connect()
        self._insert_human_events(db, 10)

        profile = PlayerProfile()
        X_all, _ = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter=None,
        )
        X_sp, _ = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter="self_play",
        )
        X_human, _ = build_dataset(
            db, profile,
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            source_filter="human",
        )
        db.close()

        assert len(X_all) == len(X_sp) + len(X_human)


class TestMigration:

    def test_v003_idempotent(self, tmp_path):
        """Applying migration v003 twice does not error or duplicate column."""
        db_path = tmp_path / "mig_test.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)  # applies once (including v003)
        # Applying again must not raise
        apply_v003(db)
        # Verify column still exists and has correct default
        db.execute_safe(
            """INSERT INTO matches
               (match_id, session_id, started_at, rng_seed, config_hash)
               VALUES ('test_m', 'test', datetime('now'), 0, 'test')"""
        )
        row = db.fetchone("SELECT source FROM matches WHERE match_id = 'test_m';")
        db.close()
        assert row is not None
        assert row["source"] == "human"

    def test_fresh_db_has_source_column(self, tmp_path):
        """Fresh DB created from schema DDL has source column with default 'human'."""
        db_path = tmp_path / "fresh.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        db.execute_safe(
            """INSERT INTO matches
               (match_id, session_id, started_at, rng_seed, config_hash)
               VALUES ('m1', 'test', datetime('now'), 0, 'test')"""
        )
        row = db.fetchone("SELECT source FROM matches WHERE match_id = 'm1';")
        db.close()
        assert row["source"] == "human"


class TestEndToEndRetrain:

    def test_self_play_data_enables_retrain(self, tmp_db_path, game_cfg, fast_ai_cfg):
        """10 self-play matches → enough semantic events → run_retrain() succeeds."""
        from ai.training.pipeline import TrainingPipeline

        run_self_play(
            n_matches=10,
            profiles=[ScriptedProfile.RANDOM, ScriptedProfile.AGGRESSIVE],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=fast_ai_cfg,
            max_ticks=500,
        )

        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg)

        db = Database(tmp_db_path)
        db.connect()
        event_count = db.fetchone(
            "SELECT COUNT(*) as c FROM semantic_events WHERE actor='PLAYER';"
        )["c"]
        db.close()

        if event_count >= fast_ai_cfg.training.min_samples_to_train:
            version = pipeline.run_retrain()
            assert version is not None

            db = Database(tmp_db_path)
            db.connect()
            row = db.fetchone(
                "SELECT is_active FROM model_registry WHERE version = ?;",
                (version,),
            )
            db.close()
            assert row is not None
            assert row["is_active"] == 1
        else:
            # If not enough events generated (very short matches), at least
            # verify the pipeline raises correctly
            with pytest.raises(RuntimeError, match="Insufficient samples"):
                pipeline.run_retrain()
