"""Tests for ai/training/pipeline.py."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

from config.config_loader import load_config, TrainingConfig
from data.db import Database
from data.migrations.migration_runner import run_migrations
from ai.training.pipeline import TrainingPipeline, PipelineResult


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
    """AIConfig with lowered training thresholds for fast unit tests."""
    fast_training = dataclasses.replace(
        ai_cfg.training,
        min_matches_to_train=1,
        min_samples_to_train=5,
        retrain_after_every_n_matches=3,
        random_forest_n_estimators=10,  # fewer trees = faster train
        random_forest_max_depth=3,
        holdout_fraction=0.0,  # no holdout: avoids label-set mismatch with tiny datasets
    )
    return dataclasses.replace(ai_cfg, training=fast_training)


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Fresh empty database path."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    db.close()
    return db_path


def _insert_match(db: Database, match_id: str, source: str = "human") -> None:
    """Insert a minimal match row."""
    db.execute_safe(
        """INSERT INTO matches
           (match_id, session_id, started_at, rng_seed, config_hash, source)
           VALUES (?, 'test', datetime('now'), 42, 'test', ?)""",
        (match_id, source),
    )


def _insert_semantic_events(db: Database, match_id: str, n: int) -> None:
    """Insert n synthetic COMMITMENT_START events for dataset building."""
    import random
    rng = random.Random(42)
    commitments = ["LIGHT_ATTACK", "HEAVY_ATTACK", "DODGE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"]
    fsm_states = ["IDLE", "MOVING"]
    spacing_zones = ["CLOSE", "MID", "FAR"]

    for i in range(n):
        db.execute_safe(
            """INSERT INTO semantic_events
               (event_type, match_id, tick_id, actor, commitment,
                opponent_fsm_state, spacing_zone,
                actor_hp, opponent_hp, actor_stamina, opponent_stamina,
                reaction_ticks, damage_dealt)
               VALUES ('COMMITMENT_START', ?, ?, 'PLAYER', ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                match_id,
                i * 60,
                rng.choice(commitments),
                rng.choice(fsm_states),
                rng.choice(spacing_zones),
                rng.randint(50, 200),
                rng.randint(50, 200),
                rng.randint(30, 100),
                rng.randint(30, 100),
                rng.randint(5, 30),
            ),
        )


class TestCheckRetrainNeeded:

    def test_returns_false_with_zero_matches(self, tmp_db, game_cfg, fast_ai_cfg):
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        assert pipeline.check_retrain_needed() is False

    def test_returns_false_below_min_matches(self, tmp_db, game_cfg, fast_ai_cfg):
        # fast_ai_cfg has min_matches=1, retrain_every=3
        # With 0 matches, should be False (covered by test above)
        # With min_matches=2 and only 1 match, also False
        fast_training_2 = dataclasses.replace(
            fast_ai_cfg.training, min_matches_to_train=2, retrain_after_every_n_matches=1,
        )
        cfg2 = dataclasses.replace(fast_ai_cfg, training=fast_training_2)
        pipeline = TrainingPipeline(tmp_db, game_cfg, cfg2)

        db = Database(tmp_db)
        db.connect()
        _insert_match(db, "m1")
        db.close()

        assert pipeline.check_retrain_needed() is False

    def test_returns_false_delta_below_threshold(self, tmp_db, game_cfg, fast_ai_cfg):
        # retrain_every=3; insert 4 matches but register a model with dataset_size=3 → delta=1
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        db = Database(tmp_db)
        db.connect()
        for i in range(4):
            _insert_match(db, f"m{i}")
        # Simulate a previously trained model with dataset_size=3
        db.execute_safe(
            """INSERT INTO model_registry
               (version, model_path, model_type, is_active, dataset_size, metadata)
               VALUES ('v_old', '/tmp/m.joblib', 'random_forest', 1, 3, '{}')"""
        )
        db.close()
        # delta = 4 - 3 = 1 < retrain_every=3 → False
        assert pipeline.check_retrain_needed() is False

    def test_returns_true_when_threshold_met(self, tmp_db, game_cfg, fast_ai_cfg):
        # min_matches=1, retrain_every=3; 3 matches, no prior model → delta=3 >= 3
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        db = Database(tmp_db)
        db.connect()
        for i in range(3):
            _insert_match(db, f"m{i}")
        db.close()
        assert pipeline.check_retrain_needed() is True


class TestRunRetrain:

    def test_raises_with_insufficient_samples(self, tmp_db, game_cfg, fast_ai_cfg):
        # min_samples=5; insert only 2 events
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        db = Database(tmp_db)
        db.connect()
        _insert_match(db, "m0")
        _insert_semantic_events(db, "m0", 2)
        db.close()

        with pytest.raises(RuntimeError, match="Insufficient samples"):
            pipeline.run_retrain()

    def test_registers_version_in_model_registry(self, tmp_db, game_cfg, fast_ai_cfg):
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        db = Database(tmp_db)
        db.connect()
        _insert_match(db, "m0")
        _insert_semantic_events(db, "m0", 10)
        db.close()

        version = pipeline.run_retrain()
        assert version is not None

        db = Database(tmp_db)
        db.connect()
        row = db.fetchone(
            "SELECT version, is_active FROM model_registry WHERE version = ?;",
            (version,),
        )
        db.close()

        assert row is not None
        assert row["is_active"] == 1

    def test_deactivates_previous_versions(self, tmp_db, game_cfg, fast_ai_cfg):
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)

        # Pre-insert an "old" active model
        db = Database(tmp_db)
        db.connect()
        db.execute_safe(
            """INSERT INTO model_registry
               (version, model_path, model_type, is_active, metadata)
               VALUES ('v_old', '/tmp/old.joblib', 'random_forest', 1, '{}')"""
        )
        _insert_match(db, "m0")
        _insert_semantic_events(db, "m0", 10)
        db.close()

        new_version = pipeline.run_retrain()

        db = Database(tmp_db)
        db.connect()
        old_row = db.fetchone(
            "SELECT is_active FROM model_registry WHERE version = 'v_old';"
        )
        new_row = db.fetchone(
            "SELECT is_active FROM model_registry WHERE version = ?;",
            (new_version,),
        )
        db.close()

        assert old_row["is_active"] == 0
        assert new_row["is_active"] == 1


class TestEvaluateCandidate:

    def test_does_not_modify_main_db_model_registry(self, tmp_db, game_cfg, fast_ai_cfg):
        """evaluate_candidate uses a fresh temp eval DB; main DB is unchanged."""
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)

        # Seed a real model in main DB
        db = Database(tmp_db)
        db.connect()
        _insert_match(db, "m0")
        _insert_semantic_events(db, "m0", 10)
        db.close()
        version = pipeline.run_retrain()

        # Note: active model count before evaluation
        db = Database(tmp_db)
        db.connect()
        rows_before = db.fetchall("SELECT version FROM model_registry;")
        db.close()

        # evaluate_candidate runs; eval writes to a *temp* DB
        eval_result = pipeline.evaluate_candidate(
            version, n_matches=1, max_ticks=200
        )

        db = Database(tmp_db)
        db.connect()
        rows_after = db.fetchall("SELECT version FROM model_registry;")
        db.close()

        # Main DB should not have gained a 'candidate' row
        versions_after = {r["version"] for r in rows_after}
        assert "candidate" not in versions_after
        assert len(rows_before) == len(rows_after)
        assert eval_result is not None


class TestRunPipeline:

    def test_retrain_not_needed_returns_early(self, tmp_db, game_cfg, fast_ai_cfg):
        # Empty DB → no matches → retrain not needed
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)
        result = pipeline.run_pipeline(auto_promote=True)
        assert result.retrain_needed is False
        assert result.retrained is False
        assert result.promoted is False
        assert result.promotion_reason == "retrain_not_needed"

    def test_auto_promote_false_returns_disabled(self, tmp_db, game_cfg, fast_ai_cfg):
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg)

        db = Database(tmp_db)
        db.connect()
        for i in range(3):
            _insert_match(db, f"m{i}")
            _insert_semantic_events(db, f"m{i}", 5)
        db.close()

        result = pipeline.run_pipeline(auto_promote=False)
        assert result.retrain_needed is True
        assert result.retrained is True
        assert result.promoted is False
        assert result.promotion_reason == "auto_promote_disabled"

    def test_auto_promote_no_baseline_promotes(self, tmp_db, game_cfg, fast_ai_cfg):
        """No baseline → promote unconditionally."""
        pipeline = TrainingPipeline(tmp_db, game_cfg, fast_ai_cfg, baseline_path=None)

        db = Database(tmp_db)
        db.connect()
        for i in range(3):
            _insert_match(db, f"m{i}")
            _insert_semantic_events(db, f"m{i}", 5)
        db.close()

        result = pipeline.run_pipeline(auto_promote=True)
        assert result.retrained is True
        assert result.promoted is True
        assert result.promotion_reason == "no_baseline"
        assert result.regression_report is None

    def test_regression_detected_blocks_promotion(self, tmp_db, tmp_path, game_cfg, fast_ai_cfg):
        """Baseline with impossible thresholds → regression detected → not promoted."""
        # Build a baseline that demands 100% win rate (impossible)
        baseline = {
            "schema_version": 1,
            "tier": "T2_FULL_ADAPTIVE",
            "match_count": 10,
            "seed_start": 42,
            "win_rate": {
                "total_matches": 10,
                "ai_wins": 10,
                "player_wins": 0,
                "draws": 0,
                "ai_win_rate": 1.0,
                "player_win_rate": 0.0,
                "draw_rate": 0.0,
            },
            "match_length": {
                "avg_ticks": 100.0,
                "min_ticks": 50,
                "max_ticks": 200,
                "median_ticks": 100.0,
            },
            "damage": {
                "avg_ai_hp_remaining": 190.0,
                "avg_player_hp_remaining": 10.0,
                "avg_hp_differential": 180.0,
            },
            "prediction": {
                "total_predictions": 50,
                "top1_correct": 45,
                "top1_accuracy": 0.90,
                "top2_correct": 50,
                "top2_accuracy": 1.0,
            },
            "planner": {
                "total_decisions_with_outcome": 50,
                "successful_decisions": 45,
                "overall_success_rate": 0.90,
                "by_mode": {},
                "by_spacing": {},
            },
            "performance": {
                "avg_ticks_per_sec": 10000.0,
                "avg_tick_ms": 0.1,
                "p50_tick_ms": 0.1,
                "p95_tick_ms": 0.1,
                "p99_tick_ms": 0.2,
                "avg_planner_ms": 0.05,
                "p95_planner_ms": 0.1,
                "match_count": 10,
            },
            "git_sha": "",
            "config_hash": "",
            "replay_verification": None,
        }
        baseline_path = tmp_path / "baseline_t2_full_adaptive_.json"
        baseline_path.write_text(json.dumps(baseline))

        pipeline = TrainingPipeline(
            tmp_db, game_cfg, fast_ai_cfg, baseline_path=baseline_path
        )

        db = Database(tmp_db)
        db.connect()
        for i in range(3):
            _insert_match(db, f"m{i}")
            _insert_semantic_events(db, f"m{i}", 5)
        db.close()

        result = pipeline.run_pipeline(auto_promote=True)

        assert result.retrained is True
        assert result.promoted is False
        assert result.promotion_reason == "regression_detected"
        assert result.regression_report is not None
        assert result.regression_report.passed is False
