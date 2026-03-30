"""Integration tests for curriculum-driven training flow."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from ai.training.curriculum import build_curriculum, CurriculumPlan
from ai.training.scripted_opponent import ScriptedProfile
from ai.training.self_play_runner import run_self_play, _expand_curriculum
from evaluation.weakness_analyzer import WeaknessReport


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
    fast_training = dataclasses.replace(
        ai_cfg.training,
        min_matches_to_train=1,
        min_samples_to_train=5,
        retrain_after_every_n_matches=1,
        random_forest_n_estimators=10,
        random_forest_max_depth=3,
        holdout_fraction=0.0,
    )
    return dataclasses.replace(ai_cfg, training=fast_training)


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    db_path = tmp_path / "curriculum_test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    db.close()
    return db_path


# ---------------------------------------------------------------------------
# Curriculum expansion tests
# ---------------------------------------------------------------------------

class TestExpandCurriculum:

    def test_expands_to_correct_total(self):
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.AGGRESSIVE, ScriptedProfile.PATTERNED],
            match_allocation={"AGGRESSIVE": 3, "PATTERNED": 2},
            focus_targets=[],
        )
        expanded, total = _expand_curriculum(plan)
        assert total == 5
        assert len(expanded) == 5

    def test_profile_counts_match_allocation(self):
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.AGGRESSIVE, ScriptedProfile.DEFENSIVE, ScriptedProfile.MIXED],
            match_allocation={"AGGRESSIVE": 4, "DEFENSIVE": 2, "MIXED": 1},
            focus_targets=[],
        )
        expanded, total = _expand_curriculum(plan)
        assert total == 7
        counts = Counter(p.value for p in expanded)
        assert counts["AGGRESSIVE"] == 4
        assert counts["DEFENSIVE"] == 2
        assert counts["MIXED"] == 1

    def test_single_profile_allocation(self):
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.RANDOM],
            match_allocation={"RANDOM": 5},
            focus_targets=[],
        )
        expanded, total = _expand_curriculum(plan)
        assert total == 5
        assert all(p == ScriptedProfile.RANDOM for p in expanded)

    def test_expansion_is_deterministic(self):
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.AGGRESSIVE, ScriptedProfile.PATTERNED],
            match_allocation={"AGGRESSIVE": 6, "PATTERNED": 4},
            focus_targets=[],
        )
        e1, t1 = _expand_curriculum(plan)
        e2, t2 = _expand_curriculum(plan)
        assert e1 == e2
        assert t1 == t2


# ---------------------------------------------------------------------------
# Self-play with curriculum param
# ---------------------------------------------------------------------------

class TestSelfPlayWithCurriculum:

    def test_curriculum_overrides_profiles_and_n_matches(self, tmp_db_path, game_cfg, ai_cfg):
        """When curriculum is set, its allocation determines what runs."""
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.AGGRESSIVE, ScriptedProfile.DEFENSIVE],
            match_allocation={"AGGRESSIVE": 2, "DEFENSIVE": 1},
            focus_targets=["test"],
        )
        result = run_self_play(
            n_matches=99,                           # overridden by curriculum
            profiles=[ScriptedProfile.RANDOM],      # overridden by curriculum
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=100,
            curriculum=plan,
        )
        assert result.matches_run == 3  # 2 + 1 from allocation

    def test_curriculum_profile_counts_in_db(self, tmp_db_path, game_cfg, ai_cfg):
        """DB should have the correct number of matches per profile."""
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.AGGRESSIVE, ScriptedProfile.RANDOM],
            match_allocation={"AGGRESSIVE": 3, "RANDOM": 2},
            focus_targets=[],
        )
        run_self_play(
            n_matches=10,
            profiles=[ScriptedProfile.MIXED],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=100,
            curriculum=plan,
        )
        db = Database(tmp_db_path)
        db.connect()
        rows = db.fetchall("SELECT match_id FROM matches WHERE source = 'self_play';")
        db.close()
        # total = 3 + 2 = 5
        assert len(rows) == 5

    def test_curriculum_matches_tagged_self_play(self, tmp_db_path, game_cfg, ai_cfg):
        plan = CurriculumPlan(
            profiles=[ScriptedProfile.PATTERNED],
            match_allocation={"PATTERNED": 2},
            focus_targets=[],
        )
        run_self_play(
            n_matches=5,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=100,
            curriculum=plan,
        )
        db = Database(tmp_db_path)
        db.connect()
        rows = db.fetchall("SELECT source FROM matches;")
        db.close()
        for r in rows:
            assert r["source"] == "self_play"

    def test_self_play_without_curriculum_uses_params(self, tmp_db_path, game_cfg, ai_cfg):
        """Without curriculum, n_matches and profiles args are used as-is."""
        result = run_self_play(
            n_matches=3,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=100,
            curriculum=None,
        )
        assert result.matches_run == 3


# ---------------------------------------------------------------------------
# Full curriculum cycle via pipeline
# ---------------------------------------------------------------------------

class TestRunCurriculumCycle:

    def test_curriculum_cycle_completes_without_error(self, tmp_db_path, game_cfg, fast_ai_cfg):
        """run_curriculum_cycle runs end-to-end without raising."""
        from ai.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg)
        result = pipeline.run_curriculum_cycle(
            n_matches=5,
            auto_promote=False,
            seed_start=0,
            max_ticks=200,
        )
        assert result is not None

    def test_curriculum_cycle_generates_self_play_matches(self, tmp_db_path, game_cfg, fast_ai_cfg):
        """After run_curriculum_cycle, self_play matches exist in DB."""
        from ai.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg)
        pipeline.run_curriculum_cycle(
            n_matches=5,
            auto_promote=False,
            seed_start=0,
            max_ticks=200,
        )

        db = Database(tmp_db_path)
        db.connect()
        row = db.fetchone("SELECT COUNT(*) as c FROM matches WHERE source = 'self_play';")
        db.close()
        assert row["c"] >= 5

    def test_curriculum_cycle_retrain_when_enough_data(self, tmp_db_path, game_cfg, fast_ai_cfg):
        """Enough self-play events → pipeline attempts retrain."""
        from ai.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg)
        result = pipeline.run_curriculum_cycle(
            n_matches=10,
            auto_promote=False,
            seed_start=0,
            max_ticks=500,
        )

        db = Database(tmp_db_path)
        db.connect()
        event_count = db.fetchone(
            "SELECT COUNT(*) as c FROM semantic_events WHERE actor = 'PLAYER';"
        )["c"]
        db.close()

        if event_count >= fast_ai_cfg.training.min_samples_to_train:
            # retrain should have occurred
            assert result.retrain_needed is True or result.retrained is True
        # If insufficient events (very short matches with quick KOs), still no error

    def test_curriculum_cycle_no_auto_promote_reason(self, tmp_db_path, game_cfg, fast_ai_cfg):
        """auto_promote=False → promotion_reason is auto_promote_disabled or retrain_not_needed."""
        from ai.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg)
        result = pipeline.run_curriculum_cycle(
            n_matches=5,
            auto_promote=False,
            seed_start=0,
            max_ticks=200,
        )
        assert result.promotion_reason in (
            "auto_promote_disabled",
            "retrain_not_needed",
            "Insufficient samples: 0 < 5",  # edge case: no events in very short matches
        ) or "Insufficient" in result.promotion_reason

    def test_curriculum_cycle_with_regression_gate(self, tmp_db_path, tmp_path, game_cfg, fast_ai_cfg):
        """Impossible baseline → regression blocks promotion even in curriculum cycle."""
        baseline = {
            "schema_version": 1,
            "tier": "T2_FULL_ADAPTIVE",
            "match_count": 10,
            "seed_start": 0,
            "win_rate": {
                "total_matches": 10, "ai_wins": 10, "player_wins": 0, "draws": 0,
                "ai_win_rate": 1.0, "player_win_rate": 0.0, "draw_rate": 0.0,
            },
            "match_length": {
                "avg_ticks": 100.0, "min_ticks": 50, "max_ticks": 200, "median_ticks": 100.0,
            },
            "damage": {
                "avg_ai_hp_remaining": 190.0, "avg_player_hp_remaining": 0.0,
                "avg_hp_differential": 190.0,
            },
            "prediction": {
                "total_predictions": 50, "top1_correct": 50, "top1_accuracy": 1.0,
                "top2_correct": 50, "top2_accuracy": 1.0,
            },
            "planner": {
                "total_decisions_with_outcome": 50, "successful_decisions": 50,
                "overall_success_rate": 1.0, "by_mode": {}, "by_spacing": {},
            },
            "performance": {
                "avg_ticks_per_sec": 100000.0, "avg_tick_ms": 0.01,
                "p50_tick_ms": 0.01, "p95_tick_ms": 0.01, "p99_tick_ms": 0.02,
                "avg_planner_ms": 0.005, "p95_planner_ms": 0.01, "match_count": 10,
            },
            "git_sha": "", "config_hash": "", "replay_verification": None,
        }
        baseline_path = tmp_path / "baseline_t2_full_adaptive_.json"
        baseline_path.write_text(json.dumps(baseline))

        from ai.training.pipeline import TrainingPipeline
        pipeline = TrainingPipeline(tmp_db_path, game_cfg, fast_ai_cfg, baseline_path=baseline_path)
        result = pipeline.run_curriculum_cycle(
            n_matches=10,
            auto_promote=True,
            seed_start=0,
            max_ticks=500,
        )

        db = Database(tmp_db_path)
        db.connect()
        event_count = db.fetchone(
            "SELECT COUNT(*) as c FROM semantic_events WHERE actor = 'PLAYER';"
        )["c"]
        db.close()

        if event_count >= fast_ai_cfg.training.min_samples_to_train:
            # If enough data was generated, regression gate should have fired
            if result.retrained:
                assert result.promoted is False
                assert result.promotion_reason == "regression_detected"


# ---------------------------------------------------------------------------
# Weakness → curriculum → self-play data loop
# ---------------------------------------------------------------------------

class TestWeaknessToCurriculumLoop:

    def test_no_weakness_uses_mixed_profile(self, tmp_db_path, game_cfg, ai_cfg):
        """No weakness → MIXED profile → self-play runs."""
        report = WeaknessReport(
            weak_prediction_labels=[],
            weak_tactical_modes=[],
            weak_spacing_zones=[],
            high_failure_scenarios=["No significant weaknesses detected"],
        )
        plan = build_curriculum(report, total_matches=4)
        assert plan.profiles == [ScriptedProfile.MIXED]

        result = run_self_play(
            n_matches=10,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=150,
            curriculum=plan,
        )
        assert result.matches_run == 4

    def test_weak_prediction_drives_aggressive_self_play(self, tmp_db_path, game_cfg, ai_cfg):
        """Weak prediction labels → AGGRESSIVE profile selected and run."""
        report = WeaknessReport(
            weak_prediction_labels=["HEAVY_ATTACK"],
            weak_tactical_modes=[],
            weak_spacing_zones=[],
            high_failure_scenarios=["Low prediction accuracy for: HEAVY_ATTACK"],
        )
        plan = build_curriculum(report, total_matches=4)
        assert ScriptedProfile.AGGRESSIVE in plan.profiles

        result = run_self_play(
            n_matches=10,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=150,
            curriculum=plan,
        )
        assert result.matches_run == 4

    def test_weak_exploit_mode_drives_patterned_self_play(self, tmp_db_path, game_cfg, ai_cfg):
        """Weak exploitation modes → PATTERNED profile selected and run."""
        report = WeaknessReport(
            weak_prediction_labels=[],
            weak_tactical_modes=["EXPLOIT_PATTERN"],
            weak_spacing_zones=[],
            high_failure_scenarios=["Poor tactical success in modes: EXPLOIT_PATTERN"],
        )
        plan = build_curriculum(report, total_matches=3)
        assert ScriptedProfile.PATTERNED in plan.profiles

        result = run_self_play(
            n_matches=10,
            profiles=[ScriptedProfile.RANDOM],
            seed_start=0,
            db_path=tmp_db_path,
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
            max_ticks=150,
            curriculum=plan,
        )
        assert result.matches_run == 3
        # DB should have 3 self-play matches
        db = Database(tmp_db_path)
        db.connect()
        row = db.fetchone("SELECT COUNT(*) as c FROM matches WHERE source='self_play';")
        db.close()
        assert row["c"] == 3
