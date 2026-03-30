"""Tests for evaluation/weakness_analyzer.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from evaluation.metrics import (
    EvaluationResult,
    WinRateMetrics,
    MatchLengthMetrics,
    DamageMetrics,
    PredictionAccuracyMetrics,
    PlannerSuccessMetrics,
    PerformanceMetrics,
)
from evaluation.weakness_analyzer import WeaknessReport, analyze_weaknesses


@pytest.fixture
def tmp_db(tmp_path) -> Database:
    db = Database(tmp_path / "test.db")
    db.connect()
    run_migrations(db)
    return db


def _make_eval_result(
    ai_win_rate: float = 0.7,
    top1_accuracy: float = 0.6,
    planner_mode_rates: dict | None = None,
    total_predictions: int = 50,
) -> EvaluationResult:
    if planner_mode_rates is None:
        planner_mode_rates = {}
    prediction = PredictionAccuracyMetrics(
        total_predictions=total_predictions,
        top1_correct=int(total_predictions * top1_accuracy),
        top1_accuracy=top1_accuracy,
        top2_correct=int(total_predictions * (top1_accuracy + 0.1)),
        top2_accuracy=top1_accuracy + 0.1,
    ) if total_predictions > 0 else None

    planner = PlannerSuccessMetrics(
        total_decisions_with_outcome=20,
        mode_success_rates=planner_mode_rates,
        overall_success_rate=0.5,
    ) if planner_mode_rates else None

    return EvaluationResult(
        tier="T2_FULL_ADAPTIVE",
        match_count=10,
        seed_start=0,
        win_rate=WinRateMetrics(
            total_matches=10,
            ai_wins=int(10 * ai_win_rate),
            player_wins=10 - int(10 * ai_win_rate),
            draws=0,
            ai_win_rate=ai_win_rate,
            player_win_rate=1 - ai_win_rate,
            draw_rate=0.0,
        ),
        match_length=MatchLengthMetrics(
            avg_ticks=3000.0, min_ticks=1000, max_ticks=5000, median_ticks=3000.0,
        ),
        damage=DamageMetrics(
            avg_ai_hp_remaining=120.0,
            avg_player_hp_remaining=80.0,
            avg_hp_differential=40.0,
        ),
        prediction=prediction,
        planner=planner,
        performance=PerformanceMetrics(
            p95_tick_ms=0.5, p95_planner_ms=0.3, avg_ticks_per_sec=2000.0,
        ),
    )


def _insert_match(db: Database, match_id: str) -> None:
    db.execute_safe(
        """INSERT INTO matches
           (match_id, session_id, started_at, rng_seed, config_hash)
           VALUES (?, 'test', datetime('now'), 0, 'test')""",
        (match_id,),
    )


def _insert_ai_decisions(
    db: Database,
    match_id: str,
    decisions: list[dict],
) -> None:
    """Insert ai_decisions rows for testing."""
    for i, d in enumerate(decisions):
        probs_json = json.dumps(d.get("pred_probs", {}))
        db.execute_safe(
            """INSERT INTO ai_decisions
               (session_id, match_id, tick_id, predicted_top, pred_confidence,
                pred_probs, tactical_mode, ai_action, outcome)
               VALUES ('test', ?, ?, ?, 0.8, ?, ?, 'LIGHT_ATTACK', ?)""",
            (
                match_id,
                i * 60,
                d.get("predicted_top"),
                probs_json,
                d.get("tactical_mode", "EXPLOIT_PATTERN"),
                d.get("outcome"),
            ),
        )


def _insert_semantic_events(
    db: Database,
    match_id: str,
    events: list[dict],
) -> None:
    for e in events:
        db.execute_safe(
            """INSERT INTO semantic_events
               (event_type, match_id, tick_id, actor, commitment,
                opponent_fsm_state, spacing_zone,
                actor_hp, opponent_hp, actor_stamina, opponent_stamina)
               VALUES ('COMMITMENT_START', ?, ?, 'PLAYER', ?, 'IDLE', ?, 150, 150, 80, 80)""",
            (
                match_id,
                e["tick_id"],
                e["commitment"],
                e.get("spacing_zone", "MID"),
            ),
        )


class TestWeaknessReportStructure:

    def test_returns_weakness_report(self, tmp_db):
        result = _make_eval_result()
        report = analyze_weaknesses(tmp_db, result)
        assert isinstance(report, WeaknessReport)
        assert isinstance(report.weak_prediction_labels, list)
        assert isinstance(report.weak_tactical_modes, list)
        assert isinstance(report.weak_spacing_zones, list)
        assert isinstance(report.high_failure_scenarios, list)
        assert len(report.high_failure_scenarios) >= 1


class TestDetectsWeakPredictionLabels:

    def test_detects_low_accuracy_labels(self, tmp_db):
        """Labels predicted often but correct rarely → appear in weak list."""
        match_id = "m0"
        _insert_match(tmp_db, match_id)

        # Insert 10 decisions predicting HEAVY_ATTACK but actual is LIGHT_ATTACK
        decisions = [
            {"predicted_top": "HEAVY_ATTACK", "pred_probs": {"HEAVY_ATTACK": 0.8, "LIGHT_ATTACK": 0.2}, "tactical_mode": "EXPLOIT_PATTERN", "outcome": "failure"}
            for _ in range(10)
        ]
        _insert_ai_decisions(tmp_db, match_id, decisions)

        # Player committed LIGHT_ATTACK after each decision
        events = [{"tick_id": i * 60 + 1, "commitment": "LIGHT_ATTACK"} for i in range(10)]
        _insert_semantic_events(tmp_db, match_id, events)

        result = _make_eval_result()
        report = analyze_weaknesses(tmp_db, result)

        # HEAVY_ATTACK was predicted 10x but only matched 0/10 → weak
        assert "HEAVY_ATTACK" in report.weak_prediction_labels

    def test_no_weak_labels_when_accurate(self, tmp_db):
        """High accuracy labels should not appear in weak list."""
        match_id = "m0"
        _insert_match(tmp_db, match_id)

        # 10 correct predictions
        decisions = [
            {"predicted_top": "LIGHT_ATTACK", "pred_probs": {"LIGHT_ATTACK": 0.9}, "tactical_mode": "EXPLOIT_PATTERN", "outcome": "success"}
            for _ in range(10)
        ]
        _insert_ai_decisions(tmp_db, match_id, decisions)
        events = [{"tick_id": i * 60 + 1, "commitment": "LIGHT_ATTACK"} for i in range(10)]
        _insert_semantic_events(tmp_db, match_id, events)

        result = _make_eval_result(top1_accuracy=0.9)
        report = analyze_weaknesses(tmp_db, result)
        assert "LIGHT_ATTACK" not in report.weak_prediction_labels


class TestDetectsWeakTacticalModes:

    def test_detects_low_success_modes(self):
        """Modes with success rate below threshold appear in weak_tactical_modes."""
        mode_rates = {
            "EXPLOIT_PATTERN": 0.20,   # very weak
            "BAIT_AND_PUNISH": 0.60,   # ok
            "DEFENSIVE_RESET": 0.15,   # weak
        }
        result = _make_eval_result(planner_mode_rates=mode_rates)

        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)

        assert "EXPLOIT_PATTERN" in report.weak_tactical_modes
        assert "DEFENSIVE_RESET" in report.weak_tactical_modes
        assert "BAIT_AND_PUNISH" not in report.weak_tactical_modes

    def test_no_weak_modes_when_all_succeed(self):
        """All strong modes → empty weak_tactical_modes."""
        mode_rates = {
            "EXPLOIT_PATTERN": 0.80,
            "BAIT_AND_PUNISH": 0.75,
        }
        result = _make_eval_result(planner_mode_rates=mode_rates)

        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)

        assert report.weak_tactical_modes == []

    def test_no_weak_modes_when_no_planner_data(self):
        result = _make_eval_result(planner_mode_rates=None)
        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)
        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)
        assert report.weak_tactical_modes == []


class TestDeterminism:

    def test_same_inputs_same_output(self, tmp_db):
        """analyze_weaknesses is deterministic for identical inputs."""
        mode_rates = {"EXPLOIT_PATTERN": 0.20, "DEFENSIVE_RESET": 0.35}
        result = _make_eval_result(planner_mode_rates=mode_rates)

        r1 = analyze_weaknesses(tmp_db, result)
        r2 = analyze_weaknesses(tmp_db, result)

        assert r1.weak_prediction_labels == r2.weak_prediction_labels
        assert r1.weak_tactical_modes == r2.weak_tactical_modes
        assert r1.weak_spacing_zones == r2.weak_spacing_zones
        assert r1.high_failure_scenarios == r2.high_failure_scenarios

    def test_weak_modes_sorted_ascending(self):
        """Weakest modes come first (deterministic ordering)."""
        mode_rates = {
            "DEFENSIVE_RESET": 0.10,
            "EXPLOIT_PATTERN": 0.25,
            "BAIT_AND_PUNISH": 0.15,
        }
        result = _make_eval_result(planner_mode_rates=mode_rates)
        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)
        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)

        # Modes should be sorted by rate ascending
        modes = report.weak_tactical_modes
        if len(modes) >= 2:
            rates = [mode_rates[m] for m in modes if m in mode_rates]
            assert rates == sorted(rates)


class TestHighFailureScenarios:

    def test_no_weakness_scenario_when_all_strong(self):
        result = _make_eval_result(top1_accuracy=0.9)
        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)
        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)
        assert any("No significant" in s for s in report.high_failure_scenarios)

    def test_scenarios_include_weak_mode_description(self):
        mode_rates = {"EXPLOIT_PATTERN": 0.15}
        result = _make_eval_result(planner_mode_rates=mode_rates)
        db_path = Path(tempfile.mktemp(suffix=".db"))
        db = Database(db_path)
        db.connect()
        run_migrations(db)
        report = analyze_weaknesses(db, result)
        db.close()
        db_path.unlink(missing_ok=True)
        assert any("mode" in s.lower() for s in report.high_failure_scenarios)
