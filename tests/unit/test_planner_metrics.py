"""Tests for analytics/planner_metrics.py."""

from __future__ import annotations

import json
import pytest

from data.db import Database
from data.migrations.migration_runner import run_migrations
from analytics.planner_metrics import compute_planner_metrics


_INSERT_DECISION = """
INSERT INTO ai_decisions (
    session_id, match_id, tick_id, predicted_top, pred_confidence,
    pred_probs, tactical_mode, ai_action, positioning_bias,
    commit_delay, reason_tags, outcome, outcome_tick
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    d.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
        "VALUES ('m1', 's1', '2025-01-01', 42, 'h');",
    )
    yield d
    d.close()


def _insert_decision(db, tick_id, mode, action, outcome=None,
                     predicted_top="LIGHT_ATTACK", pred_confidence=0.8,
                     delay=4):
    db.execute_safe(_INSERT_DECISION, (
        "s1", "m1", tick_id, predicted_top, pred_confidence,
        json.dumps({"LIGHT_ATTACK": 0.8, "HOLD": 0.2}),
        mode, action, 0.0, delay,
        json.dumps(["test"]),
        outcome, tick_id + 10 if outcome else None,
    ))


class TestComputePlannerMetrics:
    def test_t2_decisions(self, db):
        _insert_decision(db, 10, "EXPLOIT_PATTERN", "DODGE_BACKWARD", "success")
        _insert_decision(db, 20, "EXPLOIT_PATTERN", "DODGE_BACKWARD", "failure")
        _insert_decision(db, 30, "DEFENSIVE_RESET", "MOVE_LEFT", "success")
        metrics = compute_planner_metrics(db, ["m1"])

        assert metrics.total_decisions == 3
        assert metrics.mode_distribution["EXPLOIT_PATTERN"] == 2
        assert metrics.mode_distribution["DEFENSIVE_RESET"] == 1
        assert metrics.mode_outcome_rates["EXPLOIT_PATTERN"] == pytest.approx(0.5)
        assert metrics.mode_outcome_rates["DEFENSIVE_RESET"] == pytest.approx(1.0)

    def test_t1_no_outcomes(self, db):
        _insert_decision(db, 10, "EXPLOIT_PATTERN", "DODGE_BACKWARD")
        _insert_decision(db, 20, "NEUTRAL_SPACING", "MOVE_RIGHT")
        metrics = compute_planner_metrics(db, ["m1"])

        assert metrics.total_decisions == 2
        # All outcomes are None → rates should be 0.0
        assert metrics.mode_outcome_rates["EXPLOIT_PATTERN"] == 0.0

    def test_t0_no_decisions(self, db):
        metrics = compute_planner_metrics(db, ["m1"])
        assert metrics.total_decisions == 0
        assert metrics.mode_distribution == {}

    def test_empty_match_ids(self, db):
        metrics = compute_planner_metrics(db, [])
        assert metrics.total_decisions == 0

    def test_avg_commit_delay(self, db):
        _insert_decision(db, 10, "PROBE_BEHAVIOR", "LIGHT_ATTACK", delay=10)
        _insert_decision(db, 20, "NEUTRAL_SPACING", "MOVE_RIGHT", delay=0)
        metrics = compute_planner_metrics(db, ["m1"])
        assert metrics.avg_commit_delay == pytest.approx(5.0)

    def test_prediction_available_pct(self, db):
        _insert_decision(db, 10, "EXPLOIT_PATTERN", "DODGE_BACKWARD")
        _insert_decision(db, 20, "NEUTRAL_SPACING", "MOVE_RIGHT",
                         predicted_top=None, pred_confidence=None)
        metrics = compute_planner_metrics(db, ["m1"])
        assert metrics.prediction_available_pct == pytest.approx(0.5)

    def test_hold_top_label_pct(self, db):
        _insert_decision(db, 10, "NEUTRAL_SPACING", "MOVE_RIGHT",
                         predicted_top="HOLD")
        _insert_decision(db, 20, "EXPLOIT_PATTERN", "DODGE_BACKWARD",
                         predicted_top="LIGHT_ATTACK")
        metrics = compute_planner_metrics(db, ["m1"])
        assert metrics.hold_top_label_pct == pytest.approx(0.5)

    def test_action_by_mode(self, db):
        _insert_decision(db, 10, "EXPLOIT_PATTERN", "DODGE_BACKWARD", "success")
        _insert_decision(db, 20, "EXPLOIT_PATTERN", "LIGHT_ATTACK", "failure")
        metrics = compute_planner_metrics(db, ["m1"])
        assert metrics.action_by_mode["EXPLOIT_PATTERN"]["DODGE_BACKWARD"] == 1
        assert metrics.action_by_mode["EXPLOIT_PATTERN"]["LIGHT_ATTACK"] == 1

    def test_multi_match(self, db):
        db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
            "VALUES ('m2', 's1', '2025-01-02', 43, 'h');",
        )
        _insert_decision(db, 10, "EXPLOIT_PATTERN", "DODGE_BACKWARD", "success")
        db.execute_safe(_INSERT_DECISION, (
            "s1", "m2", 10, "LIGHT_ATTACK", 0.7,
            json.dumps({"LIGHT_ATTACK": 0.7}),
            "NEUTRAL_SPACING", "MOVE_RIGHT", 0.0, 0,
            json.dumps([]), "failure", 20,
        ))
        metrics = compute_planner_metrics(db, ["m1", "m2"])
        assert metrics.total_decisions == 2
