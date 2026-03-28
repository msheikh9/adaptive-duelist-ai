"""Tests for analytics/report_builder.py."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from data.db import Database
from data.migrations.migration_runner import run_migrations
from analytics.report_builder import build_match_report, build_aggregate_report
from analytics.explanation_types import ExplainabilityReport


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
        "INSERT INTO matches (match_id, session_id, started_at, ended_at, "
        "total_ticks, winner, player_hp_final, ai_hp_final, rng_seed, config_hash) "
        "VALUES ('m1', 's1', '2025-01-01', '2025-01-01', 500, 'AI', 30, 180, 42, 'h');",
    )
    # Insert a decision
    d.execute_safe(_INSERT_DECISION, (
        "s1", "m1", 50, "LIGHT_ATTACK", 0.9,
        json.dumps({"LIGHT_ATTACK": 0.9, "HOLD": 0.1}),
        "EXPLOIT_PATTERN", "DODGE_BACKWARD", -0.2, 4,
        json.dumps(["exploit", "counter_LIGHT_ATTACK"]),
        "success", 60,
    ))
    yield d
    d.close()


class TestBuildMatchReport:
    def test_report_structure(self, db):
        report = build_match_report(db, "m1")
        assert isinstance(report, ExplainabilityReport)
        assert report.match_explanation is not None
        assert report.pattern_summary is not None
        assert report.planner_metrics is not None
        assert report.generated_at is not None
        assert report.matches_included == ["m1"]

    def test_match_explanation_populated(self, db):
        report = build_match_report(db, "m1")
        me = report.match_explanation
        assert me.match_id == "m1"
        assert me.winner == "AI"
        assert me.total_ticks == 500
        assert me.decision_count == 1
        assert "EXPLOIT_PATTERN" in me.mode_usage

    def test_nonexistent_match(self, db):
        report = build_match_report(db, "nonexistent")
        me = report.match_explanation
        assert me.winner is None
        assert me.decision_count == 0
        assert me.tier_label == "T0 (baseline)"

    def test_json_serializable(self, db):
        report = build_match_report(db, "m1")
        report_dict = asdict(report)
        output = json.dumps(report_dict, default=str)
        parsed = json.loads(output)
        assert parsed["match_explanation"]["match_id"] == "m1"


class TestBuildAggregateReport:
    def test_aggregate_single_match(self, db):
        report = build_aggregate_report(db, ["m1"])
        assert report.match_explanation is not None
        assert report.planner_metrics.total_decisions == 1
        assert report.matches_included == ["m1"]

    def test_aggregate_empty(self, db):
        report = build_aggregate_report(db, [])
        assert report.match_explanation is None
        assert report.planner_metrics.total_decisions == 0

    def test_aggregate_uses_latest_match(self, db):
        db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, "
            "rng_seed, config_hash) "
            "VALUES ('m2', 's1', '2025-01-02', 43, 'h');",
        )
        report = build_aggregate_report(db, ["m1", "m2"])
        # Latest match is m2
        assert report.match_explanation.match_id == "m2"
