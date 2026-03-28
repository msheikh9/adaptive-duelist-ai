"""Tests for analytics/decision_explainer.py."""

from __future__ import annotations

import json
import pytest

from data.db import Database
from data.migrations.migration_runner import run_migrations
from analytics.decision_explainer import explain_decision


# ------------------------------------------------------------------ #
# Helpers — insert ai_decisions rows and fetch them back as Row       #
# ------------------------------------------------------------------ #

_INSERT = """
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


def _insert_and_fetch(db, **overrides):
    defaults = dict(
        session_id="s1", match_id="m1", tick_id=100,
        predicted_top="LIGHT_ATTACK", pred_confidence=0.85,
        pred_probs=json.dumps({"LIGHT_ATTACK": 0.85, "HEAVY_ATTACK": 0.15}),
        tactical_mode="EXPLOIT_PATTERN", ai_action="DODGE_BACKWARD",
        positioning_bias=-0.2, commit_delay=4,
        reason_tags=json.dumps(["exploit", "counter_LIGHT_ATTACK"]),
        outcome="success", outcome_tick=115,
    )
    defaults.update(overrides)
    db.execute_safe(_INSERT, tuple(defaults.values()))
    return db.fetchone(
        "SELECT * FROM ai_decisions WHERE tick_id = ?;",
        (defaults["tick_id"],),
    )


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestExplainDecision:
    def test_full_row_produces_complete_explanation(self, db):
        row = _insert_and_fetch(db)
        exp = explain_decision(row)
        assert exp.tick_id == 100
        assert exp.predicted_top == "LIGHT_ATTACK"
        assert exp.pred_confidence == pytest.approx(0.85)
        assert exp.pred_probs == {"LIGHT_ATTACK": 0.85, "HEAVY_ATTACK": 0.15}
        assert exp.tactical_mode == "EXPLOIT_PATTERN"
        assert exp.ai_action == "DODGE_BACKWARD"
        assert exp.positioning_bias == pytest.approx(-0.2)
        assert exp.commit_delay == 4
        assert exp.reason_tags == ["exploit", "counter_LIGHT_ATTACK"]
        assert exp.outcome == "success"
        assert exp.outcome_tick == 115

    def test_null_prediction_fields_handled(self, db):
        row = _insert_and_fetch(
            db, tick_id=200,
            predicted_top=None, pred_confidence=None, pred_probs=None,
        )
        exp = explain_decision(row)
        assert exp.predicted_top is None
        assert exp.pred_confidence is None
        assert exp.pred_probs is None
        assert "No prediction available" in exp.explanation

    def test_null_outcome(self, db):
        row = _insert_and_fetch(db, tick_id=300, outcome=None, outcome_tick=None)
        exp = explain_decision(row)
        assert exp.outcome is None
        assert "Outcome: unavailable" in exp.explanation

    def test_explanation_string_includes_prediction(self, db):
        row = _insert_and_fetch(db)
        exp = explain_decision(row)
        assert "LIGHT_ATTACK" in exp.explanation
        assert "85%" in exp.explanation

    def test_explanation_string_includes_mode(self, db):
        row = _insert_and_fetch(db)
        exp = explain_decision(row)
        assert "exploit pattern" in exp.explanation

    def test_explanation_string_includes_outcome(self, db):
        row = _insert_and_fetch(db)
        exp = explain_decision(row)
        assert "success" in exp.explanation

    def test_malformed_json_handled(self, db):
        row = _insert_and_fetch(
            db, tick_id=400,
            pred_probs="not-json",
            reason_tags="also-not-json",
        )
        exp = explain_decision(row)
        assert exp.pred_probs is None
        assert exp.reason_tags == []
