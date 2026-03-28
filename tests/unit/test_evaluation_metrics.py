"""Tests for evaluation/metrics.py — canonical metric functions."""

from __future__ import annotations

import json
import pytest

from data.db import Database
from data.migrations.migration_runner import run_migrations
from evaluation.metrics import (
    DamageMetrics,
    MatchLengthMetrics,
    PerformanceMetrics,
    ReplayVerificationMetrics,
    WinRateMetrics,
    compute_damage,
    compute_match_length,
    compute_performance,
    compute_planner_success,
    compute_prediction_accuracy,
    compute_win_rate,
)


def _make_results(outcomes: list[tuple[str, int, int, int]]) -> list[dict]:
    """Build result dicts from (winner, ticks, player_hp, ai_hp) tuples."""
    return [
        {
            "winner": w, "ticks": t, "player_hp": php, "ai_hp": ahp,
            "elapsed_s": t * 0.001,
        }
        for w, t, php, ahp in outcomes
    ]


class TestComputeWinRate:
    def test_all_ai_wins(self):
        results = _make_results([("AI", 100, 0, 150)] * 5)
        wr = compute_win_rate(results)
        assert wr.total_matches == 5
        assert wr.ai_wins == 5
        assert wr.player_wins == 0
        assert wr.draws == 0
        assert wr.ai_win_rate == 1.0

    def test_mixed_outcomes(self):
        results = _make_results([
            ("AI", 100, 0, 100),
            ("PLAYER", 200, 100, 0),
            ("DRAW", 300, 50, 50),
        ])
        wr = compute_win_rate(results)
        assert wr.ai_wins == 1
        assert wr.player_wins == 1
        assert wr.draws == 1
        assert abs(wr.ai_win_rate - 1 / 3) < 0.01

    def test_empty(self):
        wr = compute_win_rate([])
        assert wr.total_matches == 0
        assert wr.ai_win_rate == 0.0


class TestComputeMatchLength:
    def test_basic(self):
        results = _make_results([
            ("AI", 100, 0, 100),
            ("AI", 200, 0, 100),
            ("AI", 300, 0, 100),
        ])
        ml = compute_match_length(results)
        assert ml.avg_ticks == 200.0
        assert ml.min_ticks == 100
        assert ml.max_ticks == 300
        assert ml.median_ticks == 200.0

    def test_single_match(self):
        results = _make_results([("AI", 500, 0, 100)])
        ml = compute_match_length(results)
        assert ml.avg_ticks == 500.0
        assert ml.min_ticks == 500
        assert ml.max_ticks == 500


class TestComputeDamage:
    def test_ai_advantage(self):
        results = _make_results([
            ("AI", 100, 0, 150),
            ("AI", 100, 50, 100),
        ])
        dm = compute_damage(results)
        assert dm.avg_ai_hp_remaining == 125.0
        assert dm.avg_player_hp_remaining == 25.0
        assert dm.avg_hp_differential == 100.0  # AI advantage

    def test_player_advantage(self):
        results = _make_results([
            ("PLAYER", 100, 150, 0),
        ])
        dm = compute_damage(results)
        assert dm.avg_hp_differential == -150.0


class TestComputePerformance:
    def test_basic_throughput(self):
        results = [
            {"ticks": 1000, "elapsed_s": 1.0, "winner": "AI", "player_hp": 0, "ai_hp": 100},
            {"ticks": 2000, "elapsed_s": 1.0, "winner": "AI", "player_hp": 0, "ai_hp": 100},
        ]
        pf = compute_performance(results)
        assert pf.avg_ticks_per_sec == 1500.0
        assert pf.p95_tick_ms > 0

    def test_single_match(self):
        results = [
            {"ticks": 500, "elapsed_s": 0.5, "winner": "AI", "player_hp": 0, "ai_hp": 100},
        ]
        pf = compute_performance(results)
        assert pf.avg_ticks_per_sec == 1000.0

    def test_real_tick_latencies(self):
        results = [
            {
                "ticks": 100, "elapsed_s": 0.1, "winner": "AI",
                "player_hp": 0, "ai_hp": 100,
                "tick_latencies_ms": [0.5] * 95 + [2.0] * 5,
            },
        ]
        pf = compute_performance(results)
        assert pf.p95_tick_ms == 2.0

    def test_real_planner_latencies(self):
        results = [
            {
                "ticks": 100, "elapsed_s": 0.1, "winner": "AI",
                "player_hp": 0, "ai_hp": 100,
                "tick_latencies_ms": [0.5] * 100,
                "planner_latencies_ms": [0.3] * 90 + [1.5] * 10,
            },
        ]
        pf = compute_performance(results)
        assert pf.p95_planner_ms == 1.5


# ------------------------------------------------------------------ #
# DB-backed metrics                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    # Insert a match record
    d.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
        "config_hash) VALUES ('m1', 's1', '2025-01-01', 42, 'h');",
    )
    # Insert player commitment events
    d.execute_safe(
        "INSERT INTO semantic_events (event_type, match_id, tick_id, actor, "
        "commitment, actor_hp, opponent_hp, actor_stamina, opponent_stamina) VALUES "
        "('commitment', 'm1', 10, 'player', 'LIGHT_ATTACK', 200, 200, 100, 100);"
    )
    d.execute_safe(
        "INSERT INTO semantic_events (event_type, match_id, tick_id, actor, "
        "commitment, actor_hp, opponent_hp, actor_stamina, opponent_stamina) VALUES "
        "('commitment', 'm1', 30, 'player', 'HEAVY_ATTACK', 180, 200, 80, 100);"
    )
    d.execute_safe(
        "INSERT INTO semantic_events (event_type, match_id, tick_id, actor, "
        "commitment, actor_hp, opponent_hp, actor_stamina, opponent_stamina) VALUES "
        "('commitment', 'm1', 50, 'player', 'DODGE_BACKWARD', 160, 180, 60, 90);"
    )
    # Insert ai_decisions with predictions
    probs1 = json.dumps({"LIGHT_ATTACK": 0.6, "HEAVY_ATTACK": 0.3, "DODGE_BACKWARD": 0.1})
    probs2 = json.dumps({"HEAVY_ATTACK": 0.5, "LIGHT_ATTACK": 0.3, "DODGE_BACKWARD": 0.2})
    probs3 = json.dumps({"LIGHT_ATTACK": 0.5, "HEAVY_ATTACK": 0.4, "DODGE_BACKWARD": 0.1})
    d.execute_safe(
        "INSERT INTO ai_decisions (session_id, match_id, tick_id, predicted_top, "
        "pred_confidence, pred_probs, tactical_mode, ai_action, outcome) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?);",
        ("s1", "m1", 5, "LIGHT_ATTACK", 0.6, probs1,
         "EXPLOIT_PATTERN", "LIGHT_ATTACK", "success"),
    )
    d.execute_safe(
        "INSERT INTO ai_decisions (session_id, match_id, tick_id, predicted_top, "
        "pred_confidence, pred_probs, tactical_mode, ai_action, outcome) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?);",
        ("s1", "m1", 25, "HEAVY_ATTACK", 0.5, probs2,
         "EXPLOIT_PATTERN", "HEAVY_ATTACK", "success"),
    )
    d.execute_safe(
        "INSERT INTO ai_decisions (session_id, match_id, tick_id, predicted_top, "
        "pred_confidence, pred_probs, tactical_mode, ai_action, outcome) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?);",
        ("s1", "m1", 45, "LIGHT_ATTACK", 0.5, probs3,
         "DEFENSIVE_RESET", "DODGE_BACKWARD", "failure"),
    )
    yield d
    d.close()


class TestComputePredictionAccuracy:
    def test_basic_accuracy(self, db):
        result = compute_prediction_accuracy(db, ["m1"])
        assert result is not None
        assert result.total_predictions == 3
        # tick 5 predicted LIGHT_ATTACK, actual at tick 10 = LIGHT_ATTACK -> correct
        # tick 25 predicted HEAVY_ATTACK, actual at tick 30 = HEAVY_ATTACK -> correct
        # tick 45 predicted LIGHT_ATTACK, actual at tick 50 = DODGE_BACKWARD -> incorrect
        assert result.top1_correct == 2
        assert abs(result.top1_accuracy - 2 / 3) < 0.01

    def test_top2_accuracy(self, db):
        result = compute_prediction_accuracy(db, ["m1"])
        assert result is not None
        # tick 45: top-2 from probs = [LIGHT_ATTACK, HEAVY_ATTACK], actual = DODGE_BACKWARD -> miss
        assert result.top2_correct == 2

    def test_empty_match_ids(self, db):
        result = compute_prediction_accuracy(db, [])
        assert result is None

    def test_no_matching_data(self, db):
        result = compute_prediction_accuracy(db, ["nonexistent"])
        assert result is not None
        assert result.total_predictions == 0


class TestComputePlannerSuccess:
    def test_basic_success(self, db):
        result = compute_planner_success(db, ["m1"])
        assert result is not None
        assert result.total_decisions_with_outcome == 3
        assert result.overall_success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_mode_breakdown(self, db):
        result = compute_planner_success(db, ["m1"])
        assert result is not None
        assert "EXPLOIT_PATTERN" in result.mode_success_rates
        assert result.mode_success_rates["EXPLOIT_PATTERN"] == 1.0  # 2/2
        assert result.mode_success_rates["DEFENSIVE_RESET"] == 0.0  # 0/1

    def test_empty_match_ids(self, db):
        result = compute_planner_success(db, [])
        assert result is None

    def test_no_matching_data(self, db):
        result = compute_planner_success(db, ["nonexistent"])
        assert result is not None
        assert result.total_decisions_with_outcome == 0
