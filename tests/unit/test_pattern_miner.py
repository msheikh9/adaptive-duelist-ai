"""Tests for analytics/pattern_miner.py."""

from __future__ import annotations

import json
import pytest

from data.db import Database
from data.migrations.migration_runner import run_migrations
from analytics.pattern_miner import mine_patterns, _flatten_ngrams


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    yield d
    d.close()


def _insert_profile(db, **overrides):
    defaults = dict(
        player_id="player_1",
        session_count=5,
        total_ticks_observed=3000,
        action_frequencies=json.dumps({
            "LIGHT_ATTACK": 120,
            "HEAVY_ATTACK": 40,
            "DODGE_BACKWARD": 30,
            "MOVE_RIGHT": 60,
            "MOVE_LEFT": 50,
        }),
        recent_action_frequencies=json.dumps({}),
        bigrams=json.dumps({
            "LIGHT_ATTACK": {"LIGHT_ATTACK": 35, "DODGE_BACKWARD": 20, "MOVE_RIGHT": 10},
            "HEAVY_ATTACK": {"MOVE_LEFT": 15, "DODGE_BACKWARD": 10},
        }),
        trigrams=json.dumps({
            "LIGHT_ATTACK,LIGHT_ATTACK": {"HEAVY_ATTACK": 8, "DODGE_BACKWARD": 5},
        }),
        spacing_distribution=json.dumps({"CLOSE": 100, "MID": 80, "FAR": 20}),
        movement_direction_bias=0.15,
        dodge_left_pct=0.4,
        dodge_right_pct=0.6,
        dodge_frequency=0.10,
        aggression_index=0.53,
        initiative_rate=0.45,
        punish_conversion_rate=0.20,
        low_hp_action_dist=json.dumps({}),
        cornered_action_dist=json.dumps({}),
        combo_sequences=json.dumps([]),
        avg_reaction_time_ms=250.0,
        reaction_time_stddev=40.0,
        win_rate_vs_ai=0.40,
        avg_match_duration=120.0,
    )
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    db.execute_safe(
        f"INSERT INTO player_profiles ({cols}) VALUES ({placeholders});",
        tuple(defaults.values()),
    )


class TestMinePatterns:
    def test_populated_profile(self, db):
        _insert_profile(db)
        summary = mine_patterns(db)
        assert summary.player_id == "player_1"
        assert summary.total_commitments == 300
        assert summary.aggression_index == pytest.approx(0.53)
        assert summary.dodge_frequency == pytest.approx(0.10)
        assert summary.initiative_rate == pytest.approx(0.45)
        assert summary.movement_bias == pytest.approx(0.15)

    def test_top_commitments_sorted(self, db):
        _insert_profile(db)
        summary = mine_patterns(db)
        actions = [a for a, _ in summary.top_commitments]
        assert actions[0] == "LIGHT_ATTACK"

    def test_spacing_tendencies(self, db):
        _insert_profile(db)
        summary = mine_patterns(db)
        assert summary.spacing_tendencies["CLOSE"] == 100
        assert summary.spacing_tendencies["MID"] == 80

    def test_bigrams_flattened(self, db):
        _insert_profile(db)
        summary = mine_patterns(db)
        assert len(summary.top_bigrams) > 0
        # Top bigram should be LIGHT_ATTACK → LIGHT_ATTACK (35)
        assert "LIGHT_ATTACK" in summary.top_bigrams[0][0]
        assert summary.top_bigrams[0][1] == 35

    def test_trigrams_flattened(self, db):
        _insert_profile(db)
        summary = mine_patterns(db)
        assert len(summary.top_trigrams) > 0

    def test_missing_profile_returns_defaults(self, db):
        summary = mine_patterns(db)
        assert summary.total_commitments == 0
        assert summary.aggression_index == 0.0
        assert summary.exploitable_habits == []

    def test_sparse_profile(self, db):
        _insert_profile(
            db,
            action_frequencies=json.dumps({}),
            bigrams=json.dumps({}),
            trigrams=json.dumps({}),
            spacing_distribution=json.dumps({}),
        )
        summary = mine_patterns(db)
        assert summary.total_commitments == 0
        assert summary.top_commitments == []
        assert summary.top_bigrams == []


class TestExploitableHabits:
    def test_aggressive_flagged(self, db):
        _insert_profile(db, aggression_index=0.80)
        summary = mine_patterns(db)
        assert any("Aggressive" in h for h in summary.exploitable_habits)

    def test_frequent_dodger_flagged(self, db):
        _insert_profile(db, dodge_frequency=0.35)
        summary = mine_patterns(db)
        assert any("dodger" in h for h in summary.exploitable_habits)

    def test_dominant_action_flagged(self, db):
        _insert_profile(
            db,
            action_frequencies=json.dumps({
                "LIGHT_ATTACK": 200, "HEAVY_ATTACK": 10, "MOVE_RIGHT": 10,
            }),
        )
        summary = mine_patterns(db)
        assert any("Over-relies" in h for h in summary.exploitable_habits)

    def test_close_range_preference_flagged(self, db):
        _insert_profile(
            db,
            spacing_distribution=json.dumps({"CLOSE": 200, "MID": 50, "FAR": 10}),
        )
        summary = mine_patterns(db)
        assert any("close range" in h for h in summary.exploitable_habits)


class TestFlattenNgrams:
    def test_basic_flatten(self):
        ngrams = {"A": {"B": 10, "C": 5}}
        result = _flatten_ngrams(ngrams, top_n=3)
        assert result[0] == ("A \u2192 B", 10)
        assert result[1] == ("A \u2192 C", 5)

    def test_empty(self):
        assert _flatten_ngrams({}) == []

    def test_top_n_limits(self):
        ngrams = {"A": {"B": 10, "C": 5, "D": 3, "E": 1}}
        result = _flatten_ngrams(ngrams, top_n=2)
        assert len(result) == 2
