"""Tests for PlayerProfile serialization and computed properties."""

from __future__ import annotations

import json
import pytest

from ai.profile.player_profile import PlayerProfile


@pytest.fixture
def empty_profile():
    return PlayerProfile()


@pytest.fixture
def populated_profile():
    p = PlayerProfile()
    p.action_frequencies = {
        "LIGHT_ATTACK": 40,
        "HEAVY_ATTACK": 10,
        "DODGE_BACKWARD": 15,
        "MOVE_RIGHT": 25,
        "MOVE_LEFT": 10,
    }
    p.aggression_index = 0.5
    p.dodge_frequency = 0.15
    p.match_count = 5
    p.win_count = 3
    p.win_rate_vs_ai = 0.6
    p.reaction_count = 20
    p.avg_reaction_time_ms = 250.0
    p.reaction_time_stddev = 40.0
    p.reaction_M2 = 30400.0
    p.duration_sum = 600.0
    p.avg_match_duration = 120.0
    return p


class TestPlayerProfileProperties:
    def test_total_commitments(self, populated_profile):
        assert populated_profile.total_commitments == 100

    def test_total_attacks(self, populated_profile):
        assert populated_profile.total_attacks == 50

    def test_total_dodges(self, populated_profile):
        assert populated_profile.total_dodges == 15

    def test_empty_total_commitments(self, empty_profile):
        assert empty_profile.total_commitments == 0
        assert empty_profile.total_attacks == 0
        assert empty_profile.total_dodges == 0


class TestPlayerProfileSerialization:
    def test_to_db_row_length(self, populated_profile):
        row = populated_profile.to_db_row()
        # 23 schema cols + 5 accumulator cols = 28
        assert len(row) == 28

    def test_to_db_row_player_id(self, populated_profile):
        row = populated_profile.to_db_row()
        assert row[0] == "player_1"

    def test_action_frequencies_json(self, populated_profile):
        row = populated_profile.to_db_row()
        # action_frequencies is index 3
        parsed = json.loads(row[3])
        assert parsed["LIGHT_ATTACK"] == 40
        assert parsed["MOVE_RIGHT"] == 25

    def test_bigrams_json(self):
        p = PlayerProfile()
        p.bigrams = {"LIGHT_ATTACK": {"DODGE_BACKWARD": 3}}
        row = p.to_db_row()
        parsed = json.loads(row[5])  # bigrams index
        assert parsed["LIGHT_ATTACK"]["DODGE_BACKWARD"] == 3

    def test_trigrams_json(self):
        p = PlayerProfile()
        p.trigrams = {"LIGHT_ATTACK,DODGE_BACKWARD": {"MOVE_RIGHT": 2}}
        row = p.to_db_row()
        parsed = json.loads(row[6])
        assert parsed["LIGHT_ATTACK,DODGE_BACKWARD"]["MOVE_RIGHT"] == 2

    def test_accumulator_columns(self, populated_profile):
        row = populated_profile.to_db_row()
        # v002 columns are last 5
        match_count, win_count, reaction_count, reaction_M2, duration_sum = row[-5:]
        assert match_count == 5
        assert win_count == 3
        assert reaction_count == 20
        assert duration_sum == 600.0


class TestPlayerProfileFromDbRow:
    def _make_row(self, profile: PlayerProfile):
        """Simulate a sqlite3.Row using a dict proxy."""
        row = profile.to_db_row()
        cols = [
            "player_id", "session_count", "total_ticks_observed",
            "action_frequencies", "recent_action_frequencies", "bigrams", "trigrams",
            "spacing_distribution", "movement_direction_bias",
            "dodge_left_pct", "dodge_right_pct", "dodge_frequency",
            "aggression_index", "initiative_rate", "punish_conversion_rate",
            "low_hp_action_dist", "cornered_action_dist", "combo_sequences",
            "avg_reaction_time_ms", "reaction_time_stddev",
            "win_rate_vs_ai", "avg_match_duration", "last_updated",
            "match_count", "win_count", "reaction_count", "reaction_M2", "duration_sum",
        ]
        return dict(zip(cols, row))

    def test_roundtrip_scalars(self, populated_profile):
        row = self._make_row(populated_profile)
        restored = PlayerProfile.from_db_row(row)
        assert restored.match_count == populated_profile.match_count
        assert restored.win_count == populated_profile.win_count
        assert restored.win_rate_vs_ai == populated_profile.win_rate_vs_ai
        assert restored.avg_reaction_time_ms == populated_profile.avg_reaction_time_ms

    def test_roundtrip_dicts(self, populated_profile):
        row = self._make_row(populated_profile)
        restored = PlayerProfile.from_db_row(row)
        assert restored.action_frequencies == populated_profile.action_frequencies

    def test_roundtrip_nested_dicts(self):
        p = PlayerProfile()
        p.bigrams = {"LIGHT_ATTACK": {"DODGE_BACKWARD": 5, "MOVE_RIGHT": 2}}
        p.trigrams = {"LIGHT_ATTACK,DODGE_BACKWARD": {"LIGHT_ATTACK": 3}}
        row_dict = dict(zip(
            ["player_id", "session_count", "total_ticks_observed",
             "action_frequencies", "recent_action_frequencies", "bigrams", "trigrams",
             "spacing_distribution", "movement_direction_bias",
             "dodge_left_pct", "dodge_right_pct", "dodge_frequency",
             "aggression_index", "initiative_rate", "punish_conversion_rate",
             "low_hp_action_dist", "cornered_action_dist", "combo_sequences",
             "avg_reaction_time_ms", "reaction_time_stddev",
             "win_rate_vs_ai", "avg_match_duration", "last_updated",
             "match_count", "win_count", "reaction_count", "reaction_M2", "duration_sum"],
            p.to_db_row()
        ))
        restored = PlayerProfile.from_db_row(row_dict)
        assert restored.bigrams == p.bigrams
        assert restored.trigrams == p.trigrams

    def test_summary_returns_string(self, populated_profile):
        s = populated_profile.summary()
        assert isinstance(s, str)
        assert "matches=" in s
        assert "win_rate=" in s
