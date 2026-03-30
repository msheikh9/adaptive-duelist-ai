"""Tests for ai/profile/archetype_classifier.py."""

from __future__ import annotations

import pytest

from ai.profile.archetype_classifier import (
    ArchetypeLabel,
    classify_archetype,
    _top2_share,
)
from ai.profile.player_profile import PlayerProfile


def _make_profile(
    action_frequencies: dict | None = None,
    aggression_index: float = 0.0,
    dodge_frequency: float = 0.0,
    match_count: int = 10,
) -> PlayerProfile:
    profile = PlayerProfile()
    profile.match_count = match_count
    profile.aggression_index = aggression_index
    profile.dodge_frequency = dodge_frequency
    if action_frequencies is not None:
        profile.action_frequencies = action_frequencies
    return profile


class TestClassifyArchetype:

    def test_balanced_when_no_dominant_style(self):
        profile = _make_profile(
            action_frequencies={
                "LIGHT_ATTACK": 4, "HEAVY_ATTACK": 3,
                "DODGE_BACKWARD": 3, "MOVE_LEFT": 3, "MOVE_RIGHT": 3,
            },
            aggression_index=0.35,
            dodge_frequency=0.15,
        )
        assert classify_archetype(profile) == ArchetypeLabel.BALANCED

    def test_aggressive_high_attack_rate(self):
        profile = _make_profile(
            action_frequencies={
                "LIGHT_ATTACK": 8, "HEAVY_ATTACK": 6,
                "MOVE_RIGHT": 2, "DODGE_BACKWARD": 1, "MOVE_LEFT": 1,
            },
            aggression_index=0.70,
            dodge_frequency=0.05,
        )
        assert classify_archetype(profile) == ArchetypeLabel.AGGRESSIVE

    def test_defensive_low_attack_high_dodge(self):
        profile = _make_profile(
            action_frequencies={
                "DODGE_BACKWARD": 8, "MOVE_LEFT": 4,
                "MOVE_RIGHT": 3, "LIGHT_ATTACK": 2, "HEAVY_ATTACK": 1,
            },
            aggression_index=0.15,
            dodge_frequency=0.30,
        )
        assert classify_archetype(profile) == ArchetypeLabel.DEFENSIVE

    def test_evasive_very_high_dodge(self):
        profile = _make_profile(
            action_frequencies={
                "DODGE_BACKWARD": 12, "MOVE_LEFT": 3,
                "LIGHT_ATTACK": 2, "HEAVY_ATTACK": 1, "MOVE_RIGHT": 1,
            },
            aggression_index=0.20,
            dodge_frequency=0.55,
        )
        assert classify_archetype(profile) == ArchetypeLabel.EVASIVE

    def test_patterned_concentrated_distribution(self):
        profile = _make_profile(
            action_frequencies={
                "LIGHT_ATTACK": 15, "HEAVY_ATTACK": 10,
                "DODGE_BACKWARD": 2, "MOVE_LEFT": 1, "MOVE_RIGHT": 1,
            },
            aggression_index=0.43,
            dodge_frequency=0.07,
        )
        # top-2 share = (15+10)/29 ≈ 0.86 ≥ 0.70 → PATTERNED
        assert classify_archetype(profile) == ArchetypeLabel.PATTERNED

    def test_balanced_when_insufficient_commitments(self):
        profile = _make_profile(
            action_frequencies={"LIGHT_ATTACK": 3},
            aggression_index=0.9,
            dodge_frequency=0.0,
        )
        # total_commitments = 3 < _MIN_COMMITMENTS_TO_CLASSIFY (5)
        assert classify_archetype(profile) == ArchetypeLabel.BALANCED

    def test_evasive_overrides_aggressive_when_high_dodge(self):
        """High aggression AND very high dodge → EVASIVE wins (priority order)."""
        profile = _make_profile(
            action_frequencies={
                "LIGHT_ATTACK": 6, "DODGE_BACKWARD": 8,
                "HEAVY_ATTACK": 2, "MOVE_LEFT": 1, "MOVE_RIGHT": 1,
            },
            aggression_index=0.55,
            dodge_frequency=0.44,
        )
        assert classify_archetype(profile) == ArchetypeLabel.EVASIVE

    def test_at_exact_aggressive_threshold(self):
        profile = _make_profile(
            action_frequencies={
                "LIGHT_ATTACK": 5, "MOVE_RIGHT": 5,
            },
            aggression_index=0.50,
            dodge_frequency=0.05,
        )
        assert classify_archetype(profile) == ArchetypeLabel.AGGRESSIVE

    def test_at_exact_defensive_threshold(self):
        profile = _make_profile(
            action_frequencies={
                "DODGE_BACKWARD": 5, "MOVE_LEFT": 3, "LIGHT_ATTACK": 2,
            },
            aggression_index=0.20,
            dodge_frequency=0.25,
        )
        assert classify_archetype(profile) == ArchetypeLabel.DEFENSIVE

    def test_empty_action_frequencies_returns_balanced(self):
        profile = _make_profile(
            action_frequencies={},
            aggression_index=0.0,
            dodge_frequency=0.0,
        )
        assert classify_archetype(profile) == ArchetypeLabel.BALANCED


class TestDeterminism:

    def test_same_profile_same_result(self):
        profile = _make_profile(
            action_frequencies={"LIGHT_ATTACK": 7, "HEAVY_ATTACK": 5, "DODGE_BACKWARD": 2},
            aggression_index=0.60,
            dodge_frequency=0.10,
        )
        r1 = classify_archetype(profile)
        r2 = classify_archetype(profile)
        assert r1 == r2

    def test_all_labels_are_valid_enum_members(self):
        profiles = [
            _make_profile(aggression_index=0.70, dodge_frequency=0.05),
            _make_profile(aggression_index=0.15, dodge_frequency=0.35),
            _make_profile(aggression_index=0.25, dodge_frequency=0.50),
            _make_profile(
                action_frequencies={"LIGHT_ATTACK": 20, "HEAVY_ATTACK": 15, "DODGE_BACKWARD": 1},
                aggression_index=0.40,
                dodge_frequency=0.03,
            ),
            _make_profile(aggression_index=0.35, dodge_frequency=0.15),
        ]
        for profile in profiles:
            label = classify_archetype(profile)
            assert isinstance(label, ArchetypeLabel)
            assert label in list(ArchetypeLabel)


class TestTop2Share:

    def test_empty_dict_returns_zero(self):
        assert _top2_share({}) == 0.0

    def test_single_action(self):
        assert _top2_share({"LIGHT_ATTACK": 5}) == pytest.approx(1.0)

    def test_two_equal_actions(self):
        assert _top2_share({"A": 3, "B": 3}) == pytest.approx(1.0)

    def test_three_actions_top2(self):
        # top2 = 6+4 = 10 out of 12 total
        result = _top2_share({"A": 6, "B": 4, "C": 2})
        assert result == pytest.approx(10 / 12)

    def test_uniform_distribution(self):
        # 5 equal actions: top-2 = 2/5 = 0.4
        result = _top2_share({"A": 1, "B": 1, "C": 1, "D": 1, "E": 1})
        assert result == pytest.approx(0.4)
