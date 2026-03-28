"""Tests for feature_extractor: 35-feature vector construction."""

from __future__ import annotations

import pytest

from ai.features.feature_extractor import (
    NUM_FEATURES,
    HISTORY_SLOTS,
    NUM_PHASE1,
    extract_features,
)
from ai.profile.player_profile import PlayerProfile
from data.events import EventType, SemanticEvent
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone


MAX_HP = 200
MAX_STAMINA = 100
TICK_RATE = 60


def _make_event(
    actor_hp: int = 200,
    opponent_hp: int = 200,
    actor_stamina: int = 80,
    opponent_stamina: int = 80,
    spacing: SpacingZone | None = SpacingZone.MID,
    opp_fsm: FSMState | None = FSMState.IDLE,
    reaction_ticks: int = 30,
    tick_id: int = 100,
) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id="m1",
        tick_id=tick_id,
        actor=Actor.PLAYER,
        commitment=CombatCommitment.LIGHT_ATTACK,
        opponent_fsm_state=opp_fsm,
        spacing_zone=spacing,
        actor_hp=actor_hp,
        opponent_hp=opponent_hp,
        actor_stamina=actor_stamina,
        opponent_stamina=opponent_stamina,
        reaction_ticks=reaction_ticks,
    )


class TestFeatureVectorShape:
    def test_correct_length(self):
        ev = _make_event()
        profile = PlayerProfile()
        vec = extract_features(ev, [], profile, MAX_HP, MAX_STAMINA, TICK_RATE)
        assert len(vec) == NUM_FEATURES
        assert NUM_FEATURES == 35

    def test_all_values_are_floats(self):
        ev = _make_event()
        profile = PlayerProfile()
        vec = extract_features(ev, [], profile, MAX_HP, MAX_STAMINA, TICK_RATE)
        assert all(isinstance(v, float) for v in vec)


class TestGameStateFeatures:
    def test_hp_fractions(self):
        ev = _make_event(actor_hp=100, opponent_hp=150)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[0] - 0.5) < 1e-6     # actor_hp / 200
        assert abs(vec[1] - 0.75) < 1e-6    # opponent_hp / 200

    def test_stamina_fractions(self):
        ev = _make_event(actor_stamina=50, opponent_stamina=25)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[2] - 0.5) < 1e-6
        assert abs(vec[3] - 0.25) < 1e-6

    def test_spacing_ordinal(self):
        for spacing, expected in [
            (SpacingZone.CLOSE, 0.0),
            (SpacingZone.MID, 0.5),
            (SpacingZone.FAR, 1.0),
        ]:
            ev = _make_event(spacing=spacing)
            vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
            assert abs(vec[4] - expected) < 1e-6

    def test_opponent_fsm_ordinal(self):
        ev_idle = _make_event(opp_fsm=FSMState.IDLE)
        ev_active = _make_event(opp_fsm=FSMState.ATTACK_ACTIVE)
        ev_recovery = _make_event(opp_fsm=FSMState.ATTACK_RECOVERY)
        v_idle = extract_features(ev_idle, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        v_active = extract_features(ev_active, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        v_recovery = extract_features(ev_recovery, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert v_idle[5] == 0.0
        assert v_active[5] == 0.5
        assert v_recovery[5] == 1.0

    def test_hp_differential(self):
        ev = _make_event(actor_hp=150, opponent_hp=100)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[6] - 0.25) < 1e-6   # (150-100)/200

    def test_is_low_hp_flags(self):
        ev_low = _make_event(actor_hp=50, opponent_hp=50)  # 50/200 = 25% < 30%
        ev_high = _make_event(actor_hp=150, opponent_hp=150)
        v_low = extract_features(ev_low, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        v_high = extract_features(ev_high, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert v_low[8] == 1.0   # is_low_hp
        assert v_low[9] == 1.0   # is_opponent_low_hp
        assert v_high[8] == 0.0
        assert v_high[9] == 0.0

    def test_range_flags(self):
        ev_close = _make_event(spacing=SpacingZone.CLOSE)
        ev_far = _make_event(spacing=SpacingZone.FAR)
        v_close = extract_features(ev_close, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        v_far = extract_features(ev_far, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert v_close[10] == 1.0  # is_close
        assert v_close[11] == 0.0
        assert v_far[10] == 0.0
        assert v_far[11] == 1.0    # is_far

    def test_reaction_ticks_normalized(self):
        ev = _make_event(reaction_ticks=60)  # 60/120 = 0.5
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[12] - 0.5) < 1e-6

    def test_tick_fraction(self):
        ev = _make_event(tick_id=3000)  # 3000/6000 = 0.5
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[13] - 0.5) < 1e-6


class TestOneHotHistory:
    def test_empty_history_gives_all_zeros(self):
        ev = _make_event()
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        history_section = vec[14:29]
        assert all(v == 0.0 for v in history_section)

    def test_single_entry_history(self):
        ev = _make_event()
        vec = extract_features(ev, ["LIGHT_ATTACK"], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        # LIGHT_ATTACK is index 2 in PHASE1_NAMES
        # 3 slots: first 2 are empty (indices 14-23 are zeros), third slot at index 24-28
        # slot 0 (14-18): empty
        # slot 1 (19-23): empty
        # slot 2 (24-28): LIGHT_ATTACK = index 2 → position 26
        assert vec[26] == 1.0
        assert sum(vec[14:29]) == 1.0  # exactly one 1.0

    def test_full_history(self):
        ev = _make_event()
        hist = ["MOVE_LEFT", "HEAVY_ATTACK", "DODGE_BACKWARD"]
        vec = extract_features(ev, hist, PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        # slot 0 (14-18): MOVE_LEFT = idx 0 → position 14
        # slot 1 (19-23): HEAVY_ATTACK = idx 3 → position 22
        # slot 2 (24-28): DODGE_BACKWARD = idx 4 → position 28
        assert vec[14] == 1.0
        assert vec[22] == 1.0
        assert vec[28] == 1.0
        assert sum(vec[14:29]) == 3.0

    def test_history_longer_than_slots_uses_last_n(self):
        ev = _make_event()
        hist = ["MOVE_LEFT", "MOVE_RIGHT", "LIGHT_ATTACK", "HEAVY_ATTACK"]
        vec = extract_features(ev, hist, PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        # Should use last 3: MOVE_RIGHT, LIGHT_ATTACK, HEAVY_ATTACK
        # slot 0: MOVE_RIGHT = idx 1 → position 15
        # slot 1: LIGHT_ATTACK = idx 2 → position 21
        # slot 2: HEAVY_ATTACK = idx 3 → position 27
        assert vec[15] == 1.0
        assert vec[21] == 1.0
        assert vec[27] == 1.0
        assert sum(vec[14:29]) == 3.0


class TestProfileAggregates:
    def test_profile_values_appear_in_features(self):
        ev = _make_event()
        profile = PlayerProfile()
        profile.aggression_index = 0.8
        profile.initiative_rate = 0.6
        profile.dodge_frequency = 0.1
        profile.movement_direction_bias = 0.5  # → (0.5+1)/2 = 0.75
        profile.avg_reaction_time_ms = 500.0   # → 500/2000 = 0.25
        profile.win_rate_vs_ai = 0.55

        vec = extract_features(ev, [], profile, MAX_HP, MAX_STAMINA, TICK_RATE)

        assert abs(vec[29] - 0.8) < 1e-6    # aggression
        assert abs(vec[30] - 0.6) < 1e-6    # initiative
        assert abs(vec[31] - 0.1) < 1e-6    # dodge_freq
        assert abs(vec[32] - 0.75) < 1e-6   # direction_bias rescaled
        assert abs(vec[33] - 0.25) < 1e-6   # reaction_time
        assert abs(vec[34] - 0.55) < 1e-6   # win_rate


class TestEdgeCases:
    def test_none_spacing_defaults_to_mid(self):
        ev = _make_event(spacing=None)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert abs(vec[4] - 0.5) < 1e-6  # MID default

    def test_none_opponent_fsm_defaults_to_zero(self):
        ev = _make_event(opp_fsm=None)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert vec[5] == 0.0

    def test_zero_reaction_ticks(self):
        ev = _make_event(reaction_ticks=0)
        vec = extract_features(ev, [], PlayerProfile(), MAX_HP, MAX_STAMINA, TICK_RATE)
        assert vec[12] == 0.0

    def test_values_bounded_zero_to_one(self):
        """All features should be in [0, 1] or [-1, 1] range (differential)."""
        ev = _make_event(actor_hp=0, opponent_hp=200, actor_stamina=0,
                         opponent_stamina=100, reaction_ticks=9999, tick_id=99999)
        profile = PlayerProfile()
        profile.avg_reaction_time_ms = 5000.0
        vec = extract_features(ev, [], profile, MAX_HP, MAX_STAMINA, TICK_RATE)
        # Most features capped at [0, 1]; differential (6, 7) can be negative
        for i, v in enumerate(vec):
            if i in (6, 7):  # differentials
                assert -1.0 <= v <= 1.0, f"feature[{i}] = {v}"
            else:
                assert 0.0 <= v <= 1.0, f"feature[{i}] = {v}"
