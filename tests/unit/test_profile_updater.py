"""Tests for ProfileUpdater metric computation."""

from __future__ import annotations

import math
import pytest

from config.config_loader import load_config
from data.events import EventType, SemanticEvent
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone
from ai.profile.player_profile import PlayerProfile
from ai.profile.profile_updater import ProfileUpdater


@pytest.fixture
def cfg():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def updater(cfg):
    game_cfg, ai_cfg = cfg
    u = ProfileUpdater(ai_cfg, game_cfg)
    u.on_match_start("test-match")
    return u, game_cfg


@pytest.fixture
def profile():
    return PlayerProfile()


def _commitment_event(commitment: CombatCommitment, actor: Actor = Actor.PLAYER,
                      spacing: SpacingZone = SpacingZone.MID,
                      actor_hp: int = 200, opponent_hp: int = 200,
                      actor_stamina: int = 80, opponent_stamina: int = 80,
                      opponent_fsm: FSMState = FSMState.IDLE,
                      reaction_ticks: int = 30) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id="test",
        tick_id=0,
        actor=actor,
        commitment=commitment,
        opponent_fsm_state=opponent_fsm,
        spacing_zone=spacing,
        actor_hp=actor_hp,
        opponent_hp=opponent_hp,
        actor_stamina=actor_stamina,
        opponent_stamina=opponent_stamina,
        reaction_ticks=reaction_ticks,
    )


def _hit_event(actor: Actor = Actor.PLAYER,
               opponent_fsm: FSMState = FSMState.ATTACK_RECOVERY) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.HIT_LANDED,
        match_id="test",
        tick_id=0,
        actor=actor,
        commitment=CombatCommitment.LIGHT_ATTACK,
        opponent_fsm_state=opponent_fsm,
        actor_hp=200, opponent_hp=192,
        actor_stamina=80, opponent_stamina=80,
        damage_dealt=8,
    )


class TestActionFrequencies:
    def test_increments_on_player_commitment(self, updater, profile):
        u, _ = updater
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        assert profile.action_frequencies.get("LIGHT_ATTACK") == 1

    def test_multiple_commitments_accumulate(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        u.on_event(profile, _commitment_event(CombatCommitment.HEAVY_ATTACK))
        assert profile.action_frequencies["LIGHT_ATTACK"] == 5
        assert profile.action_frequencies["HEAVY_ATTACK"] == 1

    def test_ignores_ai_commitments(self, updater, profile):
        u, _ = updater
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK, actor=Actor.AI))
        assert profile.action_frequencies == {}

    def test_recent_frequencies_reflect_rolling_window(self, updater, profile):
        u, _ = updater
        for _ in range(3):
            u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        u.on_event(profile, _commitment_event(CombatCommitment.HEAVY_ATTACK))
        assert profile.recent_action_frequencies["LIGHT_ATTACK"] == 3
        assert profile.recent_action_frequencies["HEAVY_ATTACK"] == 1


class TestAggressionIndex:
    def test_pure_attack_is_one(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        assert profile.aggression_index == 1.0

    def test_pure_movement_is_zero(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_RIGHT))
        assert profile.aggression_index == 0.0

    def test_half_attacks(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_RIGHT))
        assert abs(profile.aggression_index - 0.5) < 1e-9


class TestMovementDirectionBias:
    def test_all_right_is_plus_one(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_RIGHT))
        assert profile.movement_direction_bias == 1.0

    def test_all_left_is_minus_one(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_LEFT))
        assert profile.movement_direction_bias == -1.0

    def test_equal_split_is_zero(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_RIGHT))
            u.on_event(profile, _commitment_event(CombatCommitment.MOVE_LEFT))
        assert profile.movement_direction_bias == 0.0

    def test_no_movement_stays_zero(self, updater, profile):
        u, _ = updater
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK))
        assert profile.movement_direction_bias == 0.0


class TestSpacingDistribution:
    def test_records_spacing_zone(self, updater, profile):
        u, _ = updater
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK,
                                               spacing=SpacingZone.CLOSE))
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK,
                                               spacing=SpacingZone.MID))
        u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK,
                                               spacing=SpacingZone.FAR))
        assert profile.spacing_distribution["CLOSE"] == 1
        assert profile.spacing_distribution["MID"] == 1
        assert profile.spacing_distribution["FAR"] == 1

    def test_spacing_accumulates(self, updater, profile):
        u, _ = updater
        for _ in range(4):
            u.on_event(profile, _commitment_event(CombatCommitment.LIGHT_ATTACK,
                                                   spacing=SpacingZone.CLOSE))
        assert profile.spacing_distribution["CLOSE"] == 4


class TestInitiativeRate:
    def test_full_initiative_when_opponent_always_idle(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK,
                opponent_fsm=FSMState.IDLE))
        assert profile.initiative_rate == 1.0

    def test_zero_initiative_when_opponent_always_locked(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK,
                opponent_fsm=FSMState.ATTACK_ACTIVE))
        assert profile.initiative_rate == 0.0

    def test_mixed_initiative(self, updater, profile):
        u, _ = updater
        for _ in range(3):
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK, opponent_fsm=FSMState.IDLE))
        for _ in range(2):
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK, opponent_fsm=FSMState.ATTACK_ACTIVE))
        assert abs(profile.initiative_rate - 0.6) < 1e-9


class TestPunishConversionRate:
    def test_all_punishes(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _hit_event(
                opponent_fsm=FSMState.ATTACK_RECOVERY))
        assert profile.punish_conversion_rate == 1.0

    def test_no_punishes(self, updater, profile):
        u, _ = updater
        for _ in range(5):
            u.on_event(profile, _hit_event(
                opponent_fsm=FSMState.IDLE))
        assert profile.punish_conversion_rate == 0.0

    def test_mixed_punish_rate(self, updater, profile):
        u, _ = updater
        for _ in range(2):
            u.on_event(profile, _hit_event(opponent_fsm=FSMState.ATTACK_RECOVERY))
        for _ in range(2):
            u.on_event(profile, _hit_event(opponent_fsm=FSMState.IDLE))
        assert abs(profile.punish_conversion_rate - 0.5) < 1e-9


class TestLowHpDistribution:
    def test_normal_hp_not_tracked(self, updater, profile):
        u, game_cfg = updater
        u.on_event(profile, _commitment_event(
            CombatCommitment.LIGHT_ATTACK, actor_hp=game_cfg.fighter.max_hp))
        assert profile.low_hp_action_distribution == {}

    def test_low_hp_is_tracked(self, updater, profile):
        u, game_cfg = updater
        low_hp = int(game_cfg.fighter.max_hp * 0.2)
        u.on_event(profile, _commitment_event(
            CombatCommitment.LIGHT_ATTACK, actor_hp=low_hp))
        assert profile.low_hp_action_distribution.get("LIGHT_ATTACK") == 1


class TestReactionTime:
    def test_reaction_time_recorded(self, updater, profile):
        u, game_cfg = updater
        tick_rate = game_cfg.simulation.tick_rate
        # reaction_ticks=60 → 1000ms at 60Hz
        u.on_event(profile, _commitment_event(
            CombatCommitment.LIGHT_ATTACK, reaction_ticks=60))
        # Mean is updated on match_end via Welford
        assert len(u._reaction_times_ms) == 1
        expected_ms = 60 / tick_rate * 1000
        assert abs(u._reaction_times_ms[0] - expected_ms) < 1.0

    def test_zero_reaction_ticks_not_recorded(self, updater, profile):
        u, _ = updater
        u.on_event(profile, _commitment_event(
            CombatCommitment.LIGHT_ATTACK, reaction_ticks=0))
        assert len(u._reaction_times_ms) == 0


class TestMatchEnd:
    def test_match_count_increments(self, updater, profile):
        u, game_cfg = updater
        u.on_match_end(profile, "PLAYER", 3600)
        assert profile.match_count == 1

    def test_win_count_increments_on_player_win(self, updater, profile):
        u, game_cfg = updater
        u.on_match_end(profile, "PLAYER", 3600)
        assert profile.win_count == 1
        assert profile.win_rate_vs_ai == 1.0

    def test_win_count_unchanged_on_ai_win(self, updater, profile):
        u, game_cfg = updater
        u.on_match_end(profile, "AI", 3600)
        assert profile.win_count == 0
        assert profile.win_rate_vs_ai == 0.0

    def test_win_rate_across_multiple_matches(self, updater, profile):
        u, game_cfg = updater
        u.on_match_end(profile, "PLAYER", 3600)
        u.on_match_start("m2")
        u.on_match_end(profile, "AI", 3600)
        u.on_match_start("m3")
        u.on_match_end(profile, "PLAYER", 3600)
        assert profile.match_count == 3
        assert profile.win_count == 2
        assert abs(profile.win_rate_vs_ai - 2/3) < 1e-9

    def test_avg_match_duration_computed(self, updater, profile, cfg):
        u, game_cfg = updater
        tick_rate = game_cfg.simulation.tick_rate
        u.on_match_end(profile, "AI", tick_rate * 60)   # 60 seconds
        assert abs(profile.avg_match_duration - 60.0) < 0.1

    def test_total_ticks_observed_accumulates(self, updater, profile):
        u, game_cfg = updater
        u.on_match_end(profile, "AI", 1000)
        u.on_match_start("m2")
        u.on_match_end(profile, "AI", 2000)
        assert profile.total_ticks_observed == 3000

    def test_reaction_time_welford_mean(self, updater, profile, cfg):
        u, game_cfg = updater
        tick_rate = game_cfg.simulation.tick_rate
        # Feed 3 identical reaction times of 30 ticks = 500ms
        for _ in range(3):
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK, reaction_ticks=30))
        u.on_match_end(profile, "AI", 1000)
        expected_ms = 30 / tick_rate * 1000
        assert abs(profile.avg_reaction_time_ms - expected_ms) < 1.0
        # Stddev of identical values = 0
        assert abs(profile.reaction_time_stddev) < 1e-6

    def test_reaction_time_stddev_nonzero(self, updater, profile, cfg):
        u, game_cfg = updater
        tick_rate = game_cfg.simulation.tick_rate
        # Feed 2 different reaction times
        for rt in [10, 50]:
            u.on_event(profile, _commitment_event(
                CombatCommitment.LIGHT_ATTACK, reaction_ticks=rt))
        u.on_match_end(profile, "AI", 1000)
        assert profile.reaction_time_stddev > 0.0
