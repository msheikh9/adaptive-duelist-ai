"""Tests for core enum types — existence, membership, and separation."""

from __future__ import annotations

from ai.strategy.tactics import TacticalIntent
from game.combat.actions import (
    PHASE_1_COMMITMENTS,
    FREE_STATES,
    LOCKED_STATES,
    Actor,
    CombatCommitment,
    FSMState,
    SpacingZone,
)
from game.input.input_actions import InputAction


class TestInputAction:
    def test_all_members_exist(self):
        assert len(InputAction) == 12  # Phase 20: +PRESS_SHOOT, RELEASE_SHOOT
        assert InputAction.PRESS_LEFT is not None
        assert InputAction.PRESS_RIGHT is not None
        assert InputAction.RELEASE_LEFT is not None
        assert InputAction.RELEASE_RIGHT is not None
        assert InputAction.PRESS_LIGHT_ATTACK is not None
        assert InputAction.PRESS_HEAVY_ATTACK is not None
        assert InputAction.PRESS_DODGE is not None
        assert InputAction.PRESS_BLOCK is not None
        assert InputAction.RELEASE_BLOCK is not None
        assert InputAction.PRESS_JUMP is not None

    def test_members_are_unique(self):
        values = [m.value for m in InputAction]
        assert len(values) == len(set(values))


class TestCombatCommitment:
    def test_phase_1_commitments(self):
        assert CombatCommitment.MOVE_LEFT in PHASE_1_COMMITMENTS
        assert CombatCommitment.MOVE_RIGHT in PHASE_1_COMMITMENTS
        assert CombatCommitment.LIGHT_ATTACK in PHASE_1_COMMITMENTS
        assert CombatCommitment.HEAVY_ATTACK in PHASE_1_COMMITMENTS
        assert CombatCommitment.DODGE_BACKWARD in PHASE_1_COMMITMENTS

    def test_phase_2_commitments_exist_but_not_in_phase_1(self):
        assert CombatCommitment.DODGE_LEFT not in PHASE_1_COMMITMENTS
        assert CombatCommitment.DODGE_RIGHT not in PHASE_1_COMMITMENTS

    def test_phase_3_commitments_exist_but_not_in_phase_1(self):
        assert CombatCommitment.BLOCK_START not in PHASE_1_COMMITMENTS
        assert CombatCommitment.BLOCK_RELEASE not in PHASE_1_COMMITMENTS

    def test_members_are_unique(self):
        values = [m.value for m in CombatCommitment]
        assert len(values) == len(set(values))


class TestFSMState:
    def test_free_and_locked_are_disjoint(self):
        assert FREE_STATES & LOCKED_STATES == set()

    def test_free_and_locked_cover_all_states(self):
        assert FREE_STATES | LOCKED_STATES == set(FSMState)

    def test_idle_and_moving_are_free(self):
        assert FSMState.IDLE in FREE_STATES
        assert FSMState.MOVING in FREE_STATES

    def test_attack_states_are_locked(self):
        assert FSMState.ATTACK_STARTUP in LOCKED_STATES
        assert FSMState.ATTACK_ACTIVE in LOCKED_STATES
        assert FSMState.ATTACK_RECOVERY in LOCKED_STATES

    def test_hitstun_is_locked(self):
        assert FSMState.HITSTUN in LOCKED_STATES

    def test_ko_is_locked(self):
        assert FSMState.KO in LOCKED_STATES


class TestTacticalIntent:
    def test_all_modes_exist(self):
        assert len(TacticalIntent) == 7
        assert TacticalIntent.EXPLOIT_PATTERN is not None
        assert TacticalIntent.BAIT_AND_PUNISH is not None
        assert TacticalIntent.PUNISH_RECOVERY is not None
        assert TacticalIntent.PRESSURE_STAMINA is not None
        assert TacticalIntent.DEFENSIVE_RESET is not None
        assert TacticalIntent.PROBE_BEHAVIOR is not None
        assert TacticalIntent.NEUTRAL_SPACING is not None


class TestSpacingZone:
    def test_three_zones(self):
        assert len(SpacingZone) == 3


class TestActor:
    def test_two_actors(self):
        assert len(Actor) == 2
        assert Actor.PLAYER is not None
        assert Actor.AI is not None
