"""Tests for fighter FSM transitions."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState, FREE_STATES
from game.combat.state_machine import (
    can_commit,
    enter_commitment,
    enter_exhausted,
    enter_hitstun,
    enter_ko,
    stop_moving,
    tick_fsm,
)
from game.state import FighterState


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def fighter(cfg):
    return FighterState(
        x=50000, y=30000, hp=cfg.fighter.max_hp,
        stamina=cfg.fighter.max_stamina, facing=1,
    )


class TestCanCommit:
    def test_idle_fighter_can_commit(self, fighter, cfg):
        assert can_commit(fighter, CombatCommitment.LIGHT_ATTACK, cfg)

    def test_locked_fighter_cannot_commit(self, fighter, cfg):
        fighter.fsm_state = FSMState.ATTACK_STARTUP
        assert not can_commit(fighter, CombatCommitment.LIGHT_ATTACK, cfg)

    def test_insufficient_stamina_blocks_attack(self, fighter, cfg):
        fighter.stamina = 5  # light attack costs 15
        assert not can_commit(fighter, CombatCommitment.LIGHT_ATTACK, cfg)

    def test_movement_always_allowed_with_any_stamina(self, fighter, cfg):
        fighter.stamina = 0
        assert can_commit(fighter, CombatCommitment.MOVE_LEFT, cfg)

    def test_ko_cannot_commit(self, fighter, cfg):
        fighter.fsm_state = FSMState.KO
        assert not can_commit(fighter, CombatCommitment.MOVE_LEFT, cfg)


class TestEnterCommitment:
    def test_light_attack_enters_startup(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.LIGHT_ATTACK, cfg)
        assert fighter.fsm_state == FSMState.ATTACK_STARTUP
        assert fighter.fsm_frames_remaining == cfg.actions.light_attack.startup_frames
        assert fighter.active_commitment == CombatCommitment.LIGHT_ATTACK
        assert fighter.stamina == cfg.fighter.max_stamina - cfg.actions.light_attack.stamina_cost

    def test_heavy_attack_enters_startup(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.HEAVY_ATTACK, cfg)
        assert fighter.fsm_state == FSMState.ATTACK_STARTUP
        assert fighter.fsm_frames_remaining == cfg.actions.heavy_attack.startup_frames

    def test_dodge_enters_startup(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        assert fighter.fsm_state == FSMState.DODGE_STARTUP
        assert fighter.fsm_frames_remaining == cfg.actions.dodge_backward.startup_frames

    def test_move_sets_velocity(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.MOVE_RIGHT, cfg)
        assert fighter.fsm_state == FSMState.MOVING
        expected_vel = cfg.fighter.move_speed * cfg.simulation.sub_pixel_scale
        assert fighter.velocity_x == expected_vel


class TestTickFSM:
    def test_attack_flows_startup_to_active_to_recovery_to_idle(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.LIGHT_ATTACK, cfg)

        startup = cfg.actions.light_attack.startup_frames
        active = cfg.actions.light_attack.active_frames
        recovery = cfg.actions.light_attack.recovery_frames

        # Tick through startup
        for _ in range(startup):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.ATTACK_ACTIVE

        # Tick through active
        for _ in range(active):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.ATTACK_RECOVERY

        # Tick through recovery
        for _ in range(recovery):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE
        assert fighter.active_commitment is None

    def test_heavy_attack_total_frames(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.HEAVY_ATTACK, cfg)
        total = cfg.actions.heavy_attack.total_frames
        for _ in range(total):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_dodge_flows_startup_to_dodging_to_recovery_to_idle(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

        startup = cfg.actions.dodge_backward.startup_frames
        active = cfg.actions.dodge_backward.active_frames
        recovery = cfg.actions.dodge_backward.recovery_frames

        for _ in range(startup):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.DODGING

        for _ in range(active):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.DODGE_RECOVERY

        for _ in range(recovery):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_hitstun_returns_to_idle(self, fighter, cfg):
        enter_hitstun(fighter, 10)
        assert fighter.fsm_state == FSMState.HITSTUN
        for _ in range(10):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_exhaustion_returns_to_idle(self, fighter, cfg):
        enter_exhausted(fighter, 30)
        assert fighter.fsm_state == FSMState.EXHAUSTED
        for _ in range(30):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_ko_stays_ko(self, fighter, cfg):
        enter_ko(fighter)
        assert fighter.fsm_state == FSMState.KO
        for _ in range(100):
            tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.KO

    def test_idle_stays_idle(self, fighter, cfg):
        tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_moving_stays_moving(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.MOVE_LEFT, cfg)
        tick_fsm(fighter, cfg)
        assert fighter.fsm_state == FSMState.MOVING

    def test_stop_moving_returns_to_idle(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.MOVE_RIGHT, cfg)
        assert fighter.fsm_state == FSMState.MOVING
        stop_moving(fighter)
        assert fighter.fsm_state == FSMState.IDLE
        assert fighter.velocity_x == 0
