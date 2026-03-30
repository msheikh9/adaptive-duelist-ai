"""Tests for Phase 17 dodge cooldown enforcement."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.state_machine import (
    can_commit,
    enter_commitment,
    tick_dodge_cooldown,
)
from game.state import FighterState


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def fighter(cfg):
    return FighterState(
        x=50000,
        y=cfg.arena.ground_y * cfg.simulation.sub_pixel_scale,
        hp=cfg.fighter.max_hp,
        stamina=cfg.fighter.max_stamina,
        facing=1,
    )


class TestDodgeCooldownEnforcement:
    def test_can_dodge_when_cooldown_zero(self, fighter, cfg):
        assert fighter.dodge_cooldown == 0
        assert can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_cannot_dodge_while_cooldown_active(self, fighter, cfg):
        fighter.dodge_cooldown = 10
        assert not can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_cannot_dodge_on_cooldown_1(self, fighter, cfg):
        fighter.dodge_cooldown = 1
        assert not can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_can_dodge_when_cooldown_exactly_zero(self, fighter, cfg):
        fighter.dodge_cooldown = 0
        assert can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_cooldown_set_on_commit(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        assert fighter.dodge_cooldown == cfg.actions.dodge_backward.cooldown_frames

    def test_cooldown_positive_after_commit(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        assert fighter.dodge_cooldown > 0

    def test_cooldown_not_set_for_light_attack(self, fighter, cfg):
        enter_commitment(fighter, CombatCommitment.LIGHT_ATTACK, cfg)
        assert fighter.dodge_cooldown == 0

    def test_cooldown_countdown(self, fighter, cfg):
        fighter.dodge_cooldown = 5
        tick_dodge_cooldown(fighter)
        assert fighter.dodge_cooldown == 4

    def test_cooldown_does_not_go_below_zero(self, fighter, cfg):
        fighter.dodge_cooldown = 0
        tick_dodge_cooldown(fighter)
        assert fighter.dodge_cooldown == 0

    def test_cooldown_counts_down_to_zero(self, fighter, cfg):
        fighter.dodge_cooldown = 3
        for _ in range(3):
            tick_dodge_cooldown(fighter)
        assert fighter.dodge_cooldown == 0

    def test_repeated_dodge_blocked_during_cooldown(self, fighter, cfg):
        """Second dodge attempt while cooldown is active must fail."""
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        # Fighter is now in DODGE_STARTUP and has cooldown active
        # Put back to idle to test can_commit gating specifically
        fighter.fsm_state = FSMState.IDLE
        assert fighter.dodge_cooldown > 0
        assert not can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_can_dodge_again_after_cooldown_expires(self, fighter, cfg):
        """After cooldown counts to zero, dodge is available again."""
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        fighter.fsm_state = FSMState.IDLE  # force to free state

        # Drain cooldown
        while fighter.dodge_cooldown > 0:
            tick_dodge_cooldown(fighter)

        assert fighter.dodge_cooldown == 0
        assert can_commit(fighter, CombatCommitment.DODGE_BACKWARD, cfg)

    def test_cooldown_frames_config_value(self, cfg):
        """Cooldown frames must be > 0 in config (anti-spam guarantee)."""
        assert cfg.actions.dodge_backward.cooldown_frames > 0

    def test_cooldown_longer_than_dodge_animation(self, cfg):
        """Cooldown must exceed total dodge animation so spam is impossible."""
        da = cfg.actions.dodge_backward
        total_animation = da.startup_frames + da.active_frames + da.recovery_frames
        assert da.cooldown_frames > total_animation
