"""Tests for damage application."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.collision import HitEvent
from game.combat.damage import apply_hit
from game.state import FighterState


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def defender():
    return FighterState(x=60000, y=30000, hp=200, stamina=100, facing=-1)


class TestApplyHit:
    def test_reduces_hp(self, defender, cfg):
        hit = HitEvent(
            attacker_commitment=CombatCommitment.LIGHT_ATTACK,
            damage=8, hitstun_frames=6,
            knockback_sub=3000, knockback_direction=1,
        )
        apply_hit(defender, hit)
        assert defender.hp == 192

    def test_enters_hitstun(self, defender, cfg):
        hit = HitEvent(
            attacker_commitment=CombatCommitment.LIGHT_ATTACK,
            damage=8, hitstun_frames=6,
            knockback_sub=3000, knockback_direction=1,
        )
        apply_hit(defender, hit)
        assert defender.fsm_state == FSMState.HITSTUN
        assert defender.fsm_frames_remaining == 6

    def test_applies_knockback(self, defender, cfg):
        original_x = defender.x
        hit = HitEvent(
            attacker_commitment=CombatCommitment.HEAVY_ATTACK,
            damage=22, hitstun_frames=12,
            knockback_sub=8000, knockback_direction=1,
        )
        apply_hit(defender, hit)
        assert defender.x == original_x + 8000

    def test_ko_when_hp_reaches_zero(self, defender, cfg):
        defender.hp = 10
        hit = HitEvent(
            attacker_commitment=CombatCommitment.HEAVY_ATTACK,
            damage=22, hitstun_frames=12,
            knockback_sub=8000, knockback_direction=-1,
        )
        apply_hit(defender, hit)
        assert defender.hp == 0
        assert defender.fsm_state == FSMState.KO

    def test_hp_does_not_go_negative(self, defender, cfg):
        defender.hp = 5
        hit = HitEvent(
            attacker_commitment=CombatCommitment.HEAVY_ATTACK,
            damage=22, hitstun_frames=12,
            knockback_sub=8000, knockback_direction=1,
        )
        apply_hit(defender, hit)
        assert defender.hp == 0

    def test_interrupts_active_commitment(self, defender, cfg):
        from game.combat.state_machine import enter_commitment
        enter_commitment(defender, CombatCommitment.HEAVY_ATTACK, cfg)
        assert defender.active_commitment == CombatCommitment.HEAVY_ATTACK

        hit = HitEvent(
            attacker_commitment=CombatCommitment.LIGHT_ATTACK,
            damage=8, hitstun_frames=6,
            knockback_sub=3000, knockback_direction=1,
        )
        apply_hit(defender, hit)
        assert defender.fsm_state == FSMState.HITSTUN
        assert defender.active_commitment is None
