"""Tests for stamina system."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.stamina import tick_stamina
from game.combat.state_machine import enter_commitment
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


class TestStaminaRegen:
    def test_idle_regenerates(self, fighter, cfg):
        fighter.stamina = 50
        # Tick enough times for regen to add up
        for _ in range(10):
            tick_stamina(fighter, cfg)
        # Regen idle = 3.0/tick, so after 10 ticks = +30
        assert fighter.stamina == 80

    def test_stamina_caps_at_max(self, fighter, cfg):
        fighter.stamina = cfg.fighter.max_stamina
        for _ in range(10):
            tick_stamina(fighter, cfg)
        assert fighter.stamina == cfg.fighter.max_stamina

    def test_moving_has_net_regen(self, fighter, cfg):
        # Moving: regen=1.5, cost=0.5, net=+1.0/tick
        fighter.stamina = 50
        enter_commitment(fighter, CombatCommitment.MOVE_RIGHT, cfg)
        for _ in range(10):
            tick_stamina(fighter, cfg)
        assert fighter.stamina == 60

    def test_no_regen_during_attack(self, fighter, cfg):
        # During attack states, regen = 0
        initial_stamina = cfg.fighter.max_stamina - cfg.actions.light_attack.stamina_cost
        enter_commitment(fighter, CombatCommitment.LIGHT_ATTACK, cfg)
        assert fighter.stamina == initial_stamina
        for _ in range(5):
            tick_stamina(fighter, cfg)
        assert fighter.stamina == initial_stamina


class TestExhaustion:
    def test_moving_exhausts_when_stamina_drains_to_zero(self, fighter, cfg):
        # Moving: cost=0.5/tick, regen=1.5/tick, net=+1.0/tick.
        # With default config, net regen is positive — moving alone won't exhaust.
        # To test exhaustion: set accumulator so that cost pushes acc below -1.0
        # while stamina is at 1, causing integer deduction to 0.
        fighter.stamina = 1
        # acc = -0.6: after cost(-0.5) = -1.1, regen(+1.5) = +0.4
        # But deduction check happens at acc <= -1.0 AFTER regen:
        # acc after cost: -1.1, after regen: +0.4. acc is positive, no deduction.
        # We need the net to be negative. Use acc = -1.6 to force it:
        # acc = -1.6: after cost(-0.5) = -2.1, after regen(+1.5) = -0.6
        # -0.6 is not <= -1.0. Still no deduction.
        # The only way is to have acc start very negative:
        # acc = -2.0: after cost = -2.5, after regen = -1.0 → deduct 1, stamina=0
        fighter.stamina_accumulator = -2.0
        enter_commitment(fighter, CombatCommitment.MOVE_RIGHT, cfg)
        exhausted = tick_stamina(fighter, cfg)
        assert exhausted
        assert fighter.fsm_state == FSMState.EXHAUSTED
        assert fighter.stamina == 0

    def test_no_exhaustion_while_idle(self, fighter, cfg):
        fighter.stamina = 0
        # Idle, not moving — should regen, not exhaust
        exhausted = tick_stamina(fighter, cfg)
        assert not exhausted
        assert fighter.fsm_state != FSMState.EXHAUSTED

    def test_no_changes_during_ko(self, fighter, cfg):
        fighter.fsm_state = FSMState.KO
        fighter.stamina = 50
        tick_stamina(fighter, cfg)
        assert fighter.stamina == 50


class TestStaminaCosts:
    def test_light_attack_deducts_on_commit(self, fighter, cfg):
        initial = fighter.stamina
        enter_commitment(fighter, CombatCommitment.LIGHT_ATTACK, cfg)
        assert fighter.stamina == initial - cfg.actions.light_attack.stamina_cost

    def test_heavy_attack_deducts_on_commit(self, fighter, cfg):
        initial = fighter.stamina
        enter_commitment(fighter, CombatCommitment.HEAVY_ATTACK, cfg)
        assert fighter.stamina == initial - cfg.actions.heavy_attack.stamina_cost

    def test_dodge_deducts_on_commit(self, fighter, cfg):
        initial = fighter.stamina
        enter_commitment(fighter, CombatCommitment.DODGE_BACKWARD, cfg)
        assert fighter.stamina == initial - cfg.actions.dodge_backward.stamina_cost
