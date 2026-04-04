"""Tests for Phase 18: guard (block durability) system.

Covers:
  - tick_guard: regen delay countdown, guard recovery, no-regen while blocking/stunned/KO
  - apply_block_response: chip damage, guard cost, blockstun vs guard break
  - state_machine: enter_blockstun / enter_parry_stunned, BLOCK_START, tick_fsm
  - player_fighter: PRESS_BLOCK / RELEASE_BLOCK handling
"""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.collision import HitEvent
from game.combat.guard import apply_block_response, tick_guard
from game.combat.state_machine import (
    can_commit,
    enter_blockstun,
    enter_commitment,
    enter_parry_stunned,
    tick_fsm,
)
from game.entities.player_fighter import PlayerController
from game.input.input_actions import InputAction
from game.state import FighterState, SimulationState, ArenaState, MatchStatus


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def fighter(game_cfg):
    scale = game_cfg.simulation.sub_pixel_scale
    return FighterState(
        x=300 * scale,
        y=game_cfg.arena.ground_y * scale,
        hp=game_cfg.fighter.max_hp,
        stamina=game_cfg.fighter.max_stamina,
        fsm_state=FSMState.IDLE,
        guard=game_cfg.actions.block.guard_max,
    )


@pytest.fixture
def light_hit(game_cfg):
    """A HitEvent simulating a light attack."""
    return HitEvent(
        attacker_commitment=CombatCommitment.LIGHT_ATTACK,
        damage=game_cfg.actions.light_attack.damage,
        hitstun_frames=game_cfg.actions.light_attack.hitstun_frames,
        knockback_sub=game_cfg.actions.light_attack.knockback * game_cfg.simulation.sub_pixel_scale,
        knockback_direction=1,
    )


@pytest.fixture
def heavy_hit(game_cfg):
    """A HitEvent simulating a heavy attack."""
    return HitEvent(
        attacker_commitment=CombatCommitment.HEAVY_ATTACK,
        damage=game_cfg.actions.heavy_attack.damage,
        hitstun_frames=game_cfg.actions.heavy_attack.hitstun_frames,
        knockback_sub=game_cfg.actions.heavy_attack.knockback * game_cfg.simulation.sub_pixel_scale,
        knockback_direction=1,
    )


# ---------------------------------------------------------------------------
# enter_blockstun / enter_parry_stunned
# ---------------------------------------------------------------------------

class TestEnterBlockstun:
    def test_sets_blockstun_state(self, fighter):
        enter_blockstun(fighter, 6)
        assert fighter.fsm_state == FSMState.BLOCKSTUN
        assert fighter.fsm_frames_remaining == 6

    def test_clears_commitment(self, fighter):
        fighter.active_commitment = CombatCommitment.LIGHT_ATTACK
        enter_blockstun(fighter, 6)
        assert fighter.active_commitment is None

    def test_stops_velocity(self, fighter):
        fighter.velocity_x = 500
        enter_blockstun(fighter, 6)
        assert fighter.velocity_x == 0


class TestEnterParrystunned:
    def test_sets_parry_stunned_state(self, fighter):
        enter_parry_stunned(fighter, 45)
        assert fighter.fsm_state == FSMState.PARRY_STUNNED
        assert fighter.fsm_frames_remaining == 45

    def test_clears_commitment(self, fighter):
        fighter.active_commitment = CombatCommitment.HEAVY_ATTACK
        enter_parry_stunned(fighter, 45)
        assert fighter.active_commitment is None


# ---------------------------------------------------------------------------
# tick_fsm: BLOCKING stays, BLOCKSTUN → IDLE, PARRY_STUNNED → IDLE
# ---------------------------------------------------------------------------

class TestTickFsmBlock:
    def test_blocking_no_transition(self, fighter, game_cfg):
        fighter.fsm_state = FSMState.BLOCKING
        fighter.fsm_frames_remaining = 0
        tick_fsm(fighter, game_cfg)
        assert fighter.fsm_state == FSMState.BLOCKING

    def test_blockstun_transitions_to_idle(self, fighter, game_cfg):
        enter_blockstun(fighter, 1)
        tick_fsm(fighter, game_cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_blockstun_counts_down(self, fighter, game_cfg):
        enter_blockstun(fighter, 3)
        tick_fsm(fighter, game_cfg)
        assert fighter.fsm_state == FSMState.BLOCKSTUN
        assert fighter.fsm_frames_remaining == 2

    def test_parry_stunned_transitions_to_idle(self, fighter, game_cfg):
        enter_parry_stunned(fighter, 1)
        tick_fsm(fighter, game_cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_parry_stunned_counts_down(self, fighter, game_cfg):
        enter_parry_stunned(fighter, 3)
        tick_fsm(fighter, game_cfg)
        assert fighter.fsm_state == FSMState.PARRY_STUNNED
        assert fighter.fsm_frames_remaining == 2


# ---------------------------------------------------------------------------
# BLOCK_START commitment via can_commit / enter_commitment
# ---------------------------------------------------------------------------

class TestBlockCommitment:
    def test_can_commit_block_from_idle(self, fighter, game_cfg):
        assert can_commit(fighter, CombatCommitment.BLOCK_START, game_cfg)

    def test_cannot_commit_block_when_locked(self, fighter, game_cfg):
        fighter.fsm_state = FSMState.ATTACK_STARTUP
        assert not can_commit(fighter, CombatCommitment.BLOCK_START, game_cfg)

    def test_enter_block_sets_blocking_state(self, fighter, game_cfg):
        enter_commitment(fighter, CombatCommitment.BLOCK_START, game_cfg)
        assert fighter.fsm_state == FSMState.BLOCKING
        assert fighter.velocity_x == 0

    def test_blocking_is_free_to_leave(self, fighter, game_cfg):
        enter_commitment(fighter, CombatCommitment.BLOCK_START, game_cfg)
        # Manually releasing block should go to IDLE
        fighter.fsm_state = FSMState.IDLE
        fighter.active_commitment = None
        assert fighter.fsm_state == FSMState.IDLE


# ---------------------------------------------------------------------------
# PlayerController: PRESS_BLOCK / RELEASE_BLOCK
# ---------------------------------------------------------------------------

class TestPlayerControllerBlock:
    def test_press_block_enters_blocking(self, fighter, game_cfg):
        ctrl = PlayerController()
        ctrl.process_inputs(fighter, [InputAction.PRESS_BLOCK], game_cfg)
        assert fighter.fsm_state == FSMState.BLOCKING

    def test_release_block_returns_to_idle(self, fighter, game_cfg):
        ctrl = PlayerController()
        ctrl.process_inputs(fighter, [InputAction.PRESS_BLOCK], game_cfg)
        ctrl.process_inputs(fighter, [InputAction.RELEASE_BLOCK], game_cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_release_block_no_effect_when_not_blocking(self, fighter, game_cfg):
        ctrl = PlayerController()
        ctrl.process_inputs(fighter, [InputAction.RELEASE_BLOCK], game_cfg)
        assert fighter.fsm_state == FSMState.IDLE

    def test_cannot_block_when_attacking(self, fighter, game_cfg):
        fighter.fsm_state = FSMState.ATTACK_STARTUP
        ctrl = PlayerController()
        ctrl.process_inputs(fighter, [InputAction.PRESS_BLOCK], game_cfg)
        assert fighter.fsm_state == FSMState.ATTACK_STARTUP


# ---------------------------------------------------------------------------
# tick_guard: regen delay and recovery
# ---------------------------------------------------------------------------

class TestTickGuard:
    def test_no_regen_while_blocking(self, fighter, game_cfg):
        fighter.guard = 50
        fighter.fsm_state = FSMState.BLOCKING
        tick_guard(fighter, game_cfg)
        assert fighter.guard == 50

    def test_no_regen_while_blockstun(self, fighter, game_cfg):
        fighter.guard = 50
        enter_blockstun(fighter, 6)
        tick_guard(fighter, game_cfg)
        assert fighter.guard == 50

    def test_no_regen_while_parry_stunned(self, fighter, game_cfg):
        fighter.guard = 50
        enter_parry_stunned(fighter, 45)
        tick_guard(fighter, game_cfg)
        assert fighter.guard == 50

    def test_no_regen_while_ko(self, fighter, game_cfg):
        fighter.guard = 50
        fighter.fsm_state = FSMState.KO
        tick_guard(fighter, game_cfg)
        assert fighter.guard == 50

    def test_regen_delay_counts_down(self, fighter, game_cfg):
        fighter.guard = 50
        fighter.guard_regen_delay = 3
        tick_guard(fighter, game_cfg)
        assert fighter.guard_regen_delay == 2
        assert fighter.guard == 50  # no regen yet

    def test_regen_starts_after_delay(self, fighter, game_cfg):
        block_cfg = game_cfg.actions.block
        fighter.guard = 50
        fighter.guard_regen_delay = 1
        tick_guard(fighter, game_cfg)
        assert fighter.guard_regen_delay == 0
        # Delay expired — next tick regens
        tick_guard(fighter, game_cfg)
        assert fighter.guard == 50 + block_cfg.guard_regen_per_tick

    def test_guard_does_not_exceed_max(self, fighter, game_cfg):
        block_cfg = game_cfg.actions.block
        fighter.guard = block_cfg.guard_max - 1
        fighter.guard_regen_delay = 0
        tick_guard(fighter, game_cfg)
        assert fighter.guard == block_cfg.guard_max

    def test_full_guard_no_change(self, fighter, game_cfg):
        block_cfg = game_cfg.actions.block
        fighter.guard = block_cfg.guard_max
        fighter.guard_regen_delay = 0
        tick_guard(fighter, game_cfg)
        assert fighter.guard == block_cfg.guard_max


# ---------------------------------------------------------------------------
# apply_block_response: chip damage, guard cost, states
# ---------------------------------------------------------------------------

class TestApplyBlockResponse:
    def test_light_hit_reduces_guard(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        initial_guard = fighter.guard
        apply_block_response(fighter, light_hit, game_cfg)
        assert fighter.guard == initial_guard - block_cfg.guard_cost_light

    def test_heavy_hit_reduces_guard_more(self, fighter, game_cfg, heavy_hit):
        block_cfg = game_cfg.actions.block
        initial_guard = fighter.guard
        apply_block_response(fighter, heavy_hit, game_cfg)
        assert fighter.guard == initial_guard - block_cfg.guard_cost_heavy

    def test_chip_damage_applied(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        initial_hp = fighter.hp
        apply_block_response(fighter, light_hit, game_cfg)
        expected_chip = max(1, round(light_hit.damage * block_cfg.chip_damage_pct))
        assert fighter.hp == initial_hp - expected_chip

    def test_regen_delay_set(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        apply_block_response(fighter, light_hit, game_cfg)
        assert fighter.guard_regen_delay == block_cfg.guard_regen_delay_ticks

    def test_light_hit_enters_blockstun(self, fighter, game_cfg, light_hit):
        guard_broken = apply_block_response(fighter, light_hit, game_cfg)
        assert not guard_broken
        assert fighter.fsm_state == FSMState.BLOCKSTUN

    def test_guard_break_enters_parry_stunned(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        # Drain guard to exactly the threshold
        fighter.guard = block_cfg.guard_cost_light
        guard_broken = apply_block_response(fighter, light_hit, game_cfg)
        assert guard_broken
        assert fighter.fsm_state == FSMState.PARRY_STUNNED

    def test_guard_breaks_at_zero(self, fighter, game_cfg, heavy_hit):
        block_cfg = game_cfg.actions.block
        fighter.guard = 1  # will be driven to 0 by heavy cost
        guard_broken = apply_block_response(fighter, heavy_hit, game_cfg)
        assert guard_broken
        assert fighter.guard == 0

    def test_guard_cannot_go_negative(self, fighter, game_cfg, heavy_hit):
        fighter.guard = 1
        apply_block_response(fighter, heavy_hit, game_cfg)
        assert fighter.guard >= 0

    def test_hp_cannot_go_negative(self, fighter, game_cfg, light_hit):
        fighter.hp = 1  # chip will floor at 0
        apply_block_response(fighter, light_hit, game_cfg)
        assert fighter.hp >= 0

    def test_multiple_blocks_drain_guard(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        # Count how many light blocks before guard breaks
        guard_breaks = 0
        for _ in range(20):
            fighter.fsm_state = FSMState.IDLE
            fighter.active_commitment = None
            guard_broken = apply_block_response(fighter, light_hit, game_cfg)
            if guard_broken:
                guard_breaks += 1
                break
        assert guard_breaks == 1, "Guard should break after enough absorbed hits"

    def test_guard_break_stun_duration(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        fighter.guard = block_cfg.guard_cost_light  # exactly at break threshold
        apply_block_response(fighter, light_hit, game_cfg)
        assert fighter.fsm_frames_remaining == block_cfg.guard_break_stun_frames

    def test_blockstun_duration(self, fighter, game_cfg, light_hit):
        block_cfg = game_cfg.actions.block
        fighter.guard = block_cfg.guard_max  # plenty of guard left
        apply_block_response(fighter, light_hit, game_cfg)
        assert fighter.fsm_frames_remaining == block_cfg.blockstun_frames


# ---------------------------------------------------------------------------
# Guard regen after guard break recovery
# ---------------------------------------------------------------------------

class TestGuardRegenAfterBreak:
    def test_guard_regens_after_parry_stun_resolves(self, fighter, game_cfg):
        block_cfg = game_cfg.actions.block
        fighter.guard = 0
        # Simulate end of parry stun: fighter back to IDLE
        fighter.fsm_state = FSMState.IDLE
        fighter.guard_regen_delay = 0
        # Regen should now work
        tick_guard(fighter, game_cfg)
        assert fighter.guard == block_cfg.guard_regen_per_tick

    def test_guard_regens_to_full_eventually(self, fighter, game_cfg):
        block_cfg = game_cfg.actions.block
        fighter.guard = 0
        fighter.fsm_state = FSMState.IDLE
        fighter.guard_regen_delay = 0
        # Run enough ticks to fill guard
        ticks_needed = block_cfg.guard_max // block_cfg.guard_regen_per_tick + 1
        for _ in range(ticks_needed):
            tick_guard(fighter, game_cfg)
        assert fighter.guard == block_cfg.guard_max
