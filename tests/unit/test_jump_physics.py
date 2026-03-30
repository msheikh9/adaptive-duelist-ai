"""Tests: Phase 15 jump / verticality mechanics."""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gcfg():
    from config.config_loader import load_config
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def arena(gcfg):
    from game.state import ArenaState
    scale = gcfg.simulation.sub_pixel_scale
    return ArenaState.from_config(
        gcfg.arena.width, gcfg.arena.height, gcfg.arena.ground_y, scale
    )


@pytest.fixture
def ground_fighter(gcfg, arena):
    """A fighter standing at ground level in IDLE."""
    from game.state import FighterState
    return FighterState(
        x=arena.width_sub // 2,
        y=arena.ground_y_sub,
        hp=gcfg.fighter.max_hp,
        stamina=gcfg.fighter.max_stamina,
        facing=1,
    )


# ---------------------------------------------------------------------------
# FighterState additions
# ---------------------------------------------------------------------------

class TestFighterStateAdditions:

    def test_velocity_y_default_zero(self, ground_fighter):
        assert ground_fighter.velocity_y == 0

    def test_is_airborne_false_on_ground(self, ground_fighter):
        assert ground_fighter.is_airborne is False

    def test_is_airborne_true_when_airborne(self, ground_fighter):
        from game.combat.actions import FSMState
        ground_fighter.fsm_state = FSMState.AIRBORNE
        assert ground_fighter.is_airborne is True

    def test_is_airborne_true_when_jump_startup(self, ground_fighter):
        from game.combat.actions import FSMState
        ground_fighter.fsm_state = FSMState.JUMP_STARTUP
        assert ground_fighter.is_airborne is True

    def test_is_airborne_true_when_landing(self, ground_fighter):
        from game.combat.actions import FSMState
        ground_fighter.fsm_state = FSMState.LANDING
        assert ground_fighter.is_airborne is True


# ---------------------------------------------------------------------------
# FSMState ordinals fit in uint8 (replay packing)
# ---------------------------------------------------------------------------

class TestFSMStateOrdinals:

    def test_all_fsm_ordinals_fit_uint8(self):
        from game.combat.actions import FSMState
        for s in FSMState:
            assert 0 <= s.value <= 255, f"{s.name} ordinal {s.value} exceeds uint8"

    def test_jump_startup_ordinal(self):
        from game.combat.actions import FSMState
        assert FSMState.JUMP_STARTUP.value == 15

    def test_airborne_ordinal(self):
        from game.combat.actions import FSMState
        assert FSMState.AIRBORNE.value == 16

    def test_landing_ordinal(self):
        from game.combat.actions import FSMState
        assert FSMState.LANDING.value == 17


# ---------------------------------------------------------------------------
# CombatCommitment.JUMP
# ---------------------------------------------------------------------------

class TestJumpCommitment:

    def test_jump_value_is_30(self):
        from game.combat.actions import CombatCommitment
        assert CombatCommitment.JUMP.value == 30

    def test_jump_not_in_phase1_commitments(self):
        from game.combat.actions import CombatCommitment, PHASE_1_COMMITMENTS
        assert CombatCommitment.JUMP not in PHASE_1_COMMITMENTS

    def test_jump_valid_from_idle(self, ground_fighter, gcfg):
        from game.combat.state_machine import can_commit
        from game.combat.actions import CombatCommitment
        assert can_commit(ground_fighter, CombatCommitment.JUMP, gcfg) is True

    def test_jump_invalid_when_airborne(self, ground_fighter, gcfg):
        from game.combat.state_machine import can_commit
        from game.combat.actions import CombatCommitment, FSMState
        ground_fighter.fsm_state = FSMState.AIRBORNE
        assert can_commit(ground_fighter, CombatCommitment.JUMP, gcfg) is False

    def test_jump_invalid_when_jump_startup(self, ground_fighter, gcfg):
        from game.combat.state_machine import can_commit
        from game.combat.actions import CombatCommitment, FSMState
        ground_fighter.fsm_state = FSMState.JUMP_STARTUP
        assert can_commit(ground_fighter, CombatCommitment.JUMP, gcfg) is False

    def test_jump_invalid_when_in_attack(self, ground_fighter, gcfg):
        from game.combat.state_machine import can_commit
        from game.combat.actions import CombatCommitment, FSMState
        ground_fighter.fsm_state = FSMState.ATTACK_STARTUP
        assert can_commit(ground_fighter, CombatCommitment.JUMP, gcfg) is False

    def test_jump_has_no_stamina_cost(self, gcfg):
        from game.combat.state_machine import get_stamina_cost
        from game.combat.actions import CombatCommitment
        assert get_stamina_cost(CombatCommitment.JUMP, gcfg) is None

    def test_enter_jump_sets_jump_startup(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_commitment
        from game.combat.actions import CombatCommitment, FSMState
        enter_commitment(ground_fighter, CombatCommitment.JUMP, gcfg)
        assert ground_fighter.fsm_state == FSMState.JUMP_STARTUP
        assert ground_fighter.fsm_frames_remaining == gcfg.fighter.jump_startup_frames


# ---------------------------------------------------------------------------
# FSM transitions: JUMP_STARTUP → AIRBORNE → landing
# ---------------------------------------------------------------------------

class TestJumpFSMTransitions:

    def test_jump_startup_transitions_to_airborne(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_commitment, tick_fsm
        from game.combat.actions import CombatCommitment, FSMState

        enter_commitment(ground_fighter, CombatCommitment.JUMP, gcfg)
        assert ground_fighter.fsm_state == FSMState.JUMP_STARTUP

        # Tick through startup frames
        for _ in range(gcfg.fighter.jump_startup_frames):
            tick_fsm(ground_fighter, gcfg)

        assert ground_fighter.fsm_state == FSMState.AIRBORNE

    def test_airborne_velocity_set_on_launch(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_commitment, tick_fsm
        from game.combat.actions import CombatCommitment, FSMState

        enter_commitment(ground_fighter, CombatCommitment.JUMP, gcfg)
        for _ in range(gcfg.fighter.jump_startup_frames):
            tick_fsm(ground_fighter, gcfg)

        expected_vy = -(gcfg.fighter.jump_velocity * gcfg.simulation.sub_pixel_scale)
        assert ground_fighter.velocity_y == expected_vy

    def test_airborne_stays_airborne_in_tick_fsm(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_commitment, tick_fsm
        from game.combat.actions import CombatCommitment, FSMState

        enter_commitment(ground_fighter, CombatCommitment.JUMP, gcfg)
        for _ in range(gcfg.fighter.jump_startup_frames):
            tick_fsm(ground_fighter, gcfg)

        assert ground_fighter.fsm_state == FSMState.AIRBORNE
        # AIRBORNE is inert in tick_fsm (landing handled by physics)
        tick_fsm(ground_fighter, gcfg)
        assert ground_fighter.fsm_state == FSMState.AIRBORNE

    def test_enter_landing_sets_landing_state(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_landing
        from game.combat.actions import FSMState

        ground_fighter.fsm_state = FSMState.AIRBORNE
        enter_landing(ground_fighter, gcfg.fighter.landing_recovery_frames)
        assert ground_fighter.fsm_state == FSMState.LANDING
        assert ground_fighter.velocity_y == 0
        assert ground_fighter.fsm_frames_remaining == gcfg.fighter.landing_recovery_frames

    def test_landing_transitions_to_idle(self, ground_fighter, gcfg):
        from game.combat.state_machine import enter_landing, tick_fsm
        from game.combat.actions import FSMState

        ground_fighter.fsm_state = FSMState.AIRBORNE
        enter_landing(ground_fighter, gcfg.fighter.landing_recovery_frames)
        for _ in range(gcfg.fighter.landing_recovery_frames):
            tick_fsm(ground_fighter, gcfg)
        assert ground_fighter.fsm_state == FSMState.IDLE


# ---------------------------------------------------------------------------
# Physics: gravity and landing
# ---------------------------------------------------------------------------

class TestGravityPhysics:

    def test_gravity_applies_above_ground(self, ground_fighter, gcfg, arena):
        from game.combat.physics import apply_gravity
        from game.combat.actions import FSMState

        ground_fighter.fsm_state = FSMState.AIRBORNE
        ground_fighter.y = arena.ground_y_sub - 500  # above ground
        ground_fighter.velocity_y = 0

        apply_gravity(ground_fighter, arena, gcfg)

        expected = gcfg.fighter.jump_gravity * gcfg.simulation.sub_pixel_scale
        assert ground_fighter.velocity_y == expected

    def test_gravity_does_not_apply_at_ground(self, ground_fighter, gcfg, arena):
        from game.combat.physics import apply_gravity

        ground_fighter.velocity_y = 0
        apply_gravity(ground_fighter, arena, gcfg)
        assert ground_fighter.velocity_y == 0

    def test_apply_velocity_moves_y(self, ground_fighter, gcfg):
        from game.combat.physics import apply_velocity

        ground_fighter.velocity_y = -300
        y_before = ground_fighter.y
        apply_velocity(ground_fighter)
        assert ground_fighter.y == y_before - 300

    def test_apply_velocity_still_moves_x(self, ground_fighter, gcfg):
        from game.combat.physics import apply_velocity

        ground_fighter.velocity_x = 500
        x_before = ground_fighter.x
        apply_velocity(ground_fighter)
        assert ground_fighter.x == x_before + 500


class TestLandingDetection:

    def test_handle_landing_returns_true_on_descent(self, ground_fighter, gcfg, arena):
        from game.combat.physics import apply_velocity, handle_landing
        from game.combat.actions import FSMState

        ground_fighter.fsm_state = FSMState.AIRBORNE
        ground_fighter.y = arena.ground_y_sub - 1  # just above ground
        ground_fighter.velocity_y = 10             # moving downward

        apply_velocity(ground_fighter)
        result = handle_landing(ground_fighter, arena)
        assert result is True
        assert ground_fighter.y == arena.ground_y_sub
        assert ground_fighter.velocity_y == 0

    def test_handle_landing_returns_false_at_ground_level(self, ground_fighter, gcfg, arena):
        from game.combat.physics import handle_landing
        # Already on ground with no downward velocity
        result = handle_landing(ground_fighter, arena)
        assert result is False

    def test_handle_landing_returns_false_ascending(self, ground_fighter, gcfg, arena):
        from game.combat.physics import handle_landing
        from game.combat.actions import FSMState

        ground_fighter.fsm_state = FSMState.AIRBORNE
        ground_fighter.y = arena.ground_y_sub - 100
        ground_fighter.velocity_y = -500   # going up
        result = handle_landing(ground_fighter, arena)
        assert result is False

    def test_ceiling_clamp_prevents_negative_y(self, ground_fighter, gcfg, arena):
        from game.combat.physics import clamp_to_arena

        fighter_w_sub = gcfg.fighter.width * gcfg.simulation.sub_pixel_scale
        ground_fighter.y = -10
        ground_fighter.velocity_y = -100
        clamp_to_arena(ground_fighter, arena, fighter_w_sub)
        assert ground_fighter.y == 0
        assert ground_fighter.velocity_y == 0


# ---------------------------------------------------------------------------
# Full jump arc (integration via engine headless ticks)
# ---------------------------------------------------------------------------

class TestJumpArcIntegration:

    @pytest.fixture
    def engine_factory(self, tmp_path):
        from config.config_loader import load_config
        from data.db import Database
        from data.migrations.migration_runner import run_migrations

        game_cfg, ai_cfg, display_cfg = load_config()
        db_path = tmp_path / "jump_test.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        engines = []

        def _make():
            from game.engine import Engine
            from ai.layers.tactical_planner import AITier
            e = Engine(game_cfg, ai_cfg, display_cfg, db,
                       headless=True, ai_tier=AITier.T0_BASELINE)
            engines.append(e)
            return e

        yield _make
        db.close()

    def test_player_can_jump_and_land(self, engine_factory, gcfg, arena):
        """Player commits JUMP; after enough ticks the fighter lands back on ground."""
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment

        engine = engine_factory()
        engine._start_match()

        player = engine.state.player
        # Manually commit JUMP (bypasses input)
        enter_commitment(player, CombatCommitment.JUMP, gcfg)

        # Run ticks until back on ground (max 200 for safety)
        for _ in range(200):
            engine._run_headless_tick()
            if player.fsm_state == FSMState.IDLE and player.y == arena.ground_y_sub:
                break

        assert player.fsm_state == FSMState.IDLE
        assert player.y == arena.ground_y_sub
        assert player.velocity_y == 0

    def test_jump_peak_is_above_ground(self, engine_factory, gcfg, arena):
        """Fighter rises above ground during AIRBORNE phase."""
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment

        engine = engine_factory()
        engine._start_match()

        player = engine.state.player
        enter_commitment(player, CombatCommitment.JUMP, gcfg)

        min_y = player.y  # track the highest point (lowest y value)
        for _ in range(50):
            engine._run_headless_tick()
            min_y = min(min_y, player.y)

        assert min_y < arena.ground_y_sub, "Fighter never rose above ground"

    def test_no_double_jump(self, engine_factory, gcfg):
        """Attempting a second JUMP while AIRBORNE is rejected."""
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment, can_commit

        engine = engine_factory()
        engine._start_match()

        player = engine.state.player
        enter_commitment(player, CombatCommitment.JUMP, gcfg)
        # Tick into AIRBORNE
        for _ in range(gcfg.fighter.jump_startup_frames + 1):
            engine._run_headless_tick()

        assert player.fsm_state == FSMState.AIRBORNE
        # Second jump must be rejected
        assert can_commit(player, CombatCommitment.JUMP, gcfg) is False
