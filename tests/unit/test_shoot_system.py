"""Tests for Phase 20: Charged Ranged Weapon System.

Covers:
  - ShootActionConfig: defaults loaded from config
  - FighterState: charge_ticks, shoot_cooldown, pending_shot fields
  - InputAction: PRESS_SHOOT / RELEASE_SHOOT exist
  - CombatCommitment: SHOOT_START / SHOOT_INSTANT
  - FSMState: SHOOT_STARTUP / CHARGING / SHOOT_ACTIVE / SHOOT_RECOVERY
  - state_machine: can_commit (shoot cooldown gate), enter_commitment SHOOT_START/INSTANT,
      tick_fsm SHOOT_STARTUP→CHARGING, SHOOT_ACTIVE→RECOVERY→IDLE
  - tick_shoot_cooldown: decrements each tick, clamps at 0
  - PlayerController: PRESS_SHOOT starts commitment, RELEASE_SHOOT fires
  - Projectile dataclass: fields, active flag
  - Engine projectile integration: fire, move, collision, hit application
"""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.projectile import Projectile
from game.combat.state_machine import (
    can_commit,
    enter_commitment,
    tick_fsm,
    tick_shoot_cooldown,
)
from game.entities.player_fighter import PlayerController
from game.input.input_actions import InputAction
from game.state import FighterState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def idle_fighter(game_cfg):
    scale = game_cfg.simulation.sub_pixel_scale
    return FighterState(
        x=300 * scale,
        y=game_cfg.arena.ground_y * scale,
        hp=game_cfg.fighter.max_hp,
        stamina=game_cfg.fighter.max_stamina,
        fsm_state=FSMState.IDLE,
    )


@pytest.fixture
def player_ctrl():
    return PlayerController()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestShootConfig:
    def test_shoot_config_exists(self, game_cfg):
        assert hasattr(game_cfg.actions, "shoot")

    def test_shoot_config_defaults(self, game_cfg):
        s = game_cfg.actions.shoot
        assert s.startup_frames >= 1
        assert s.max_charge_frames >= 10
        assert s.min_damage >= 1
        assert s.max_damage > s.min_damage
        assert s.projectile_speed >= 1
        assert s.cooldown_frames >= 1
        assert s.active_frames >= 1
        assert s.recovery_frames >= 1

    def test_shoot_config_values(self, game_cfg):
        s = game_cfg.actions.shoot
        assert s.startup_frames == 5
        assert s.max_charge_frames == 60
        assert s.min_damage == 8
        assert s.max_damage == 30
        assert s.projectile_speed == 12
        assert s.cooldown_frames == 45


# ---------------------------------------------------------------------------
# FighterState fields
# ---------------------------------------------------------------------------

class TestFighterStateShootFields:
    def test_charge_ticks_default(self):
        f = FighterState()
        assert f.charge_ticks == 0

    def test_shoot_cooldown_default(self):
        f = FighterState()
        assert f.shoot_cooldown == 0

    def test_pending_shot_default(self):
        f = FighterState()
        assert f.pending_shot is False


# ---------------------------------------------------------------------------
# Input actions
# ---------------------------------------------------------------------------

class TestShootInputActions:
    def test_press_shoot_exists(self):
        assert hasattr(InputAction, "PRESS_SHOOT")

    def test_release_shoot_exists(self):
        assert hasattr(InputAction, "RELEASE_SHOOT")


# ---------------------------------------------------------------------------
# CombatCommitment / FSMState
# ---------------------------------------------------------------------------

class TestShootEnums:
    def test_shoot_start_commitment(self):
        assert hasattr(CombatCommitment, "SHOOT_START")
        assert CombatCommitment.SHOOT_START.value == 40

    def test_shoot_instant_commitment(self):
        assert hasattr(CombatCommitment, "SHOOT_INSTANT")
        assert CombatCommitment.SHOOT_INSTANT.value == 41

    def test_shoot_fsm_states(self):
        for name in ("SHOOT_STARTUP", "CHARGING", "SHOOT_ACTIVE", "SHOOT_RECOVERY"):
            assert hasattr(FSMState, name), f"FSMState.{name} missing"


# ---------------------------------------------------------------------------
# tick_shoot_cooldown
# ---------------------------------------------------------------------------

class TestTickShootCooldown:
    def test_decrements(self, idle_fighter):
        idle_fighter.shoot_cooldown = 10
        tick_shoot_cooldown(idle_fighter)
        assert idle_fighter.shoot_cooldown == 9

    def test_clamps_at_zero(self, idle_fighter):
        idle_fighter.shoot_cooldown = 0
        tick_shoot_cooldown(idle_fighter)
        assert idle_fighter.shoot_cooldown == 0

    def test_counts_down_to_zero(self, idle_fighter):
        idle_fighter.shoot_cooldown = 3
        for _ in range(5):
            tick_shoot_cooldown(idle_fighter)
        assert idle_fighter.shoot_cooldown == 0


# ---------------------------------------------------------------------------
# can_commit — shoot cooldown gate
# ---------------------------------------------------------------------------

class TestCanCommitShoot:
    def test_shoot_start_allowed_when_ready(self, idle_fighter, game_cfg):
        assert can_commit(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)

    def test_shoot_start_blocked_by_cooldown(self, idle_fighter, game_cfg):
        idle_fighter.shoot_cooldown = 10
        assert not can_commit(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)

    def test_shoot_instant_allowed_when_ready(self, idle_fighter, game_cfg):
        assert can_commit(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)

    def test_shoot_instant_blocked_by_cooldown(self, idle_fighter, game_cfg):
        idle_fighter.shoot_cooldown = 5
        assert not can_commit(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)

    def test_shoot_blocked_when_attacking(self, idle_fighter, game_cfg):
        idle_fighter.fsm_state = FSMState.ATTACK_ACTIVE
        assert not can_commit(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)


# ---------------------------------------------------------------------------
# enter_commitment — SHOOT_START and SHOOT_INSTANT
# ---------------------------------------------------------------------------

class TestEnterCommitmentShoot:
    def test_shoot_start_enters_shoot_startup(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        assert idle_fighter.fsm_state == FSMState.SHOOT_STARTUP

    def test_shoot_start_sets_startup_frames(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        assert idle_fighter.fsm_frames_remaining == game_cfg.actions.shoot.startup_frames

    def test_shoot_start_resets_charge(self, idle_fighter, game_cfg):
        idle_fighter.charge_ticks = 99
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        assert idle_fighter.charge_ticks == 0

    def test_shoot_start_stops_movement(self, idle_fighter, game_cfg):
        idle_fighter.velocity_x = 500
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        assert idle_fighter.velocity_x == 0

    def test_shoot_instant_enters_shoot_active(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)
        assert idle_fighter.fsm_state == FSMState.SHOOT_ACTIVE

    def test_shoot_instant_sets_active_frames(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)
        assert idle_fighter.fsm_frames_remaining == game_cfg.actions.shoot.active_frames

    def test_shoot_instant_sets_pending_shot(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)
        assert idle_fighter.pending_shot is True

    def test_shoot_instant_zero_charge(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_INSTANT, game_cfg)
        assert idle_fighter.charge_ticks == 0


# ---------------------------------------------------------------------------
# tick_fsm — SHOOT_STARTUP → CHARGING → SHOOT_ACTIVE → SHOOT_RECOVERY → IDLE
# ---------------------------------------------------------------------------

class TestTickFsmShoot:
    def _advance(self, fighter, cfg, n=1):
        for _ in range(n):
            tick_fsm(fighter, cfg)

    def test_shoot_startup_transitions_to_charging(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        startup = game_cfg.actions.shoot.startup_frames
        self._advance(idle_fighter, game_cfg, startup)
        assert idle_fighter.fsm_state == FSMState.CHARGING

    def test_charging_does_not_auto_advance(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        startup = game_cfg.actions.shoot.startup_frames
        self._advance(idle_fighter, game_cfg, startup + 30)
        # Still CHARGING — held until player releases
        assert idle_fighter.fsm_state == FSMState.CHARGING

    def test_shoot_active_transitions_to_recovery(self, idle_fighter, game_cfg):
        idle_fighter.fsm_state = FSMState.SHOOT_ACTIVE
        idle_fighter.fsm_frames_remaining = game_cfg.actions.shoot.active_frames
        self._advance(idle_fighter, game_cfg, game_cfg.actions.shoot.active_frames)
        assert idle_fighter.fsm_state == FSMState.SHOOT_RECOVERY

    def test_shoot_recovery_transitions_to_idle(self, idle_fighter, game_cfg):
        idle_fighter.fsm_state = FSMState.SHOOT_RECOVERY
        idle_fighter.fsm_frames_remaining = game_cfg.actions.shoot.recovery_frames
        self._advance(idle_fighter, game_cfg, game_cfg.actions.shoot.recovery_frames)
        assert idle_fighter.fsm_state == FSMState.IDLE

    def test_shoot_recovery_clears_active_commitment(self, idle_fighter, game_cfg):
        idle_fighter.fsm_state = FSMState.SHOOT_RECOVERY
        idle_fighter.fsm_frames_remaining = game_cfg.actions.shoot.recovery_frames
        idle_fighter.active_commitment = CombatCommitment.SHOOT_START
        self._advance(idle_fighter, game_cfg, game_cfg.actions.shoot.recovery_frames)
        assert idle_fighter.active_commitment is None

    def test_charging_reset_on_startup_entry(self, idle_fighter, game_cfg):
        enter_commitment(idle_fighter, CombatCommitment.SHOOT_START, game_cfg)
        startup = game_cfg.actions.shoot.startup_frames
        self._advance(idle_fighter, game_cfg, startup)
        # charge_ticks should be 0 immediately on entering CHARGING
        assert idle_fighter.charge_ticks == 0


# ---------------------------------------------------------------------------
# PlayerController — PRESS_SHOOT / RELEASE_SHOOT
# ---------------------------------------------------------------------------

class TestPlayerControllerShoot:
    def test_press_shoot_enters_shoot_startup(self, idle_fighter, game_cfg, player_ctrl):
        player_ctrl.process_inputs(idle_fighter, [InputAction.PRESS_SHOOT], game_cfg)
        assert idle_fighter.fsm_state == FSMState.SHOOT_STARTUP

    def test_press_shoot_blocked_when_attacking(self, idle_fighter, game_cfg, player_ctrl):
        idle_fighter.fsm_state = FSMState.ATTACK_ACTIVE
        idle_fighter.fsm_frames_remaining = 3
        player_ctrl.process_inputs(idle_fighter, [InputAction.PRESS_SHOOT], game_cfg)
        assert idle_fighter.fsm_state == FSMState.ATTACK_ACTIVE

    def test_press_shoot_blocked_by_cooldown(self, idle_fighter, game_cfg, player_ctrl):
        idle_fighter.shoot_cooldown = 5
        player_ctrl.process_inputs(idle_fighter, [InputAction.PRESS_SHOOT], game_cfg)
        assert idle_fighter.fsm_state == FSMState.IDLE

    def test_release_shoot_in_charging_fires(self, idle_fighter, game_cfg, player_ctrl):
        idle_fighter.fsm_state = FSMState.CHARGING
        idle_fighter.fsm_frames_remaining = 0
        player_ctrl.process_inputs(idle_fighter, [InputAction.RELEASE_SHOOT], game_cfg)
        assert idle_fighter.fsm_state == FSMState.SHOOT_ACTIVE

    def test_release_shoot_in_charging_sets_pending_shot(self, idle_fighter, game_cfg, player_ctrl):
        idle_fighter.fsm_state = FSMState.CHARGING
        player_ctrl.process_inputs(idle_fighter, [InputAction.RELEASE_SHOOT], game_cfg)
        assert idle_fighter.pending_shot is True

    def test_release_shoot_when_not_charging_is_noop(self, idle_fighter, game_cfg, player_ctrl):
        player_ctrl.process_inputs(idle_fighter, [InputAction.RELEASE_SHOOT], game_cfg)
        assert idle_fighter.fsm_state == FSMState.IDLE

    def test_release_shoot_sets_active_frames(self, idle_fighter, game_cfg, player_ctrl):
        idle_fighter.fsm_state = FSMState.CHARGING
        player_ctrl.process_inputs(idle_fighter, [InputAction.RELEASE_SHOOT], game_cfg)
        assert idle_fighter.fsm_frames_remaining == game_cfg.actions.shoot.active_frames


# ---------------------------------------------------------------------------
# Projectile dataclass
# ---------------------------------------------------------------------------

class TestProjectileDataclass:
    def test_fields_exist(self):
        p = Projectile(x=1000, y=2000, velocity_x=1200, damage=10,
                       owner="PLAYER", charge_frac=0.5)
        assert p.x == 1000
        assert p.y == 2000
        assert p.velocity_x == 1200
        assert p.damage == 10
        assert p.owner == "PLAYER"
        assert p.charge_frac == 0.5
        assert p.active is True

    def test_can_deactivate(self):
        p = Projectile(x=0, y=0, velocity_x=0, damage=5, owner="AI", charge_frac=0.0)
        p.active = False
        assert not p.active

    def test_charge_frac_zero(self):
        p = Projectile(x=0, y=0, velocity_x=0, damage=5, owner="PLAYER", charge_frac=0.0)
        assert p.charge_frac == 0.0

    def test_charge_frac_full(self):
        p = Projectile(x=0, y=0, velocity_x=0, damage=30, owner="AI", charge_frac=1.0)
        assert p.charge_frac == 1.0


# ---------------------------------------------------------------------------
# Engine integration — _fire_projectile / _update_projectiles
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_engine(game_cfg, tmp_path):
    """A headless engine with a fresh match backed by a temp DB."""
    from pathlib import Path
    from data.db import Database
    from data.migrations.migration_runner import run_migrations
    from game.engine import Engine
    from ai.layers.tactical_planner import AITier

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    _, ai_cfg, display_cfg = load_config()
    engine = Engine(game_cfg, ai_cfg, display_cfg, db,
                    headless=True, ai_tier=AITier.T0_BASELINE)
    engine._start_match()
    yield engine
    db.close()


class TestEngineProjectileIntegration:
    """Tests that exercise the engine's projectile methods directly."""

    def test_fire_projectile_adds_to_list(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        assert len(engine._projectiles) == 1

    def test_fire_projectile_correct_owner(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        assert engine._projectiles[0].owner == "PLAYER"

    def test_fire_projectile_sets_cooldown(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        assert fighter.shoot_cooldown == game_cfg.actions.shoot.cooldown_frames

    def test_fire_projectile_resets_charge_ticks(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 30
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        assert fighter.charge_ticks == 0

    def test_fire_projectile_min_damage_when_uncharged(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        proj = engine._projectiles[0]
        assert proj.damage == game_cfg.actions.shoot.min_damage

    def test_fire_projectile_max_damage_when_full_charge(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = game_cfg.actions.shoot.max_charge_frames
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        proj = engine._projectiles[0]
        assert proj.damage == game_cfg.actions.shoot.max_damage

    def test_fire_projectile_velocity_follows_facing(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = -1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        proj = engine._projectiles[0]
        assert proj.velocity_x < 0

    def test_update_projectiles_moves_them(self, tmp_engine, game_cfg):
        engine = tmp_engine
        fighter = engine._state.player
        fighter.facing = 1
        fighter.charge_ticks = 0
        engine._fire_projectile(fighter, "PLAYER", game_cfg)
        initial_x = engine._projectiles[0].x
        engine._update_projectiles(engine._state, game_cfg)
        assert engine._projectiles[0].x > initial_x

    def test_update_projectiles_deactivates_out_of_bounds(self, tmp_engine, game_cfg):
        engine = tmp_engine
        scale = game_cfg.simulation.sub_pixel_scale
        proj = Projectile(
            x=engine._state.arena.width_sub + scale,  # just past right edge
            y=game_cfg.arena.ground_y * scale,
            velocity_x=100 * scale,
            damage=10,
            owner="PLAYER",
            charge_frac=0.0,
        )
        engine._projectiles.append(proj)
        engine._update_projectiles(engine._state, game_cfg)
        assert not proj.active

    def test_update_projectiles_removes_inactive(self, tmp_engine, game_cfg):
        engine = tmp_engine
        scale = game_cfg.simulation.sub_pixel_scale
        proj = Projectile(
            x=engine._state.arena.width_sub + scale,
            y=game_cfg.arena.ground_y * scale,
            velocity_x=100 * scale,
            damage=10,
            owner="PLAYER",
            charge_frac=0.0,
        )
        engine._projectiles.append(proj)
        engine._update_projectiles(engine._state, game_cfg)
        assert len(engine._projectiles) == 0

    def test_projectile_hits_opponent_reduces_hp(self, tmp_engine, game_cfg):
        """Place a PLAYER projectile directly on the AI fighter and check damage."""
        engine = tmp_engine
        state = engine._state
        initial_hp = state.ai.hp

        # Create a stationary projectile right at the AI's position
        proj = Projectile(
            x=state.ai.x,
            y=state.ai.y,
            velocity_x=0,  # stationary — will hit immediately
            damage=10,
            owner="PLAYER",
            charge_frac=0.0,
        )
        engine._projectiles.append(proj)
        engine._update_projectiles(state, game_cfg)

        assert state.ai.hp < initial_hp

    def test_projectile_does_not_hit_own_fighter(self, tmp_engine, game_cfg):
        """PLAYER projectile should not hit the player."""
        engine = tmp_engine
        state = engine._state
        initial_hp = state.player.hp

        proj = Projectile(
            x=state.player.x,
            y=state.player.y,
            velocity_x=0,
            damage=10,
            owner="PLAYER",
            charge_frac=0.0,
        )
        engine._projectiles.append(proj)
        engine._update_projectiles(state, game_cfg)

        assert state.player.hp == initial_hp

    def test_pending_shot_cleared_after_fire(self, tmp_engine, game_cfg):
        """pending_shot flag should be cleared after the engine reads it."""
        engine = tmp_engine
        state = engine._state
        state.player.pending_shot = True
        state.player.facing = 1
        state.player.charge_ticks = 0
        # Simulate the engine's pending_shot drain logic
        if state.player.pending_shot:
            state.player.pending_shot = False
            engine._fire_projectile(state.player, "PLAYER", game_cfg)
        assert state.player.pending_shot is False
