"""Tests: balancing changes — heavy cooldown, dodge tightening, combo corners."""

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
def engine_factory(tmp_path):
    from config.config_loader import load_config
    from data.db import Database
    from data.migrations.migration_runner import run_migrations

    game_cfg, ai_cfg, display_cfg = load_config()
    db_path = tmp_path / "balance_test.db"
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


@pytest.fixture
def renderer(gcfg):
    from config.config_loader import load_config
    _, _, display_cfg = load_config()
    from rendering.renderer import Renderer
    r = Renderer(gcfg, display_cfg)
    r.init()
    return r


# ---------------------------------------------------------------------------
# Config: heavy cooldown value exists and is loaded
# ---------------------------------------------------------------------------

class TestHeavyAttackCooldownConfig:

    def test_heavy_attack_config_has_cooldown_ticks(self, gcfg):
        assert hasattr(gcfg.actions.heavy_attack, "cooldown_ticks")

    def test_heavy_cooldown_ticks_positive(self, gcfg):
        assert gcfg.actions.heavy_attack.cooldown_ticks > 0

    def test_heavy_cooldown_default_at_least_30(self, gcfg):
        # Should be a meaningful delay, not a token value
        assert gcfg.actions.heavy_attack.cooldown_ticks >= 30

    def test_light_attack_cooldown_is_zero(self, gcfg):
        # Light attacks have no extra cooldown beyond their animation
        assert gcfg.actions.light_attack.cooldown_ticks == 0

    def test_heavy_cooldown_greater_than_total_animation(self, gcfg):
        # Cooldown must outlast the animation to create a real lockout window
        heavy = gcfg.actions.heavy_attack
        assert heavy.cooldown_ticks > heavy.total_frames


# ---------------------------------------------------------------------------
# State: heavy_cooldown field on FighterState
# ---------------------------------------------------------------------------

class TestHeavyCooldownState:

    def test_fighter_state_has_heavy_cooldown(self):
        from game.state import FighterState
        f = FighterState()
        assert hasattr(f, "heavy_cooldown")

    def test_heavy_cooldown_starts_zero(self):
        from game.state import FighterState
        assert FighterState().heavy_cooldown == 0


# ---------------------------------------------------------------------------
# state_machine: tick_heavy_cooldown
# ---------------------------------------------------------------------------

class TestTickHeavyCooldown:

    def test_tick_heavy_cooldown_decrements(self):
        from game.state import FighterState
        from game.combat.state_machine import tick_heavy_cooldown
        f = FighterState(heavy_cooldown=10)
        tick_heavy_cooldown(f)
        assert f.heavy_cooldown == 9

    def test_tick_heavy_cooldown_does_not_go_below_zero(self):
        from game.state import FighterState
        from game.combat.state_machine import tick_heavy_cooldown
        f = FighterState(heavy_cooldown=0)
        tick_heavy_cooldown(f)
        assert f.heavy_cooldown == 0

    def test_tick_heavy_cooldown_counts_to_zero(self):
        from game.state import FighterState
        from game.combat.state_machine import tick_heavy_cooldown
        f = FighterState(heavy_cooldown=3)
        for _ in range(3):
            tick_heavy_cooldown(f)
        assert f.heavy_cooldown == 0


# ---------------------------------------------------------------------------
# state_machine: can_commit blocks heavy while cooldown active
# ---------------------------------------------------------------------------

class TestHeavyCooldownEnforcement:

    def _free_fighter(self, gcfg):
        from game.state import FighterState
        from game.combat.actions import FSMState
        scale = gcfg.simulation.sub_pixel_scale
        return FighterState(
            x=300 * scale,
            y=gcfg.arena.ground_y * scale,
            hp=gcfg.fighter.max_hp,
            stamina=gcfg.fighter.max_stamina,
            fsm_state=FSMState.IDLE,
            facing=1,
        )

    def test_heavy_blocked_when_cooldown_active(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import can_commit
        f = self._free_fighter(gcfg)
        f.heavy_cooldown = 10
        assert not can_commit(f, CombatCommitment.HEAVY_ATTACK, gcfg)

    def test_heavy_allowed_when_cooldown_zero(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import can_commit
        f = self._free_fighter(gcfg)
        f.heavy_cooldown = 0
        assert can_commit(f, CombatCommitment.HEAVY_ATTACK, gcfg)

    def test_light_not_blocked_when_heavy_cooldown_active(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import can_commit
        f = self._free_fighter(gcfg)
        f.heavy_cooldown = 20
        assert can_commit(f, CombatCommitment.LIGHT_ATTACK, gcfg)

    def test_enter_heavy_sets_cooldown(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import enter_commitment
        f = self._free_fighter(gcfg)
        enter_commitment(f, CombatCommitment.HEAVY_ATTACK, gcfg)
        assert f.heavy_cooldown == gcfg.actions.heavy_attack.cooldown_ticks

    def test_enter_light_does_not_set_heavy_cooldown(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import enter_commitment
        f = self._free_fighter(gcfg)
        enter_commitment(f, CombatCommitment.LIGHT_ATTACK, gcfg)
        assert f.heavy_cooldown == 0

    def test_heavy_cooldown_greater_than_zero_after_first_use(self, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import enter_commitment
        f = self._free_fighter(gcfg)
        enter_commitment(f, CombatCommitment.HEAVY_ATTACK, gcfg)
        assert f.heavy_cooldown > 0

    def test_heavy_blocked_immediately_after_commitment(self, gcfg):
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment, can_commit
        f = self._free_fighter(gcfg)
        enter_commitment(f, CombatCommitment.HEAVY_ATTACK, gcfg)
        # Put back in FREE state (simulate recovery), cooldown still active
        f.fsm_state = FSMState.IDLE
        f.fsm_frames_remaining = 0
        f.active_commitment = None
        f.stamina = gcfg.fighter.max_stamina
        assert not can_commit(f, CombatCommitment.HEAVY_ATTACK, gcfg)

    def test_heavy_available_after_cooldown_expires(self, gcfg):
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import (
            enter_commitment, can_commit, tick_heavy_cooldown
        )
        f = self._free_fighter(gcfg)
        enter_commitment(f, CombatCommitment.HEAVY_ATTACK, gcfg)
        f.fsm_state = FSMState.IDLE
        f.fsm_frames_remaining = 0
        f.active_commitment = None
        f.stamina = gcfg.fighter.max_stamina

        # Tick down the full cooldown
        for _ in range(gcfg.actions.heavy_attack.cooldown_ticks):
            tick_heavy_cooldown(f)

        assert can_commit(f, CombatCommitment.HEAVY_ATTACK, gcfg)


# ---------------------------------------------------------------------------
# Engine: tick_heavy_cooldown called each simulate tick
# ---------------------------------------------------------------------------

class TestEngineHeavyCooldownTick:

    def test_engine_imports_tick_heavy_cooldown(self):
        from game.engine import Engine  # noqa — just verifying import chain

    def test_heavy_cooldown_decrements_during_match(self, engine_factory, gcfg):
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment
        engine = engine_factory()
        engine._start_match()

        # Manually set a heavy cooldown on the player
        enter_commitment(engine._state.player, CombatCommitment.HEAVY_ATTACK, gcfg)
        initial_cd = engine._state.player.heavy_cooldown
        assert initial_cd > 0

        # Run one headless tick — cooldown should decrement
        engine._run_headless_tick()
        assert engine._state.player.heavy_cooldown < initial_cd

    def test_heavy_cooldown_counts_down_to_zero_over_ticks(self, engine_factory, gcfg):
        from game.combat.actions import CombatCommitment
        from game.combat.state_machine import enter_commitment
        engine = engine_factory()
        engine._start_match()

        enter_commitment(engine._state.player, CombatCommitment.HEAVY_ATTACK, gcfg)
        cd = gcfg.actions.heavy_attack.cooldown_ticks

        for _ in range(cd + 5):
            engine._run_headless_tick()

        assert engine._state.player.heavy_cooldown == 0


# ---------------------------------------------------------------------------
# Dodge cooldown: stricter value
# ---------------------------------------------------------------------------

class TestDodgeCooldownStricter:

    def test_dodge_cooldown_frames_at_least_60(self, gcfg):
        # Was 45 — now must be meaningfully higher to punish spam
        assert gcfg.actions.dodge_backward.cooldown_frames >= 60

    def test_dodge_cooldown_longer_than_animation(self, gcfg):
        dodge = gcfg.actions.dodge_backward
        animation_total = dodge.startup_frames + dodge.active_frames + dodge.recovery_frames
        assert dodge.cooldown_frames > animation_total

    def test_dodge_cooldown_post_recovery_lockout_is_substantial(self, gcfg):
        # The lockout window AFTER the animation should be a meaningful fraction
        # of a second (at 60 ticks/sec, at least 30 ticks = 0.5 sec).
        dodge = gcfg.actions.dodge_backward
        animation_total = dodge.startup_frames + dodge.active_frames + dodge.recovery_frames
        lockout_after_recovery = dodge.cooldown_frames - animation_total
        assert lockout_after_recovery >= 30

    def test_dodge_blocked_immediately_after_commit(self, gcfg):
        from game.state import FighterState
        from game.combat.actions import CombatCommitment, FSMState
        from game.combat.state_machine import enter_commitment, can_commit
        scale = gcfg.simulation.sub_pixel_scale
        f = FighterState(
            x=300 * scale, y=gcfg.arena.ground_y * scale,
            hp=gcfg.fighter.max_hp, stamina=gcfg.fighter.max_stamina,
            fsm_state=FSMState.IDLE, facing=1,
        )
        enter_commitment(f, CombatCommitment.DODGE_BACKWARD, gcfg)
        # Simulate recovery complete but cooldown still active
        f.fsm_state = FSMState.IDLE
        f.fsm_frames_remaining = 0
        f.active_commitment = None
        f.stamina = gcfg.fighter.max_stamina
        assert not can_commit(f, CombatCommitment.DODGE_BACKWARD, gcfg)


# ---------------------------------------------------------------------------
# Renderer: combo corner display
# ---------------------------------------------------------------------------

class TestComboCornerDisplay:

    def _make_state(self, gcfg):
        from game.state import SimulationState, ArenaState, FighterState, MatchStatus
        scale = gcfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            gcfg.arena.width, gcfg.arena.height, gcfg.arena.ground_y, scale)
        return SimulationState(
            player=FighterState(x=arena.width_sub // 3, y=arena.ground_y_sub),
            ai=FighterState(x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )

    def test_combo_corner_method_exists(self, renderer):
        assert hasattr(renderer, "_draw_combo_corner")

    def test_combo_corner_no_error_combo_zero(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 0, 0, "left")
        renderer._draw_combo_corner(screen, 0, 0, "right")

    def test_combo_corner_no_error_combo_one(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 1, 0, "left")

    def test_combo_corner_no_error_combo_two(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 2, 0, "left")
        renderer._draw_combo_corner(screen, 2, 0, "right")

    def test_combo_corner_no_error_combo_five_with_flash(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 5, 12, "left")
        renderer._draw_combo_corner(screen, 5, 12, "right")

    def test_render_accepts_heavy_cd_params(self, renderer, gcfg):
        state = self._make_state(gcfg)
        renderer.render(state, player_heavy_cd=20, ai_heavy_cd=10)

    def test_render_heavy_cd_zero_no_error(self, renderer, gcfg):
        state = self._make_state(gcfg)
        renderer.render(state, player_heavy_cd=0, ai_heavy_cd=0)

    def test_render_combo_and_heavy_together(self, renderer, gcfg):
        state = self._make_state(gcfg)
        renderer.render(
            state,
            player_combo=4, ai_combo=2,
            player_combo_flash=8, ai_combo_flash=0,
            player_heavy_cd=30, ai_heavy_cd=0,
        )

    def test_draw_fighter_hud_accepts_heavy_cd(self, renderer, gcfg):
        from game.state import FighterState
        screen = renderer._screen
        fighter = FighterState(x=0, y=gcfg.arena.ground_y * gcfg.simulation.sub_pixel_scale)
        renderer._draw_fighter_hud(
            screen, fighter, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            20, 16, "PLAYER",
            heavy_cd=15, max_heavy_cd=45,
        )

    def test_draw_fighter_hud_heavy_cd_zero(self, renderer, gcfg):
        from game.state import FighterState
        screen = renderer._screen
        fighter = FighterState(x=0, y=gcfg.arena.ground_y * gcfg.simulation.sub_pixel_scale)
        renderer._draw_fighter_hud(
            screen, fighter, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            20, 16, "PLAYER",
            heavy_cd=0, max_heavy_cd=45,
        )

    def test_fighter_hud_no_longer_has_combo_param(self, renderer, gcfg):
        """_draw_fighter_hud should NOT accept combo/combo_flash anymore."""
        import inspect
        sig = inspect.signature(renderer._draw_fighter_hud)
        assert "combo" not in sig.parameters
        assert "combo_flash" not in sig.parameters

    def test_combo_not_in_draw_hud_params(self, renderer):
        import inspect
        sig = inspect.signature(renderer._draw_hud)
        assert "combo" not in sig.parameters

    def test_render_full_params_no_error(self, renderer, gcfg):
        state = self._make_state(gcfg)
        renderer.render(
            state,
            show_help=False,
            player_flash=0, ai_flash=0,
            player_whiff=0, ai_whiff=0,
            player_dodge_cd=20, ai_dodge_cd=10,
            player_heavy_cd=30, ai_heavy_cd=0,
            shake_x=0, shake_y=0,
            player_combo=3, ai_combo=0,
            player_combo_flash=10, ai_combo_flash=0,
        )


# ---------------------------------------------------------------------------
# Config loader: field-level default fallback (_build_dataclass fix)
# ---------------------------------------------------------------------------

class TestConfigLoaderFieldDefaults:

    def test_attack_config_cooldown_ticks_field_has_default(self):
        from config.config_loader import AttackActionConfig
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(AttackActionConfig)}
        assert "cooldown_ticks" in fields
        assert fields["cooldown_ticks"].default == 0

    def test_load_config_succeeds(self):
        from config.config_loader import load_config
        cfg, _, _ = load_config()
        assert cfg is not None

    def test_heavy_cooldown_loaded_from_yaml(self):
        from config.config_loader import load_config
        cfg, _, _ = load_config()
        # Value should come from game_config.yaml, not the default of 0
        assert cfg.actions.heavy_attack.cooldown_ticks > 0

    def test_light_cooldown_defaults_to_zero_from_missing_yaml_key(self):
        from config.config_loader import load_config
        cfg, _, _ = load_config()
        # light_attack has no cooldown_ticks in YAML — should fall back to field default
        assert cfg.actions.light_attack.cooldown_ticks == 0
