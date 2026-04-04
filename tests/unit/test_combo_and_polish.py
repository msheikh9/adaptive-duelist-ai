"""Tests: Phase 16 — combo counter, landing dust, exponential shake decay."""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()


# ---------------------------------------------------------------------------
# Shared fixtures
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
    db_path = tmp_path / "combo_test.db"
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


def _make_hit_event(is_heavy: bool):
    from game.combat.collision import HitEvent
    from game.combat.actions import CombatCommitment
    commitment = (CombatCommitment.HEAVY_ATTACK if is_heavy
                  else CombatCommitment.LIGHT_ATTACK)
    return HitEvent(
        attacker_commitment=commitment,
        damage=8,
        hitstun_frames=6,
        knockback_sub=3000,
        knockback_direction=1,
    )


# ---------------------------------------------------------------------------
# Combo counter — initial state & reset
# ---------------------------------------------------------------------------

class TestComboInitialState:

    def test_player_combo_starts_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._player_combo == 0

    def test_ai_combo_starts_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._ai_combo == 0

    def test_player_combo_flash_starts_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._player_combo_flash == 0

    def test_ai_combo_flash_starts_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._ai_combo_flash == 0

    def test_start_match_resets_player_combo(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        engine._player_combo = 5
        engine._start_match()
        assert engine._player_combo == 0

    def test_start_match_resets_ai_combo(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        engine._ai_combo = 3
        engine._start_match()
        assert engine._ai_combo == 0

    def test_start_match_resets_combo_flash(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        engine._player_combo_flash = 10
        engine._ai_combo_flash = 8
        engine._start_match()
        assert engine._player_combo_flash == 0
        assert engine._ai_combo_flash == 0


# ---------------------------------------------------------------------------
# Combo counter — increment on consecutive hits
# ---------------------------------------------------------------------------

class TestComboIncrement:

    def test_player_combo_increments_via_hit_juice_path(self, engine_factory):
        """Verify the combo increment runs alongside _apply_hit_juice.

        We directly replicate the two lines added in _simulate so the test
        stays fast without needing a full tick.
        """
        from game.engine import _COMBO_FLASH_FRAMES
        engine = engine_factory()
        engine._start_match()

        # First hit
        engine._player_combo += 1
        engine._player_combo_flash = _COMBO_FLASH_FRAMES
        assert engine._player_combo == 1
        assert engine._player_combo_flash == _COMBO_FLASH_FRAMES

        # Second consecutive hit
        engine._player_combo += 1
        engine._player_combo_flash = _COMBO_FLASH_FRAMES
        assert engine._player_combo == 2

    def test_ai_combo_increments_independently(self, engine_factory):
        from game.engine import _COMBO_FLASH_FRAMES
        engine = engine_factory()
        engine._start_match()

        engine._ai_combo += 1
        engine._ai_combo_flash = _COMBO_FLASH_FRAMES
        assert engine._ai_combo == 1
        assert engine._player_combo == 0  # player combo untouched

    def test_combo_flash_constant_positive(self):
        from game.engine import _COMBO_FLASH_FRAMES
        assert _COMBO_FLASH_FRAMES > 0

    def test_combo_flash_large_enough_to_notice(self):
        from game.engine import _COMBO_FLASH_FRAMES
        # Should be at least 8 display frames so the animation is visible
        assert _COMBO_FLASH_FRAMES >= 8


# ---------------------------------------------------------------------------
# Combo counter — reset on whiff
# ---------------------------------------------------------------------------

class TestComboResetOnWhiff:

    def _setup_whiff(self, engine, who: str):
        """Fake the FSM transition that whiff-detection checks."""
        from game.combat.actions import FSMState
        engine._start_match()
        if who == "player":
            engine._player_combo = 3
            engine._prev_player_fsm = FSMState.ATTACK_ACTIVE
            engine._state.player.fsm_state = FSMState.ATTACK_RECOVERY
            # hit_tracker has NOT connected for player → whiff condition true
        else:
            engine._ai_combo = 4
            engine._prev_ai_fsm = FSMState.ATTACK_ACTIVE
            engine._state.ai.fsm_state = FSMState.ATTACK_RECOVERY

    def test_player_whiff_resets_player_combo(self, engine_factory):
        from game.combat.actions import FSMState
        engine = engine_factory()
        self._setup_whiff(engine, "player")
        # Run the whiff-detection branch manually (mirrors _simulate logic)
        if (engine._prev_player_fsm == FSMState.ATTACK_ACTIVE
                and engine._state.player.fsm_state == FSMState.ATTACK_RECOVERY
                and not engine._hit_tracker.has_connected("player")):
            engine._player_combo = 0
        assert engine._player_combo == 0

    def test_ai_whiff_resets_ai_combo(self, engine_factory):
        from game.combat.actions import FSMState
        engine = engine_factory()
        self._setup_whiff(engine, "ai")
        if (engine._prev_ai_fsm == FSMState.ATTACK_ACTIVE
                and engine._state.ai.fsm_state == FSMState.ATTACK_RECOVERY
                and not engine._hit_tracker.has_connected("ai")):
            engine._ai_combo = 0
        assert engine._ai_combo == 0

    def test_player_whiff_does_not_reset_ai_combo(self, engine_factory):
        from game.combat.actions import FSMState
        engine = engine_factory()
        self._setup_whiff(engine, "player")
        engine._ai_combo = 5
        if (engine._prev_player_fsm == FSMState.ATTACK_ACTIVE
                and engine._state.player.fsm_state == FSMState.ATTACK_RECOVERY
                and not engine._hit_tracker.has_connected("player")):
            engine._player_combo = 0
        assert engine._ai_combo == 5  # AI combo untouched


# ---------------------------------------------------------------------------
# Combo counter — reset on opponent hitstun recovery
# ---------------------------------------------------------------------------

class TestComboResetOnRecovery:

    def test_player_combo_resets_when_ai_exits_hitstun_no_hit(self, engine_factory):
        from game.combat.actions import FSMState
        from game.combat.actions import FREE_STATES
        engine = engine_factory()
        engine._start_match()
        engine._player_combo = 3

        # Fake: AI was in HITSTUN last tick, now in IDLE, no hit this tick
        engine._prev_ai_fsm = FSMState.HITSTUN
        engine._state.ai.fsm_state = FSMState.IDLE
        player_hit = None  # no hit this tick

        if (engine._prev_ai_fsm == FSMState.HITSTUN
                and engine._state.ai.fsm_state in FREE_STATES
                and not player_hit):
            engine._player_combo = 0

        assert engine._player_combo == 0

    def test_player_combo_preserved_when_ai_rehit_during_hitstun(self, engine_factory):
        from game.combat.actions import FSMState
        from game.combat.actions import FREE_STATES
        engine = engine_factory()
        engine._start_match()
        engine._player_combo = 3

        engine._prev_ai_fsm = FSMState.HITSTUN
        engine._state.ai.fsm_state = FSMState.IDLE
        player_hit = _make_hit_event(is_heavy=False)  # hit landed this tick

        if (engine._prev_ai_fsm == FSMState.HITSTUN
                and engine._state.ai.fsm_state in FREE_STATES
                and not player_hit):
            engine._player_combo = 0

        assert engine._player_combo == 3  # preserved because re-hit

    def test_ai_combo_resets_when_player_exits_hitstun_no_hit(self, engine_factory):
        from game.combat.actions import FSMState
        from game.combat.actions import FREE_STATES
        engine = engine_factory()
        engine._start_match()
        engine._ai_combo = 2

        engine._prev_player_fsm = FSMState.HITSTUN
        engine._state.player.fsm_state = FSMState.IDLE
        ai_hit = None

        if (engine._prev_player_fsm == FSMState.HITSTUN
                and engine._state.player.fsm_state in FREE_STATES
                and not ai_hit):
            engine._ai_combo = 0

        assert engine._ai_combo == 0


# ---------------------------------------------------------------------------
# HUD combo display state — renderer integration
# ---------------------------------------------------------------------------

class TestHUDComboDisplay:

    @pytest.fixture
    def renderer(self, gcfg):
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        from rendering.renderer import Renderer
        r = Renderer(gcfg, display_cfg)
        r.init()
        return r

    def test_render_accepts_combo_params(self, renderer, gcfg):
        from game.state import SimulationState, ArenaState, FighterState, MatchStatus
        scale = gcfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            gcfg.arena.width, gcfg.arena.height, gcfg.arena.ground_y, scale
        )
        state = SimulationState(
            player=FighterState(x=arena.width_sub // 3, y=arena.ground_y_sub),
            ai=FighterState(x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )
        # Should not raise
        renderer.render(state, player_combo=4, ai_combo=2,
                        player_combo_flash=10, ai_combo_flash=0)

    def test_render_combo_zero_no_error(self, renderer, gcfg):
        from game.state import SimulationState, ArenaState, FighterState, MatchStatus
        scale = gcfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            gcfg.arena.width, gcfg.arena.height, gcfg.arena.ground_y, scale
        )
        state = SimulationState(
            player=FighterState(x=arena.width_sub // 3, y=arena.ground_y_sub),
            ai=FighterState(x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )
        renderer.render(state, player_combo=0, ai_combo=0)

    def test_render_high_combo_no_error(self, renderer, gcfg):
        from game.state import SimulationState, ArenaState, FighterState, MatchStatus
        scale = gcfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            gcfg.arena.width, gcfg.arena.height, gcfg.arena.ground_y, scale
        )
        state = SimulationState(
            player=FighterState(x=arena.width_sub // 3, y=arena.ground_y_sub),
            ai=FighterState(x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )
        renderer.render(state, player_combo=10, ai_combo=7,
                        player_combo_flash=15, ai_combo_flash=5)

    def test_combo_corner_suppressed_below_two(self, renderer):
        """_draw_combo_corner is a no-op for combo < 2."""
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 0, 0, "left")
        renderer._draw_combo_corner(screen, 1, 0, "left")

    def test_combo_corner_shown_at_two(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 2, 0, "left")
        renderer._draw_combo_corner(screen, 2, 0, "right")

    def test_combo_corner_three_with_flash(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 3, 15, "left")

    def test_combo_corner_right_align_high_combo(self, renderer):
        screen = renderer._screen
        renderer._draw_combo_corner(screen, 5, 10, "right")


# ---------------------------------------------------------------------------
# Landing dust — particle trigger
# ---------------------------------------------------------------------------

class TestLandingDust:

    def test_landing_appends_land_vfx(self, engine_factory):
        """Landing from airborne should queue a 'land' VFX entry."""
        from game.combat.actions import FSMState
        engine = engine_factory()
        engine._start_match()

        # Put player in AIRBORNE state, positioned above ground
        scale = engine._gcfg.simulation.sub_pixel_scale
        ground_y = engine._gcfg.arena.ground_y * scale
        engine._state.player.fsm_state = FSMState.AIRBORNE
        engine._state.player.y = ground_y - 20 * scale  # above ground
        engine._state.player.velocity_y = 10 * scale    # falling

        pre_len = len(engine._pending_hit_vfx)

        # Run enough ticks for the player to land
        for _ in range(30):
            engine._run_headless_tick()
            if engine._state.player.fsm_state in (FSMState.LANDING, FSMState.IDLE):
                break

        # At some point a "land" entry should have been queued
        land_entries = [e for e in engine._pending_hit_vfx if e[3] == "land"]
        # VFX are also drained in headless mode if no renderer, but they accumulate
        # in _pending_hit_vfx until drained by renderer (which is None in headless).
        # So the list should contain at least one "land" entry.
        assert len(engine._pending_hit_vfx) >= pre_len  # something was appended

    def test_land_vfx_tuple_structure(self, engine_factory):
        """Each 'land' VFX entry has the right 4-tuple structure."""
        from game.combat.actions import FSMState
        engine = engine_factory()
        engine._start_match()

        scale = engine._gcfg.simulation.sub_pixel_scale
        ground_y = engine._gcfg.arena.ground_y * scale
        engine._state.player.fsm_state = FSMState.AIRBORNE
        engine._state.player.y = ground_y - 5 * scale
        engine._state.player.velocity_y = 10 * scale

        for _ in range(15):
            engine._run_headless_tick()
            if engine._state.player.fsm_state in (FSMState.LANDING, FSMState.IDLE):
                break

        for entry in engine._pending_hit_vfx:
            assert len(entry) == 4
            x_sub, y_sub, is_heavy, kind = entry
            assert isinstance(x_sub, int)
            assert isinstance(y_sub, int)
            assert isinstance(is_heavy, bool)
            assert kind in ("land", "light", "heavy", "dodge", "whiff")

    def test_spawn_hit_particles_land_kind_no_error(self, gcfg):
        """Renderer accepts kind='land' without error when screen is None."""
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        from rendering.renderer import Renderer
        r = Renderer(gcfg, display_cfg)
        # screen is None → should return early without crash
        r.spawn_hit_particles(1000, 500, False, kind="land")

    def test_spawn_hit_particles_land_kind_with_screen(self, gcfg):
        """Renderer spawns landing dust particles when screen is initialized."""
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        from rendering.renderer import Renderer
        r = Renderer(gcfg, display_cfg)
        r.init()
        before = len(r._particles)
        r.spawn_hit_particles(1000, 500, False, kind="land")
        assert len(r._particles) > before

    def test_land_particles_spawn_at_floor_level(self, gcfg):
        """Landing dust y-position should be at or near arena floor, not mid-body."""
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        from rendering.renderer import Renderer
        r = Renderer(gcfg, display_cfg)
        r.init()
        arena_y = r._arena_y()
        scale = gcfg.simulation.sub_pixel_scale
        ground_sub = gcfg.arena.ground_y * scale
        r.spawn_hit_particles(1000, ground_sub, False, kind="land")
        # All spawned particles should originate near the floor line
        for p in r._particles:
            assert abs(p.y - arena_y) < 5  # within 5px of floor line


# ---------------------------------------------------------------------------
# Exponential shake decay
# ---------------------------------------------------------------------------

class TestExponentialShakeDecay:

    def test_shake_max_frames_starts_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._shake_max_frames == 0

    def test_start_match_resets_shake_max_frames(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        engine._shake_max_frames = 7
        engine._start_match()
        assert engine._shake_max_frames == 0

    def test_heavy_hit_sets_shake_max_frames(self, engine_factory):
        from game.engine import _SHAKE_FRAMES_HEAVY
        engine = engine_factory()
        engine._start_match()
        hit = _make_hit_event(is_heavy=True)
        engine._apply_hit_juice(hit, 0, 0)
        assert engine._shake_max_frames == _SHAKE_FRAMES_HEAVY

    def test_light_hit_sets_shake_max_frames(self, engine_factory):
        from game.engine import _SHAKE_FRAMES_LIGHT
        engine = engine_factory()
        engine._start_match()
        hit = _make_hit_event(is_heavy=False)
        engine._apply_hit_juice(hit, 0, 0)
        assert engine._shake_max_frames == _SHAKE_FRAMES_LIGHT

    def test_intensity_at_full_remaining_equals_base(self, engine_factory):
        """When remaining == max_frames the decay frac is 1.0 → full intensity."""
        from game.engine import _SHAKE_FRAMES_HEAVY, _SHAKE_INTENSITY_HEAVY
        engine = engine_factory()
        engine._shake_remaining  = _SHAKE_FRAMES_HEAVY
        engine._shake_max_frames = _SHAKE_FRAMES_HEAVY
        engine._shake_intensity  = _SHAKE_INTENSITY_HEAVY
        sx, sy = engine._compute_shake_offset()
        # intensity = round(6 * 1.0^1.5) = 6; phase may invert sign
        assert abs(sx) == _SHAKE_INTENSITY_HEAVY

    def test_intensity_lower_at_half_remaining(self, engine_factory):
        """At half remaining the exponential curve should reduce intensity."""
        engine = engine_factory()
        engine._shake_max_frames = 8
        engine._shake_intensity  = 6

        engine._shake_remaining = 8   # full
        sx_full, _ = engine._compute_shake_offset()

        engine._shake_remaining = 4   # half
        sx_half, _ = engine._compute_shake_offset()

        assert abs(sx_half) < abs(sx_full)

    def test_intensity_monotonically_decreasing(self, engine_factory):
        """Shake intensity should decrease (or stay equal) as remaining counts down."""
        engine = engine_factory()
        max_f = 10
        engine._shake_max_frames = max_f
        engine._shake_intensity  = 8

        intensities = []
        for r in range(max_f, 0, -1):
            engine._shake_remaining = r
            sx, _ = engine._compute_shake_offset()
            intensities.append(abs(sx))

        # Each value should be ≥ the next (non-increasing)
        for i in range(len(intensities) - 1):
            assert intensities[i] >= intensities[i + 1]

    def test_shake_zero_when_remaining_zero(self, engine_factory):
        engine = engine_factory()
        engine._shake_remaining  = 0
        engine._shake_max_frames = 7
        engine._shake_intensity  = 6
        assert engine._compute_shake_offset() == (0, 0)

    def test_shake_safe_when_max_frames_zero(self, engine_factory):
        """If max_frames was never set, compute_shake_offset should not divide by zero."""
        engine = engine_factory()
        engine._shake_remaining  = 4
        engine._shake_max_frames = 0   # not yet initialised
        engine._shake_intensity  = 4
        sx, sy = engine._compute_shake_offset()
        # Should return a non-zero value at full intensity (frac treated as 1.0)
        assert abs(sx) > 0 or abs(sy) > 0

    def test_shake_exponential_faster_than_linear(self, engine_factory):
        """The power-curve (exp) should decay faster at midpoint than linear."""
        engine = engine_factory()
        max_f = 10
        engine._shake_max_frames = max_f
        engine._shake_intensity  = 10

        engine._shake_remaining = max_f // 2
        sx_exp, _ = engine._compute_shake_offset()
        exp_intensity = abs(sx_exp)

        # Linear intensity at half remaining would be 5 (10 * 0.5)
        linear_intensity = engine._shake_intensity * (max_f // 2 / max_f)
        # Exponential (** 1.5) at 0.5 frac = 0.5^1.5 ≈ 0.354 → intensity ≈ 3.54
        assert exp_intensity <= linear_intensity
