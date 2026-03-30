"""Tests: Phase 15 combat juice — hitstop, screen shake, particles, sound."""

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
    db_path = tmp_path / "juice_test.db"
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


# ---------------------------------------------------------------------------
# Sound manager
# ---------------------------------------------------------------------------

class TestSoundManager:

    def test_null_sound_manager_exists(self):
        from game.sound import NullSoundManager
        s = NullSoundManager()
        assert s is not None

    def test_null_sound_manager_all_methods_callable(self):
        from game.sound import NullSoundManager
        s = NullSoundManager()
        s.play_hit_light()
        s.play_hit_heavy()
        s.play_jump()
        s.play_land()
        s.play_whiff()

    def test_null_sound_manager_all_return_none(self):
        from game.sound import NullSoundManager
        s = NullSoundManager()
        assert s.play_hit_light()  is None
        assert s.play_hit_heavy()  is None
        assert s.play_jump()       is None
        assert s.play_land()       is None
        assert s.play_whiff()      is None

    def test_engine_has_sound_manager(self, engine_factory):
        from game.sound import NullSoundManager
        engine = engine_factory()
        assert isinstance(engine._sound, NullSoundManager)


# ---------------------------------------------------------------------------
# Hitstop constants
# ---------------------------------------------------------------------------

class TestHitstopConstants:

    def test_hitstop_light_less_than_heavy(self):
        from game.engine import _HITSTOP_LIGHT, _HITSTOP_HEAVY
        assert _HITSTOP_LIGHT < _HITSTOP_HEAVY

    def test_hitstop_light_positive(self):
        from game.engine import _HITSTOP_LIGHT
        assert _HITSTOP_LIGHT > 0

    def test_hitstop_heavy_positive(self):
        from game.engine import _HITSTOP_HEAVY
        assert _HITSTOP_HEAVY > 0

    def test_engine_initial_hitstop_is_zero(self, engine_factory):
        engine = engine_factory()
        assert engine._hitstop_remaining == 0


# ---------------------------------------------------------------------------
# Hitstop + shake set on hit (via _apply_hit_juice)
# ---------------------------------------------------------------------------

class TestHitJuice:

    def _make_hit_event(self, is_heavy: bool):
        from game.combat.collision import HitEvent
        from game.combat.actions import CombatCommitment
        commitment = CombatCommitment.HEAVY_ATTACK if is_heavy else CombatCommitment.LIGHT_ATTACK
        return HitEvent(
            attacker_commitment=commitment,
            damage=8,
            hitstun_frames=6,
            knockback_sub=3000,
            knockback_direction=1,
        )

    def test_light_hit_sets_hitstop(self, engine_factory):
        from game.engine import _HITSTOP_LIGHT
        engine = engine_factory()
        engine._start_match()
        hit = self._make_hit_event(is_heavy=False)
        engine._apply_hit_juice(hit, 0, 0)
        assert engine._hitstop_remaining == _HITSTOP_LIGHT

    def test_heavy_hit_sets_longer_hitstop(self, engine_factory):
        from game.engine import _HITSTOP_HEAVY
        engine = engine_factory()
        engine._start_match()
        hit = self._make_hit_event(is_heavy=True)
        engine._apply_hit_juice(hit, 0, 0)
        assert engine._hitstop_remaining == _HITSTOP_HEAVY

    def test_heavy_hitstop_greater_than_light(self, engine_factory):
        from game.engine import _HITSTOP_LIGHT, _HITSTOP_HEAVY
        engine = engine_factory()
        engine._start_match()

        light_hit = self._make_hit_event(is_heavy=False)
        engine._apply_hit_juice(light_hit, 0, 0)
        light_hs = engine._hitstop_remaining

        engine._hitstop_remaining = 0
        heavy_hit = self._make_hit_event(is_heavy=True)
        engine._apply_hit_juice(heavy_hit, 0, 0)
        heavy_hs = engine._hitstop_remaining

        assert heavy_hs > light_hs

    def test_hit_sets_shake(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        hit = self._make_hit_event(is_heavy=False)
        engine._apply_hit_juice(hit, 0, 0)
        assert engine._shake_remaining > 0
        assert engine._shake_intensity > 0

    def test_heavy_hit_sets_larger_shake(self, engine_factory):
        from game.engine import _SHAKE_INTENSITY_LIGHT, _SHAKE_INTENSITY_HEAVY
        engine = engine_factory()
        engine._start_match()
        heavy_hit = self._make_hit_event(is_heavy=True)
        engine._apply_hit_juice(heavy_hit, 0, 0)
        assert engine._shake_intensity == _SHAKE_INTENSITY_HEAVY

    def test_pending_vfx_populated_on_hit(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        hit = self._make_hit_event(is_heavy=False)
        engine._apply_hit_juice(hit, 1000, 2000)
        assert len(engine._pending_hit_vfx) == 1
        x_sub, y_sub, is_heavy = engine._pending_hit_vfx[0]
        assert x_sub == 1000
        assert y_sub == 2000
        assert is_heavy is False

    def test_pending_vfx_heavy_flag(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        hit = self._make_hit_event(is_heavy=True)
        engine._apply_hit_juice(hit, 0, 0)
        _, _, is_heavy = engine._pending_hit_vfx[0]
        assert is_heavy is True

    def test_start_match_clears_juice_state(self, engine_factory):
        engine = engine_factory()
        engine._start_match()
        engine._hitstop_remaining = 5
        engine._shake_remaining = 3
        engine._pending_hit_vfx.append((0, 0, False))
        engine._start_match()
        assert engine._hitstop_remaining == 0
        assert engine._shake_remaining == 0
        assert len(engine._pending_hit_vfx) == 0

    def test_compute_shake_offset_zero_when_inactive(self, engine_factory):
        engine = engine_factory()
        engine._shake_remaining = 0
        assert engine._compute_shake_offset() == (0, 0)

    def test_compute_shake_offset_nonzero_when_active(self, engine_factory):
        engine = engine_factory()
        engine._shake_remaining = 3
        engine._shake_intensity = 4
        sx, sy = engine._compute_shake_offset()
        # At least one axis should be non-zero
        assert sx != 0 or sy != 0

    def test_simultaneous_hits_take_max_hitstop(self, engine_factory):
        """If two hits land in the same simulate call, take the max hitstop."""
        from game.engine import _HITSTOP_HEAVY
        engine = engine_factory()
        engine._start_match()

        light = self._make_hit_event(is_heavy=False)
        heavy = self._make_hit_event(is_heavy=True)
        engine._apply_hit_juice(light, 0, 0)
        engine._apply_hit_juice(heavy, 0, 0)
        assert engine._hitstop_remaining == _HITSTOP_HEAVY


# ---------------------------------------------------------------------------
# Renderer particle system (headless-safe tests)
# ---------------------------------------------------------------------------

class TestParticles:

    def test_renderer_has_particles_list(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        assert hasattr(r, "_particles")
        assert isinstance(r._particles, list)

    def test_spawn_hit_particles_no_crash_without_screen(self):
        """spawn_hit_particles is a no-op when screen is not initialised."""
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        # No init() called → _screen is None; should not raise
        r.spawn_hit_particles(10000, 30000, False)

    def test_spawn_adds_particles_after_init(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        r.init()
        r.spawn_hit_particles(60000, 30000, False)
        assert len(r._particles) > 0

    def test_heavy_hit_spawns_more_particles(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        r.init()

        r.spawn_hit_particles(60000, 30000, False)
        light_count = len(r._particles)
        r._particles.clear()

        r.spawn_hit_particles(60000, 30000, True)
        heavy_count = len(r._particles)

        assert heavy_count > light_count

    def test_particles_have_positive_lifetime(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        r.init()
        r.spawn_hit_particles(60000, 30000, True)
        for p in r._particles:
            assert p.lifetime > 0


# ---------------------------------------------------------------------------
# HUD airborne indicator
# ---------------------------------------------------------------------------

class TestHUDAirborne:

    def test_renderer_has_ground_y_sub(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        gcfg, _, dcfg = load_config()
        r = Renderer(gcfg, dcfg)
        expected = gcfg.arena.ground_y * gcfg.simulation.sub_pixel_scale
        assert r._ground_y_sub == expected

    def test_fighter_is_airborne_property_used_in_hud_label(self):
        """is_airborne property exists on FighterState."""
        from game.state import FighterState
        from game.combat.actions import FSMState
        f = FighterState(fsm_state=FSMState.AIRBORNE)
        assert f.is_airborne is True
        f2 = FighterState(fsm_state=FSMState.IDLE)
        assert f2.is_airborne is False
