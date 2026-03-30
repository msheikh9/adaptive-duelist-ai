"""Tests for Phase 13 UX screen wiring and overlay state management."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()


# ---------------------------------------------------------------------------
# InputHandler: toggle_help_requested
# ---------------------------------------------------------------------------

class TestInputHandlerHelpToggle:

    def test_toggle_help_false_by_default(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        assert h.toggle_help_requested is False

    def test_toggle_help_false_after_poll_with_no_events(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        pygame.event.clear()
        h.poll()
        assert h.toggle_help_requested is False

    def test_toggle_help_true_on_h_key(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        pygame.event.clear()
        # Simulate H key press
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_h, "mod": 0, "unicode": "h", "scancode": 0,
        }))
        h.poll()
        assert h.toggle_help_requested is True

    def test_toggle_help_reset_on_next_poll(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_h, "mod": 0, "unicode": "h", "scancode": 0,
        }))
        h.poll()
        assert h.toggle_help_requested is True
        pygame.event.clear()
        h.poll()
        assert h.toggle_help_requested is False

    def test_h_key_does_not_produce_input_action(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_h, "mod": 0, "unicode": "h", "scancode": 0,
        }))
        actions = h.poll()
        assert actions == []  # H is consumed internally, not forwarded

    def test_quit_requested_still_works(self):
        from game.input.input_handler import InputHandler
        h = InputHandler()
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        h.poll()
        assert h.quit_requested is True


# ---------------------------------------------------------------------------
# Engine: show_help flag and hit-flash counters (headless / attribute-level)
# ---------------------------------------------------------------------------

class TestEngineUXState:

    @pytest.fixture
    def headless_engine(self, tmp_path):
        """Build a headless engine with a temp database."""
        from config.config_loader import load_config
        from data.db import Database
        from data.migrations.migration_runner import run_migrations
        from game.engine import Engine

        game_cfg, ai_cfg, display_cfg = load_config()
        db_path = tmp_path / "ux_test.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        engine = Engine(game_cfg, ai_cfg, display_cfg, db,
                        headless=True)
        yield engine
        db.close()

    def test_show_help_false_by_default(self, headless_engine):
        assert headless_engine._show_help is False

    def test_hit_flash_counters_zero_by_default(self, headless_engine):
        assert headless_engine._player_hit_flash == 0
        assert headless_engine._ai_hit_flash == 0

    def test_hit_flash_reset_on_start_match(self, headless_engine):
        headless_engine._player_hit_flash = 99
        headless_engine._ai_hit_flash = 99
        headless_engine._show_help = True
        headless_engine._start_match()
        assert headless_engine._player_hit_flash == 0
        assert headless_engine._ai_hit_flash == 0
        assert headless_engine._show_help is False

    def test_show_help_reset_on_start_match(self, headless_engine):
        headless_engine._show_help = True
        headless_engine._start_match()
        assert headless_engine._show_help is False


# ---------------------------------------------------------------------------
# Renderer: initialises without error (dummy display)
# ---------------------------------------------------------------------------

class TestRendererInit:

    def test_renderer_init(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        _, _, display_cfg = load_config()
        game_cfg, _, _ = load_config()
        r = Renderer(game_cfg, display_cfg, ai_tier_name="T2_FULL_ADAPTIVE")
        # init() requires a display; skip — just confirm construction works
        assert r._ai_tier_name == "T2_FULL_ADAPTIVE"

    def test_renderer_tier_name_stored(self):
        from config.config_loader import load_config
        from rendering.renderer import Renderer
        game_cfg, _, display_cfg = load_config()
        for tier in ("T0_BASELINE", "T1_MARKOV_ONLY", "T2_FULL_ADAPTIVE"):
            r = Renderer(game_cfg, display_cfg, ai_tier_name=tier)
            assert r._ai_tier_name == tier


# ---------------------------------------------------------------------------
# Config: new colour and HUD fields loaded correctly
# ---------------------------------------------------------------------------

class TestDisplayConfigNewFields:

    def test_new_color_fields_present(self):
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        colors = display_cfg.colors
        # These were added in Phase 13
        assert hasattr(colors, "accent")
        assert hasattr(colors, "hit_flash")
        assert hasattr(colors, "health_critical")
        assert hasattr(colors, "stamina_low")
        assert hasattr(colors, "panel_bg")
        assert hasattr(colors, "tier_badge")
        assert hasattr(colors, "border")

    def test_new_color_fields_are_rgb_tuples(self):
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        colors = display_cfg.colors
        for field in ("accent", "hit_flash", "health_critical",
                      "stamina_low", "panel_bg", "tier_badge", "border"):
            val = getattr(colors, field)
            assert isinstance(val, tuple), f"{field} must be a tuple"
            assert len(val) == 3, f"{field} must have 3 components"
            assert all(0 <= c <= 255 for c in val), f"{field} values out of range"

    def test_font_size_large_present(self):
        from config.config_loader import load_config
        _, _, display_cfg = load_config()
        assert hasattr(display_cfg.hud, "font_size_large")
        assert display_cfg.hud.font_size_large > 0
