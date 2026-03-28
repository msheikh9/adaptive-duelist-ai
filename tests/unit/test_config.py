"""Tests for configuration loading and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from config.config_loader import (
    AIConfig,
    ConfigError,
    DisplayConfig,
    GameConfig,
    load_config,
)


class TestConfigLoadsCorrectly:
    def test_loads_default_configs(self, config_dir):
        game_cfg, ai_cfg, display_cfg = load_config(config_dir)
        assert isinstance(game_cfg, GameConfig)
        assert isinstance(ai_cfg, AIConfig)
        assert isinstance(display_cfg, DisplayConfig)

    def test_game_config_values(self, game_config):
        assert game_config.arena.width == 1200
        assert game_config.fighter.max_hp == 200
        assert game_config.fighter.max_stamina == 100
        assert game_config.simulation.tick_rate == 60
        assert game_config.simulation.sub_pixel_scale == 100

    def test_game_config_action_frame_data(self, game_config):
        light = game_config.actions.light_attack
        assert light.startup_frames == 4
        assert light.active_frames == 3
        assert light.recovery_frames == 8
        assert light.total_frames == 15
        assert light.damage == 8
        assert light.stamina_cost == 15

        heavy = game_config.actions.heavy_attack
        assert heavy.startup_frames == 12
        assert heavy.total_frames == 34
        assert heavy.damage == 22

    def test_ai_config_values(self, ai_config):
        assert ai_config.prediction.window_ticks == 60
        assert ai_config.prediction.markov_order == 3
        assert ai_config.training.min_matches_to_train == 3
        assert ai_config.strategy.softmax_temperature == 0.5

    def test_display_config_values(self, display_config):
        assert display_config.window.width == 1280
        assert display_config.window.height == 720
        assert display_config.window.title == "Adaptive Duelist"

    def test_spacing_config(self, game_config):
        assert game_config.spacing.close_max < game_config.spacing.mid_max

    def test_missing_config_files_use_defaults(self, tmp_path):
        game_cfg, ai_cfg, display_cfg = load_config(tmp_path)
        assert game_cfg.arena.width == 1200
        assert ai_cfg.prediction.window_ticks == 60
        assert display_cfg.window.width == 1280


class TestConfigValidation:
    def test_rejects_negative_arena_width(self, tmp_path):
        (tmp_path / "game_config.yaml").write_text(
            yaml.dump({"arena": {"width": -100}})
        )
        with pytest.raises(ConfigError, match="arena.width"):
            load_config(tmp_path)

    def test_rejects_zero_hp(self, tmp_path):
        (tmp_path / "game_config.yaml").write_text(
            yaml.dump({"fighter": {"max_hp": 0}})
        )
        with pytest.raises(ConfigError, match="fighter.max_hp"):
            load_config(tmp_path)

    def test_rejects_stamina_cost_exceeding_max(self, tmp_path):
        cfg = {
            "fighter": {"max_stamina": 10},
            "actions": {
                "light_attack": {
                    "startup_frames": 4, "active_frames": 3, "recovery_frames": 8,
                    "stamina_cost": 15, "damage": 8, "reach": 80,
                    "hitstun_frames": 6, "knockback": 30,
                },
            },
        }
        (tmp_path / "game_config.yaml").write_text(yaml.dump(cfg))
        with pytest.raises(ConfigError, match="stamina_cost.*exceeds.*max_stamina"):
            load_config(tmp_path)

    def test_rejects_close_max_gte_mid_max(self, tmp_path):
        cfg = {"spacing": {"close_max": 400, "mid_max": 350}}
        (tmp_path / "game_config.yaml").write_text(yaml.dump(cfg))
        with pytest.raises(ConfigError, match="close_max.*less than.*mid_max"):
            load_config(tmp_path)

    def test_rejects_invalid_yaml(self, tmp_path):
        (tmp_path / "game_config.yaml").write_text("{{invalid: yaml: [")
        with pytest.raises(ConfigError, match="Failed to parse"):
            load_config(tmp_path)

    def test_rejects_prediction_window_exceeds_max(self, tmp_path):
        cfg = {"prediction": {"window_ticks": 100, "window_max_ticks": 90}}
        (tmp_path / "ai_config.yaml").write_text(yaml.dump(cfg))
        with pytest.raises(ConfigError, match="window_ticks.*exceeds.*window_max_ticks"):
            load_config(tmp_path)
