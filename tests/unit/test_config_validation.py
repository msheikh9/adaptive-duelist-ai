"""Tests for hardened config validation."""

from __future__ import annotations

import pytest
from dataclasses import replace

from config.config_loader import (
    ConfigError,
    AIConfig,
    PredictionConfig,
    EnsembleConfig,
    PlannerMemoryConfig,
    load_config,
    _validate_ai_config,
)


@pytest.fixture
def ai_cfg():
    _, cfg, _ = load_config()
    return cfg


class TestAIConfigValidation:
    def test_valid_config_passes(self, ai_cfg):
        """Default config should pass validation."""
        _validate_ai_config(ai_cfg)

    def test_window_ticks_exceeds_max(self, ai_cfg):
        bad = AIConfig(
            prediction=PredictionConfig(window_ticks=200, window_max_ticks=100),
            ensemble=ai_cfg.ensemble,
            training=ai_cfg.training,
            strategy=ai_cfg.strategy,
            planner_memory=ai_cfg.planner_memory,
            action_resolver=ai_cfg.action_resolver,
            reaction=ai_cfg.reaction,
            profile=ai_cfg.profile,
        )
        with pytest.raises(ConfigError, match="exceeds"):
            _validate_ai_config(bad)

    def test_drain_exceeds_regen(self, ai_cfg):
        bad_mem = PlannerMemoryConfig(
            exploration_budget_drain_rate=0.50,
            exploration_budget_regen_rate=0.10,
        )
        bad = AIConfig(
            prediction=ai_cfg.prediction,
            ensemble=ai_cfg.ensemble,
            training=ai_cfg.training,
            strategy=ai_cfg.strategy,
            planner_memory=bad_mem,
            action_resolver=ai_cfg.action_resolver,
            reaction=ai_cfg.reaction,
            profile=ai_cfg.profile,
        )
        with pytest.raises(ConfigError, match="drain_rate"):
            _validate_ai_config(bad)

    def test_negative_markov_order(self, ai_cfg):
        bad = AIConfig(
            prediction=PredictionConfig(markov_order=-1),
            ensemble=ai_cfg.ensemble,
            training=ai_cfg.training,
            strategy=ai_cfg.strategy,
            planner_memory=ai_cfg.planner_memory,
            action_resolver=ai_cfg.action_resolver,
            reaction=ai_cfg.reaction,
            profile=ai_cfg.profile,
        )
        with pytest.raises(ConfigError, match="positive"):
            _validate_ai_config(bad)

    def test_holdout_fraction_out_of_range(self, ai_cfg):
        from config.config_loader import TrainingConfig

        bad_train = TrainingConfig(holdout_fraction=1.5)
        bad = AIConfig(
            prediction=ai_cfg.prediction,
            ensemble=ai_cfg.ensemble,
            training=bad_train,
            strategy=ai_cfg.strategy,
            planner_memory=ai_cfg.planner_memory,
            action_resolver=ai_cfg.action_resolver,
            reaction=ai_cfg.reaction,
            profile=ai_cfg.profile,
        )
        with pytest.raises(ConfigError, match="between 0.0 and 1.0"):
            _validate_ai_config(bad)

    def test_ema_alpha_out_of_range(self, ai_cfg):
        bad_ens = EnsembleConfig(weight_update_ema_alpha=2.0)
        bad = AIConfig(
            prediction=ai_cfg.prediction,
            ensemble=bad_ens,
            training=ai_cfg.training,
            strategy=ai_cfg.strategy,
            planner_memory=ai_cfg.planner_memory,
            action_resolver=ai_cfg.action_resolver,
            reaction=ai_cfg.reaction,
            profile=ai_cfg.profile,
        )
        with pytest.raises(ConfigError, match="between 0.0 and 1.0"):
            _validate_ai_config(bad)
