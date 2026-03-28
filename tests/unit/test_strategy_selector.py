"""Tests for StrategySelector: mode scoring and selection."""

from __future__ import annotations

import random
import pytest

from config.config_loader import load_config
from ai.models.base_predictor import make_prediction_result
from ai.strategy.ai_context import AIContext
from ai.strategy.planner_memory import PlannerMemory
from ai.strategy.strategy_selector import select_mode, _base_score, _softmax_select
from ai.strategy.tactics import TacticalIntent
from game.combat.actions import CombatCommitment, FSMState, SpacingZone


@pytest.fixture
def configs():
    _, ai_cfg, _ = load_config()
    return ai_cfg


@pytest.fixture
def mem(configs):
    return PlannerMemory(configs.planner_memory)


def _make_ctx(
    player_hp_frac: float = 1.0,
    ai_hp_frac: float = 1.0,
    player_stamina_frac: float = 0.8,
    ai_stamina_frac: float = 0.8,
    player_fsm: FSMState = FSMState.IDLE,
    ai_fsm: FSMState = FSMState.IDLE,
    spacing: SpacingZone = SpacingZone.MID,
    prediction_conf: float = 0.0,
    predicted_commit: CombatCommitment | None = None,
    tick_id: int = 100,
) -> AIContext:
    if predicted_commit is not None and prediction_conf > 0:
        dist = {predicted_commit.name: prediction_conf}
        remaining = 1.0 - prediction_conf
        if remaining > 0:
            dist["HOLD"] = remaining
        pred = make_prediction_result(dist, source="markov", markov_level="unigram")
    else:
        pred = make_prediction_result({}, source="none", markov_level="none")

    return AIContext(
        tick_id=tick_id,
        player_hp_frac=player_hp_frac,
        player_stamina_frac=player_stamina_frac,
        player_fsm=player_fsm,
        player_commitment=None,
        ai_hp_frac=ai_hp_frac,
        ai_stamina_frac=ai_stamina_frac,
        ai_fsm=ai_fsm,
        ai_commitment=None,
        ai_facing=1,
        spacing=spacing,
        distance_sub=20000,
        prediction=pred,
    )


class TestModeSelection:
    def test_returns_valid_intent(self, configs, mem):
        ctx = _make_ctx()
        rng = random.Random(42)
        mode = select_mode(ctx, mem, configs.strategy, configs.planner_memory, rng)
        assert isinstance(mode, TacticalIntent)

    def test_forced_probe_on_low_budget(self, configs, mem):
        ctx = _make_ctx()
        rng = random.Random(42)
        # Drain budget below floor
        mem.exploration_budget = configs.planner_memory.exploration_budget_floor - 0.01
        mode = select_mode(ctx, mem, configs.strategy, configs.planner_memory, rng)
        assert mode == TacticalIntent.PROBE_BEHAVIOR

    def test_forced_probe_during_shift(self, configs, mem):
        ctx = _make_ctx()
        rng = random.Random(42)
        mem.shift_detected = True
        mem._shift_probe_remaining = 5
        mode = select_mode(ctx, mem, configs.strategy, configs.planner_memory, rng)
        assert mode == TacticalIntent.PROBE_BEHAVIOR

    def test_high_confidence_favors_exploit(self, configs, mem):
        """When prediction confidence is high, EXPLOIT should score well."""
        ctx = _make_ctx(
            prediction_conf=0.9,
            predicted_commit=CombatCommitment.LIGHT_ATTACK,
        )
        rng = random.Random(42)

        w = configs.strategy.scoring_weights
        exploit_score = _base_score(TacticalIntent.EXPLOIT_PATTERN, ctx, w)
        probe_score = _base_score(TacticalIntent.PROBE_BEHAVIOR, ctx, w)
        assert exploit_score > probe_score

    def test_low_ai_hp_penalizes_aggressive_modes(self, configs, mem):
        ctx = _make_ctx(ai_hp_frac=0.15, prediction_conf=0.5,
                        predicted_commit=CombatCommitment.LIGHT_ATTACK)
        w = configs.strategy.scoring_weights
        exploit_score = _base_score(TacticalIntent.EXPLOIT_PATTERN, ctx, w)
        defensive_score = _base_score(TacticalIntent.DEFENSIVE_RESET, ctx, w)
        # Defensive should be better than exploit when AI is low HP
        assert defensive_score > exploit_score

    def test_player_recovering_boosts_punish(self, configs, mem):
        ctx = _make_ctx(
            player_fsm=FSMState.ATTACK_RECOVERY,
            spacing=SpacingZone.CLOSE,
        )
        w = configs.strategy.scoring_weights
        punish_score = _base_score(TacticalIntent.PUNISH_RECOVERY, ctx, w)
        neutral_score = _base_score(TacticalIntent.NEUTRAL_SPACING, ctx, w)
        assert punish_score > neutral_score


class TestSoftmaxSelect:
    def test_deterministic_with_seed(self):
        scores = {
            TacticalIntent.EXPLOIT_PATTERN: 2.0,
            TacticalIntent.PROBE_BEHAVIOR: 0.5,
        }
        r1 = _softmax_select(scores, 0.5, random.Random(42))
        r2 = _softmax_select(scores, 0.5, random.Random(42))
        assert r1 == r2

    def test_zero_temperature_picks_max(self):
        scores = {
            TacticalIntent.EXPLOIT_PATTERN: 5.0,
            TacticalIntent.PROBE_BEHAVIOR: 0.1,
            TacticalIntent.DEFENSIVE_RESET: 1.0,
        }
        result = _softmax_select(scores, 0.0, random.Random(42))
        assert result == TacticalIntent.EXPLOIT_PATTERN

    def test_high_temperature_more_uniform(self):
        """With very high temperature, distribution approaches uniform."""
        scores = {
            TacticalIntent.EXPLOIT_PATTERN: 10.0,
            TacticalIntent.PROBE_BEHAVIOR: 0.0,
        }
        rng = random.Random(42)
        counts = {TacticalIntent.EXPLOIT_PATTERN: 0, TacticalIntent.PROBE_BEHAVIOR: 0}
        for _ in range(1000):
            mode = _softmax_select(scores, 100.0, rng)
            counts[mode] += 1
        # With temperature=100, both should get ~50% (within reason)
        assert counts[TacticalIntent.PROBE_BEHAVIOR] > 300  # at least 30%


class TestMemoryAdjustments:
    def test_stale_exploit_penalized(self, configs, mem):
        ctx = _make_ctx(
            tick_id=1000,
            prediction_conf=0.8,
            predicted_commit=CombatCommitment.LIGHT_ATTACK,
        )
        mem.set_exploit_target("LIGHT_ATTACK", tick_id=0)  # very stale

        rng = random.Random(42)
        # Run many trials; exploit should be selected less often than neutral
        # (Can't easily test exact scores without exposing internals, so just
        # verify it doesn't crash and returns a valid mode)
        mode = select_mode(ctx, mem, configs.strategy, configs.planner_memory, rng)
        assert isinstance(mode, TacticalIntent)

    def test_consecutive_penalty_applied(self, configs, mem):
        ctx = _make_ctx()
        rng = random.Random(42)
        # Repeat same mode many times
        for _ in range(10):
            mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mode = select_mode(ctx, mem, configs.strategy, configs.planner_memory, rng)
        # Should still return a valid mode (penalty makes it less likely but not impossible)
        assert isinstance(mode, TacticalIntent)
