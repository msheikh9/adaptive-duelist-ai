"""Tests for ActionResolver: mode → commitment resolution."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from ai.models.base_predictor import make_prediction_result
from ai.strategy.action_resolver import ResolverOutput, resolve
from ai.strategy.ai_context import AIContext
from ai.strategy.planner_memory import PlannerMemory
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
    spacing: SpacingZone = SpacingZone.MID,
    player_fsm: FSMState = FSMState.IDLE,
    ai_hp_frac: float = 1.0,
    ai_stamina_frac: float = 0.8,
    ai_facing: int = 1,
    predicted_commit: CombatCommitment | None = None,
    prediction_conf: float = 0.0,
    tick_id: int = 100,
) -> AIContext:
    if predicted_commit is not None and prediction_conf > 0:
        dist = {predicted_commit.name: prediction_conf}
        pred = make_prediction_result(dist, source="markov", markov_level="unigram")
    else:
        pred = make_prediction_result({}, source="none", markov_level="none")

    return AIContext(
        tick_id=tick_id,
        player_hp_frac=1.0,
        player_stamina_frac=0.8,
        player_fsm=player_fsm,
        player_commitment=None,
        ai_hp_frac=ai_hp_frac,
        ai_stamina_frac=ai_stamina_frac,
        ai_fsm=FSMState.IDLE,
        ai_commitment=None,
        ai_facing=ai_facing,
        spacing=spacing,
        distance_sub=20000,
        prediction=pred,
    )


class TestResolverOutput:
    def test_output_has_required_fields(self, configs, mem):
        ctx = _make_ctx()
        out = resolve(TacticalIntent.NEUTRAL_SPACING, ctx, mem,
                      configs.action_resolver)
        assert isinstance(out, ResolverOutput)
        assert isinstance(out.commitment, CombatCommitment)
        assert isinstance(out.positioning_bias, float)
        assert isinstance(out.commit_delay, int)
        assert isinstance(out.reason_tags, tuple)

    def test_positioning_bias_in_range(self, configs, mem):
        for mode in TacticalIntent:
            ctx = _make_ctx(predicted_commit=CombatCommitment.LIGHT_ATTACK,
                            prediction_conf=0.8)
            out = resolve(mode, ctx, mem, configs.action_resolver)
            assert -1.0 <= out.positioning_bias <= 1.0

    def test_commit_delay_bounded(self, configs, mem):
        for mode in TacticalIntent:
            ctx = _make_ctx(predicted_commit=CombatCommitment.LIGHT_ATTACK,
                            prediction_conf=0.8)
            out = resolve(mode, ctx, mem, configs.action_resolver)
            assert 0 <= out.commit_delay <= configs.action_resolver.max_commit_delay_ticks


class TestExploitResolver:
    def test_counters_light_attack(self, configs, mem):
        ctx = _make_ctx(predicted_commit=CombatCommitment.LIGHT_ATTACK,
                        prediction_conf=0.9)
        out = resolve(TacticalIntent.EXPLOIT_PATTERN, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.DODGE_BACKWARD

    def test_counters_heavy_attack(self, configs, mem):
        ctx = _make_ctx(predicted_commit=CombatCommitment.HEAVY_ATTACK,
                        prediction_conf=0.9)
        out = resolve(TacticalIntent.EXPLOIT_PATTERN, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.LIGHT_ATTACK

    def test_no_prediction_defaults_to_attack(self, configs, mem):
        ctx = _make_ctx()  # no prediction
        out = resolve(TacticalIntent.EXPLOIT_PATTERN, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.LIGHT_ATTACK


class TestPunishRecoveryResolver:
    def test_heavy_attack_on_close_recovery(self, configs, mem):
        ctx = _make_ctx(
            player_fsm=FSMState.ATTACK_RECOVERY,
            spacing=SpacingZone.CLOSE,
            ai_stamina_frac=0.5,
        )
        out = resolve(TacticalIntent.PUNISH_RECOVERY, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.HEAVY_ATTACK

    def test_light_attack_on_mid_recovery(self, configs, mem):
        ctx = _make_ctx(
            player_fsm=FSMState.ATTACK_RECOVERY,
            spacing=SpacingZone.MID,
        )
        out = resolve(TacticalIntent.PUNISH_RECOVERY, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.LIGHT_ATTACK

    def test_advance_when_not_recovering(self, configs, mem):
        ctx = _make_ctx(
            player_fsm=FSMState.IDLE,
            spacing=SpacingZone.FAR,
            ai_facing=1,
        )
        out = resolve(TacticalIntent.PUNISH_RECOVERY, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_RIGHT


class TestDefensiveResolver:
    def test_dodge_at_close_range(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.CLOSE)
        out = resolve(TacticalIntent.DEFENSIVE_RESET, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.DODGE_BACKWARD
        assert out.positioning_bias < 0.0

    def test_retreat_at_mid_range(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.MID, ai_facing=1)
        out = resolve(TacticalIntent.DEFENSIVE_RESET, ctx, mem,
                      configs.action_resolver)
        # Should move away from player
        assert out.commitment == CombatCommitment.MOVE_LEFT


class TestPressureResolver:
    def test_attack_at_close(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.CLOSE)
        out = resolve(TacticalIntent.PRESSURE_STAMINA, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.LIGHT_ATTACK

    def test_advance_at_far(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.FAR, ai_facing=1)
        out = resolve(TacticalIntent.PRESSURE_STAMINA, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_RIGHT


class TestNeutralResolver:
    def test_retreat_at_close(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.CLOSE, ai_facing=1)
        out = resolve(TacticalIntent.NEUTRAL_SPACING, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_LEFT

    def test_advance_at_far(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.FAR, ai_facing=1)
        out = resolve(TacticalIntent.NEUTRAL_SPACING, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_RIGHT


class TestProbeResolver:
    def test_approach_at_far(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.FAR, ai_facing=1)
        out = resolve(TacticalIntent.PROBE_BEHAVIOR, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_RIGHT

    def test_provoke_at_close(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.CLOSE)
        out = resolve(TacticalIntent.PROBE_BEHAVIOR, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.LIGHT_ATTACK
        assert out.commit_delay > 0


class TestBaitResolver:
    def test_approach_at_far(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.FAR, ai_facing=1)
        out = resolve(TacticalIntent.BAIT_AND_PUNISH, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.MOVE_RIGHT

    def test_dodge_at_close(self, configs, mem):
        ctx = _make_ctx(spacing=SpacingZone.CLOSE)
        out = resolve(TacticalIntent.BAIT_AND_PUNISH, ctx, mem,
                      configs.action_resolver)
        assert out.commitment == CombatCommitment.DODGE_BACKWARD
