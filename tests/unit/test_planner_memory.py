"""Tests for PlannerMemory: mode tracking, prediction accuracy,
exploit staleness, exploration budget, shift detection."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from ai.strategy.planner_memory import (
    ExploitTarget,
    ModeOutcome,
    PlannerMemory,
    PredictionOutcome,
    RollingRate,
)
from ai.strategy.tactics import TacticalIntent


@pytest.fixture
def cfg():
    _, ai_cfg, _ = load_config()
    return ai_cfg.planner_memory


@pytest.fixture
def mem(cfg):
    return PlannerMemory(cfg)


class TestRollingRate:
    def test_empty_rate_is_zero(self):
        r = RollingRate(10)
        assert r.rate == 0.0
        assert r.count == 0

    def test_all_correct(self):
        r = RollingRate(5)
        for _ in range(5):
            r.push(True)
        assert r.rate == 1.0

    def test_half_correct(self):
        r = RollingRate(10)
        for _ in range(5):
            r.push(True)
        for _ in range(5):
            r.push(False)
        assert abs(r.rate - 0.5) < 1e-9

    def test_window_rolls(self):
        r = RollingRate(3)
        r.push(True)
        r.push(True)
        r.push(True)
        assert r.rate == 1.0
        r.push(False)  # drops first True
        assert abs(r.rate - 2 / 3) < 1e-9

    def test_clear(self):
        r = RollingRate(5)
        r.push(True)
        r.clear()
        assert r.count == 0


class TestModeTracking:
    def test_record_mode_appends(self, mem):
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        assert list(mem.recent_modes) == [TacticalIntent.EXPLOIT_PATTERN]

    def test_consecutive_count(self, mem):
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        assert mem.consecutive_same_mode == 3

    def test_consecutive_resets_on_different_mode(self, mem):
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mem.record_mode(TacticalIntent.DEFENSIVE_RESET)
        assert mem.consecutive_same_mode == 1

    def test_per_mode_success_rate_default(self, mem):
        # No data → neutral prior
        assert mem.per_mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) == 0.5

    def test_per_mode_success_rate_computed(self, mem):
        for i in range(4):
            mem.record_outcome(ModeOutcome(
                mode=TacticalIntent.EXPLOIT_PATTERN,
                tick_id=i * 10,
                success=i < 3,  # 3 successes out of 4
            ))
        assert abs(mem.per_mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) - 0.75) < 1e-9


class TestExplorationBudget:
    def test_budget_starts_at_one(self, mem):
        assert mem.exploration_budget == 1.0

    def test_non_probe_drains_budget(self, mem, cfg):
        initial = mem.exploration_budget
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        assert mem.exploration_budget < initial
        assert abs(mem.exploration_budget - (1.0 - cfg.exploration_budget_drain_rate)) < 1e-9

    def test_probe_regenerates_budget(self, mem, cfg):
        # First drain
        for _ in range(10):
            mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        drained = mem.exploration_budget
        # Then probe
        mem.record_mode(TacticalIntent.PROBE_BEHAVIOR)
        assert mem.exploration_budget > drained

    def test_budget_has_floor(self, mem, cfg):
        for _ in range(200):
            mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        assert mem.exploration_budget >= cfg.exploration_budget_floor


class TestExploitStaleness:
    def test_no_target_zero_staleness(self, mem):
        assert mem.exploit_staleness(100) == 0.0

    def test_fresh_target_low_staleness(self, mem, cfg):
        mem.set_exploit_target("LIGHT_ATTACK", tick_id=100)
        assert mem.exploit_staleness(110) < 0.1

    def test_old_target_high_staleness(self, mem, cfg):
        mem.set_exploit_target("LIGHT_ATTACK", tick_id=0)
        staleness = mem.exploit_staleness(cfg.exploit_staleness_threshold_ticks)
        assert abs(staleness - 1.0) < 1e-6

    def test_staleness_capped_at_one(self, mem, cfg):
        mem.set_exploit_target("LIGHT_ATTACK", tick_id=0)
        assert mem.exploit_staleness(99999) == 1.0

    def test_clear_exploit_target(self, mem):
        mem.set_exploit_target("LIGHT_ATTACK", tick_id=0)
        mem.clear_exploit_target()
        assert mem.current_exploit_target is None
        assert mem.exploit_staleness(100) == 0.0


class TestPredictionAccuracy:
    def test_recording_prediction(self, mem):
        mem.record_prediction(PredictionOutcome(
            predicted_label="LIGHT_ATTACK",
            actual_label="LIGHT_ATTACK",
            confidence=0.8,
            tick_id=10,
        ))
        assert mem.prediction_accuracy_short == 1.0

    def test_mixed_accuracy(self, mem):
        mem.record_prediction(PredictionOutcome("L", "L", 0.8, 10))
        mem.record_prediction(PredictionOutcome("L", "H", 0.8, 20))
        assert abs(mem.prediction_accuracy_short - 0.5) < 1e-9

    def test_accuracy_trend_zero_initially(self, mem):
        assert mem.accuracy_trend == 0.0

    def test_accuracy_trend_computed(self, mem):
        # Feed enough for both windows
        for i in range(15):
            mem.record_prediction(PredictionOutcome("L", "L", 0.8, i))
        # Now feed wrong predictions (recent = bad, medium includes good)
        for i in range(10):
            mem.record_prediction(PredictionOutcome("L", "H", 0.8, 15 + i))
        # Short accuracy should be worse than medium → negative trend
        assert mem.accuracy_trend < 0.0


class TestShiftDetection:
    def test_no_shift_initially(self, mem):
        assert not mem.shift_detected

    def test_shift_detected_on_accuracy_drop(self, mem, cfg):
        # Build up good accuracy (fill short window of 10)
        for i in range(10):
            mem.record_prediction(PredictionOutcome("L", "L", 0.8, i))
        # At this point _prev_accuracy = 1.0 (all correct in short window)
        # Now push 10 wrong predictions all at once on a single tick
        # to create a sharp drop
        for i in range(10):
            mem.record_prediction(PredictionOutcome("L", "H", 0.8, 100 + i))
        # After 10 wrong, short accuracy = 0.0, prev was 1.0 → drop = 1.0
        assert mem.shift_detected

    def test_shift_clears_exploit_target(self, mem, cfg):
        mem.set_exploit_target("LIGHT_ATTACK", 0)
        for i in range(10):
            mem.record_prediction(PredictionOutcome("L", "L", 0.8, i))
        for i in range(10):
            mem.record_prediction(PredictionOutcome("L", "H", 0.8, 100 + i))
        if mem.shift_detected:
            assert mem.current_exploit_target is None


class TestReset:
    def test_reset_clears_all(self, mem):
        mem.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        mem.set_exploit_target("LIGHT_ATTACK", 0)
        mem.record_prediction(PredictionOutcome("L", "L", 0.8, 0))
        mem.reset()
        assert len(mem.recent_modes) == 0
        assert mem.current_exploit_target is None
        assert mem.exploration_budget == 1.0
        assert mem.consecutive_same_mode == 0
        assert not mem.shift_detected
