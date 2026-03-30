"""Tests for ai/strategy/session_memory.py."""

from __future__ import annotations

import pytest

from ai.strategy.session_memory import SessionMemory
from ai.strategy.tactics import TacticalIntent


class TestSessionMemoryBasic:

    def test_fresh_memory_has_zero_samples(self):
        sm = SessionMemory()
        assert sm.total_samples() == 0.0

    def test_mode_success_rate_returns_none_when_empty(self):
        sm = SessionMemory()
        assert sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) is None

    def test_mode_success_rate_returns_none_below_min_samples(self):
        sm = SessionMemory(min_samples=10)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 4)})
        # total = 4 < 10
        assert sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) is None

    def test_mode_success_rate_returns_value_at_threshold(self):
        sm = SessionMemory(min_samples=5)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 5)})
        rate = sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN)
        assert rate == pytest.approx(3 / 5)

    def test_total_samples_sums_all_modes(self):
        sm = SessionMemory()
        sm.record_match_outcomes({
            "EXPLOIT_PATTERN": (2, 4),
            "DEFENSIVE_RESET": (1, 3),
        })
        assert sm.total_samples() == pytest.approx(7.0)

    def test_reset_clears_all_data(self):
        sm = SessionMemory()
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 5)})
        sm.reset()
        assert sm.total_samples() == 0.0
        assert sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) is None

    def test_unknown_mode_returns_none_even_when_threshold_met(self):
        sm = SessionMemory(min_samples=3)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (2, 5)})
        # Threshold met for total, but DEFENSIVE_RESET has no data
        assert sm.mode_success_rate(TacticalIntent.DEFENSIVE_RESET) is None


class TestDecay:

    def test_decay_reduces_total_samples(self):
        sm = SessionMemory(decay_factor=0.5)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (4, 8)})
        before = sm.total_samples()
        sm.record_match_outcomes({})  # empty second match still applies decay
        after = sm.total_samples()
        assert after == pytest.approx(before * 0.5)

    def test_decay_factor_1_preserves_data(self):
        sm = SessionMemory(decay_factor=1.0)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 6)})
        sm.record_match_outcomes({})
        assert sm.total_samples() == pytest.approx(6.0)

    def test_decay_factor_0_clears_old_data(self):
        sm = SessionMemory(decay_factor=0.0)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 6)})
        sm.record_match_outcomes({"DEFENSIVE_RESET": (1, 2)})
        # Old EXPLOIT_PATTERN data decayed to 0; only new DEFENSIVE_RESET remains
        assert sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN) is None
        assert sm.total_samples() == pytest.approx(2.0)

    def test_decay_preserves_success_rate_magnitude(self):
        """Decay should not change the success RATE, only the weight."""
        sm = SessionMemory(decay_factor=0.5, min_samples=1)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (4, 8)})
        # Apply decay via empty match
        sm.record_match_outcomes({})
        rate = sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN)
        assert rate == pytest.approx(0.5)

    def test_multiple_matches_accumulate_with_decay(self):
        """After 3 matches the oldest has the smallest contribution."""
        sm = SessionMemory(decay_factor=0.8, min_samples=1)
        # Match 1: 10/10 → weight 0.64 after two more decays
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (10, 10)})
        # Match 2: 0/10 → weight 0.8
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (0, 10)})
        # Match 3: 0/10 → weight 1.0 (most recent, no decay applied yet)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (0, 10)})

        total = sm.total_samples()
        # Expected: 10*0.64 + 10*0.8 + 10 = 24.4
        assert total == pytest.approx(10 * 0.64 + 10 * 0.8 + 10, rel=1e-6)


class TestDeterminism:

    def test_same_inputs_same_output(self):
        sm1 = SessionMemory(decay_factor=0.8, min_samples=5)
        sm2 = SessionMemory(decay_factor=0.8, min_samples=5)
        outcomes = [
            {"EXPLOIT_PATTERN": (3, 5), "DEFENSIVE_RESET": (2, 4)},
            {"BAIT_AND_PUNISH": (1, 3)},
        ]
        for o in outcomes:
            sm1.record_match_outcomes(o)
            sm2.record_match_outcomes(o)

        for mode in TacticalIntent:
            r1 = sm1.mode_success_rate(mode)
            r2 = sm2.mode_success_rate(mode)
            assert r1 == r2

    def test_mode_stats_keys_match_recorded_modes(self):
        sm = SessionMemory()
        sm.record_match_outcomes({
            "EXPLOIT_PATTERN": (2, 4),
            "PROBE_BEHAVIOR": (1, 2),
        })
        stats = sm.mode_stats()
        assert set(stats.keys()) == {"EXPLOIT_PATTERN", "PROBE_BEHAVIOR"}

    def test_all_tactical_intents_can_be_queried(self):
        """No mode raises on mode_success_rate — they all return None or float."""
        sm = SessionMemory(min_samples=3)
        sm.record_match_outcomes({m.name: (1, 2) for m in TacticalIntent})
        for mode in TacticalIntent:
            result = sm.mode_success_rate(mode)
            assert result is None or isinstance(result, float)


class TestEdgeCases:

    def test_zero_total_in_outcome_ignored(self):
        sm = SessionMemory(min_samples=1)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (0, 0)})
        assert sm.total_samples() == 0.0

    def test_success_greater_than_total_not_validated(self):
        """SessionMemory doesn't validate inputs — caller is responsible."""
        sm = SessionMemory(min_samples=1)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (5, 3)})
        rate = sm.mode_success_rate(TacticalIntent.EXPLOIT_PATTERN)
        # Rate > 1.0 is caller's responsibility; no crash expected
        assert rate is not None

    def test_record_match_outcomes_empty_dict_applies_decay(self):
        sm = SessionMemory(decay_factor=0.5, min_samples=1)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (4, 8)})
        sm.record_match_outcomes({})
        assert sm.total_samples() == pytest.approx(4.0)
