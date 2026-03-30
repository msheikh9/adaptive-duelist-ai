"""Tests for ai/training/curriculum.py."""

from __future__ import annotations

import pytest

from ai.training.curriculum import CurriculumPlan, build_curriculum, _allocate
from ai.training.scripted_opponent import ScriptedProfile
from evaluation.weakness_analyzer import WeaknessReport


def _make_report(
    weak_labels: list[str] | None = None,
    weak_modes: list[str] | None = None,
    weak_zones: list[str] | None = None,
) -> WeaknessReport:
    return WeaknessReport(
        weak_prediction_labels=weak_labels or [],
        weak_tactical_modes=weak_modes or [],
        weak_spacing_zones=weak_zones or [],
        high_failure_scenarios=[],
    )


class TestProfileMapping:

    def test_weak_prediction_labels_selects_aggressive(self):
        report = _make_report(weak_labels=["HEAVY_ATTACK"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.AGGRESSIVE in plan.profiles

    def test_weak_defensive_modes_selects_defensive(self):
        report = _make_report(weak_modes=["DEFENSIVE_RESET"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.DEFENSIVE in plan.profiles

    def test_weak_exploit_modes_selects_patterned(self):
        report = _make_report(weak_modes=["EXPLOIT_PATTERN"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.PATTERNED in plan.profiles

    def test_weak_spacing_zones_selects_mixed(self):
        report = _make_report(weak_zones=["CLOSE"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.MIXED in plan.profiles

    def test_no_weakness_selects_mixed_only(self):
        report = _make_report()
        plan = build_curriculum(report, total_matches=10)
        assert plan.profiles == [ScriptedProfile.MIXED]

    def test_multiple_weaknesses_selects_multiple_profiles(self):
        report = _make_report(
            weak_labels=["LIGHT_ATTACK"],
            weak_modes=["DEFENSIVE_RESET", "EXPLOIT_PATTERN"],
        )
        plan = build_curriculum(report, total_matches=20)
        profile_values = {p.value for p in plan.profiles}
        assert "AGGRESSIVE" in profile_values
        assert "DEFENSIVE" in profile_values
        assert "PATTERNED" in profile_values

    def test_neutral_spacing_mode_maps_to_defensive(self):
        report = _make_report(weak_modes=["NEUTRAL_SPACING"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.DEFENSIVE in plan.profiles

    def test_bait_and_punish_mode_maps_to_patterned(self):
        report = _make_report(weak_modes=["BAIT_AND_PUNISH"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.PATTERNED in plan.profiles

    def test_punish_recovery_mode_maps_to_patterned(self):
        report = _make_report(weak_modes=["PUNISH_RECOVERY"])
        plan = build_curriculum(report, total_matches=10)
        assert ScriptedProfile.PATTERNED in plan.profiles

    def test_no_duplicate_profiles(self):
        # Both exploit modes present → only one PATTERNED
        report = _make_report(weak_modes=["EXPLOIT_PATTERN", "BAIT_AND_PUNISH", "PUNISH_RECOVERY"])
        plan = build_curriculum(report, total_matches=10)
        assert len(plan.profiles) == len(set(plan.profiles))


class TestAllocationSum:

    def test_allocation_sums_to_total_matches_single_profile(self):
        report = _make_report()
        for total in [1, 5, 10, 50, 100]:
            plan = build_curriculum(report, total_matches=total)
            assert sum(plan.match_allocation.values()) == total

    def test_allocation_sums_to_total_matches_multiple_profiles(self):
        report = _make_report(
            weak_labels=["HEAVY_ATTACK"],
            weak_modes=["DEFENSIVE_RESET", "EXPLOIT_PATTERN"],
            weak_zones=["CLOSE"],
        )
        for total in [4, 10, 20, 50, 99, 100]:
            plan = build_curriculum(report, total_matches=total)
            assert sum(plan.match_allocation.values()) == total, (
                f"total={total}, allocation={plan.match_allocation}"
            )

    def test_all_profiles_have_at_least_one_match(self):
        report = _make_report(
            weak_labels=["HEAVY_ATTACK"],
            weak_modes=["DEFENSIVE_RESET"],
        )
        plan = build_curriculum(report, total_matches=3)
        for profile in plan.profiles:
            assert plan.match_allocation.get(profile.value, 0) >= 1

    def test_allocation_keys_match_profiles(self):
        report = _make_report(weak_labels=["LIGHT_ATTACK"], weak_modes=["EXPLOIT_PATTERN"])
        plan = build_curriculum(report, total_matches=20)
        assert set(plan.match_allocation.keys()) == {p.value for p in plan.profiles}

    def test_total_matches_1(self):
        report = _make_report()
        plan = build_curriculum(report, total_matches=1)
        assert sum(plan.match_allocation.values()) == 1

    def test_invalid_total_raises(self):
        report = _make_report()
        with pytest.raises(ValueError):
            build_curriculum(report, total_matches=0)

    def test_invalid_negative_total_raises(self):
        report = _make_report()
        with pytest.raises(ValueError):
            build_curriculum(report, total_matches=-5)


class TestBiasLogic:

    def test_aggressive_gets_higher_allocation_than_mixed_when_weak_labels(self):
        """When weak_prediction_labels are set, AGGRESSIVE gets a boost (×1.5)."""
        report = _make_report(
            weak_labels=["HEAVY_ATTACK"],
            weak_zones=["CLOSE"],  # → both AGGRESSIVE and MIXED
        )
        plan = build_curriculum(report, total_matches=50)
        aggressive_alloc = plan.match_allocation.get("AGGRESSIVE", 0)
        mixed_alloc = plan.match_allocation.get("MIXED", 0)
        assert aggressive_alloc > mixed_alloc

    def test_patterned_boosted_when_exploit_modes_weak(self):
        """PATTERNED gets boost over MIXED when exploitation mode is weak."""
        report = _make_report(
            weak_modes=["EXPLOIT_PATTERN"],
            weak_zones=["MID"],  # → both PATTERNED and MIXED
        )
        plan = build_curriculum(report, total_matches=50)
        patterned_alloc = plan.match_allocation.get("PATTERNED", 0)
        mixed_alloc = plan.match_allocation.get("MIXED", 0)
        assert patterned_alloc > mixed_alloc

    def test_defensive_boosted_when_defensive_modes_weak(self):
        """DEFENSIVE gets boost over MIXED when defensive modes are weak."""
        report = _make_report(
            weak_modes=["DEFENSIVE_RESET"],
            weak_zones=["FAR"],  # → both DEFENSIVE and MIXED
        )
        plan = build_curriculum(report, total_matches=50)
        defensive_alloc = plan.match_allocation.get("DEFENSIVE", 0)
        mixed_alloc = plan.match_allocation.get("MIXED", 0)
        assert defensive_alloc > mixed_alloc

    def test_no_boost_when_no_weakness(self):
        """No weakness → only MIXED, no boost to apply."""
        report = _make_report()
        plan = build_curriculum(report, total_matches=20)
        assert plan.profiles == [ScriptedProfile.MIXED]
        assert plan.match_allocation["MIXED"] == 20

    def test_boosted_profile_outranks_unboosted_mixed(self):
        """Boosted AGGRESSIVE (weak labels) gets more matches than unboosted MIXED (weak zones)."""
        report = _make_report(
            weak_labels=["HEAVY_ATTACK"],   # → AGGRESSIVE boosted (×1.5)
            weak_zones=["CLOSE"],            # → MIXED selected, no extra boost
        )
        plan = build_curriculum(report, total_matches=50)
        # AGGRESSIVE weight=1.5, MIXED weight=1.0 → AGGRESSIVE gets more matches
        aggressive = plan.match_allocation.get("AGGRESSIVE", 0)
        mixed = plan.match_allocation.get("MIXED", 0)
        assert aggressive > mixed


class TestDeterminism:

    def test_same_inputs_same_plan(self):
        report = _make_report(
            weak_labels=["HEAVY_ATTACK"],
            weak_modes=["EXPLOIT_PATTERN", "DEFENSIVE_RESET"],
            weak_zones=["CLOSE"],
        )
        plan1 = build_curriculum(report, total_matches=30)
        plan2 = build_curriculum(report, total_matches=30)
        assert plan1.profiles == plan2.profiles
        assert plan1.match_allocation == plan2.match_allocation
        assert plan1.focus_targets == plan2.focus_targets

    def test_focus_targets_populated(self):
        report = _make_report(weak_labels=["HEAVY_ATTACK"])
        plan = build_curriculum(report, total_matches=10)
        assert len(plan.focus_targets) >= 1

    def test_focus_targets_empty_weakness_uses_fallback(self):
        report = _make_report()
        plan = build_curriculum(report, total_matches=10)
        assert any("no specific weakness" in t.lower() for t in plan.focus_targets)


class TestCurriculumPlanType:

    def test_returns_curriculum_plan(self):
        report = _make_report()
        plan = build_curriculum(report, total_matches=5)
        assert isinstance(plan, CurriculumPlan)
        assert isinstance(plan.profiles, list)
        assert isinstance(plan.match_allocation, dict)
        assert isinstance(plan.focus_targets, list)

    def test_profiles_are_scripted_profile_enum(self):
        report = _make_report(weak_labels=["HEAVY_ATTACK"])
        plan = build_curriculum(report, total_matches=10)
        for p in plan.profiles:
            assert isinstance(p, ScriptedProfile)
