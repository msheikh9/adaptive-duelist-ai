"""Curriculum planning: map AI weaknesses to scripted opponent profiles.

Translates a WeaknessReport into a CurriculumPlan that drives the
self-play runner to generate targeted training data.

Mapping rules (spec):
    weak prediction  → AGGRESSIVE  (frequent varied commitments)
    weak defense     → DEFENSIVE   (lots of DODGE_BACKWARD stress)
    poor exploitation→ PATTERNED   (deterministic patterns to exploit)
    unknown/mixed    → MIXED
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.training.scripted_opponent import ScriptedProfile
from evaluation.weakness_analyzer import WeaknessReport

# Tactical modes considered "defensive" — suggest DEFENSIVE opponent
_DEFENSIVE_MODES = frozenset({
    "DEFENSIVE_RESET",
    "NEUTRAL_SPACING",
})

# Tactical modes considered "exploitation" — suggest PATTERNED opponent
_EXPLOIT_MODES = frozenset({
    "EXPLOIT_PATTERN",
    "BAIT_AND_PUNISH",
    "PUNISH_RECOVERY",
})

# Minimum fraction of total_matches assigned to any active profile
_MIN_PROFILE_FRACTION = 0.10


@dataclass
class CurriculumPlan:
    """Describes which profiles to use and how many matches to run per profile."""

    profiles: list[ScriptedProfile]
    match_allocation: dict[str, int]   # profile.value → match count
    focus_targets: list[str]           # human-readable weakness targets


def build_curriculum(
    weakness: WeaknessReport,
    total_matches: int,
) -> CurriculumPlan:
    """Map a WeaknessReport to a CurriculumPlan.

    Allocation rules:
    - Start with equal base shares among selected profiles.
    - Boost profiles that target detected weaknesses.
    - Allocation sums exactly to total_matches.
    - At least one profile is always selected.
    - Deterministic: same inputs → same output.
    """
    if total_matches <= 0:
        raise ValueError(f"total_matches must be positive, got {total_matches}")

    # --- Select profiles based on weaknesses ---
    selected: list[ScriptedProfile] = []
    focus_targets: list[str] = []

    # Weak prediction labels → AGGRESSIVE generates varied attack commitments
    if weakness.weak_prediction_labels:
        selected.append(ScriptedProfile.AGGRESSIVE)
        focus_targets.append(
            f"prediction labels {weakness.weak_prediction_labels}"
        )

    # Weak defensive modes → DEFENSIVE stresses the AI's defense
    if any(m in _DEFENSIVE_MODES for m in weakness.weak_tactical_modes):
        selected.append(ScriptedProfile.DEFENSIVE)
        focus_targets.append(
            f"defensive modes {[m for m in weakness.weak_tactical_modes if m in _DEFENSIVE_MODES]}"
        )

    # Weak exploitation modes → PATTERNED gives exploitable patterns
    if any(m in _EXPLOIT_MODES for m in weakness.weak_tactical_modes):
        selected.append(ScriptedProfile.PATTERNED)
        focus_targets.append(
            f"exploitation modes {[m for m in weakness.weak_tactical_modes if m in _EXPLOIT_MODES]}"
        )

    # Weak spacing zones → MIXED covers diverse spacing scenarios
    if weakness.weak_spacing_zones:
        selected.append(ScriptedProfile.MIXED)
        focus_targets.append(
            f"spacing zones {weakness.weak_spacing_zones}"
        )

    # Fallback: no clear weakness detected → MIXED
    if not selected:
        selected.append(ScriptedProfile.MIXED)
        focus_targets.append("no specific weakness detected — using MIXED")

    # Deduplicate while preserving order (deterministic)
    seen: set[ScriptedProfile] = set()
    unique_profiles: list[ScriptedProfile] = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            unique_profiles.append(p)

    # --- Build allocation ---
    allocation = _allocate(unique_profiles, total_matches, weakness)

    return CurriculumPlan(
        profiles=unique_profiles,
        match_allocation=allocation,
        focus_targets=focus_targets,
    )


def _allocate(
    profiles: list[ScriptedProfile],
    total_matches: int,
    weakness: WeaknessReport,
) -> dict[str, int]:
    """Distribute total_matches across profiles.

    Applies a boost factor for profiles directly targeting detected weaknesses,
    then floors each to at least 1 match, and normalises to sum exactly.
    """
    n = len(profiles)

    # Compute raw weights — boost profiles that map to known weaknesses
    weights: dict[str, float] = {}
    for p in profiles:
        weights[p.value] = 1.0

    if ScriptedProfile.AGGRESSIVE in profiles and weakness.weak_prediction_labels:
        weights[ScriptedProfile.AGGRESSIVE.value] *= 1.5

    if ScriptedProfile.PATTERNED in profiles and any(
        m in _EXPLOIT_MODES for m in weakness.weak_tactical_modes
    ):
        weights[ScriptedProfile.PATTERNED.value] *= 1.5

    if ScriptedProfile.DEFENSIVE in profiles and any(
        m in _DEFENSIVE_MODES for m in weakness.weak_tactical_modes
    ):
        weights[ScriptedProfile.DEFENSIVE.value] *= 1.5

    total_weight = sum(weights.values())

    # Convert to integer counts — floor first, then distribute remainder
    raw: dict[str, float] = {
        p.value: (weights[p.value] / total_weight) * total_matches
        for p in profiles
    }
    counts: dict[str, int] = {k: max(1, int(v)) for k, v in raw.items()}
    current_sum = sum(counts.values())
    remainder = total_matches - current_sum

    if remainder != 0:
        # Distribute remainder to profiles with largest fractional parts,
        # sorted by profile name for determinism.
        fractional = sorted(
            profiles,
            key=lambda p: (-(raw[p.value] - int(raw[p.value])), p.value),
        )
        for i in range(abs(remainder)):
            p = fractional[i % len(fractional)]
            if remainder > 0:
                counts[p.value] += 1
            else:
                if counts[p.value] > 1:
                    counts[p.value] -= 1

    assert sum(counts.values()) == total_matches, (
        f"Allocation sum {sum(counts.values())} != {total_matches}"
    )
    return counts
