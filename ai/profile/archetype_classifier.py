"""Player archetype classification from behavioral profile.

A pure, deterministic function over PlayerProfile fields. No DB access,
no randomness, no side effects.

Archetypes:
  AGGRESSIVE  — high attack rate (aggression_index >= 0.50)
  DEFENSIVE   — low attack rate + moderate dodge rate
  PATTERNED   — concentrated action distribution (top-2 actions >= 70%)
  EVASIVE     — very high dodge frequency (>= 0.40)
  BALANCED    — no dominant style detected, or insufficient data
"""

from __future__ import annotations

import enum

from ai.profile.player_profile import PlayerProfile

# Classification thresholds
_EVASIVE_DODGE_THRESHOLD     = 0.40   # dodge_frequency >= this → EVASIVE
_AGGRESSIVE_AGG_THRESHOLD    = 0.50   # aggression_index >= this → AGGRESSIVE
_DEFENSIVE_AGG_CEILING       = 0.25   # aggression_index < this ...
_DEFENSIVE_DODGE_FLOOR       = 0.25   # ... and dodge_frequency >= this → DEFENSIVE
_PATTERNED_TOP2_THRESHOLD    = 0.70   # top-2 actions cover >= this share → PATTERNED
_MIN_COMMITMENTS_TO_CLASSIFY = 5      # fewer total commitments → BALANCED


class ArchetypeLabel(enum.Enum):
    AGGRESSIVE = "AGGRESSIVE"
    DEFENSIVE  = "DEFENSIVE"
    PATTERNED  = "PATTERNED"
    EVASIVE    = "EVASIVE"
    BALANCED   = "BALANCED"


def classify_archetype(profile: PlayerProfile) -> ArchetypeLabel:
    """Infer the dominant player archetype from a PlayerProfile.

    Rules are evaluated in priority order (most specific first).
    Returns BALANCED if the profile has too few observations or no
    dominant style is detected.

    Args:
        profile:  PlayerProfile instance. Only existing public fields are
                  read.  Never modifies the profile.

    Returns:
        An ArchetypeLabel. Fully deterministic for the same profile state.
    """
    if profile.total_commitments < _MIN_COMMITMENTS_TO_CLASSIFY:
        return ArchetypeLabel.BALANCED

    agg   = profile.aggression_index
    dodge = profile.dodge_frequency
    top2  = _top2_share(profile.action_frequencies)

    # Priority order — EVASIVE overrides AGGRESSIVE when dodge dominates
    if dodge >= _EVASIVE_DODGE_THRESHOLD:
        return ArchetypeLabel.EVASIVE

    if agg >= _AGGRESSIVE_AGG_THRESHOLD:
        return ArchetypeLabel.AGGRESSIVE

    if agg < _DEFENSIVE_AGG_CEILING and dodge >= _DEFENSIVE_DODGE_FLOOR:
        return ArchetypeLabel.DEFENSIVE

    if top2 >= _PATTERNED_TOP2_THRESHOLD:
        return ArchetypeLabel.PATTERNED

    return ArchetypeLabel.BALANCED


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _top2_share(action_frequencies: dict[str, int]) -> float:
    """Fraction of total commitments covered by the two most-used actions."""
    if not action_frequencies:
        return 0.0
    total = sum(action_frequencies.values())
    if total == 0:
        return 0.0
    top2 = sum(sorted(action_frequencies.values(), reverse=True)[:2])
    return top2 / total
