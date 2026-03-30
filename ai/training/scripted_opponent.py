"""Scripted opponents for self-play training data generation.

Five deterministic profiles simulate distinct player archetypes.
All profiles use only PHASE_1_COMMITMENTS — no Phase 2/3 actions.
No Pygame imports. Uses attempt_commitment() for FSM compliance.
"""
from __future__ import annotations
import enum
import random
from typing import TYPE_CHECKING

from game.combat.actions import CombatCommitment, PHASE_1_COMMITMENTS
from game.combat.state_machine import can_commit
from game.entities.fighter import attempt_commitment

if TYPE_CHECKING:
    from config.config_loader import GameConfig
    from game.state import FighterState, SimulationState


class ScriptedProfile(enum.Enum):
    RANDOM     = "RANDOM"
    AGGRESSIVE = "AGGRESSIVE"
    DEFENSIVE  = "DEFENSIVE"
    PATTERNED  = "PATTERNED"
    MIXED      = "MIXED"


# Weight tables over PHASE_1_COMMITMENTS only
_WEIGHTS: dict[ScriptedProfile, dict[CombatCommitment, float]] = {
    ScriptedProfile.RANDOM: {
        CombatCommitment.MOVE_LEFT:      1.0,
        CombatCommitment.MOVE_RIGHT:     1.0,
        CombatCommitment.LIGHT_ATTACK:   1.0,
        CombatCommitment.HEAVY_ATTACK:   1.0,
        CombatCommitment.DODGE_BACKWARD: 1.0,
    },
    ScriptedProfile.AGGRESSIVE: {
        CombatCommitment.MOVE_LEFT:      1.0,
        CombatCommitment.MOVE_RIGHT:     1.5,
        CombatCommitment.LIGHT_ATTACK:   4.0,
        CombatCommitment.HEAVY_ATTACK:   3.0,
        CombatCommitment.DODGE_BACKWARD: 0.5,
    },
    ScriptedProfile.DEFENSIVE: {
        CombatCommitment.MOVE_LEFT:      1.5,
        CombatCommitment.MOVE_RIGHT:     1.5,
        CombatCommitment.LIGHT_ATTACK:   0.5,
        CombatCommitment.HEAVY_ATTACK:   0.3,
        CombatCommitment.DODGE_BACKWARD: 4.0,
    },
}

_PATTERNED_CYCLE = [
    CombatCommitment.LIGHT_ATTACK,
    CombatCommitment.LIGHT_ATTACK,
    CombatCommitment.HEAVY_ATTACK,
    CombatCommitment.DODGE_BACKWARD,
    CombatCommitment.MOVE_RIGHT,
]

_MIXABLE = [
    ScriptedProfile.RANDOM,
    ScriptedProfile.AGGRESSIVE,
    ScriptedProfile.DEFENSIVE,
    ScriptedProfile.PATTERNED,
]

DECISION_INTERVAL_MIN = 10
DECISION_INTERVAL_MAX = 40


class ScriptedOpponent:
    """Deterministic scripted fighter for the player slot in self-play.

    Args:
        profile:          Scripted behaviour profile.
        seed:             RNG seed for determinism.
        focus_commitment: Optional commitment to emphasise (weight ×2.5).
                          Must be in PHASE_1_COMMITMENTS; ignored otherwise.
    """

    def __init__(
        self,
        profile: ScriptedProfile,
        seed: int = 0,
        focus_commitment: CombatCommitment | None = None,
    ) -> None:
        self._base_profile = profile
        self._active_profile = profile
        self._rng = random.Random(seed)
        self._pattern_idx = 0
        self._ticks_until_decision = self._sample_interval()
        # Validate: only Phase 1 commitments may be focused
        if focus_commitment is not None and focus_commitment not in PHASE_1_COMMITMENTS:
            focus_commitment = None
        self._focus_commitment: CombatCommitment | None = focus_commitment

    def _sample_interval(self) -> int:
        return self._rng.randint(DECISION_INTERVAL_MIN, DECISION_INTERVAL_MAX)

    def decide(
        self,
        fighter: "FighterState",
        sim: "SimulationState",
        config: "GameConfig",
    ) -> CombatCommitment | None:
        if not fighter.is_alive:
            return None
        if fighter.is_locked:
            return None
        self._ticks_until_decision -= 1
        if self._ticks_until_decision > 0:
            return None
        self._ticks_until_decision = self._sample_interval()

        # collect valid Phase 1 commitments
        valid = [c for c in PHASE_1_COMMITMENTS if can_commit(fighter, c, config)]
        if not valid:
            return None

        chosen = self._choose(valid, fighter, sim, config)
        if chosen is None:
            return None
        if attempt_commitment(fighter, chosen, config):
            return chosen
        return None

    def _choose(self, valid: list, fighter, sim, config) -> CombatCommitment | None:
        profile = self._active_profile
        if profile == ScriptedProfile.PATTERNED:
            # Try to pick next in cycle; if not valid, fallback to random
            for _ in range(len(_PATTERNED_CYCLE)):
                candidate = _PATTERNED_CYCLE[self._pattern_idx % len(_PATTERNED_CYCLE)]
                self._pattern_idx += 1
                if candidate in valid:
                    return candidate
            # fallback
            return self._rng.choice(valid)

        weights_table = _WEIGHTS.get(profile, _WEIGHTS[ScriptedProfile.RANDOM])
        weights = [weights_table.get(c, 1.0) for c in valid]

        # Distance bias for AGGRESSIVE: prefer advancing when far
        if profile == ScriptedProfile.AGGRESSIVE and sim is not None:
            scale = config.simulation.sub_pixel_scale
            far_threshold = config.spacing.mid_max * scale
            if sim.distance_sub() > far_threshold:
                advance = (CombatCommitment.MOVE_RIGHT
                           if fighter.facing > 0
                           else CombatCommitment.MOVE_LEFT)
                for i, c in enumerate(valid):
                    if c == advance:
                        weights[i] *= 3.0

        # Apply focus_commitment boost (×2.5) when set
        if self._focus_commitment is not None and self._focus_commitment in valid:
            focus_idx = valid.index(self._focus_commitment)
            weights[focus_idx] *= 2.5

        return self._rng.choices(valid, weights=weights, k=1)[0]

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self._pattern_idx = 0
        self._ticks_until_decision = self._sample_interval()
        if self._base_profile == ScriptedProfile.MIXED:
            self._active_profile = self._rng.choice(_MIXABLE)
        else:
            self._active_profile = self._base_profile
