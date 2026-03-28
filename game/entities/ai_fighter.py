"""AI-controlled fighter with a simple baseline policy.

The baseline AI makes random valid commitments at random intervals.
It respects all FSM and stamina constraints — no cheating.
This will be replaced by the full AI pipeline in Phase 5.
"""

from __future__ import annotations

import random

from game.combat.actions import CombatCommitment, FSMState, PHASE_1_COMMITMENTS
from game.combat.state_machine import can_commit
from game.entities.fighter import attempt_commitment
from game.state import FighterState, SimulationState
from config.config_loader import GameConfig

# Weighted probabilities for the baseline AI's random choices.
# Higher weight = more likely to be chosen.
BASELINE_WEIGHTS: dict[CombatCommitment, float] = {
    CombatCommitment.MOVE_LEFT: 2.0,
    CombatCommitment.MOVE_RIGHT: 2.0,
    CombatCommitment.LIGHT_ATTACK: 3.0,
    CombatCommitment.HEAVY_ATTACK: 1.5,
    CombatCommitment.DODGE_BACKWARD: 1.0,
}

# When idle, the AI waits this many ticks (on average) before acting.
# Uniformly sampled from [MIN, MAX].
DECISION_INTERVAL_MIN = 10
DECISION_INTERVAL_MAX = 40


class BaselineAIController:
    """Simple random-policy AI controller.

    Uses a seeded RNG for deterministic behavior during replay.
    """

    def __init__(self, rng_seed: int) -> None:
        self._rng = random.Random(rng_seed)
        self._ticks_until_decision = self._sample_interval()

    def _sample_interval(self) -> int:
        return self._rng.randint(DECISION_INTERVAL_MIN, DECISION_INTERVAL_MAX)

    def decide(self, ai_state: FighterState, sim: SimulationState,
               config: GameConfig) -> CombatCommitment | None:
        """Choose an action for this tick. Returns commitment if one was entered."""
        if not ai_state.is_alive:
            return None

        # If locked, do nothing — FSM will complete the current commitment
        if ai_state.is_locked:
            return None

        # Countdown until next decision
        self._ticks_until_decision -= 1
        if self._ticks_until_decision > 0:
            return None

        self._ticks_until_decision = self._sample_interval()

        # Collect valid commitments
        valid = []
        weights = []
        for commitment in PHASE_1_COMMITMENTS:
            if can_commit(ai_state, commitment, config):
                valid.append(commitment)
                weights.append(BASELINE_WEIGHTS.get(commitment, 1.0))

        if not valid:
            return None

        # Bias toward advancing when far from player
        distance_sub = sim.distance_sub()
        far_threshold = config.spacing.mid_max * config.simulation.sub_pixel_scale
        if distance_sub > far_threshold:
            # Prefer moving toward player
            advance = (CombatCommitment.MOVE_RIGHT
                       if ai_state.facing > 0
                       else CombatCommitment.MOVE_LEFT)
            for i, c in enumerate(valid):
                if c == advance:
                    weights[i] *= 3.0

        chosen = self._rng.choices(valid, weights=weights, k=1)[0]

        if attempt_commitment(ai_state, chosen, config):
            return chosen

        return None

    def reset(self, rng_seed: int) -> None:
        self._rng = random.Random(rng_seed)
        self._ticks_until_decision = self._sample_interval()
