"""Compact frozen context snapshot for tactical planning.

AIContext captures the scalar game state needed for strategy selection
and action resolution. It is built once per planning cycle from the
mutable SimulationState — no deep copy, just scalar extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.models.base_predictor import PredictionResult
from game.combat.actions import CombatCommitment, FSMState, SpacingZone


@dataclass(frozen=True)
class AIContext:
    """Immutable snapshot of state relevant to AI planning."""

    tick_id: int

    # Player state
    player_hp_frac: float
    player_stamina_frac: float
    player_fsm: FSMState
    player_commitment: CombatCommitment | None

    # AI state
    ai_hp_frac: float
    ai_stamina_frac: float
    ai_fsm: FSMState
    ai_commitment: CombatCommitment | None
    ai_facing: int

    # Spatial
    spacing: SpacingZone
    distance_sub: int

    # Current prediction
    prediction: PredictionResult

    # Derived convenience
    @property
    def hp_advantage(self) -> float:
        """Positive = AI has more HP fraction."""
        return self.ai_hp_frac - self.player_hp_frac

    @property
    def stamina_advantage(self) -> float:
        return self.ai_stamina_frac - self.player_stamina_frac

    @property
    def player_is_locked(self) -> bool:
        return self.player_fsm not in (FSMState.IDLE, FSMState.MOVING)

    @property
    def ai_is_free(self) -> bool:
        return self.ai_fsm in (FSMState.IDLE, FSMState.MOVING)

    @property
    def player_is_recovering(self) -> bool:
        return self.player_fsm in (
            FSMState.ATTACK_RECOVERY,
            FSMState.DODGE_RECOVERY,
            FSMState.EXHAUSTED,
        )
