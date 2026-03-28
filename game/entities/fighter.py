"""Fighter base: shared logic for translating commitments to FSM entries.

Both PlayerFighter and AIFighter delegate to this module to attempt
commitments through the FSM. This ensures both sides obey identical rules.
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment
from game.combat.state_machine import can_commit, enter_commitment, stop_moving
from game.state import FighterState
from config.config_loader import GameConfig


def attempt_commitment(fighter: FighterState, commitment: CombatCommitment,
                       config: GameConfig) -> bool:
    """Attempt to enter a commitment. Returns True if accepted by FSM."""
    if not can_commit(fighter, commitment, config):
        return False
    enter_commitment(fighter, commitment, config)
    return True


def attempt_stop_move(fighter: FighterState) -> None:
    """Stop moving if currently moving."""
    stop_moving(fighter)
