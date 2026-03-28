"""Stamina system: regeneration, continuous costs, and exhaustion.

Stamina is stored as an integer (0–max_stamina). Regeneration uses a
float accumulator to handle fractional regen rates without floating-point
position math. When the accumulator reaches ≥1.0, integer stamina
increments.

Movement cost is also tracked through the accumulator. Exhaustion triggers
when integer stamina hits 0 while moving.
"""

from __future__ import annotations

from game.combat.actions import FSMState
from game.combat.state_machine import enter_exhausted
from game.state import FighterState
from config.config_loader import GameConfig


def tick_stamina(fighter: FighterState, config: GameConfig) -> bool:
    """Update stamina for one tick. Returns True if exhaustion was triggered.

    Handles:
    - Regen while idle/moving/locked
    - Continuous cost while moving
    - Exhaustion entry when stamina reaches 0 during movement
    """
    max_stam = config.fighter.max_stamina
    fsm = fighter.fsm_state

    if fsm in (FSMState.KO, FSMState.EXHAUSTED):
        return False

    # Determine regen rate based on current state
    if fsm == FSMState.IDLE:
        regen = config.fighter.stamina_regen_idle
    elif fsm == FSMState.MOVING:
        regen = config.fighter.stamina_regen_moving
    else:
        regen = config.fighter.stamina_regen_attacking

    # Apply continuous movement cost before regen
    if fsm == FSMState.MOVING:
        move_cost = config.actions.move.stamina_cost_per_tick
        fighter.stamina_accumulator -= move_cost

    # Apply regen
    fighter.stamina_accumulator += regen

    # Convert accumulator to integer stamina changes
    if fighter.stamina_accumulator >= 1.0:
        add = int(fighter.stamina_accumulator)
        fighter.stamina = min(max_stam, fighter.stamina + add)
        fighter.stamina_accumulator -= add
    elif fighter.stamina_accumulator <= -1.0:
        sub = int(-fighter.stamina_accumulator)
        new_stamina = fighter.stamina - sub
        fighter.stamina_accumulator += sub

        if new_stamina <= 0 and fsm == FSMState.MOVING:
            fighter.stamina = 0
            fighter.stamina_accumulator = 0.0
            enter_exhausted(fighter, config.fighter.exhaustion_recovery_frames)
            return True

        fighter.stamina = max(0, new_stamina)

    return False
