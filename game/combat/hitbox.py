"""Hitbox definitions for attack commitments.

Hitboxes are axis-aligned rectangles anchored to the attacker's position.
They are only active during ATTACK_ACTIVE frames. Each attack type has
a reach (horizontal extent in front of the attacker) derived from config.

All coordinates are in sub-pixel units.
"""

from __future__ import annotations

from dataclasses import dataclass

from game.combat.actions import CombatCommitment, FSMState
from game.state import FighterState
from config.config_loader import GameConfig


@dataclass(frozen=True, slots=True)
class Hitbox:
    """Axis-aligned rectangle in sub-pixel coordinates."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def overlaps(self, other: Hitbox) -> bool:
        return (
            self.x_min < other.x_max
            and self.x_max > other.x_min
            and self.y_min < other.y_max
            and self.y_max > other.y_min
        )


def get_attack_hitbox(attacker: FighterState, config: GameConfig) -> Hitbox | None:
    """Get the active hitbox for an attacker, or None if no hitbox is active.

    Hitbox is only present during ATTACK_ACTIVE state.
    """
    if attacker.fsm_state != FSMState.ATTACK_ACTIVE:
        return None

    commitment = attacker.active_commitment
    if commitment == CombatCommitment.LIGHT_ATTACK:
        reach_sub = config.actions.light_attack.reach * config.simulation.sub_pixel_scale
    elif commitment == CombatCommitment.HEAVY_ATTACK:
        reach_sub = config.actions.heavy_attack.reach * config.simulation.sub_pixel_scale
    else:
        return None

    fighter_half_w = (config.fighter.width * config.simulation.sub_pixel_scale) // 2
    fighter_h = config.fighter.height * config.simulation.sub_pixel_scale

    if attacker.facing > 0:
        x_min = attacker.x + fighter_half_w
        x_max = attacker.x + fighter_half_w + reach_sub
    else:
        x_min = attacker.x - fighter_half_w - reach_sub
        x_max = attacker.x - fighter_half_w

    y_min = attacker.y - fighter_h
    y_max = attacker.y

    return Hitbox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def get_fighter_hurtbox(fighter: FighterState, config: GameConfig) -> Hitbox:
    """Get the hurtbox (body) of a fighter."""
    half_w = (config.fighter.width * config.simulation.sub_pixel_scale) // 2
    h = config.fighter.height * config.simulation.sub_pixel_scale

    return Hitbox(
        x_min=fighter.x - half_w,
        x_max=fighter.x + half_w,
        y_min=fighter.y - h,
        y_max=fighter.y,
    )
