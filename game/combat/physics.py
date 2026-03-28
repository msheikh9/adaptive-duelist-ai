"""Physics system: position updates, arena clamping, knockback.

All positions are in sub-pixel units (int32). No floating point in
position math. Velocity is in sub-pixel units per tick.
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment, FSMState
from game.state import FighterState, ArenaState
from config.config_loader import GameConfig


def apply_velocity(fighter: FighterState) -> None:
    """Apply velocity to position. Single tick update."""
    fighter.x += fighter.velocity_x


def apply_dodge_velocity(fighter: FighterState, config: GameConfig) -> None:
    """Apply dodge movement during DODGING state.

    Dodge moves the fighter backward (away from opponent) over the
    active frames. Total distance = dodge_distance * sub_pixel_scale,
    spread evenly across active frames.
    """
    if fighter.fsm_state != FSMState.DODGING:
        return

    dodge_cfg = config.actions.dodge_backward
    total_distance_sub = dodge_cfg.distance * config.simulation.sub_pixel_scale
    velocity_per_frame = total_distance_sub // dodge_cfg.active_frames

    # Dodge moves backward (opposite of facing direction)
    fighter.velocity_x = -fighter.facing * velocity_per_frame


def clamp_to_arena(fighter: FighterState, arena: ArenaState,
                   fighter_width_sub: int) -> None:
    """Clamp fighter position to arena boundaries.

    Fighters cannot move past the left (0) or right (arena_width - fighter_width) edges.
    """
    half_width = fighter_width_sub // 2
    min_x = half_width
    max_x = arena.width_sub - half_width

    if fighter.x < min_x:
        fighter.x = min_x
        fighter.velocity_x = 0
    elif fighter.x > max_x:
        fighter.x = max_x
        fighter.velocity_x = 0


def apply_knockback(fighter: FighterState, knockback_sub: int, direction: int) -> None:
    """Apply instantaneous knockback displacement.

    Args:
        fighter: The fighter being knocked back.
        knockback_sub: Knockback distance in sub-pixel units.
        direction: +1 for rightward, -1 for leftward.
    """
    fighter.x += knockback_sub * direction


def update_facing(player: FighterState, ai: FighterState) -> None:
    """Update facing directions so fighters always face each other."""
    if player.x < ai.x:
        player.facing = 1
        ai.facing = -1
    elif player.x > ai.x:
        player.facing = -1
        ai.facing = 1
    # If equal position, keep current facing
