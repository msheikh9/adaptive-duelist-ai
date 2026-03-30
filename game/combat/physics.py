"""Physics system: position updates, arena clamping, knockback, gravity.

All positions are in sub-pixel units (int32). No floating point in
position math. Velocity is in sub-pixel units per tick.

Phase 15 additions:
  - apply_gravity()  — downward acceleration when above ground
  - handle_landing() — clamp to floor and report touch-down events
  - apply_velocity() now also applies velocity_y (vertical)
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment, FSMState
from game.state import FighterState, ArenaState
from config.config_loader import GameConfig


def apply_velocity(fighter: FighterState) -> None:
    """Apply velocity to position. Single tick update (horizontal + vertical)."""
    fighter.x += fighter.velocity_x
    fighter.y += fighter.velocity_y


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


def apply_gravity(fighter: FighterState, arena: ArenaState,
                  config: GameConfig) -> None:
    """Apply downward gravity when fighter is above ground level.

    Gravity is applied regardless of FSM state so that fighters in
    HITSTUN or KO also fall back to the floor naturally.
    """
    if fighter.y < arena.ground_y_sub:
        fighter.velocity_y += config.fighter.jump_gravity * config.simulation.sub_pixel_scale


def handle_landing(fighter: FighterState, arena: ArenaState) -> bool:
    """Clamp fighter to ground if they've descended to or past floor level.

    Returns True if the fighter was in downward flight and just touched
    down (i.e. a landing event occurred that the caller should act on).
    The caller is responsible for FSM transitions (enter_landing / etc.).
    """
    if fighter.y >= arena.ground_y_sub:
        just_landed = fighter.velocity_y > 0  # was moving downward
        if just_landed:
            fighter.y = arena.ground_y_sub
            fighter.velocity_y = 0
        return just_landed
    return False


def clamp_to_arena(fighter: FighterState, arena: ArenaState,
                   fighter_width_sub: int) -> None:
    """Clamp fighter position to arena boundaries.

    Horizontal: fighters cannot move past the left/right edges.
    Vertical ceiling: fighters cannot go above the arena top (y < 0).
    Floor clamping is handled separately by handle_landing().
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

    # Ceiling clamp: can't go above top of arena
    if fighter.y < 0:
        fighter.y = 0
        if fighter.velocity_y < 0:
            fighter.velocity_y = 0


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
