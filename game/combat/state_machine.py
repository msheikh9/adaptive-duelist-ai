"""Fighter finite state machine.

Manages FSM transitions and frame countdown. The FSM is the authority
on what a fighter can and cannot do. When a commitment is entered, the
FSM locks the fighter through startup → active → recovery phases.

All timing is in integer tick counts. No floating point.
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment, FSMState, FREE_STATES
from game.state import FighterState
from config.config_loader import GameConfig


def can_commit(fighter: FighterState, commitment: CombatCommitment,
               config: GameConfig) -> bool:
    """Check if a fighter can enter a new commitment right now."""
    if fighter.fsm_state not in FREE_STATES:
        return False

    if fighter.fsm_state == FSMState.KO:
        return False

    # Check stamina requirements
    stamina_cost = get_stamina_cost(commitment, config)
    if stamina_cost is not None and fighter.stamina < stamina_cost:
        return False

    return True


def get_stamina_cost(commitment: CombatCommitment, config: GameConfig) -> int | None:
    """Get the upfront stamina cost for a commitment. None for continuous-cost actions."""
    match commitment:
        case CombatCommitment.LIGHT_ATTACK:
            return config.actions.light_attack.stamina_cost
        case CombatCommitment.HEAVY_ATTACK:
            return config.actions.heavy_attack.stamina_cost
        case CombatCommitment.DODGE_BACKWARD:
            return config.actions.dodge_backward.stamina_cost
        case CombatCommitment.MOVE_LEFT | CombatCommitment.MOVE_RIGHT:
            return None  # continuous cost, checked per-tick in stamina system
        case _:
            return None


def enter_commitment(fighter: FighterState, commitment: CombatCommitment,
                     config: GameConfig) -> None:
    """Transition a fighter into a new commitment. Caller must verify can_commit first."""
    fighter.active_commitment = commitment

    match commitment:
        case CombatCommitment.LIGHT_ATTACK:
            fighter.fsm_state = FSMState.ATTACK_STARTUP
            fighter.fsm_frames_remaining = config.actions.light_attack.startup_frames
            fighter.stamina -= config.actions.light_attack.stamina_cost

        case CombatCommitment.HEAVY_ATTACK:
            fighter.fsm_state = FSMState.ATTACK_STARTUP
            fighter.fsm_frames_remaining = config.actions.heavy_attack.startup_frames
            fighter.stamina -= config.actions.heavy_attack.stamina_cost

        case CombatCommitment.DODGE_BACKWARD:
            fighter.fsm_state = FSMState.DODGE_STARTUP
            fighter.fsm_frames_remaining = config.actions.dodge_backward.startup_frames
            fighter.stamina -= config.actions.dodge_backward.stamina_cost

        case CombatCommitment.MOVE_LEFT:
            fighter.fsm_state = FSMState.MOVING
            fighter.fsm_frames_remaining = 0
            fighter.velocity_x = -config.fighter.move_speed * config.simulation.sub_pixel_scale

        case CombatCommitment.MOVE_RIGHT:
            fighter.fsm_state = FSMState.MOVING
            fighter.fsm_frames_remaining = 0
            fighter.velocity_x = config.fighter.move_speed * config.simulation.sub_pixel_scale


def stop_moving(fighter: FighterState) -> None:
    """Return a moving fighter to idle."""
    if fighter.fsm_state == FSMState.MOVING:
        fighter.fsm_state = FSMState.IDLE
        fighter.fsm_frames_remaining = 0
        fighter.velocity_x = 0
        fighter.active_commitment = None


def enter_hitstun(fighter: FighterState, hitstun_frames: int) -> None:
    """Force a fighter into hitstun (from being hit)."""
    fighter.fsm_state = FSMState.HITSTUN
    fighter.fsm_frames_remaining = hitstun_frames
    fighter.active_commitment = None
    fighter.velocity_x = 0


def enter_exhausted(fighter: FighterState, recovery_frames: int) -> None:
    """Force a fighter into exhaustion (stamina depleted to 0)."""
    fighter.fsm_state = FSMState.EXHAUSTED
    fighter.fsm_frames_remaining = recovery_frames
    fighter.active_commitment = None
    fighter.velocity_x = 0


def enter_ko(fighter: FighterState) -> None:
    """Fighter is knocked out (HP reached 0)."""
    fighter.fsm_state = FSMState.KO
    fighter.fsm_frames_remaining = 0
    fighter.active_commitment = None
    fighter.velocity_x = 0
    fighter.hp = 0


def _get_attack_config(commitment: CombatCommitment, config: GameConfig):
    """Get the AttackActionConfig for an attack commitment."""
    if commitment == CombatCommitment.LIGHT_ATTACK:
        return config.actions.light_attack
    elif commitment == CombatCommitment.HEAVY_ATTACK:
        return config.actions.heavy_attack
    return None


def tick_fsm(fighter: FighterState, config: GameConfig) -> None:
    """Advance the fighter's FSM by one tick.

    Handles frame countdown and automatic state transitions.
    Does NOT handle external events (hits, exhaustion) — those are
    applied by their respective systems before this is called.
    """
    state = fighter.fsm_state

    if state in (FSMState.IDLE, FSMState.MOVING, FSMState.KO):
        return

    if fighter.fsm_frames_remaining > 0:
        fighter.fsm_frames_remaining -= 1
        if fighter.fsm_frames_remaining > 0:
            return

    # Frame count reached 0 — transition to next state
    match state:
        case FSMState.ATTACK_STARTUP:
            atk_cfg = _get_attack_config(fighter.active_commitment, config)
            fighter.fsm_state = FSMState.ATTACK_ACTIVE
            fighter.fsm_frames_remaining = atk_cfg.active_frames

        case FSMState.ATTACK_ACTIVE:
            atk_cfg = _get_attack_config(fighter.active_commitment, config)
            fighter.fsm_state = FSMState.ATTACK_RECOVERY
            fighter.fsm_frames_remaining = atk_cfg.recovery_frames

        case FSMState.ATTACK_RECOVERY:
            fighter.fsm_state = FSMState.IDLE
            fighter.fsm_frames_remaining = 0
            fighter.active_commitment = None

        case FSMState.DODGE_STARTUP:
            fighter.fsm_state = FSMState.DODGING
            fighter.fsm_frames_remaining = config.actions.dodge_backward.active_frames

        case FSMState.DODGING:
            fighter.fsm_state = FSMState.DODGE_RECOVERY
            fighter.fsm_frames_remaining = config.actions.dodge_backward.recovery_frames
            fighter.velocity_x = 0  # stop dodge movement

        case FSMState.DODGE_RECOVERY:
            fighter.fsm_state = FSMState.IDLE
            fighter.fsm_frames_remaining = 0
            fighter.active_commitment = None

        case FSMState.HITSTUN:
            fighter.fsm_state = FSMState.IDLE
            fighter.fsm_frames_remaining = 0
            fighter.velocity_x = 0

        case FSMState.EXHAUSTED:
            fighter.fsm_state = FSMState.IDLE
            fighter.fsm_frames_remaining = 0
