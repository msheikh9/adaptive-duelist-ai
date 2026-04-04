"""Vocabulary B: Combat commitments and fighter FSM states.

CombatCommitment represents an irreversible fighter action. Once entered,
the FSM carries it to completion. This is the observable game-level action
that all systems outside of input/ operate on.

FSMState represents the fighter's current animation/logic state within
the state machine.

SpacingZone classifies the distance between fighters.
"""

from enum import Enum, auto


class CombatCommitment(Enum):
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    LIGHT_ATTACK = auto()
    HEAVY_ATTACK = auto()
    DODGE_BACKWARD = auto()
    # Phase 2: directional dodge
    DODGE_LEFT = 10
    DODGE_RIGHT = 11
    # Phase 3: block
    BLOCK_START = 20
    BLOCK_RELEASE = 21
    # Phase 15: jump
    JUMP = 30
    # Phase 20: ranged weapon
    SHOOT_START   = 40   # player presses E — begin startup + charge
    SHOOT_INSTANT = 41   # AI shortcut — enter SHOOT_ACTIVE immediately (no charge)


# The subset of commitments active in Phase 1.
PHASE_1_COMMITMENTS = frozenset({
    CombatCommitment.MOVE_LEFT,
    CombatCommitment.MOVE_RIGHT,
    CombatCommitment.LIGHT_ATTACK,
    CombatCommitment.HEAVY_ATTACK,
    CombatCommitment.DODGE_BACKWARD,
})


class FSMState(Enum):
    """Fighter finite state machine states."""
    IDLE = auto()
    MOVING = auto()
    ATTACK_STARTUP = auto()
    ATTACK_ACTIVE = auto()
    ATTACK_RECOVERY = auto()
    DODGE_STARTUP = auto()
    DODGING = auto()
    DODGE_RECOVERY = auto()
    BLOCKING = auto()
    BLOCKSTUN = auto()
    HITSTUN = auto()
    PARRY_STUNNED = auto()
    EXHAUSTED = auto()
    KO = auto()
    # Phase 15: jump states (ordinals 15–17, fit in uint8 for replay packing)
    JUMP_STARTUP = auto()
    AIRBORNE = auto()
    LANDING = auto()
    # Phase 20: ranged weapon states
    SHOOT_STARTUP  = auto()   # brief startup before charging begins
    CHARGING       = auto()   # hold to accumulate charge (hold E)
    SHOOT_ACTIVE   = auto()   # projectile fired — brief active window
    SHOOT_RECOVERY = auto()   # recovery after firing


# States where the fighter can accept a new commitment.
FREE_STATES = frozenset({FSMState.IDLE, FSMState.MOVING})

# States where the fighter is locked into an animation.
LOCKED_STATES = frozenset(s for s in FSMState if s not in FREE_STATES)


class SpacingZone(Enum):
    CLOSE = auto()
    MID = auto()
    FAR = auto()


class Actor(Enum):
    """Identifies which fighter performed an action or received an event."""
    PLAYER = auto()
    AI = auto()
