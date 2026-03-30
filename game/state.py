"""Mutable authoritative simulation state with phase-gated write access.

SimulationState is the single source of truth for the game. It is mutated
ONLY during the SIMULATE phase of the tick loop. External systems (AI,
renderer, logger) read it during their designated phases when no mutation
occurs.

In debug mode (__debug__ is True, i.e. not running with -O), a phase lock
enforces that mutations only happen during the SIMULATE phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from game.combat.actions import CombatCommitment, FSMState


class TickPhase(Enum):
    """Phases of the game loop tick. Used for phase-lock enforcement."""
    INPUT = auto()
    SIMULATE = auto()
    AI_OBSERVE = auto()
    LOG = auto()
    RENDER = auto()


class MatchStatus(Enum):
    WAITING = auto()
    ACTIVE = auto()
    ENDED = auto()


class PhaseLockError(Exception):
    """Raised when SimulationState is mutated outside the SIMULATE phase."""


class PhaseLock:
    """Debug-only enforcement that state mutations happen in the correct phase.

    When __debug__ is True (default), set_phase / check_write are active.
    When running with python -O, check_write is a no-op.
    """

    def __init__(self) -> None:
        self._phase: TickPhase = TickPhase.INPUT

    @property
    def phase(self) -> TickPhase:
        return self._phase

    def set_phase(self, phase: TickPhase) -> None:
        self._phase = phase

    def check_write(self) -> None:
        if __debug__ and self._phase != TickPhase.SIMULATE:
            raise PhaseLockError(
                f"SimulationState mutation attempted during {self._phase.name} phase. "
                f"Mutations are only allowed during SIMULATE phase."
            )


@dataclass
class FighterState:
    """Mutable state of a single fighter."""

    # Position in sub-pixel units (pixels * sub_pixel_scale)
    x: int = 0
    y: int = 0
    velocity_x: int = 0
    # Phase 15: vertical velocity (sub-pixel/tick; negative = upward)
    velocity_y: int = 0

    # Resources
    hp: int = 200
    stamina: int = 100
    stamina_accumulator: float = 0.0

    # FSM
    fsm_state: FSMState = FSMState.IDLE
    fsm_frames_remaining: int = 0

    # Active commitment (None when in FREE state)
    active_commitment: CombatCommitment | None = None

    # Facing direction: +1 = facing right, -1 = facing left
    facing: int = 1

    # Phase 17: dodge cooldown counter (ticks until next dodge is legal; 0 = ready)
    dodge_cooldown: int = 0

    @property
    def is_free(self) -> bool:
        return self.fsm_state in (FSMState.IDLE, FSMState.MOVING)

    @property
    def is_locked(self) -> bool:
        return not self.is_free

    @property
    def is_alive(self) -> bool:
        return self.fsm_state != FSMState.KO

    @property
    def is_airborne(self) -> bool:
        """True while fighter is off the ground (jump startup, in air, landing)."""
        return self.fsm_state in (FSMState.JUMP_STARTUP, FSMState.AIRBORNE, FSMState.LANDING)


@dataclass
class ArenaState:
    """Arena boundary state. Immutable after initialization."""

    width_sub: int = 0     # arena width in sub-pixel units
    height_sub: int = 0
    ground_y_sub: int = 0

    @classmethod
    def from_config(cls, arena_width: int, arena_height: int,
                    ground_y: int, sub_pixel_scale: int) -> ArenaState:
        return cls(
            width_sub=arena_width * sub_pixel_scale,
            height_sub=arena_height * sub_pixel_scale,
            ground_y_sub=ground_y * sub_pixel_scale,
        )


@dataclass
class SimulationState:
    """The single mutable authoritative game state.

    Owned by the game engine. Mutated only during SIMULATE phase.
    Read by AI, renderer, and logger during their respective phases.
    """

    tick_id: int = 0
    rng_seed: int = 0

    player: FighterState = field(default_factory=FighterState)
    ai: FighterState = field(default_factory=FighterState)
    arena: ArenaState = field(default_factory=ArenaState)

    match_status: MatchStatus = MatchStatus.WAITING
    winner: str | None = None  # "PLAYER", "AI", or None

    phase_lock: PhaseLock = field(default_factory=PhaseLock)

    def distance_sub(self) -> int:
        """Absolute distance between fighters in sub-pixel units."""
        return abs(self.player.x - self.ai.x)

    def distance_px(self, sub_pixel_scale: int) -> float:
        """Distance in pixels."""
        return self.distance_sub() / sub_pixel_scale

    def set_phase(self, phase: TickPhase) -> None:
        """Transition to a new tick phase. Called by the engine."""
        self.phase_lock.set_phase(phase)
