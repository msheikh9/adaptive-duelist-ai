"""Sparse semantic event records.

SemanticEvents are recorded only on meaningful state transitions.
They are the primary data source for ML training, profile updates,
and post-match analytics. Never recorded per-tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone


class EventType(Enum):
    COMMITMENT_START = auto()
    COMMITMENT_END = auto()
    HIT_LANDED = auto()
    HIT_BLOCKED = auto()
    PARRY_SUCCESS = auto()
    DODGE_SUCCESS = auto()
    DODGE_NO_THREAT = auto()
    STAMINA_EXHAUSTED = auto()
    MATCH_START = auto()
    MATCH_END = auto()


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    """A single meaningful game event.

    Fields use concrete types matching the SQLite schema.
    Optional fields are None when not applicable to the event type.
    """

    event_type: EventType
    match_id: str
    tick_id: int
    actor: Actor
    commitment: CombatCommitment | None = None
    opponent_fsm_state: FSMState | None = None
    opponent_commitment: CombatCommitment | None = None
    spacing_zone: SpacingZone | None = None
    actor_hp: int = 0
    opponent_hp: int = 0
    actor_stamina: int = 0
    opponent_stamina: int = 0
    damage_dealt: int | None = None
    reaction_ticks: int | None = None
