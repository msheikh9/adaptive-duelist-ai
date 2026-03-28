"""Dense per-tick snapshot records.

TickSnapshots are recorded every simulation tick. They contain only the
minimal state needed for replay reconstruction and timeline scrubbing.
ML training and profile updates NEVER read TickSnapshots — they use
SemanticEvents exclusively.

All spatial values are in sub-pixel units (int32).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from game.combat.actions import FSMState
from game.state import SimulationState

# Struct format for binary packing.
# I = uint32, i = int32, H = uint16, B = uint8
# Total: 4 + 4+4+4+4 + 2+2 + 1+1 + 1+1+1+1 + 4 = 34 bytes
TICK_SNAPSHOT_FORMAT = "<I iiii HH BB BB BB I"
TICK_SNAPSHOT_SIZE = struct.calcsize(TICK_SNAPSHOT_FORMAT)


@dataclass(frozen=True, slots=True)
class TickSnapshot:
    """Dense per-tick state record. 34 bytes packed."""

    tick_id: int            # uint32
    player_x: int           # int32 (sub-pixel)
    player_y: int           # int32 (sub-pixel)
    ai_x: int               # int32 (sub-pixel)
    ai_y: int               # int32 (sub-pixel)
    player_hp: int           # uint16
    ai_hp: int               # uint16
    player_stamina: int      # uint8
    ai_stamina: int          # uint8
    player_fsm: int          # uint8 (FSMState ordinal)
    ai_fsm: int              # uint8 (FSMState ordinal)
    player_fsm_frame: int    # uint8
    ai_fsm_frame: int        # uint8
    distance: int            # uint32 (sub-pixel)

    def pack(self) -> bytes:
        return struct.pack(
            TICK_SNAPSHOT_FORMAT,
            self.tick_id,
            self.player_x, self.player_y,
            self.ai_x, self.ai_y,
            self.player_hp, self.ai_hp,
            self.player_stamina, self.ai_stamina,
            self.player_fsm, self.ai_fsm,
            self.player_fsm_frame, self.ai_fsm_frame,
            self.distance,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TickSnapshot:
        values = struct.unpack(TICK_SNAPSHOT_FORMAT, data)
        return cls(*values)

    @classmethod
    def from_state(cls, state: SimulationState) -> TickSnapshot:
        return cls(
            tick_id=state.tick_id,
            player_x=state.player.x,
            player_y=state.player.y,
            ai_x=state.ai.x,
            ai_y=state.ai.y,
            player_hp=state.player.hp,
            ai_hp=state.ai.hp,
            player_stamina=int(state.player.stamina),
            ai_stamina=int(state.ai.stamina),
            player_fsm=state.player.fsm_state.value,
            ai_fsm=state.ai.fsm_state.value,
            player_fsm_frame=min(state.player.fsm_frames_remaining, 255),
            ai_fsm_frame=min(state.ai.fsm_frames_remaining, 255),
            distance=state.distance_sub(),
        )

    def fsm_state_player(self) -> FSMState:
        return FSMState(self.player_fsm)

    def fsm_state_ai(self) -> FSMState:
        return FSMState(self.ai_fsm)
