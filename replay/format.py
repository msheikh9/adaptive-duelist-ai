"""Replay binary format definitions.

The replay file is organized into three layers:
  Layer A (Authoritative): header + initial state + commitment stream
  Layer B (Verification):  periodic state checksums + final state hash
  Layer C (Inspection):    tick snapshots + match metadata

All integer fields use little-endian byte order.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

from game.combat.actions import Actor, CombatCommitment, FSMState
from game.state import FighterState, ArenaState, SimulationState, MatchStatus

MAGIC = b"ADUEL\x00"
FORMAT_VERSION = 1
ENGINE_VERSION = "0.2.0"
HEADER_SIZE = 256
DEFAULT_CHECKSUM_INTERVAL = 300

# --- Commitment Record: 6 bytes each ---
# I=tick_id(4), B=actor(1), B=commitment(1)
COMMITMENT_FORMAT = "<IBB"
COMMITMENT_SIZE = struct.calcsize(COMMITMENT_FORMAT)

# --- Checksum Record: 20 bytes each ---
# I=tick_id(4), 16s=md5(16)
CHECKSUM_FORMAT = "<I16s"
CHECKSUM_SIZE = struct.calcsize(CHECKSUM_FORMAT)


@dataclass
class CommitmentRecord:
    tick_id: int
    actor: Actor
    commitment: CombatCommitment

    def pack(self) -> bytes:
        return struct.pack(COMMITMENT_FORMAT, self.tick_id,
                           self.actor.value, self.commitment.value)

    @classmethod
    def unpack(cls, data: bytes) -> CommitmentRecord:
        tick_id, actor_v, commit_v = struct.unpack(COMMITMENT_FORMAT, data)
        return cls(tick_id=tick_id, actor=Actor(actor_v),
                   commitment=CombatCommitment(commit_v))


@dataclass
class ChecksumRecord:
    tick_id: int
    state_md5: bytes  # 16 bytes

    def pack(self) -> bytes:
        return struct.pack(CHECKSUM_FORMAT, self.tick_id, self.state_md5)

    @classmethod
    def unpack(cls, data: bytes) -> ChecksumRecord:
        tick_id, md5 = struct.unpack(CHECKSUM_FORMAT, data)
        return cls(tick_id=tick_id, state_md5=md5)


@dataclass
class ReplayHeader:
    format_version: int = FORMAT_VERSION
    engine_version: str = ENGINE_VERSION
    config_hash: str = ""
    frame_data_hash: str = ""
    rng_seed: int = 0
    match_id: str = ""
    total_ticks: int = 0
    winner: int = 2  # 0=player, 1=ai, 2=draw/none
    checksum_interval: int = DEFAULT_CHECKSUM_INTERVAL

    def pack(self) -> bytes:
        """Pack header into exactly HEADER_SIZE bytes."""
        buf = bytearray(HEADER_SIZE)
        offset = 0

        # format_version: uint16
        struct.pack_into("<H", buf, offset, self.format_version)
        offset += 2

        # engine_version: 16 bytes (utf-8 padded)
        ev = self.engine_version.encode("utf-8")[:16].ljust(16, b"\x00")
        buf[offset:offset+16] = ev
        offset += 16

        # config_hash: 32 bytes (hex string → raw bytes, or first 32 chars)
        ch = self.config_hash.encode("ascii")[:32].ljust(32, b"\x00")
        buf[offset:offset+32] = ch
        offset += 32

        # frame_data_hash: 32 bytes
        fh = self.frame_data_hash.encode("ascii")[:32].ljust(32, b"\x00")
        buf[offset:offset+32] = fh
        offset += 32

        # rng_seed: uint64
        struct.pack_into("<Q", buf, offset, self.rng_seed)
        offset += 8

        # match_id: 36 bytes (UUID string)
        mid = self.match_id.encode("ascii")[:36].ljust(36, b"\x00")
        buf[offset:offset+36] = mid
        offset += 36

        # total_ticks: uint32
        struct.pack_into("<I", buf, offset, self.total_ticks)
        offset += 4

        # winner: uint8
        struct.pack_into("<B", buf, offset, self.winner)
        offset += 1

        # checksum_interval: uint16
        struct.pack_into("<H", buf, offset, self.checksum_interval)
        offset += 2

        # rest is reserved/padding (already zeroed)
        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes) -> ReplayHeader:
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {HEADER_SIZE}")

        offset = 0
        fmt_ver = struct.unpack_from("<H", data, offset)[0]
        offset += 2

        eng_ver = data[offset:offset+16].rstrip(b"\x00").decode("utf-8")
        offset += 16

        cfg_hash = data[offset:offset+32].rstrip(b"\x00").decode("ascii")
        offset += 32

        fd_hash = data[offset:offset+32].rstrip(b"\x00").decode("ascii")
        offset += 32

        rng_seed = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        match_id = data[offset:offset+36].rstrip(b"\x00").decode("ascii")
        offset += 36

        total_ticks = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        winner = struct.unpack_from("<B", data, offset)[0]
        offset += 1

        checksum_interval = struct.unpack_from("<H", data, offset)[0]
        offset += 2

        return cls(
            format_version=fmt_ver, engine_version=eng_ver,
            config_hash=cfg_hash, frame_data_hash=fd_hash,
            rng_seed=rng_seed, match_id=match_id,
            total_ticks=total_ticks, winner=winner,
            checksum_interval=checksum_interval,
        )


def serialize_initial_state(state: SimulationState) -> bytes:
    """Serialize the initial SimulationState to JSON bytes."""
    d = {
        "tick_id": state.tick_id,
        "rng_seed": state.rng_seed,
        "match_status": state.match_status.value,
        "player": _serialize_fighter(state.player),
        "ai": _serialize_fighter(state.ai),
        "arena": {
            "width_sub": state.arena.width_sub,
            "height_sub": state.arena.height_sub,
            "ground_y_sub": state.arena.ground_y_sub,
        },
    }
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def deserialize_initial_state(data: bytes) -> SimulationState:
    """Deserialize initial SimulationState from JSON bytes."""
    d = json.loads(data.decode("utf-8"))
    return SimulationState(
        tick_id=d["tick_id"],
        rng_seed=d["rng_seed"],
        match_status=MatchStatus(d["match_status"]),
        player=_deserialize_fighter(d["player"]),
        ai=_deserialize_fighter(d["ai"]),
        arena=ArenaState(
            width_sub=d["arena"]["width_sub"],
            height_sub=d["arena"]["height_sub"],
            ground_y_sub=d["arena"]["ground_y_sub"],
        ),
    )


def _serialize_fighter(f: FighterState) -> dict:
    return {
        "x": f.x, "y": f.y, "velocity_x": f.velocity_x,
        "hp": f.hp, "stamina": f.stamina,
        "stamina_accumulator": f.stamina_accumulator,
        "fsm_state": f.fsm_state.value,
        "fsm_frames_remaining": f.fsm_frames_remaining,
        "active_commitment": f.active_commitment.value if f.active_commitment else None,
        "facing": f.facing,
    }


def _deserialize_fighter(d: dict) -> FighterState:
    return FighterState(
        x=d["x"], y=d["y"], velocity_x=d["velocity_x"],
        hp=d["hp"], stamina=d["stamina"],
        stamina_accumulator=d["stamina_accumulator"],
        fsm_state=FSMState(d["fsm_state"]),
        fsm_frames_remaining=d["fsm_frames_remaining"],
        active_commitment=(CombatCommitment(d["active_commitment"])
                           if d["active_commitment"] is not None else None),
        facing=d["facing"],
    )


def compute_state_hash(state: SimulationState) -> bytes:
    """Compute MD5 hash of essential state fields for checksum comparison."""
    h = hashlib.md5()
    h.update(struct.pack("<iiiiHHBBBBii",
                         state.player.x, state.player.y,
                         state.ai.x, state.ai.y,
                         state.player.hp, state.ai.hp,
                         state.player.stamina, state.ai.stamina,
                         state.player.fsm_state.value, state.ai.fsm_state.value,
                         state.player.fsm_frames_remaining,
                         state.ai.fsm_frames_remaining))
    return h.digest()


def compute_frame_data_hash(game_cfg) -> str:
    """Hash the action frame data table for replay compatibility checking."""
    h = hashlib.sha256()
    for name in ("light_attack", "heavy_attack"):
        atk = getattr(game_cfg.actions, name)
        h.update(f"{name}:{atk.startup_frames},{atk.active_frames},"
                 f"{atk.recovery_frames},{atk.stamina_cost},{atk.damage},"
                 f"{atk.reach},{atk.hitstun_frames},{atk.knockback}".encode())
    dodge = game_cfg.actions.dodge_backward
    h.update(f"dodge:{dodge.startup_frames},{dodge.active_frames},"
             f"{dodge.recovery_frames},{dodge.stamina_cost},{dodge.distance}".encode())
    return h.hexdigest()[:32]
