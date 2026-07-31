"""Replay player: deterministic match reconstruction from Layer A data.

Reconstructs the full match by replaying the initial state and commitment
stream through the same simulation logic. Only needs Layer A (header +
initial state + commitments) — Layer B checksums are used for verification,
Layer C snapshots are optional inspection data.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config.config_loader import GameConfig
from data.tick_snapshot import TickSnapshot
from game.combat.actions import Actor, CombatCommitment, FSMState, FREE_STATES
from game.entities.fighter import attempt_commitment
from game.simulation_step import SimulationContext, step_simulation
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    SimulationState,
    TickPhase,
)
from replay.format import (
    MAGIC,
    HEADER_SIZE,
    COMMITMENT_SIZE,
    CHECKSUM_SIZE,
    CommitmentRecord,
    ChecksumRecord,
    ReplayHeader,
    compute_state_hash,
    deserialize_initial_state,
)

log = logging.getLogger(__name__)


class ReplayError(Exception):
    """Raised when replay loading or verification fails."""


@dataclass
class VerificationResult:
    """Result of replaying and verifying a match."""
    passed: bool = True
    total_checksums: int = 0
    failed_checksums: int = 0
    checksum_failures: list[tuple[int, bytes, bytes]] = field(default_factory=list)
    final_state_match: bool = True
    error: str | None = None


@dataclass
class ReplayData:
    """Parsed replay file contents."""
    header: ReplayHeader
    initial_state: SimulationState
    commitments: list[CommitmentRecord]
    checksums: list[ChecksumRecord]
    snapshots_raw: bytes  # Raw snapshot bytes (Layer C), may be empty
    metadata_raw: bytes   # Raw metadata JSON (Layer C), may be empty


def load_replay(path: Path) -> ReplayData:
    """Load and parse a replay file from disk."""
    data = path.read_bytes()

    # Validate magic
    if not data.startswith(MAGIC):
        raise ReplayError(f"Invalid replay file: bad magic bytes")

    offset = len(MAGIC)

    # Parse header
    header = ReplayHeader.unpack(data[offset:offset + HEADER_SIZE])
    offset += HEADER_SIZE

    # Parse section table: 4 sections × (offset: uint32, count: uint32)
    section_table = struct.unpack_from("<IIIIIIII", data, offset)
    offset += 32

    initial_state_offset, initial_state_len = section_table[0], section_table[1]
    commitments_offset, commitment_count = section_table[2], section_table[3]
    checksums_offset, checksum_count = section_table[4], section_table[5]
    snapshots_offset, snapshot_count = section_table[6], section_table[7]

    # Parse Layer A: initial state (skip the uint32 length prefix)
    state_data_start = initial_state_offset + 4
    state_data = data[state_data_start:state_data_start + initial_state_len]
    initial_state = deserialize_initial_state(state_data)

    # Parse Layer A: commitments
    commitments = []
    pos = commitments_offset
    for _ in range(commitment_count):
        rec = CommitmentRecord.unpack(data[pos:pos + COMMITMENT_SIZE])
        commitments.append(rec)
        pos += COMMITMENT_SIZE

    # Parse Layer B: checksums
    checksums = []
    pos = checksums_offset
    for _ in range(checksum_count):
        rec = ChecksumRecord.unpack(data[pos:pos + CHECKSUM_SIZE])
        checksums.append(rec)
        pos += CHECKSUM_SIZE

    # Parse Layer C: snapshots (raw bytes) + metadata
    from data.tick_snapshot import TICK_SNAPSHOT_SIZE
    snapshot_bytes_len = snapshot_count * TICK_SNAPSHOT_SIZE
    snapshots_raw = data[snapshots_offset:snapshots_offset + snapshot_bytes_len]

    metadata_raw = b""
    metadata_offset = snapshots_offset + snapshot_bytes_len
    if metadata_offset + 4 <= len(data):
        meta_len = struct.unpack_from("<I", data, metadata_offset)[0]
        metadata_raw = data[metadata_offset + 4:metadata_offset + 4 + meta_len]

    return ReplayData(
        header=header,
        initial_state=initial_state,
        commitments=commitments,
        checksums=checksums,
        snapshots_raw=snapshots_raw,
        metadata_raw=metadata_raw,
    )


def _drive_replay(
    replay: ReplayData,
    game_cfg: GameConfig,
    on_tick_end: "Callable[[int, SimulationState], None] | None" = None,
) -> SimulationState:
    """Re-simulate a match from its recorded commitment stream.

    Runs the same `step_simulation` the live engine runs; the only difference is
    where commitments come from. `on_tick_end` is called after each tick with
    (tick, state), before the tick counter advances — that is exactly when the
    recorder writes its checksums.
    """
    state = _clone_initial_state(replay.initial_state)
    state.set_phase(TickPhase.SIMULATE)
    ctx = SimulationContext()

    commitments_by_tick: dict[int, list[CommitmentRecord]] = {}
    for c in replay.commitments:
        commitments_by_tick.setdefault(c.tick_id, []).append(c)

    def replay_actor(records, actor: Actor, fighter) -> list[CombatCommitment]:
        """Re-enter the commitments recorded for `actor` on this tick, in order."""
        entered: list[CombatCommitment] = []
        for rec in records:
            if rec.actor != actor:
                continue
            if attempt_commitment(fighter, rec.commitment, game_cfg):
                entered.append(rec.commitment)
        return entered

    # header.total_ticks is the *index* of the final simulated tick, not a count
    # (the recorder writes final_state.tick_id, and the engine finalizes before
    # incrementing). So the inclusive bound is total_ticks, and iterating
    # range(total_ticks) stopped one tick short — which left the final-state hash
    # mismatching even when every periodic checksum agreed. The ENDED check below
    # keeps this self-limiting, so the extra iteration is a no-op after a KO.
    for tick in range(replay.header.total_ticks + 1):
        if state.match_status == MatchStatus.ENDED:
            break

        records = commitments_by_tick.get(tick, ())
        step_simulation(
            state, game_cfg, ctx,
            lambda: replay_actor(records, Actor.PLAYER, state.player),
            lambda: replay_actor(records, Actor.AI, state.ai),
        )

        if on_tick_end is not None:
            on_tick_end(tick, state)

        state.tick_id += 1

    return state


def replay_match(replay: ReplayData, game_cfg: GameConfig) -> SimulationState:
    """Reconstruct a match by replaying commitments through simulation.

    Uses only Layer A data (initial state + commitment stream).
    Returns the final SimulationState.
    """
    return _drive_replay(replay, game_cfg)


def verify_replay(replay: ReplayData, game_cfg: GameConfig) -> VerificationResult:
    """Replay the match and verify against Layer B checksums.

    Returns a VerificationResult with pass/fail status and details.
    """
    result = VerificationResult()

    try:
        checksums_by_tick: dict[int, ChecksumRecord] = {
            cs.tick_id: cs for cs in replay.checksums
        }
        result.total_checksums = len(replay.checksums)

        def check(tick: int, state: SimulationState) -> None:
            expected = checksums_by_tick.get(tick)
            if expected is None:
                return
            actual_hash = compute_state_hash(state)
            if actual_hash != expected.state_md5:
                result.passed = False
                result.failed_checksums += 1
                result.checksum_failures.append(
                    (tick, expected.state_md5, actual_hash)
                )

        state = _drive_replay(replay, game_cfg, on_tick_end=check)

        # Verify final state checksum (last entry in checksums list)
        if replay.checksums:
            final_cs = replay.checksums[-1]
            if final_cs.tick_id == state.tick_id:
                actual_final = compute_state_hash(state)
                if actual_final != final_cs.state_md5:
                    result.passed = False
                    result.final_state_match = False

    except Exception as e:
        result.passed = False
        result.error = str(e)

    return result


def _clone_initial_state(source: SimulationState) -> SimulationState:
    """Create a fresh SimulationState from the initial state data."""
    return SimulationState(
        tick_id=source.tick_id,
        rng_seed=source.rng_seed,
        match_status=source.match_status,
        player=FighterState(
            x=source.player.x,
            y=source.player.y,
            velocity_x=source.player.velocity_x,
            hp=source.player.hp,
            stamina=source.player.stamina,
            stamina_accumulator=source.player.stamina_accumulator,
            fsm_state=source.player.fsm_state,
            fsm_frames_remaining=source.player.fsm_frames_remaining,
            active_commitment=source.player.active_commitment,
            facing=source.player.facing,
        ),
        ai=FighterState(
            x=source.ai.x,
            y=source.ai.y,
            velocity_x=source.ai.velocity_x,
            hp=source.ai.hp,
            stamina=source.ai.stamina,
            stamina_accumulator=source.ai.stamina_accumulator,
            fsm_state=source.ai.fsm_state,
            fsm_frames_remaining=source.ai.fsm_frames_remaining,
            active_commitment=source.ai.active_commitment,
            facing=source.ai.facing,
        ),
        arena=ArenaState(
            width_sub=source.arena.width_sub,
            height_sub=source.arena.height_sub,
            ground_y_sub=source.arena.ground_y_sub,
        ),
    )
