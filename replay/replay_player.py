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

from config.config_loader import GameConfig
from data.tick_snapshot import TickSnapshot
from game.combat.actions import Actor, CombatCommitment, FSMState, FREE_STATES
from game.combat.collision import HitTracker, check_hit
from game.combat.damage import apply_hit
from game.combat.physics import (
    apply_dodge_velocity,
    apply_velocity,
    clamp_to_arena,
    update_facing,
)
from game.combat.stamina import tick_stamina
from game.combat.state_machine import tick_fsm
from game.entities.fighter import attempt_commitment
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


def replay_match(replay: ReplayData, game_cfg: GameConfig) -> SimulationState:
    """Reconstruct a match by replaying commitments through simulation.

    Uses only Layer A data (initial state + commitment stream).
    Returns the final SimulationState.
    """
    state = _clone_initial_state(replay.initial_state)
    state.set_phase(TickPhase.SIMULATE)

    hit_tracker = HitTracker()
    scale = game_cfg.simulation.sub_pixel_scale
    fighter_w_sub = game_cfg.fighter.width * scale

    # Index commitments by tick for fast lookup
    commitments_by_tick: dict[int, list[CommitmentRecord]] = {}
    for c in replay.commitments:
        commitments_by_tick.setdefault(c.tick_id, []).append(c)

    total_ticks = replay.header.total_ticks

    for tick in range(total_ticks):
        if state.match_status == MatchStatus.ENDED:
            break

        # Apply commitments for this tick
        tick_commits = commitments_by_tick.get(tick, [])
        for c in tick_commits:
            fighter = state.player if c.actor == Actor.PLAYER else state.ai
            attempt_commitment(fighter, c.commitment, game_cfg)

        # Physics
        apply_dodge_velocity(state.player, game_cfg)
        apply_dodge_velocity(state.ai, game_cfg)
        apply_velocity(state.player)
        apply_velocity(state.ai)
        clamp_to_arena(state.player, state.arena, fighter_w_sub)
        clamp_to_arena(state.ai, state.arena, fighter_w_sub)
        update_facing(state.player, state.ai)

        # Collision
        player_hit = check_hit(state.player, state.ai, "player", hit_tracker, game_cfg)
        ai_hit = check_hit(state.ai, state.player, "ai", hit_tracker, game_cfg)

        if player_hit:
            apply_hit(state.ai, player_hit)
        if ai_hit:
            apply_hit(state.player, ai_hit)

        # Stamina
        tick_stamina(state.player, game_cfg)
        tick_stamina(state.ai, game_cfg)

        # Reset hit tracker for free fighters
        if state.player.is_free:
            hit_tracker.reset("player")
        if state.ai.is_free:
            hit_tracker.reset("ai")

        # FSM advance
        tick_fsm(state.player, game_cfg)
        tick_fsm(state.ai, game_cfg)

        # KO check
        if state.player.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "AI"
        elif state.ai.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "PLAYER"

        state.tick_id += 1

    return state


def verify_replay(replay: ReplayData, game_cfg: GameConfig) -> VerificationResult:
    """Replay the match and verify against Layer B checksums.

    Returns a VerificationResult with pass/fail status and details.
    """
    result = VerificationResult()

    try:
        state = _clone_initial_state(replay.initial_state)
        state.set_phase(TickPhase.SIMULATE)

        hit_tracker = HitTracker()
        scale = game_cfg.simulation.sub_pixel_scale
        fighter_w_sub = game_cfg.fighter.width * scale

        # Index commitments and checksums by tick
        commitments_by_tick: dict[int, list[CommitmentRecord]] = {}
        for c in replay.commitments:
            commitments_by_tick.setdefault(c.tick_id, []).append(c)

        checksums_by_tick: dict[int, ChecksumRecord] = {}
        for cs in replay.checksums:
            checksums_by_tick[cs.tick_id] = cs

        result.total_checksums = len(replay.checksums)
        total_ticks = replay.header.total_ticks

        for tick in range(total_ticks):
            if state.match_status == MatchStatus.ENDED:
                break

            # Apply commitments
            tick_commits = commitments_by_tick.get(tick, [])
            for c in tick_commits:
                fighter = state.player if c.actor == Actor.PLAYER else state.ai
                attempt_commitment(fighter, c.commitment, game_cfg)

            # Physics
            apply_dodge_velocity(state.player, game_cfg)
            apply_dodge_velocity(state.ai, game_cfg)
            apply_velocity(state.player)
            apply_velocity(state.ai)
            clamp_to_arena(state.player, state.arena, fighter_w_sub)
            clamp_to_arena(state.ai, state.arena, fighter_w_sub)
            update_facing(state.player, state.ai)

            # Collision
            player_hit = check_hit(state.player, state.ai, "player",
                                   hit_tracker, game_cfg)
            ai_hit = check_hit(state.ai, state.player, "ai",
                               hit_tracker, game_cfg)

            if player_hit:
                apply_hit(state.ai, player_hit)
            if ai_hit:
                apply_hit(state.player, ai_hit)

            # Stamina
            tick_stamina(state.player, game_cfg)
            tick_stamina(state.ai, game_cfg)

            # Reset hit tracker
            if state.player.is_free:
                hit_tracker.reset("player")
            if state.ai.is_free:
                hit_tracker.reset("ai")

            # FSM advance
            tick_fsm(state.player, game_cfg)
            tick_fsm(state.ai, game_cfg)

            # Verify checksum after simulation (matches recorder timing: LOG phase)
            if tick in checksums_by_tick:
                expected = checksums_by_tick[tick]
                actual_hash = compute_state_hash(state)
                if actual_hash != expected.state_md5:
                    result.passed = False
                    result.failed_checksums += 1
                    result.checksum_failures.append(
                        (tick, expected.state_md5, actual_hash)
                    )

            # KO check
            if state.player.fsm_state == FSMState.KO:
                state.match_status = MatchStatus.ENDED
                state.winner = "AI"
            elif state.ai.fsm_state == FSMState.KO:
                state.match_status = MatchStatus.ENDED
                state.winner = "PLAYER"

            state.tick_id += 1

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
