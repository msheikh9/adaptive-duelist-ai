"""Replay recorder: captures match data into a binary replay file.

Records three layers during a match:
  Layer A (Authoritative): header + initial state + commitment stream
  Layer B (Verification):  periodic state checksums + final state hash
  Layer C (Inspection):    tick snapshots + match metadata

The recorder accumulates data in memory and writes a single file at
match end via finalize().
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

from config.config_loader import GameConfig, config_hash, CONFIG_DIR
from data.tick_snapshot import TickSnapshot, TICK_SNAPSHOT_SIZE
from game.combat.actions import Actor, CombatCommitment
from game.state import SimulationState
from replay.format import (
    MAGIC,
    HEADER_SIZE,
    DEFAULT_CHECKSUM_INTERVAL,
    CommitmentRecord,
    ChecksumRecord,
    ReplayHeader,
    compute_frame_data_hash,
    compute_state_hash,
    serialize_initial_state,
)

log = logging.getLogger(__name__)

REPLAY_DIR = Path(__file__).parent.parent / "replays"


class ReplayRecorder:
    """Records match data and writes a binary replay file on finalize.

    Usage:
        recorder = ReplayRecorder(initial_state, game_cfg)
        # During simulation:
        recorder.record_commitment(tick_id, actor, commitment)
        # During LOG phase:
        recorder.record_checksum_if_due(state)
        # At match end:
        replay_path = recorder.finalize(final_state, snapshots, match_id)
    """

    def __init__(self, initial_state: SimulationState,
                 game_cfg: GameConfig,
                 checksum_interval: int = DEFAULT_CHECKSUM_INTERVAL) -> None:
        self._game_cfg = game_cfg
        self._checksum_interval = checksum_interval

        # Layer A data
        self._initial_state_bytes = serialize_initial_state(initial_state)
        self._rng_seed = initial_state.rng_seed
        self._commitments: list[CommitmentRecord] = []

        # Layer B data
        self._checksums: list[ChecksumRecord] = []
        self._last_checksum_tick = -checksum_interval  # Force first checksum at tick 0

        # Precompute hashes
        self._config_hash = config_hash(CONFIG_DIR / "game_config.yaml")[:32]
        self._frame_data_hash = compute_frame_data_hash(game_cfg)

    def record_commitment(self, tick_id: int, actor: Actor,
                          commitment: CombatCommitment) -> None:
        """Record a commitment event. Called during SIMULATE phase."""
        self._commitments.append(
            CommitmentRecord(tick_id=tick_id, actor=actor, commitment=commitment)
        )

    def record_checksum_if_due(self, state: SimulationState) -> None:
        """Record a state checksum if enough ticks have passed since the last one."""
        if state.tick_id - self._last_checksum_tick >= self._checksum_interval:
            checksum = compute_state_hash(state)
            self._checksums.append(
                ChecksumRecord(tick_id=state.tick_id, state_md5=checksum)
            )
            self._last_checksum_tick = state.tick_id

    def finalize(self, final_state: SimulationState,
                 snapshots: list[TickSnapshot],
                 match_id: str) -> Path | None:
        """Write the replay file. Returns the file path, or None on failure."""
        try:
            return self._write_replay(final_state, snapshots, match_id)
        except Exception:
            log.exception("Failed to write replay file for match %s", match_id[:8])
            return None

    def _write_replay(self, final_state: SimulationState,
                      snapshots: list[TickSnapshot],
                      match_id: str) -> Path:
        """Build and write the binary replay file."""
        # Record final checksum
        final_checksum = compute_state_hash(final_state)
        self._checksums.append(
            ChecksumRecord(tick_id=final_state.tick_id, state_md5=final_checksum)
        )

        # Determine winner code: 0=player, 1=ai, 2=draw/none
        winner_code = 2
        if final_state.winner == "PLAYER":
            winner_code = 0
        elif final_state.winner == "AI":
            winner_code = 1

        # Build header
        header = ReplayHeader(
            config_hash=self._config_hash,
            frame_data_hash=self._frame_data_hash,
            rng_seed=self._rng_seed,
            match_id=match_id,
            total_ticks=final_state.tick_id,
            winner=winner_code,
            checksum_interval=self._checksum_interval,
        )

        # --- Pack binary sections ---

        # Layer A: initial state
        initial_state_data = self._initial_state_bytes
        initial_state_len = len(initial_state_data)

        # Layer A: commitments
        commitment_data = b"".join(c.pack() for c in self._commitments)
        commitment_count = len(self._commitments)

        # Layer B: checksums
        checksum_data = b"".join(c.pack() for c in self._checksums)
        checksum_count = len(self._checksums)

        # Layer C: tick snapshots
        snapshot_data = b"".join(s.pack() for s in snapshots)
        snapshot_count = len(snapshots)

        # Layer C: metadata (JSON)
        metadata = {
            "match_id": match_id,
            "total_ticks": final_state.tick_id,
            "winner": final_state.winner,
            "player_hp_final": final_state.player.hp,
            "ai_hp_final": final_state.ai.hp,
            "commitment_count": commitment_count,
            "checksum_count": checksum_count,
            "snapshot_count": snapshot_count,
        }
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        # --- Build the file ---
        # Layout:
        #   [MAGIC 6B]
        #   [HEADER 256B]
        #   [Section Table 32B: 4 x (offset: uint32, count: uint32)]
        #   [Layer A: initial_state_len(uint32) + initial_state_data + commitments]
        #   [Layer B: checksums]
        #   [Layer C: snapshots + metadata_len(uint32) + metadata]

        section_table_size = 4 * 8  # 4 sections × (4B offset + 4B count)
        base_offset = len(MAGIC) + HEADER_SIZE + section_table_size

        # Section offsets (relative to start of file)
        layer_a_offset = base_offset
        layer_a_size = 4 + initial_state_len + len(commitment_data)

        layer_b_offset = layer_a_offset + layer_a_size
        layer_b_size = len(checksum_data)

        layer_c_offset = layer_b_offset + layer_b_size
        layer_c_size = len(snapshot_data) + 4 + len(metadata_bytes)

        # Pack section table: (offset, count) for each section
        # Section 0: initial state (offset, length in bytes)
        # Section 1: commitments (offset, count)
        # Section 2: checksums (offset, count)
        # Section 3: snapshots (offset, count)
        section_table = struct.pack(
            "<IIIIIIII",
            layer_a_offset, initial_state_len,
            layer_a_offset + 4 + initial_state_len, commitment_count,
            layer_b_offset, checksum_count,
            layer_c_offset, snapshot_count,
        )

        # Assemble file
        parts = [
            MAGIC,
            header.pack(),
            section_table,
            # Layer A
            struct.pack("<I", initial_state_len),
            initial_state_data,
            commitment_data,
            # Layer B
            checksum_data,
            # Layer C
            snapshot_data,
            struct.pack("<I", len(metadata_bytes)),
            metadata_bytes,
        ]

        replay_bytes = b"".join(parts)

        # Write to disk
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{match_id}.replay"
        path = REPLAY_DIR / filename
        path.write_bytes(replay_bytes)

        log.debug("Replay written: %s (%d bytes, %d commitments, %d checksums, %d snapshots)",
                  filename, len(replay_bytes), commitment_count, checksum_count, snapshot_count)

        return path
