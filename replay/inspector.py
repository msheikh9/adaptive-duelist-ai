"""Replay inspector: high-level API for querying replay data.

Provides tick-level access to replay state without re-simulating.
Uses Layer C snapshots for fast random access and Layer A commitments
for event queries.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from data.tick_snapshot import TickSnapshot, TICK_SNAPSHOT_SIZE
from game.combat.actions import Actor, CombatCommitment
from replay.format import CommitmentRecord
from replay.replay_player import ReplayData, load_replay

from pathlib import Path


class InspectorError(Exception):
    """Raised when inspector operations fail."""


class ReplayInspector:
    """Query API for replay data.

    Provides:
    - get_snapshot_at_tick(t): TickSnapshot at a given tick
    - get_commitments_at_tick(t): list of commitments at a given tick
    - get_commitments_in_range(start, end): commitments in a tick range
    - get_all_commitments_for(actor): all commitments by a given actor
    - metadata: parsed Layer C metadata dict
    """

    def __init__(self, replay: ReplayData) -> None:
        self._replay = replay
        self._snapshots: list[TickSnapshot] | None = None
        self._commitments_by_tick: dict[int, list[CommitmentRecord]] | None = None
        self._metadata: dict | None = None

    @classmethod
    def from_file(cls, path: Path) -> ReplayInspector:
        """Load a replay file and create an inspector."""
        replay = load_replay(path)
        return cls(replay)

    @property
    def header(self):
        return self._replay.header

    @property
    def total_ticks(self) -> int:
        return self._replay.header.total_ticks

    @property
    def metadata(self) -> dict:
        """Parsed Layer C metadata."""
        if self._metadata is None:
            if self._replay.metadata_raw:
                self._metadata = json.loads(self._replay.metadata_raw.decode("utf-8"))
            else:
                self._metadata = {}
        return self._metadata

    @property
    def commitment_count(self) -> int:
        return len(self._replay.commitments)

    @property
    def checksum_count(self) -> int:
        return len(self._replay.checksums)

    def get_snapshot_at_tick(self, tick: int) -> TickSnapshot | None:
        """Get the TickSnapshot for a specific tick. Returns None if unavailable."""
        snapshots = self._get_snapshots()
        if not snapshots:
            return None
        if tick < 0 or tick >= len(snapshots):
            return None
        return snapshots[tick]

    def get_commitments_at_tick(self, tick: int) -> list[CommitmentRecord]:
        """Get all commitments that occurred at a specific tick."""
        index = self._get_commitment_index()
        return index.get(tick, [])

    def get_commitments_in_range(self, start: int, end: int) -> list[CommitmentRecord]:
        """Get all commitments in the tick range [start, end)."""
        result = []
        for c in self._replay.commitments:
            if start <= c.tick_id < end:
                result.append(c)
        return result

    def get_all_commitments_for(self, actor: Actor) -> list[CommitmentRecord]:
        """Get all commitments made by a specific actor."""
        return [c for c in self._replay.commitments if c.actor == actor]

    def get_snapshot_range(self, start: int, end: int) -> list[TickSnapshot]:
        """Get snapshots for tick range [start, end). Returns available snapshots."""
        snapshots = self._get_snapshots()
        if not snapshots:
            return []
        start = max(0, start)
        end = min(len(snapshots), end)
        return snapshots[start:end]

    def _get_snapshots(self) -> list[TickSnapshot]:
        """Lazily parse Layer C snapshot bytes."""
        if self._snapshots is None:
            self._snapshots = []
            raw = self._replay.snapshots_raw
            if raw:
                for i in range(0, len(raw), TICK_SNAPSHOT_SIZE):
                    chunk = raw[i:i + TICK_SNAPSHOT_SIZE]
                    if len(chunk) == TICK_SNAPSHOT_SIZE:
                        self._snapshots.append(TickSnapshot.unpack(chunk))
        return self._snapshots

    def _get_commitment_index(self) -> dict[int, list[CommitmentRecord]]:
        """Lazily build tick→commitments index."""
        if self._commitments_by_tick is None:
            self._commitments_by_tick = {}
            for c in self._replay.commitments:
                self._commitments_by_tick.setdefault(c.tick_id, []).append(c)
        return self._commitments_by_tick
