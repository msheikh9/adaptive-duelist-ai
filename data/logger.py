"""Buffered game event and tick snapshot logger.

Maintains in-memory ring buffers for TickSnapshots and SemanticEvents.
Batch-flushes to SQLite on a configurable interval (default: every 60
ticks / 1 second) and always on match end. Never blocks the game loop.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from data.events import SemanticEvent
from data.serializer import semantic_event_to_row
from data.tick_snapshot import TickSnapshot

if TYPE_CHECKING:
    from data.db import Database

log = logging.getLogger(__name__)

SEMANTIC_EVENT_INSERT = """
INSERT INTO semantic_events (
    event_type, match_id, tick_id, actor, commitment,
    opponent_fsm_state, opponent_commitment, spacing_zone,
    actor_hp, opponent_hp, actor_stamina, opponent_stamina,
    damage_dealt, reaction_ticks
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class GameLogger:
    """Buffered logger for tick snapshots and semantic events.

    Tick snapshots are stored in memory only (flushed to replay file
    by the recorder, not to SQLite). Semantic events are batch-flushed
    to SQLite.
    """

    def __init__(self, db: Database, flush_interval: int = 60) -> None:
        self._db = db
        self._flush_interval = flush_interval
        self._tick_buffer: deque[TickSnapshot] = deque()
        self._event_buffer: deque[SemanticEvent] = deque()
        self._ticks_since_flush = 0

    @property
    def tick_buffer(self) -> deque[TickSnapshot]:
        return self._tick_buffer

    @property
    def event_buffer(self) -> deque[SemanticEvent]:
        return self._event_buffer

    def record_tick(self, snapshot: TickSnapshot) -> None:
        """Buffer a tick snapshot. Called every simulation tick."""
        self._tick_buffer.append(snapshot)
        self._ticks_since_flush += 1
        if self._ticks_since_flush >= self._flush_interval:
            self.flush_events()

    def record_event(self, event: SemanticEvent) -> None:
        """Buffer a semantic event. Called on meaningful state transitions."""
        self._event_buffer.append(event)

    def flush_events(self) -> None:
        """Write all buffered semantic events to SQLite."""
        if not self._event_buffer:
            self._ticks_since_flush = 0
            return

        rows = [semantic_event_to_row(e) for e in self._event_buffer]
        success = self._db.executemany_safe(SEMANTIC_EVENT_INSERT, rows)

        if success:
            self._event_buffer.clear()
        else:
            log.warning(
                "Failed to flush %d events to database. Events retained in buffer.",
                len(rows),
            )
        self._ticks_since_flush = 0

    def flush_all(self) -> None:
        """Flush all buffers. Called on match end."""
        self.flush_events()

    def drain_tick_buffer(self) -> list[TickSnapshot]:
        """Remove and return all tick snapshots. Used by replay recorder."""
        snapshots = list(self._tick_buffer)
        self._tick_buffer.clear()
        return snapshots

    def clear(self) -> None:
        """Clear all buffers without flushing. Used on reset."""
        self._tick_buffer.clear()
        self._event_buffer.clear()
        self._ticks_since_flush = 0
