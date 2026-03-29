"""Build labeled training datasets from the semantic_events table.

Each sample is a player COMMITMENT_START event. The label is the *next*
player COMMITMENT_START commitment within `window_ticks`, or "HOLD" if
the player made no commitment in that window.

Data source is strictly semantic_events (no TickSnapshot coupling).
The builder reconstructs a rolling history of recent commitment names
per match to supply one-hot history features.

Chronological ordering is preserved — no shuffling — so callers can
perform time-based train/test splits without cross-match leakage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai.features.feature_extractor import (
    HISTORY_SLOTS,
    NUM_FEATURES,
    extract_features,
)
from ai.models.base_predictor import LABEL_HOLD, PHASE1_NAMES
from data.events import EventType
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone

if TYPE_CHECKING:
    from data.db import Database
    from ai.profile.player_profile import PlayerProfile

log = logging.getLogger(__name__)

_PHASE1_SET = frozenset(PHASE1_NAMES)

# SQL to pull all player COMMITMENT_START events ordered chronologically.
_EVENTS_SQL = """
SELECT
    match_id, tick_id, commitment, opponent_fsm_state, spacing_zone,
    actor_hp, opponent_hp, actor_stamina, opponent_stamina,
    reaction_ticks
FROM semantic_events
WHERE event_type = 'COMMITMENT_START' AND actor = 'PLAYER'
ORDER BY match_id, tick_id;
"""

_EVENTS_SQL_SOURCE_FILTERED = """
SELECT
    se.match_id, se.tick_id, se.commitment, se.opponent_fsm_state,
    se.spacing_zone, se.actor_hp, se.opponent_hp,
    se.actor_stamina, se.opponent_stamina, se.reaction_ticks
FROM semantic_events se
JOIN matches m ON se.match_id = m.match_id
WHERE se.event_type = 'COMMITMENT_START' AND se.actor = 'PLAYER'
  AND m.source = ?
ORDER BY se.match_id, se.tick_id;
"""


def _reconstruct_event(row, match_id: str) -> tuple:
    """Convert a DB row into the fields needed for feature extraction.

    Returns (match_id, tick_id, commitment_name, event-like namespace).
    """
    commitment_name = row["commitment"]
    spacing = None
    if row["spacing_zone"]:
        try:
            spacing = SpacingZone[row["spacing_zone"]]
        except KeyError:
            pass

    opp_fsm = None
    if row["opponent_fsm_state"]:
        try:
            opp_fsm = FSMState[row["opponent_fsm_state"]]
        except KeyError:
            pass

    # Lightweight namespace matching SemanticEvent's attribute API
    class _Ev:
        pass

    ev = _Ev()
    ev.actor_hp = row["actor_hp"]
    ev.opponent_hp = row["opponent_hp"]
    ev.actor_stamina = row["actor_stamina"]
    ev.opponent_stamina = row["opponent_stamina"]
    ev.spacing_zone = spacing
    ev.opponent_fsm_state = opp_fsm
    ev.reaction_ticks = row["reaction_ticks"]
    ev.tick_id = row["tick_id"]
    return commitment_name, ev


def build_dataset(
    db: Database,
    profile: PlayerProfile,
    max_hp: int,
    max_stamina: int,
    tick_rate: int,
    window_ticks: int = 60,
    source_filter: str | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Build (X, y) from all stored semantic events.

    Returns
    -------
    X : list of feature vectors (each length NUM_FEATURES).
    y : list of label strings (Phase1 commitment names or "HOLD").

    Rows are in chronological order. No shuffling is applied.
    """
    if source_filter is not None:
        rows = db.fetchall(_EVENTS_SQL_SOURCE_FILTERED, (source_filter,))
    else:
        rows = db.fetchall(_EVENTS_SQL)
    if not rows:
        return [], []

    # Group rows by match_id while preserving order
    matches: list[list] = []
    current_match: str | None = None
    current_rows: list = []
    for row in rows:
        mid = row["match_id"]
        if mid != current_match:
            if current_rows:
                matches.append(current_rows)
            current_match = mid
            current_rows = []
        current_rows.append(row)
    if current_rows:
        matches.append(current_rows)

    X: list[list[float]] = []
    y: list[str] = []

    for match_rows in matches:
        _build_match_samples(
            match_rows, profile, max_hp, max_stamina, tick_rate, window_ticks,
            X, y,
        )

    log.info("Dataset built: %d samples (%d matches)", len(X), len(matches))
    return X, y


def _build_match_samples(
    match_rows: list,
    profile: PlayerProfile,
    max_hp: int,
    max_stamina: int,
    tick_rate: int,
    window_ticks: int,
    X_out: list[list[float]],
    y_out: list[str],
) -> None:
    """Process one match's commitment events into samples.

    For each event i, the label is:
      - The commitment name of event i+1 if it arrives within window_ticks.
      - "HOLD" otherwise.

    Only Phase1 commitment names produce valid labels; non-Phase1
    commitments are skipped as samples but still tracked in history.
    """
    history: list[str] = []

    for i, row in enumerate(match_rows):
        name, ev = _reconstruct_event(row, row["match_id"])

        # Only Phase1 commitments are valid observation points
        if name not in _PHASE1_SET:
            history.append(name)
            if len(history) > HISTORY_SLOTS:
                history = history[-HISTORY_SLOTS:]
            continue

        # Determine label from next event
        label = LABEL_HOLD
        if i + 1 < len(match_rows):
            next_row = match_rows[i + 1]
            next_name = next_row["commitment"]
            gap = next_row["tick_id"] - row["tick_id"]
            if gap <= window_ticks and next_name in _PHASE1_SET:
                label = next_name

        features = extract_features(
            ev, history, profile, max_hp, max_stamina, tick_rate,
        )
        X_out.append(features)
        y_out.append(label)

        # Advance history
        history.append(name)
        if len(history) > HISTORY_SLOTS:
            history = history[-HISTORY_SLOTS:]
