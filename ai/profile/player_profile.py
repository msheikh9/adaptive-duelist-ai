"""Player profile: persistent behavioral fingerprint of a human player.

Tracks cumulative statistics across all sessions. Updated incrementally
from SemanticEvents by ProfileUpdater. Persisted to SQLite as a single
canonical row per player_id.

All frequency/distribution fields use string keys (commitment names, zone
names) so the JSON roundtrip is lossless without enum reconstruction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


PLAYER_ID = "player_1"

# Thresholds for contextual distributions
LOW_HP_FRACTION = 0.30        # actor_hp below this fraction → low-HP context
CLOSE_ZONE_NAME = "CLOSE"     # spacing zone name for cornered proxy


@dataclass
class PlayerProfile:
    """All behavioral statistics tracked for a player.

    Mutable. Updated in-place by ProfileUpdater and BehaviorModel.
    Serializable to/from the SQLite player_profiles row.
    """

    player_id: str = PLAYER_ID

    # --- Observation volume ---
    session_count: int = 0           # Engine starts (incremented on load)
    total_ticks_observed: int = 0    # Sum of all match tick counts

    # --- Action frequency tables (commitment.name → count) ---
    action_frequencies: dict[str, int] = field(default_factory=dict)
    recent_action_frequencies: dict[str, int] = field(default_factory=dict)

    # --- N-gram transition counts ---
    # bigrams:  {prev_name: {next_name: count}}
    # trigrams: {"prev2_name,prev1_name": {next_name: count}}
    bigrams: dict[str, dict[str, int]] = field(default_factory=dict)
    trigrams: dict[str, dict[str, int]] = field(default_factory=dict)

    # --- Spatial behavior ---
    spacing_distribution: dict[str, int] = field(default_factory=dict)
    movement_direction_bias: float = 0.0   # -1=all-left, +1=all-right

    # --- Dodge behavior ---
    dodge_frequency: float = 0.0
    dodge_left_pct: float = 0.5
    dodge_right_pct: float = 0.5

    # --- Tactical tendencies ---
    aggression_index: float = 0.0          # attacks / total commitments
    initiative_rate: float = 0.0           # commits-while-opponent-free / total
    punish_conversion_rate: float = 0.0    # punish successes / hit opportunities

    # --- Context-specific action distributions ---
    low_hp_action_distribution: dict[str, int] = field(default_factory=dict)
    cornered_action_distribution: dict[str, int] = field(default_factory=dict)

    # --- Reaction time (Welford online stats, ms) ---
    avg_reaction_time_ms: float = 0.0
    reaction_time_stddev: float = 0.0

    # --- Match-level outcomes ---
    win_rate_vs_ai: float = 0.0
    avg_match_duration: float = 0.0        # seconds

    # --- Accumulators (stored via migration v002) ---
    match_count: int = 0
    win_count: int = 0
    reaction_count: int = 0        # observations for Welford
    reaction_M2: float = 0.0       # Welford M2 accumulator
    duration_sum: float = 0.0      # total match seconds

    # ------------------------------------------------------------------ #
    # Read-only convenience properties                                     #
    # ------------------------------------------------------------------ #

    @property
    def total_commitments(self) -> int:
        return sum(self.action_frequencies.values())

    @property
    def total_attacks(self) -> int:
        return (self.action_frequencies.get("LIGHT_ATTACK", 0)
                + self.action_frequencies.get("HEAVY_ATTACK", 0))

    @property
    def total_dodges(self) -> int:
        return (self.action_frequencies.get("DODGE_BACKWARD", 0)
                + self.action_frequencies.get("DODGE_LEFT", 0)
                + self.action_frequencies.get("DODGE_RIGHT", 0))

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_db_row(self) -> tuple:
        """Return values for INSERT/UPDATE matching the player_profiles schema."""
        now = datetime.now(timezone.utc).isoformat()
        return (
            self.player_id,
            self.session_count,
            self.total_ticks_observed,
            json.dumps(self.action_frequencies),
            json.dumps(self.recent_action_frequencies),
            json.dumps(self.bigrams),
            json.dumps(self.trigrams),
            json.dumps(self.spacing_distribution),
            self.movement_direction_bias,
            self.dodge_left_pct,
            self.dodge_right_pct,
            self.dodge_frequency,
            self.aggression_index,
            self.initiative_rate,
            self.punish_conversion_rate,
            json.dumps(self.low_hp_action_distribution),
            json.dumps(self.cornered_action_distribution),
            json.dumps([]),          # combo_sequences placeholder
            self.avg_reaction_time_ms,
            self.reaction_time_stddev,
            self.win_rate_vs_ai,
            self.avg_match_duration,
            now,
            # v002 accumulator columns
            self.match_count,
            self.win_count,
            self.reaction_count,
            self.reaction_M2,
            self.duration_sum,
        )

    @classmethod
    def from_db_row(cls, row) -> PlayerProfile:
        """Reconstruct a PlayerProfile from a sqlite3.Row."""

        def _j(col: str, default):
            raw = row[col]
            if raw is None:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        def _f(col: str, default: float = 0.0) -> float:
            v = row[col]
            return float(v) if v is not None else default

        def _i(col: str, default: int = 0) -> int:
            v = row[col]
            return int(v) if v is not None else default

        # v002 columns may not exist in older DBs
        def _i_safe(col: str, default: int = 0) -> int:
            try:
                return _i(col, default)
            except (IndexError, KeyError):
                return default

        def _f_safe(col: str, default: float = 0.0) -> float:
            try:
                return _f(col, default)
            except (IndexError, KeyError):
                return default

        return cls(
            player_id=row["player_id"],
            session_count=_i("session_count"),
            total_ticks_observed=_i("total_ticks_observed"),
            action_frequencies=_j("action_frequencies", {}),
            recent_action_frequencies=_j("recent_action_frequencies", {}),
            bigrams=_j("bigrams", {}),
            trigrams=_j("trigrams", {}),
            spacing_distribution=_j("spacing_distribution", {}),
            movement_direction_bias=_f("movement_direction_bias"),
            dodge_left_pct=_f("dodge_left_pct", 0.5),
            dodge_right_pct=_f("dodge_right_pct", 0.5),
            dodge_frequency=_f("dodge_frequency"),
            aggression_index=_f("aggression_index"),
            initiative_rate=_f("initiative_rate"),
            punish_conversion_rate=_f("punish_conversion_rate"),
            low_hp_action_distribution=_j("low_hp_action_dist", {}),
            cornered_action_distribution=_j("cornered_action_dist", {}),
            avg_reaction_time_ms=_f("avg_reaction_time_ms"),
            reaction_time_stddev=_f("reaction_time_stddev"),
            win_rate_vs_ai=_f("win_rate_vs_ai"),
            avg_match_duration=_f("avg_match_duration"),
            match_count=_i_safe("match_count"),
            win_count=_i_safe("win_count"),
            reaction_count=_i_safe("reaction_count"),
            reaction_M2=_f_safe("reaction_M2"),
            duration_sum=_f_safe("duration_sum"),
        )

    def summary(self) -> str:
        """Human-readable single-line summary."""
        return (
            f"PlayerProfile("
            f"matches={self.match_count}, "
            f"win_rate={self.win_rate_vs_ai:.1%}, "
            f"aggression={self.aggression_index:.2f}, "
            f"init_rate={self.initiative_rate:.2f}, "
            f"react_ms={self.avg_reaction_time_ms:.0f}±{self.reaction_time_stddev:.0f}, "
            f"commitments={self.total_commitments})"
        )
