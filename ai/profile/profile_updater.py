"""Profile updater: processes SemanticEvents to update a PlayerProfile.

Maintains per-match transient accumulators (reaction times, punish counts,
initiative counts) and flushes derived metrics into the PlayerProfile on
match end. Handles the rolling recent-action window.

Does NOT update bigrams/trigrams — those are owned by MarkovPredictor
and synced to the profile by BehaviorModel at save time.
"""

from __future__ import annotations

import math
from collections import Counter, deque

from config.config_loader import AIConfig, GameConfig
from data.events import EventType, SemanticEvent
from game.combat.actions import (
    Actor,
    CombatCommitment,
    FSMState,
    FREE_STATES,
    SpacingZone,
)
from ai.profile.player_profile import LOW_HP_FRACTION, CLOSE_ZONE_NAME

# FSM states where the opponent is vulnerable to punishment
_PUNISH_STATES = frozenset({
    FSMState.ATTACK_RECOVERY,
    FSMState.DODGE_RECOVERY,
    FSMState.EXHAUSTED,
})

# Offensive commitments that count toward aggression_index
_ATTACK_NAMES = frozenset({
    CombatCommitment.LIGHT_ATTACK.name,
    CombatCommitment.HEAVY_ATTACK.name,
})

# All dodge variants
_DODGE_NAMES = frozenset({
    CombatCommitment.DODGE_BACKWARD.name,
    CombatCommitment.DODGE_LEFT.name,
    CombatCommitment.DODGE_RIGHT.name,
})


class ProfileUpdater:
    """Processes SemanticEvents and updates PlayerProfile fields in-place.

    Instances are long-lived (one per Engine). Per-match transient state
    is reset via on_match_start().
    """

    def __init__(self, ai_cfg: AIConfig, game_cfg: GameConfig) -> None:
        self._window_size = ai_cfg.profile.rolling_window_size
        self._tick_rate = game_cfg.simulation.tick_rate
        self._max_hp = game_cfg.fighter.max_hp

        # Rolling window for recent_action_frequencies
        self._rolling_window: deque[str] = deque(maxlen=self._window_size)

        # Per-match transient accumulators
        self._reaction_times_ms: list[float] = []
        self._punish_opps: int = 0       # player HIT_LANDED events
        self._punish_hits: int = 0       # subset where opponent was in punish state
        self._initiative_total: int = 0  # total player COMMITMENT_START events
        self._initiative_taken: int = 0  # subset where opponent was in FREE state

    def on_match_start(self, match_id: str) -> None:
        """Reset per-match accumulators."""
        self._reaction_times_ms.clear()
        self._punish_opps = 0
        self._punish_hits = 0
        self._initiative_total = 0
        self._initiative_taken = 0

    def on_event(self, profile, event: SemanticEvent) -> None:
        """Route event to appropriate handler."""
        if event.event_type == EventType.COMMITMENT_START and event.actor == Actor.PLAYER:
            self._on_player_commitment(profile, event)
        elif event.event_type == EventType.HIT_LANDED and event.actor == Actor.PLAYER:
            self._on_player_hit(profile, event)

    def on_match_end(self, profile, winner: str | None,
                     total_ticks: int) -> None:
        """Update match-level and session-level derived metrics."""
        # Match count and outcomes
        profile.match_count += 1
        if winner == "PLAYER":
            profile.win_count += 1
        profile.win_rate_vs_ai = (
            profile.win_count / profile.match_count
            if profile.match_count > 0 else 0.0
        )

        # Match duration
        duration_s = total_ticks / self._tick_rate
        profile.duration_sum += duration_s
        profile.avg_match_duration = profile.duration_sum / profile.match_count

        # Tick volume
        profile.total_ticks_observed += total_ticks

        # Reaction time: Welford online update over all observations this match
        for rt_ms in self._reaction_times_ms:
            _welford_update(profile, rt_ms)
        if profile.reaction_count >= 2:
            profile.reaction_time_stddev = math.sqrt(
                profile.reaction_M2 / (profile.reaction_count - 1)
            )

        # Punish conversion rate: lifetime (total hits tracked in profile field)
        # Already updated per-event; final sync of per-match punish data is in
        # _on_player_hit. Nothing extra needed here.

    # ------------------------------------------------------------------ #
    # Private handlers                                                     #
    # ------------------------------------------------------------------ #

    def _on_player_commitment(self, profile, event: SemanticEvent) -> None:
        key = event.commitment.name  # type: ignore[union-attr]

        # 1. Global action frequencies
        profile.action_frequencies[key] = (
            profile.action_frequencies.get(key, 0) + 1
        )

        # 2. Rolling window → recent_action_frequencies
        self._rolling_window.append(key)
        profile.recent_action_frequencies = dict(Counter(self._rolling_window))

        # 3. Spacing distribution
        if event.spacing_zone is not None:
            zone = event.spacing_zone.name
            profile.spacing_distribution[zone] = (
                profile.spacing_distribution.get(zone, 0) + 1
            )

        # 4. Low-HP context distribution
        if event.actor_hp < self._max_hp * LOW_HP_FRACTION:
            profile.low_hp_action_distribution[key] = (
                profile.low_hp_action_distribution.get(key, 0) + 1
            )

        # 5. Cornered proxy: CLOSE spacing with low stamina (≤20)
        if (event.spacing_zone is not None
                and event.spacing_zone == SpacingZone.CLOSE
                and event.actor_stamina <= 20):
            profile.cornered_action_distribution[key] = (
                profile.cornered_action_distribution.get(key, 0) + 1
            )

        # 6. Initiative tracking
        self._initiative_total += 1
        if (event.opponent_fsm_state is not None
                and event.opponent_fsm_state in FREE_STATES):
            self._initiative_taken += 1

        # 7. Reaction time (milliseconds)
        if event.reaction_ticks and event.reaction_ticks > 0:
            rt_ms = event.reaction_ticks / self._tick_rate * 1000.0
            self._reaction_times_ms.append(rt_ms)

        # --- Derived metrics (recomputed from totals) ---
        total = sum(profile.action_frequencies.values())
        if total == 0:
            return

        # Aggression index
        attack_total = sum(profile.action_frequencies.get(a, 0) for a in _ATTACK_NAMES)
        profile.aggression_index = attack_total / total

        # Dodge frequency and direction split
        dodge_total = sum(profile.action_frequencies.get(d, 0) for d in _DODGE_NAMES)
        profile.dodge_frequency = dodge_total / total
        left_d = profile.action_frequencies.get(CombatCommitment.DODGE_LEFT.name, 0)
        right_d = profile.action_frequencies.get(CombatCommitment.DODGE_RIGHT.name, 0)
        back_d = profile.action_frequencies.get(CombatCommitment.DODGE_BACKWARD.name, 0)
        directional = left_d + right_d
        if directional > 0:
            profile.dodge_left_pct = left_d / directional
            profile.dodge_right_pct = right_d / directional

        # Movement direction bias
        right_mv = profile.action_frequencies.get(CombatCommitment.MOVE_RIGHT.name, 0)
        left_mv = profile.action_frequencies.get(CombatCommitment.MOVE_LEFT.name, 0)
        move_total = right_mv + left_mv
        if move_total > 0:
            profile.movement_direction_bias = (right_mv - left_mv) / move_total

        # Initiative rate
        if self._initiative_total > 0:
            profile.initiative_rate = self._initiative_taken / self._initiative_total

    def _on_player_hit(self, profile, event: SemanticEvent) -> None:
        """Track punish conversion."""
        self._punish_opps += 1
        if (event.opponent_fsm_state is not None
                and event.opponent_fsm_state in _PUNISH_STATES):
            self._punish_hits += 1
        if self._punish_opps > 0:
            profile.punish_conversion_rate = self._punish_hits / self._punish_opps


# ------------------------------------------------------------------ #
# Welford's online algorithm for mean and variance                    #
# ------------------------------------------------------------------ #

def _welford_update(profile, new_value: float) -> None:
    """Update Welford running mean/M2 in profile with a single new observation."""
    profile.reaction_count += 1
    delta = new_value - profile.avg_reaction_time_ms
    profile.avg_reaction_time_ms += delta / profile.reaction_count
    delta2 = new_value - profile.avg_reaction_time_ms
    profile.reaction_M2 += delta * delta2
