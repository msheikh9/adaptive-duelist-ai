"""Behavior Model Layer: orchestrates profile updates, Markov prediction,
and persistence.

This is the primary entry point for all behavioral intelligence. It:
  - Owns the canonical PlayerProfile
  - Owns the MarkovPredictor (which tracks n-gram state)
  - Delegates event processing to ProfileUpdater
  - Loads from DB on startup
  - Persists to DB on match end

The engine calls on_event() for every SemanticEvent it emits. All
other AI layers read from this model via its public properties.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.config_loader import AIConfig, GameConfig
from data.events import EventType, SemanticEvent
from game.combat.actions import Actor, CombatCommitment
from ai.models.markov_predictor import MarkovPredictor, PredictionResult
from ai.profile.player_profile import PLAYER_ID, PlayerProfile
from ai.profile.profile_updater import ProfileUpdater

if TYPE_CHECKING:
    from data.db import Database

log = logging.getLogger(__name__)

_UPSERT_SQL = """
INSERT INTO player_profiles (
    player_id, session_count, total_ticks_observed,
    action_frequencies, recent_action_frequencies, bigrams, trigrams,
    spacing_distribution, movement_direction_bias,
    dodge_left_pct, dodge_right_pct, dodge_frequency,
    aggression_index, initiative_rate, punish_conversion_rate,
    low_hp_action_dist, cornered_action_dist, combo_sequences,
    avg_reaction_time_ms, reaction_time_stddev,
    win_rate_vs_ai, avg_match_duration, last_updated,
    match_count, win_count, reaction_count, reaction_M2, duration_sum
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(player_id) DO UPDATE SET
    session_count           = excluded.session_count,
    total_ticks_observed    = excluded.total_ticks_observed,
    action_frequencies      = excluded.action_frequencies,
    recent_action_frequencies = excluded.recent_action_frequencies,
    bigrams                 = excluded.bigrams,
    trigrams                = excluded.trigrams,
    spacing_distribution    = excluded.spacing_distribution,
    movement_direction_bias = excluded.movement_direction_bias,
    dodge_left_pct          = excluded.dodge_left_pct,
    dodge_right_pct         = excluded.dodge_right_pct,
    dodge_frequency         = excluded.dodge_frequency,
    aggression_index        = excluded.aggression_index,
    initiative_rate         = excluded.initiative_rate,
    punish_conversion_rate  = excluded.punish_conversion_rate,
    low_hp_action_dist      = excluded.low_hp_action_dist,
    cornered_action_dist    = excluded.cornered_action_dist,
    combo_sequences         = excluded.combo_sequences,
    avg_reaction_time_ms    = excluded.avg_reaction_time_ms,
    reaction_time_stddev    = excluded.reaction_time_stddev,
    win_rate_vs_ai          = excluded.win_rate_vs_ai,
    avg_match_duration      = excluded.avg_match_duration,
    last_updated            = excluded.last_updated,
    match_count             = excluded.match_count,
    win_count               = excluded.win_count,
    reaction_count          = excluded.reaction_count,
    reaction_M2             = excluded.reaction_M2,
    duration_sum            = excluded.duration_sum;
"""

_SELECT_SQL = """
SELECT * FROM player_profiles WHERE player_id = ? LIMIT 1;
"""


class BehaviorModel:
    """Orchestrates player behavior tracking for a single session.

    Lifecycle:
        model = BehaviorModel(db, ai_cfg, game_cfg)
        model.load_profile()         # once, on engine startup

        # per match:
        model.on_match_start(match_id)
        # during match:
        model.on_event(event)        # called for every SemanticEvent
        # on match end:
        model.on_match_end(winner, total_ticks)
    """

    def __init__(self, db: Database, ai_cfg: AIConfig,
                 game_cfg: GameConfig) -> None:
        self._db = db
        self._ai_cfg = ai_cfg
        self._game_cfg = game_cfg

        self._profile = PlayerProfile()
        self._updater = ProfileUpdater(ai_cfg, game_cfg)
        self._predictor = MarkovPredictor(order=ai_cfg.prediction.markov_order)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load_profile(self) -> None:
        """Load saved profile from DB. Safe to call when no record exists."""
        try:
            row = self._db.fetchone(_SELECT_SQL, (PLAYER_ID,))
            if row is not None:
                self._profile = PlayerProfile.from_db_row(row)
                self._predictor.load_from_profile(self._profile)
                log.info("Profile loaded: %s", self._profile.summary())
            else:
                log.info("No existing profile found. Starting fresh.")
        except Exception:
            log.exception("Failed to load player profile. Using empty profile.")

        # Increment session counter (this Engine run = new session)
        self._profile.session_count += 1

    def on_match_start(self, match_id: str) -> None:
        """Reset per-match accumulators."""
        self._updater.on_match_start(match_id)
        self._predictor.reset_history()

    def on_event(self, event: SemanticEvent) -> None:
        """Process a SemanticEvent. Called from the engine for every event."""
        # ProfileUpdater handles all non-ngram profile fields
        self._updater.on_event(self._profile, event)

        # MarkovPredictor handles n-gram state (player commitments only)
        if (event.event_type == EventType.COMMITMENT_START
                and event.actor == Actor.PLAYER
                and event.commitment is not None):
            self._predictor.update(event.commitment)

    def on_match_end(self, winner: str | None, total_ticks: int) -> None:
        """Finalize match-level metrics and persist profile to DB."""
        self._updater.on_match_end(self._profile, winner, total_ticks)
        # Sync n-gram counts from predictor into profile before saving
        self._predictor.sync_to_profile(self._profile)
        self._save_profile()

    # ------------------------------------------------------------------ #
    # Public read API                                                      #
    # ------------------------------------------------------------------ #

    @property
    def profile(self) -> PlayerProfile:
        return self._profile

    @property
    def predictor(self) -> MarkovPredictor:
        return self._predictor

    def predict_next(self) -> PredictionResult:
        """Return the Markov predictor's current prediction."""
        return self._predictor.predict()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save_profile(self) -> None:
        """Write profile to DB via UPSERT."""
        try:
            row = self._profile.to_db_row()
            self._db.execute_safe(_UPSERT_SQL, row)
            log.debug("Profile saved: %s", self._profile.summary())
        except Exception:
            log.exception("Failed to save player profile.")
