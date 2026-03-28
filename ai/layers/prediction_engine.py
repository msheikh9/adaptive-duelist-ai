"""Prediction Engine: owns the ensemble, manages inference triggers,
feature extraction, model lifecycle, and weight updates.

This is the primary entry point for all prediction intelligence. It:
  - Owns the EnsemblePredictor (Markov + sklearn blend)
  - Manages inference triggers (COMMITMENT_END + idle timeout)
  - Extracts features from SemanticEvents for sklearn
  - Loads/activates sklearn model after sufficient matches
  - Triggers retraining when configured
  - Tracks last prediction for weight update feedback

The engine calls on_event() for every SemanticEvent. The tactical
planner (Phase 5) reads from predict().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai.features.feature_extractor import extract_features, HISTORY_SLOTS
from ai.models.base_predictor import LABEL_HOLD, PHASE1_NAMES, PredictionResult, make_prediction_result
from ai.models.ensemble_predictor import EnsemblePredictor
from ai.models.sklearn_predictor import SklearnPredictor
from ai.training.model_trainer import load_latest_model
from config.config_loader import AIConfig, GameConfig
from data.events import EventType, SemanticEvent
from game.combat.actions import Actor, CombatCommitment

if TYPE_CHECKING:
    from ai.layers.behavior_model import BehaviorModel
    from data.db import Database

log = logging.getLogger(__name__)

_PHASE1_SET = frozenset(PHASE1_NAMES)


class PredictionEngine:
    """Orchestrates prediction across Markov and sklearn models.

    Lifecycle:
        engine = PredictionEngine(db, behavior_model, ai_cfg, game_cfg)
        engine.try_load_sklearn()     # once, on startup

        # per match:
        engine.on_match_start(match_id)
        # during match:
        engine.on_event(event)        # triggers re-prediction as needed
        # read:
        engine.predict()              # returns latest PredictionResult
        # on match end:
        engine.on_match_end()
    """

    def __init__(
        self,
        db: Database,
        behavior_model: BehaviorModel,
        ai_cfg: AIConfig,
        game_cfg: GameConfig,
    ) -> None:
        self._db = db
        self._bm = behavior_model
        self._ai_cfg = ai_cfg
        self._game_cfg = game_cfg

        # Config shortcuts
        self._window_ticks = ai_cfg.prediction.window_ticks
        self._idle_reeval_ticks = ai_cfg.prediction.inference_reeval_idle_ticks
        self._sklearn_activation_matches = ai_cfg.ensemble.sklearn_activation_matches
        self._max_hp = game_cfg.fighter.max_hp
        self._max_stamina = game_cfg.fighter.max_stamina
        self._tick_rate = game_cfg.simulation.tick_rate

        # Ensemble owns the Markov predictor from BehaviorModel
        self._ensemble = EnsemblePredictor(
            behavior_model.predictor,
            initial_markov_weight=ai_cfg.ensemble.initial_markov_weight,
            initial_sklearn_weight=ai_cfg.ensemble.initial_sklearn_weight,
            ema_alpha=ai_cfg.ensemble.weight_update_ema_alpha,
        )

        # Inference state
        self._last_prediction: PredictionResult | None = None
        self._last_prediction_tick: int = -1
        self._last_player_commit_tick: int = -1
        self._history: list[str] = []  # recent commitment names for features
        self._last_event: SemanticEvent | None = None  # most recent player event for features
        self._match_id: str = ""

        # Pending feedback: the prediction we made before the player committed
        self._pending_markov_result: PredictionResult | None = None
        self._pending_sklearn_result: PredictionResult | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def try_load_sklearn(self) -> None:
        """Attempt to load the active sklearn model from the registry."""
        match_count = self._bm.profile.match_count
        if match_count < self._sklearn_activation_matches:
            log.info(
                "Sklearn not activated: %d/%d matches.",
                match_count, self._sklearn_activation_matches,
            )
            return

        clf, version = load_latest_model(self._db)
        if clf is not None and version is not None:
            self._ensemble.activate_sklearn(SklearnPredictor(clf, version))
        else:
            log.info("No trained sklearn model found in registry.")

    def on_match_start(self, match_id: str) -> None:
        """Reset per-match prediction state."""
        self._match_id = match_id
        self._last_prediction = None
        self._last_prediction_tick = -1
        self._last_player_commit_tick = -1
        self._history.clear()
        self._last_event = None
        self._pending_markov_result = None
        self._pending_sklearn_result = None

    def on_event(self, event: SemanticEvent) -> None:
        """Process an event and trigger re-prediction if appropriate.

        Inference triggers:
          1. COMMITMENT_END from player → predict what comes next.
          2. Idle timeout: if no player commitment for idle_reeval_ticks
             after last prediction, re-predict.

        We also track player COMMITMENT_START for weight update feedback.
        """
        if event.actor != Actor.PLAYER:
            return

        if event.event_type == EventType.COMMITMENT_START:
            self._on_player_commit_start(event)
        elif event.event_type == EventType.COMMITMENT_END:
            self._on_player_commit_end(event)

    def on_tick(self, tick_id: int) -> None:
        """Check idle timeout trigger. Called each tick from the engine."""
        if self._last_prediction is None:
            return
        if self._last_player_commit_tick < 0:
            return
        idle_ticks = tick_id - self._last_player_commit_tick
        if idle_ticks >= self._idle_reeval_ticks:
            if tick_id - self._last_prediction_tick >= self._idle_reeval_ticks:
                self._run_prediction(tick_id)

    def on_match_end(self) -> None:
        """Finalize match-level prediction tracking."""
        # Check if we should activate sklearn for next match
        match_count = self._bm.profile.match_count + 1  # about to increment
        if (not self._ensemble.sklearn_active
                and match_count >= self._sklearn_activation_matches):
            self.try_load_sklearn()

    # ------------------------------------------------------------------ #
    # Prediction API                                                       #
    # ------------------------------------------------------------------ #

    def predict(self) -> PredictionResult:
        """Return the latest prediction, or a null result if none exists."""
        if self._last_prediction is not None:
            return self._last_prediction
        return make_prediction_result({}, source="none", markov_level="none")

    @property
    def ensemble(self) -> EnsemblePredictor:
        return self._ensemble

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _on_player_commit_start(self, event: SemanticEvent) -> None:
        """Player started a new commitment. Use for weight update feedback."""
        if event.commitment is None:
            return

        name = event.commitment.name
        self._last_player_commit_tick = event.tick_id
        self._last_event = event

        # Weight update: compare last prediction to actual
        if (self._pending_markov_result is not None
                and name in _PHASE1_SET):
            self._ensemble.update_weights(
                name,
                self._pending_markov_result,
                self._pending_sklearn_result,
            )
            self._pending_markov_result = None
            self._pending_sklearn_result = None

        # Update history
        if name in _PHASE1_SET:
            self._history.append(name)
            if len(self._history) > HISTORY_SLOTS:
                self._history = self._history[-HISTORY_SLOTS:]

    def _on_player_commit_end(self, event: SemanticEvent) -> None:
        """Player finished a commitment. Trigger prediction for what comes next."""
        self._run_prediction(event.tick_id)

    def _run_prediction(self, tick_id: int) -> None:
        """Execute the ensemble prediction pipeline."""
        features: list[float] | None = None

        if self._ensemble.sklearn_active and self._last_event is not None:
            features = extract_features(
                self._last_event,
                self._history,
                self._bm.profile,
                self._max_hp,
                self._max_stamina,
                self._tick_rate,
            )

        # Store individual results for weight update feedback
        self._pending_markov_result = self._bm.predictor.predict()
        if (self._ensemble.sklearn_active
                and self._ensemble._sklearn is not None
                and features is not None):
            self._pending_sklearn_result = self._ensemble._sklearn.predict_from_features(features)
        else:
            self._pending_sklearn_result = None

        # Get blended result
        result = self._ensemble.predict(features)
        self._last_prediction = result
        self._last_prediction_tick = tick_id
