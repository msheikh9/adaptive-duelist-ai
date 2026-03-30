"""Tactical planner: orchestrates prediction reading, context building,
mode selection, action resolution, and decision caching.

This is the top-level AI decision layer. The engine calls decide() each
tick; the planner returns None (wait) or a CombatCommitment (act now).

The planner supports three tiers:
  T0: baseline random AI (handled externally by BaselineAIController)
  T1: markov-only simplified planner (prediction + strategy, no memory updates)
  T2: full adaptive planner with memory

Tier is set at construction and does not change mid-session.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from ai.layers.prediction_engine import PredictionEngine
from ai.models.base_predictor import LABEL_HOLD, PredictionResult, make_prediction_result
from ai.strategy.action_resolver import ResolverOutput, resolve
from ai.strategy.ai_context import AIContext
from ai.strategy.planner_memory import ModeOutcome, PlannerMemory, PredictionOutcome
from ai.strategy.session_memory import SessionMemory
from ai.strategy.strategy_selector import select_mode
from ai.strategy.tactics import TacticalIntent
from game.arena import classify_spacing
from game.combat.actions import (
    Actor,
    CombatCommitment,
    FSMState,
    FREE_STATES,
    SpacingZone,
)
from game.combat.state_machine import can_commit
from game.entities.fighter import attempt_commitment

if TYPE_CHECKING:
    from ai.layers.behavior_model import BehaviorModel
    from config.config_loader import AIConfig, GameConfig
    from data.db import Database
    from game.state import FighterState, SimulationState

log = logging.getLogger(__name__)


class AITier(Enum):
    T0_BASELINE = 0
    T1_MARKOV_ONLY = 1
    T2_FULL_ADAPTIVE = 2


_AI_DECISION_INSERT = """
INSERT INTO ai_decisions (
    session_id, match_id, tick_id, predicted_top, pred_confidence,
    pred_probs, tactical_mode, ai_action, positioning_bias,
    commit_delay, reason_tags, outcome, outcome_tick
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class TacticalPlanner:
    """Full adaptive AI planner.

    Lifecycle:
        planner = TacticalPlanner(db, prediction_engine, ai_cfg, game_cfg, tier)
        planner.on_match_start(match_id, session_id, rng_seed)
        # each tick during SIMULATE phase:
        commitment = planner.decide(ai_state, sim_state, game_cfg)
        # on events:
        planner.on_player_commit(commitment, tick_id)
        planner.on_ai_commit_end(tick_id, damage_dealt, damage_taken)
        planner.on_match_end()
    """

    def __init__(
        self,
        db: Database,
        prediction_engine: PredictionEngine,
        ai_cfg: AIConfig,
        game_cfg: GameConfig,
        tier: AITier = AITier.T2_FULL_ADAPTIVE,
        behavior_model: "BehaviorModel | None" = None,
    ) -> None:
        self._db = db
        self._pe = prediction_engine
        self._ai_cfg = ai_cfg
        self._game_cfg = game_cfg
        self._tier = tier
        self._behavior_model = behavior_model

        self._memory = PlannerMemory(ai_cfg.planner_memory)
        # Session memory persists across matches — NOT reset on match start
        self._session_memory = SessionMemory(
            decay_factor=ai_cfg.session_adaptation.decay_factor,
            min_samples=ai_cfg.session_adaptation.min_session_samples,
        )
        self._rng = random.Random(42)

        # Per-match state
        self._match_id = ""
        self._session_id = ""

        # Current pending action
        self._pending: ResolverOutput | None = None
        self._pending_mode: TacticalIntent | None = None
        self._pending_tick: int = -1
        self._pending_context_tick: int = -1  # tick_id when context was captured

        # Delay countdown
        self._delay_remaining: int = 0

        # Outcome tracking
        self._last_ai_action: CombatCommitment | None = None
        self._last_ai_action_tick: int = -1
        self._last_ai_mode: TacticalIntent | None = None  # mode that produced the action
        self._ai_hp_at_action: int = 0
        self._opponent_hp_at_action: int = 0

        # Last prediction for accuracy tracking
        self._last_prediction_label: str | None = None
        self._last_prediction_confidence: float = 0.0

    @property
    def tier(self) -> AITier:
        return self._tier

    @property
    def memory(self) -> PlannerMemory:
        return self._memory

    @property
    def session_memory(self) -> SessionMemory:
        return self._session_memory

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def on_match_start(self, match_id: str, session_id: str,
                       rng_seed: int) -> None:
        self._match_id = match_id
        self._session_id = session_id
        self._rng = random.Random(rng_seed)
        self._memory.reset()
        self._pending = None
        self._pending_mode = None
        self._pending_tick = -1
        self._pending_context_tick = -1
        self._delay_remaining = 0
        self._last_ai_action = None
        self._last_ai_action_tick = -1
        self._last_ai_mode = None
        self._last_prediction_label = None
        self._last_prediction_confidence = 0.0

    def on_match_end(self) -> None:
        """Aggregate match outcomes into session memory before match reset."""
        if self._tier == AITier.T2_FULL_ADAPTIVE:
            mode_stats: dict[str, tuple[int, int]] = {}
            for outcome in self._memory.mode_outcome_log:
                name = outcome.mode.name
                s, t = mode_stats.get(name, (0, 0))
                mode_stats[name] = (s + (1 if outcome.success else 0), t + 1)
            self._session_memory.record_match_outcomes(mode_stats)

    # ------------------------------------------------------------------ #
    # Core decision                                                        #
    # ------------------------------------------------------------------ #

    def decide(
        self,
        ai_state: FighterState,
        sim: SimulationState,
        config: GameConfig,
    ) -> CombatCommitment | None:
        """Called each tick. Returns a commitment if one should fire now."""
        if not ai_state.is_alive:
            return None
        if ai_state.is_locked:
            return None

        # If we have a delayed pending action, count down
        if self._pending is not None and self._delay_remaining > 0:
            self._delay_remaining -= 1
            if self._delay_remaining > 0:
                return None

            # Check if context is still valid (no major change since plan)
            if self._is_context_stale(sim):
                self._pending = None
                self._pending_mode = None
                # Fall through to re-plan
            else:
                return self._execute_pending(ai_state, sim, config)

        # Plan a new action
        return self._plan_and_maybe_execute(ai_state, sim, config)

    def _plan_and_maybe_execute(
        self,
        ai_state: FighterState,
        sim: SimulationState,
        config: GameConfig,
    ) -> CombatCommitment | None:
        """Build context, select mode, resolve action, optionally execute."""
        ctx = self._build_context(sim, config)
        prediction = ctx.prediction

        # Store prediction for accuracy tracking
        if prediction.has_prediction:
            self._last_prediction_label = prediction.top_label
            self._last_prediction_confidence = prediction.top_label_confidence

        # Resolve archetype if a behavior model is available
        archetype = (
            self._behavior_model.current_archetype()
            if self._behavior_model is not None
            else None
        )

        # Select tactical mode (session context passed when available)
        mode = select_mode(
            ctx, self._memory,
            self._ai_cfg.strategy,
            self._ai_cfg.planner_memory,
            self._rng,
            session_memory=self._session_memory,
            archetype=archetype,
            archetype_table=self._ai_cfg.archetype_mode_alignment,
        )

        # Record mode in memory
        if self._tier == AITier.T2_FULL_ADAPTIVE:
            self._memory.record_mode(mode)

            # Update exploit target
            if (mode == TacticalIntent.EXPLOIT_PATTERN
                    and prediction.top_commitment is not None
                    and prediction.has_prediction):
                if (self._memory.current_exploit_target is None
                        or self._memory.current_exploit_target.predicted_commitment
                        != prediction.top_commitment.name):
                    self._memory.set_exploit_target(
                        prediction.top_commitment.name, sim.tick_id)

        # Resolve to action
        output = resolve(mode, ctx, self._memory, self._ai_cfg.action_resolver)

        if output.commit_delay > 0:
            self._pending = output
            self._pending_mode = mode
            self._pending_tick = sim.tick_id
            self._pending_context_tick = sim.tick_id
            self._delay_remaining = output.commit_delay
            return None

        return self._fire_action(
            ai_state, sim, config, output, mode)

    def _execute_pending(
        self,
        ai_state: FighterState,
        sim: SimulationState,
        config: GameConfig,
    ) -> CombatCommitment | None:
        """Execute a previously planned pending action."""
        output = self._pending
        mode = self._pending_mode
        self._pending = None
        self._pending_mode = None
        return self._fire_action(ai_state, sim, config, output, mode)

    def _fire_action(
        self,
        ai_state: FighterState,
        sim: SimulationState,
        config: GameConfig,
        output: ResolverOutput,
        mode: TacticalIntent | None,
    ) -> CombatCommitment | None:
        """Attempt to enter the resolved commitment. Log the decision."""
        commitment = output.commitment

        if not can_commit(ai_state, commitment, config):
            # Fallback: try a safe alternative
            commitment = self._fallback_commitment(ai_state, sim, config)
            if commitment is None:
                return None

        if attempt_commitment(ai_state, commitment, config):
            # Track for outcome resolution
            self._last_ai_action = commitment
            self._last_ai_action_tick = sim.tick_id
            self._last_ai_mode = mode  # exact mode that produced this action
            self._ai_hp_at_action = sim.ai.hp
            self._opponent_hp_at_action = sim.player.hp

            # Log decision
            self._log_decision(sim, output, mode, commitment)
            return commitment

        return None

    # ------------------------------------------------------------------ #
    # Outcome resolution                                                   #
    # ------------------------------------------------------------------ #

    def on_player_commit(self, commitment: CombatCommitment, tick_id: int) -> None:
        """Called when player starts a new commitment. Update prediction accuracy."""
        if self._tier != AITier.T2_FULL_ADAPTIVE:
            return
        if self._last_prediction_label is not None:
            outcome = PredictionOutcome(
                predicted_label=self._last_prediction_label,
                actual_label=commitment.name,
                confidence=self._last_prediction_confidence,
                tick_id=tick_id,
            )
            self._memory.record_prediction(outcome)
            self._last_prediction_label = None

    def on_ai_commit_end(
        self,
        tick_id: int,
        current_ai_hp: int,
        current_opponent_hp: int,
    ) -> None:
        """Called when AI finishes a commitment. Assess outcome."""
        if self._tier != AITier.T2_FULL_ADAPTIVE:
            return
        if self._last_ai_action is None:
            return

        damage_dealt = max(0, self._opponent_hp_at_action - current_opponent_hp)
        damage_taken = max(0, self._ai_hp_at_action - current_ai_hp)

        # Simple outcome: success if dealt more damage than taken
        success = damage_dealt > damage_taken

        mode = self._pending_mode_for_outcome()
        outcome = ModeOutcome(
            mode=mode,
            tick_id=tick_id,
            success=success,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
        )
        self._memory.record_outcome(outcome)

        # Persist outcome to the ai_decisions row for this action
        self._db.execute_safe(
            "UPDATE ai_decisions SET outcome=?, outcome_tick=? "
            "WHERE match_id=? AND tick_id=?;",
            ("success" if success else "failure",
             tick_id, self._match_id, self._last_ai_action_tick),
        )

        self._last_ai_action = None
        self._last_ai_mode = None

    def _pending_mode_for_outcome(self) -> TacticalIntent:
        """Get the mode that produced the last action."""
        if self._last_ai_mode is not None:
            return self._last_ai_mode
        return TacticalIntent.NEUTRAL_SPACING

    # ------------------------------------------------------------------ #
    # Context building                                                     #
    # ------------------------------------------------------------------ #

    def _build_context(
        self, sim: SimulationState, config: GameConfig,
    ) -> AIContext:
        prediction = self._pe.predict()
        max_hp = config.fighter.max_hp
        max_stamina = config.fighter.max_stamina
        scale = config.simulation.sub_pixel_scale

        spacing = classify_spacing(
            sim.distance_sub(),
            config.spacing.close_max,
            config.spacing.mid_max,
            scale,
        )

        return AIContext(
            tick_id=sim.tick_id,
            player_hp_frac=sim.player.hp / max_hp if max_hp > 0 else 0.0,
            player_stamina_frac=sim.player.stamina / max_stamina if max_stamina > 0 else 0.0,
            player_fsm=sim.player.fsm_state,
            player_commitment=sim.player.active_commitment,
            ai_hp_frac=sim.ai.hp / max_hp if max_hp > 0 else 0.0,
            ai_stamina_frac=sim.ai.stamina / max_stamina if max_stamina > 0 else 0.0,
            ai_fsm=sim.ai.fsm_state,
            ai_commitment=sim.ai.active_commitment,
            ai_facing=sim.ai.facing,
            spacing=spacing,
            distance_sub=sim.distance_sub(),
            prediction=prediction,
        )

    def _is_context_stale(self, sim: SimulationState) -> bool:
        """Check if the game state has changed enough to invalidate pending action."""
        if self._pending_context_tick < 0:
            return True
        elapsed = sim.tick_id - self._pending_context_tick
        # Stale if too old or player has moved to a different FSM state
        return elapsed > self._ai_cfg.action_resolver.max_commit_delay_ticks

    def _fallback_commitment(
        self,
        ai_state: FighterState,
        sim: SimulationState,
        config: GameConfig,
    ) -> CombatCommitment | None:
        """When primary action fails, find a safe alternative."""
        # Try light attack, then move toward player, then move away
        fallbacks = [
            CombatCommitment.LIGHT_ATTACK,
            CombatCommitment.MOVE_RIGHT if ai_state.facing > 0 else CombatCommitment.MOVE_LEFT,
            CombatCommitment.MOVE_LEFT if ai_state.facing > 0 else CombatCommitment.MOVE_RIGHT,
        ]
        for fb in fallbacks:
            if can_commit(ai_state, fb, config):
                return fb
        return None

    # ------------------------------------------------------------------ #
    # Decision logging                                                     #
    # ------------------------------------------------------------------ #

    def _log_decision(
        self,
        sim: SimulationState,
        output: ResolverOutput,
        mode: TacticalIntent | None,
        actual_commitment: CombatCommitment,
    ) -> None:
        """Insert into ai_decisions table. Non-fatal on failure."""
        pred = self._pe.predict()
        pred_top = pred.top_label if pred.has_prediction else None
        pred_conf = pred.top_label_confidence if pred.has_prediction else None
        pred_probs = json.dumps(pred.distribution) if pred.distribution else None

        self._db.execute_safe(
            _AI_DECISION_INSERT,
            (
                self._session_id,
                self._match_id,
                sim.tick_id,
                pred_top,
                pred_conf,
                pred_probs,
                mode.name if mode else "UNKNOWN",
                actual_commitment.name,
                output.positioning_bias,
                output.commit_delay,
                json.dumps(output.reason_tags),
                None,  # outcome filled later
                None,  # outcome_tick filled later
            ),
        )
