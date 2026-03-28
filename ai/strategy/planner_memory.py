"""Planner memory: persistent tactical state across decisions within a match.

Tracks mode outcomes, prediction accuracy, exploit staleness, exploration
budget, pattern shift detection, and consecutive mode penalties. All values
are reset on match start. Cross-match learning happens through the
PlayerProfile, not here.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ai.strategy.tactics import TacticalIntent
from config.config_loader import PlannerMemoryConfig


@dataclass
class ModeOutcome:
    """Record of a single tactical mode execution and its result."""
    mode: TacticalIntent
    tick_id: int
    success: bool  # did the AI's action achieve its goal?
    damage_dealt: int = 0
    damage_taken: int = 0


@dataclass
class PredictionOutcome:
    """Record of a single prediction vs actual."""
    predicted_label: str
    actual_label: str
    confidence: float
    tick_id: int

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.actual_label


@dataclass
class ExploitTarget:
    """Current pattern being exploited."""
    predicted_commitment: str
    first_seen_tick: int
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0


class RollingRate:
    """Computes accuracy over a fixed-size window of bool observations."""

    def __init__(self, capacity: int) -> None:
        self._window: deque[bool] = deque(maxlen=capacity)

    def push(self, correct: bool) -> None:
        self._window.append(correct)

    @property
    def rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def count(self) -> int:
        return len(self._window)

    def clear(self) -> None:
        self._window.clear()


class PlannerMemory:
    """Tactical memory for one match. Reset on match start.

    All capacities and thresholds come from PlannerMemoryConfig.
    """

    def __init__(self, cfg: PlannerMemoryConfig) -> None:
        self._cfg = cfg

        # Mode history
        self.recent_modes: deque[TacticalIntent] = deque(maxlen=cfg.recent_modes_capacity)
        self.mode_outcome_log: deque[ModeOutcome] = deque(maxlen=cfg.mode_outcome_capacity)

        # Per-mode success tracking
        self._mode_attempts: dict[TacticalIntent, int] = {}
        self._mode_successes: dict[TacticalIntent, int] = {}

        # Prediction accuracy
        self.recent_predictions: deque[PredictionOutcome] = deque(
            maxlen=cfg.recent_predictions_capacity)
        self._accuracy_short = RollingRate(10)
        self._accuracy_medium = RollingRate(30)

        # Exploit state
        self.current_exploit_target: ExploitTarget | None = None
        self._exploit_set_tick: int = 0

        # Exploration budget (starts full)
        self.exploration_budget: float = 1.0

        # Consecutive mode tracking
        self._consecutive_mode: TacticalIntent | None = None
        self._consecutive_count: int = 0

        # Pattern shift detection
        self.shift_detected: bool = False
        self.shift_detected_at_tick: int = 0
        self._shift_probe_remaining: int = 0

    # ------------------------------------------------------------------ #
    # Match lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all per-match state."""
        self.recent_modes.clear()
        self.mode_outcome_log.clear()
        self._mode_attempts.clear()
        self._mode_successes.clear()
        self.recent_predictions.clear()
        self._accuracy_short.clear()
        self._accuracy_medium.clear()
        self.current_exploit_target = None
        self._exploit_set_tick = 0
        self.exploration_budget = 1.0
        self._consecutive_mode = None
        self._consecutive_count = 0
        self.shift_detected = False
        self.shift_detected_at_tick = 0
        self._shift_probe_remaining = 0

    # ------------------------------------------------------------------ #
    # Mode tracking                                                        #
    # ------------------------------------------------------------------ #

    def record_mode(self, mode: TacticalIntent) -> None:
        """Record a mode selection. Updates consecutive counter."""
        self.recent_modes.append(mode)

        if mode == self._consecutive_mode:
            self._consecutive_count += 1
        else:
            self._consecutive_mode = mode
            self._consecutive_count = 1

        # Drain exploration budget when not probing
        if mode != TacticalIntent.PROBE_BEHAVIOR:
            self.exploration_budget = max(
                self._cfg.exploration_budget_floor,
                self.exploration_budget - self._cfg.exploration_budget_drain_rate,
            )
        else:
            # Regenerate budget when probing
            self.exploration_budget = min(
                1.0,
                self.exploration_budget + self._cfg.exploration_budget_regen_rate,
            )
            if self._shift_probe_remaining > 0:
                self._shift_probe_remaining -= 1
                if self._shift_probe_remaining <= 0:
                    self.shift_detected = False

    def record_outcome(self, outcome: ModeOutcome) -> None:
        """Record the result of a mode execution."""
        self.mode_outcome_log.append(outcome)
        mode = outcome.mode
        self._mode_attempts[mode] = self._mode_attempts.get(mode, 0) + 1
        if outcome.success:
            self._mode_successes[mode] = self._mode_successes.get(mode, 0) + 1

        # Update exploit target if relevant
        if (mode == TacticalIntent.EXPLOIT_PATTERN
                and self.current_exploit_target is not None):
            self.current_exploit_target.attempts += 1
            if outcome.success:
                self.current_exploit_target.successes += 1

    def per_mode_success_rate(self, mode: TacticalIntent) -> float:
        """Lifetime success rate for a given mode this match."""
        attempts = self._mode_attempts.get(mode, 0)
        if attempts == 0:
            return 0.5  # neutral prior
        return self._mode_successes.get(mode, 0) / attempts

    @property
    def consecutive_same_mode(self) -> int:
        return self._consecutive_count

    # ------------------------------------------------------------------ #
    # Prediction accuracy                                                  #
    # ------------------------------------------------------------------ #

    def record_prediction(self, outcome: PredictionOutcome) -> None:
        """Record a prediction result and update rolling accuracy."""
        self.recent_predictions.append(outcome)
        self._accuracy_short.push(outcome.correct)
        self._accuracy_medium.push(outcome.correct)
        self._check_shift(outcome.tick_id)

    @property
    def prediction_accuracy_short(self) -> float:
        return self._accuracy_short.rate

    @property
    def prediction_accuracy_medium(self) -> float:
        return self._accuracy_medium.rate

    @property
    def accuracy_trend(self) -> float:
        """Positive = improving, negative = degrading."""
        if self._accuracy_short.count < 3 or self._accuracy_medium.count < 5:
            return 0.0
        return self._accuracy_short.rate - self._accuracy_medium.rate

    # ------------------------------------------------------------------ #
    # Exploit staleness                                                    #
    # ------------------------------------------------------------------ #

    def set_exploit_target(self, predicted_commitment: str, tick_id: int) -> None:
        """Set a new exploit target commitment."""
        self.current_exploit_target = ExploitTarget(
            predicted_commitment=predicted_commitment,
            first_seen_tick=tick_id,
        )
        self._exploit_set_tick = tick_id

    def exploit_staleness(self, current_tick: int) -> float:
        """How stale the current exploit target is, as a 0-1 fraction."""
        if self.current_exploit_target is None:
            return 0.0
        age = current_tick - self._exploit_set_tick
        threshold = self._cfg.exploit_staleness_threshold_ticks
        return min(age / threshold, 1.0) if threshold > 0 else 0.0

    def clear_exploit_target(self) -> None:
        self.current_exploit_target = None
        self._exploit_set_tick = 0

    # ------------------------------------------------------------------ #
    # Shift detection                                                      #
    # ------------------------------------------------------------------ #

    def _check_shift(self, tick_id: int) -> None:
        """Detect if prediction accuracy dropped suddenly.

        Compares short-term accuracy against medium-term accuracy.
        A significant gap (short << medium) indicates a pattern shift.
        """
        if self._accuracy_short.count < 5 or self._accuracy_medium.count < 5:
            return
        short = self._accuracy_short.rate
        medium = self._accuracy_medium.rate
        drop = medium - short
        if drop >= self._cfg.shift_detection_confidence_drop and not self.shift_detected:
            self.shift_detected = True
            self.shift_detected_at_tick = tick_id
            self._shift_probe_remaining = self._cfg.shift_probe_duration_decisions
            # Clear exploit target on shift
            self.clear_exploit_target()

    @property
    def in_shift_probe(self) -> bool:
        return self.shift_detected and self._shift_probe_remaining > 0
