"""Dataclasses for the explainability / analytics layer.

These types form the API surface between analytics modules and consumers
(CLI scripts, future UI, JSON export).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionExplanation:
    """Human-readable explanation of one AI decision."""

    tick_id: int
    predicted_top: str | None
    pred_confidence: float | None
    pred_probs: dict[str, float] | None
    tactical_mode: str
    ai_action: str
    positioning_bias: float | None
    commit_delay: int | None
    reason_tags: list[str]
    outcome: str | None
    outcome_tick: int | None
    explanation: str  # concise human-readable summary


@dataclass
class PredictionAuditRow:
    """Audits one prediction against the actual player action."""

    tick_id: int
    predicted_top: str | None
    actual_commitment: str | None
    correct: bool | None  # None when not enough data to evaluate
    confidence: float | None


@dataclass
class MatchExplanation:
    """Full explanation of one match from the AI's perspective."""

    match_id: str
    winner: str | None
    total_ticks: int | None
    player_hp_final: int | None
    ai_hp_final: int | None
    decision_count: int
    mode_usage: dict[str, int]
    mode_success: dict[str, tuple[int, int]]  # mode → (successes, total_with_outcome)
    prediction_accuracy: float | None
    top2_accuracy: float | None
    player_top_commitments: list[tuple[str, int]]
    exploit_targets: list[str]
    notable_decisions: list[DecisionExplanation]
    prediction_audit: list[PredictionAuditRow]
    tier_label: str


@dataclass
class TacticalPatternSummary:
    """Summary of player tactical patterns."""

    player_id: str
    total_commitments: int
    top_commitments: list[tuple[str, int]]
    top_bigrams: list[tuple[str, int]]
    top_trigrams: list[tuple[str, int]]
    spacing_tendencies: dict[str, int]
    aggression_index: float
    dodge_frequency: float
    initiative_rate: float
    movement_bias: float
    exploitable_habits: list[str]


@dataclass
class PlannerMetricsSummary:
    """Summary of planner quality metrics from ai_decisions."""

    total_decisions: int
    mode_distribution: dict[str, int]
    mode_outcome_rates: dict[str, float]  # mode → success rate
    avg_commit_delay: float
    action_by_mode: dict[str, dict[str, int]]  # mode → {action: count}
    hold_top_label_pct: float
    prediction_available_pct: float


@dataclass
class ExplainabilityReport:
    """Top-level report combining match, pattern, and planner analysis."""

    match_explanation: MatchExplanation | None
    pattern_summary: TacticalPatternSummary | None
    planner_metrics: PlannerMetricsSummary | None
    generated_at: str  # ISO timestamp
    matches_included: list[str] = field(default_factory=list)
