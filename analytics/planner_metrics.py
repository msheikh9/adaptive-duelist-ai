"""Summarize planner quality metrics from ai_decisions.

Computes mode distribution, per-mode outcome rates, delay averages,
action distribution, and prediction availability.  Tolerates T0/T1/T2
data — returns zeroed metrics when no decisions exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analytics.explanation_types import PlannerMetricsSummary

if TYPE_CHECKING:
    from data.db import Database


def compute_planner_metrics(
    db: Database,
    match_ids: list[str],
) -> PlannerMetricsSummary:
    """Build PlannerMetricsSummary from ai_decisions across given matches."""
    if not match_ids:
        return _empty_metrics()

    placeholders = ",".join("?" for _ in match_ids)
    rows = db.fetchall(
        f"SELECT * FROM ai_decisions WHERE match_id IN ({placeholders}) "
        "ORDER BY tick_id;",
        tuple(match_ids),
    )

    if not rows:
        return _empty_metrics()

    total = len(rows)
    mode_dist: dict[str, int] = {}
    mode_successes: dict[str, int] = {}
    mode_outcome_total: dict[str, int] = {}
    action_by_mode: dict[str, dict[str, int]] = {}
    delay_sum = 0
    delay_count = 0
    hold_count = 0
    pred_available = 0

    for row in rows:
        mode = row["tactical_mode"]
        action = row["ai_action"]
        outcome = row["outcome"]
        delay = row["commit_delay"]
        predicted_top = row["predicted_top"]

        mode_dist[mode] = mode_dist.get(mode, 0) + 1

        if mode not in action_by_mode:
            action_by_mode[mode] = {}
        action_by_mode[mode][action] = action_by_mode[mode].get(action, 0) + 1

        if outcome is not None:
            mode_outcome_total[mode] = mode_outcome_total.get(mode, 0) + 1
            if outcome == "success":
                mode_successes[mode] = mode_successes.get(mode, 0) + 1

        if delay is not None:
            delay_sum += delay
            delay_count += 1

        if predicted_top is not None:
            pred_available += 1
            if predicted_top == "HOLD":
                hold_count += 1

    mode_outcome_rates: dict[str, float] = {}
    for mode in mode_dist:
        t = mode_outcome_total.get(mode, 0)
        mode_outcome_rates[mode] = (
            mode_successes.get(mode, 0) / t if t > 0 else 0.0
        )

    return PlannerMetricsSummary(
        total_decisions=total,
        mode_distribution=mode_dist,
        mode_outcome_rates=mode_outcome_rates,
        avg_commit_delay=delay_sum / delay_count if delay_count > 0 else 0.0,
        action_by_mode=action_by_mode,
        hold_top_label_pct=hold_count / total if total > 0 else 0.0,
        prediction_available_pct=pred_available / total if total > 0 else 0.0,
    )


def _empty_metrics() -> PlannerMetricsSummary:
    return PlannerMetricsSummary(
        total_decisions=0,
        mode_distribution={},
        mode_outcome_rates={},
        avg_commit_delay=0.0,
        action_by_mode={},
        hold_top_label_pct=0.0,
        prediction_available_pct=0.0,
    )
