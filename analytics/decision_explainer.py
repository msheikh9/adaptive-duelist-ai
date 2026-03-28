"""Produces human-readable explanations for individual AI decisions.

Given one ai_decisions row, builds a DecisionExplanation with a concise
natural-language summary.  Gracefully handles null prediction fields
so T0/T1 rows never crash.
"""

from __future__ import annotations

import json
import sqlite3

from analytics.explanation_types import DecisionExplanation


def explain_decision(row: sqlite3.Row) -> DecisionExplanation:
    """Build a DecisionExplanation from one ai_decisions row."""
    predicted_top = row["predicted_top"]
    pred_confidence = row["pred_confidence"]
    pred_probs_raw = row["pred_probs"]
    tactical_mode = row["tactical_mode"]
    ai_action = row["ai_action"]
    positioning_bias = row["positioning_bias"]
    commit_delay = row["commit_delay"]
    reason_tags_raw = row["reason_tags"]
    outcome = row["outcome"]
    outcome_tick = row["outcome_tick"]
    tick_id = row["tick_id"]

    pred_probs = _safe_json(pred_probs_raw)
    reason_tags: list[str] = _safe_json(reason_tags_raw) or []

    explanation = _build_explanation(
        predicted_top, pred_confidence, tactical_mode,
        ai_action, reason_tags, outcome,
    )

    return DecisionExplanation(
        tick_id=tick_id,
        predicted_top=predicted_top,
        pred_confidence=pred_confidence,
        pred_probs=pred_probs,
        tactical_mode=tactical_mode,
        ai_action=ai_action,
        positioning_bias=positioning_bias,
        commit_delay=commit_delay,
        reason_tags=reason_tags,
        outcome=outcome,
        outcome_tick=outcome_tick,
        explanation=explanation,
    )


def _build_explanation(
    predicted_top: str | None,
    pred_confidence: float | None,
    tactical_mode: str,
    ai_action: str,
    reason_tags: list[str],
    outcome: str | None,
) -> str:
    """Compose a concise human-readable explanation string."""
    parts: list[str] = []

    # Prediction context
    if predicted_top is not None and pred_confidence is not None:
        parts.append(
            f"Predicted player would {predicted_top} "
            f"({pred_confidence:.0%} confidence)."
        )
    else:
        parts.append("No prediction available.")

    # Tactical mode and action
    mode_label = tactical_mode.replace("_", " ").lower()
    action_label = ai_action.replace("_", " ").lower()
    parts.append(f"Selected {mode_label} mode \u2192 {action_label}.")

    # Reason tags
    if reason_tags:
        parts.append(f"Reasons: {', '.join(reason_tags)}.")

    # Outcome
    if outcome is not None:
        parts.append(f"Outcome: {outcome}.")
    else:
        parts.append("Outcome: unavailable.")

    return " ".join(parts)


def _safe_json(raw: str | None):
    """Parse JSON or return None."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
