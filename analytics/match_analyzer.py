"""Build a MatchExplanation from DB contents for one match.

Reads matches, ai_decisions, and semantic_events.  Handles missing data
gracefully — T0 matches produce a valid reduced report with clear
"data unavailable" sections instead of errors.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analytics.decision_explainer import explain_decision
from analytics.explanation_types import (
    DecisionExplanation,
    MatchExplanation,
    PredictionAuditRow,
)

if TYPE_CHECKING:
    from data.db import Database


def analyze_match(
    db: Database,
    match_id: str,
    max_notable: int = 10,
) -> MatchExplanation:
    """Build a full MatchExplanation for one match."""
    # Match metadata
    match_row = db.fetchone(
        "SELECT * FROM matches WHERE match_id = ?;", (match_id,),
    )
    winner = match_row["winner"] if match_row else None
    total_ticks = match_row["total_ticks"] if match_row else None
    player_hp_final = match_row["player_hp_final"] if match_row else None
    ai_hp_final = match_row["ai_hp_final"] if match_row else None

    # AI decisions
    decisions = db.fetchall(
        "SELECT * FROM ai_decisions WHERE match_id = ? ORDER BY tick_id;",
        (match_id,),
    )

    # Player commitment events
    player_commits = db.fetchall(
        "SELECT * FROM semantic_events "
        "WHERE match_id = ? AND actor = 'PLAYER' "
        "AND event_type = 'COMMITMENT_START' "
        "ORDER BY tick_id;",
        (match_id,),
    )

    tier_label = _detect_tier(decisions)
    mode_usage, mode_success = _compute_mode_stats(decisions)
    audit = _build_prediction_audit(decisions, player_commits)
    prediction_accuracy = _compute_accuracy(audit)
    top2_accuracy = _compute_top2_accuracy(decisions, player_commits)
    player_top_commits = _count_player_commitments(player_commits)
    exploit_targets = _extract_exploit_targets(decisions)
    notable = _select_notable_decisions(decisions, max_notable)

    return MatchExplanation(
        match_id=match_id,
        winner=winner,
        total_ticks=total_ticks,
        player_hp_final=player_hp_final,
        ai_hp_final=ai_hp_final,
        decision_count=len(decisions),
        mode_usage=mode_usage,
        mode_success=mode_success,
        prediction_accuracy=prediction_accuracy,
        top2_accuracy=top2_accuracy,
        player_top_commitments=player_top_commits,
        exploit_targets=exploit_targets,
        notable_decisions=notable,
        prediction_audit=audit,
        tier_label=tier_label,
    )


# ------------------------------------------------------------------ #
# Tier detection                                                       #
# ------------------------------------------------------------------ #

def _detect_tier(decisions: list) -> str:
    if not decisions:
        return "T0 (baseline)"
    has_outcome = any(row["outcome"] is not None for row in decisions)
    if has_outcome:
        return "T2 (full adaptive)"
    return "T1 (markov-only)"


# ------------------------------------------------------------------ #
# Mode stats                                                           #
# ------------------------------------------------------------------ #

def _compute_mode_stats(
    decisions: list,
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    usage: dict[str, int] = {}
    success_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}

    for row in decisions:
        mode = row["tactical_mode"]
        outcome = row["outcome"]
        usage[mode] = usage.get(mode, 0) + 1
        if outcome is not None:
            outcome_counts[mode] = outcome_counts.get(mode, 0) + 1
            if outcome == "success":
                success_counts[mode] = success_counts.get(mode, 0) + 1

    mode_success: dict[str, tuple[int, int]] = {}
    for mode in usage:
        s = success_counts.get(mode, 0)
        t = outcome_counts.get(mode, 0)
        mode_success[mode] = (s, t)

    return usage, mode_success


# ------------------------------------------------------------------ #
# Prediction audit                                                     #
# ------------------------------------------------------------------ #

def _build_prediction_audit(
    decisions: list,
    player_commits: list,
) -> list[PredictionAuditRow]:
    """Match each AI prediction to the next player commitment."""
    audit: list[PredictionAuditRow] = []
    commit_idx = 0

    for row in decisions:
        predicted_top = row["predicted_top"]
        confidence = row["pred_confidence"]
        tick = row["tick_id"]

        # Advance to the first player commit at or after this tick
        while (commit_idx < len(player_commits)
               and player_commits[commit_idx]["tick_id"] < tick):
            commit_idx += 1

        actual = None
        correct = None
        if commit_idx < len(player_commits):
            actual = player_commits[commit_idx]["commitment"]
            if predicted_top is not None and actual is not None:
                correct = predicted_top == actual

        audit.append(PredictionAuditRow(
            tick_id=tick,
            predicted_top=predicted_top,
            actual_commitment=actual,
            correct=correct,
            confidence=confidence,
        ))

    return audit


def _compute_accuracy(audit: list[PredictionAuditRow]) -> float | None:
    evaluated = [a for a in audit if a.correct is not None]
    if not evaluated:
        return None
    return sum(1 for a in evaluated if a.correct) / len(evaluated)


def _compute_top2_accuracy(
    decisions: list, player_commits: list,
) -> float | None:
    """Was the actual commitment among the top-2 predicted labels?"""
    commit_idx = 0
    correct = 0
    total = 0

    for row in decisions:
        pred_probs_raw = row["pred_probs"]
        tick = row["tick_id"]

        if not pred_probs_raw:
            continue
        try:
            probs = json.loads(pred_probs_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not probs:
            continue

        while (commit_idx < len(player_commits)
               and player_commits[commit_idx]["tick_id"] < tick):
            commit_idx += 1
        if commit_idx >= len(player_commits):
            continue

        actual = player_commits[commit_idx]["commitment"]
        if actual is None:
            continue

        sorted_labels = sorted(probs.keys(), key=lambda k: -probs[k])[:2]
        total += 1
        if actual in sorted_labels:
            correct += 1

    return correct / total if total > 0 else None


# ------------------------------------------------------------------ #
# Player commitments                                                   #
# ------------------------------------------------------------------ #

def _count_player_commitments(
    player_commits: list,
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in player_commits:
        c = row["commitment"]
        if c:
            counts[c] = counts.get(c, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:5]


# ------------------------------------------------------------------ #
# Exploit targets                                                      #
# ------------------------------------------------------------------ #

def _extract_exploit_targets(decisions: list) -> list[str]:
    """Unique exploit targets from EXPLOIT_PATTERN decisions."""
    targets: list[str] = []
    seen: set[str] = set()
    for row in decisions:
        if row["tactical_mode"] == "EXPLOIT_PATTERN" and row["predicted_top"]:
            target = row["predicted_top"]
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return targets


# ------------------------------------------------------------------ #
# Notable decisions                                                    #
# ------------------------------------------------------------------ #

def _select_notable_decisions(
    decisions: list, max_count: int,
) -> list[DecisionExplanation]:
    """Pick the most notable decisions for the report."""
    if not decisions:
        return []

    scored: list[tuple[float, int]] = []
    for i, row in enumerate(decisions):
        score = 0.0
        conf = row["pred_confidence"]
        outcome = row["outcome"]

        if conf is not None:
            score += conf * 2.0
        if outcome == "success":
            score += 1.5
        elif outcome == "failure":
            score += 1.0
        if i == 0:
            score += 3.0
        if i == len(decisions) - 1:
            score += 2.0
        if row["tactical_mode"] == "EXPLOIT_PATTERN":
            score += 0.5

        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    selected_indices = sorted(idx for _, idx in scored[:max_count])
    return [explain_decision(decisions[i]) for i in selected_indices]
