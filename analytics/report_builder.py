"""Compose individual analytics into a unified ExplainabilityReport.

Supports both single-match and multi-match aggregate reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from analytics.explanation_types import ExplainabilityReport
from analytics.match_analyzer import analyze_match
from analytics.pattern_miner import mine_patterns
from analytics.planner_metrics import compute_planner_metrics

if TYPE_CHECKING:
    from data.db import Database


def build_match_report(
    db: Database,
    match_id: str,
) -> ExplainabilityReport:
    """Build a complete report for a single match."""
    match_exp = analyze_match(db, match_id)
    pattern = mine_patterns(db)
    metrics = compute_planner_metrics(db, [match_id])

    return ExplainabilityReport(
        match_explanation=match_exp,
        pattern_summary=pattern,
        planner_metrics=metrics,
        generated_at=datetime.now(timezone.utc).isoformat(),
        matches_included=[match_id],
    )


def build_aggregate_report(
    db: Database,
    match_ids: list[str],
) -> ExplainabilityReport:
    """Build an aggregate report over multiple matches.

    Uses the most recent match for MatchExplanation;
    aggregates all matches for planner metrics.
    """
    latest = match_ids[-1] if match_ids else None
    match_exp = analyze_match(db, latest) if latest else None
    pattern = mine_patterns(db)
    metrics = compute_planner_metrics(db, match_ids)

    return ExplainabilityReport(
        match_explanation=match_exp,
        pattern_summary=pattern,
        planner_metrics=metrics,
        generated_at=datetime.now(timezone.utc).isoformat(),
        matches_included=list(match_ids),
    )
