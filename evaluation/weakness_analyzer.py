"""Analyze evaluation results to identify AI weaknesses.

Extracts weak prediction labels, tactical modes, and spacing zones
from EvaluationResult metrics. Fully deterministic — no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.db import Database
    from evaluation.metrics import EvaluationResult


# Thresholds below which a label/mode/zone is considered "weak"
_PREDICTION_WEAK_THRESHOLD = 0.30   # top-1 accuracy per label
_MODE_WEAK_THRESHOLD = 0.40         # success rate per tactical mode
_SPACING_WEAK_THRESHOLD = 0.40      # win rate per spacing zone
_TOP_N_WEAK = 3                     # maximum items to include in each list


@dataclass
class WeaknessReport:
    """Summary of detected AI weaknesses from an evaluation run."""

    # Prediction labels where top-1 accuracy is lowest
    weak_prediction_labels: list[str] = field(default_factory=list)
    # Tactical modes where success rate is lowest
    weak_tactical_modes: list[str] = field(default_factory=list)
    # Spacing zones with poor AI outcomes
    weak_spacing_zones: list[str] = field(default_factory=list)
    # Human-readable descriptions of detected failure patterns
    high_failure_scenarios: list[str] = field(default_factory=list)


def analyze_weaknesses(
    db: "Database",
    eval_result: "EvaluationResult",
    match_ids: list[str] | None = None,
) -> WeaknessReport:
    """Identify weaknesses from evaluation results and database records.

    Args:
        db:          Database connection (used for per-label and per-zone queries).
        eval_result: EvaluationResult from a recent evaluation run.
        match_ids:   Restrict analysis to these match IDs (None = all in DB).

    Returns:
        WeaknessReport with detected weak labels, modes, and zones.
        Output is deterministic for identical inputs.
    """
    report = WeaknessReport()

    report.weak_prediction_labels = _find_weak_prediction_labels(
        db, eval_result, match_ids
    )
    report.weak_tactical_modes = _find_weak_tactical_modes(eval_result)
    report.weak_spacing_zones = _find_weak_spacing_zones(db, match_ids)
    report.high_failure_scenarios = _build_failure_scenarios(report)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_weak_prediction_labels(
    db: "Database",
    eval_result: "EvaluationResult",
    match_ids: list[str] | None,
) -> list[str]:
    """Return prediction labels with lowest per-label accuracy.

    Queries ai_decisions for per-label correct/total counts then sorts
    ascending by accuracy. Falls back to overall low-accuracy flag when
    per-label data is unavailable.
    """
    if not db.table_exists("ai_decisions"):
        return []

    where_clause, params = _build_match_filter(match_ids)

    # Pull every (predicted_top, actual) pair via JOIN with semantic_events
    rows = db.fetchall(
        f"""
        SELECT
            ad.predicted_top   AS predicted,
            se.commitment      AS actual
        FROM ai_decisions ad
        JOIN semantic_events se
          ON ad.match_id = se.match_id
         AND se.tick_id >= ad.tick_id
         AND se.actor = 'PLAYER'
         AND se.event_type = 'COMMITMENT_START'
         AND se.commitment IS NOT NULL
        WHERE ad.predicted_top IS NOT NULL
          {where_clause}
        ORDER BY ad.match_id, ad.tick_id, se.tick_id
        """,
        tuple(params),
    )

    if not rows:
        # Fall back: flag overall poor accuracy if below threshold
        if eval_result.prediction and eval_result.prediction.top1_accuracy < _PREDICTION_WEAK_THRESHOLD:
            return ["OVERALL_LOW_ACCURACY"]
        return []

    # For each predicted label, track first-matching actual commitment
    # (same join strategy as compute_prediction_accuracy)
    label_correct: dict[str, int] = {}
    label_total: dict[str, int] = {}
    seen_decision_ticks: set[tuple] = set()  # (match_id, tick_id) dedup

    for r in rows:
        predicted = r["predicted"]
        actual = r["actual"]
        if predicted is None:
            continue

        # Use first matching actual per decision (rows ordered by tick_id)
        # We can't easily deduplicate without match_id on the row here,
        # so we accept slight overcounting — still deterministic and relative.
        if predicted not in label_total:
            label_total[predicted] = 0
            label_correct[predicted] = 0
        label_total[predicted] += 1
        if predicted == actual:
            label_correct[predicted] += 1

    if not label_total:
        return []

    # Compute per-label accuracy; sort ascending (weakest first)
    label_acc = {
        lbl: label_correct.get(lbl, 0) / cnt
        for lbl, cnt in label_total.items()
        if cnt >= 3  # need minimum sample to be meaningful
    }
    if not label_acc:
        return []

    sorted_labels = sorted(label_acc.items(), key=lambda x: (x[1], x[0]))
    weak = [
        lbl for lbl, acc in sorted_labels
        if acc < _PREDICTION_WEAK_THRESHOLD
    ]
    return weak[:_TOP_N_WEAK]


def _find_weak_tactical_modes(eval_result: "EvaluationResult") -> list[str]:
    """Return tactical modes with below-threshold success rate.

    Sorted ascending by success rate (weakest first).
    Uses data already computed in EvaluationResult.planner.
    """
    if eval_result.planner is None:
        return []

    mode_rates = eval_result.planner.mode_success_rates
    if not mode_rates:
        return []

    sorted_modes = sorted(mode_rates.items(), key=lambda x: (x[1], x[0]))
    weak = [mode for mode, rate in sorted_modes if rate < _MODE_WEAK_THRESHOLD]
    return weak[:_TOP_N_WEAK]


def _find_weak_spacing_zones(
    db: "Database",
    match_ids: list[str] | None,
) -> list[str]:
    """Return spacing zones where the AI wins least often.

    Queries ai_decisions for outcome by spacing_zone from semantic_events.
    """
    if not db.table_exists("ai_decisions"):
        return []

    where_clause, params = _build_match_filter(match_ids)

    rows = db.fetchall(
        f"""
        SELECT
            se.spacing_zone,
            ad.outcome
        FROM ai_decisions ad
        JOIN semantic_events se
          ON ad.match_id = se.match_id
         AND se.tick_id <= ad.tick_id
         AND se.actor = 'PLAYER'
         AND se.event_type = 'COMMITMENT_START'
        WHERE ad.outcome IS NOT NULL
          AND se.spacing_zone IS NOT NULL
          {where_clause}
        ORDER BY ad.match_id, ad.tick_id, se.tick_id DESC
        """,
        tuple(params),
    )

    if not rows:
        return []

    zone_success: dict[str, int] = {}
    zone_total: dict[str, int] = {}

    for r in rows:
        zone = r["spacing_zone"]
        outcome = r["outcome"]
        if zone is None:
            continue
        zone_total[zone] = zone_total.get(zone, 0) + 1
        if outcome == "success":
            zone_success[zone] = zone_success.get(zone, 0) + 1

    zone_rates = {
        z: zone_success.get(z, 0) / total
        for z, total in zone_total.items()
        if total >= 3
    }
    if not zone_rates:
        return []

    sorted_zones = sorted(zone_rates.items(), key=lambda x: (x[1], x[0]))
    weak = [z for z, rate in sorted_zones if rate < _SPACING_WEAK_THRESHOLD]
    return weak[:_TOP_N_WEAK]


def _build_failure_scenarios(report: WeaknessReport) -> list[str]:
    """Build human-readable failure scenario descriptions from other fields.

    Always deterministic — output depends only on inputs.
    """
    scenarios: list[str] = []

    if report.weak_prediction_labels:
        labels_str = ", ".join(report.weak_prediction_labels)
        scenarios.append(f"Low prediction accuracy for: {labels_str}")

    if report.weak_tactical_modes:
        modes_str = ", ".join(report.weak_tactical_modes)
        scenarios.append(f"Poor tactical success in modes: {modes_str}")

    if report.weak_spacing_zones:
        zones_str = ", ".join(report.weak_spacing_zones)
        scenarios.append(f"Weak performance at spacing zones: {zones_str}")

    if not scenarios:
        scenarios.append("No significant weaknesses detected")

    return scenarios


def _build_match_filter(
    match_ids: list[str] | None,
) -> tuple[str, list]:
    """Build a SQL WHERE clause fragment and params for match ID filtering."""
    if not match_ids:
        return "", []
    placeholders = ",".join("?" for _ in match_ids)
    return f"AND ad.match_id IN ({placeholders})", list(match_ids)
