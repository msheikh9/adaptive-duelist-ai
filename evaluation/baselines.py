"""Baseline snapshot schema, save/load under baselines/.

A baseline is a frozen JSON artifact capturing the EvaluationResult
for a specific tier at a specific commit. Used by regression_checker
to compare current vs stored results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import (
    DamageMetrics,
    EvaluationResult,
    MatchLengthMetrics,
    PerformanceMetrics,
    PlannerSuccessMetrics,
    PredictionAccuracyMetrics,
    WinRateMetrics,
)

BASELINES_DIR = Path(__file__).parent.parent / "baselines"


def baseline_filename(tier: str, tag: str = "") -> str:
    """Generate a canonical baseline filename.

    Format: baseline_<TIER>_<tag>.json  (tag optional)
    """
    parts = ["baseline", tier.lower()]
    if tag:
        parts.append(tag)
    return "_".join(parts) + ".json"


def save_baseline(
    result: EvaluationResult,
    tag: str = "",
    directory: Path | None = None,
    git_sha: str = "",
    config_hash: str = "",
) -> Path:
    """Save an EvaluationResult as a baseline artifact.

    Returns the path to the saved file.
    """
    dest = directory or BASELINES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config_hash": config_hash,
        "tier": result.tier,
        "match_count": result.match_count,
        "seed_start": result.seed_start,
        "win_rate": asdict(result.win_rate),
        "match_length": asdict(result.match_length),
        "damage": asdict(result.damage),
        "prediction": asdict(result.prediction) if result.prediction else None,
        "planner": asdict(result.planner) if result.planner else None,
        "performance": asdict(result.performance),
        "replay_verification": asdict(result.replay_verification) if result.replay_verification else None,
    }

    fname = baseline_filename(result.tier, tag)
    path = dest / fname
    path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
    return path


def load_baseline(path: Path) -> dict:
    """Load a baseline artifact from JSON.

    Returns the raw dict (schema_version, metrics, metadata).
    Raises FileNotFoundError if path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")
    return json.loads(path.read_text())


def find_baseline(
    tier: str,
    tag: str = "",
    directory: Path | None = None,
) -> Path | None:
    """Find the matching baseline file for a tier and optional tag.

    Returns None if no matching baseline exists.
    """
    dest = directory or BASELINES_DIR
    fname = baseline_filename(tier, tag)
    path = dest / fname
    return path if path.exists() else None


def list_baselines(directory: Path | None = None) -> list[Path]:
    """List all baseline files in the baselines directory."""
    dest = directory or BASELINES_DIR
    if not dest.exists():
        return []
    return sorted(dest.glob("baseline_*.json"))
