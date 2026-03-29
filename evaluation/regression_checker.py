"""Compare current evaluation results against a stored baseline.

Flags regressions when metrics degrade beyond configured thresholds.
Returns structured results with per-metric pass/fail and a CI-friendly
overall verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evaluation.baselines import load_baseline
from evaluation.metrics import EvaluationResult

EVAL_CONFIG_PATH = Path(__file__).parent.parent / "config" / "eval_config.yaml"


@dataclass
class ThresholdConfig:
    """Regression thresholds — how much worse a metric can be before flagging.

    All values are expressed as absolute deltas (not percentages).
    A positive threshold means "current may be this much lower than baseline."
    """

    # Win rate: AI win rate may drop by at most this much (e.g. 0.10 = 10pp)
    ai_win_rate_drop: float = 0.10
    # Match length: avg ticks may increase by at most this factor
    avg_ticks_increase_pct: float = 0.25
    # Damage: HP differential may drop by at most this much
    hp_differential_drop: float = 30.0
    # Prediction: top-1 accuracy may drop by at most this much
    prediction_accuracy_drop: float = 0.10
    # Planner: overall success rate may drop by at most this much
    planner_success_drop: float = 0.10
    # Performance: p95 tick time may increase by at most this factor
    p95_tick_increase_pct: float = 0.50
    # Replay: pass rate may drop by at most this much (0.0 = no failures allowed)
    replay_pass_rate_drop: float = 0.0


def load_threshold_config(path: Path | None = None) -> ThresholdConfig:
    """Load regression thresholds from eval_config.yaml.

    Falls back to ThresholdConfig defaults for any missing keys.
    If the file does not exist, returns pure defaults.
    """
    config_path = path or EVAL_CONFIG_PATH
    if not config_path.exists():
        return ThresholdConfig()
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("regression_thresholds", {})
    if not raw:
        return ThresholdConfig()
    fields = {f.name for f in ThresholdConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in fields}
    return ThresholdConfig(**kwargs)


def load_eval_defaults(path: Path | None = None) -> dict:
    """Load evaluation defaults from eval_config.yaml.

    Returns dict with 'matches', 'seed_start', 'max_ticks'.
    Falls back to hardcoded defaults if file is missing.
    """
    config_path = path or EVAL_CONFIG_PATH
    defaults = {"matches": 50, "seed_start": 0, "max_ticks": 20000}
    if not config_path.exists():
        return defaults
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("evaluation_defaults", {})
    defaults.update({k: v for k, v in raw.items() if k in defaults})
    return defaults


@dataclass
class CheckResult:
    """Result of one metric comparison."""

    metric: str
    baseline_value: float
    current_value: float
    threshold: float
    passed: bool
    detail: str


@dataclass
class RegressionReport:
    """Full regression check report."""

    tier: str
    baseline_path: str
    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def check_regression(
    current: EvaluationResult,
    baseline_path: Path,
    thresholds: ThresholdConfig | None = None,
) -> RegressionReport:
    """Compare current evaluation results against baseline.

    Args:
        current: Fresh evaluation results.
        baseline_path: Path to the stored baseline JSON.
        thresholds: Regression thresholds (defaults if None).

    Returns:
        RegressionReport with per-metric checks and overall pass/fail.
    """
    if thresholds is None:
        thresholds = load_threshold_config()

    baseline = load_baseline(baseline_path)
    report = RegressionReport(tier=current.tier, baseline_path=str(baseline_path))

    # --- Win rate ---
    bl_wr = baseline["win_rate"]["ai_win_rate"]
    cur_wr = current.win_rate.ai_win_rate
    drop = bl_wr - cur_wr
    passed = drop <= thresholds.ai_win_rate_drop
    report.checks.append(CheckResult(
        metric="ai_win_rate",
        baseline_value=bl_wr,
        current_value=cur_wr,
        threshold=thresholds.ai_win_rate_drop,
        passed=passed,
        detail=f"drop={drop:+.3f} (max allowed={thresholds.ai_win_rate_drop})",
    ))
    if not passed:
        report.passed = False

    # --- Match length ---
    bl_ticks = baseline["match_length"]["avg_ticks"]
    cur_ticks = current.match_length.avg_ticks
    if bl_ticks > 0:
        increase_pct = (cur_ticks - bl_ticks) / bl_ticks
    else:
        increase_pct = 0.0
    passed = increase_pct <= thresholds.avg_ticks_increase_pct
    report.checks.append(CheckResult(
        metric="avg_match_ticks",
        baseline_value=bl_ticks,
        current_value=cur_ticks,
        threshold=thresholds.avg_ticks_increase_pct,
        passed=passed,
        detail=f"increase={increase_pct:+.1%} (max allowed={thresholds.avg_ticks_increase_pct:.0%})",
    ))
    if not passed:
        report.passed = False

    # --- HP differential ---
    bl_diff = baseline["damage"]["avg_hp_differential"]
    cur_diff = current.damage.avg_hp_differential
    drop = bl_diff - cur_diff
    passed = drop <= thresholds.hp_differential_drop
    report.checks.append(CheckResult(
        metric="hp_differential",
        baseline_value=bl_diff,
        current_value=cur_diff,
        threshold=thresholds.hp_differential_drop,
        passed=passed,
        detail=f"drop={drop:+.1f} (max allowed={thresholds.hp_differential_drop})",
    ))
    if not passed:
        report.passed = False

    # --- Prediction accuracy (only if both have data) ---
    bl_pred = baseline.get("prediction")
    if bl_pred and current.prediction and bl_pred["total_predictions"] > 0:
        bl_acc = bl_pred["top1_accuracy"]
        cur_acc = current.prediction.top1_accuracy
        drop = bl_acc - cur_acc
        passed = drop <= thresholds.prediction_accuracy_drop
        report.checks.append(CheckResult(
            metric="prediction_top1_accuracy",
            baseline_value=bl_acc,
            current_value=cur_acc,
            threshold=thresholds.prediction_accuracy_drop,
            passed=passed,
            detail=f"drop={drop:+.3f} (max allowed={thresholds.prediction_accuracy_drop})",
        ))
        if not passed:
            report.passed = False

    # --- Planner success (only if both have data) ---
    bl_plan = baseline.get("planner")
    if bl_plan and current.planner and bl_plan["total_decisions_with_outcome"] > 0:
        bl_sr = bl_plan["overall_success_rate"]
        cur_sr = current.planner.overall_success_rate
        drop = bl_sr - cur_sr
        passed = drop <= thresholds.planner_success_drop
        report.checks.append(CheckResult(
            metric="planner_success_rate",
            baseline_value=bl_sr,
            current_value=cur_sr,
            threshold=thresholds.planner_success_drop,
            passed=passed,
            detail=f"drop={drop:+.3f} (max allowed={thresholds.planner_success_drop})",
        ))
        if not passed:
            report.passed = False

    # --- Performance: p95 tick time ---
    bl_p95 = baseline["performance"]["p95_tick_ms"]
    cur_p95 = current.performance.p95_tick_ms
    if bl_p95 > 0:
        increase_pct = (cur_p95 - bl_p95) / bl_p95
    else:
        increase_pct = 0.0
    passed = increase_pct <= thresholds.p95_tick_increase_pct
    report.checks.append(CheckResult(
        metric="p95_tick_ms",
        baseline_value=bl_p95,
        current_value=cur_p95,
        threshold=thresholds.p95_tick_increase_pct,
        passed=passed,
        detail=f"increase={increase_pct:+.1%} (max allowed={thresholds.p95_tick_increase_pct:.0%})",
    ))
    if not passed:
        report.passed = False

    # --- Replay verification pass rate ---
    bl_replay = baseline.get("replay_verification")
    if bl_replay and current.replay_verification and bl_replay["total_replays"] > 0:
        bl_rpr = bl_replay["pass_rate"]
        cur_rpr = current.replay_verification.pass_rate
        drop = bl_rpr - cur_rpr
        passed = drop <= thresholds.replay_pass_rate_drop
        report.checks.append(CheckResult(
            metric="replay_pass_rate",
            baseline_value=bl_rpr,
            current_value=cur_rpr,
            threshold=thresholds.replay_pass_rate_drop,
            passed=passed,
            detail=f"drop={drop:+.3f} (max allowed={thresholds.replay_pass_rate_drop})",
        ))
        if not passed:
            report.passed = False

    return report
