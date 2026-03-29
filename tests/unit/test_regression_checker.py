"""Tests for evaluation/regression_checker.py."""

from __future__ import annotations

import json
import pytest

from evaluation.metrics import (
    DamageMetrics,
    EvaluationResult,
    MatchLengthMetrics,
    PerformanceMetrics,
    PlannerSuccessMetrics,
    PredictionAccuracyMetrics,
    ReplayVerificationMetrics,
    WinRateMetrics,
)
from evaluation.regression_checker import (
    CheckResult,
    RegressionReport,
    ThresholdConfig,
    check_regression,
    load_eval_defaults,
    load_threshold_config,
)


def _make_baseline(
    ai_win_rate=0.70,
    avg_ticks=3000.0,
    hp_diff=40.0,
    top1_acc=0.30,
    planner_sr=0.50,
    p95_tick=0.15,
    replay_pass_rate=None,
) -> dict:
    """Build a baseline dict matching the save format."""
    bl = {
        "schema_version": 1,
        "tier": "T2_FULL_ADAPTIVE",
        "match_count": 50,
        "seed_start": 0,
        "win_rate": {
            "total_matches": 50,
            "ai_wins": 35,
            "player_wins": 15,
            "draws": 0,
            "ai_win_rate": ai_win_rate,
            "player_win_rate": 1 - ai_win_rate,
            "draw_rate": 0.0,
        },
        "match_length": {
            "avg_ticks": avg_ticks,
            "min_ticks": 1000,
            "max_ticks": 5000,
            "median_ticks": 3000.0,
        },
        "damage": {
            "avg_ai_hp_remaining": 100.0,
            "avg_player_hp_remaining": 60.0,
            "avg_hp_differential": hp_diff,
        },
        "prediction": {
            "total_predictions": 100,
            "top1_correct": 30,
            "top1_accuracy": top1_acc,
            "top2_correct": 50,
            "top2_accuracy": 0.50,
        },
        "planner": {
            "total_decisions_with_outcome": 200,
            "mode_success_rates": {"EXPLOIT_PATTERN": 0.55},
            "overall_success_rate": planner_sr,
        },
        "performance": {
            "p95_tick_ms": p95_tick,
            "p95_planner_ms": 0.10,
            "avg_ticks_per_sec": 50000.0,
        },
        "replay_verification": None,
    }
    if replay_pass_rate is not None:
        bl["replay_verification"] = {
            "total_replays": 10,
            "passed": int(10 * replay_pass_rate),
            "failed": 10 - int(10 * replay_pass_rate),
            "pass_rate": replay_pass_rate,
        }
    return bl


def _make_eval_result(
    ai_win_rate=0.70,
    avg_ticks=3000.0,
    hp_diff=40.0,
    top1_acc=0.30,
    planner_sr=0.50,
    p95_tick=0.15,
    replay_pass_rate=None,
) -> EvaluationResult:
    replay = None
    if replay_pass_rate is not None:
        replay = ReplayVerificationMetrics(
            total_replays=10,
            passed=int(10 * replay_pass_rate),
            failed=10 - int(10 * replay_pass_rate),
            pass_rate=replay_pass_rate,
        )
    return EvaluationResult(
        tier="T2_FULL_ADAPTIVE",
        match_count=50,
        seed_start=0,
        win_rate=WinRateMetrics(
            total_matches=50, ai_wins=35, player_wins=15, draws=0,
            ai_win_rate=ai_win_rate, player_win_rate=1 - ai_win_rate,
            draw_rate=0.0,
        ),
        match_length=MatchLengthMetrics(
            avg_ticks=avg_ticks, min_ticks=1000, max_ticks=5000,
            median_ticks=3000.0,
        ),
        damage=DamageMetrics(
            avg_ai_hp_remaining=100.0, avg_player_hp_remaining=60.0,
            avg_hp_differential=hp_diff,
        ),
        prediction=PredictionAccuracyMetrics(
            total_predictions=100, top1_correct=30,
            top1_accuracy=top1_acc, top2_correct=50, top2_accuracy=0.50,
        ),
        planner=PlannerSuccessMetrics(
            total_decisions_with_outcome=200,
            mode_success_rates={"EXPLOIT_PATTERN": 0.55},
            overall_success_rate=planner_sr,
        ),
        performance=PerformanceMetrics(
            p95_tick_ms=p95_tick, p95_planner_ms=0.10,
            avg_ticks_per_sec=50000.0,
        ),
        replay_verification=replay,
    )


class TestCheckRegression:
    def test_identical_passes(self, tmp_path):
        bl = _make_baseline()
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result()
        report = check_regression(current, bl_path)
        assert report.passed
        assert len(report.failures) == 0

    def test_win_rate_regression(self, tmp_path):
        bl = _make_baseline(ai_win_rate=0.70)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(ai_win_rate=0.50)  # 20pp drop
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "ai_win_rate" in failed

    def test_match_length_regression(self, tmp_path):
        bl = _make_baseline(avg_ticks=3000.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(avg_ticks=4500.0)  # 50% increase
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "avg_match_ticks" in failed

    def test_hp_differential_regression(self, tmp_path):
        bl = _make_baseline(hp_diff=40.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(hp_diff=-10.0)  # 50pt drop
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "hp_differential" in failed

    def test_prediction_regression(self, tmp_path):
        bl = _make_baseline(top1_acc=0.35)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(top1_acc=0.20)  # 15pp drop
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "prediction_top1_accuracy" in failed

    def test_planner_regression(self, tmp_path):
        bl = _make_baseline(planner_sr=0.55)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(planner_sr=0.40)  # 15pp drop
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "planner_success_rate" in failed

    def test_performance_regression(self, tmp_path):
        bl = _make_baseline(p95_tick=0.10)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(p95_tick=0.20)  # 100% increase
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "p95_tick_ms" in failed

    def test_custom_thresholds(self, tmp_path):
        bl = _make_baseline(ai_win_rate=0.70)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        # 20pp drop would normally fail, but with relaxed threshold it passes
        current = _make_eval_result(ai_win_rate=0.50)
        thresholds = ThresholdConfig(ai_win_rate_drop=0.25)
        report = check_regression(current, bl_path, thresholds)
        wr_check = next(c for c in report.checks if c.metric == "ai_win_rate")
        assert wr_check.passed

    def test_missing_prediction_skipped(self, tmp_path):
        bl = _make_baseline()
        bl["prediction"] = None
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result()
        current.prediction = None
        report = check_regression(current, bl_path)
        metrics = [c.metric for c in report.checks]
        assert "prediction_top1_accuracy" not in metrics

    def test_within_threshold_passes(self, tmp_path):
        bl = _make_baseline(ai_win_rate=0.70)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        # 5pp drop is within the default 10pp threshold
        current = _make_eval_result(ai_win_rate=0.65)
        report = check_regression(current, bl_path)
        wr_check = next(c for c in report.checks if c.metric == "ai_win_rate")
        assert wr_check.passed

    def test_replay_pass_rate_regression(self, tmp_path):
        """Any drop in replay pass rate should fail with default threshold (0.0)."""
        bl = _make_baseline(replay_pass_rate=1.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(replay_pass_rate=0.8)
        report = check_regression(current, bl_path)
        assert not report.passed
        failed = [c.metric for c in report.failures]
        assert "replay_pass_rate" in failed

    def test_replay_pass_rate_within_threshold(self, tmp_path):
        """Perfect replay on both sides should pass."""
        bl = _make_baseline(replay_pass_rate=1.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result(replay_pass_rate=1.0)
        report = check_regression(current, bl_path)
        rp_check = next(c for c in report.checks if c.metric == "replay_pass_rate")
        assert rp_check.passed

    def test_replay_skipped_when_absent(self, tmp_path):
        """No replay data on either side should not produce a replay check."""
        bl = _make_baseline()  # replay_verification = None
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        current = _make_eval_result()
        report = check_regression(current, bl_path)
        metrics = [c.metric for c in report.checks]
        assert "replay_pass_rate" not in metrics

    def test_replay_corruption_blocks_release(self, tmp_path):
        """Even a single replay failure from perfect baseline should block."""
        bl = _make_baseline(replay_pass_rate=1.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(bl))

        # 9/10 pass = 0.9 pass rate, a 0.1 drop from 1.0
        current = _make_eval_result(replay_pass_rate=0.9)
        report = check_regression(current, bl_path)
        assert not report.passed
        rp_check = next(c for c in report.checks if c.metric == "replay_pass_rate")
        assert not rp_check.passed


class TestLoadThresholdConfig:
    def test_loads_from_yaml(self, tmp_path):
        yaml_content = (
            "regression_thresholds:\n"
            "  ai_win_rate_drop: 0.20\n"
            "  p95_tick_increase_pct: 0.75\n"
        )
        cfg_path = tmp_path / "eval_config.yaml"
        cfg_path.write_text(yaml_content)
        tc = load_threshold_config(cfg_path)
        assert tc.ai_win_rate_drop == 0.20
        assert tc.p95_tick_increase_pct == 0.75
        # Unspecified fields should keep defaults
        assert tc.hp_differential_drop == 30.0

    def test_missing_file_returns_defaults(self, tmp_path):
        tc = load_threshold_config(tmp_path / "nonexistent.yaml")
        assert tc.ai_win_rate_drop == 0.10

    def test_empty_yaml_returns_defaults(self, tmp_path):
        cfg_path = tmp_path / "eval_config.yaml"
        cfg_path.write_text("")
        tc = load_threshold_config(cfg_path)
        assert tc.ai_win_rate_drop == 0.10

    def test_loads_real_eval_config(self):
        """Verify the actual config/eval_config.yaml loads correctly."""
        tc = load_threshold_config()
        assert tc.ai_win_rate_drop == 0.10
        assert tc.avg_ticks_increase_pct == 0.25
        assert tc.p95_tick_increase_pct == 0.50


class TestLoadEvalDefaults:
    def test_loads_from_yaml(self, tmp_path):
        yaml_content = (
            "evaluation_defaults:\n"
            "  matches: 100\n"
            "  seed_start: 42\n"
            "  max_ticks: 10000\n"
        )
        cfg_path = tmp_path / "eval_config.yaml"
        cfg_path.write_text(yaml_content)
        defaults = load_eval_defaults(cfg_path)
        assert defaults["matches"] == 100
        assert defaults["seed_start"] == 42
        assert defaults["max_ticks"] == 10000

    def test_missing_file_returns_defaults(self, tmp_path):
        defaults = load_eval_defaults(tmp_path / "nonexistent.yaml")
        assert defaults["matches"] == 50
        assert defaults["seed_start"] == 0

    def test_loads_real_eval_config(self):
        """Verify the actual config/eval_config.yaml loads correctly."""
        defaults = load_eval_defaults()
        assert defaults["matches"] == 50
        assert defaults["seed_start"] == 0
        assert defaults["max_ticks"] == 20000
