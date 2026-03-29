"""Integration tests for the full evaluation → baseline → regression flow."""

from __future__ import annotations

import json
import pytest

from config.config_loader import load_config
from ai.layers.tactical_planner import AITier
from evaluation.baselines import save_baseline, load_baseline, find_baseline
from evaluation.match_runner import run_evaluation
from evaluation.metrics import EvaluationResult
from evaluation.regression_checker import (
    ThresholdConfig,
    check_regression,
)


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def ai_cfg():
    _, cfg, _ = load_config()
    return cfg


class TestEvaluationT0:
    """T0 baseline evaluation — fast, no DB required."""

    def test_run_evaluation_t0(self, game_cfg):
        result = run_evaluation(
            n_matches=3,
            seed_start=42,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        assert result.tier == "T0_BASELINE"
        assert result.match_count == 3
        assert result.win_rate.total_matches == 3
        assert result.match_length.avg_ticks > 0
        assert result.prediction is None  # T0 has no planner
        assert result.planner is None

    def test_win_rate_sane(self, game_cfg):
        result = run_evaluation(
            n_matches=5,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=5000,
            game_cfg=game_cfg,
        )
        wr = result.win_rate
        assert wr.ai_wins + wr.player_wins + wr.draws == 5
        assert 0.0 <= wr.ai_win_rate <= 1.0

    def test_performance_populated(self, game_cfg):
        result = run_evaluation(
            n_matches=3,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        assert result.performance.avg_ticks_per_sec > 0
        assert result.performance.p95_tick_ms >= 0


class TestEvaluationT2:
    """T2 full adaptive evaluation — exercises full planner."""

    def test_run_evaluation_t2(self, game_cfg, ai_cfg, tmp_path):
        result = run_evaluation(
            n_matches=2,
            seed_start=42,
            tier=AITier.T2_FULL_ADAPTIVE,
            max_ticks=500,
            db_path=tmp_path / "eval.db",
            game_cfg=game_cfg,
            ai_cfg=ai_cfg,
        )
        assert result.tier == "T2_FULL_ADAPTIVE"
        assert result.match_count == 2
        assert result.win_rate.total_matches == 2
        assert result.match_length.avg_ticks > 0


class TestBaselineRoundtrip:
    """Test save → load → regression check flow."""

    def test_save_and_find(self, game_cfg, tmp_path):
        result = run_evaluation(
            n_matches=3,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        path = save_baseline(result, directory=tmp_path)
        assert path.exists()

        found = find_baseline("T0_BASELINE", directory=tmp_path)
        assert found is not None

        data = load_baseline(found)
        assert data["tier"] == "T0_BASELINE"
        assert data["match_count"] == 3

    def test_regression_check_self(self, game_cfg, tmp_path):
        """An evaluation compared against itself should always pass."""
        result = run_evaluation(
            n_matches=3,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        path = save_baseline(result, directory=tmp_path)

        # Same evaluation vs its own baseline
        report = check_regression(result, path)
        assert report.passed
        assert len(report.failures) == 0

    def test_regression_detected_on_worse_results(self, game_cfg, tmp_path):
        """Verify that worse metrics trip the regression checker."""
        result = run_evaluation(
            n_matches=3,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        path = save_baseline(result, directory=tmp_path)
        baseline = load_baseline(path)

        # Fabricate a much worse current result
        from evaluation.metrics import (
            WinRateMetrics, MatchLengthMetrics, DamageMetrics,
            PerformanceMetrics,
        )
        worse = EvaluationResult(
            tier="T0_BASELINE",
            match_count=3,
            seed_start=0,
            win_rate=WinRateMetrics(
                total_matches=3, ai_wins=0, player_wins=3, draws=0,
                ai_win_rate=0.0, player_win_rate=1.0, draw_rate=0.0,
            ),
            match_length=MatchLengthMetrics(
                avg_ticks=20000.0, min_ticks=20000, max_ticks=20000,
                median_ticks=20000.0,
            ),
            damage=DamageMetrics(
                avg_ai_hp_remaining=0.0, avg_player_hp_remaining=200.0,
                avg_hp_differential=-200.0,
            ),
            prediction=None,
            planner=None,
            performance=PerformanceMetrics(
                p95_tick_ms=100.0, p95_planner_ms=100.0,
                avg_ticks_per_sec=10.0,
            ),
        )

        report = check_regression(worse, path)
        assert not report.passed
        assert len(report.failures) > 0

    def test_replay_regression_blocks_release(self, game_cfg, tmp_path):
        """Baseline with perfect replays vs degraded current fails the gate."""
        from evaluation.metrics import (
            WinRateMetrics, MatchLengthMetrics, DamageMetrics,
            PerformanceMetrics, ReplayVerificationMetrics,
        )

        # Create a baseline with perfect replay verification
        result = run_evaluation(
            n_matches=3,
            seed_start=0,
            tier=AITier.T0_BASELINE,
            max_ticks=2000,
            game_cfg=game_cfg,
        )
        # Inject perfect replay verification into the result
        result.replay_verification = ReplayVerificationMetrics(
            total_replays=10, passed=10, failed=0, pass_rate=1.0,
        )
        path = save_baseline(result, directory=tmp_path)

        # Fabricate a current result with degraded replay pass rate
        degraded = EvaluationResult(
            tier=result.tier,
            match_count=result.match_count,
            seed_start=result.seed_start,
            win_rate=result.win_rate,
            match_length=result.match_length,
            damage=result.damage,
            prediction=None,
            planner=None,
            performance=result.performance,
            replay_verification=ReplayVerificationMetrics(
                total_replays=10, passed=8, failed=2, pass_rate=0.8,
            ),
        )

        report = check_regression(degraded, path)
        assert not report.passed
        failed_metrics = [c.metric for c in report.failures]
        assert "replay_pass_rate" in failed_metrics
