"""Tests for evaluation/baselines.py — save/load/find baseline artifacts."""

from __future__ import annotations

import json
import pytest

from evaluation.baselines import (
    baseline_filename,
    find_baseline,
    list_baselines,
    load_baseline,
    save_baseline,
)
from evaluation.metrics import (
    DamageMetrics,
    EvaluationResult,
    MatchLengthMetrics,
    PerformanceMetrics,
    WinRateMetrics,
)


def _make_eval_result(tier="T2_FULL_ADAPTIVE") -> EvaluationResult:
    return EvaluationResult(
        tier=tier,
        match_count=10,
        seed_start=0,
        win_rate=WinRateMetrics(
            total_matches=10, ai_wins=7, player_wins=3, draws=0,
            ai_win_rate=0.7, player_win_rate=0.3, draw_rate=0.0,
        ),
        match_length=MatchLengthMetrics(
            avg_ticks=2000.0, min_ticks=1000, max_ticks=3000,
            median_ticks=2000.0,
        ),
        damage=DamageMetrics(
            avg_ai_hp_remaining=100.0, avg_player_hp_remaining=50.0,
            avg_hp_differential=50.0,
        ),
        prediction=None,
        planner=None,
        performance=PerformanceMetrics(
            p95_tick_ms=0.15, p95_planner_ms=0.10,
            avg_ticks_per_sec=50000.0,
        ),
    )


class TestBaselineFilename:
    def test_no_tag(self):
        assert baseline_filename("T0_BASELINE") == "baseline_t0_baseline.json"

    def test_with_tag(self):
        assert baseline_filename("T2_FULL_ADAPTIVE", "v1.0") == \
            "baseline_t2_full_adaptive_v1.0.json"


class TestSaveBaseline:
    def test_creates_file(self, tmp_path):
        result = _make_eval_result()
        path = save_baseline(result, directory=tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        assert data["tier"] == "T2_FULL_ADAPTIVE"
        assert data["win_rate"]["ai_win_rate"] == 0.7

    def test_with_tag(self, tmp_path):
        result = _make_eval_result()
        path = save_baseline(result, tag="test", directory=tmp_path)
        assert "test" in path.name

    def test_with_git_sha(self, tmp_path):
        result = _make_eval_result()
        path = save_baseline(result, directory=tmp_path, git_sha="abc123")
        data = json.loads(path.read_text())
        assert data["git_sha"] == "abc123"

    def test_creates_directory(self, tmp_path):
        dest = tmp_path / "nested" / "dir"
        result = _make_eval_result()
        path = save_baseline(result, directory=dest)
        assert path.exists()


class TestLoadBaseline:
    def test_roundtrip(self, tmp_path):
        result = _make_eval_result()
        path = save_baseline(result, directory=tmp_path)
        data = load_baseline(path)
        assert data["match_count"] == 10
        assert data["performance"]["p95_tick_ms"] == 0.15

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")


class TestFindBaseline:
    def test_found(self, tmp_path):
        result = _make_eval_result()
        save_baseline(result, directory=tmp_path)
        found = find_baseline("T2_FULL_ADAPTIVE", directory=tmp_path)
        assert found is not None
        assert found.exists()

    def test_not_found(self, tmp_path):
        found = find_baseline("T0_BASELINE", directory=tmp_path)
        assert found is None

    def test_with_tag(self, tmp_path):
        result = _make_eval_result()
        save_baseline(result, tag="v2", directory=tmp_path)
        found = find_baseline("T2_FULL_ADAPTIVE", tag="v2", directory=tmp_path)
        assert found is not None


class TestListBaselines:
    def test_empty(self, tmp_path):
        assert list_baselines(tmp_path) == []

    def test_multiple(self, tmp_path):
        r1 = _make_eval_result("T0_BASELINE")
        r2 = _make_eval_result("T2_FULL_ADAPTIVE")
        save_baseline(r1, directory=tmp_path)
        save_baseline(r2, directory=tmp_path)
        found = list_baselines(tmp_path)
        assert len(found) == 2

    def test_nonexistent_dir(self, tmp_path):
        assert list_baselines(tmp_path / "nope") == []
