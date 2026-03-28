"""Tests for performance profiling and benchmark helpers."""

from __future__ import annotations

import json
import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    d.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
        "config_hash) VALUES ('m1', 's1', '2025-01-01', 42, 'h');",
    )
    yield d
    d.close()


class TestPerfProfiler:
    def test_percentiles(self):
        from scripts.perf_profiler import _percentiles

        samples = [0.001 * i for i in range(100)]
        stats = _percentiles(samples)
        assert stats["count"] == 100
        assert stats["mean_ms"] > 0
        assert stats["p50_ms"] <= stats["p90_ms"]
        assert stats["p90_ms"] <= stats["p95_ms"]
        assert stats["p95_ms"] <= stats["p99_ms"]
        assert stats["p99_ms"] <= stats["max_ms"]

    def test_empty_percentiles(self):
        from scripts.perf_profiler import _percentiles

        stats = _percentiles([])
        assert stats == {"count": 0}

    def test_profile_engine_tick(self, game_cfg):
        from scripts.perf_profiler import profile_engine_tick

        stats = profile_engine_tick(game_cfg, n_ticks=50, seed=42)
        assert stats["count"] > 0
        assert stats["mean_ms"] > 0

    def test_profile_prediction(self, game_cfg, db):
        from scripts.perf_profiler import profile_prediction

        _, ai_cfg, _ = load_config()
        stats = profile_prediction(ai_cfg, game_cfg, db, n_ticks=20, seed=42)
        assert stats["count"] == 20
        assert stats["mean_ms"] >= 0

    def test_profile_analytics(self, db):
        from scripts.perf_profiler import profile_analytics

        stats = profile_analytics(db, "m1")
        assert stats["count"] == 20
        assert stats["mean_ms"] >= 0


class TestBenchmarkHelpers:
    def test_single_match(self, game_cfg):
        from scripts.benchmark import benchmark_single_match

        result = benchmark_single_match(game_cfg, seed=42)
        assert result["ticks"] > 0
        assert result["elapsed_s"] > 0
        assert result["winner"] in ("PLAYER", "AI", "DRAW")

    def test_batch_matches(self, game_cfg):
        from scripts.benchmark import benchmark_batch_matches

        result = benchmark_batch_matches(game_cfg, n=3, seed=42)
        assert result["matches"] == 3
        assert result["total_ticks"] > 0
        assert result["ticks_per_sec"] > 0

    def test_report_generation(self, db):
        from scripts.benchmark import benchmark_report_generation

        result = benchmark_report_generation(db, ["m1"])
        assert result["count"] == 1

    def test_report_generation_empty(self, db):
        from scripts.benchmark import benchmark_report_generation

        result = benchmark_report_generation(db, [])
        assert result["count"] == 0


class TestExportResults:
    def test_simulation_summary(self, db):
        from scripts.export_results import export_simulation_summary

        result = export_simulation_summary(db)
        assert result["total_matches"] == 1
        assert len(result["matches"]) == 1
        assert result["matches"][0]["match_id"] == "m1"

    def test_simulation_summary_json(self, db):
        from scripts.export_results import export_simulation_summary

        result = export_simulation_summary(db)
        output = json.dumps(result, default=str)
        parsed = json.loads(output)
        assert parsed["total_matches"] == 1
