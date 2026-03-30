"""Integration tests for Phase 12: Local API."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """TestClient wired to a fresh temp DB."""
    db_path = tmp_path / "api_test.db"
    monkeypatch.setenv("ADAPTIVE_DUELIST_DB", str(db_path))
    app = create_app()
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Health / System
# ---------------------------------------------------------------------------

class TestSystemRoutes:

    def test_health_ok(self, api_client):
        res = api_client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_config_returns_game_and_ai(self, api_client):
        res = api_client.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert "game" in data
        assert "ai" in data

    def test_stats_empty_db(self, api_client):
        res = api_client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["total_matches"] == 0
        assert data["active_model_version"] is None


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

class TestMatchRoutes:

    def test_recent_matches_empty(self, api_client):
        res = api_client.get("/api/matches/recent")
        assert res.status_code == 200
        assert res.json()["matches"] == []

    def test_recent_matches_limit_param(self, api_client):
        res = api_client.get("/api/matches/recent?limit=5")
        assert res.status_code == 200

    def test_match_report_404(self, api_client):
        res = api_client.get("/api/matches/nonexistent-id/report")
        assert res.status_code == 404

    def test_self_play_runs(self, api_client):
        res = api_client.post("/api/matches/self-play", json={
            "n_matches": 1,
            "seed": 42,
            "max_ticks": 500,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["matches_run"] == 1
        assert len(data["match_ids"]) == 1

    def test_self_play_invalid_profile(self, api_client):
        res = api_client.post("/api/matches/self-play", json={
            "n_matches": 1,
            "profiles": ["NOT_A_PROFILE"],
        })
        assert res.status_code == 422

    def test_evaluate_t0(self, api_client):
        res = api_client.post("/api/matches/evaluate", json={
            "n_matches": 1,
            "tier": "T0_BASELINE",
            "seed": 0,
            "max_ticks": 500,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["tier"] == "T0_BASELINE"
        assert data["match_count"] == 1

    def test_evaluate_invalid_tier(self, api_client):
        res = api_client.post("/api/matches/evaluate", json={
            "n_matches": 1,
            "tier": "INVALID_TIER",
        })
        assert res.status_code == 422

    def test_stats_after_self_play(self, api_client):
        api_client.post("/api/matches/self-play", json={
            "n_matches": 2,
            "seed": 0,
            "max_ticks": 500,
        })
        res = api_client.get("/api/stats")
        assert res.status_code == 200
        assert res.json()["total_matches"] >= 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrainingRoutes:

    def test_training_status_empty(self, api_client):
        res = api_client.get("/api/training/status")
        assert res.status_code == 200
        data = res.json()
        assert "total_matches" in data
        assert "retrain_needed" in data

    def test_run_training_not_needed(self, api_client):
        res = api_client.post("/api/training/run", json={"auto_promote": False})
        assert res.status_code == 200
        data = res.json()
        assert data["retrain_needed"] is False

    def test_curriculum_endpoint(self, api_client):
        res = api_client.post("/api/training/curriculum", json={
            "n_matches": 2,
            "auto_promote": False,
            "seed": 0,
            "max_ticks": 500,
        })
        assert res.status_code == 200


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModelRoutes:

    def test_model_status_empty(self, api_client):
        res = api_client.get("/api/models/status")
        assert res.status_code == 200
        data = res.json()
        assert data["active_version"] is None
        assert data["all_versions"] == []

    def test_check_regression_no_baseline(self, api_client):
        res = api_client.post("/api/models/check-regression", json={
            "n_matches": 1,
            "tier": "T0_BASELINE",
            "baseline_tag": "nonexistent-tag-xyz",
        })
        assert res.status_code == 404
