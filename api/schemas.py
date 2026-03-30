"""Pydantic request and response schemas for the Adaptive Duelist API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    game: dict[str, Any]
    ai: dict[str, Any]


class StatsResponse(BaseModel):
    total_matches: int
    human_matches: int
    self_play_matches: int
    active_model_version: str | None


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

class SelfPlayRequest(BaseModel):
    n_matches: int = Field(default=10, ge=1, le=500)
    profiles: list[str] = Field(default_factory=list)
    seed: int = Field(default=0)
    max_ticks: int = Field(default=5000, ge=100, le=50000)


class SelfPlayResponse(BaseModel):
    matches_run: int
    profiles_used: list[str]
    semantic_events_inserted: int
    match_ids: list[str]


class EvaluateRequest(BaseModel):
    n_matches: int = Field(default=10, ge=1, le=200)
    tier: str = Field(default="T2_FULL_ADAPTIVE")
    seed: int = Field(default=0)
    max_ticks: int = Field(default=5000, ge=100, le=50000)


class EvaluateResponse(BaseModel):
    tier: str
    match_count: int
    seed_start: int
    win_rate: dict[str, Any]
    match_length: dict[str, Any]
    damage: dict[str, Any]
    prediction: dict[str, Any] | None
    planner: dict[str, Any] | None
    performance: dict[str, Any]


class RecentMatchesResponse(BaseModel):
    matches: list[dict[str, Any]]


class MatchReportResponse(BaseModel):
    match_id: str
    report: dict[str, Any]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    auto_promote: bool = Field(default=False)
    source_filter: str | None = Field(default=None)


class TrainResponse(BaseModel):
    retrain_needed: bool
    retrained: bool
    version: str | None
    holdout_accuracy: float | None
    promoted: bool
    promotion_reason: str


class CurriculumRequest(BaseModel):
    n_matches: int = Field(default=20, ge=1, le=500)
    auto_promote: bool = Field(default=False)
    seed: int = Field(default=0)
    max_ticks: int = Field(default=5000, ge=100, le=50000)


class TrainingStatusResponse(BaseModel):
    total_matches: int
    last_dataset_size: int
    delta: int
    retrain_threshold: int
    retrain_needed: bool


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelStatusResponse(BaseModel):
    active_version: str | None
    active_accuracy: float | None
    all_versions: list[dict[str, Any]]


class BaselineRequest(BaseModel):
    n_matches: int = Field(default=10, ge=1, le=200)
    tag: str = Field(default="")
    tier: str = Field(default="T2_FULL_ADAPTIVE")
    seed: int = Field(default=0)


class BaselineResponse(BaseModel):
    path: str
    tier: str
    tag: str


class RegressionRequest(BaseModel):
    n_matches: int = Field(default=10, ge=1, le=200)
    tier: str = Field(default="T2_FULL_ADAPTIVE")
    seed: int = Field(default=0)
    baseline_tag: str = Field(default="")


class RegressionResponse(BaseModel):
    passed: bool
    checks: list[dict[str, Any]]
    failures: list[dict[str, Any]]
