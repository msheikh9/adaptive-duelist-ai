"""System routes: health, config, stats."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_configs, get_db, get_db_path, safe_dict
from api.schemas import ConfigResponse, HealthResponse, StatsResponse
from data.db import Database

router = APIRouter(prefix="/api", tags=["system"])

VERSION = "12.0.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@router.get("/config", response_model=ConfigResponse)
def get_config(configs=Depends(get_configs)) -> ConfigResponse:
    game_cfg, ai_cfg = configs
    return ConfigResponse(
        game=safe_dict(game_cfg),
        ai=safe_dict(ai_cfg),
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Database = Depends(get_db)) -> StatsResponse:
    total_row = db.fetchone("SELECT COUNT(*) as c FROM matches;")
    total = total_row["c"] if total_row else 0

    try:
        human_row = db.fetchone(
            "SELECT COUNT(*) as c FROM matches WHERE source = 'human';"
        )
        sp_row = db.fetchone(
            "SELECT COUNT(*) as c FROM matches WHERE source = 'self_play';"
        )
        human = human_row["c"] if human_row else 0
        self_play = sp_row["c"] if sp_row else 0
    except Exception:
        human = total
        self_play = 0

    active = db.fetchone(
        "SELECT version FROM model_registry WHERE is_active = 1 "
        "ORDER BY created_at DESC LIMIT 1;"
    )
    version = active["version"] if active else None

    return StatsResponse(
        total_matches=total,
        human_matches=human,
        self_play_matches=self_play,
        active_model_version=version,
    )
