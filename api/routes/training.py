"""Training routes: run pipeline, curriculum, status."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_configs, get_db, get_db_path
from api.schemas import (
    CurriculumRequest,
    TrainRequest,
    TrainResponse,
    TrainingStatusResponse,
)
from data.db import Database

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/run", response_model=TrainResponse)
def run_training(
    body: TrainRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> TrainResponse:
    from ai.training.pipeline import TrainingPipeline

    game_cfg, ai_cfg = configs
    pipeline = TrainingPipeline(db_path, game_cfg, ai_cfg)
    result = pipeline.run_pipeline(
        auto_promote=body.auto_promote,
        source_filter=body.source_filter,
    )

    return TrainResponse(
        retrain_needed=result.retrain_needed,
        retrained=result.retrained,
        version=result.version,
        holdout_accuracy=result.holdout_accuracy,
        promoted=result.promoted,
        promotion_reason=result.promotion_reason,
    )


@router.post("/curriculum", response_model=TrainResponse)
def run_curriculum(
    body: CurriculumRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> TrainResponse:
    from ai.training.pipeline import TrainingPipeline

    game_cfg, ai_cfg = configs
    pipeline = TrainingPipeline(db_path, game_cfg, ai_cfg)
    result = pipeline.run_curriculum_cycle(
        n_matches=body.n_matches,
        auto_promote=body.auto_promote,
        seed_start=body.seed,
        max_ticks=body.max_ticks,
    )

    return TrainResponse(
        retrain_needed=result.retrain_needed,
        retrained=result.retrained,
        version=result.version,
        holdout_accuracy=result.holdout_accuracy,
        promoted=result.promoted,
        promotion_reason=result.promotion_reason,
    )


@router.get("/status", response_model=TrainingStatusResponse)
def training_status(
    db: Database = Depends(get_db),
    configs=Depends(get_configs),
) -> TrainingStatusResponse:
    _, ai_cfg = configs

    total_row = db.fetchone("SELECT COUNT(*) as c FROM matches;")
    total = total_row["c"] if total_row else 0

    last_row = db.fetchone(
        "SELECT dataset_size FROM model_registry ORDER BY created_at DESC LIMIT 1;"
    )
    last_size = last_row["dataset_size"] if last_row and last_row["dataset_size"] else 0

    retrain_every = ai_cfg.training.retrain_after_every_n_matches
    delta = total - last_size
    retrain_needed = delta >= retrain_every and total >= ai_cfg.training.min_matches_to_train

    return TrainingStatusResponse(
        total_matches=total,
        last_dataset_size=last_size,
        delta=delta,
        retrain_threshold=retrain_every,
        retrain_needed=retrain_needed,
    )
