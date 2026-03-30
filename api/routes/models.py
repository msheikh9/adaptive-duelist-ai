"""Model routes: status, baseline, regression check."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_configs, get_db, get_db_path, safe_dict
from api.schemas import (
    BaselineRequest,
    BaselineResponse,
    ModelStatusResponse,
    RegressionRequest,
    RegressionResponse,
)
from data.db import Database

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/status", response_model=ModelStatusResponse)
def model_status(db: Database = Depends(get_db)) -> ModelStatusResponse:
    active = db.fetchone(
        "SELECT version, eval_accuracy FROM model_registry "
        "WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1;"
    )
    all_versions = db.fetchall(
        "SELECT version, eval_accuracy, is_active, created_at "
        "FROM model_registry ORDER BY created_at DESC;"
    )

    return ModelStatusResponse(
        active_version=active["version"] if active else None,
        active_accuracy=active["eval_accuracy"] if active else None,
        all_versions=[dict(r) for r in all_versions] if all_versions else [],
    )


@router.post("/baseline", response_model=BaselineResponse)
def create_baseline(
    body: BaselineRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> BaselineResponse:
    from ai.layers.tactical_planner import AITier
    from evaluation.baselines import save_baseline
    from evaluation.match_runner import run_evaluation

    game_cfg, ai_cfg = configs

    try:
        tier = AITier[body.tier]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown tier: {body.tier}")

    result = run_evaluation(
        n_matches=body.n_matches,
        seed_start=body.seed,
        tier=tier,
        db_path=db_path,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
    )
    path = save_baseline(result, tag=body.tag)

    return BaselineResponse(
        path=str(path),
        tier=result.tier,
        tag=body.tag,
    )


@router.post("/check-regression", response_model=RegressionResponse)
def check_regression_endpoint(
    body: RegressionRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> RegressionResponse:
    from ai.layers.tactical_planner import AITier
    from evaluation.baselines import find_baseline
    from evaluation.match_runner import run_evaluation
    from evaluation.regression_checker import check_regression

    game_cfg, ai_cfg = configs

    try:
        tier = AITier[body.tier]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown tier: {body.tier}")

    baseline_path = find_baseline(body.tier, tag=body.baseline_tag)
    if baseline_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline found for tier={body.tier} tag={body.baseline_tag!r}",
        )

    result = run_evaluation(
        n_matches=body.n_matches,
        seed_start=body.seed,
        tier=tier,
        db_path=db_path,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
    )
    report = check_regression(result, baseline_path)

    return RegressionResponse(
        passed=report.passed,
        checks=[safe_dict(c) for c in report.checks],
        failures=[safe_dict(c) for c in report.failures],
    )
