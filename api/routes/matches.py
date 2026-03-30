"""Match routes: self-play, evaluate, report, recent."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_configs, get_db, get_db_path, safe_dict
from api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    MatchReportResponse,
    RecentMatchesResponse,
    SelfPlayRequest,
    SelfPlayResponse,
)
from data.db import Database

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.post("/self-play", response_model=SelfPlayResponse)
def run_self_play_endpoint(
    body: SelfPlayRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> SelfPlayResponse:
    from ai.training.scripted_opponent import ScriptedProfile
    from ai.training.self_play_runner import run_self_play

    game_cfg, ai_cfg = configs

    if body.profiles:
        try:
            profiles = [ScriptedProfile(p) for p in body.profiles]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        profiles = list(ScriptedProfile)

    result = run_self_play(
        n_matches=body.n_matches,
        profiles=profiles,
        seed_start=body.seed,
        db_path=db_path,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
        max_ticks=body.max_ticks,
    )

    return SelfPlayResponse(
        matches_run=result.matches_run,
        profiles_used=result.profiles_used,
        semantic_events_inserted=result.semantic_events_inserted,
        match_ids=result.match_ids,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(
    body: EvaluateRequest,
    db_path=Depends(get_db_path),
    configs=Depends(get_configs),
) -> EvaluateResponse:
    from ai.layers.tactical_planner import AITier
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
        max_ticks=body.max_ticks,
        db_path=db_path,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
    )

    return EvaluateResponse(
        tier=result.tier,
        match_count=result.match_count,
        seed_start=result.seed_start,
        win_rate=safe_dict(result.win_rate),
        match_length=safe_dict(result.match_length),
        damage=safe_dict(result.damage),
        prediction=safe_dict(result.prediction) if result.prediction else None,
        planner=safe_dict(result.planner) if result.planner else None,
        performance=safe_dict(result.performance),
    )


@router.get("/recent", response_model=RecentMatchesResponse)
def recent_matches(
    limit: int = 20,
    db: Database = Depends(get_db),
) -> RecentMatchesResponse:
    rows = db.fetchall(
        "SELECT match_id, winner, player_hp_final, ai_hp_final, total_ticks, started_at "
        "FROM matches ORDER BY started_at DESC LIMIT ?;",
        (limit,),
    )
    matches = [dict(r) for r in rows] if rows else []
    return RecentMatchesResponse(matches=matches)


@router.get("/{match_id}/report", response_model=MatchReportResponse)
def match_report(
    match_id: str,
    db: Database = Depends(get_db),
) -> MatchReportResponse:
    from analytics.report_builder import build_match_report

    row = db.fetchone(
        "SELECT match_id FROM matches WHERE match_id = ?;", (match_id,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    report = build_match_report(db, match_id)
    return MatchReportResponse(match_id=match_id, report=safe_dict(report))
