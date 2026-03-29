"""Deterministic batch evaluator for AI tiers.

Runs N matches at fixed seeds per tier, collects results, and computes
all evaluation metrics. Supports T0/T1/T2 with identical game configs.
Collects real per-tick latency samples for accurate p95 measurements.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from ai.layers.tactical_planner import AITier
from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from evaluation.metrics import (
    EvaluationResult,
    compute_damage,
    compute_match_length,
    compute_performance,
    compute_planner_success,
    compute_prediction_accuracy,
    compute_replay_verification,
    compute_win_rate,
)


def _run_baseline_match_instrumented(game_cfg, seed: int, max_ticks: int) -> dict:
    """Run a T0 baseline match, collecting per-tick latency samples."""
    from tests.fixtures.headless_engine import HeadlessMatch
    from game.state import MatchStatus

    match = HeadlessMatch(game_cfg, rng_seed=seed)
    tick_latencies: list[float] = []
    t0 = time.perf_counter()
    for _ in range(max_ticks):
        if match.state.match_status == MatchStatus.ENDED:
            break
        t_tick = time.perf_counter()
        match.tick()
        tick_latencies.append((time.perf_counter() - t_tick) * 1000)
    elapsed = time.perf_counter() - t0
    s = match.state
    return {
        "ticks": s.tick_id,
        "winner": s.winner or "DRAW",
        "player_hp": s.player.hp,
        "ai_hp": s.ai.hp,
        "elapsed_s": elapsed,
        "tick_latencies_ms": tick_latencies,
    }


def _run_planner_match_instrumented(
    game_cfg, ai_cfg, db: Database,
    seed: int, tier: AITier, max_ticks: int,
    match_id: str,
    force_sklearn: bool = False,
) -> dict:
    """Run a T1/T2 match, collecting per-tick and planner latency samples."""
    from ai.layers.behavior_model import BehaviorModel
    from ai.layers.prediction_engine import PredictionEngine
    from ai.layers.tactical_planner import TacticalPlanner
    from game.combat.actions import FSMState
    from game.combat.collision import HitTracker, check_hit
    from game.combat.damage import apply_hit
    from game.combat.physics import (
        apply_dodge_velocity, apply_velocity, clamp_to_arena, update_facing,
    )
    from game.combat.stamina import tick_stamina
    from game.combat.state_machine import tick_fsm
    from game.entities.ai_fighter import BaselineAIController
    from game.state import (
        ArenaState, FighterState, MatchStatus, SimulationState, TickPhase,
    )

    bm = BehaviorModel(db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
    if force_sklearn:
        pe.try_load_sklearn(force=True)
    planner = TacticalPlanner(db, pe, ai_cfg, game_cfg, tier)

    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    sim = SimulationState(
        tick_id=0, rng_seed=seed,
        player=FighterState(
            x=arena.width_sub // 3, y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
            facing=1,
        ),
        ai=FighterState(
            x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
            facing=-1,
        ),
        arena=arena, match_status=MatchStatus.ACTIVE,
    )

    db.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
        "config_hash) VALUES (?, 'eval', '2025-01-01', ?, 'eval');",
        (match_id, seed),
    )

    bm.on_match_start(match_id)
    pe.on_match_start(match_id)
    planner.on_match_start(match_id, "eval", seed)

    baseline = BaselineAIController(seed)
    hit_tracker = HitTracker()
    fighter_w_sub = game_cfg.fighter.width * scale

    tick_latencies: list[float] = []
    planner_latencies: list[float] = []
    t0 = time.perf_counter()
    for tick in range(max_ticks):
        if sim.match_status == MatchStatus.ENDED:
            break
        sim.tick_id = tick
        sim.set_phase(TickPhase.SIMULATE)

        t_tick = time.perf_counter()

        player_commit = baseline.decide(sim.player, sim, game_cfg)

        t_plan = time.perf_counter()
        ai_commit = planner.decide(sim.ai, sim, game_cfg)
        planner_latencies.append((time.perf_counter() - t_plan) * 1000)

        apply_dodge_velocity(sim.player, game_cfg)
        apply_dodge_velocity(sim.ai, game_cfg)
        apply_velocity(sim.player)
        apply_velocity(sim.ai)
        clamp_to_arena(sim.player, sim.arena, fighter_w_sub)
        clamp_to_arena(sim.ai, sim.arena, fighter_w_sub)
        update_facing(sim.player, sim.ai)

        p_hit = check_hit(sim.player, sim.ai, "player", hit_tracker, game_cfg)
        a_hit = check_hit(sim.ai, sim.player, "ai", hit_tracker, game_cfg)
        if p_hit:
            apply_hit(sim.ai, p_hit)
        if a_hit:
            apply_hit(sim.player, a_hit)

        tick_stamina(sim.player, game_cfg)
        tick_stamina(sim.ai, game_cfg)

        if sim.player.is_free:
            hit_tracker.reset("player")
        if sim.ai.is_free:
            hit_tracker.reset("ai")

        tick_fsm(sim.player, game_cfg)
        tick_fsm(sim.ai, game_cfg)

        if sim.player.fsm_state == FSMState.KO:
            sim.match_status = MatchStatus.ENDED
            sim.winner = "AI"
        elif sim.ai.fsm_state == FSMState.KO:
            sim.match_status = MatchStatus.ENDED
            sim.winner = "PLAYER"

        tick_latencies.append((time.perf_counter() - t_tick) * 1000)

    elapsed = time.perf_counter() - t0
    return {
        "ticks": sim.tick_id,
        "winner": sim.winner or "DRAW",
        "player_hp": sim.player.hp,
        "ai_hp": sim.ai.hp,
        "elapsed_s": elapsed,
        "tick_latencies_ms": tick_latencies,
        "planner_latencies_ms": planner_latencies,
    }


def run_evaluation(
    n_matches: int = 50,
    seed_start: int = 0,
    tier: AITier = AITier.T2_FULL_ADAPTIVE,
    max_ticks: int = 20000,
    db_path: Path | None = None,
    game_cfg=None,
    ai_cfg=None,
    replay_dir: Path | None = None,
    candidate_model_path: Path | None = None,
) -> EvaluationResult:
    """Run a deterministic evaluation batch and return all metrics.

    Args:
        n_matches: Number of matches to run.
        seed_start: First RNG seed (incremented per match).
        tier: AI tier to evaluate.
        max_ticks: Maximum ticks per match before draw.
        db_path: Database path for T1/T2 (temp if None).
        game_cfg: GameConfig override (loaded from defaults if None).
        ai_cfg: AIConfig override (loaded from defaults if None).
        replay_dir: Directory of replay files to verify (optional).

    Returns:
        EvaluationResult with all metrics computed.
    """
    if game_cfg is None or ai_cfg is None:
        _game, _ai, _ = load_config()
        game_cfg = game_cfg or _game
        ai_cfg = ai_cfg or _ai

    db = None
    match_ids: list[str] = []

    if tier != AITier.T0_BASELINE:
        if db_path is None:
            import tempfile
            db_path = Path(tempfile.mkdtemp()) / "eval.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        if candidate_model_path is not None:
            db.execute_safe(
                """INSERT INTO model_registry
                   (version, model_path, model_type, is_active, metadata)
                   VALUES ('candidate', ?, 'random_forest', 1, '{}')""",
                (str(candidate_model_path),),
            )

    results: list[dict] = []
    for i in range(n_matches):
        seed = seed_start + i
        if tier == AITier.T0_BASELINE:
            r = _run_baseline_match_instrumented(game_cfg, seed, max_ticks)
        else:
            mid = f"eval-{tier.name}-{i:06d}"
            match_ids.append(mid)
            r = _run_planner_match_instrumented(
                game_cfg, ai_cfg, db, seed, tier, max_ticks, mid,
                force_sklearn=(candidate_model_path is not None),
            )
        results.append(r)

    # Compute all metrics
    win_rate = compute_win_rate(results)
    match_length = compute_match_length(results)
    damage = compute_damage(results)
    performance = compute_performance(results)

    prediction = None
    planner = None
    if db is not None and match_ids:
        prediction = compute_prediction_accuracy(db, match_ids)
        planner = compute_planner_success(db, match_ids)
        db.close()

    replay_verification = None
    if replay_dir is not None:
        replay_verification = compute_replay_verification(replay_dir)

    return EvaluationResult(
        tier=tier.name,
        match_count=n_matches,
        seed_start=seed_start,
        win_rate=win_rate,
        match_length=match_length,
        damage=damage,
        prediction=prediction,
        planner=planner,
        performance=performance,
        replay_verification=replay_verification,
        raw_results=results,
    )
