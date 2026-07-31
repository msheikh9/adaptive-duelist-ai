"""Deterministic batch evaluator for AI tiers.

Runs N matches at fixed seeds per tier through the *same* `step_simulation` the
live engine uses, and the *same* `EventRouter`, so an evaluated match is the
match the game would play. Before that was true this module carried its own
partial copy of the tick loop, which silently dropped gravity, guard regen, all
three cooldown timers and the projectile subsystem, and never emitted semantic
events — so evaluated AI could use heavy attack and dodge at most once per match
and prediction accuracy was structurally unmeasurable.

All three tiers now differ *only* in the AI policy:

    T0  UniformRandomAIController — uniform over legal commitments
    T1  TacticalPlanner(T1_MARKOV_ONLY)
    T2  TacticalPlanner(T2_FULL_ADAPTIVE)

The opponent on the player side is `BaselineAIController(seed)` in every tier, so
a cross-tier comparison is a controlled one.
"""

from __future__ import annotations

import time
from pathlib import Path

from ai.layers.tactical_planner import AITier
from config.config_loader import load_config
from data.db import Database
from data.logger import GameLogger
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
from data.tick_snapshot import TickSnapshot
from game.combat.actions import Actor, CombatCommitment
from game.entities.ai_fighter import (
    BaselineAIController,
    UniformRandomAIController,
)
from game.entities.fighter import attempt_commitment
from game.semantic_events import EventRouter
from game.simulation_step import SimulationContext, step_simulation
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    SimulationState,
    TickPhase,
)
from replay.recorder import ReplayRecorder

# Offset applied to the AI-side RNG seed so the AI and its opponent do not run
# correlated random streams (identical seeds would make their decision-interval
# countdowns line up).
_AI_SEED_OFFSET = 7919


def _fresh_state(game_cfg, seed: int) -> SimulationState:
    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    return SimulationState(
        tick_id=0, rng_seed=seed,
        player=FighterState(
            x=arena.width_sub // 3, y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp,
            stamina=game_cfg.fighter.max_stamina, facing=1,
        ),
        ai=FighterState(
            x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp,
            stamina=game_cfg.fighter.max_stamina, facing=-1,
        ),
        arena=arena, match_status=MatchStatus.ACTIVE,
    )


def _run_match(
    game_cfg,
    ai_cfg,
    db: Database | None,
    seed: int,
    tier: AITier,
    max_ticks: int,
    match_id: str,
    force_sklearn: bool = False,
    record_to: Path | None = None,
) -> dict:
    """Run one evaluation match and return its result row.

    Collects per-tick and per-planner-call latency samples so p95 figures are
    measured rather than estimated.

    `record_to` writes a replay file, recorded exactly as the live engine records
    one: every entered commitment into the Layer A stream, a Layer B checksum
    during the LOG phase. This is the only headless way to produce a replay —
    before it existed replays came solely from interactive play, so replay
    verification could not be exercised without a human at the keyboard.
    """
    # --- AI side: the thing under test ---
    planner = None
    prediction_engine = None
    behavior_model = None
    random_ai = None

    if tier == AITier.T0_BASELINE:
        random_ai = UniformRandomAIController(seed + _AI_SEED_OFFSET)
    else:
        from ai.layers.behavior_model import BehaviorModel
        from ai.layers.prediction_engine import PredictionEngine
        from ai.layers.tactical_planner import TacticalPlanner

        behavior_model = BehaviorModel(db, ai_cfg, game_cfg)
        behavior_model.load_profile()
        prediction_engine = PredictionEngine(db, behavior_model, ai_cfg, game_cfg)
        # The engine loads the active model unconditionally at construction, so
        # do the same here for parity. Note this is a no-op when the evaluation
        # database is a fresh temp file: its model_registry is empty, so nothing
        # loads and the "ensemble" runs Markov-only. Pass candidate_model_path to
        # run_evaluation to actually exercise the Random Forest.
        prediction_engine.try_load_sklearn(force=True)
        planner = TacticalPlanner(
            db, prediction_engine, ai_cfg, game_cfg, tier,
            behavior_model=behavior_model,
        )

    # --- Player side: identical opponent for every tier ---
    opponent = BaselineAIController(seed)

    state = _fresh_state(game_cfg, seed)
    ctx = SimulationContext()

    # Recorder must see the pristine initial state before any tick runs.
    recorder = None
    snapshots: list[TickSnapshot] = []
    if record_to is not None:
        recorder = ReplayRecorder(state, game_cfg, out_dir=record_to)

    sinks = []
    if db is not None:
        logger = GameLogger(db)
        logger.clear()
        sinks.append(logger.record_event)
        db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
            "config_hash) VALUES (?, 'eval', '2025-01-01', ?, 'eval');",
            (match_id, seed),
        )
    else:
        logger = None
    if behavior_model is not None:
        sinks.append(behavior_model.on_event)
    if prediction_engine is not None:
        sinks.append(prediction_engine.on_event)

    router = EventRouter(game_cfg, sinks=sinks, planner=planner)

    if behavior_model is not None:
        behavior_model.on_match_start(match_id)
    if prediction_engine is not None:
        prediction_engine.on_match_start(match_id)
    if planner is not None:
        planner.on_match_start(match_id, "eval", seed)

    def decide_player() -> list[CombatCommitment]:
        entered = opponent.decide(state.player, state, game_cfg)
        return [entered] if entered is not None else []

    planner_latencies: list[float] = []

    def decide_ai() -> list[CombatCommitment]:
        entered: list[CombatCommitment] = []
        t_plan = time.perf_counter()
        if planner is not None:
            c = planner.decide(state.ai, state, game_cfg)
            # Extras (e.g. a guard release) were entered before `c`.
            entered.extend(planner.take_extra_commitments())
        else:
            c = random_ai.decide(state.ai, state, game_cfg)
        planner_latencies.append((time.perf_counter() - t_plan) * 1000)
        if c is not None:
            entered.append(c)

        # Mirror the engine's periodic AI shoot policy so evaluated behaviour
        # matches played behaviour.
        if (state.tick_id % 180 == 90
                and state.ai.is_free
                and state.ai.shoot_cooldown == 0):
            if attempt_commitment(
                    state.ai, CombatCommitment.SHOOT_INSTANT, game_cfg):
                entered.append(CombatCommitment.SHOOT_INSTANT)
        return entered

    tick_latencies: list[float] = []
    t0 = time.perf_counter()
    for tick in range(max_ticks):
        if state.match_status == MatchStatus.ENDED:
            break
        state.tick_id = tick
        state.set_phase(TickPhase.SIMULATE)

        t_tick = time.perf_counter()
        fx = step_simulation(state, game_cfg, ctx, decide_player, decide_ai)
        router.route(state, match_id, ctx, fx)
        if prediction_engine is not None:
            prediction_engine.on_tick(state.tick_id)
        tick_latencies.append((time.perf_counter() - t_tick) * 1000)

        if recorder is not None:
            for actor, commitments in (
                (Actor.PLAYER, fx.player_commitments),
                (Actor.AI, fx.ai_commitments),
            ):
                for commitment in commitments:
                    recorder.record_commitment(state.tick_id, actor, commitment)
            # LOG phase: same point in the tick the engine checksums at.
            state.set_phase(TickPhase.LOG)
            snapshots.append(TickSnapshot.from_state(state))
            recorder.record_checksum_if_due(state)

    elapsed = time.perf_counter() - t0

    if recorder is not None:
        record_to.mkdir(parents=True, exist_ok=True)
        recorder.finalize(state, snapshots, match_id)

    if planner is not None:
        planner.on_match_end()
    if prediction_engine is not None:
        prediction_engine.on_match_end()
    if logger is not None:
        logger.flush_events()

    return {
        "ticks": state.tick_id,
        "winner": state.winner or "DRAW",
        "player_hp": state.player.hp,
        "ai_hp": state.ai.hp,
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
    record_replays_to: Path | None = None,
) -> EvaluationResult:
    """Run a deterministic evaluation batch and return all metrics.

    Args:
        n_matches: Number of matches to run.
        seed_start: First RNG seed (incremented per match).
        tier: AI tier to evaluate.
        max_ticks: Maximum ticks per match before draw.
        db_path: Database path (temp if None).
        game_cfg: GameConfig override (loaded from defaults if None).
        ai_cfg: AIConfig override (loaded from defaults if None).
        replay_dir: Directory of replay files to verify (optional).
        candidate_model_path: Evaluate this sklearn model instead of the
            registry's active one.

    Returns:
        EvaluationResult with all metrics computed.
    """
    if game_cfg is None or ai_cfg is None:
        _game, _ai, _ = load_config()
        game_cfg = game_cfg or _game
        ai_cfg = ai_cfg or _ai

    # Every tier gets a database now: T0 has no predictor, but it still emits the
    # event stream, and logging it keeps the tiers on one code path.
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
    match_ids: list[str] = []
    for i in range(n_matches):
        mid = f"eval-{tier.name}-{i:06d}"
        match_ids.append(mid)
        results.append(_run_match(
            game_cfg, ai_cfg, db, seed_start + i, tier, max_ticks, mid,
            force_sklearn=(candidate_model_path is not None),
            record_to=record_replays_to,
        ))

    win_rate = compute_win_rate(results)
    match_length = compute_match_length(results)
    damage = compute_damage(results)
    performance = compute_performance(results)
    # T0 has no predictor and no tactical planner, so these are not applicable
    # rather than zero. Reporting 0% for a tier that never predicts would read as
    # a measured failure instead of an absent component.
    if tier == AITier.T0_BASELINE:
        prediction = None
        planner = None
    else:
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
