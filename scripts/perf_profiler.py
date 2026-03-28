"""Performance profiler for core subsystems.

Measures per-component latency across:
  - Engine tick
  - Prediction engine inference
  - Tactical planner decision
  - Analytics report generation

Usage: python3 scripts/perf_profiler.py [--ticks N] [--seed S] [--tier T]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from game.combat.actions import CombatCommitment, FSMState
from game.state import (
    ArenaState, FighterState, MatchStatus, SimulationState, TickPhase,
)
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.layers.tactical_planner import AITier, TacticalPlanner
from analytics.report_builder import build_match_report
from tests.fixtures.headless_engine import HeadlessMatch


def _percentiles(samples: list[float]) -> dict[str, float]:
    """Compute p50, p90, p95, p99, mean, max for a list of durations (ms)."""
    if not samples:
        return {"count": 0}
    ms = [s * 1000 for s in samples]
    ms.sort()
    n = len(ms)
    return {
        "count": n,
        "mean_ms": statistics.mean(ms),
        "p50_ms": ms[n // 2],
        "p90_ms": ms[int(n * 0.9)],
        "p95_ms": ms[int(n * 0.95)],
        "p99_ms": ms[int(n * 0.99)],
        "max_ms": ms[-1],
    }


def profile_engine_tick(game_cfg, n_ticks: int, seed: int) -> dict:
    """Profile raw engine tick throughput (baseline AI, no planner)."""
    match = HeadlessMatch(game_cfg, rng_seed=seed)
    samples: list[float] = []
    for _ in range(n_ticks):
        if match.state.match_status == MatchStatus.ENDED:
            break
        t0 = time.perf_counter()
        match.tick()
        samples.append(time.perf_counter() - t0)
    return _percentiles(samples)


def profile_prediction(ai_cfg, game_cfg, db, n_ticks: int, seed: int) -> dict:
    """Profile prediction engine inference latency."""
    bm = BehaviorModel(db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
    pe.on_match_start("perf-test")

    samples: list[float] = []
    for _ in range(n_ticks):
        t0 = time.perf_counter()
        pe.predict()
        samples.append(time.perf_counter() - t0)
    return _percentiles(samples)


def profile_planner(
    ai_cfg, game_cfg, db, n_ticks: int, seed: int, tier: AITier,
) -> dict:
    """Profile tactical planner decision latency."""
    bm = BehaviorModel(db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
    planner = TacticalPlanner(db, pe, ai_cfg, game_cfg, tier)

    bm.on_match_start("perf-test")
    pe.on_match_start("perf-test")
    planner.on_match_start("perf-test", "perf-session", seed)

    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    ai_state = FighterState(
        x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub,
        hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
        facing=-1,
    )
    sim = SimulationState(
        tick_id=0, rng_seed=seed,
        player=FighterState(
            x=arena.width_sub // 3, y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
            facing=1,
        ),
        ai=ai_state, arena=arena, match_status=MatchStatus.ACTIVE,
    )

    samples: list[float] = []
    for tick in range(n_ticks):
        sim.tick_id = tick
        sim.set_phase(TickPhase.SIMULATE)
        sim.ai.fsm_state = FSMState.IDLE
        sim.ai.active_commitment = None
        t0 = time.perf_counter()
        planner.decide(sim.ai, sim, game_cfg)
        samples.append(time.perf_counter() - t0)
    return _percentiles(samples)


def profile_analytics(db, match_id: str) -> dict:
    """Profile analytics report generation latency."""
    samples: list[float] = []
    for _ in range(20):
        t0 = time.perf_counter()
        build_match_report(db, match_id)
        samples.append(time.perf_counter() - t0)
    return _percentiles(samples)


def _format_stats(label: str, stats: dict) -> str:
    if stats.get("count", 0) == 0:
        return f"  {label:30s}  no data"
    return (
        f"  {label:30s}  n={stats['count']:5d}  "
        f"mean={stats['mean_ms']:.3f}ms  "
        f"p50={stats['p50_ms']:.3f}ms  "
        f"p90={stats['p90_ms']:.3f}ms  "
        f"p95={stats['p95_ms']:.3f}ms  "
        f"p99={stats['p99_ms']:.3f}ms  "
        f"max={stats['max_ms']:.3f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile core subsystems")
    parser.add_argument("--ticks", type=int, default=1000,
                        help="Ticks per component")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2],
                        help="AI tier for planner profile (1 or 2)")
    parser.add_argument("--db", default=None,
                        help="DB path (default: temp DB)")
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()
    tier = AITier(args.tier)

    if args.db:
        db_path = Path(args.db)
    else:
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "perf.db"

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    # Insert a match record for planner/analytics to reference
    db.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
        "config_hash) VALUES ('perf-test', 'perf-session', "
        "'2025-01-01', ?, 'perf');",
        (args.seed,),
    )

    print(f"Profiling with {args.ticks} ticks, seed={args.seed}, tier={tier.name}")
    print(f"{'=' * 100}")

    engine_stats = profile_engine_tick(game_cfg, args.ticks, args.seed)
    print(_format_stats("Engine tick (baseline)", engine_stats))

    pred_stats = profile_prediction(ai_cfg, game_cfg, db, args.ticks, args.seed)
    print(_format_stats("Prediction inference", pred_stats))

    planner_stats = profile_planner(
        ai_cfg, game_cfg, db, args.ticks, args.seed, tier)
    print(_format_stats(f"Planner decide ({tier.name})", planner_stats))

    analytics_stats = profile_analytics(db, "perf-test")
    print(_format_stats("Analytics report gen", analytics_stats))

    print(f"{'=' * 100}")

    tick_rate = game_cfg.simulation.tick_rate
    if engine_stats.get("count", 0) > 0:
        budget_ms = 1000.0 / tick_rate
        used_pct = engine_stats["p95_ms"] / budget_ms * 100
        print(f"\n  Tick budget: {budget_ms:.1f}ms @ {tick_rate}Hz  |  "
              f"Engine p95 uses {used_pct:.1f}% of budget")

    db.close()


if __name__ == "__main__":
    main()
