"""Production-grade bulk match runner.

Runs large self-play / evaluation batches with configurable seeds,
AI tiers, and result export (CSV or JSONL).

Usage:
  python3 scripts/bulk_simulate.py --count 1000 --seed 0 --tier 2
  python3 scripts/bulk_simulate.py --count 50 --format csv -o results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
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
from game.combat.actions import FSMState
from game.state import (
    ArenaState, FighterState, MatchStatus, SimulationState, TickPhase,
)
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.layers.tactical_planner import AITier, TacticalPlanner
from tests.fixtures.headless_engine import HeadlessMatch


RESULT_FIELDS = [
    "match_index", "seed", "tier", "ticks", "winner",
    "player_hp", "ai_hp", "elapsed_s",
]


def run_baseline_match(game_cfg, seed: int, max_ticks: int) -> dict:
    """Run a T0 baseline match."""
    t0 = time.perf_counter()
    match = HeadlessMatch(game_cfg, rng_seed=seed)
    match.run_until_end(max_ticks=max_ticks)
    elapsed = time.perf_counter() - t0
    s = match.state
    return {
        "ticks": s.tick_id,
        "winner": s.winner or "DRAW",
        "player_hp": s.player.hp,
        "ai_hp": s.ai.hp,
        "elapsed_s": elapsed,
    }


def run_planner_match(
    game_cfg, ai_cfg, db: Database,
    seed: int, tier: AITier, max_ticks: int,
    match_id: str,
) -> dict:
    """Run a T1/T2 match using the tactical planner."""
    bm = BehaviorModel(db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
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

    # Insert match record for FK
    db.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, "
        "config_hash) VALUES (?, 'bulk', '2025-01-01', ?, 'bulk');",
        (match_id, seed),
    )

    bm.on_match_start(match_id)
    pe.on_match_start(match_id)
    planner.on_match_start(match_id, "bulk", seed)

    # Minimal simulation loop — AI decides, baseline drives player
    from game.entities.ai_fighter import BaselineAIController
    from game.combat.state_machine import tick_fsm
    from game.combat.physics import (
        apply_dodge_velocity, apply_velocity, clamp_to_arena, update_facing,
    )
    from game.combat.collision import HitTracker, check_hit
    from game.combat.damage import apply_hit
    from game.combat.stamina import tick_stamina

    baseline = BaselineAIController(seed)
    hit_tracker = HitTracker()
    fighter_w_sub = game_cfg.fighter.width * scale

    t0 = time.perf_counter()
    for tick in range(max_ticks):
        if sim.match_status == MatchStatus.ENDED:
            break
        sim.tick_id = tick
        sim.set_phase(TickPhase.SIMULATE)

        # Player uses baseline AI
        player_commit = baseline.decide(sim.player, sim, game_cfg)
        # AI uses planner
        ai_commit = planner.decide(sim.ai, sim, game_cfg)

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

    elapsed = time.perf_counter() - t0
    return {
        "ticks": sim.tick_id,
        "winner": sim.winner or "DRAW",
        "player_hp": sim.player.hp,
        "ai_hp": sim.ai.hp,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk headless match runner")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of matches")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--tier", type=int, default=0, choices=[0, 1, 2],
                        help="AI tier (0=baseline, 1=markov, 2=full)")
    parser.add_argument("--max-ticks", type=int, default=20000,
                        help="Max ticks per match")
    parser.add_argument("--format", choices=["csv", "jsonl"],
                        default="jsonl", help="Export format")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: stdout summary only)")
    parser.add_argument("--db", default=None,
                        help="DB path for T1/T2 (default: temp)")
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()
    tier = AITier(args.tier)

    db = None
    if tier != AITier.T0_BASELINE:
        import tempfile
        db_path = Path(args.db) if args.db else (
            Path(tempfile.mkdtemp()) / "bulk.db")
        db = Database(db_path)
        db.connect()
        run_migrations(db)

    results: list[dict] = []
    wins = {"PLAYER": 0, "AI": 0, "DRAW": 0}
    total_ticks = 0
    wall_start = time.perf_counter()

    for i in range(args.count):
        seed = args.seed + i
        if tier == AITier.T0_BASELINE:
            r = run_baseline_match(game_cfg, seed, args.max_ticks)
        else:
            match_id = f"bulk-{i:06d}"
            r = run_planner_match(
                game_cfg, ai_cfg, db, seed, tier, args.max_ticks, match_id)

        row = {
            "match_index": i,
            "seed": seed,
            "tier": tier.name,
            **r,
        }
        results.append(row)
        total_ticks += r["ticks"]
        wins[r["winner"]] = wins.get(r["winner"], 0) + 1

        if (i + 1) % max(1, args.count // 10) == 0 or i == args.count - 1:
            print(f"  [{i+1}/{args.count}] seed={seed} "
                  f"ticks={r['ticks']:5d} winner={r['winner']:6s} "
                  f"P_hp={r['player_hp']:3d} AI_hp={r['ai_hp']:3d} "
                  f"({r['elapsed_s']:.3f}s)")

    wall_elapsed = time.perf_counter() - wall_start

    # --- Export ---
    if args.output:
        out_path = Path(args.output)
        if args.format == "csv":
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                for row in results:
                    writer.writerow({k: row.get(k) for k in RESULT_FIELDS})
        else:
            with open(out_path, "w") as f:
                for row in results:
                    f.write(json.dumps(row, default=str) + "\n")
        print(f"\nResults exported to {out_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Bulk simulation: {args.count} matches, tier={tier.name}")
    print(f"  Total ticks:  {total_ticks:,}")
    print(f"  Wall time:    {wall_elapsed:.2f}s "
          f"({total_ticks / wall_elapsed:,.0f} ticks/s)")
    print(f"  AI wins: {wins.get('AI', 0)}  "
          f"Player wins: {wins.get('PLAYER', 0)}  "
          f"Draws: {wins.get('DRAW', 0)}")
    print(f"{'=' * 60}")

    if db:
        db.close()


if __name__ == "__main__":
    main()
