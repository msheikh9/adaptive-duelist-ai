"""Repeatable benchmark suite for core systems.

Benchmarks:
  - 1 headless match
  - N headless matches (default 100)
  - Replay verification batch
  - Analytics report generation batch

Usage: python3 scripts/benchmark.py [--matches N] [--seed S] [--output FILE]
"""

from __future__ import annotations

import argparse
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
from game.state import MatchStatus
from tests.fixtures.headless_engine import HeadlessMatch


def benchmark_single_match(game_cfg, seed: int) -> dict:
    """Benchmark a single headless match."""
    t0 = time.perf_counter()
    match = HeadlessMatch(game_cfg, rng_seed=seed)
    match.run_until_end(max_ticks=20000)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "ticks": match.state.tick_id,
        "ticks_per_sec": match.state.tick_id / elapsed if elapsed > 0 else 0,
        "winner": match.state.winner or "DRAW",
    }


def benchmark_batch_matches(game_cfg, n: int, seed: int) -> dict:
    """Benchmark N headless matches."""
    total_ticks = 0
    wins = {"PLAYER": 0, "AI": 0, "DRAW": 0}
    t0 = time.perf_counter()
    for i in range(n):
        match = HeadlessMatch(game_cfg, rng_seed=seed + i)
        match.run_until_end(max_ticks=20000)
        total_ticks += match.state.tick_id
        w = match.state.winner or "DRAW"
        wins[w] = wins.get(w, 0) + 1
    elapsed = time.perf_counter() - t0
    return {
        "matches": n,
        "elapsed_s": elapsed,
        "total_ticks": total_ticks,
        "ticks_per_sec": total_ticks / elapsed if elapsed > 0 else 0,
        "avg_ticks_per_match": total_ticks / n if n > 0 else 0,
        "wins": wins,
    }


def benchmark_replay_verification(replay_dir: Path) -> dict:
    """Benchmark verifying all replay files in a directory."""
    from replay.replay_player import load_replay, verify_replay
    from config.config_loader import load_config as _lc

    game_cfg, _, _ = _lc()
    files = sorted(replay_dir.glob("*.replay"))
    if not files:
        return {"count": 0, "elapsed_s": 0, "note": "no replay files found"}

    passed = 0
    failed = 0
    t0 = time.perf_counter()
    for f in files:
        try:
            replay = load_replay(f)
            result = verify_replay(replay, game_cfg)
            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    elapsed = time.perf_counter() - t0
    return {
        "count": len(files),
        "passed": passed,
        "failed": failed,
        "elapsed_s": elapsed,
        "per_replay_ms": (elapsed / len(files) * 1000) if files else 0,
    }


def benchmark_report_generation(db, match_ids: list[str]) -> dict:
    """Benchmark analytics report generation."""
    from analytics.report_builder import build_match_report

    if not match_ids:
        return {"count": 0, "elapsed_s": 0, "note": "no matches to report on"}

    t0 = time.perf_counter()
    for mid in match_ids:
        build_match_report(db, mid)
    elapsed = time.perf_counter() - t0
    return {
        "count": len(match_ids),
        "elapsed_s": elapsed,
        "per_report_ms": (elapsed / len(match_ids) * 1000) if match_ids else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark suite")
    parser.add_argument("--matches", type=int, default=100,
                        help="Batch match count")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", "-o", default=None,
                        help="Export results to JSON file")
    parser.add_argument("--replay-dir", default="replays",
                        help="Replay directory for verification benchmark")
    parser.add_argument("--db", default=None,
                        help="DB path for report benchmark (default: temp)")
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()
    results: dict[str, dict] = {}

    # --- Single match ---
    print("Benchmarking: single match...")
    results["single_match"] = benchmark_single_match(game_cfg, args.seed)
    r = results["single_match"]
    print(f"  {r['ticks']} ticks in {r['elapsed_s']:.3f}s "
          f"({r['ticks_per_sec']:,.0f} ticks/s)  winner={r['winner']}")

    # --- Batch matches ---
    print(f"\nBenchmarking: {args.matches} matches...")
    results["batch_matches"] = benchmark_batch_matches(
        game_cfg, args.matches, args.seed)
    r = results["batch_matches"]
    print(f"  {r['total_ticks']:,} total ticks in {r['elapsed_s']:.3f}s "
          f"({r['ticks_per_sec']:,.0f} ticks/s)")
    print(f"  Avg {r['avg_ticks_per_match']:.0f} ticks/match  "
          f"AI={r['wins'].get('AI',0)} Player={r['wins'].get('PLAYER',0)}")

    # --- Replay verification ---
    replay_dir = Path(args.replay_dir)
    if replay_dir.is_dir():
        print(f"\nBenchmarking: replay verification in {replay_dir}...")
        results["replay_verification"] = benchmark_replay_verification(replay_dir)
        r = results["replay_verification"]
        if r["count"] > 0:
            print(f"  {r['count']} replays: {r['passed']} passed, "
                  f"{r['failed']} failed in {r['elapsed_s']:.3f}s "
                  f"({r['per_replay_ms']:.1f}ms/replay)")
        else:
            print("  No replay files found.")
    else:
        print(f"\nSkipping replay verification: {replay_dir} not found.")
        results["replay_verification"] = {"skipped": True}

    # --- Report generation ---
    import tempfile
    db_path = Path(args.db) if args.db else Path(tempfile.mkdtemp()) / "bench.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)

    match_ids = [r["match_id"] for r in
                 db.fetchall("SELECT match_id FROM matches LIMIT 20;")]
    if match_ids:
        print(f"\nBenchmarking: report generation ({len(match_ids)} matches)...")
        results["report_generation"] = benchmark_report_generation(db, match_ids)
        r = results["report_generation"]
        print(f"  {r['count']} reports in {r['elapsed_s']:.3f}s "
              f"({r['per_report_ms']:.1f}ms/report)")
    else:
        print("\nSkipping report benchmark: no matches in DB.")
        results["report_generation"] = {"skipped": True}

    db.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Benchmark complete.")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"Results exported to {args.output}")


if __name__ == "__main__":
    main()
