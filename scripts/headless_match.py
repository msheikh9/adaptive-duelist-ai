"""Run headless matches for testing and benchmarking.

Usage: python3 scripts/headless_match.py [--count N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from tests.fixtures.headless_engine import HeadlessMatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless AI vs AI matches")
    parser.add_argument("--count", type=int, default=5, help="Number of matches")
    parser.add_argument("--seed", type=int, default=42, help="Starting RNG seed")
    parser.add_argument("--max-ticks", type=int, default=10000, help="Max ticks per match")
    args = parser.parse_args()

    game_cfg, _, _ = load_config()

    wins = {"PLAYER": 0, "AI": 0}
    total_ticks = 0

    start = time.perf_counter()

    for i in range(args.count):
        seed = args.seed + i
        match = HeadlessMatch(game_cfg, rng_seed=seed)
        match.run_until_end(max_ticks=args.max_ticks)

        winner = match.state.winner or "DRAW"
        ticks = match.state.tick_id
        total_ticks += ticks

        p_hp = match.state.player.hp
        ai_hp = match.state.ai.hp
        hits = len(match.events)

        print(f"Match {i+1:3d} | seed={seed:6d} | ticks={ticks:5d} | "
              f"winner={winner:6s} | P_hp={p_hp:3d} AI_hp={ai_hp:3d} | "
              f"events={hits:3d}")

        if winner in wins:
            wins[winner] += 1

    elapsed = time.perf_counter() - start

    print(f"\n--- Summary ---")
    print(f"Matches: {args.count}")
    print(f"AI wins: {wins['AI']}, Player(baseline) wins: {wins['PLAYER']}")
    print(f"Total ticks: {total_ticks:,}")
    print(f"Elapsed: {elapsed:.2f}s ({total_ticks / elapsed:,.0f} ticks/sec)")


if __name__ == "__main__":
    main()
