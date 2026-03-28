"""One-shot AI evaluation CLI.

Runs a deterministic batch of matches for a given tier, prints metrics,
and optionally exports the result as JSON.

Usage:
  python3 scripts/evaluate_ai.py --tier 2 --matches 50 --seed 0
  python3 scripts/evaluate_ai.py --tier 0 --matches 100 -o eval_t0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ai.layers.tactical_planner import AITier
from evaluation.match_runner import run_evaluation
from evaluation.regression_checker import load_eval_defaults


def _print_result(result) -> None:
    wr = result.win_rate
    ml = result.match_length
    dm = result.damage
    pf = result.performance

    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {result.tier}  |  {result.match_count} matches  |  seed={result.seed_start}")
    print(f"{'=' * 60}")
    print(f"\n  Win Rate")
    print(f"    AI: {wr.ai_win_rate:.1%}  Player: {wr.player_win_rate:.1%}  Draw: {wr.draw_rate:.1%}")
    print(f"    ({wr.ai_wins}W / {wr.player_wins}L / {wr.draws}D)")
    print(f"\n  Match Length")
    print(f"    avg={ml.avg_ticks:.0f}  median={ml.median_ticks:.0f}  "
          f"min={ml.min_ticks}  max={ml.max_ticks}")
    print(f"\n  Damage")
    print(f"    AI HP remaining:     {dm.avg_ai_hp_remaining:.1f}")
    print(f"    Player HP remaining: {dm.avg_player_hp_remaining:.1f}")
    print(f"    HP differential:     {dm.avg_hp_differential:+.1f}")
    print(f"\n  Performance")
    print(f"    p95 tick:    {pf.p95_tick_ms:.3f} ms")
    print(f"    p95 planner: {pf.p95_planner_ms:.3f} ms")
    print(f"    throughput:  {pf.avg_ticks_per_sec:,.0f} ticks/s")

    if result.prediction:
        p = result.prediction
        print(f"\n  Prediction")
        print(f"    top-1: {p.top1_accuracy:.1%} ({p.top1_correct}/{p.total_predictions})")
        print(f"    top-2: {p.top2_accuracy:.1%} ({p.top2_correct}/{p.total_predictions})")

    if result.planner:
        pl = result.planner
        print(f"\n  Planner")
        print(f"    overall success: {pl.overall_success_rate:.1%} "
              f"({pl.total_decisions_with_outcome} decisions)")
        for mode, rate in sorted(pl.mode_success_rates.items()):
            print(f"    {mode:30s} {rate:.1%}")

    if result.replay_verification:
        rv = result.replay_verification
        print(f"\n  Replay Verification")
        print(f"    pass rate: {rv.pass_rate:.1%} ({rv.passed}/{rv.total_replays})")

    print(f"\n{'=' * 60}")


def main() -> None:
    eval_defaults = load_eval_defaults()

    parser = argparse.ArgumentParser(description="Evaluate AI quality")
    parser.add_argument("--tier", type=int, default=2, choices=[0, 1, 2],
                        help="AI tier (0=baseline, 1=markov, 2=full)")
    parser.add_argument("--matches", type=int, default=eval_defaults["matches"],
                        help="Number of matches")
    parser.add_argument("--seed", type=int, default=eval_defaults["seed_start"],
                        help="Starting seed")
    parser.add_argument("--max-ticks", type=int, default=eval_defaults["max_ticks"],
                        help="Max ticks per match")
    parser.add_argument("--replay-dir", default=None,
                        help="Replay directory to verify (optional)")
    parser.add_argument("--output", "-o", default=None,
                        help="Export results to JSON file")
    args = parser.parse_args()

    tier = AITier(args.tier)
    result = run_evaluation(
        n_matches=args.matches,
        seed_start=args.seed,
        tier=tier,
        max_ticks=args.max_ticks,
        replay_dir=Path(args.replay_dir) if args.replay_dir else None,
    )

    _print_result(result)

    if args.output:
        # Exclude raw_results for cleaner export
        d = asdict(result)
        del d["raw_results"]
        Path(args.output).write_text(json.dumps(d, indent=2, default=str) + "\n")
        print(f"\nExported to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
