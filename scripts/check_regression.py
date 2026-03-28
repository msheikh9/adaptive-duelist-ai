"""Regression check with non-zero exit on failure.

Runs an evaluation, compares against the stored baseline for the tier,
and exits with code 0 (pass) or 1 (regression detected). Designed for
CI pipelines and pre-release gates.

Usage:
  python3 scripts/check_regression.py --tier 2 --matches 50
  python3 scripts/check_regression.py --tier 0 --baseline baselines/baseline_t0_baseline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ai.layers.tactical_planner import AITier
from evaluation.baselines import find_baseline
from evaluation.match_runner import run_evaluation
from evaluation.regression_checker import (
    RegressionReport,
    ThresholdConfig,
    check_regression,
    load_eval_defaults,
    load_threshold_config,
)


def _print_report(report: RegressionReport) -> None:
    verdict = "PASS" if report.passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"  Regression Check: {report.tier}  [{verdict}]")
    print(f"  Baseline: {report.baseline_path}")
    print(f"{'=' * 60}")

    for c in report.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.metric:30s}  "
              f"baseline={c.baseline_value:.4f}  "
              f"current={c.current_value:.4f}  "
              f"{c.detail}")

    if report.failures:
        print(f"\n  {len(report.failures)} regression(s) detected!")
    else:
        print(f"\n  All {len(report.checks)} checks passed.")
    print(f"{'=' * 60}")


def main() -> None:
    eval_defaults = load_eval_defaults()

    parser = argparse.ArgumentParser(
        description="Check for AI regressions against baseline")
    parser.add_argument("--tier", type=int, default=2, choices=[0, 1, 2],
                        help="AI tier to evaluate")
    parser.add_argument("--matches", type=int, default=eval_defaults["matches"],
                        help="Number of matches to run")
    parser.add_argument("--seed", type=int, default=eval_defaults["seed_start"],
                        help="Starting seed")
    parser.add_argument("--max-ticks", type=int, default=eval_defaults["max_ticks"],
                        help="Max ticks per match")
    parser.add_argument("--baseline", default=None,
                        help="Path to baseline file (default: auto-find)")
    parser.add_argument("--baselines-dir", default=None,
                        help="Directory to search for baselines")
    parser.add_argument("--eval-config", default=None,
                        help="Path to eval_config.yaml for custom thresholds")
    args = parser.parse_args()

    tier = AITier(args.tier)

    # Find baseline
    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        baseline_path = find_baseline(
            tier.name,
            directory=Path(args.baselines_dir) if args.baselines_dir else None,
        )

    if baseline_path is None or not baseline_path.exists():
        print(f"Error: no baseline found for tier {tier.name}", file=sys.stderr)
        print("Run scripts/create_baseline.py first.", file=sys.stderr)
        sys.exit(2)

    # Run evaluation
    print(f"Running {args.matches} matches at tier {tier.name}...")
    result = run_evaluation(
        n_matches=args.matches,
        seed_start=args.seed,
        tier=tier,
        max_ticks=args.max_ticks,
    )

    # Check regression — load thresholds from eval_config.yaml
    config_path = Path(args.eval_config) if args.eval_config else None
    thresholds = load_threshold_config(config_path)
    report = check_regression(result, baseline_path, thresholds)
    _print_report(report)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
