"""Create a frozen baseline artifact from an evaluation run.

Usage:
  python3 scripts/create_baseline.py --tier 2 --matches 50 --seed 0
  python3 scripts/create_baseline.py --tier 0 --matches 100 --tag v1.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ai.layers.tactical_planner import AITier
from config.config_loader import config_hash as compute_config_hash
from evaluation.baselines import save_baseline
from evaluation.match_runner import run_evaluation
from evaluation.regression_checker import load_eval_defaults

CONFIG_DIR = PROJECT_ROOT / "config"


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def main() -> None:
    eval_defaults = load_eval_defaults()

    parser = argparse.ArgumentParser(
        description="Create a baseline artifact from evaluation")
    parser.add_argument("--tier", type=int, default=2, choices=[0, 1, 2],
                        help="AI tier to evaluate")
    parser.add_argument("--matches", type=int, default=eval_defaults["matches"],
                        help="Number of matches")
    parser.add_argument("--seed", type=int, default=eval_defaults["seed_start"],
                        help="Starting seed")
    parser.add_argument("--max-ticks", type=int, default=eval_defaults["max_ticks"],
                        help="Max ticks per match")
    parser.add_argument("--tag", default="",
                        help="Optional tag for the baseline filename")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: baselines/)")
    args = parser.parse_args()

    tier = AITier(args.tier)
    git_sha = _get_git_sha()

    # Compute config hash from game + AI config files
    cfg_hash = ""
    game_yaml = CONFIG_DIR / "game_config.yaml"
    ai_yaml = CONFIG_DIR / "ai_config.yaml"
    if game_yaml.exists() and ai_yaml.exists():
        cfg_hash = compute_config_hash(game_yaml) + ":" + compute_config_hash(ai_yaml)
    elif game_yaml.exists():
        cfg_hash = compute_config_hash(game_yaml)

    print(f"Running {args.matches} matches at tier {tier.name} (seed={args.seed})...")
    result = run_evaluation(
        n_matches=args.matches,
        seed_start=args.seed,
        tier=tier,
        max_ticks=args.max_ticks,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    path = save_baseline(
        result, tag=args.tag, directory=output_dir,
        git_sha=git_sha, config_hash=cfg_hash,
    )

    print(f"\nBaseline saved: {path}")
    print(f"  tier={result.tier}  matches={result.match_count}  "
          f"ai_win_rate={result.win_rate.ai_win_rate:.1%}  "
          f"git={git_sha or 'unknown'}")


if __name__ == "__main__":
    main()
