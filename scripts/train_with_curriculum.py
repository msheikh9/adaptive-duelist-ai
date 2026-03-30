#!/usr/bin/env python3
"""Curriculum-driven training: evaluate → analyze weaknesses → self-play → retrain → promote.

Usage:
    python -m scripts.train_with_curriculum [--matches N] [--auto-promote] [--db PATH]

Exit codes:
    0  promoted, retrain not needed, or auto-promote disabled
    1  retrained but regression detected (not promoted)
    2  insufficient data to retrain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    from config.config_loader import load_config
    from data.db import DEFAULT_DB_PATH

    parser = argparse.ArgumentParser(
        description="Curriculum-driven training: evaluate → self-play → retrain → promote"
    )
    parser.add_argument("--matches", type=int, default=50,
                        help="Total self-play matches to generate (default: 50)")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Promote model when regression gates pass")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="Database path")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline JSON for regression check")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed start for self-play and evaluation (default: 0)")
    parser.add_argument("--max-ticks", type=int, default=5000,
                        help="Tick cap per match (default: 5000)")
    args = parser.parse_args()

    from ai.training.curriculum import build_curriculum
    from ai.training.pipeline import TrainingPipeline
    from ai.training.self_play_runner import run_self_play
    from ai.layers.tactical_planner import AITier
    from evaluation.match_runner import run_evaluation
    from evaluation.weakness_analyzer import analyze_weaknesses

    game_cfg, ai_cfg, _ = load_config()

    # Step 1: Evaluate current model
    log.info("Evaluating current model to detect weaknesses...")
    eval_n = max(5, args.matches // 10)
    current_eval = run_evaluation(
        n_matches=eval_n,
        seed_start=args.seed,
        tier=AITier.T2_FULL_ADAPTIVE,
        max_ticks=args.max_ticks,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
    )

    # Step 2: Analyze weaknesses
    from data.db import Database
    from data.migrations.migration_runner import run_migrations
    db = Database(args.db)
    db.connect()
    run_migrations(db)
    weakness = analyze_weaknesses(db, current_eval)
    db.close()

    print(f"\nDetected weaknesses:")
    for scenario in weakness.high_failure_scenarios:
        print(f"  - {scenario}")
    if weakness.weak_prediction_labels:
        print(f"  Weak prediction labels: {weakness.weak_prediction_labels}")
    if weakness.weak_tactical_modes:
        print(f"  Weak tactical modes: {weakness.weak_tactical_modes}")
    if weakness.weak_spacing_zones:
        print(f"  Weak spacing zones: {weakness.weak_spacing_zones}")

    # Step 3: Build curriculum
    curriculum = build_curriculum(weakness, args.matches)

    print(f"\nCurriculum plan:")
    print(f"  Profiles: {[p.value for p in curriculum.profiles]}")
    print(f"  Allocation: {curriculum.match_allocation}")
    print(f"  Focus: {curriculum.focus_targets}")

    # Step 4: Run self-play using curriculum
    log.info("Running %d self-play matches with curriculum...", args.matches)
    sp_result = run_self_play(
        n_matches=args.matches,
        profiles=curriculum.profiles,
        seed_start=args.seed,
        db_path=args.db,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
        max_ticks=args.max_ticks,
        curriculum=curriculum,
    )

    print(f"\nSelf-play summary:")
    print(f"  Matches run:              {sp_result.matches_run}")
    print(f"  Semantic events inserted: {sp_result.semantic_events_inserted}")

    # Step 5–8: Retrain and promote
    pipeline = TrainingPipeline(
        db_path=args.db,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
        baseline_path=args.baseline,
    )
    result = pipeline.run_pipeline(auto_promote=args.auto_promote)

    print(f"\nTraining pipeline result:")
    print(f"  Retrain needed: {result.retrain_needed}")
    print(f"  Retrained:      {result.retrained}")
    print(f"  Version:        {result.version or 'N/A'}")
    if result.holdout_accuracy is not None:
        print(f"  Holdout acc:    {result.holdout_accuracy:.3f}")
    print(f"  Promoted:       {result.promoted}")
    print(f"  Reason:         {result.promotion_reason}")

    if result.regression_report is not None:
        print(f"  Regression passed: {result.regression_report.passed}")
        for c in result.regression_report.failures:
            print(f"    FAIL {c.metric}: {c.detail}")

    # Exit codes mirror train_and_promote.py
    if not result.retrained and not result.retrain_needed:
        sys.exit(0)
    if result.promotion_reason == "auto_promote_disabled":
        sys.exit(0)
    if result.promoted:
        sys.exit(0)
    if result.promotion_reason == "regression_detected":
        sys.exit(1)
    sys.exit(2)


if __name__ == "__main__":
    main()
