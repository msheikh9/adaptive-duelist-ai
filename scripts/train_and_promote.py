#!/usr/bin/env python3
"""Run training pipeline: check → retrain → evaluate → (optionally) promote.

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

    parser = argparse.ArgumentParser(description="Training pipeline: retrain and promote")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--source-filter", choices=["human", "self_play"], default=None,
                        help="Filter training data by source (default: all)")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Promote model when regression gates pass")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline JSON for regression check")
    parser.add_argument("--eval-matches", type=int, default=10,
                        help="Matches for candidate evaluation (default: 10)")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="Seed for candidate evaluation (default: 42)")
    args = parser.parse_args()

    from ai.training.pipeline import TrainingPipeline

    game_cfg, ai_cfg, _ = load_config()
    pipeline = TrainingPipeline(
        db_path=args.db,
        game_cfg=game_cfg,
        ai_cfg=ai_cfg,
        baseline_path=args.baseline,
    )

    result = pipeline.run_pipeline(
        auto_promote=args.auto_promote,
        source_filter=args.source_filter,
    )

    print(f"\nPipeline result:")
    print(f"  Retrain needed:    {result.retrain_needed}")
    print(f"  Retrained:         {result.retrained}")
    print(f"  Version:           {result.version or 'N/A'}")
    if result.holdout_accuracy is not None:
        print(f"  Holdout accuracy:  {result.holdout_accuracy:.3f}")
    print(f"  Promoted:          {result.promoted}")
    print(f"  Reason:            {result.promotion_reason}")

    if result.regression_report is not None:
        print(f"  Regression passed: {result.regression_report.passed}")
        for c in result.regression_report.failures:
            print(f"    FAIL {c.metric}: {c.detail}")

    if not result.retrained and not result.retrain_needed:
        sys.exit(0)
    if result.promotion_reason == "auto_promote_disabled":
        sys.exit(0)
    if result.promoted:
        sys.exit(0)
    if result.promotion_reason == "regression_detected":
        sys.exit(1)
    # insufficient data
    sys.exit(2)


if __name__ == "__main__":
    main()
