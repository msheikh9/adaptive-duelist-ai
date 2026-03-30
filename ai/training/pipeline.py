"""Training pipeline: check retrain → retrain → evaluate candidate → promote.

Uses existing infrastructure throughout:
  - build_dataset() + train_model() + register_model() for retraining
  - run_evaluation() with candidate_model_path for evaluation
  - check_regression() for promotion gating
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config_loader import AIConfig, GameConfig
    from evaluation.metrics import EvaluationResult
    from evaluation.regression_checker import RegressionReport

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    retrain_needed: bool
    retrained: bool
    version: str | None
    holdout_accuracy: float | None
    promoted: bool
    promotion_reason: str
    regression_report: "RegressionReport | None"


class TrainingPipeline:
    def __init__(
        self,
        db_path: Path,
        game_cfg: "GameConfig",
        ai_cfg: "AIConfig",
        baseline_path: Path | None = None,
    ) -> None:
        self._db_path = db_path
        self._game_cfg = game_cfg
        self._ai_cfg = ai_cfg
        self._baseline_path = baseline_path

    def _open_db(self):
        from data.db import Database
        from data.migrations.migration_runner import run_migrations
        db = Database(self._db_path)
        db.connect()
        run_migrations(db)
        return db

    def check_retrain_needed(self) -> bool:
        """True when new matches since last train >= retrain_after_every_n_matches
        AND total matches >= min_matches_to_train."""
        db = self._open_db()
        try:
            min_matches = self._ai_cfg.training.min_matches_to_train
            retrain_every = self._ai_cfg.training.retrain_after_every_n_matches

            total_row = db.fetchone("SELECT COUNT(*) as c FROM matches;")
            total = total_row["c"] if total_row else 0

            if total < min_matches:
                return False

            last_row = db.fetchone(
                "SELECT dataset_size FROM model_registry ORDER BY created_at DESC LIMIT 1;"
            )
            last_dataset_size = last_row["dataset_size"] if last_row and last_row["dataset_size"] else 0
            delta = total - last_dataset_size
            return delta >= retrain_every
        finally:
            db.close()

    def run_retrain(self, source_filter: str | None = None) -> str:
        """build_dataset → train_model → register_model. Returns version."""
        from ai.profile.player_profile import PlayerProfile, PLAYER_ID
        from ai.training.dataset_builder import build_dataset
        from ai.training.model_trainer import train_model, register_model

        db = self._open_db()
        try:
            row = db.fetchone(
                "SELECT * FROM player_profiles WHERE player_id = ? LIMIT 1;",
                (PLAYER_ID,),
            )
            profile = PlayerProfile.from_db_row(row) if row else PlayerProfile()

            X, y = build_dataset(
                db, profile,
                max_hp=self._game_cfg.fighter.max_hp,
                max_stamina=self._game_cfg.fighter.max_stamina,
                tick_rate=self._game_cfg.simulation.tick_rate,
                window_ticks=self._ai_cfg.prediction.window_ticks,
                source_filter=source_filter,
            )

            if len(X) < self._ai_cfg.training.min_samples_to_train:
                raise RuntimeError(
                    f"Insufficient samples: {len(X)} < {self._ai_cfg.training.min_samples_to_train}"
                )

            result = train_model(
                X, y,
                n_estimators=self._ai_cfg.training.random_forest_n_estimators,
                max_depth=self._ai_cfg.training.random_forest_max_depth,
                min_samples_leaf=self._ai_cfg.training.random_forest_min_samples_leaf,
                holdout_fraction=self._ai_cfg.training.holdout_fraction,
            )
            register_model(db, result)
            return result.version
        finally:
            db.close()

    def evaluate_candidate(
        self,
        candidate_version: str,
        n_matches: int = 10,
        seed_start: int = 42,
        max_ticks: int = 5000,
    ) -> "EvaluationResult":
        """Evaluate a specific candidate model without touching main DB."""
        from ai.layers.tactical_planner import AITier
        from evaluation.match_runner import run_evaluation

        # Resolve model path from registry
        db = self._open_db()
        try:
            row = db.fetchone(
                "SELECT model_path FROM model_registry WHERE version = ?;",
                (candidate_version,),
            )
            if row is None:
                raise ValueError(f"Model version not found: {candidate_version}")
            model_path = Path(row["model_path"])
        finally:
            db.close()

        return run_evaluation(
            n_matches=n_matches,
            seed_start=seed_start,
            tier=AITier.T2_FULL_ADAPTIVE,
            max_ticks=max_ticks,
            game_cfg=self._game_cfg,
            ai_cfg=self._ai_cfg,
            candidate_model_path=model_path,
        )

    def promote(self, version: str) -> None:
        """Set is_active=1 for version, 0 for all others in model_registry."""
        db = self._open_db()
        try:
            db.execute_safe(
                "UPDATE model_registry SET is_active = 0;",
            )
            db.execute_safe(
                "UPDATE model_registry SET is_active = 1 WHERE version = ?;",
                (version,),
            )
        finally:
            db.close()

    def run_pipeline(
        self,
        auto_promote: bool = False,
        source_filter: str | None = None,
    ) -> PipelineResult:
        """Full pipeline: check → retrain → evaluate → promote."""
        from evaluation.regression_checker import check_regression

        if not self.check_retrain_needed():
            return PipelineResult(
                retrain_needed=False,
                retrained=False,
                version=None,
                holdout_accuracy=None,
                promoted=False,
                promotion_reason="retrain_not_needed",
                regression_report=None,
            )

        try:
            version = self.run_retrain(source_filter=source_filter)
        except RuntimeError as e:
            log.error("Retrain failed: %s", e)
            return PipelineResult(
                retrain_needed=True,
                retrained=False,
                version=None,
                holdout_accuracy=None,
                promoted=False,
                promotion_reason=str(e),
                regression_report=None,
            )

        # Get holdout accuracy from registry
        db = self._open_db()
        try:
            row = db.fetchone(
                "SELECT eval_accuracy FROM model_registry WHERE version = ?;",
                (version,),
            )
            holdout_accuracy = float(row["eval_accuracy"]) if row and row["eval_accuracy"] is not None else None
        finally:
            db.close()

        if not auto_promote:
            return PipelineResult(
                retrain_needed=True,
                retrained=True,
                version=version,
                holdout_accuracy=holdout_accuracy,
                promoted=False,
                promotion_reason="auto_promote_disabled",
                regression_report=None,
            )

        # Determine baseline path
        baseline_path = self._baseline_path
        if baseline_path is None:
            from evaluation.baselines import find_baseline
            baseline_path = find_baseline("T2_FULL_ADAPTIVE")

        if baseline_path is None:
            self.promote(version)
            return PipelineResult(
                retrain_needed=True,
                retrained=True,
                version=version,
                holdout_accuracy=holdout_accuracy,
                promoted=True,
                promotion_reason="no_baseline",
                regression_report=None,
            )

        # Evaluate candidate
        eval_result = self.evaluate_candidate(version)
        report = check_regression(eval_result, baseline_path)

        if report.passed:
            self.promote(version)
            return PipelineResult(
                retrain_needed=True,
                retrained=True,
                version=version,
                holdout_accuracy=holdout_accuracy,
                promoted=True,
                promotion_reason="passed_eval_gates",
                regression_report=report,
            )
        else:
            return PipelineResult(
                retrain_needed=True,
                retrained=True,
                version=version,
                holdout_accuracy=holdout_accuracy,
                promoted=False,
                promotion_reason="regression_detected",
                regression_report=report,
            )

    def run_curriculum_cycle(
        self,
        n_matches: int = 50,
        auto_promote: bool = False,
        seed_start: int = 0,
        max_ticks: int = 5000,
    ) -> PipelineResult:
        """Full curriculum cycle:
        1. Evaluate current model.
        2. Analyze weaknesses.
        3. Build curriculum.
        4. Run self-play using the curriculum.
        5. Retrain.
        6. Evaluate candidate.
        7. Check regression (if baseline set).
        8. Promote if allowed.

        Reuses run_self_play(), run_retrain(), evaluate_candidate(), and
        check_regression() from the existing pipeline.
        """
        from ai.layers.tactical_planner import AITier
        from ai.training.curriculum import build_curriculum
        from ai.training.self_play_runner import run_self_play
        from evaluation.match_runner import run_evaluation
        from evaluation.weakness_analyzer import analyze_weaknesses

        # Step 1: Evaluate current model to find weaknesses
        current_eval = run_evaluation(
            n_matches=max(5, n_matches // 10),
            seed_start=seed_start,
            tier=AITier.T2_FULL_ADAPTIVE,
            max_ticks=max_ticks,
            game_cfg=self._game_cfg,
            ai_cfg=self._ai_cfg,
        )

        # Step 2 & 3: Analyze weaknesses and build curriculum
        db = self._open_db()
        try:
            weakness = analyze_weaknesses(db, current_eval)
        finally:
            db.close()

        curriculum = build_curriculum(weakness, n_matches)

        # Step 4: Run self-play using curriculum
        run_self_play(
            n_matches=n_matches,
            profiles=curriculum.profiles,
            seed_start=seed_start,
            db_path=self._db_path,
            game_cfg=self._game_cfg,
            ai_cfg=self._ai_cfg,
            max_ticks=max_ticks,
            curriculum=curriculum,
        )

        # Step 5–8: Run standard pipeline (retrain → eval → promote)
        return self.run_pipeline(
            auto_promote=auto_promote,
            source_filter=None,
        )
