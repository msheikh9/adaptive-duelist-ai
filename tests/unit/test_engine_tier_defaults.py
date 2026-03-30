"""Tests: live engine uses T2 by default and loads the promoted model."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_factory(tmp_path):
    """Return a factory that builds a headless Engine with a temp DB."""
    from config.config_loader import load_config
    from data.db import Database
    from data.migrations.migration_runner import run_migrations

    game_cfg, ai_cfg, display_cfg = load_config()
    db_path = tmp_path / "tier_test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)

    engines = []

    def _make(ai_tier=None):
        from game.engine import Engine
        from ai.layers.tactical_planner import AITier
        kwargs = {"headless": True}
        if ai_tier is not None:
            kwargs["ai_tier"] = ai_tier
        e = Engine(game_cfg, ai_cfg, display_cfg, db, **kwargs)
        engines.append(e)
        return e

    yield _make
    db.close()


# ---------------------------------------------------------------------------
# Default tier is T2_FULL_ADAPTIVE
# ---------------------------------------------------------------------------

class TestDefaultTier:

    def test_engine_default_is_t2(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        from game.engine import Engine
        import inspect
        sig = inspect.signature(Engine.__init__)
        default = sig.parameters["ai_tier"].default
        assert default == AITier.T2_FULL_ADAPTIVE, (
            f"Engine default tier should be T2_FULL_ADAPTIVE, got {default}"
        )

    def test_engine_without_explicit_tier_uses_t2(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory()
        assert engine._ai_tier == AITier.T2_FULL_ADAPTIVE

    def test_engine_with_t2_has_tactical_planner(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T2_FULL_ADAPTIVE)
        assert engine._tactical_planner is not None

    def test_engine_with_t1_has_tactical_planner(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T1_MARKOV_ONLY)
        assert engine._tactical_planner is not None

    def test_engine_with_t0_has_no_tactical_planner(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T0_BASELINE)
        assert engine._tactical_planner is None


# ---------------------------------------------------------------------------
# main.py passes T2 to Engine
# ---------------------------------------------------------------------------

class TestMainEntryPoint:

    def test_main_imports_t2(self):
        """main.py must reference T2_FULL_ADAPTIVE explicitly."""
        import ast
        from pathlib import Path
        src = (Path(__file__).parent.parent.parent / "main.py").read_text()
        tree = ast.parse(src)
        # Look for T2_FULL_ADAPTIVE anywhere in the source text
        assert "T2_FULL_ADAPTIVE" in src, (
            "main.py must pass ai_tier=AITier.T2_FULL_ADAPTIVE to Engine"
        )


# ---------------------------------------------------------------------------
# try_load_sklearn is called with force=True in Engine.__init__
# ---------------------------------------------------------------------------

class TestSklearnForceLoad:

    def test_force_load_called_in_constructor(self):
        """Engine.__init__ source must call try_load_sklearn(force=True)."""
        import inspect
        from game.engine import Engine
        src = inspect.getsource(Engine.__init__)
        assert "try_load_sklearn(force=True)" in src, (
            "Engine.__init__ must call try_load_sklearn(force=True) so the "
            "active promoted model is loaded for live play regardless of match count."
        )

    def test_no_active_model_does_not_crash(self, engine_factory):
        """With an empty DB (no trained model), Engine still constructs fine."""
        engine = engine_factory()
        assert engine is not None
        assert engine._prediction_engine is not None

    def test_active_model_loaded_when_registered(self, tmp_path):
        """When a model is in the registry, PredictionEngine activates sklearn."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from config.config_loader import load_config
        from data.db import Database
        from data.migrations.migration_runner import run_migrations
        from ai.layers.behavior_model import BehaviorModel
        from ai.layers.prediction_engine import PredictionEngine
        from ai.models.base_predictor import ALL_LABELS

        game_cfg, ai_cfg, _ = load_config()
        db_path = tmp_path / "model_load_test.db"
        db = Database(db_path)
        db.connect()
        run_migrations(db)

        # Train a minimal model and register it
        clf = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)
        import numpy as np
        X = np.zeros((len(ALL_LABELS), 10))
        y = ALL_LABELS
        clf.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(clf, model_path)
        db.execute_safe(
            "INSERT INTO model_registry "
            "(version, model_path, model_type, is_active, eval_accuracy, "
            " eval_top2_acc, dataset_size, metadata) "
            "VALUES ('v_test', ?, 'random_forest', 1, 0.5, 0.8, 100, '{}');",
            (str(model_path),),
        )

        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        pe.try_load_sklearn(force=True)

        # After force load, sklearn predictor should be active
        assert pe._ensemble._sklearn is not None, (
            "Sklearn predictor should be activated after try_load_sklearn(force=True)"
        )
        db.close()


# ---------------------------------------------------------------------------
# _apply_tier switches the planner and tier attribute
# ---------------------------------------------------------------------------

class TestApplyTier:

    def test_apply_tier_t0_clears_planner(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T2_FULL_ADAPTIVE)
        assert engine._tactical_planner is not None
        engine._apply_tier(AITier.T0_BASELINE)
        assert engine._ai_tier == AITier.T0_BASELINE
        assert engine._tactical_planner is None

    def test_apply_tier_t2_creates_planner(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T0_BASELINE)
        assert engine._tactical_planner is None
        engine._apply_tier(AITier.T2_FULL_ADAPTIVE)
        assert engine._ai_tier == AITier.T2_FULL_ADAPTIVE
        assert engine._tactical_planner is not None

    def test_apply_same_tier_is_noop(self, engine_factory):
        from ai.layers.tactical_planner import AITier
        engine = engine_factory(ai_tier=AITier.T2_FULL_ADAPTIVE)
        planner_before = engine._tactical_planner
        engine._apply_tier(AITier.T2_FULL_ADAPTIVE)
        # same object — no rebuild
        assert engine._tactical_planner is planner_before

    def test_apply_tier_updates_renderer_badge(self, tmp_path):
        """_apply_tier updates _renderer._ai_tier_name when renderer exists."""
        import pygame
        from config.config_loader import load_config
        from data.db import Database
        from data.migrations.migration_runner import run_migrations
        from game.engine import Engine
        from ai.layers.tactical_planner import AITier
        from rendering.renderer import Renderer

        game_cfg, ai_cfg, display_cfg = load_config()
        db = Database(tmp_path / "badge_test.db")
        db.connect()
        run_migrations(db)

        engine = Engine(game_cfg, ai_cfg, display_cfg, db,
                        headless=True, ai_tier=AITier.T2_FULL_ADAPTIVE)
        # Manually attach a renderer (headless engine skips init)
        renderer = Renderer(game_cfg, display_cfg, ai_tier_name="T2_FULL_ADAPTIVE")
        engine._renderer = renderer

        engine._apply_tier(AITier.T0_BASELINE)
        assert engine._renderer._ai_tier_name == "T0_BASELINE"
        db.close()


# ---------------------------------------------------------------------------
# Renderer tier selector display
# ---------------------------------------------------------------------------

class TestRendererTierSelector:

    def test_render_title_accepts_tier_name(self):
        """render_title() signature accepts selected_tier kwarg."""
        import inspect
        from rendering.renderer import Renderer
        sig = inspect.signature(Renderer.render_title)
        assert "selected_tier" in sig.parameters

    def test_render_title_default_is_t2(self):
        from rendering.renderer import Renderer
        import inspect
        sig = inspect.signature(Renderer.render_title)
        default = sig.parameters["selected_tier"].default
        assert default == "T2_FULL_ADAPTIVE"

    def test_title_screen_tier_order_in_engine(self):
        """_run_title_screen cycles T2 → T1 → T0 (T2 is index 0)."""
        import inspect
        from game.engine import Engine
        src = inspect.getsource(Engine._run_title_screen)
        # T2 must appear before T1 and T0 in the tier list
        t2_pos = src.index("T2_FULL_ADAPTIVE")
        t1_pos = src.index("T1_MARKOV_ONLY")
        t0_pos = src.index("T0_BASELINE")
        assert t2_pos < t1_pos < t0_pos, (
            "T2 should be first (default) in the title screen tier list"
        )
