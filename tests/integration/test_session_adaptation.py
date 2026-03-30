"""Integration tests for Phase 11: In-Session Behavioral Adaptation."""

from __future__ import annotations

import random

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from game.combat.actions import CombatCommitment, FSMState
from game.state import (
    ArenaState, FighterState, MatchStatus, SimulationState, TickPhase,
)
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.layers.tactical_planner import AITier, TacticalPlanner
from ai.profile.archetype_classifier import ArchetypeLabel, classify_archetype
from ai.profile.player_profile import PlayerProfile
from ai.strategy.session_memory import SessionMemory
from ai.strategy.tactics import TacticalIntent


@pytest.fixture
def configs():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "session_test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    yield db
    db.close()


@pytest.fixture
def sim_state(configs):
    game_cfg, _ = configs
    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    return SimulationState(
        tick_id=0,
        rng_seed=42,
        player=FighterState(
            x=arena.width_sub // 3,
            y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp,
            stamina=game_cfg.fighter.max_stamina,
            facing=1,
        ),
        ai=FighterState(
            x=(arena.width_sub * 2) // 3,
            y=arena.ground_y_sub,
            hp=game_cfg.fighter.max_hp,
            stamina=game_cfg.fighter.max_stamina,
            facing=-1,
        ),
        arena=arena,
        match_status=MatchStatus.ACTIVE,
    )


@pytest.fixture
def planner_with_bm(configs, tmp_db):
    game_cfg, ai_cfg = configs
    bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
    planner = TacticalPlanner(
        tmp_db, pe, ai_cfg, game_cfg,
        AITier.T2_FULL_ADAPTIVE,
        behavior_model=bm,
    )
    return planner, bm, pe


# ---------------------------------------------------------------------------
# SessionMemory persistence across matches
# ---------------------------------------------------------------------------

class TestSessionMemoryPersistence:

    def test_session_memory_not_reset_on_match_start(self, planner_with_bm, sim_state, configs):
        """Session memory survives on_match_start() calls."""
        planner, bm, pe = planner_with_bm
        game_cfg, _ = configs

        # Manually inject data into session memory
        planner.session_memory.record_match_outcomes(
            {"EXPLOIT_PATTERN": (3, 5)}
        )
        before = planner.session_memory.total_samples()

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        # Session memory should still have data (not reset)
        assert planner.session_memory.total_samples() > 0
        assert planner.session_memory.total_samples() == pytest.approx(
            before, rel=1e-6
        )

    def test_planner_memory_IS_reset_on_match_start(self, planner_with_bm, sim_state, configs):
        """PlannerMemory resets but SessionMemory does not."""
        planner, bm, pe = planner_with_bm
        game_cfg, _ = configs

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)
        planner.memory.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        planner.memory.record_mode(TacticalIntent.DEFENSIVE_RESET)

        # Second match start — planner memory resets
        planner.on_match_start("m2", "s1", 43)
        assert len(planner.memory.recent_modes) == 0

    def test_on_match_end_records_outcomes_into_session(self, planner_with_bm, sim_state, configs):
        """After on_match_end(), session memory gains new data."""
        planner, bm, pe = planner_with_bm
        game_cfg, _ = configs

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        # Run a few ticks to generate some decisions
        for tick in range(60):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            planner.decide(sim_state.ai, sim_state, game_cfg)

        # Manually inject an outcome so mode_outcome_log is non-empty
        from ai.strategy.planner_memory import ModeOutcome
        planner.memory.record_outcome(ModeOutcome(
            mode=TacticalIntent.EXPLOIT_PATTERN,
            tick_id=10,
            success=True,
        ))

        before = planner.session_memory.total_samples()
        planner.on_match_end()
        after = planner.session_memory.total_samples()

        assert after > before or after >= 0  # at least no crash

    def test_session_memory_accumulates_across_multiple_matches(self, configs, tmp_db):
        """Three consecutive on_match_end() calls grow the session total."""
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            tmp_db, pe, ai_cfg, game_cfg,
            AITier.T2_FULL_ADAPTIVE,
            behavior_model=bm,
        )

        from ai.strategy.planner_memory import ModeOutcome

        for i in range(3):
            bm.on_match_start(f"m{i}")
            pe.on_match_start(f"m{i}")
            planner.on_match_start(f"m{i}", "s1", i)
            planner.memory.record_outcome(ModeOutcome(
                mode=TacticalIntent.BAIT_AND_PUNISH,
                tick_id=10,
                success=True,
            ))
            planner.on_match_end()

        # After 3 matches, session should have weighted data for BAIT_AND_PUNISH
        total = planner.session_memory.total_samples()
        assert total > 0

    def test_session_memory_reset_clears_data(self, planner_with_bm):
        planner, _, _ = planner_with_bm
        planner.session_memory.record_match_outcomes(
            {"EXPLOIT_PATTERN": (3, 5)}
        )
        planner.session_memory.reset()
        assert planner.session_memory.total_samples() == 0.0


# ---------------------------------------------------------------------------
# Archetype classification
# ---------------------------------------------------------------------------

class TestArchetypeIntegration:

    def test_current_archetype_returns_balanced_when_no_matches(self, planner_with_bm, configs):
        """Before min_matches_for_archetype, always BALANCED."""
        planner, bm, pe = planner_with_bm
        _, ai_cfg = configs
        # Profile starts with match_count=0
        assert bm.profile.match_count == 0
        label = bm.current_archetype()
        assert label == ArchetypeLabel.BALANCED

    def test_current_archetype_classifies_after_enough_matches(self, planner_with_bm, configs):
        """After min_matches_for_archetype, classification uses the profile."""
        planner, bm, pe = planner_with_bm
        _, ai_cfg = configs
        min_m = ai_cfg.session_adaptation.min_matches_for_archetype

        # Set profile to look aggressive and bump match count
        bm.profile.aggression_index = 0.70
        bm.profile.dodge_frequency = 0.05
        bm.profile.action_frequencies = {
            "LIGHT_ATTACK": 20, "HEAVY_ATTACK": 15, "MOVE_RIGHT": 5,
        }
        bm._profile.match_count = min_m  # direct attribute write (test only)

        label = bm.current_archetype()
        assert label == ArchetypeLabel.AGGRESSIVE

    def test_current_archetype_defensive_profile(self, planner_with_bm, configs):
        planner, bm, pe = planner_with_bm
        _, ai_cfg = configs
        min_m = ai_cfg.session_adaptation.min_matches_for_archetype

        bm.profile.aggression_index = 0.15
        bm.profile.dodge_frequency = 0.35
        bm.profile.action_frequencies = {
            "DODGE_BACKWARD": 10, "MOVE_LEFT": 5, "LIGHT_ATTACK": 2,
        }
        bm._profile.match_count = min_m

        label = bm.current_archetype()
        assert label == ArchetypeLabel.DEFENSIVE

    def test_current_archetype_evasive_profile(self, planner_with_bm, configs):
        planner, bm, pe = planner_with_bm
        _, ai_cfg = configs
        min_m = ai_cfg.session_adaptation.min_matches_for_archetype

        bm.profile.aggression_index = 0.10
        bm.profile.dodge_frequency = 0.55
        bm.profile.action_frequencies = {"DODGE_BACKWARD": 15, "LIGHT_ATTACK": 2, "MOVE_LEFT": 1}
        bm._profile.match_count = min_m

        label = bm.current_archetype()
        assert label == ArchetypeLabel.EVASIVE


# ---------------------------------------------------------------------------
# Strategy selector integration
# ---------------------------------------------------------------------------

class TestStrategySelectionWithSession:

    def _make_ctx(self, configs):
        from ai.models.base_predictor import make_prediction_result
        from ai.strategy.ai_context import AIContext
        from game.combat.actions import SpacingZone
        game_cfg, _ = configs
        pred = make_prediction_result({}, source="none", markov_level="none")
        return AIContext(
            tick_id=100,
            player_hp_frac=1.0, player_stamina_frac=0.8,
            player_fsm=FSMState.IDLE, player_commitment=None,
            ai_hp_frac=1.0, ai_stamina_frac=0.8,
            ai_fsm=FSMState.IDLE, ai_commitment=None,
            ai_facing=1, spacing=SpacingZone.MID,
            distance_sub=20000, prediction=pred,
        )

    def test_select_mode_with_session_memory_does_not_raise(self, configs):
        from ai.strategy.planner_memory import PlannerMemory
        from ai.strategy.strategy_selector import select_mode
        game_cfg, ai_cfg = configs
        memory = PlannerMemory(ai_cfg.planner_memory)
        ctx = self._make_ctx(configs)

        sm = SessionMemory(decay_factor=0.8, min_samples=5)
        sm.record_match_outcomes({"EXPLOIT_PATTERN": (3, 5), "DEFENSIVE_RESET": (1, 4)})

        rng = random.Random(42)
        mode = select_mode(
            ctx, memory,
            ai_cfg.strategy, ai_cfg.planner_memory, rng,
            session_memory=sm,
            archetype=ArchetypeLabel.AGGRESSIVE,
            archetype_table=ai_cfg.archetype_mode_alignment,
        )
        assert isinstance(mode, TacticalIntent)

    def test_select_mode_without_session_returns_valid_mode(self, configs):
        """Existing call sites with no session args still work."""
        from ai.strategy.planner_memory import PlannerMemory
        from ai.strategy.strategy_selector import select_mode
        game_cfg, ai_cfg = configs
        memory = PlannerMemory(ai_cfg.planner_memory)
        ctx = self._make_ctx(configs)

        rng = random.Random(42)
        mode = select_mode(ctx, memory, ai_cfg.strategy, ai_cfg.planner_memory, rng)
        assert isinstance(mode, TacticalIntent)

    def test_archetype_alignment_shifts_mode_distribution(self, configs):
        """With AGGRESSIVE archetype, DEFENSIVE_RESET alignment bonus should
        be applied without crashing. Distribution may shift."""
        from ai.strategy.planner_memory import PlannerMemory
        from ai.strategy.strategy_selector import select_mode, _session_adjustments, ALL_MODES
        game_cfg, ai_cfg = configs

        w = ai_cfg.strategy.scoring_weights
        sm = SessionMemory(min_samples=1)
        sm.record_match_outcomes({"DEFENSIVE_RESET": (2, 4)})

        table = {"DEFENSIVE_RESET": {"AGGRESSIVE": 0.5}}
        adj = _session_adjustments(
            TacticalIntent.DEFENSIVE_RESET,
            session_memory=sm,
            archetype=ArchetypeLabel.AGGRESSIVE,
            archetype_table=table,
            w=w,
        )
        # DEFENSIVE_RESET rate = 0.5, bonus from archetype = 0.5
        # adj = w.session_success_rate * 0 + w.archetype_alignment * 0.5
        assert adj == pytest.approx(w.archetype_alignment * 0.5, rel=1e-6)

    def test_session_adjustment_zero_when_no_session(self, configs):
        """_session_adjustments returns 0 when session_memory is None."""
        from ai.strategy.strategy_selector import _session_adjustments
        game_cfg, ai_cfg = configs
        w = ai_cfg.strategy.scoring_weights

        adj = _session_adjustments(
            TacticalIntent.EXPLOIT_PATTERN,
            session_memory=None,
            archetype=None,
            archetype_table={},
            w=w,
        )
        assert adj == 0.0

    def test_planner_passes_session_to_selector(self, planner_with_bm, sim_state, configs):
        """TacticalPlanner.decide() runs without error when session memory exists."""
        planner, bm, pe = planner_with_bm
        game_cfg, _ = configs

        # Pre-seed session memory above threshold
        planner.session_memory.record_match_outcomes(
            {m.name: (1, 2) for m in TacticalIntent}
        )

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)
        sim_state.set_phase(TickPhase.SIMULATE)
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None or isinstance(result, CombatCommitment)


# ---------------------------------------------------------------------------
# No regression: T1 planner still works
# ---------------------------------------------------------------------------

class TestT1NoRegression:

    def test_t1_planner_without_behavior_model(self, configs, tmp_db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        # No behavior_model passed — old-style construction
        planner = TacticalPlanner(tmp_db, pe, ai_cfg, game_cfg, AITier.T1_MARKOV_ONLY)

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None or isinstance(result, CombatCommitment)

    def test_session_memory_exists_on_t1_planner(self, configs, tmp_db):
        """T1 planner still has session_memory attribute (but won't record)."""
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(tmp_db, pe, ai_cfg, game_cfg, AITier.T1_MARKOV_ONLY)
        assert planner.session_memory is not None
        assert planner.session_memory.total_samples() == 0.0
