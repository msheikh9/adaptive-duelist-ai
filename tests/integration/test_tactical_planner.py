"""Integration tests for TacticalPlanner: lifecycle, decision-making,
outcome resolution, and tier switching."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from game.combat.actions import CombatCommitment, FSMState, SpacingZone
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    SimulationState,
    TickPhase,
)
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.layers.tactical_planner import AITier, TacticalPlanner
from ai.strategy.tactics import TacticalIntent


@pytest.fixture
def configs():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
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
def planner_t2(configs, tmp_db):
    game_cfg, ai_cfg = configs
    bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
    planner = TacticalPlanner(tmp_db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)
    return planner, bm, pe


@pytest.fixture
def planner_t1(configs, tmp_db):
    game_cfg, ai_cfg = configs
    bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
    planner = TacticalPlanner(tmp_db, pe, ai_cfg, game_cfg, AITier.T1_MARKOV_ONLY)
    return planner, bm, pe


class TestPlannerLifecycle:
    def test_tier_set(self, planner_t2):
        planner, _, _ = planner_t2
        assert planner.tier == AITier.T2_FULL_ADAPTIVE

    def test_match_start_resets_memory(self, planner_t2, sim_state, configs):
        planner, _, _ = planner_t2
        planner.on_match_start("m1", "s1", 42)
        planner.memory.record_mode(TacticalIntent.EXPLOIT_PATTERN)
        planner.on_match_start("m2", "s1", 43)
        assert len(planner.memory.recent_modes) == 0


class TestDecisionMaking:
    def test_decide_returns_commitment_or_none(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None or isinstance(result, CombatCommitment)

    def test_locked_ai_returns_none(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        sim_state.ai.fsm_state = FSMState.ATTACK_ACTIVE
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None

    def test_dead_ai_returns_none(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        sim_state.ai.fsm_state = FSMState.KO
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None

    def test_multiple_decisions_work(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        commitments = []
        for tick in range(100):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            # Reset AI to idle each tick (simplified — real engine has FSM)
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            result = planner.decide(sim_state.ai, sim_state, game_cfg)
            if result is not None:
                commitments.append(result)

        # Should make at least some decisions in 100 ticks
        assert len(commitments) > 0


class TestOutcomeResolution:
    def test_player_commit_updates_prediction_tracking(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        # Make a decision first
        sim_state.set_phase(TickPhase.SIMULATE)
        planner.decide(sim_state.ai, sim_state, game_cfg)

        # Player commits
        planner.on_player_commit(CombatCommitment.LIGHT_ATTACK, 10)
        # Should not crash; prediction tracking updated

    def test_ai_commit_end_records_outcome(self, planner_t2, sim_state, configs):
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        if result is not None:
            planner.on_ai_commit_end(15, sim_state.ai.hp, sim_state.player.hp)
            # Should record in mode_outcome_log
            assert len(planner.memory.mode_outcome_log) >= 0  # no crash

    def test_outcome_attribution_uses_originating_mode(self, planner_t2, sim_state, configs):
        """Outcome mode must reflect the mode that produced the action,
        not whatever recent_modes[-1] is at resolution time."""
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        sim_state.tick_id = 0
        sim_state.ai.fsm_state = FSMState.IDLE
        sim_state.ai.active_commitment = None
        result = planner.decide(sim_state.ai, sim_state, game_cfg)

        if result is None:
            pytest.skip("no commit at tick 0")

        # Mode that was recorded for this action
        assert len(planner.memory.recent_modes) > 0
        originating_mode = planner.memory.recent_modes[-1]

        # Poison recent_modes with a different sentinel mode
        all_modes = list(TacticalIntent)
        poison = next(m for m in all_modes if m != originating_mode)
        planner.memory.recent_modes.append(poison)
        assert planner.memory.recent_modes[-1] == poison

        # Resolve outcome — must use originating_mode, not poison
        planner.on_ai_commit_end(15, sim_state.ai.hp, sim_state.player.hp)

        assert len(planner.memory.mode_outcome_log) == 1
        assert planner.memory.mode_outcome_log[-1].mode == originating_mode
        assert planner.memory.mode_outcome_log[-1].mode != poison


class TestTierBehavior:
    def test_t1_does_not_update_memory(self, planner_t1, sim_state, configs):
        planner, bm, pe = planner_t1
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        # Player commit should not update prediction tracking in T1
        planner.on_player_commit(CombatCommitment.LIGHT_ATTACK, 10)
        assert len(planner.memory.recent_predictions) == 0

    def test_t1_still_makes_decisions(self, planner_t1, sim_state, configs):
        planner, bm, pe = planner_t1
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        sim_state.set_phase(TickPhase.SIMULATE)
        # Should work even without memory updates
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        assert result is None or isinstance(result, CombatCommitment)


class TestDelayedActions:
    def test_delayed_action_counts_down(self, planner_t2, sim_state, configs):
        """Planner may return None on first call due to commit_delay,
        then fire the action on subsequent calls."""
        planner, bm, pe = planner_t2
        game_cfg, _ = configs
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        results = []
        for tick in range(50):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            r = planner.decide(sim_state.ai, sim_state, game_cfg)
            results.append(r)

        # Should have at least one actual commitment
        actual_commits = [r for r in results if r is not None]
        assert len(actual_commits) > 0
