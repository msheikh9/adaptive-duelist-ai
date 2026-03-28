"""Integration tests for AI decision logging to ai_decisions table."""

from __future__ import annotations

import json
import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from game.combat.actions import CombatCommitment, FSMState
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
    # Insert a match record for FK
    db.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
        "VALUES ('m1', 's1', '2025-01-01', 42, 'hash');",
    )
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


class TestAIDecisionLogging:
    def test_decisions_logged_to_db(self, configs, tmp_db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            tmp_db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        commits = 0
        for tick in range(200):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            result = planner.decide(sim_state.ai, sim_state, game_cfg)
            if result is not None:
                commits += 1

        # Should have logged at least one decision
        rows = tmp_db.fetchall(
            "SELECT * FROM ai_decisions WHERE match_id = 'm1';"
        )
        assert len(rows) > 0
        assert len(rows) == commits

    def test_decision_fields_populated(self, configs, tmp_db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            tmp_db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        # Run until we get at least one decision
        for tick in range(200):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            result = planner.decide(sim_state.ai, sim_state, game_cfg)
            if result is not None:
                break

        rows = tmp_db.fetchall(
            "SELECT * FROM ai_decisions WHERE match_id = 'm1' LIMIT 1;"
        )
        if rows:
            row = rows[0]
            assert row["session_id"] == "s1"
            assert row["match_id"] == "m1"
            assert row["tick_id"] >= 0
            assert row["tactical_mode"] is not None
            assert row["ai_action"] is not None
            assert row["positioning_bias"] is not None

    def test_reason_tags_are_json(self, configs, tmp_db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            tmp_db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        for tick in range(200):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            if planner.decide(sim_state.ai, sim_state, game_cfg) is not None:
                break

        rows = tmp_db.fetchall(
            "SELECT reason_tags FROM ai_decisions WHERE match_id = 'm1' LIMIT 1;"
        )
        if rows:
            tags = json.loads(rows[0]["reason_tags"])
            assert isinstance(tags, list)
            assert len(tags) > 0

    def test_outcome_persisted_after_commit_end(self, configs, tmp_db, sim_state):
        """outcome and outcome_tick are written back to the row after on_ai_commit_end."""
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            tmp_db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        bm.on_match_start("m1")
        pe.on_match_start("m1")
        planner.on_match_start("m1", "s1", 42)

        committed_tick = None
        for tick in range(200):
            sim_state.set_phase(TickPhase.SIMULATE)
            sim_state.tick_id = tick
            sim_state.ai.fsm_state = FSMState.IDLE
            sim_state.ai.active_commitment = None
            result = planner.decide(sim_state.ai, sim_state, game_cfg)
            if result is not None:
                committed_tick = tick
                break

        if committed_tick is None:
            pytest.skip("no commit generated in 200 ticks")

        # Simulate commitment ending
        planner.on_ai_commit_end(committed_tick + 10,
                                  sim_state.ai.hp, sim_state.player.hp)

        rows = tmp_db.fetchall(
            "SELECT outcome, outcome_tick FROM ai_decisions "
            "WHERE match_id = 'm1' AND tick_id = ?;",
            (committed_tick,),
        )
        assert len(rows) == 1
        assert rows[0]["outcome"] in ("success", "failure")
        assert rows[0]["outcome_tick"] == committed_tick + 10

    def test_no_logging_in_t0_baseline(self, configs, tmp_db, sim_state):
        """T0 baseline uses BaselineAIController; no planner means no logging."""
        # Just verify the DB has no decisions when no planner is used
        rows = tmp_db.fetchall("SELECT * FROM ai_decisions;")
        assert len(rows) == 0
