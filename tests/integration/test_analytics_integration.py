"""Integration tests for the analytics / explainability layer.

Tests the explain_match CLI, analyze_player CLI, and full report
generation from actual planner-generated DB contents.
"""

from __future__ import annotations

import json
from dataclasses import asdict

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
from analytics.match_analyzer import analyze_match
from analytics.pattern_miner import mine_patterns
from analytics.planner_metrics import compute_planner_metrics
from analytics.report_builder import build_match_report, build_aggregate_report
from analytics.explanation_types import ExplainabilityReport


@pytest.fixture
def configs():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.connect()
    run_migrations(d)
    d.execute_safe(
        "INSERT INTO matches (match_id, session_id, started_at, ended_at, "
        "total_ticks, winner, player_hp_final, ai_hp_final, rng_seed, config_hash) "
        "VALUES ('m1', 's1', '2025-01-01', '2025-01-01', 500, 'AI', 30, 180, 42, 'h');",
    )
    yield d
    d.close()


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


def _run_planner_ticks(planner, bm, pe, sim_state, game_cfg,
                       match_id="m1", n_ticks=200):
    """Run the planner for n_ticks and return the number of commitments."""
    bm.on_match_start(match_id)
    pe.on_match_start(match_id)
    planner.on_match_start(match_id, "s1", 42)

    commits = 0
    for tick in range(n_ticks):
        sim_state.set_phase(TickPhase.SIMULATE)
        sim_state.tick_id = tick
        sim_state.ai.fsm_state = FSMState.IDLE
        sim_state.ai.active_commitment = None
        result = planner.decide(sim_state.ai, sim_state, game_cfg)
        if result is not None:
            commits += 1
            # Simulate outcome resolution
            planner.on_ai_commit_end(tick + 5, sim_state.ai.hp, sim_state.player.hp)
    return commits


# ------------------------------------------------------------------ #
# Full match report from planner output                                #
# ------------------------------------------------------------------ #

class TestFullMatchReport:
    def test_report_from_planner_output(self, configs, db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        commits = _run_planner_ticks(
            planner, bm, pe, sim_state, game_cfg)

        report = build_match_report(db, "m1")
        me = report.match_explanation
        assert me.match_id == "m1"
        assert me.decision_count == commits
        assert me.decision_count > 0
        assert me.tier_label == "T2 (full adaptive)"
        assert len(me.mode_usage) > 0
        assert len(me.notable_decisions) > 0

    def test_report_json_roundtrip(self, configs, db, sim_state):
        game_cfg, ai_cfg = configs
        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        _run_planner_ticks(planner, bm, pe, sim_state, game_cfg)

        report = build_match_report(db, "m1")
        report_dict = asdict(report)
        output = json.dumps(report_dict, indent=2, default=str)
        parsed = json.loads(output)
        assert parsed["match_explanation"]["match_id"] == "m1"
        assert isinstance(parsed["match_explanation"]["mode_usage"], dict)
        assert isinstance(parsed["match_explanation"]["notable_decisions"], list)

    def test_t0_match_produces_valid_report(self, db):
        """T0 has no ai_decisions — report should be valid with empty data."""
        db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, "
            "rng_seed, config_hash) "
            "VALUES ('m_t0', 's1', '2025-01-01', 42, 'h');",
        )
        report = build_match_report(db, "m_t0")
        assert report.match_explanation.decision_count == 0
        assert report.match_explanation.tier_label == "T0 (baseline)"
        assert report.match_explanation.prediction_accuracy is None


# ------------------------------------------------------------------ #
# explain_match script core logic                                      #
# ------------------------------------------------------------------ #

class TestExplainMatchScript:
    def test_format_report_runs(self, configs, db, sim_state):
        from scripts.explain_match import format_report

        game_cfg, ai_cfg = configs
        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        _run_planner_ticks(planner, bm, pe, sim_state, game_cfg)

        report = build_match_report(db, "m1")
        output = format_report(report)
        assert "MATCH EXPLANATION" in output
        assert "m1" in output
        assert "Prediction Performance" in output
        assert "Tactical Mode Usage" in output

    def test_format_handles_no_data(self, db):
        from scripts.explain_match import format_report

        report = build_match_report(db, "nonexistent")
        output = format_report(report)
        assert "unavailable" in output.lower() or "MATCH EXPLANATION" in output


# ------------------------------------------------------------------ #
# analyze_player script core logic                                     #
# ------------------------------------------------------------------ #

class TestAnalyzePlayerScript:
    def test_format_player_report_runs(self, db):
        from scripts.analyze_player import format_player_report

        pattern = mine_patterns(db)
        metrics = compute_planner_metrics(db, ["m1"])
        output = format_player_report(pattern, metrics, 1)
        assert "PLAYER ANALYSIS" in output
        assert "player_1" in output

    def test_player_report_no_decisions(self, db):
        from scripts.analyze_player import format_player_report

        pattern = mine_patterns(db)
        metrics = compute_planner_metrics(db, [])
        output = format_player_report(pattern, metrics, 0)
        assert "T0 baseline" in output or "No AI planner data" in output


# ------------------------------------------------------------------ #
# Aggregate report                                                     #
# ------------------------------------------------------------------ #

class TestAggregateReport:
    def test_multi_match_aggregate(self, configs, db, sim_state):
        game_cfg, ai_cfg = configs

        db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, "
            "rng_seed, config_hash) "
            "VALUES ('m2', 's1', '2025-01-02', 43, 'h');",
        )

        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(
            db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        _run_planner_ticks(planner, bm, pe, sim_state, game_cfg,
                           match_id="m1", n_ticks=100)
        _run_planner_ticks(planner, bm, pe, sim_state, game_cfg,
                           match_id="m2", n_ticks=100)

        report = build_aggregate_report(db, ["m1", "m2"])
        assert report.matches_included == ["m1", "m2"]
        assert report.planner_metrics.total_decisions > 0
        assert report.match_explanation.match_id == "m2"
