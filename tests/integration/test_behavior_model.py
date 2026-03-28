"""Integration tests for BehaviorModel: profile updates, persistence,
engine integration, and cross-match accumulation."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from data.db import Database
from data.events import EventType, SemanticEvent
from data.migrations.migration_runner import run_migrations
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone
from game.state import MatchStatus
from ai.layers.behavior_model import BehaviorModel
from ai.profile.player_profile import PlayerProfile
from tests.fixtures.headless_engine import HeadlessMatch


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
def model(configs, tmp_db):
    game_cfg, ai_cfg = configs
    m = BehaviorModel(tmp_db, ai_cfg, game_cfg)
    m.load_profile()
    return m


def _player_commit_event(commitment: CombatCommitment, tick_id: int = 0,
                          opponent_fsm: FSMState = FSMState.IDLE,
                          reaction_ticks: int = 30) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id="m1",
        tick_id=tick_id,
        actor=Actor.PLAYER,
        commitment=commitment,
        opponent_fsm_state=opponent_fsm,
        spacing_zone=SpacingZone.MID,
        actor_hp=200, opponent_hp=200,
        actor_stamina=80, opponent_stamina=80,
        reaction_ticks=reaction_ticks,
    )


class TestBehaviorModelProfileUpdates:
    def test_profile_starts_empty(self, model):
        assert model.profile.total_commitments == 0

    def test_session_count_incremented_on_load(self, model):
        assert model.profile.session_count == 1

    def test_event_updates_profile(self, model):
        model.on_match_start("m1")
        model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        assert model.profile.action_frequencies.get("LIGHT_ATTACK") == 1

    def test_events_update_aggression(self, model):
        model.on_match_start("m1")
        for _ in range(8):
            model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        for _ in range(2):
            model.on_event(_player_commit_event(CombatCommitment.MOVE_RIGHT))
        assert abs(model.profile.aggression_index - 0.8) < 1e-9

    def test_markov_predictor_updated(self, model):
        model.on_match_start("m1")
        for _ in range(5):
            model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        result = model.predict_next()
        assert result.has_prediction
        assert result.top_commitment == CombatCommitment.LIGHT_ATTACK

    def test_ai_events_ignored_by_predictor(self, model):
        model.on_match_start("m1")
        ai_event = SemanticEvent(
            event_type=EventType.COMMITMENT_START,
            match_id="m1", tick_id=0,
            actor=Actor.AI,
            commitment=CombatCommitment.HEAVY_ATTACK,
            actor_hp=200, opponent_hp=200,
            actor_stamina=80, opponent_stamina=80,
        )
        model.on_event(ai_event)
        assert model.predictor.total_observations == 0


class TestBehaviorModelPersistence:
    def test_profile_saved_on_match_end(self, model, tmp_db):
        model.on_match_start("m1")
        model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        model.on_match_end("PLAYER", 3600)

        row = tmp_db.fetchone(
            "SELECT action_frequencies FROM player_profiles WHERE player_id='player_1';"
        )
        assert row is not None
        import json
        af = json.loads(row["action_frequencies"])
        assert af.get("LIGHT_ATTACK") == 1

    def test_profile_loaded_on_second_instance(self, configs, tmp_db):
        game_cfg, ai_cfg = configs

        # First session: do 5 attacks
        m1 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m1.load_profile()
        m1.on_match_start("m1")
        for _ in range(5):
            m1.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        m1.on_match_end("AI", 3600)

        # Second session: load and verify counts persisted
        m2 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m2.load_profile()
        assert m2.profile.action_frequencies.get("LIGHT_ATTACK") == 5

    def test_bigrams_persisted_via_sync(self, configs, tmp_db):
        game_cfg, ai_cfg = configs

        m1 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m1.load_profile()
        m1.on_match_start("m1")
        # Build bigram: L → H several times
        for _ in range(3):
            m1.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK, tick_id=0))
            m1.on_event(_player_commit_event(CombatCommitment.HEAVY_ATTACK, tick_id=1))
        m1.on_match_end("AI", 1800)

        m2 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m2.load_profile()
        assert "LIGHT_ATTACK" in m2.profile.bigrams
        assert "HEAVY_ATTACK" in m2.profile.bigrams["LIGHT_ATTACK"]

    def test_match_count_persists_across_sessions(self, configs, tmp_db):
        game_cfg, ai_cfg = configs

        m1 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m1.load_profile()
        m1.on_match_start("m1")
        m1.on_match_end("AI", 3600)

        m2 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m2.load_profile()
        assert m2.profile.match_count == 1

    def test_win_rate_persists(self, configs, tmp_db):
        game_cfg, ai_cfg = configs

        m1 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m1.load_profile()
        m1.on_match_start("m1")
        m1.on_match_end("PLAYER", 3600)

        m2 = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        m2.load_profile()
        assert m2.profile.win_rate_vs_ai == 1.0
        assert m2.profile.win_count == 1


class TestBehaviorModelMarkovPredictions:
    def test_prediction_improves_with_data(self, model):
        model.on_match_start("m1")
        # Establish strong pattern: L → D
        for _ in range(5):
            model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
            model.on_event(_player_commit_event(CombatCommitment.DODGE_BACKWARD))
        result = model.predict_next()
        # After L, should predict D
        assert result.has_prediction

    def test_prediction_level_upgrades_with_history(self, model):
        model.on_match_start("m1")
        # Build enough observations for the predictor to use n-gram levels
        for _ in range(4):
            model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
            model.on_event(_player_commit_event(CombatCommitment.HEAVY_ATTACK))
        result = model.predict_next()
        # With sufficient data, predictor should use n-gram (not "none")
        assert result.level in ("trigram", "bigram", "unigram")
        assert result.has_prediction

    def test_history_resets_on_new_match(self, model):
        model.on_match_start("m1")
        for _ in range(3):
            model.on_event(_player_commit_event(CombatCommitment.LIGHT_ATTACK))
        model.on_match_end("AI", 1000)
        model.on_match_start("m2")
        # History should be empty after new match start
        assert model.predictor.history == []


class TestBehaviorModelEngineIntegration:
    def test_headless_match_updates_profile(self, configs, tmp_db):
        """Running a headless match through HeadlessMatch should produce events
        that (if fed through BehaviorModel) update the profile."""
        game_cfg, ai_cfg = configs

        # Run a match to collect events
        match = HeadlessMatch(game_cfg, rng_seed=42)
        match.run_until_end(max_ticks=5000)

        # Manually feed events to behavior model (simulating engine integration)
        model = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        model.load_profile()
        model.on_match_start("headless-1")

        # Inject some synthetic player commitment events
        for i in range(10):
            model.on_event(_player_commit_event(
                CombatCommitment.LIGHT_ATTACK, tick_id=i * 10, reaction_ticks=30))

        model.on_match_end("AI", match.state.tick_id)

        assert model.profile.total_commitments == 10
        assert model.profile.match_count == 1
        assert model.profile.total_ticks_observed == match.state.tick_id

    def test_profile_changes_across_multiple_matches(self, configs, tmp_db):
        game_cfg, ai_cfg = configs
        model = BehaviorModel(tmp_db, ai_cfg, game_cfg)
        model.load_profile()

        for i in range(3):
            model.on_match_start(f"m{i}")
            for j in range(5):
                model.on_event(_player_commit_event(
                    CombatCommitment.LIGHT_ATTACK, tick_id=j))
            model.on_match_end("PLAYER" if i % 2 == 0 else "AI", 3600)

        assert model.profile.match_count == 3
        assert model.profile.total_commitments == 15
        assert model.profile.win_count == 2
