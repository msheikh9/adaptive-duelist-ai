"""Integration tests for PredictionEngine: lifecycle, inference triggers,
ensemble integration, and engine wiring."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from data.db import Database
from data.events import EventType, SemanticEvent
from data.migrations.migration_runner import run_migrations
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.models.base_predictor import LABEL_HOLD


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
def engine(configs, tmp_db):
    game_cfg, ai_cfg = configs
    bm = BehaviorModel(tmp_db, ai_cfg, game_cfg)
    bm.load_profile()
    pe = PredictionEngine(tmp_db, bm, ai_cfg, game_cfg)
    return pe, bm


def _player_commit_start(commitment: CombatCommitment, tick_id: int = 0) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id="m1",
        tick_id=tick_id,
        actor=Actor.PLAYER,
        commitment=commitment,
        opponent_fsm_state=FSMState.IDLE,
        spacing_zone=SpacingZone.MID,
        actor_hp=200, opponent_hp=200,
        actor_stamina=80, opponent_stamina=80,
        reaction_ticks=30,
    )


def _player_commit_end(tick_id: int = 0) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_END,
        match_id="m1",
        tick_id=tick_id,
        actor=Actor.PLAYER,
        actor_hp=200, opponent_hp=200,
        actor_stamina=80, opponent_stamina=80,
    )


def _ai_commit_start(commitment: CombatCommitment, tick_id: int = 0) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id="m1",
        tick_id=tick_id,
        actor=Actor.AI,
        commitment=commitment,
        actor_hp=200, opponent_hp=200,
        actor_stamina=80, opponent_stamina=80,
    )


class TestPredictionEngineLifecycle:
    def test_initial_predict_returns_empty(self, engine):
        pe, bm = engine
        pe.on_match_start("m1")
        result = pe.predict()
        assert not result.has_prediction
        assert result.source == "none"

    def test_sklearn_not_activated_before_threshold(self, engine):
        pe, bm = engine
        assert not pe.ensemble.sklearn_active

    def test_match_start_resets_prediction(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")
        # Feed some events and get a prediction
        for _ in range(3):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=50))
        assert pe.predict().has_prediction  # should have prediction now

        # Start new match — prediction should reset
        pe.on_match_start("m2")
        result = pe.predict()
        assert result.source == "none"


class TestInferenceTriggers:
    def test_commitment_end_triggers_prediction(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        # Feed player commitments to build Markov state
        for i in range(5):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)

        # COMMITMENT_END triggers prediction
        pe.on_event(_player_commit_end(tick_id=50))
        result = pe.predict()
        assert result.has_prediction
        assert result.source == "markov"

    def test_ai_events_ignored(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        ai_ev = _ai_commit_start(CombatCommitment.HEAVY_ATTACK)
        pe.on_event(ai_ev)
        result = pe.predict()
        assert not result.has_prediction

    def test_idle_timeout_triggers_reprediction(self, engine, configs):
        pe, bm = engine
        _, ai_cfg = configs
        idle_ticks = ai_cfg.prediction.inference_reeval_idle_ticks

        bm.on_match_start("m1")
        pe.on_match_start("m1")

        # Build some Markov state
        for i in range(5):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)

        pe.on_event(_player_commit_end(tick_id=50))
        first_result = pe.predict()
        first_tick = pe._last_prediction_tick

        # Simulate idle ticks passing
        pe.on_tick(first_tick + idle_ticks + 1)
        second_tick = pe._last_prediction_tick
        assert second_tick > first_tick


class TestMarkovOnlyPrediction:
    def test_strong_pattern_predicted(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        # Establish L → H → L → H pattern
        for i in range(10):
            commitment = (CombatCommitment.LIGHT_ATTACK if i % 2 == 0
                          else CombatCommitment.HEAVY_ATTACK)
            ev = _player_commit_start(commitment, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)

        pe.on_event(_player_commit_end(tick_id=100))
        result = pe.predict()
        assert result.has_prediction
        assert result.top_commitment is not None

    def test_single_observation_produces_unigram(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=0)
        bm.on_event(ev)
        pe.on_event(ev)

        pe.on_event(_player_commit_end(tick_id=15))
        result = pe.predict()
        assert result.has_prediction
        assert result.markov_level == "unigram"


class TestWeightUpdateFeedback:
    def test_weight_update_on_next_commit(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        # Build state and trigger prediction
        for i in range(5):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=50))

        # Prediction is pending; now actual next commitment arrives
        actual = _player_commit_start(CombatCommitment.HEAVY_ATTACK, tick_id=60)
        bm.on_event(actual)
        pe.on_event(actual)

        # No crash, weights updated (no sklearn active so no change expected)
        assert pe.ensemble.markov_weight == 1.0


class TestPredictionResultContract:
    def test_distribution_sums_to_one(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        for i in range(5):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=50))

        result = pe.predict()
        if result.distribution:
            total = sum(result.distribution.values())
            assert abs(total - 1.0) < 1e-6

    def test_backward_compatible_aliases(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        for i in range(3):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=30))

        result = pe.predict()
        # Phase 3 aliases
        assert result.probabilities is result.distribution
        assert result.confidence == result.commitment_confidence
        assert result.level == result.markov_level

    def test_has_prediction_property(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        for i in range(3):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=30))

        result = pe.predict()
        assert result.has_prediction
        assert result.top_commitment is not None
        assert result.commitment_confidence > 0.0

    def test_hold_probability_present(self, engine):
        pe, bm = engine
        bm.on_match_start("m1")
        pe.on_match_start("m1")

        for i in range(3):
            ev = _player_commit_start(CombatCommitment.LIGHT_ATTACK, tick_id=i * 10)
            bm.on_event(ev)
            pe.on_event(ev)
        pe.on_event(_player_commit_end(tick_id=30))

        result = pe.predict()
        # Markov doesn't produce HOLD, so hold_probability should be 0
        assert result.hold_probability == 0.0
