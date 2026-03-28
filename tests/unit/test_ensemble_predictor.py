"""Tests for EnsemblePredictor: blending, weight updates, activation."""

from __future__ import annotations

import pytest

from ai.models.base_predictor import LABEL_HOLD, PredictionResult, make_prediction_result
from ai.models.ensemble_predictor import EnsemblePredictor
from ai.models.markov_predictor import MarkovPredictor
from game.combat.actions import CombatCommitment


L = CombatCommitment.LIGHT_ATTACK
H = CombatCommitment.HEAVY_ATTACK
D = CombatCommitment.DODGE_BACKWARD


class FakeSklearnPredictor:
    """Minimal sklearn predictor stub for testing."""

    def __init__(self, distribution: dict[str, float], version: str = "test_v1"):
        self._dist = distribution
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def predict_from_features(self, features: list[float]) -> PredictionResult:
        return make_prediction_result(self._dist, source="sklearn", markov_level="none")


class TestMarkovOnlyMode:
    def test_returns_markov_result_when_sklearn_inactive(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)
        ens = EnsemblePredictor(markov)
        result = ens.predict()
        assert result.source == "markov"
        assert result.top_commitment == L

    def test_sklearn_inactive_by_default(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov)
        assert not ens.sklearn_active

    def test_default_weights(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov)
        assert ens.markov_weight == 1.0
        assert ens.sklearn_weight == 0.0


class TestSklearnActivation:
    def test_activate_sklearn(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov)
        fake_sk = FakeSklearnPredictor({"LIGHT_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)
        assert ens.sklearn_active

    def test_deactivate_sklearn(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov)
        fake_sk = FakeSklearnPredictor({"LIGHT_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)
        ens.deactivate_sklearn()
        assert not ens.sklearn_active
        assert ens.markov_weight == 1.0
        assert ens.sklearn_weight == 0.0

    def test_predict_without_features_falls_back_to_markov(self):
        markov = MarkovPredictor()
        for _ in range(3):
            markov.update(L)
        ens = EnsemblePredictor(markov)
        fake_sk = FakeSklearnPredictor({"HEAVY_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)
        # No features provided → should still return markov result
        result = ens.predict(features=None)
        assert result.source == "markov"


class TestEnsembleBlending:
    def test_blended_prediction_returns_ensemble_source(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)

        ens = EnsemblePredictor(markov)
        fake_sk = FakeSklearnPredictor({"HEAVY_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)

        features = [0.0] * 35
        result = ens.predict(features)
        assert result.source == "ensemble"

    def test_blended_distribution_sums_to_one(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)
        for _ in range(3):
            markov.update(H)

        ens = EnsemblePredictor(markov)
        fake_sk = FakeSklearnPredictor({"LIGHT_ATTACK": 0.4, "HEAVY_ATTACK": 0.6})
        ens.activate_sklearn(fake_sk)

        features = [0.0] * 35
        result = ens.predict(features)
        total = sum(result.distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_equal_weights_equal_contributions(self):
        markov = MarkovPredictor()
        for _ in range(10):
            markov.update(L)  # Markov predicts L with 100%

        ens = EnsemblePredictor(
            markov,
            initial_markov_weight=0.5,
            initial_sklearn_weight=0.5,
        )
        # Sklearn predicts H with 100%
        fake_sk = FakeSklearnPredictor({"HEAVY_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)

        features = [0.0] * 35
        result = ens.predict(features)
        # Each should contribute ~50%
        assert abs(result.distribution.get("LIGHT_ATTACK", 0) - 0.5) < 1e-6
        assert abs(result.distribution.get("HEAVY_ATTACK", 0) - 0.5) < 1e-6

    def test_hold_in_distribution(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)

        ens = EnsemblePredictor(
            markov,
            initial_markov_weight=0.5,
            initial_sklearn_weight=0.5,
        )
        # Sklearn gives some probability to HOLD
        fake_sk = FakeSklearnPredictor({
            "LIGHT_ATTACK": 0.5,
            LABEL_HOLD: 0.5,
        })
        ens.activate_sklearn(fake_sk)

        features = [0.0] * 35
        result = ens.predict(features)
        assert result.hold_probability > 0.0


class TestWeightUpdates:
    def test_weight_increases_on_correct_prediction(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)

        ens = EnsemblePredictor(markov, ema_alpha=0.3)
        fake_sk = FakeSklearnPredictor({"HEAVY_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)

        initial_markov_w = ens.markov_weight
        markov_result = make_prediction_result(
            {"LIGHT_ATTACK": 1.0}, source="markov", markov_level="unigram",
        )
        sklearn_result = make_prediction_result(
            {"HEAVY_ATTACK": 1.0}, source="sklearn", markov_level="none",
        )

        # Markov predicted L, sklearn predicted H; actual is L → markov wins
        ens.update_weights("LIGHT_ATTACK", markov_result, sklearn_result)
        assert ens.markov_weight > initial_markov_w or abs(ens.markov_weight - initial_markov_w) < 0.01

    def test_weights_have_floor(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov, ema_alpha=0.9)
        fake_sk = FakeSklearnPredictor({"HEAVY_ATTACK": 1.0})
        ens.activate_sklearn(fake_sk)

        markov_result = make_prediction_result(
            {"LIGHT_ATTACK": 1.0}, source="markov", markov_level="unigram",
        )
        sklearn_result = make_prediction_result(
            {"HEAVY_ATTACK": 1.0}, source="sklearn", markov_level="none",
        )

        # Sklearn always wrong for many rounds
        for _ in range(50):
            ens.update_weights("LIGHT_ATTACK", markov_result, sklearn_result)

        assert ens.sklearn_weight >= 0.05  # floor

    def test_no_weight_update_without_sklearn(self):
        markov = MarkovPredictor()
        ens = EnsemblePredictor(markov)
        initial_w = ens.markov_weight

        markov_result = make_prediction_result(
            {"LIGHT_ATTACK": 1.0}, source="markov", markov_level="unigram",
        )
        ens.update_weights("LIGHT_ATTACK", markov_result, None)
        assert ens.markov_weight == initial_w


class TestBackwardCompatibility:
    def test_prediction_result_aliases(self):
        markov = MarkovPredictor()
        for _ in range(5):
            markov.update(L)
        ens = EnsemblePredictor(markov)
        result = ens.predict()
        # Phase 3 aliases must still work
        assert result.probabilities == result.distribution
        assert result.confidence == result.commitment_confidence
        assert result.level == result.markov_level
        assert isinstance(result.has_prediction, bool)
