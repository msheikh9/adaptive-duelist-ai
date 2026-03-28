"""Tests for MarkovPredictor."""

from __future__ import annotations

import pytest

from game.combat.actions import CombatCommitment
from ai.models.markov_predictor import MarkovPredictor
from ai.models.base_predictor import PredictionResult
from ai.profile.player_profile import PlayerProfile


L = CombatCommitment.LIGHT_ATTACK
H = CombatCommitment.HEAVY_ATTACK
D = CombatCommitment.DODGE_BACKWARD
MR = CombatCommitment.MOVE_RIGHT
ML = CombatCommitment.MOVE_LEFT


class TestPredictorBasics:
    def test_no_prediction_before_any_data(self):
        p = MarkovPredictor()
        result = p.predict()
        assert result.level == "none"
        assert result.top_commitment is None
        assert result.confidence == 0.0
        assert not result.has_prediction

    def test_unigram_after_single_commitment(self):
        p = MarkovPredictor()
        p.update(L)
        result = p.predict()
        assert result.level == "unigram"
        assert result.top_commitment == L
        assert result.confidence == 1.0

    def test_probabilities_sum_to_one(self):
        p = MarkovPredictor()
        for c in [L, H, D, L, H, D, MR]:
            p.update(c)
        result = p.predict()
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9

    def test_history_starts_empty(self):
        p = MarkovPredictor()
        assert p.history == []

    def test_history_tracks_recent(self):
        p = MarkovPredictor(order=3)
        p.update(L)
        p.update(H)
        p.update(D)
        p.update(L)
        # Deque maxlen=3; after 4 updates the oldest (L) is dropped
        assert p.history == ["HEAVY_ATTACK", "DODGE_BACKWARD", "LIGHT_ATTACK"]

    def test_reset_history_clears_deque(self):
        p = MarkovPredictor()
        p.update(L)
        p.update(H)
        p.reset_history()
        assert p.history == []
        # Counts should be preserved
        assert p.total_observations == 2

    def test_total_observations_increments(self):
        p = MarkovPredictor()
        for _ in range(5):
            p.update(L)
        assert p.total_observations == 5


class TestFallbackBehavior:
    def test_uses_bigram_over_unigram_when_sufficient(self):
        p = MarkovPredictor(order=2)
        # Build bigram: after L → H appears 3 times
        for _ in range(3):
            p.update(L)
            p.update(H)
        # Sequence: L,H,L,H,L,H → history ends at H
        # Manually position history to end at L for bigram lookup
        p.reset_history()
        p._history.append("LIGHT_ATTACK")
        result = p.predict()
        assert result.level == "bigram"
        assert result.top_commitment == H

    def test_falls_back_to_unigram_when_bigram_sparse(self):
        p = MarkovPredictor(order=2)
        # Only 1 bigram observation (below _MIN_BIGRAM_OBS=2)
        p.update(L)
        p.update(H)
        # Only one bigram L→H seen; history ends at H.
        # bigram for H has 0 observations → fall back to unigram
        result = p.predict()
        # unigram: L=1, H=1 → tied; either is valid
        assert result.level == "unigram"
        assert result.top_commitment in (L, H)

    def test_trigram_preferred_over_bigram(self):
        p = MarkovPredictor(order=3)
        # Build trigram "L,H" → D by repeating L,H,D pattern
        for _ in range(3):
            p.update(L)
            p.update(H)
            p.update(D)
        # After L,H,D × 3, history = [L, H, D]
        # Manually verify: trigram["L,H"] should have D with count 3
        assert p._trigrams.get("LIGHT_ATTACK,HEAVY_ATTACK", {}).get("DODGE_BACKWARD", 0) >= 2
        # Position history so last 2 = [L, H] for trigram key "L,H"
        p.reset_history()
        p._history.extend(["LIGHT_ATTACK", "HEAVY_ATTACK"])
        result = p.predict()
        assert result.level == "trigram"
        assert result.top_commitment == D

    def test_order_1_only_uses_unigram(self):
        p = MarkovPredictor(order=1)
        for _ in range(5):
            p.update(L)
        p.update(H)
        result = p.predict()
        assert result.level == "unigram"
        assert result.top_commitment == L

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            MarkovPredictor(order=4)


class TestMarkovProbabilityCorrectness:
    def test_bigram_probabilities(self):
        p = MarkovPredictor(order=2)
        # After L: H appears 3 times, D appears 1 time → P(H|L)=0.75, P(D|L)=0.25
        for _ in range(3):
            p.update(L)
            p.update(H)
        p.update(L)
        p.update(D)
        # Set history to just [L] for bigram prediction
        p.reset_history()
        p._history.append("LIGHT_ATTACK")
        result = p.predict()
        assert result.level == "bigram"
        assert abs(result.probabilities.get("HEAVY_ATTACK", 0) - 0.75) < 1e-6
        assert abs(result.probabilities.get("DODGE_BACKWARD", 0) - 0.25) < 1e-6

    def test_confidence_equals_top_probability(self):
        p = MarkovPredictor()
        for _ in range(7):
            p.update(L)
        for _ in range(3):
            p.update(H)
        result = p.predict()
        assert abs(result.confidence - result.probabilities[result.top_commitment.name]) < 1e-9


class TestProfileIntegration:
    def test_load_from_profile_seeds_counts(self):
        profile = PlayerProfile()
        profile.action_frequencies = {"LIGHT_ATTACK": 10, "HEAVY_ATTACK": 5}
        profile.bigrams = {"LIGHT_ATTACK": {"DODGE_BACKWARD": 4}}
        profile.trigrams = {"LIGHT_ATTACK,DODGE_BACKWARD": {"MOVE_RIGHT": 2}}

        p = MarkovPredictor()
        p.load_from_profile(profile)

        assert p.total_observations == 15
        assert p._bigrams == {"LIGHT_ATTACK": {"DODGE_BACKWARD": 4}}
        assert p._trigrams == {"LIGHT_ATTACK,DODGE_BACKWARD": {"MOVE_RIGHT": 2}}

    def test_load_from_profile_does_not_alias(self):
        profile = PlayerProfile()
        profile.bigrams = {"LIGHT_ATTACK": {"HEAVY_ATTACK": 3}}

        p = MarkovPredictor()
        p.load_from_profile(profile)

        # Modifying predictor's bigrams must not affect profile
        p._bigrams["LIGHT_ATTACK"]["NEW"] = 999
        assert "NEW" not in profile.bigrams["LIGHT_ATTACK"]

    def test_sync_to_profile(self):
        p = MarkovPredictor()
        p.update(L)
        p.update(H)
        p.update(L)
        p.update(D)

        profile = PlayerProfile()
        p.sync_to_profile(profile)

        assert "LIGHT_ATTACK" in profile.bigrams
        assert "HEAVY_ATTACK" in profile.bigrams["LIGHT_ATTACK"]

    def test_load_empty_profile(self):
        profile = PlayerProfile()
        p = MarkovPredictor()
        p.load_from_profile(profile)
        result = p.predict()
        assert result.level == "none"
