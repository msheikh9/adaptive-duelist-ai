"""Ensemble predictor: blends Markov and sklearn predictions.

Weight dynamics:
  - Before sklearn activation (match_count < sklearn_activation_matches):
    Markov weight=1.0, sklearn weight=0.0. Markov-only.
  - After activation: both contribute. Weights updated per prediction
    via EMA on correctness.

The ensemble merges the two probability distributions via weighted
average, then produces a unified PredictionResult.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai.models.base_predictor import (
    ALL_LABELS,
    LABEL_HOLD,
    PredictionResult,
    make_prediction_result,
)

if TYPE_CHECKING:
    from ai.models.markov_predictor import MarkovPredictor
    from ai.models.sklearn_predictor import SklearnPredictor

log = logging.getLogger(__name__)


class EnsemblePredictor:
    """Blends Markov and sklearn predictions with adaptive weights.

    The ensemble is stateless in terms of history — it delegates
    history tracking to MarkovPredictor and feature computation to
    the caller (PredictionEngine).
    """

    def __init__(
        self,
        markov: MarkovPredictor,
        *,
        initial_markov_weight: float = 1.0,
        initial_sklearn_weight: float = 0.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self._markov = markov
        self._sklearn: SklearnPredictor | None = None

        self._w_markov = initial_markov_weight
        self._w_sklearn = initial_sklearn_weight
        self._ema_alpha = ema_alpha

        self._sklearn_active = False

    # ------------------------------------------------------------------ #
    # Sklearn lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def activate_sklearn(self, sklearn_pred: SklearnPredictor) -> None:
        """Attach and activate the sklearn predictor."""
        self._sklearn = sklearn_pred
        self._sklearn_active = True
        # Give sklearn a starting weight so it contributes
        if self._w_sklearn <= 0.0:
            self._w_sklearn = 0.3
            self._w_markov = 0.7
        log.info(
            "Sklearn activated (v=%s). Weights: markov=%.2f sklearn=%.2f",
            sklearn_pred.version, self._w_markov, self._w_sklearn,
        )

    def deactivate_sklearn(self) -> None:
        """Remove the sklearn predictor. Ensemble falls back to Markov only."""
        self._sklearn = None
        self._sklearn_active = False
        self._w_markov = 1.0
        self._w_sklearn = 0.0

    @property
    def sklearn_active(self) -> bool:
        return self._sklearn_active and self._sklearn is not None

    @property
    def markov_weight(self) -> float:
        return self._w_markov

    @property
    def sklearn_weight(self) -> float:
        return self._w_sklearn

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, features: list[float] | None = None) -> PredictionResult:
        """Produce a blended prediction.

        Parameters
        ----------
        features : Feature vector for sklearn. None if sklearn is inactive
                   or unavailable (Markov-only mode).
        """
        markov_result = self._markov.predict()

        if not self.sklearn_active or features is None or self._sklearn is None:
            return markov_result

        sklearn_result = self._sklearn.predict_from_features(features)

        # Blend distributions
        blended = self._blend_distributions(
            markov_result.distribution,
            sklearn_result.distribution,
        )

        return make_prediction_result(
            blended, source="ensemble", markov_level=markov_result.markov_level,
        )

    def _blend_distributions(
        self,
        markov_dist: dict[str, float],
        sklearn_dist: dict[str, float],
    ) -> dict[str, float]:
        """Weighted average of two probability distributions."""
        w_total = self._w_markov + self._w_sklearn
        if w_total <= 0:
            return markov_dist

        wm = self._w_markov / w_total
        ws = self._w_sklearn / w_total

        # Collect all labels from both distributions
        all_keys = set(markov_dist) | set(sklearn_dist)
        blended: dict[str, float] = {}
        for key in all_keys:
            p_m = markov_dist.get(key, 0.0)
            p_s = sklearn_dist.get(key, 0.0)
            blended[key] = wm * p_m + ws * p_s

        # Renormalize (should be close to 1.0 already)
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    # ------------------------------------------------------------------ #
    # Weight updates                                                       #
    # ------------------------------------------------------------------ #

    def update_weights(
        self,
        actual_label: str,
        markov_result: PredictionResult,
        sklearn_result: PredictionResult | None,
    ) -> None:
        """EMA update on weights based on which predictor was correct.

        Called after observing the actual next commitment. Only updates
        weights when sklearn is active — Markov-only mode has fixed w=1.0.
        """
        if not self.sklearn_active or sklearn_result is None:
            return

        markov_correct = 1.0 if markov_result.top_label == actual_label else 0.0
        sklearn_correct = 1.0 if sklearn_result.top_label == actual_label else 0.0

        alpha = self._ema_alpha
        self._w_markov = (1 - alpha) * self._w_markov + alpha * markov_correct
        self._w_sklearn = (1 - alpha) * self._w_sklearn + alpha * sklearn_correct

        # Floor: don't let either weight go to zero entirely
        self._w_markov = max(self._w_markov, 0.05)
        self._w_sklearn = max(self._w_sklearn, 0.05)
