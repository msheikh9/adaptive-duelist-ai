"""Sklearn-based predictor wrapping a trained RandomForest classifier.

Takes a fitted sklearn model and produces PredictionResults from feature
vectors. Does NOT implement BasePredictor (no update/reset_history) since
it is a stateless inference wrapper — the EnsemblePredictor composes it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai.features.feature_extractor import NUM_FEATURES, extract_features
from ai.models.base_predictor import (
    ALL_LABELS,
    LABEL_HOLD,
    PredictionResult,
    make_prediction_result,
)
from data.events import SemanticEvent

log = logging.getLogger(__name__)


class SklearnPredictor:
    """Wraps a fitted sklearn classifier for commitment prediction.

    Not a BasePredictor subclass — it has no history or update mechanism.
    The EnsemblePredictor calls predict_from_features() directly.
    """

    def __init__(self, clf: Any, version: str) -> None:
        self._clf = clf
        self._version = version
        # Cache the class order from the fitted model
        self._classes: list[str] = list(clf.classes_)

    @property
    def version(self) -> str:
        return self._version

    @property
    def classes(self) -> list[str]:
        return self._classes

    def predict_from_features(self, features: list[float]) -> PredictionResult:
        """Run inference on a single feature vector.

        Returns a PredictionResult with source="sklearn".
        """
        X = np.asarray([features], dtype=np.float32)
        proba = self._clf.predict_proba(X)[0]

        distribution: dict[str, float] = {}
        for cls_name, prob in zip(self._classes, proba):
            distribution[cls_name] = float(prob)

        return make_prediction_result(
            distribution, source="sklearn", markov_level="none",
        )
