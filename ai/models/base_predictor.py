"""Abstract base class for next-commitment predictors.

PredictionResult is the shared output contract for all predictors.
It includes HOLD as a valid label, provides a non-HOLD fallback
commitment for tactical planning, and tracks prediction source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from game.combat.actions import CombatCommitment

LABEL_HOLD = "HOLD"

# Phase 1 commitment names that can appear as labels (plus HOLD).
PHASE1_NAMES: tuple[str, ...] = (
    "MOVE_LEFT", "MOVE_RIGHT",
    "LIGHT_ATTACK", "HEAVY_ATTACK",
    "DODGE_BACKWARD",
)

ALL_LABELS: tuple[str, ...] = PHASE1_NAMES + (LABEL_HOLD,)


@dataclass
class PredictionResult:
    """Output of a predictor for a single prediction query.

    distribution:           full label→prob dict (includes HOLD). sums to 1.0.
    top_label:              argmax of distribution (may be "HOLD").
    top_label_confidence:   distribution[top_label].
    top_commitment:         best non-HOLD CombatCommitment, or None.
    commitment_confidence:  raw (un-renormalized) prob of top_commitment.
    hold_probability:       distribution.get("HOLD", 0.0).
    source:                 "markov" | "sklearn" | "ensemble" | "none".
    markov_level:           "trigram" | "bigram" | "unigram" | "none".
    """

    distribution: dict[str, float]
    top_label: str
    top_label_confidence: float
    top_commitment: CombatCommitment | None
    commitment_confidence: float
    hold_probability: float
    source: str
    markov_level: str

    @property
    def has_prediction(self) -> bool:
        return self.top_commitment is not None and self.commitment_confidence > 0.0

    @property
    def is_hold(self) -> bool:
        return self.top_label == LABEL_HOLD

    @property
    def commitment_prob_normalized(self) -> float:
        """P(top_commitment | player actually commits) renormalized over non-HOLD."""
        denom = 1.0 - self.hold_probability
        if denom <= 0 or self.top_commitment is None:
            return 0.0
        return self.commitment_confidence / denom

    # Backward-compatible aliases used by Phase 3 tests
    @property
    def probabilities(self) -> dict[str, float]:
        return self.distribution

    @property
    def confidence(self) -> float:
        return self.commitment_confidence

    @property
    def level(self) -> str:
        return self.markov_level


def make_prediction_result(
    distribution: dict[str, float],
    source: str = "none",
    markov_level: str = "none",
) -> PredictionResult:
    """Build a PredictionResult from a probability distribution.

    Handles HOLD extraction and non-HOLD fallback construction.
    """
    if not distribution:
        return PredictionResult(
            distribution={},
            top_label=LABEL_HOLD,
            top_label_confidence=0.0,
            top_commitment=None,
            commitment_confidence=0.0,
            hold_probability=0.0,
            source=source,
            markov_level=markov_level,
        )

    top_label = max(distribution, key=distribution.__getitem__)
    top_label_confidence = distribution[top_label]
    hold_probability = distribution.get(LABEL_HOLD, 0.0)

    # Non-HOLD fallback
    non_hold = {k: v for k, v in distribution.items() if k != LABEL_HOLD}
    if non_hold:
        best_name = max(non_hold, key=non_hold.__getitem__)
        try:
            top_commitment = CombatCommitment[best_name]
        except KeyError:
            top_commitment = None
        commitment_confidence = non_hold[best_name] if top_commitment else 0.0
    else:
        top_commitment = None
        commitment_confidence = 0.0

    return PredictionResult(
        distribution=distribution,
        top_label=top_label,
        top_label_confidence=top_label_confidence,
        top_commitment=top_commitment,
        commitment_confidence=commitment_confidence,
        hold_probability=hold_probability,
        source=source,
        markov_level=markov_level,
    )


class BasePredictor(ABC):
    """Shared interface for all commitment predictors."""

    @abstractmethod
    def update(self, commitment: CombatCommitment) -> None:
        """Record a new observed player commitment."""

    @abstractmethod
    def predict(self) -> PredictionResult:
        """Return the predicted next player commitment."""

    @abstractmethod
    def reset_history(self) -> None:
        """Clear per-match history (keep learned counts)."""
