"""Markov chain predictor for next CombatCommitment.

Maintains unigram, bigram, and trigram transition counts over the
player's commitment history. Prediction falls back from trigram →
bigram → unigram so the predictor works from the very first commitment.

The predictor is seeded from a saved PlayerProfile on startup and
continues accumulating counts throughout each session. BehaviorModel
syncs its counts back to the profile before each DB write.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from game.combat.actions import CombatCommitment
from ai.models.base_predictor import BasePredictor, PredictionResult, make_prediction_result

if TYPE_CHECKING:
    from ai.profile.player_profile import PlayerProfile

# Minimum observations required at a given n-gram level before using it.
_MIN_TRIGRAM_OBS = 2
_MIN_BIGRAM_OBS = 2


class MarkovPredictor(BasePredictor):
    """Variable-order Markov chain (order 1–3) for next-commitment prediction.

    Counts are maintained as nested dicts keyed by commitment name strings
    to stay JSON-serializable and consistent with PlayerProfile storage.

    Fallback chain:
      trigram (last 2 → next)  if obs ≥ _MIN_TRIGRAM_OBS
      bigram  (last 1 → next)  if obs ≥ _MIN_BIGRAM_OBS
      unigram (overall freq)   always available after ≥1 observation
    """

    def __init__(self, order: int = 3) -> None:
        if order not in (1, 2, 3):
            raise ValueError(f"Markov order must be 1, 2, or 3; got {order}")
        self._order = order
        self._history: deque[str] = deque(maxlen=order)

        # {prev_name: {next_name: count}}
        self._bigrams: dict[str, dict[str, int]] = {}
        # {"prev2,prev1": {next_name: count}}
        self._trigrams: dict[str, dict[str, int]] = {}
        # {name: count}
        self._unigrams: dict[str, int] = {}
        self._total: int = 0

    # ------------------------------------------------------------------ #
    # BasePredictor interface                                              #
    # ------------------------------------------------------------------ #

    def update(self, commitment: CombatCommitment) -> None:
        """Record a new observed player commitment and update all counts."""
        name = commitment.name

        # Bigram update
        if self._order >= 2 and len(self._history) >= 1:
            prev = self._history[-1]
            inner = self._bigrams.setdefault(prev, {})
            inner[name] = inner.get(name, 0) + 1

        # Trigram update
        if self._order >= 3 and len(self._history) >= 2:
            h = list(self._history)
            tkey = f"{h[-2]},{h[-1]}"
            inner = self._trigrams.setdefault(tkey, {})
            inner[name] = inner.get(name, 0) + 1

        # Unigram update
        self._unigrams[name] = self._unigrams.get(name, 0) + 1
        self._total += 1

        # Advance history
        self._history.append(name)

    def predict(self) -> PredictionResult:
        """Return best-available prediction with normalized probabilities."""
        if self._total == 0:
            return make_prediction_result({}, source="markov", markov_level="none")

        # --- Trigram ---
        if self._order >= 3 and len(self._history) >= 2:
            h = list(self._history)
            tkey = f"{h[-2]},{h[-1]}"
            counts = self._trigrams.get(tkey)
            if counts:
                total = sum(counts.values())
                if total >= _MIN_TRIGRAM_OBS:
                    return self._make_result(counts, total, "trigram")

        # --- Bigram ---
        if self._order >= 2 and len(self._history) >= 1:
            prev = self._history[-1]
            counts = self._bigrams.get(prev)
            if counts:
                total = sum(counts.values())
                if total >= _MIN_BIGRAM_OBS:
                    return self._make_result(counts, total, "bigram")

        # --- Unigram fallback ---
        return self._make_result(self._unigrams, self._total, "unigram")

    def reset_history(self) -> None:
        """Clear the recent-history deque. Counts are preserved."""
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Profile integration                                                  #
    # ------------------------------------------------------------------ #

    def load_from_profile(self, profile: PlayerProfile) -> None:
        """Seed this predictor from a saved profile (deep copy)."""
        self._bigrams = {k: dict(v) for k, v in profile.bigrams.items()}
        self._trigrams = {k: dict(v) for k, v in profile.trigrams.items()}
        self._unigrams = dict(profile.action_frequencies)
        self._total = sum(self._unigrams.values())
        self._history.clear()

    def sync_to_profile(self, profile: PlayerProfile) -> None:
        """Write current counts into the profile for persistence (deep copy)."""
        profile.bigrams = {k: dict(v) for k, v in self._bigrams.items()}
        profile.trigrams = {k: dict(v) for k, v in self._trigrams.items()}
        # action_frequencies is managed by ProfileUpdater; don't overwrite here.

    # ------------------------------------------------------------------ #
    # Inspection                                                           #
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> list[str]:
        return list(self._history)

    @property
    def total_observations(self) -> int:
        return self._total

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_result(counts: dict[str, int], total: int,
                     level: str) -> PredictionResult:
        probs = {k: v / total for k, v in counts.items()}
        return make_prediction_result(probs, source="markov", markov_level=level)
