"""Cross-match session memory with exponential decay.

Tracks aggregate mode outcomes across matches within a single planner
instance (one game session). Unlike PlannerMemory, this is NOT reset
on match start.

All arithmetic is deterministic — no randomness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.strategy.tactics import TacticalIntent


class SessionMemory:
    """Weighted per-mode success tracker across matches.

    Each call to record_match_outcomes() applies exponential decay to
    existing data before adding new outcomes, so recent matches have
    proportionally higher weight.

    Args:
        decay_factor:  Multiplied against existing weights each call (0–1).
                       0.8 means prior data counts 80% as much after one
                       match.
        min_samples:   Minimum total weighted samples (across all modes)
                       before mode_success_rate() returns a non-None value.
    """

    def __init__(
        self,
        decay_factor: float = 0.8,
        min_samples: int = 5,
    ) -> None:
        self._decay_factor = decay_factor
        self._min_samples = min_samples
        self._mode_weighted_successes: dict[str, float] = {}
        self._mode_weighted_totals: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all session state."""
        self._mode_weighted_successes.clear()
        self._mode_weighted_totals.clear()

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_match_outcomes(
        self,
        mode_outcomes: dict[str, tuple[int, int]],
    ) -> None:
        """Decay existing data then record new match outcomes.

        Args:
            mode_outcomes:  {mode_name: (successes, total)} for each mode
                            used this match.  Entries with total <= 0 are
                            ignored.
        """
        # Apply decay to all existing entries first
        for mode_name in list(self._mode_weighted_successes):
            self._mode_weighted_successes[mode_name] *= self._decay_factor
            self._mode_weighted_totals[mode_name] *= self._decay_factor

        # Accumulate new match data
        for mode_name, (successes, total) in mode_outcomes.items():
            if total <= 0:
                continue
            self._mode_weighted_successes[mode_name] = (
                self._mode_weighted_successes.get(mode_name, 0.0) + successes
            )
            self._mode_weighted_totals[mode_name] = (
                self._mode_weighted_totals.get(mode_name, 0.0) + total
            )

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def mode_success_rate(self, mode: "TacticalIntent") -> float | None:
        """Weighted success rate for one mode across the session.

        Returns None when the total session sample count is below
        min_samples, or when this specific mode has no data.
        """
        if self.total_samples() < self._min_samples:
            return None
        total = self._mode_weighted_totals.get(mode.name, 0.0)
        if total < 1.0:
            return None
        successes = self._mode_weighted_successes.get(mode.name, 0.0)
        return successes / total

    def total_samples(self) -> float:
        """Total weighted sample count across all modes."""
        return sum(self._mode_weighted_totals.values())

    def mode_stats(self) -> dict[str, tuple[float, float]]:
        """Return {mode_name: (weighted_successes, weighted_total)}."""
        return {
            name: (
                self._mode_weighted_successes.get(name, 0.0),
                total,
            )
            for name, total in self._mode_weighted_totals.items()
        }
