"""Sound hook interface for combat juice.

Provides a NullSoundManager (all no-ops) as the default implementation.
Replace with a real SoundManager when audio assets are available.

Phase 15: placeholder hooks; wired but silent until sound files are added.
Phase 17: added play_dodge_avoid, play_block, play_dodge_start for outcome clarity.
"""

from __future__ import annotations


class NullSoundManager:
    """Default sound manager: all hooks are no-ops.

    Swap this out with a real implementation once sound files exist.
    The method signatures define the audio event contract.
    """

    def play_hit_light(self) -> None:
        """Light attack connected."""

    def play_hit_heavy(self) -> None:
        """Heavy attack connected."""

    def play_jump(self) -> None:
        """Fighter left the ground."""

    def play_land(self) -> None:
        """Fighter touched down after jumping."""

    def play_whiff(self) -> None:
        """Attack ended without connecting (miss / whiff)."""

    def play_dodge_start(self) -> None:
        """Fighter began a dodge."""

    def play_dodge_avoid(self) -> None:
        """Defender's dodge absorbed an incoming hit (real near-miss)."""

    def play_block(self) -> None:
        """Incoming hit absorbed by block."""

    def play_guard_break(self) -> None:
        """Guard meter depleted — guard break triggered."""
