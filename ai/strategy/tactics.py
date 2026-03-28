"""Vocabulary C: AI tactical intents.

TacticalIntent represents what the AI wants to achieve strategically.
These exist only inside ai/ and are never seen by the game engine.
The ActionResolver translates a TacticalIntent into a CombatCommitment.
"""

from enum import Enum, auto


class TacticalIntent(Enum):
    EXPLOIT_PATTERN = auto()
    BAIT_AND_PUNISH = auto()
    PUNISH_RECOVERY = auto()
    PRESSURE_STAMINA = auto()
    DEFENSIVE_RESET = auto()
    PROBE_BEHAVIOR = auto()
    NEUTRAL_SPACING = auto()
