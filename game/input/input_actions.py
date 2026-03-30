"""Vocabulary A: Raw player input actions.

These represent physical keyboard events. They have no game semantics —
an InputAction is a request, not a commitment. The fighter's FSM may
reject the request if the fighter is in a locked state.

Only game/input/ and PlayerFighter.process_input() should reference these.
The AI never sees InputAction.
"""

from enum import Enum, auto


class InputAction(Enum):
    PRESS_LEFT = auto()
    PRESS_RIGHT = auto()
    RELEASE_LEFT = auto()
    RELEASE_RIGHT = auto()
    PRESS_LIGHT_ATTACK = auto()
    PRESS_HEAVY_ATTACK = auto()
    PRESS_DODGE = auto()
    PRESS_BLOCK = auto()
    RELEASE_BLOCK = auto()
    # Phase 15: jump
    PRESS_JUMP = auto()
