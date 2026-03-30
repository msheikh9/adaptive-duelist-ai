"""Source-of-truth keybind information for display purposes.

Derives all game keybinds from the live KEYDOWN_MAP so the help screen
always reflects actual bindings. Match/UI controls (R, Esc, H) are
appended as static entries because they are handled at the engine level.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from game.input.input_actions import InputAction
from game.input.input_map import KEYDOWN_MAP

# ---------------------------------------------------------------------------
# Category constants (canonical display order)
# ---------------------------------------------------------------------------

CAT_MOVEMENT = "Movement"
CAT_ATTACKS = "Attacks"
CAT_DEFENSE = "Dodge / Defense"
CAT_MATCH = "Match / UI"

CATEGORY_ORDER: tuple[str, ...] = (CAT_MOVEMENT, CAT_ATTACKS, CAT_DEFENSE, CAT_MATCH)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeybindEntry:
    label: str               # Human-readable action name, e.g. "Move Left"
    keys: tuple[str, ...]    # Human-readable key name(s), e.g. ("A", "←")
    category: str            # Category name for grouping


# ---------------------------------------------------------------------------
# Mapping: InputAction → (display label, category)
# Only actions listed here will appear in the help screen.
# ---------------------------------------------------------------------------

_ACTION_INFO: dict[InputAction, tuple[str, str]] = {
    InputAction.PRESS_LEFT:         ("Move Left",       CAT_MOVEMENT),
    InputAction.PRESS_RIGHT:        ("Move Right",      CAT_MOVEMENT),
    InputAction.PRESS_JUMP:         ("Jump",            CAT_MOVEMENT),
    InputAction.PRESS_LIGHT_ATTACK: ("Light Attack",    CAT_ATTACKS),
    InputAction.PRESS_HEAVY_ATTACK: ("Heavy Attack",    CAT_ATTACKS),
    InputAction.PRESS_DODGE:        ("Dodge Backward",  CAT_DEFENSE),
    InputAction.PRESS_BLOCK:        ("Block",           CAT_DEFENSE),
}

# Static entries for engine-level controls not in KEYDOWN_MAP
_MATCH_UI_ENTRIES: tuple[KeybindEntry, ...] = (
    KeybindEntry(label="Restart match",    keys=("R",),   category=CAT_MATCH),
    KeybindEntry(label="Quit",             keys=("Esc",), category=CAT_MATCH),
    KeybindEntry(label="Controls / Help",  keys=("H",),   category=CAT_MATCH),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: dict[str, str] = {
    "left":       "←",
    "right":      "→",
    "up":         "↑",
    "down":       "↓",
    "space":      "Space",
    "return":     "Enter",
    "escape":     "Esc",
    "backspace":  "Bksp",
    "tab":        "Tab",
}


def _key_display(key_const: int) -> str:
    """Return a human-readable display name for a pygame key constant."""
    name = pygame.key.name(key_const)
    return _DISPLAY_NAMES.get(name.lower(), name.upper())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_keybind_entries() -> list[KeybindEntry]:
    """Build keybind display entries derived from the live KEYDOWN_MAP.

    - Groups alternate keys for the same action into one entry.
    - Omits any action that has no assigned key.
    - Appends static Match/UI entries at the end.
    """
    action_keys: dict[InputAction, list[str]] = {}
    for key_const, action in KEYDOWN_MAP.items():
        if action not in _ACTION_INFO:
            continue
        action_keys.setdefault(action, []).append(_key_display(key_const))

    entries: list[KeybindEntry] = []
    for action, (label, category) in _ACTION_INFO.items():
        keys = action_keys.get(action)
        if not keys:
            continue  # unassigned — do not show
        entries.append(KeybindEntry(label=label, keys=tuple(keys), category=category))

    entries.extend(_MATCH_UI_ENTRIES)
    return entries


def build_keybind_entries_by_category() -> dict[str, list[KeybindEntry]]:
    """Return entries grouped by category in canonical display order.

    Categories with no entries are omitted.
    """
    all_entries = build_keybind_entries()
    result: dict[str, list[KeybindEntry]] = {}
    for cat in CATEGORY_ORDER:
        cat_entries = [e for e in all_entries if e.category == cat]
        if cat_entries:
            result[cat] = cat_entries
    return result
