"""Tests for game/input/keybind_registry.py."""

from __future__ import annotations

import os

import pygame
import pytest

from game.input.input_actions import InputAction
from game.input.input_map import KEYDOWN_MAP
from game.input.keybind_registry import (
    CATEGORY_ORDER,
    CAT_ATTACKS,
    CAT_DEFENSE,
    CAT_MATCH,
    CAT_MOVEMENT,
    KeybindEntry,
    _ACTION_INFO,
    _MATCH_UI_ENTRIES,
    build_keybind_entries,
    build_keybind_entries_by_category,
)

# pygame.key.name() needs pygame to be initialised
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame.init()


# ---------------------------------------------------------------------------
# KeybindEntry
# ---------------------------------------------------------------------------

class TestKeybindEntry:

    def test_is_frozen(self):
        entry = KeybindEntry(label="Move Left", keys=("A", "←"), category=CAT_MOVEMENT)
        with pytest.raises((AttributeError, TypeError)):
            entry.label = "other"  # type: ignore[misc]

    def test_fields(self):
        entry = KeybindEntry(label="Light Attack", keys=("J",), category=CAT_ATTACKS)
        assert entry.label == "Light Attack"
        assert entry.keys == ("J",)
        assert entry.category == CAT_ATTACKS


# ---------------------------------------------------------------------------
# build_keybind_entries
# ---------------------------------------------------------------------------

class TestBuildKeybindEntries:

    def test_returns_list_of_entries(self):
        entries = build_keybind_entries()
        assert isinstance(entries, list)
        assert all(isinstance(e, KeybindEntry) for e in entries)

    def test_all_assigned_actions_present(self):
        entries = build_keybind_entries()
        labels = {e.label for e in entries}
        # Every action in _ACTION_INFO that has a key in KEYDOWN_MAP must appear
        for action, (label, _) in _ACTION_INFO.items():
            if any(v == action for v in KEYDOWN_MAP.values()):
                assert label in labels, f"Expected '{label}' in help entries"

    def test_no_unassigned_actions(self):
        """Actions in _ACTION_INFO with no KEYDOWN_MAP entry must not appear."""
        # This is implicitly true because current map covers everything,
        # but test the property directly.
        mapped_actions = set(KEYDOWN_MAP.values())
        for action, (label, _) in _ACTION_INFO.items():
            if action not in mapped_actions:
                # must not be in entries
                entries = build_keybind_entries()
                assert all(e.label != label for e in entries), (
                    f"Unassigned action '{label}' should not appear"
                )

    def test_match_ui_entries_appended(self):
        entries = build_keybind_entries()
        match_entries = [e for e in entries if e.category == CAT_MATCH]
        static_labels = {e.label for e in _MATCH_UI_ENTRIES}
        for label in static_labels:
            assert any(e.label == label for e in match_entries), (
                f"Static Match/UI entry '{label}' not found"
            )

    def test_alternate_keys_grouped(self):
        """Actions with multiple keys in KEYDOWN_MAP should appear as one entry
        with multiple keys listed."""
        entries = build_keybind_entries()
        move_left = next(e for e in entries if e.label == "Move Left")
        # KEYDOWN_MAP has both K_a and K_LEFT → two keys in one entry
        assert len(move_left.keys) >= 2, "Move Left should have at least 2 key bindings"

    def test_keys_are_strings(self):
        entries = build_keybind_entries()
        for entry in entries:
            assert all(isinstance(k, str) for k in entry.keys), (
                f"All keys must be strings, got: {entry.keys}"
            )

    def test_categories_are_valid(self):
        valid = set(CATEGORY_ORDER)
        entries = build_keybind_entries()
        for entry in entries:
            assert entry.category in valid, f"Unknown category: {entry.category}"


# ---------------------------------------------------------------------------
# build_keybind_entries_by_category
# ---------------------------------------------------------------------------

class TestBuildByCategory:

    def test_returns_dict(self):
        result = build_keybind_entries_by_category()
        assert isinstance(result, dict)

    def test_canonical_order_respected(self):
        result = build_keybind_entries_by_category()
        keys = list(result.keys())
        # Keys present must appear in CATEGORY_ORDER order
        order_indices = [CATEGORY_ORDER.index(k) for k in keys]
        assert order_indices == sorted(order_indices)

    def test_movement_category_present(self):
        result = build_keybind_entries_by_category()
        assert CAT_MOVEMENT in result

    def test_attacks_category_present(self):
        result = build_keybind_entries_by_category()
        assert CAT_ATTACKS in result

    def test_match_ui_category_present(self):
        result = build_keybind_entries_by_category()
        assert CAT_MATCH in result

    def test_all_entries_in_their_category(self):
        result = build_keybind_entries_by_category()
        for cat, entries in result.items():
            for entry in entries:
                assert entry.category == cat

    def test_empty_categories_omitted(self):
        """Categories with no entries are not included."""
        result = build_keybind_entries_by_category()
        for entries in result.values():
            assert len(entries) > 0

    def test_total_count_matches_flat_list(self):
        flat = build_keybind_entries()
        by_cat = build_keybind_entries_by_category()
        total = sum(len(v) for v in by_cat.values())
        assert total == len(flat)

    def test_deterministic(self):
        r1 = build_keybind_entries_by_category()
        r2 = build_keybind_entries_by_category()
        assert list(r1.keys()) == list(r2.keys())
        for cat in r1:
            assert [e.label for e in r1[cat]] == [e.label for e in r2[cat]]


# ---------------------------------------------------------------------------
# Binding accuracy: sourced from live KEYDOWN_MAP
# ---------------------------------------------------------------------------

class TestBindingAccuracy:

    def test_move_left_key_matches_keydown_map(self):
        entries = build_keybind_entries()
        entry = next(e for e in entries if e.label == "Move Left")
        # Verify keys in entry correspond to actual KEYDOWN_MAP entries
        import pygame
        actual_keys = [
            k for k, v in KEYDOWN_MAP.items()
            if v == InputAction.PRESS_LEFT
        ]
        actual_names = {pygame.key.name(k).upper() for k in actual_keys}
        # At least one key name should overlap (may be normalised differently)
        entry_keys_upper = {k.upper() for k in entry.keys}
        # Allow arrow symbols
        arrow_map = {"←": "LEFT", "→": "RIGHT", "↑": "UP", "↓": "DOWN"}
        normalised = {arrow_map.get(k, k) for k in entry_keys_upper}
        assert normalised & actual_names or entry_keys_upper & actual_names

    def test_escape_key_in_match_ui(self):
        entries = build_keybind_entries()
        quit_entry = next(e for e in entries if e.label == "Quit")
        assert "Esc" in quit_entry.keys

    def test_h_key_in_match_ui(self):
        entries = build_keybind_entries()
        help_entry = next(e for e in entries if "Help" in e.label or "Controls" in e.label)
        assert "H" in help_entry.keys
