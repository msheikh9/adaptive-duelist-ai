"""Translate pygame events into InputAction sequences.

This module is the ONLY place where pygame keyboard events are read.
It produces a list of InputAction per tick, which the player fighter
consumes.
"""

from __future__ import annotations

import pygame

from game.input.input_actions import InputAction
from game.input.input_map import KEYDOWN_MAP, KEYUP_MAP


class InputHandler:
    """Collects pygame keyboard events and converts to InputAction list."""

    def __init__(self) -> None:
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def poll(self) -> list[InputAction]:
        """Read all pending pygame events and return InputActions.

        Must be called once per tick during the INPUT phase.
        """
        actions: list[InputAction] = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                    continue
                mapped = KEYDOWN_MAP.get(event.key)
                if mapped is not None:
                    actions.append(mapped)

            elif event.type == pygame.KEYUP:
                mapped = KEYUP_MAP.get(event.key)
                if mapped is not None:
                    actions.append(mapped)

        return actions
