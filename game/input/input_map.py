"""Keybinding definitions: pygame key constants → InputAction mapping."""

from __future__ import annotations

import pygame

from game.input.input_actions import InputAction

# Default keyboard mappings.
# KEYDOWN events:
KEYDOWN_MAP: dict[int, InputAction] = {
    pygame.K_a: InputAction.PRESS_LEFT,
    pygame.K_LEFT: InputAction.PRESS_LEFT,
    pygame.K_d: InputAction.PRESS_RIGHT,
    pygame.K_RIGHT: InputAction.PRESS_RIGHT,
    pygame.K_j: InputAction.PRESS_LIGHT_ATTACK,
    pygame.K_z: InputAction.PRESS_LIGHT_ATTACK,
    pygame.K_k: InputAction.PRESS_HEAVY_ATTACK,
    pygame.K_x: InputAction.PRESS_HEAVY_ATTACK,
    pygame.K_SPACE: InputAction.PRESS_DODGE,
    pygame.K_l: InputAction.PRESS_BLOCK,
    pygame.K_c: InputAction.PRESS_BLOCK,
    # Phase 15: jump
    pygame.K_w: InputAction.PRESS_JUMP,
    pygame.K_UP: InputAction.PRESS_JUMP,
    # Phase 20: ranged weapon (E / Right-Ctrl; K is already heavy attack)
    pygame.K_e: InputAction.PRESS_SHOOT,
    pygame.K_RCTRL: InputAction.PRESS_SHOOT,
}

# KEYUP events:
KEYUP_MAP: dict[int, InputAction] = {
    pygame.K_a: InputAction.RELEASE_LEFT,
    pygame.K_LEFT: InputAction.RELEASE_LEFT,
    pygame.K_d: InputAction.RELEASE_RIGHT,
    pygame.K_RIGHT: InputAction.RELEASE_RIGHT,
    pygame.K_l: InputAction.RELEASE_BLOCK,
    pygame.K_c: InputAction.RELEASE_BLOCK,
    # Phase 20: release to fire
    pygame.K_e: InputAction.RELEASE_SHOOT,
    pygame.K_RCTRL: InputAction.RELEASE_SHOOT,
}
