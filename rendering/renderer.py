"""Minimal game renderer: arena, fighters, HP bars, stamina bars.

Reads SimulationState during the RENDER phase. Never mutates state.
All drawing is done to a single pygame surface.
"""

from __future__ import annotations

import pygame

from game.combat.actions import FSMState
from game.state import SimulationState, MatchStatus
from config.config_loader import GameConfig, DisplayConfig


class Renderer:
    """Draws the game state to a pygame window."""

    def __init__(self, game_cfg: GameConfig, display_cfg: DisplayConfig) -> None:
        self._gcfg = game_cfg
        self._dcfg = display_cfg
        self._scale = game_cfg.simulation.sub_pixel_scale

        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None

    def init(self) -> pygame.Surface:
        pygame.display.set_caption(self._dcfg.window.title)
        self._screen = pygame.display.set_mode(
            (self._dcfg.window.width, self._dcfg.window.height)
        )
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", self._dcfg.hud.font_size_primary)
        self._small_font = pygame.font.SysFont("monospace", self._dcfg.hud.font_size_secondary)
        return self._screen

    def render(self, state: SimulationState) -> None:
        if self._screen is None:
            return

        screen = self._screen
        colors = self._dcfg.colors
        gcfg = self._gcfg

        # Background
        screen.fill(colors.background)

        # Arena floor
        arena_render_y = self._dcfg.window.height - 120
        floor_rect = pygame.Rect(
            40, arena_render_y,
            self._dcfg.window.width - 80, 4,
        )
        pygame.draw.rect(screen, colors.arena_floor, floor_rect)

        # Fighters
        self._draw_fighter(screen, state.player, colors.player,
                           arena_render_y, "PLAYER")
        self._draw_fighter(screen, state.ai, colors.ai,
                           arena_render_y, "AI")

        # HUD
        self._draw_hp_bar(screen, state.player.hp, gcfg.fighter.max_hp,
                          20, 20, colors.hp_bar_fill, colors.hp_bar_empty, "Player")
        self._draw_hp_bar(screen, state.ai.hp, gcfg.fighter.max_hp,
                          self._dcfg.window.width - self._dcfg.hud.hp_bar_width - 20, 20,
                          colors.hp_bar_fill, colors.hp_bar_empty, "AI")

        self._draw_stamina_bar(screen, state.player.stamina, gcfg.fighter.max_stamina,
                               20, 48, colors.stamina_bar_fill, colors.stamina_bar_empty)
        self._draw_stamina_bar(screen, state.ai.stamina, gcfg.fighter.max_stamina,
                               self._dcfg.window.width - self._dcfg.hud.stamina_bar_width - 20,
                               48, colors.stamina_bar_fill, colors.stamina_bar_empty)

        # State labels
        p_state_text = self._font.render(
            state.player.fsm_state.name, True, colors.text_secondary
        )
        screen.blit(p_state_text, (20, 68))

        ai_state_text = self._font.render(
            state.ai.fsm_state.name, True, colors.text_secondary
        )
        ai_text_x = self._dcfg.window.width - ai_state_text.get_width() - 20
        screen.blit(ai_state_text, (ai_text_x, 68))

        # Tick counter
        tick_text = self._small_font.render(
            f"Tick: {state.tick_id}", True, colors.text_secondary
        )
        screen.blit(tick_text, (self._dcfg.window.width // 2 - 40, 10))

        # Match status overlay
        if state.match_status == MatchStatus.ENDED:
            self._draw_match_end(screen, state)

        pygame.display.flip()

    def _draw_fighter(self, screen: pygame.Surface, fighter, color: tuple,
                      arena_y: int, label: str) -> None:
        gcfg = self._gcfg
        fw = gcfg.fighter.width
        fh = gcfg.fighter.height

        # Convert sub-pixel position to screen position
        screen_x = fighter.x // self._scale
        # Center the fighter rectangle on screen_x
        render_x = 40 + screen_x - fw // 2  # 40 = arena left padding
        render_y = arena_y - fh

        # Tint based on state
        draw_color = color
        if fighter.fsm_state == FSMState.HITSTUN:
            draw_color = (255, 255, 255)
        elif fighter.fsm_state == FSMState.DODGING:
            draw_color = tuple(min(255, c + 60) for c in color)
        elif fighter.fsm_state == FSMState.EXHAUSTED:
            draw_color = tuple(c // 2 for c in color)
        elif fighter.fsm_state == FSMState.KO:
            draw_color = (80, 80, 80)
        elif fighter.fsm_state == FSMState.ATTACK_ACTIVE:
            draw_color = (255, 255, 100)

        rect = pygame.Rect(render_x, render_y, fw, fh)
        pygame.draw.rect(screen, draw_color, rect)

        # Facing indicator (small triangle)
        if fighter.facing > 0:
            tri = [
                (render_x + fw, render_y + fh // 2 - 5),
                (render_x + fw + 10, render_y + fh // 2),
                (render_x + fw, render_y + fh // 2 + 5),
            ]
        else:
            tri = [
                (render_x, render_y + fh // 2 - 5),
                (render_x - 10, render_y + fh // 2),
                (render_x, render_y + fh // 2 + 5),
            ]
        pygame.draw.polygon(screen, draw_color, tri)

        # Label
        label_text = self._small_font.render(label, True, color)
        screen.blit(label_text, (render_x + fw // 2 - label_text.get_width() // 2,
                                 render_y - 20))

    def _draw_hp_bar(self, screen: pygame.Surface, hp: int, max_hp: int,
                     x: int, y: int, fill_color: tuple, empty_color: tuple,
                     label: str) -> None:
        w = self._dcfg.hud.hp_bar_width
        h = self._dcfg.hud.hp_bar_height
        fill_w = max(0, int(w * hp / max_hp))

        # Background (empty)
        pygame.draw.rect(screen, empty_color, pygame.Rect(x, y, w, h))
        # Fill
        if fill_w > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(x, y, fill_w, h))
        # Border
        pygame.draw.rect(screen, self._dcfg.colors.text_secondary,
                         pygame.Rect(x, y, w, h), 1)

    def _draw_stamina_bar(self, screen: pygame.Surface, stamina: int, max_stamina: int,
                          x: int, y: int, fill_color: tuple, empty_color: tuple) -> None:
        w = self._dcfg.hud.stamina_bar_width
        h = self._dcfg.hud.stamina_bar_height
        fill_w = max(0, int(w * stamina / max_stamina))

        pygame.draw.rect(screen, empty_color, pygame.Rect(x, y, w, h))
        if fill_w > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(x, y, fill_w, h))
        pygame.draw.rect(screen, self._dcfg.colors.text_secondary,
                         pygame.Rect(x, y, w, h), 1)

    def _draw_match_end(self, screen: pygame.Surface, state: SimulationState) -> None:
        if state.winner == "PLAYER":
            text = "YOU WIN"
            color = self._dcfg.colors.player
        elif state.winner == "AI":
            text = "AI WINS"
            color = self._dcfg.colors.ai
        else:
            text = "DRAW"
            color = self._dcfg.colors.text_primary

        big_font = pygame.font.SysFont("monospace", 48, bold=True)
        rendered = big_font.render(text, True, color)
        x = self._dcfg.window.width // 2 - rendered.get_width() // 2
        y = self._dcfg.window.height // 2 - rendered.get_height() // 2 - 60
        screen.blit(rendered, (x, y))

        restart_text = self._font.render("Press R to restart, ESC to quit", True,
                                         self._dcfg.colors.text_secondary)
        rx = self._dcfg.window.width // 2 - restart_text.get_width() // 2
        screen.blit(restart_text, (rx, y + 70))
