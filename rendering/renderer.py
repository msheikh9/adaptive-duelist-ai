"""Game renderer: arena, fighters, HUD, overlays.

Reads SimulationState during the RENDER phase. Never mutates state.
All drawing is done to a single pygame surface.

Phase 13 additions:
  - render_title()  — pre-match title / controls screen
  - render() accepts show_help, player_flash, ai_flash for UX polish
  - hit-flash tinting, health-critical and stamina-low color warnings
  - cleaner HUD layout with explicit labels and numeric HP/stamina
  - better match-end overlay (semi-transparent, HP summary)
  - AI tier badge shown in the HUD

Phase 15 additions:
  - Vertical fighter position (fighter.y used for jump height rendering)
  - Jump shadow (ellipse on ground under airborne fighter)
  - Impact particle system (spawn_hit_particles, _update_draw_particles)
  - Screen shake (shake_x/shake_y params, copy-blit technique)
  - New FSM state tints: AIRBORNE, JUMP_STARTUP, LANDING

Phase 17 additions:
  - spawn_hit_particles() accepts kind= ("light","heavy","dodge","whiff")
  - render() accepts player_whiff, ai_whiff, player_dodge_cd, ai_dodge_cd
  - Whiff tint on attacker when whiff flash is active
  - Dodge cooldown pip in HUD below stamina bar
  - Floating damage text (_FloatText, spawn_damage_text)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from config.config_loader import DisplayConfig, GameConfig
from game.combat.actions import FSMState
from game.state import MatchStatus, SimulationState

# Threshold fractions for "warning" colour overrides
_HP_CRITICAL_FRAC    = 0.25   # HP ≤ 25%  → orange bar
_STAMINA_LOW_FRAC    = 0.30   # Stamina ≤ 30% → orange bar


@dataclass
class _Particle:
    """Single impact particle (renderer-side, purely cosmetic)."""
    x: float
    y: float
    vx: float
    vy: float
    lifetime: int
    max_lifetime: int
    color: tuple[int, int, int]


@dataclass
class _FloatText:
    """Floating damage number (renderer-side, purely cosmetic)."""
    x: float
    y: float
    text: str
    lifetime: int
    max_lifetime: int
    color: tuple[int, int, int]


class Renderer:
    """Draws the game state to a pygame window."""

    def __init__(self, game_cfg: GameConfig, display_cfg: DisplayConfig,
                 ai_tier_name: str = "T0") -> None:
        self._gcfg = game_cfg
        self._dcfg = display_cfg
        self._scale = game_cfg.simulation.sub_pixel_scale
        self._ai_tier_name = ai_tier_name

        # Pre-compute ground position for vertical rendering
        self._ground_y_sub: int = (
            game_cfg.arena.ground_y * game_cfg.simulation.sub_pixel_scale
        )

        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._large_font: pygame.font.Font | None = None
        self._tiny_font: pygame.font.Font | None = None

        # Phase 15: particle system
        self._particles: list[_Particle] = []
        self._particle_rng = random.Random(42)  # fixed seed → deterministic sparks

        # Phase 17: floating damage text
        self._float_texts: list[_FloatText] = []

    def init(self) -> pygame.Surface:
        pygame.display.set_caption(self._dcfg.window.title)
        self._screen = pygame.display.set_mode(
            (self._dcfg.window.width, self._dcfg.window.height)
        )
        pygame.font.init()
        hud = self._dcfg.hud
        self._font       = pygame.font.SysFont("monospace", hud.font_size_primary)
        self._small_font = pygame.font.SysFont("monospace", hud.font_size_secondary)
        self._large_font = pygame.font.SysFont("monospace", hud.font_size_large, bold=True)
        self._tiny_font  = pygame.font.SysFont("monospace", 11)
        return self._screen

    # ------------------------------------------------------------------
    # Particle system
    # ------------------------------------------------------------------

    def spawn_hit_particles(self, x_sub: int, y_sub: int, is_heavy: bool,
                            kind: str = "light") -> None:
        """Spawn impact particles at the given sub-pixel position.

        Called by the engine with the defender's sub-pixel coords.
        kind ∈ {"light", "heavy", "dodge", "whiff"}
        """
        if self._screen is None:
            return

        scale   = self._scale
        arena_y = self._arena_y()
        fh      = self._gcfg.fighter.height

        screen_x = 40 + x_sub // scale
        y_offset = (y_sub - self._ground_y_sub) // scale
        screen_y = arena_y - fh // 2 + y_offset  # mid-body of fighter

        # Backward compat: is_heavy=True without explicit kind → heavy burst
        if kind == "light" and is_heavy:
            kind = "heavy"

        # Per-kind particle parameters
        if kind == "heavy":
            n, speed, lifetime = 10, 5.5, 15
            color = (255, 130, 40)   # orange-red
        elif kind == "dodge":
            n, speed, lifetime = 6, 4.0, 11
            color = (80, 200, 220)   # teal/cyan
        elif kind == "whiff":
            n, speed, lifetime = 3, 1.8, 7
            color = (140, 140, 140)  # grey puff
        else:  # "light"
            n, speed, lifetime = 6, 3.2, 10
            color = (255, 255, 160)  # pale yellow

        rng = self._particle_rng
        for i in range(n):
            angle = (2 * math.pi * i / n) + rng.uniform(-0.4, 0.4)
            spd   = speed + rng.uniform(-0.8, 0.8)
            self._particles.append(_Particle(
                x=float(screen_x),
                y=float(screen_y),
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd - 1.0,  # slight upward bias
                lifetime=lifetime,
                max_lifetime=lifetime,
                color=color,
            ))

    def spawn_damage_text(self, amount: int, x_sub: int, y_sub: int,
                          is_heavy: bool) -> None:
        """Spawn a floating damage number above the hit location."""
        if self._screen is None:
            return

        scale   = self._scale
        arena_y = self._arena_y()
        fh      = self._gcfg.fighter.height

        screen_x = float(40 + x_sub // scale)
        y_offset = (y_sub - self._ground_y_sub) // scale
        screen_y = float(arena_y - fh + y_offset - 8)  # above fighter top

        color    = (255, 130, 40) if is_heavy else (255, 255, 160)
        lifetime = 28 if is_heavy else 22
        self._float_texts.append(_FloatText(
            x=screen_x,
            y=screen_y,
            text=f"-{amount}",
            lifetime=lifetime,
            max_lifetime=lifetime,
            color=color,
        ))

    def _update_draw_particles(self, screen: pygame.Surface) -> None:
        """Advance and draw all active particles. Called each render frame."""
        alive: list[_Particle] = []
        for p in self._particles:
            p.lifetime -= 1
            if p.lifetime <= 0:
                continue
            p.x  += p.vx
            p.y  += p.vy
            p.vy += 0.25  # particle gravity
            frac = p.lifetime / p.max_lifetime
            r = max(0, int(p.color[0] * frac))
            g = max(0, int(p.color[1] * frac))
            b = max(0, int(p.color[2] * frac))
            size = max(1, round(3 * frac))
            pygame.draw.circle(screen, (r, g, b), (int(p.x), int(p.y)), size)
            alive.append(p)
        self._particles = alive

    def _update_draw_float_texts(self, screen: pygame.Surface) -> None:
        """Advance and draw all active floating damage texts."""
        if self._tiny_font is None:
            return
        alive: list[_FloatText] = []
        for ft in self._float_texts:
            ft.lifetime -= 1
            if ft.lifetime <= 0:
                continue
            ft.y -= 0.6  # float upward
            frac  = ft.lifetime / ft.max_lifetime
            alpha = max(0, int(255 * frac))
            r = ft.color[0]
            g = ft.color[1]
            b = ft.color[2]
            surf = self._tiny_font.render(ft.text, True, (r, g, b))
            surf.set_alpha(alpha)
            screen.blit(surf, (int(ft.x) - surf.get_width() // 2, int(ft.y)))
            alive.append(ft)
        self._float_texts = alive

    # ------------------------------------------------------------------
    # Main render entry-point
    # ------------------------------------------------------------------

    def render(
        self,
        state: SimulationState,
        *,
        show_help: bool = False,
        player_flash: int = 0,
        ai_flash: int = 0,
        player_whiff: int = 0,
        ai_whiff: int = 0,
        player_dodge_cd: int = 0,
        ai_dodge_cd: int = 0,
        shake_x: int = 0,
        shake_y: int = 0,
    ) -> None:
        """Render one frame of the game.

        Args:
            state:           Current simulation state (read-only).
            show_help:       If True, draw the controls overlay on top.
            player_flash:    Ticks remaining for player hit-flash effect.
            ai_flash:        Ticks remaining for AI hit-flash effect.
            player_whiff:    Ticks remaining for player whiff-flash effect.
            ai_whiff:        Ticks remaining for AI whiff-flash effect.
            player_dodge_cd: Remaining dodge cooldown ticks for the player.
            ai_dodge_cd:     Remaining dodge cooldown ticks for the AI.
            shake_x:         Horizontal screen-shake offset (pixels).
            shake_y:         Vertical screen-shake offset (pixels).
        """
        if self._screen is None:
            return

        screen = self._screen
        colors = self._dcfg.colors

        # ------------------------------------------------------------------
        # Background + arena
        # ------------------------------------------------------------------
        screen.fill(colors.background)
        arena_render_y = self._arena_y()
        self._draw_arena(screen, arena_render_y)

        # ------------------------------------------------------------------
        # Fighters
        # ------------------------------------------------------------------
        max_dodge_cd = self._gcfg.actions.dodge_backward.cooldown_frames
        self._draw_fighter(screen, state.player, colors.player, arena_render_y,
                           "PLAYER", player_flash, whiff_ticks=player_whiff)
        self._draw_fighter(screen, state.ai, colors.ai, arena_render_y,
                           "AI", ai_flash, whiff_ticks=ai_whiff)

        # ------------------------------------------------------------------
        # HUD
        # ------------------------------------------------------------------
        self._draw_hud(screen, state, player_dodge_cd=player_dodge_cd,
                       ai_dodge_cd=ai_dodge_cd, max_dodge_cd=max_dodge_cd)

        # ------------------------------------------------------------------
        # Match-end overlay
        # ------------------------------------------------------------------
        if state.match_status == MatchStatus.ENDED:
            self._draw_match_end(screen, state)

        # ------------------------------------------------------------------
        # Help / controls overlay (drawn last, on top of everything)
        # ------------------------------------------------------------------
        if show_help:
            from ui.screens.help_screen import draw_help_overlay
            draw_help_overlay(screen, self._font, self._small_font, colors)

        # ------------------------------------------------------------------
        # Phase 15: screen shake — shift entire frame
        # ------------------------------------------------------------------
        if shake_x != 0 or shake_y != 0:
            copy = screen.copy()
            screen.fill(colors.background)
            screen.blit(copy, (shake_x, shake_y))

        # ------------------------------------------------------------------
        # Phase 15: impact particles (drawn on top, unaffected by shake)
        # ------------------------------------------------------------------
        self._update_draw_particles(screen)

        # ------------------------------------------------------------------
        # Phase 17: floating damage text (on top of particles)
        # ------------------------------------------------------------------
        self._update_draw_float_texts(screen)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Title screen
    # ------------------------------------------------------------------

    def render_title(self, selected_tier: str = "T2_FULL_ADAPTIVE") -> None:
        """Render the pre-match title / controls + tier-selection screen.

        Args:
            selected_tier: The currently selected AI tier name (for display).
        """
        if self._screen is None:
            return

        screen = self._screen
        colors = self._dcfg.colors
        w, h   = self._dcfg.window.width, self._dcfg.window.height

        screen.fill(colors.background)

        # Subtle floor line for visual grounding
        arena_y = self._arena_y()
        pygame.draw.line(screen, colors.arena_floor, (40, arena_y), (w - 40, arena_y), 2)

        # ---- Title block ----
        title_font = pygame.font.SysFont("monospace", 52, bold=True)
        sub_font   = pygame.font.SysFont("monospace", 18)

        title_surf = title_font.render("ADAPTIVE DUELIST", True, colors.accent)
        sub_surf   = sub_font.render(
            "An AI that learns your fighting style.", True, colors.text_secondary
        )

        title_x = w // 2 - title_surf.get_width() // 2
        title_y = h // 4 - title_surf.get_height() // 2
        screen.blit(title_surf, (title_x, title_y))
        screen.blit(sub_surf,   (w // 2 - sub_surf.get_width() // 2, title_y + 60))

        # ---- Tier selector ----
        self._draw_tier_selector(screen, selected_tier, w // 2, title_y + 108)

        # ---- Controls summary ----
        from ui.screens.help_screen import draw_help_overlay
        draw_help_overlay(screen, self._font, self._small_font, colors)

        # ---- Start prompt (drawn over the overlay's dismiss hint) ----
        cw, ch = self._dcfg.window.width, self._dcfg.window.height
        prompt_font = pygame.font.SysFont("monospace", 17)
        prompt = prompt_font.render(
            "SPACE / ENTER  —  start match", True, colors.text_primary
        )
        px = cw // 2 - prompt.get_width() // 2 - 10
        py = ch - 50
        pygame.draw.rect(screen, colors.panel_bg, (px, py, prompt.get_width() + 20, 28), border_radius=4)
        screen.blit(prompt, (cw // 2 - prompt.get_width() // 2, py + 4))

        pygame.display.flip()

    def _draw_tier_selector(
        self, screen: pygame.Surface, selected_tier: str, cx: int, y: int
    ) -> None:
        """Draw the tier selection widget (← tier name →)."""
        colors = self._dcfg.colors

        _TIER_LABELS: dict[str, str] = {
            "T0_BASELINE":     "T0  Baseline  (random)",
            "T1_MARKOV_ONLY":  "T1  Markov only",
            "T2_FULL_ADAPTIVE": "T2  Full adaptive  ★",
        }
        label = _TIER_LABELS.get(selected_tier, selected_tier)

        tier_surf = self._font.render(label, True, colors.tier_badge)
        arrows    = self._small_font.render("←  /  →  change tier", True, colors.text_secondary)

        box_w  = max(tier_surf.get_width(), arrows.get_width()) + 28
        box_h  = tier_surf.get_height() + arrows.get_height() + 14
        box_x  = cx - box_w // 2
        pygame.draw.rect(screen, colors.panel_bg, (box_x, y, box_w, box_h), border_radius=6)
        pygame.draw.rect(screen, colors.border,   (box_x, y, box_w, box_h), 1, border_radius=6)
        screen.blit(tier_surf, (cx - tier_surf.get_width() // 2, y + 5))
        screen.blit(arrows,    (cx - arrows.get_width() // 2,
                                y + 5 + tier_surf.get_height() + 4))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, screen: pygame.Surface, state: SimulationState,
                  player_dodge_cd: int = 0, ai_dodge_cd: int = 0,
                  max_dodge_cd: int = 45) -> None:
        colors  = self._dcfg.colors
        gcfg    = self._gcfg
        hud     = self._dcfg.hud
        w       = self._dcfg.window.width

        BAR_W   = hud.hp_bar_width
        LEFT_X  = 20
        RIGHT_X = w - BAR_W - 20

        # ---- Player side (left) ----
        self._draw_fighter_hud(
            screen, state.player, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            LEFT_X, 16, "PLAYER", align="left",
            dodge_cd=player_dodge_cd, max_dodge_cd=max_dodge_cd,
        )

        # ---- AI side (right) ----
        self._draw_fighter_hud(
            screen, state.ai, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            RIGHT_X, 16, "AI", align="right",
            dodge_cd=ai_dodge_cd, max_dodge_cd=max_dodge_cd,
        )

        # ---- Center info: tier badge + tick ----
        cx = w // 2
        self._draw_tier_badge(screen, cx, 10)

        tick_surf = self._tiny_font.render(
            f"tick {state.tick_id}", True, colors.text_secondary
        )
        screen.blit(tick_surf, (cx - tick_surf.get_width() // 2, 36))

    def _draw_fighter_hud(
        self,
        screen: pygame.Surface,
        fighter,
        max_hp: int,
        max_stamina: int,
        x: int,
        y: int,
        label: str,
        align: str = "left",
        dodge_cd: int = 0,
        max_dodge_cd: int = 45,
    ) -> None:
        colors = self._dcfg.colors
        hud    = self._dcfg.hud
        BAR_W  = hud.hp_bar_width
        HP_H   = hud.hp_bar_height
        STM_H  = hud.stamina_bar_height

        base_color = colors.player if label == "PLAYER" else colors.ai

        # Name label (with airborne indicator)
        name = label
        if fighter.is_airborne:
            name = label + " ↑"
        name_surf = self._small_font.render(name, True, base_color)
        if align == "right":
            screen.blit(name_surf, (x + BAR_W - name_surf.get_width(), y))
        else:
            screen.blit(name_surf, (x, y))

        bar_y = y + name_surf.get_height() + 2

        # HP bar
        hp_frac = max(0.0, fighter.hp / max_hp)
        hp_fill_color = (
            colors.health_critical if hp_frac <= _HP_CRITICAL_FRAC else colors.hp_bar_fill
        )
        self._draw_bar(screen, x, bar_y, BAR_W, HP_H,
                       hp_frac, hp_fill_color, colors.hp_bar_empty, colors.border)

        # HP numeric
        hp_text = self._tiny_font.render(f"{max(0, fighter.hp)}", True, colors.text_primary)
        if align == "right":
            screen.blit(hp_text, (x + BAR_W - hp_text.get_width() - 3,
                                  bar_y + (HP_H - hp_text.get_height()) // 2))
        else:
            screen.blit(hp_text, (x + 3, bar_y + (HP_H - hp_text.get_height()) // 2))

        stm_y = bar_y + HP_H + 3

        # Stamina bar
        stm_frac = max(0.0, fighter.stamina / max_stamina)
        stm_fill_color = (
            colors.stamina_low if stm_frac <= _STAMINA_LOW_FRAC else colors.stamina_bar_fill
        )
        self._draw_bar(screen, x, stm_y, BAR_W, STM_H,
                       stm_frac, stm_fill_color, colors.stamina_bar_empty, colors.border)

        # Phase 17: dodge cooldown pip (thin bar below stamina)
        dodge_y = stm_y + STM_H + 2
        if max_dodge_cd > 0 and dodge_cd > 0:
            cd_frac = dodge_cd / max_dodge_cd
            # Filled portion shows remaining cooldown (red → fades left as it counts down)
            self._draw_bar(screen, x, dodge_y, BAR_W, 3,
                           cd_frac, (200, 80, 80), (40, 40, 40), colors.border)
        else:
            # Ready: draw a dim green ready indicator
            self._draw_bar(screen, x, dodge_y, BAR_W, 3,
                           1.0, (60, 140, 60), (40, 40, 40), colors.border)

        # FSM state label
        state_y = dodge_y + 3 + 4
        fsm_surf = self._tiny_font.render(
            fighter.fsm_state.name.replace("_", " "), True, colors.text_secondary
        )
        if align == "right":
            screen.blit(fsm_surf, (x + BAR_W - fsm_surf.get_width(), state_y))
        else:
            screen.blit(fsm_surf, (x, state_y))

    def _draw_bar(
        self,
        screen: pygame.Surface,
        x: int, y: int, w: int, h: int,
        frac: float,
        fill_color: tuple,
        empty_color: tuple,
        border_color: tuple,
    ) -> None:
        fill_w = max(0, int(w * frac))
        pygame.draw.rect(screen, empty_color, pygame.Rect(x, y, w, h))
        if fill_w > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(x, y, fill_w, h))
        pygame.draw.rect(screen, border_color, pygame.Rect(x, y, w, h), 1)

    def _draw_tier_badge(self, screen: pygame.Surface, cx: int, y: int) -> None:
        colors    = self._dcfg.colors
        badge_txt = f"AI  {self._ai_tier_name}"
        surf = self._tiny_font.render(badge_txt, True, colors.tier_badge)
        bx = cx - surf.get_width() // 2 - 6
        bw = surf.get_width() + 12
        bh = surf.get_height() + 6
        pygame.draw.rect(screen, colors.panel_bg, (bx, y, bw, bh), border_radius=4)
        pygame.draw.rect(screen, colors.border,   (bx, y, bw, bh), 1, border_radius=4)
        screen.blit(surf, (bx + 6, y + 3))

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------

    def _arena_y(self) -> int:
        return self._dcfg.window.height - 120

    def _draw_arena(self, screen: pygame.Surface, arena_y: int) -> None:
        colors = self._dcfg.colors
        w      = self._dcfg.window.width

        # Main floor line (thicker)
        pygame.draw.rect(screen, colors.arena_floor,
                         pygame.Rect(40, arena_y, w - 80, 3))
        # Subtle secondary line
        pygame.draw.rect(screen, tuple(max(0, c - 15) for c in colors.arena_floor),
                         pygame.Rect(40, arena_y + 3, w - 80, 1))

    # ------------------------------------------------------------------
    # Fighter
    # ------------------------------------------------------------------

    def _draw_fighter(
        self,
        screen: pygame.Surface,
        fighter,
        base_color: tuple,
        arena_y: int,
        label: str,
        flash_ticks: int = 0,
        whiff_ticks: int = 0,
    ) -> None:
        gcfg   = self._gcfg
        colors = self._dcfg.colors
        fw, fh = gcfg.fighter.width, gcfg.fighter.height
        scale  = self._scale

        # Horizontal position (unchanged)
        screen_x = fighter.x // scale
        render_x = 40 + screen_x - fw // 2

        # Phase 15: vertical position from fighter.y relative to ground
        # At ground: fighter.y == ground_y_sub → offset = 0 → render_y = arena_y - fh
        # In air:    fighter.y < ground_y_sub  → offset < 0 → render_y decreases (higher)
        screen_y_offset = (fighter.y - self._ground_y_sub) // scale
        render_y = arena_y - fh + screen_y_offset

        # Phase 15: draw jump shadow on the ground when airborne
        if fighter.fsm_state in (FSMState.JUMP_STARTUP, FSMState.AIRBORNE):
            self._draw_jump_shadow(screen, fighter, arena_y, fw, scale)

        # Tint based on FSM state
        state = fighter.fsm_state
        if flash_ticks > 0:
            draw_color = colors.hit_flash
        elif whiff_ticks > 0:
            # Attacker whiffed: dim desaturated yellow-grey
            draw_color = (160, 150, 80)
        elif state == FSMState.HITSTUN:
            draw_color = (255, 255, 255)
        elif state == FSMState.ATTACK_ACTIVE:
            draw_color = (255, 230, 60)
        elif state == FSMState.AIRBORNE:
            # Slightly brightened while in flight
            draw_color = tuple(min(255, c + 25) for c in base_color)
        elif state == FSMState.JUMP_STARTUP:
            draw_color = tuple(min(255, c + 12) for c in base_color)
        elif state == FSMState.LANDING:
            draw_color = tuple(max(0, c - 25) for c in base_color)
        elif state == FSMState.DODGING:
            draw_color = tuple(min(255, c + 50) for c in base_color)
        elif state == FSMState.EXHAUSTED:
            draw_color = tuple(max(0, c - 60) for c in base_color)
        elif state == FSMState.KO:
            draw_color = (70, 70, 75)
        else:
            draw_color = base_color

        rect = pygame.Rect(render_x, render_y, fw, fh)

        # Body
        pygame.draw.rect(screen, draw_color, rect)
        # Outline (slightly lighter than body for depth)
        outline = tuple(min(255, c + 40) for c in draw_color)
        pygame.draw.rect(screen, outline, rect, 2)

        # KO marker
        if state == FSMState.KO:
            self._draw_ko_mark(screen, render_x, render_y, fw, fh)

        # Facing triangle
        self._draw_facing_triangle(screen, fighter, draw_color, render_x, render_y, fw, fh)

    def _draw_jump_shadow(
        self,
        screen: pygame.Surface,
        fighter,
        arena_y: int,
        fw: int,
        scale: int,
    ) -> None:
        """Draw a fading ground shadow beneath an airborne fighter."""
        height_above_px = max(0, (self._ground_y_sub - fighter.y) // scale)
        # Shadow fades to invisible at 60+ pixels above ground
        alpha = max(0, min(110, 110 - height_above_px * 2))
        if alpha <= 0:
            return
        w = max(8, fw - height_above_px // 3)
        h = max(3, 8 - height_above_px // 10)
        shadow_x = 40 + fighter.x // scale - w // 2
        shadow_y = arena_y - h // 2
        shadow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, alpha), (0, 0, w, h))
        screen.blit(shadow_surf, (shadow_x, shadow_y))

    def _draw_facing_triangle(
        self, screen, fighter, color, rx, ry, fw, fh
    ) -> None:
        cy = ry + fh // 2
        if fighter.facing > 0:
            tri = [(rx + fw, cy - 6), (rx + fw + 12, cy), (rx + fw, cy + 6)]
        else:
            tri = [(rx, cy - 6), (rx - 12, cy), (rx, cy + 6)]
        pygame.draw.polygon(screen, color, tri)
        outline = tuple(min(255, c + 30) for c in color)
        pygame.draw.polygon(screen, outline, tri, 1)

    def _draw_ko_mark(self, screen, rx, ry, fw, fh) -> None:
        """Draw an X over a KO'd fighter."""
        col = (180, 60, 60)
        pad = 12
        pygame.draw.line(screen, col, (rx + pad, ry + pad), (rx + fw - pad, ry + fh - pad), 3)
        pygame.draw.line(screen, col, (rx + fw - pad, ry + pad), (rx + pad, ry + fh - pad), 3)

    # ------------------------------------------------------------------
    # Match-end overlay
    # ------------------------------------------------------------------

    def _draw_match_end(self, screen: pygame.Surface, state: SimulationState) -> None:
        colors = self._dcfg.colors
        w, h   = self._dcfg.window.width, self._dcfg.window.height

        # Semi-transparent dim
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        screen.blit(dim, (0, 0))

        # Winner text
        if state.winner == "PLAYER":
            result_text  = "YOU WIN"
            result_color = colors.player
        elif state.winner == "AI":
            result_text  = "AI WINS"
            result_color = colors.ai
        else:
            result_text  = "DRAW"
            result_color = colors.text_primary

        big_surf = self._large_font.render(result_text, True, result_color)
        bx = w // 2 - big_surf.get_width() // 2
        by = h // 2 - 80
        screen.blit(big_surf, (bx, by))

        # HP summary
        p_hp = max(0, state.player.hp)
        a_hp = max(0, state.ai.hp)
        summary = f"Player {p_hp} HP  ·  AI {a_hp} HP"
        sum_surf = self._small_font.render(summary, True, colors.text_secondary)
        screen.blit(sum_surf, (w // 2 - sum_surf.get_width() // 2, by + big_surf.get_height() + 12))

        # Controls prompt
        prompt = "R  restart       Esc  quit"
        prompt_surf = self._font.render(prompt, True, colors.text_secondary)
        screen.blit(prompt_surf, (w // 2 - prompt_surf.get_width() // 2,
                                   by + big_surf.get_height() + 44))

        # H for help hint (small)
        help_hint = self._tiny_font.render("H  ·  controls", True, colors.border)
        screen.blit(help_hint, (w // 2 - help_hint.get_width() // 2,
                                 by + big_surf.get_height() + 70))
