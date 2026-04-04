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

Phase 16 additions:
  - spawn_hit_particles() adds kind="land" (landing dust at floor level)
  - render() accepts player_combo, ai_combo, player_combo_flash, ai_combo_flash
  - _draw_fighter_hud() renders combo counter (×N) below FSM label
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
    """Floating damage number or popup text (renderer-side, purely cosmetic)."""
    x: float
    y: float
    text: str
    lifetime: int
    max_lifetime: int
    color: tuple[int, int, int]
    large: bool = False   # Phase 19: use _font (18px) instead of _tiny_font (11px)


@dataclass
class _Ring:
    """Expanding impact ring (renderer-side, purely cosmetic)."""
    x: float
    y: float
    radius: float
    max_radius: float
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

        # Phase 19: impact rings (heavy hits / guard breaks)
        self._rings: list[_Ring] = []

        # Phase 19: attack trails — list of (screen_x, screen_y, age_frames)
        # Separate trail per fighter; updated each render call.
        self._player_trail: list[tuple[float, float, int]] = []
        self._ai_trail:     list[tuple[float, float, int]] = []

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
        kind ∈ {"light", "heavy", "dodge", "whiff", "land"}
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
        vy_bias  = -1.0   # default: slight upward bias
        ring_r   = 0      # non-zero → also spawn an expanding ring
        ring_col = (255, 255, 255)
        if kind == "heavy":
            n, speed, lifetime = 14, 6.0, 17   # Phase 19: more particles
            color   = (255, 130, 40)   # orange-red
            ring_r  = 40               # Phase 19: impact ring
            ring_col = (255, 160, 60)
        elif kind == "dodge":
            n, speed, lifetime = 6, 4.0, 11
            color = (80, 200, 220)   # teal/cyan
        elif kind == "whiff":
            n, speed, lifetime = 6, 2.4, 9    # Phase 19: more visible whiff
            color = (160, 160, 160)  # brighter grey puff
        elif kind == "land":
            # Phase 16: landing dust — spawns at ground level, spreads outward
            n, speed, lifetime = 8, 2.0, 9
            color    = (160, 140, 100)  # dusty tan
            screen_y = arena_y          # override to floor line
            vy_bias  = -0.3             # mostly horizontal spread
        elif kind == "block":
            # Phase 18: block absorption — teal sparks, inward burst
            n, speed, lifetime = 8, 3.5, 11
            color = (60, 210, 230)   # cyan/teal
        elif kind == "guard_break":
            # Phase 18/19: guard break — large burst with ring
            n, speed, lifetime = 18, 7.0, 20
            color    = (255, 90, 30)    # orange-red
            ring_r   = 60
            ring_col = (255, 60, 20)
        elif kind == "combo_ring":
            # Phase 20: combo pulse — accent ring at attacker, no particles
            n, speed, lifetime = 0, 0.0, 0
            color    = (160, 120, 240)  # accent purple (unused, n=0)
            ring_r   = 34
            ring_col = (160, 120, 240)  # accent purple ring
        else:  # "light"
            n, speed, lifetime = 8, 3.5, 11   # Phase 19: slightly more
            color = (255, 255, 160)  # pale yellow

        rng = self._particle_rng
        for i in range(n):
            angle = (2 * math.pi * i / n) + rng.uniform(-0.4, 0.4)
            spd   = speed + rng.uniform(-0.8, 0.8)
            self._particles.append(_Particle(
                x=float(screen_x),
                y=float(screen_y),
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd + vy_bias,
                lifetime=lifetime,
                max_lifetime=lifetime,
                color=color,
            ))

        # Phase 19: spawn impact ring for heavy hits / guard breaks
        if ring_r > 0:
            self._rings.append(_Ring(
                x=float(screen_x),
                y=float(screen_y),
                radius=4.0,
                max_radius=float(ring_r),
                lifetime=12,
                max_lifetime=12,
                color=ring_col,
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

    def spawn_text_popup(self, text: str, x_sub: int, y_sub: int,
                        is_large: bool = False) -> None:
        """Spawn a floating text popup (MISS, GUARD BREAK!, combo milestones).

        Phase 19: large=True uses the 18px font; False uses 11px tiny font.
        """
        if self._screen is None:
            return

        scale   = self._scale
        arena_y = self._arena_y()
        fh      = self._gcfg.fighter.height

        screen_x = float(40 + x_sub // scale)
        y_offset = (y_sub - self._ground_y_sub) // scale
        # Slightly above the fighter, extra clearance for large text
        y_lift = fh + (18 if is_large else 8)
        screen_y = float(arena_y - y_lift + y_offset)

        if is_large:
            color    = (255, 220, 60)   # bright yellow for milestones / guard break
            lifetime = 50
        else:
            color    = (200, 200, 200)  # grey for MISS
            lifetime = 28

        self._float_texts.append(_FloatText(
            x=screen_x,
            y=screen_y,
            text=text,
            lifetime=lifetime,
            max_lifetime=lifetime,
            color=color,
            large=is_large,
        ))

    def _get_weapon_tip(self, fighter, arena_y: int) -> tuple[float, float] | None:
        """Return screen-space (x, y) of the weapon tip during attack states.

        Returns None if the fighter is not in an attack state.
        """
        from game.combat.actions import CombatCommitment
        state = fighter.fsm_state
        if state not in (FSMState.ATTACK_STARTUP, FSMState.ATTACK_ACTIVE,
                         FSMState.ATTACK_RECOVERY):
            return None

        commitment = fighter.active_commitment
        if commitment == CombatCommitment.LIGHT_ATTACK:
            reach = self._gcfg.actions.light_attack.reach
        elif commitment == CombatCommitment.HEAVY_ATTACK:
            reach = self._gcfg.actions.heavy_attack.reach
        else:
            return None

        gcfg   = self._gcfg
        fw, fh = gcfg.fighter.width, gcfg.fighter.height
        scale  = self._scale

        screen_x = fighter.x // scale
        render_x = 40 + screen_x - fw // 2
        y_offset = (fighter.y - self._ground_y_sub) // scale
        mid_y    = float(arena_y - fh + y_offset + fh // 2)

        if fighter.facing > 0:
            tip_x = float(render_x + fw + reach)
        else:
            tip_x = float(render_x - reach)

        return tip_x, mid_y

    def _update_attack_trail(
        self,
        trail: list[tuple[float, float, int]],
        fighter,
        arena_y: int,
    ) -> list[tuple[float, float, int]]:
        """Age trail points and optionally append the current weapon tip.

        New tip is appended only while the fighter is in attack startup or
        active frames.  During recovery the trail fades naturally without
        receiving new points.  Points older than _TRAIL_MAX_AGE are discarded.
        """
        _TRAIL_MAX_AGE = 9

        # Age existing points; drop stale ones
        aged = [(x, y, age + 1) for x, y, age in trail if age < _TRAIL_MAX_AGE]

        tip = self._get_weapon_tip(fighter, arena_y)
        if tip is not None and fighter.fsm_state in (
                FSMState.ATTACK_STARTUP, FSMState.ATTACK_ACTIVE):
            aged.append((tip[0], tip[1], 0))

        return aged

    def _draw_attack_trail(
        self,
        screen: pygame.Surface,
        trail: list[tuple[float, float, int]],
        commitment,
    ) -> None:
        """Draw fading line segments connecting weapon-tip trail points."""
        if len(trail) < 2:
            return

        from game.combat.actions import CombatCommitment
        is_heavy  = (commitment == CombatCommitment.HEAVY_ATTACK)
        base_r, base_g, base_b = (255, 120, 30) if is_heavy else (255, 240, 80)

        for i in range(len(trail) - 1):
            x1, y1, age1 = trail[i]
            x2, y2, age2 = trail[i + 1]
            frac  = max(0.0, 1.0 - age1 / 9.0)
            r     = int(base_r * frac)
            g     = int(base_g * frac)
            b     = int(base_b * frac)
            width = max(1, 3 if is_heavy else 2)
            pygame.draw.line(screen, (r, g, b),
                             (int(x1), int(y1)), (int(x2), int(y2)), width)

    def _update_draw_rings(self, screen: pygame.Surface) -> None:
        """Advance and draw all active impact rings."""
        alive: list[_Ring] = []
        for ring in self._rings:
            ring.lifetime -= 1
            if ring.lifetime <= 0:
                continue
            frac   = ring.lifetime / ring.max_lifetime
            ring.radius += (ring.max_radius - ring.radius) * 0.25
            alpha  = max(0, int(200 * frac))
            r = max(0, int(ring.color[0] * frac))
            g = max(0, int(ring.color[1] * frac))
            b = max(0, int(ring.color[2] * frac))
            thickness = max(1, round(3 * frac))
            surf = pygame.Surface(
                (int(ring.radius * 2 + 4), int(ring.radius * 2 + 4)),
                pygame.SRCALPHA,
            )
            cx = surf.get_width() // 2
            cy = surf.get_height() // 2
            pygame.draw.circle(surf, (r, g, b, alpha), (cx, cy),
                               max(1, int(ring.radius)), thickness)
            screen.blit(surf, (int(ring.x - ring.radius - 2),
                               int(ring.y - ring.radius - 2)))
            alive.append(ring)
        self._rings = alive

    def _draw_hitbox_overlay(
        self,
        screen: pygame.Surface,
        state,
        arena_y: int,
    ) -> None:
        """Draw attack hitboxes and fighter hurtboxes for debug inspection.

        Phase 19: toggled by F1.  Uses semi-transparent SRCALPHA surfaces.
        Attack hitbox = red; hurtbox = cyan.
        """
        from game.combat.hitbox import get_attack_hitbox, get_fighter_hurtbox

        scale    = self._scale
        gcfg     = self._gcfg
        y_origin = arena_y  # screen y for ground (y_sub = ground_y_sub)

        def sub_rect_to_screen(hbox) -> pygame.Rect:
            # sub-pixel hitbox → screen pixels
            sx = 40 + hbox.x_min // scale
            # y grows downward on screen but upward in sub-pixel space
            # y_sub=0 is at screen bottom, y_sub=ground_y_sub is at arena_y
            sy = y_origin + (hbox.y_min - self._ground_y_sub) // scale
            sw = (hbox.x_max - hbox.x_min) // scale
            sh = (hbox.y_max - hbox.y_min) // scale
            return pygame.Rect(sx, sy, max(1, sw), max(1, sh))

        hit_surf  = pygame.Surface(
            (self._dcfg.window.width, self._dcfg.window.height), pygame.SRCALPHA)
        hurt_surf = pygame.Surface(
            (self._dcfg.window.width, self._dcfg.window.height), pygame.SRCALPHA)

        for fighter in (state.player, state.ai):
            # Hurtbox (cyan, semi-transparent fill)
            hbox = get_fighter_hurtbox(fighter, gcfg)
            r    = sub_rect_to_screen(hbox)
            pygame.draw.rect(hurt_surf, (0, 220, 220, 35), r)
            pygame.draw.rect(hurt_surf, (0, 220, 220, 160), r, 1)

            # Attack hitbox (red), only during ATTACK_ACTIVE
            ahbox = get_attack_hitbox(fighter, gcfg)
            if ahbox is not None:
                ar = sub_rect_to_screen(ahbox)
                pygame.draw.rect(hit_surf, (255, 60, 60, 55), ar)
                pygame.draw.rect(hit_surf, (255, 60, 60, 200), ar, 2)

        screen.blit(hurt_surf, (0, 0))
        screen.blit(hit_surf,  (0, 0))

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
        """Advance and draw all active floating texts.

        Phase 19: large=True uses _font (18px) for milestone / guard-break
        popups; small uses _tiny_font (11px) for damage numbers / MISS.
        """
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
            font  = self._font if (ft.large and self._font is not None) \
                    else self._tiny_font
            surf  = font.render(ft.text, True, (r, g, b))
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
        player_heavy_cd: int = 0,
        ai_heavy_cd: int = 0,
        shake_x: int = 0,
        shake_y: int = 0,
        player_combo: int = 0,
        ai_combo: int = 0,
        player_combo_flash: int = 0,
        ai_combo_flash: int = 0,
        player_guard: int = 100,
        ai_guard: int = 100,
        player_block_flash: int = 0,
        ai_block_flash: int = 0,
        player_guard_break_flash: int = 0,
        ai_guard_break_flash: int = 0,
        show_hitboxes: bool = False,
    ) -> None:
        """Render one frame of the game.

        Args:
            state:            Current simulation state (read-only).
            show_help:        If True, draw the controls overlay on top.
            player_flash:     Ticks remaining for player hit-flash effect.
            ai_flash:         Ticks remaining for AI hit-flash effect.
            player_whiff:     Ticks remaining for player whiff-flash effect.
            ai_whiff:         Ticks remaining for AI whiff-flash effect.
            player_dodge_cd:  Remaining dodge cooldown ticks for the player.
            ai_dodge_cd:      Remaining dodge cooldown ticks for the AI.
            player_heavy_cd:  Remaining heavy attack cooldown ticks for the player.
            ai_heavy_cd:      Remaining heavy attack cooldown ticks for the AI.
            shake_x:          Horizontal screen-shake offset (pixels).
            shake_y:          Vertical screen-shake offset (pixels).
            player_combo:     Player's current hit streak.
            ai_combo:         AI's current hit streak.
            player_combo_flash: Display frames remaining for player combo emphasis.
            ai_combo_flash:   Display frames remaining for AI combo emphasis.
            player_guard:     Player's current guard meter value.
            ai_guard:         AI's current guard meter value.
            player_block_flash:       Display frames for block-absorbed-hit flash.
            ai_block_flash:           Display frames for AI block-absorbed-hit flash.
            player_guard_break_flash: Display frames for player guard break flash.
            ai_guard_break_flash:     Display frames for AI guard break flash.
            show_hitboxes:            If True, draw hitbox/hurtbox debug overlay.
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
        # Phase 19: update attack trails (before drawing fighters)
        # ------------------------------------------------------------------
        self._player_trail = self._update_attack_trail(
            self._player_trail, state.player, arena_render_y)
        self._ai_trail = self._update_attack_trail(
            self._ai_trail, state.ai, arena_render_y)

        # ------------------------------------------------------------------
        # Fighters
        # ------------------------------------------------------------------
        max_guard = self._gcfg.actions.block.guard_max
        max_dodge_cd = self._gcfg.actions.dodge_backward.cooldown_frames
        self._draw_fighter(screen, state.player, colors.player, arena_render_y,
                           "PLAYER", player_flash, whiff_ticks=player_whiff,
                           block_flash=player_block_flash,
                           guard_break_flash=player_guard_break_flash)
        self._draw_fighter(screen, state.ai, colors.ai, arena_render_y,
                           "AI", ai_flash, whiff_ticks=ai_whiff,
                           block_flash=ai_block_flash,
                           guard_break_flash=ai_guard_break_flash)

        # ------------------------------------------------------------------
        # Phase 19: attack trails (drawn after fighters, before HUD)
        # ------------------------------------------------------------------
        self._draw_attack_trail(screen, self._player_trail,
                                state.player.active_commitment)
        self._draw_attack_trail(screen, self._ai_trail,
                                state.ai.active_commitment)

        # ------------------------------------------------------------------
        # Phase 19: hitbox debug overlay (drawn after fighters)
        # ------------------------------------------------------------------
        if show_hitboxes:
            self._draw_hitbox_overlay(screen, state, arena_render_y)

        # ------------------------------------------------------------------
        # HUD
        # ------------------------------------------------------------------
        self._draw_hud(screen, state,
                       player_dodge_cd=player_dodge_cd, ai_dodge_cd=ai_dodge_cd,
                       max_dodge_cd=max_dodge_cd,
                       player_heavy_cd=player_heavy_cd, ai_heavy_cd=ai_heavy_cd,
                       player_guard=player_guard, ai_guard=ai_guard,
                       max_guard=max_guard)

        # ------------------------------------------------------------------
        # Combo corner display (secondary — always visible at 2+)
        # ------------------------------------------------------------------
        self._draw_combo_corner(screen, player_combo, player_combo_flash, align="left")
        self._draw_combo_corner(screen, ai_combo,     ai_combo_flash,     align="right")

        # ------------------------------------------------------------------
        # Phase 20: Big center combo text (primary, shown at 3+)
        # ------------------------------------------------------------------
        self._draw_big_combo(screen, player_combo, ai_combo,
                             player_combo_flash, ai_combo_flash)

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
        # Phase 19: impact rings (on top of particles)
        # ------------------------------------------------------------------
        self._update_draw_rings(screen)

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
                  max_dodge_cd: int = 45,
                  player_heavy_cd: int = 0, ai_heavy_cd: int = 0,
                  player_guard: int = 100, ai_guard: int = 100,
                  max_guard: int = 100) -> None:
        colors  = self._dcfg.colors
        gcfg    = self._gcfg
        hud     = self._dcfg.hud
        w       = self._dcfg.window.width

        BAR_W         = hud.hp_bar_width
        LEFT_X        = 20
        RIGHT_X       = w - BAR_W - 20
        max_heavy_cd  = gcfg.actions.heavy_attack.cooldown_ticks

        # ---- Player side (left) ----
        self._draw_fighter_hud(
            screen, state.player, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            LEFT_X, 16, "PLAYER", align="left",
            dodge_cd=player_dodge_cd, max_dodge_cd=max_dodge_cd,
            heavy_cd=player_heavy_cd, max_heavy_cd=max_heavy_cd,
            guard=player_guard, max_guard=max_guard,
        )

        # ---- AI side (right) ----
        self._draw_fighter_hud(
            screen, state.ai, gcfg.fighter.max_hp, gcfg.fighter.max_stamina,
            RIGHT_X, 16, "AI", align="right",
            dodge_cd=ai_dodge_cd, max_dodge_cd=max_dodge_cd,
            heavy_cd=ai_heavy_cd, max_heavy_cd=max_heavy_cd,
            guard=ai_guard, max_guard=max_guard,
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
        heavy_cd: int = 0,
        max_heavy_cd: int = 0,
        guard: int = 100,
        max_guard: int = 100,
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
            self._draw_bar(screen, x, dodge_y, BAR_W, 3,
                           cd_frac, (200, 80, 80), (40, 40, 40), colors.border)
        else:
            self._draw_bar(screen, x, dodge_y, BAR_W, 3,
                           1.0, (60, 140, 60), (40, 40, 40), colors.border)

        # Phase 17b: heavy attack cooldown pip (below dodge pip)
        heavy_y = dodge_y + 3 + 2
        if max_heavy_cd > 0 and heavy_cd > 0:
            h_frac = heavy_cd / max_heavy_cd
            self._draw_bar(screen, x, heavy_y, BAR_W, 3,
                           h_frac, (220, 140, 40), (40, 40, 40), colors.border)
        elif max_heavy_cd > 0:
            self._draw_bar(screen, x, heavy_y, BAR_W, 3,
                           1.0, (60, 140, 60), (40, 40, 40), colors.border)

        # Phase 18: guard bar (6px, teal when healthy, red when critical/broken)
        guard_y = heavy_y + 3 + 2
        if max_guard > 0:
            g_frac = max(0.0, guard / max_guard)
            # Teal/cyan when ≥25%, orange-red when low, dark red when 0
            if g_frac <= 0.0:
                g_fill = (160, 30, 30)   # broken: dark red
            elif g_frac <= 0.25:
                g_fill = (220, 80, 40)   # critical: orange-red
            else:
                g_fill = (40, 190, 200)  # healthy: teal/cyan
            self._draw_bar(screen, x, guard_y, BAR_W, 6,
                           g_frac, g_fill, (30, 30, 40), colors.border)

        # FSM state label
        state_y = guard_y + 6 + 4
        fsm_surf = self._tiny_font.render(
            fighter.fsm_state.name.replace("_", " "), True, colors.text_secondary
        )
        if align == "right":
            screen.blit(fsm_surf, (x + BAR_W - fsm_surf.get_width(), state_y))
        else:
            screen.blit(fsm_surf, (x, state_y))

    def _draw_combo_corner(
        self,
        screen: pygame.Surface,
        combo: int,
        combo_flash: int,
        align: str,
    ) -> None:
        """Draw the arcade-style combo counter in the upper-left or upper-right corner.

        Shown only when combo >= 2. At 3+ uses accent color with a highlighted
        border. combo_flash > 0 draws a thicker border for the pulse effect.
        """
        if combo < 2 or self._large_font is None or self._small_font is None:
            return

        colors  = self._dcfg.colors
        w_win   = self._dcfg.window.width

        use_accent   = combo >= 3
        combo_color  = colors.accent if use_accent else colors.text_secondary

        number_surf = self._large_font.render(f"\xd7{combo}", True, combo_color)
        label_surf  = self._small_font.render("COMBO", True, combo_color)

        PAD     = 10
        panel_w = max(number_surf.get_width(), label_surf.get_width()) + PAD * 2
        panel_h = number_surf.get_height() + label_surf.get_height() + PAD + 6

        # Vertical anchor: just below the HUD bars (~120px from top)
        panel_y = 120

        if align == "left":
            panel_x = 20
            num_x   = panel_x + PAD
            lbl_x   = panel_x + PAD
        else:
            panel_x = w_win - 20 - panel_w
            num_x   = panel_x + panel_w - PAD - number_surf.get_width()
            lbl_x   = panel_x + panel_w - PAD - label_surf.get_width()

        # Background — brighter panel when flashing
        bg_color = (35, 25, 55) if (combo_flash > 0 and use_accent) else colors.panel_bg
        pygame.draw.rect(screen, bg_color,
                         (panel_x, panel_y, panel_w, panel_h), border_radius=6)

        # Border: thicker + accent color while flashing or when 3+
        border_color  = colors.accent if (use_accent or combo_flash > 0) else colors.border
        border_width  = 2 if combo_flash > 0 else 1
        pygame.draw.rect(screen, border_color,
                         (panel_x, panel_y, panel_w, panel_h),
                         border_width, border_radius=6)

        # Glow effect: semi-transparent multi-offset blits when flashing at 3+
        if combo_flash > 0 and use_accent and self._large_font is not None:
            glow_surf = self._large_font.render(f"\xd7{combo}", True, colors.accent)
            glow_surf.set_alpha(60)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                screen.blit(glow_surf, (num_x + dx, panel_y + PAD // 2 + dy))

        # Number (×N) and label
        screen.blit(number_surf, (num_x, panel_y + PAD // 2))
        screen.blit(label_surf,  (lbl_x, panel_y + PAD // 2 + number_surf.get_height() + 3))

    def _draw_big_combo(
        self,
        screen: pygame.Surface,
        player_combo: int,
        ai_combo: int,
        player_combo_flash: int,
        ai_combo_flash: int,
    ) -> None:
        """Render the primary large combo display centered near the top of the screen.

        Phase 20: shown for the active combo >= 3.  Includes:
          - Scale-up animation during the first 10 display frames after increment.
          - Alpha fade-out during the final 15 display frames of persistence.
          - Subtle background panel for readability.
        """
        if self._large_font is None:
            return

        colors = self._dcfg.colors
        w_win  = self._dcfg.window.width

        # Choose which combo to feature (higher flash = more recently incremented)
        if player_combo >= 3 and ai_combo >= 3:
            if player_combo_flash >= ai_combo_flash:
                combo, flash = player_combo, player_combo_flash
            else:
                combo, flash = ai_combo, ai_combo_flash
        elif player_combo >= 3:
            combo, flash = player_combo, player_combo_flash
        elif ai_combo >= 3:
            combo, flash = ai_combo, ai_combo_flash
        else:
            return

        if flash <= 0:
            return

        _PERSIST   = 60   # total frames (== _COMBO_FLASH_FRAMES)
        _FADE_IN   = 10   # scale-pop frames at the start
        _FADE_OUT  = 15   # alpha-fade frames at the end

        # Alpha: full except during fade-out window
        alpha = 255 if flash >= _FADE_OUT else int(255 * flash / _FADE_OUT)

        # Scale animation: peak during the first _FADE_IN frames (flash > 50)
        thresh = _PERSIST - _FADE_IN   # 50
        if flash > thresh:
            t = (flash - thresh) / _FADE_IN   # 0..1
            scale_f = 1.0 + 0.22 * math.sin(t * math.pi)
        else:
            scale_f = 1.0

        # Render text surface
        text = f"\xd7{combo} COMBO"
        base_surf = self._large_font.render(text, True, colors.accent)

        if abs(scale_f - 1.0) > 0.005:
            new_w = max(1, int(base_surf.get_width()  * scale_f))
            new_h = max(1, int(base_surf.get_height() * scale_f))
            surf  = pygame.transform.scale(base_surf, (new_w, new_h))
        else:
            surf = base_surf

        surf = surf.copy()
        surf.set_alpha(alpha)

        # Background panel (semi-transparent, anchored to text bounds)
        PAD     = 12
        panel_w = surf.get_width()  + PAD * 2
        panel_h = surf.get_height() + PAD
        y_pos   = 68   # below tier badge / tick counter
        panel_x = w_win // 2 - panel_w // 2

        panel_alpha = max(0, alpha - 80)
        p_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        p_surf.fill((10, 8, 22, panel_alpha))
        screen.blit(p_surf, (panel_x, y_pos - PAD // 2))

        # Text centered
        screen.blit(surf, (w_win // 2 - surf.get_width() // 2, y_pos))

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
        block_flash: int = 0,
        guard_break_flash: int = 0,
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
        if guard_break_flash > 0:
            # Guard break: bright red-orange flash
            draw_color = (255, 80, 40)
        elif block_flash > 0:
            # Block absorbed a hit: cyan/teal flash
            draw_color = (40, 220, 230)
        elif flash_ticks > 0:
            draw_color = colors.hit_flash
        elif whiff_ticks > 0:
            # Attacker whiffed: dim desaturated yellow-grey
            draw_color = (160, 150, 80)
        elif state == FSMState.BLOCKING:
            # Actively blocking: desaturated blue-grey shield tint
            draw_color = tuple(min(255, int(c * 0.5 + 80)) for c in base_color)
        elif state == FSMState.BLOCKSTUN:
            draw_color = (80, 140, 200)
        elif state == FSMState.PARRY_STUNNED:
            draw_color = (200, 60, 200)   # magenta: guard-break stunned
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

        # Phase 18: shield bubble when blocking
        if state == FSMState.BLOCKING:
            shield_surf = pygame.Surface((fw + 16, fh + 16), pygame.SRCALPHA)
            pygame.draw.rect(shield_surf, (80, 200, 255, 70),
                             (0, 0, fw + 16, fh + 16), border_radius=8)
            pygame.draw.rect(shield_surf, (120, 220, 255, 160),
                             (0, 0, fw + 16, fh + 16), 2, border_radius=8)
            screen.blit(shield_surf, (render_x - 8, render_y - 8))

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
