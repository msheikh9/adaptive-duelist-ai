"""Help / controls overlay rendering.

Provides draw_help_overlay() — draws a semi-transparent controls panel
onto an existing pygame surface. Used during active matches and on the
title screen.
"""

from __future__ import annotations

import pygame

from game.input.keybind_registry import build_keybind_entries_by_category


def draw_help_overlay(
    surface: pygame.Surface,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
    colors: object,
) -> None:
    """Render the controls/help overlay onto *surface*.

    The overlay is drawn on top of whatever is already on the surface,
    with a semi-transparent background so match state remains visible.

    Args:
        surface:    Target pygame surface (the game window).
        font_title: Font used for the "CONTROLS" header.
        font_body:  Font used for category headers and keybind lines.
        colors:     DisplayConfig.colors (duck-typed; uses getattr with defaults).
    """
    sw, sh = surface.get_size()

    # ---------------------------------------------------------------------------
    # Colours (with safe fallbacks so this works without a polished display cfg)
    # ---------------------------------------------------------------------------
    accent        = getattr(colors, "accent",        (160, 120, 240))
    panel_bg      = getattr(colors, "panel_bg",      (18, 18, 28))
    border_col    = getattr(colors, "border",        (60, 65, 90))
    text_primary  = getattr(colors, "text_primary",  (240, 240, 240))
    text_secondary = getattr(colors, "text_secondary", (160, 160, 170))
    player_col    = getattr(colors, "player",        (80, 160, 255))

    # ---------------------------------------------------------------------------
    # Dim the whole screen first
    # ---------------------------------------------------------------------------
    dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 185))
    surface.blit(dim, (0, 0))

    # ---------------------------------------------------------------------------
    # Panel geometry
    # ---------------------------------------------------------------------------
    entries_by_cat = build_keybind_entries_by_category()
    n_entries = sum(len(v) for v in entries_by_cat.values())
    n_cats = len(entries_by_cat)

    LINE_H   = 26
    HDR_H    = 34
    CAT_GAP  = 10
    PADDING  = 28

    content_h = (n_cats * HDR_H
                 + n_entries * LINE_H
                 + max(0, n_cats - 1) * CAT_GAP)
    panel_h = min(content_h + PADDING * 2 + 54 + 24, sh - 80)
    panel_w = min(580, sw - 80)

    panel_x = (sw - panel_w) // 2
    panel_y = (sh - panel_h) // 2

    # ---------------------------------------------------------------------------
    # Panel background + border
    # ---------------------------------------------------------------------------
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((*panel_bg, 245))
    surface.blit(panel_surf, (panel_x, panel_y))
    pygame.draw.rect(
        surface, border_col,
        (panel_x, panel_y, panel_w, panel_h), 2, border_radius=8,
    )

    # ---------------------------------------------------------------------------
    # Title row
    # ---------------------------------------------------------------------------
    title_surf = font_title.render("CONTROLS", True, accent)
    surface.blit(
        title_surf,
        (panel_x + (panel_w - title_surf.get_width()) // 2, panel_y + 16),
    )
    # Divider under title
    div_y = panel_y + 16 + title_surf.get_height() + 6
    pygame.draw.line(
        surface, border_col,
        (panel_x + 16, div_y), (panel_x + panel_w - 16, div_y), 1,
    )

    # ---------------------------------------------------------------------------
    # Keybind rows
    # ---------------------------------------------------------------------------
    KEY_COL_X   = panel_x + PADDING
    LABEL_COL_X = panel_x + PADDING + 160
    cy = div_y + 10

    for cat, entries in entries_by_cat.items():
        # Category header
        cat_surf = font_body.render(cat.upper(), True, accent)
        surface.blit(cat_surf, (KEY_COL_X, cy))
        cy += HDR_H

        for entry in entries:
            key_str = " / ".join(entry.keys)
            key_surf  = font_body.render(key_str,    True, player_col)
            lbl_surf  = font_body.render(entry.label, True, text_primary)
            surface.blit(key_surf,  (KEY_COL_X,   cy))
            surface.blit(lbl_surf,  (LABEL_COL_X, cy))
            cy += LINE_H

        cy += CAT_GAP

    # ---------------------------------------------------------------------------
    # Dismiss hint at bottom of panel
    # ---------------------------------------------------------------------------
    hint_surf = font_body.render("H  ·  close", True, text_secondary)
    surface.blit(
        hint_surf,
        (panel_x + (panel_w - hint_surf.get_width()) // 2,
         panel_y + panel_h - 22),
    )
