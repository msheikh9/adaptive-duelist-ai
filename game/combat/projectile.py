"""Projectile data model for Phase 20: Charged Ranged Weapon System.

A Projectile is a fire-and-forget entity created when a fighter releases
a charged shot. The engine owns the list; the renderer draws it.

Coordinates are in sub-pixel units (same as FighterState.x / .y).
velocity_x is in sub-pixels per tick and carries sign (positive = right).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Projectile:
    """A single in-flight projectile."""

    x: int              # sub-pixel x position (centre of projectile)
    y: int              # sub-pixel y position (centre of projectile)
    velocity_x: int     # sub-pixels / tick; sign encodes direction
    damage: int         # hit-point damage on contact
    owner: str          # "PLAYER" or "AI"
    charge_frac: float  # 0.0 = uncharged, 1.0 = full charge (visual scale)
    active: bool = True # False once it hits or leaves the arena
