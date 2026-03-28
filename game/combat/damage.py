"""Damage application system.

Applies damage from HitEvents to defenders. Handles HP reduction,
hitstun entry, knockback, and KO detection.
"""

from __future__ import annotations

from game.combat.collision import HitEvent
from game.combat.physics import apply_knockback
from game.combat.state_machine import enter_hitstun, enter_ko
from game.state import FighterState


def apply_hit(defender: FighterState, hit: HitEvent) -> None:
    """Apply a confirmed hit to the defender.

    Reduces HP, applies knockback, and enters hitstun.
    If HP reaches 0, enters KO instead.
    """
    defender.hp = max(0, defender.hp - hit.damage)

    if defender.hp <= 0:
        enter_ko(defender)
        apply_knockback(defender, hit.knockback_sub, hit.knockback_direction)
        return

    # Interrupt whatever the defender was doing
    enter_hitstun(defender, hit.hitstun_frames)
    apply_knockback(defender, hit.knockback_sub, hit.knockback_direction)
