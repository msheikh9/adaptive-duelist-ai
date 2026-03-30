"""Collision detection between hitboxes and hurtboxes.

Generates hit events when an active attack hitbox overlaps a defender's
hurtbox. Enforces one-hit-per-swing: once a hit connects during an
attack's active frames, it cannot hit again until a new attack starts.

Dodge invincibility is checked here: fighters in DODGING state are
immune to hits.
"""

from __future__ import annotations

from dataclasses import dataclass

from game.combat.actions import CombatCommitment, FSMState
from game.combat.hitbox import get_attack_hitbox, get_fighter_hurtbox
from game.state import FighterState
from config.config_loader import GameConfig


@dataclass(frozen=True, slots=True)
class HitEvent:
    """Records a single hit between attacker and defender."""
    attacker_commitment: CombatCommitment
    damage: int
    hitstun_frames: int
    knockback_sub: int
    knockback_direction: int  # +1 or -1


class HitTracker:
    """Tracks which attacks have already connected to enforce one-hit-per-swing.

    Keyed by (attacker_id, commitment_start_tick). Reset when the attacker
    re-enters a free state.
    """

    def __init__(self) -> None:
        self._connected: set[str] = set()

    def has_connected(self, attacker_id: str) -> bool:
        return attacker_id in self._connected

    def mark_connected(self, attacker_id: str) -> None:
        self._connected.add(attacker_id)

    def reset(self, attacker_id: str) -> None:
        self._connected.discard(attacker_id)

    def clear(self) -> None:
        self._connected.clear()


def check_hit(attacker: FighterState, defender: FighterState,
              attacker_id: str, hit_tracker: HitTracker,
              config: GameConfig) -> HitEvent | None:
    """Check if attacker's hitbox overlaps defender's hurtbox.

    Returns a HitEvent if a new hit occurs, None otherwise.
    """
    # Already connected this swing
    if hit_tracker.has_connected(attacker_id):
        return None

    # Get active hitbox (only during ATTACK_ACTIVE)
    hitbox = get_attack_hitbox(attacker, config)
    if hitbox is None:
        return None

    # Defender is invincible during dodge active frames
    if defender.fsm_state == FSMState.DODGING:
        return None

    hurtbox = get_fighter_hurtbox(defender, config)
    if not hitbox.overlaps(hurtbox):
        return None

    # Hit confirmed
    hit_tracker.mark_connected(attacker_id)

    commitment = attacker.active_commitment
    if commitment == CombatCommitment.LIGHT_ATTACK:
        atk_cfg = config.actions.light_attack
    elif commitment == CombatCommitment.HEAVY_ATTACK:
        atk_cfg = config.actions.heavy_attack
    else:
        return None

    # Knockback direction: push defender away from attacker
    knockback_dir = 1 if attacker.x < defender.x else -1

    return HitEvent(
        attacker_commitment=commitment,
        damage=atk_cfg.damage,
        hitstun_frames=atk_cfg.hitstun_frames,
        knockback_sub=atk_cfg.knockback * config.simulation.sub_pixel_scale,
        knockback_direction=knockback_dir,
    )


def was_dodge_avoided(attacker: FighterState, defender: FighterState,
                      attacker_id: str, hit_tracker: HitTracker,
                      config: GameConfig) -> bool:
    """Return True if the attacker's hitbox would have connected but
    the defender's dodge invulnerability absorbed it.

    Conditions:
    - Attacker is in ATTACK_ACTIVE and hasn't already connected this swing.
    - Defender is in DODGING state (invulnerable).
    - The hitbox and hurtbox actually overlap (real near-miss, not a whiff).
    """
    if hit_tracker.has_connected(attacker_id):
        return False
    hitbox = get_attack_hitbox(attacker, config)
    if hitbox is None:
        return False
    if defender.fsm_state != FSMState.DODGING:
        return False
    hurtbox = get_fighter_hurtbox(defender, config)
    return hitbox.overlaps(hurtbox)
