"""Projectile data model and simulation for Phase 20: Charged Ranged Weapon.

A Projectile is a fire-and-forget entity created when a fighter releases
a charged shot. The caller owns the list; the renderer draws it.

Coordinates are in sub-pixel units (same as FighterState.x / .y).
velocity_x is in sub-pixels per tick and carries sign (positive = right).

The simulation functions here are presentation-free: they mutate fighter and
projectile state and *describe* what happened via ProjectileHitEffect. Sound,
VFX, flashes and popups are the caller's job. This is what lets the engine, the
replay verifier, the batch evaluator and the test fixture share one
implementation instead of four divergent ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from game.combat.actions import Actor, CombatCommitment, FSMState
from game.combat.guard import apply_block_response
from game.combat.state_machine import enter_hitstun, enter_ko

if TYPE_CHECKING:
    from config.config_loader import GameConfig
    from game.state import FighterState, SimulationState

# Hitstun applied by a projectile, in ticks. Shorter than melee.
PROJECTILE_HITSTUN_HEAVY = 8
PROJECTILE_HITSTUN_LIGHT = 5

# A charge fraction at or above this counts as a "heavy" shot.
HEAVY_CHARGE_THRESHOLD = 0.5


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


@dataclass
class ProjectileHitEffect:
    """Describes a projectile connecting, for the caller's presentation layer."""

    x: int                     # impact position (sub-pixel)
    y: int
    target_actor: Actor        # who got hit
    is_heavy: bool
    damage: int
    blocked: bool = False      # target was BLOCKING; damage went to guard
    guard_broken: bool = False # only meaningful when blocked
    ko: bool = False           # this hit reduced the target to 0 HP


def fire_projectile(
    shooter: FighterState,
    owner: str,
    gcfg: GameConfig,
    projectiles: list[Projectile],
) -> Projectile:
    """Spawn a projectile from the shooter, start its cooldown, clear charge.

    Appends to `projectiles` and returns the new projectile.
    """
    shoot_cfg = gcfg.actions.shoot
    charge_frac = min(
        1.0, shooter.charge_ticks / max(1, shoot_cfg.max_charge_frames)
    )
    damage = round(
        shoot_cfg.min_damage
        + charge_frac * (shoot_cfg.max_damage - shoot_cfg.min_damage)
    )
    speed_sub = shoot_cfg.projectile_speed * gcfg.simulation.sub_pixel_scale

    proj = Projectile(
        x=shooter.x,
        y=shooter.y,
        velocity_x=speed_sub * shooter.facing,
        damage=damage,
        owner=owner,
        charge_frac=charge_frac,
    )
    projectiles.append(proj)

    shooter.shoot_cooldown = shoot_cfg.cooldown_frames
    shooter.charge_ticks = 0
    return proj


def resolve_projectile_hit(
    proj: Projectile,
    target: FighterState,
    target_actor: Actor,
    gcfg: GameConfig,
) -> ProjectileHitEffect | None:
    """Apply a projectile hit to `target`. Returns None if the hit is ignored.

    Mirrors melee resolution order: a blocking target routes the damage through
    the guard system, a KO'd target absorbs nothing.
    """
    is_heavy = proj.charge_frac >= HEAVY_CHARGE_THRESHOLD

    if target.fsm_state == FSMState.BLOCKING:
        # Chip damage through guard. Costed as a light hit regardless of charge,
        # matching the original engine behaviour.
        guard_broken = apply_block_response(
            target, _GuardCost(proj.damage), gcfg
        )
        return ProjectileHitEffect(
            x=target.x, y=target.y, target_actor=target_actor,
            is_heavy=is_heavy, damage=proj.damage,
            blocked=True, guard_broken=guard_broken,
        )

    if target.fsm_state == FSMState.KO:
        return None

    target.hp = max(0, target.hp - proj.damage)
    ko = target.hp == 0
    if ko:
        enter_ko(target)
    else:
        enter_hitstun(
            target,
            PROJECTILE_HITSTUN_HEAVY if is_heavy else PROJECTILE_HITSTUN_LIGHT,
        )

    return ProjectileHitEffect(
        x=proj.x, y=proj.y, target_actor=target_actor,
        is_heavy=is_heavy, damage=proj.damage, ko=ko,
    )


class _GuardCost:
    """Minimal hit-shaped object for `apply_block_response`.

    Projectiles are costed as light attacks against the guard meter.
    """

    attacker_commitment = CombatCommitment.LIGHT_ATTACK

    def __init__(self, damage: int) -> None:
        self.damage = damage


def update_projectiles(
    state: SimulationState,
    gcfg: GameConfig,
    projectiles: list[Projectile],
) -> list[ProjectileHitEffect]:
    """Advance every active projectile, resolve collisions, drop spent ones.

    Mutates `projectiles` in place (spent entries are removed) and returns the
    hits that landed this tick.
    """
    effects: list[ProjectileHitEffect] = []
    if not projectiles:
        return effects

    arena_w = state.arena.width_sub
    scale = gcfg.simulation.sub_pixel_scale
    fighter_w = gcfg.fighter.width * scale
    fighter_h = gcfg.fighter.height * scale
    # Projectile is treated as a square of 1/6 the fighter's width.
    proj_r = fighter_w // 6

    for proj in projectiles:
        if not proj.active:
            continue

        proj.x += proj.velocity_x

        if proj.x < 0 or proj.x > arena_w:
            proj.active = False
            continue

        if proj.owner == "PLAYER":
            target, target_actor = state.ai, Actor.AI
        else:
            target, target_actor = state.player, Actor.PLAYER

        t_left = target.x - fighter_w // 2
        t_right = target.x + fighter_w // 2
        t_top = target.y - fighter_h
        t_bot = target.y

        if (t_left - proj_r <= proj.x <= t_right + proj_r
                and t_top - proj_r <= proj.y <= t_bot + proj_r):
            effect = resolve_projectile_hit(proj, target, target_actor, gcfg)
            if effect is not None:
                effects.append(effect)
            proj.active = False

    projectiles[:] = [p for p in projectiles if p.active]
    return effects
