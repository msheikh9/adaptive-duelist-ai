"""Guard (block durability) system.

Manages the guard meter: reduction on blocked hits, regen delay, and guard break.

Phase 18:
  - tick_guard()          — per-tick regen countdown and guard recovery
  - apply_block_response() — chip damage, guard cost, blockstun or guard break
"""

from __future__ import annotations

from game.combat.actions import CombatCommitment, FSMState
from game.combat.state_machine import enter_blockstun, enter_parry_stunned
from game.state import FighterState
from config.config_loader import GameConfig


def tick_guard(fighter: FighterState, config: GameConfig) -> None:
    """Advance the guard regen countdown and restore guard when eligible.

    Guard does NOT regenerate while the fighter is blocking, in blockstun,
    parry-stunned, or KO'd. The regen delay starts when a blocked hit lands
    and counts down to zero before regen resumes.
    """
    block_cfg = config.actions.block

    if fighter.fsm_state in (
        FSMState.BLOCKING, FSMState.BLOCKSTUN, FSMState.PARRY_STUNNED, FSMState.KO
    ):
        return

    if fighter.guard_regen_delay > 0:
        fighter.guard_regen_delay -= 1
        return

    if fighter.guard < block_cfg.guard_max:
        fighter.guard = min(block_cfg.guard_max,
                            fighter.guard + block_cfg.guard_regen_per_tick)


def apply_block_response(defender: FighterState, hit, config: GameConfig) -> bool:
    """Apply a hit that was absorbed by the defender's block.

    Deals chip damage, reduces guard by the hit's guard cost, resets regen
    delay, and either enters blockstun (guard held) or parry-stunned (guard
    broken).

    Returns:
        True  — guard was broken (defender enters PARRY_STUNNED).
        False — guard held (defender enters BLOCKSTUN).
    """
    block_cfg = config.actions.block
    is_heavy = (hit.attacker_commitment == CombatCommitment.HEAVY_ATTACK)

    # Chip damage: always at least 1
    chip = max(1, round(hit.damage * block_cfg.chip_damage_pct))
    defender.hp = max(0, defender.hp - chip)

    # Guard cost
    cost = block_cfg.guard_cost_heavy if is_heavy else block_cfg.guard_cost_light
    defender.guard = max(0, defender.guard - cost)

    # Start regen delay
    defender.guard_regen_delay = block_cfg.guard_regen_delay_ticks

    if defender.guard <= 0:
        enter_parry_stunned(defender, block_cfg.guard_break_stun_frames)
        return True
    else:
        enter_blockstun(defender, block_cfg.blockstun_frames)
        return False
