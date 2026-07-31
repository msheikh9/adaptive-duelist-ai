"""The single authoritative simulation tick.

Every caller that advances a match runs through `step_simulation`: the live
engine, the replay verifier, the batch evaluator, and the headless test
fixture. Before this module existed each of those had its own hand-written copy
of the tick loop and only the engine's was complete — the other three silently
omitted gravity, guard regen, all three cooldown timers and the whole projectile
subsystem, which is why replay verification could never reproduce a match and
why evaluated AI tiers could use heavy attack and dodge at most once per match.

The step is presentation-free. It mutates `SimulationState` and *describes* what
happened in a `TickEffects` record; sound, VFX queues, combo counters, hit
flashes, screen shake and hitstop are the caller's business. Semantic events are
likewise described rather than emitted, so a caller that wants them (the engine,
the evaluator) can build and route them, and a caller that does not (the replay
verifier) can ignore them at no cost.

Policy is *not* part of the step. Who acts, and what they choose, is supplied by
the caller as two deciders — which is what lets the replay verifier substitute
"replay the recorded commitment stream" for "ask the planner" and get an
identical simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from game.combat.actions import (
    Actor,
    CombatCommitment,
    FREE_STATES,
    FSMState,
)
from game.combat.collision import HitTracker, check_hit, was_dodge_avoided
from game.combat.damage import apply_hit
from game.combat.guard import apply_block_response, tick_guard
from game.combat.physics import (
    apply_dodge_velocity,
    apply_gravity,
    apply_velocity,
    clamp_to_arena,
    handle_landing,
    update_facing,
)
from game.combat.projectile import (
    Projectile,
    ProjectileHitEffect,
    fire_projectile,
    update_projectiles,
)
from game.combat.stamina import tick_stamina
from game.combat.state_machine import (
    enter_landing,
    tick_dodge_cooldown,
    tick_fsm,
    tick_heavy_cooldown,
    tick_shoot_cooldown,
)
from game.state import MatchStatus

if TYPE_CHECKING:
    from config.config_loader import GameConfig
    from game.state import SimulationState

# A decider enters commitments on its fighter and returns the ones it entered,
# in the order entered. Usually empty or one element, but more than one is legal
# and does happen: MOVING is a free state, so the engine's periodic AI shoot
# policy can override a movement commitment entered earlier in the same tick.
# The replay verifier relies on this to replay a tick's full commitment record.
Decider = Callable[[], "Sequence[CombatCommitment]"]


@dataclass
class SimulationContext:
    """Per-match mutable state that outlives a single tick.

    Create one per match and pass it to every `step_simulation` call for that
    match. Holding this outside the step is what keeps the step itself a pure
    function of (state, config, context, deciders).
    """

    hit_tracker: HitTracker = field(default_factory=HitTracker)
    projectiles: list[Projectile] = field(default_factory=list)
    # FSM states as of the end of the previous tick, for edge detection.
    prev_player_fsm: FSMState = FSMState.IDLE
    prev_ai_fsm: FSMState = FSMState.IDLE
    # Tick of each side's last commitment, for reaction-time measurement.
    # Starts at 0, not -1, so a first-tick commitment reports 0 reaction rather
    # than 1 — matching the original engine bookkeeping.
    last_player_commit_tick: int = 0
    last_ai_commit_tick: int = 0

    def reaction_ticks(self, actor: Actor, tick_id: int) -> int:
        """Ticks since the *opponent* last committed — i.e. reaction latency."""
        if actor == Actor.PLAYER:
            return tick_id - self.last_ai_commit_tick
        return tick_id - self.last_player_commit_tick


@dataclass
class TickEffects:
    """Everything a caller might want to react to, for one tick.

    Fields describing an attack (`player_hit`, `player_whiffed`) are from the
    *attacker's* perspective. Fields describing defence (`ai_blocked`,
    `ai_guard_broken`) are from the *defender's* perspective.
    """

    # Commitments entered this tick, in order (already applied to state).
    player_commitments: list[CombatCommitment] = field(default_factory=list)
    ai_commitments: list[CombatCommitment] = field(default_factory=list)

    # Melee hits that landed and were applied. None if nothing or if blocked.
    player_hit: object | None = None   # HitEvent from the player's attack
    ai_hit: object | None = None       # HitEvent from the AI's attack

    # Defence. `ai_blocked` means the AI blocked the player's incoming attack.
    player_blocked: bool = False
    ai_blocked: bool = False
    player_guard_broken: bool = False
    ai_guard_broken: bool = False

    # An attack passed through a dodge without connecting.
    player_dodge_avoided: bool = False
    ai_dodge_avoided: bool = False

    # Touched the ground from AIRBORNE this tick.
    player_landed: bool = False
    ai_landed: bool = False

    # ATTACK_ACTIVE -> ATTACK_RECOVERY with nothing connected.
    player_whiffed: bool = False
    ai_whiffed: bool = False

    player_exhausted: bool = False
    ai_exhausted: bool = False

    # Opponent recovered from hitstun without being re-hit, so a streak ended.
    player_combo_reset: bool = False
    ai_combo_reset: bool = False

    # Left a locked state and became actionable again.
    player_commitment_end: bool = False
    ai_commitment_end: bool = False

    projectiles_fired: list[Projectile] = field(default_factory=list)
    projectile_hits: list[ProjectileHitEffect] = field(default_factory=list)

    # "AI" or "PLAYER" — the winner — set on the tick a KO occurs.
    ko_winner: str | None = None


def step_simulation(
    state: SimulationState,
    gcfg: GameConfig,
    ctx: SimulationContext,
    decide_player: Decider,
    decide_ai: Decider,
) -> TickEffects:
    """Advance the match by exactly one tick.

    `decide_player` runs before `decide_ai`, so an AI policy may legitimately
    react to a commitment the player entered on this same tick. That ordering is
    load-bearing for the T2 planner's frame-perfect reads — do not swap it.

    Returns a `TickEffects` describing the tick. The caller is responsible for
    presentation, replay recording, and semantic event emission.
    """
    fx = TickEffects()
    scale = gcfg.simulation.sub_pixel_scale
    fighter_w_sub = gcfg.fighter.width * scale

    # --- Decisions (policy lives in the caller) ---
    # Note: `last_*_commit_tick` is deliberately NOT updated here. Reaction time
    # is measured against the opponent's *previous* commitment, and the caller
    # updates those fields as it emits each commitment event — preserving the
    # original ordering where a same-tick reply reads as reaction_ticks == 0.
    fx.player_commitments = list(decide_player())
    fx.ai_commitments = list(decide_ai())

    # --- Accumulate charge while holding a shot ---
    max_charge = gcfg.actions.shoot.max_charge_frames
    if state.player.fsm_state == FSMState.CHARGING:
        state.player.charge_ticks = min(state.player.charge_ticks + 1, max_charge)
    if state.ai.fsm_state == FSMState.CHARGING:
        state.ai.charge_ticks = min(state.ai.charge_ticks + 1, max_charge)

    # --- Release pending shots ---
    if state.player.pending_shot:
        state.player.pending_shot = False
        fx.projectiles_fired.append(
            fire_projectile(state.player, "PLAYER", gcfg, ctx.projectiles))
    if state.ai.pending_shot:
        state.ai.pending_shot = False
        fx.projectiles_fired.append(
            fire_projectile(state.ai, "AI", gcfg, ctx.projectiles))

    # --- Cooldowns tick every tick, regardless of FSM state ---
    tick_dodge_cooldown(state.player)
    tick_dodge_cooldown(state.ai)
    tick_heavy_cooldown(state.player)
    tick_heavy_cooldown(state.ai)
    tick_shoot_cooldown(state.player)
    tick_shoot_cooldown(state.ai)

    # --- Guard meter regen ---
    tick_guard(state.player, gcfg)
    tick_guard(state.ai, gcfg)

    # --- Movement: gravity, then dodge impulse, then integrate ---
    apply_gravity(state.player, state.arena, gcfg)
    apply_gravity(state.ai, state.arena, gcfg)
    apply_dodge_velocity(state.player, gcfg)
    apply_dodge_velocity(state.ai, gcfg)
    apply_velocity(state.player)
    apply_velocity(state.ai)
    clamp_to_arena(state.player, state.arena, fighter_w_sub)
    clamp_to_arena(state.ai, state.arena, fighter_w_sub)

    # --- Landing ---
    if handle_landing(state.player, state.arena):
        if state.player.fsm_state == FSMState.AIRBORNE:
            enter_landing(state.player, gcfg.fighter.landing_recovery_frames)
            fx.player_landed = True
    if handle_landing(state.ai, state.arena):
        if state.ai.fsm_state == FSMState.AIRBORNE:
            enter_landing(state.ai, gcfg.fighter.landing_recovery_frames)
            fx.ai_landed = True

    update_facing(state.player, state.ai)

    # --- Melee collision ---
    player_hit = check_hit(
        state.player, state.ai, "player", ctx.hit_tracker, gcfg)
    ai_hit = check_hit(
        state.ai, state.player, "ai", ctx.hit_tracker, gcfg)

    # Dodge avoidance must be sampled before damage, while the attacker is
    # still in its active window.
    fx.player_dodge_avoided = was_dodge_avoided(
        state.player, state.ai, "player", ctx.hit_tracker, gcfg)
    fx.ai_dodge_avoided = was_dodge_avoided(
        state.ai, state.player, "ai", ctx.hit_tracker, gcfg)

    # --- Guard intercepts damage before it lands ---
    if player_hit and state.ai.fsm_state == FSMState.BLOCKING:
        fx.ai_guard_broken = apply_block_response(state.ai, player_hit, gcfg)
        fx.ai_blocked = True
        player_hit = None
    if ai_hit and state.player.fsm_state == FSMState.BLOCKING:
        fx.player_guard_broken = apply_block_response(
            state.player, ai_hit, gcfg)
        fx.player_blocked = True
        ai_hit = None

    if player_hit:
        apply_hit(state.ai, player_hit)
        fx.player_hit = player_hit
    if ai_hit:
        apply_hit(state.player, ai_hit)
        fx.ai_hit = ai_hit

    # --- Stamina ---
    fx.player_exhausted = tick_stamina(state.player, gcfg)
    fx.ai_exhausted = tick_stamina(state.ai, gcfg)

    # --- Advance FSMs ---
    if state.player.is_free:
        ctx.hit_tracker.reset("player")
    if state.ai.is_free:
        ctx.hit_tracker.reset("ai")

    tick_fsm(state.player, gcfg)
    tick_fsm(state.ai, gcfg)

    # --- Projectiles move and resolve after the FSMs advance ---
    fx.projectile_hits = update_projectiles(state, gcfg, ctx.projectiles)

    # --- Whiff detection on the ACTIVE -> RECOVERY edge ---
    if (ctx.prev_player_fsm == FSMState.ATTACK_ACTIVE
            and state.player.fsm_state == FSMState.ATTACK_RECOVERY
            and not ctx.hit_tracker.has_connected("player")):
        fx.player_whiffed = True
    if (ctx.prev_ai_fsm == FSMState.ATTACK_ACTIVE
            and state.ai.fsm_state == FSMState.ATTACK_RECOVERY
            and not ctx.hit_tracker.has_connected("ai")):
        fx.ai_whiffed = True

    # --- A streak ends when the opponent escapes hitstun un-re-hit ---
    if (ctx.prev_ai_fsm == FSMState.HITSTUN
            and state.ai.fsm_state in FREE_STATES
            and not player_hit):
        fx.player_combo_reset = True
    if (ctx.prev_player_fsm == FSMState.HITSTUN
            and state.player.fsm_state in FREE_STATES
            and not ai_hit):
        fx.ai_combo_reset = True

    # --- Commitment end edges ---
    if (ctx.prev_player_fsm not in FREE_STATES
            and state.player.fsm_state in FREE_STATES):
        fx.player_commitment_end = True
    if (ctx.prev_ai_fsm not in FREE_STATES
            and state.ai.fsm_state in FREE_STATES):
        fx.ai_commitment_end = True

    ctx.prev_player_fsm = state.player.fsm_state
    ctx.prev_ai_fsm = state.ai.fsm_state

    # --- KO ---
    if state.player.fsm_state == FSMState.KO:
        state.match_status = MatchStatus.ENDED
        state.winner = "AI"
        fx.ko_winner = "AI"
    elif state.ai.fsm_state == FSMState.KO:
        state.match_status = MatchStatus.ENDED
        state.winner = "PLAYER"
        fx.ko_winner = "PLAYER"

    return fx
