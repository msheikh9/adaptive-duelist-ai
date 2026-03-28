"""Action resolver: translates a TacticalIntent into a single CombatCommitment.

Returns exactly one commitment + positioning_bias + commit_delay.
No plan queues — multi-step tactics emerge through re-evaluation.

When HOLD is top_label, the resolver still picks a non-HOLD action using
the prediction's commitment fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai.strategy.tactics import TacticalIntent
from game.combat.actions import CombatCommitment, SpacingZone

if TYPE_CHECKING:
    from ai.strategy.ai_context import AIContext
    from ai.strategy.planner_memory import PlannerMemory
    from config.config_loader import ActionResolverConfig

# Counter-action table: given predicted player commitment, what beats it?
_COUNTER_MAP: dict[CombatCommitment, CombatCommitment] = {
    CombatCommitment.LIGHT_ATTACK: CombatCommitment.DODGE_BACKWARD,
    CombatCommitment.HEAVY_ATTACK: CombatCommitment.LIGHT_ATTACK,  # punish startup
    CombatCommitment.DODGE_BACKWARD: CombatCommitment.MOVE_RIGHT,  # advance on retreat
    CombatCommitment.MOVE_LEFT: CombatCommitment.MOVE_RIGHT,       # pursue
    CombatCommitment.MOVE_RIGHT: CombatCommitment.MOVE_LEFT,       # cut off
}


@dataclass(frozen=True)
class ResolverOutput:
    """Result of action resolution."""
    commitment: CombatCommitment
    positioning_bias: float   # -1.0 (retreat) to +1.0 (advance)
    commit_delay: int         # ticks to wait before committing (0 = immediate)
    reason_tags: tuple[str, ...]


def resolve(
    mode: TacticalIntent,
    ctx: AIContext,
    memory: PlannerMemory,
    resolver_cfg: ActionResolverConfig,
) -> ResolverOutput:
    """Resolve a TacticalIntent into a concrete action.

    Always returns an actionable commitment. HOLD predictions are
    handled via non-HOLD fallback from the PredictionResult.
    """
    pred = ctx.prediction
    predicted_commit = pred.top_commitment  # non-HOLD fallback

    match mode:
        case TacticalIntent.EXPLOIT_PATTERN:
            return _resolve_exploit(ctx, predicted_commit, resolver_cfg)

        case TacticalIntent.BAIT_AND_PUNISH:
            return _resolve_bait(ctx, resolver_cfg)

        case TacticalIntent.PUNISH_RECOVERY:
            return _resolve_punish_recovery(ctx, resolver_cfg)

        case TacticalIntent.PRESSURE_STAMINA:
            return _resolve_pressure(ctx, resolver_cfg)

        case TacticalIntent.DEFENSIVE_RESET:
            return _resolve_defensive(ctx, resolver_cfg)

        case TacticalIntent.PROBE_BEHAVIOR:
            return _resolve_probe(ctx, resolver_cfg)

        case TacticalIntent.NEUTRAL_SPACING:
            return _resolve_neutral(ctx, resolver_cfg)

        case _:
            return ResolverOutput(
                commitment=CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT,
                positioning_bias=0.0,
                commit_delay=0,
                reason_tags=("fallback",),
            )


# ------------------------------------------------------------------ #
# Per-mode resolvers                                                   #
# ------------------------------------------------------------------ #

def _resolve_exploit(
    ctx: AIContext,
    predicted_commit: CombatCommitment | None,
    cfg: ActionResolverConfig,
) -> ResolverOutput:
    """Counter the predicted player commitment."""
    if predicted_commit is not None:
        action = _COUNTER_MAP.get(predicted_commit, CombatCommitment.LIGHT_ATTACK)
        # If we predict attack, dodge; if dodge, advance; etc.
        delay = min(4, cfg.max_commit_delay_ticks)  # slight delay for timing
        return ResolverOutput(
            commitment=action,
            positioning_bias=0.2 if action in _ADVANCE_ACTIONS else -0.2,
            commit_delay=delay,
            reason_tags=("exploit", f"counter_{predicted_commit.name}"),
        )

    # No prediction available — default to light attack at close range
    return ResolverOutput(
        commitment=CombatCommitment.LIGHT_ATTACK,
        positioning_bias=0.3,
        commit_delay=0,
        reason_tags=("exploit", "no_prediction"),
    )


def _resolve_bait(ctx: AIContext, cfg: ActionResolverConfig) -> ResolverOutput:
    """Move into range to bait an attack, then prepare to dodge/punish."""
    if ctx.spacing == SpacingZone.FAR:
        # Advance into mid range
        advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
        return ResolverOutput(
            commitment=advance,
            positioning_bias=0.8,
            commit_delay=0,
            reason_tags=("bait", "approach"),
        )

    if ctx.spacing == SpacingZone.CLOSE:
        # At close range, dodge backward to create punish window
        return ResolverOutput(
            commitment=CombatCommitment.DODGE_BACKWARD,
            positioning_bias=-0.5,
            commit_delay=min(6, cfg.max_commit_delay_ticks),
            reason_tags=("bait", "dodge_create_window"),
        )

    # Mid range — wait at optimal distance
    return ResolverOutput(
        commitment=CombatCommitment.DODGE_BACKWARD,
        positioning_bias=-0.2,
        commit_delay=min(8, cfg.max_commit_delay_ticks),
        reason_tags=("bait", "hold_mid"),
    )


def _resolve_punish_recovery(
    ctx: AIContext, cfg: ActionResolverConfig,
) -> ResolverOutput:
    """Punish player during recovery frames."""
    if ctx.player_is_recovering and ctx.spacing in (SpacingZone.CLOSE, SpacingZone.MID):
        # Heavy attack for max damage if close enough
        if ctx.spacing == SpacingZone.CLOSE and ctx.ai_stamina_frac > 0.3:
            return ResolverOutput(
                commitment=CombatCommitment.HEAVY_ATTACK,
                positioning_bias=0.5,
                commit_delay=0,
                reason_tags=("punish", "heavy"),
            )
        return ResolverOutput(
            commitment=CombatCommitment.LIGHT_ATTACK,
            positioning_bias=0.5,
            commit_delay=0,
            reason_tags=("punish", "light"),
        )

    # Player not in recovery — advance for position
    advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
    return ResolverOutput(
        commitment=advance,
        positioning_bias=0.6,
        commit_delay=0,
        reason_tags=("punish", "advance"),
    )


def _resolve_pressure(ctx: AIContext, cfg: ActionResolverConfig) -> ResolverOutput:
    """Pressure stamina with fast attacks and movement."""
    if ctx.spacing == SpacingZone.CLOSE:
        # Fast attack to drain opponent stamina via blockstun
        return ResolverOutput(
            commitment=CombatCommitment.LIGHT_ATTACK,
            positioning_bias=0.4,
            commit_delay=0,
            reason_tags=("pressure", "attack"),
        )

    if ctx.spacing == SpacingZone.MID:
        # Advance into range
        advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
        return ResolverOutput(
            commitment=advance,
            positioning_bias=0.7,
            commit_delay=0,
            reason_tags=("pressure", "advance"),
        )

    # Far — close distance
    advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
    return ResolverOutput(
        commitment=advance,
        positioning_bias=1.0,
        commit_delay=0,
        reason_tags=("pressure", "close_gap"),
    )


def _resolve_defensive(ctx: AIContext, cfg: ActionResolverConfig) -> ResolverOutput:
    """Retreat and recover stamina."""
    if ctx.spacing == SpacingZone.CLOSE:
        return ResolverOutput(
            commitment=CombatCommitment.DODGE_BACKWARD,
            positioning_bias=-1.0,
            commit_delay=0,
            reason_tags=("defensive", "escape"),
        )

    # Move away
    retreat = CombatCommitment.MOVE_LEFT if ctx.ai_facing > 0 else CombatCommitment.MOVE_RIGHT
    return ResolverOutput(
        commitment=retreat,
        positioning_bias=-0.8,
        commit_delay=0,
        reason_tags=("defensive", "retreat"),
    )


def _resolve_probe(ctx: AIContext, cfg: ActionResolverConfig) -> ResolverOutput:
    """Exploratory action to gather information about player behavior."""
    if ctx.spacing == SpacingZone.FAR:
        advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
        return ResolverOutput(
            commitment=advance,
            positioning_bias=0.5,
            commit_delay=0,
            reason_tags=("probe", "approach"),
        )

    # At mid/close, use a light attack to provoke a response
    delay = min(10, cfg.max_commit_delay_ticks)
    return ResolverOutput(
        commitment=CombatCommitment.LIGHT_ATTACK,
        positioning_bias=0.0,
        commit_delay=delay,
        reason_tags=("probe", "provoke"),
    )


def _resolve_neutral(ctx: AIContext, cfg: ActionResolverConfig) -> ResolverOutput:
    """Maintain optimal spacing without commitment."""
    if ctx.spacing == SpacingZone.CLOSE:
        retreat = CombatCommitment.MOVE_LEFT if ctx.ai_facing > 0 else CombatCommitment.MOVE_RIGHT
        return ResolverOutput(
            commitment=retreat,
            positioning_bias=-0.4,
            commit_delay=0,
            reason_tags=("neutral", "create_space"),
        )

    if ctx.spacing == SpacingZone.FAR:
        advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
        return ResolverOutput(
            commitment=advance,
            positioning_bias=0.4,
            commit_delay=0,
            reason_tags=("neutral", "close_gap"),
        )

    # Mid range — hold position
    advance = CombatCommitment.MOVE_RIGHT if ctx.ai_facing > 0 else CombatCommitment.MOVE_LEFT
    return ResolverOutput(
        commitment=advance,
        positioning_bias=0.1,
        commit_delay=min(5, cfg.max_commit_delay_ticks),
        reason_tags=("neutral", "hold"),
    )


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_ADVANCE_ACTIONS = frozenset({
    CombatCommitment.MOVE_LEFT,
    CombatCommitment.MOVE_RIGHT,
    CombatCommitment.LIGHT_ATTACK,
})
