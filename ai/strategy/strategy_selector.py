"""Strategy selector: scores and selects a TacticalIntent for the current context.

Scoring combines base heuristics (prediction alignment, HP differential,
stamina suitability, risk) with six PlannerMemory-driven adjustments:
  - staleness_penalty
  - consecutive_penalty
  - exploration_pressure
  - accuracy_trend_adjustment
  - shift_response
  - historical_mode_success

Final selection uses softmax-weighted sampling from the score vector,
with a forced PROBE override when exploration_budget < budget_floor.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ai.strategy.tactics import TacticalIntent
from game.combat.actions import FSMState, SpacingZone

if TYPE_CHECKING:
    from ai.strategy.ai_context import AIContext
    from ai.strategy.planner_memory import PlannerMemory
    from config.config_loader import StrategyConfig, PlannerMemoryConfig

ALL_MODES = list(TacticalIntent)


def select_mode(
    ctx: AIContext,
    memory: PlannerMemory,
    strategy_cfg: StrategyConfig,
    memory_cfg: PlannerMemoryConfig,
    rng: random.Random,
) -> TacticalIntent:
    """Score all tactical modes and select one.

    Returns PROBE_BEHAVIOR unconditionally when exploration budget is
    below the configured floor.
    """
    # Forced probe on low exploration budget
    if memory.exploration_budget < memory_cfg.exploration_budget_floor:
        return TacticalIntent.PROBE_BEHAVIOR

    # Forced probe during shift detection probe window
    if memory.in_shift_probe:
        return TacticalIntent.PROBE_BEHAVIOR

    w = strategy_cfg.scoring_weights
    scores: dict[TacticalIntent, float] = {}

    for mode in ALL_MODES:
        base = _base_score(mode, ctx, w)
        adj = _memory_adjustments(mode, ctx, memory, memory_cfg, w)
        scores[mode] = base + adj

    return _softmax_select(scores, strategy_cfg.softmax_temperature, rng)


# ------------------------------------------------------------------ #
# Base scoring                                                         #
# ------------------------------------------------------------------ #

def _base_score(mode: TacticalIntent, ctx: AIContext, w) -> float:
    """Compute base score from game context only (no memory)."""
    s = 0.0

    # --- Prediction alignment ---
    # Modes that rely on prediction quality benefit from high confidence
    pred_conf = ctx.prediction.commitment_confidence
    if mode == TacticalIntent.EXPLOIT_PATTERN:
        s += w.prediction_alignment * pred_conf
    elif mode == TacticalIntent.BAIT_AND_PUNISH:
        s += w.prediction_alignment * pred_conf * 0.7
    elif mode == TacticalIntent.PROBE_BEHAVIOR:
        # Probe is more valuable when confidence is low
        s += w.prediction_alignment * (1.0 - pred_conf) * 0.5

    # --- HP differential ---
    hp_adv = ctx.hp_advantage
    if mode == TacticalIntent.PRESSURE_STAMINA:
        s += w.hp_differential * max(hp_adv, 0.0)
    elif mode == TacticalIntent.DEFENSIVE_RESET:
        s += w.hp_differential * max(-hp_adv, 0.0)

    # --- Stamina suitability ---
    ai_stam = ctx.ai_stamina_frac
    if mode in (TacticalIntent.PRESSURE_STAMINA, TacticalIntent.EXPLOIT_PATTERN):
        s += w.stamina_suitability * ai_stam
    elif mode == TacticalIntent.DEFENSIVE_RESET:
        s += w.stamina_suitability * (1.0 - ai_stam) * 0.5

    # --- Risk penalty / survival instinct ---
    # Aggressive modes penalized at low HP; defensive boosted
    if ctx.ai_hp_frac < 0.3:
        risk_severity = (0.3 - ctx.ai_hp_frac) / 0.3  # 0.0 at 30%, 1.0 at 0%
        if mode in (TacticalIntent.EXPLOIT_PATTERN, TacticalIntent.PRESSURE_STAMINA):
            s -= w.risk_penalty * risk_severity
        elif mode == TacticalIntent.DEFENSIVE_RESET:
            s += w.risk_penalty * risk_severity

    # --- Spacing context ---
    if mode == TacticalIntent.PUNISH_RECOVERY:
        # Only viable when player is in recovery and close
        if ctx.player_is_recovering and ctx.spacing == SpacingZone.CLOSE:
            s += 1.5
        elif ctx.player_is_recovering:
            s += 0.5
        else:
            s -= 0.5  # not applicable

    if mode == TacticalIntent.NEUTRAL_SPACING:
        if ctx.spacing == SpacingZone.FAR:
            s += 0.3
        elif ctx.spacing == SpacingZone.CLOSE:
            s += 0.1

    if mode == TacticalIntent.BAIT_AND_PUNISH:
        if ctx.spacing == SpacingZone.MID:
            s += 0.3

    # Exploration bonus baseline for PROBE
    if mode == TacticalIntent.PROBE_BEHAVIOR:
        s += w.exploration_bonus * 0.3

    return s


# ------------------------------------------------------------------ #
# Memory-driven adjustments                                            #
# ------------------------------------------------------------------ #

def _memory_adjustments(
    mode: TacticalIntent,
    ctx: AIContext,
    memory: PlannerMemory,
    memory_cfg: PlannerMemoryConfig,
    w,
) -> float:
    """Six adjustments from PlannerMemory."""
    adj = 0.0

    # 1. Staleness penalty: penalize EXPLOIT when exploit target is stale
    if mode == TacticalIntent.EXPLOIT_PATTERN:
        staleness = memory.exploit_staleness(ctx.tick_id)
        adj -= w.staleness_penalty * staleness

    # 2. Consecutive penalty: penalize repeating the same mode
    if memory.consecutive_same_mode >= memory_cfg.max_consecutive_before_penalty:
        if mode == (memory._consecutive_mode):
            excess = memory.consecutive_same_mode - memory_cfg.max_consecutive_before_penalty
            adj -= w.consecutive_penalty * min(excess * 0.2, 1.0)

    # 3. Exploration pressure: boost PROBE as budget drains
    if mode == TacticalIntent.PROBE_BEHAVIOR:
        deficit = 1.0 - memory.exploration_budget
        adj += w.exploration_pressure * deficit * 0.5

    # 4. Accuracy trend: when accuracy is improving, favor exploit;
    #    when degrading, favor defensive/probe
    trend = memory.accuracy_trend
    if mode == TacticalIntent.EXPLOIT_PATTERN:
        adj += w.accuracy_trend * trend
    elif mode in (TacticalIntent.DEFENSIVE_RESET, TacticalIntent.PROBE_BEHAVIOR):
        adj -= w.accuracy_trend * trend  # these get boosted when trend is negative

    # 5. Shift response: on detected shift, boost probe and defensive
    if memory.shift_detected:
        if mode == TacticalIntent.PROBE_BEHAVIOR:
            adj += w.shift_response * 0.8
        elif mode == TacticalIntent.DEFENSIVE_RESET:
            adj += w.shift_response * 0.4
        elif mode == TacticalIntent.EXPLOIT_PATTERN:
            adj -= w.shift_response * 0.6

    # 6. Historical mode success rate
    success = memory.per_mode_success_rate(mode)
    adj += w.mode_success_rate * (success - 0.5)  # center at 0.5

    return adj


# ------------------------------------------------------------------ #
# Softmax selection                                                    #
# ------------------------------------------------------------------ #

def _softmax_select(
    scores: dict[TacticalIntent, float],
    temperature: float,
    rng: random.Random,
) -> TacticalIntent:
    """Select a mode via softmax-weighted random sampling."""
    if temperature <= 0:
        return max(scores, key=scores.__getitem__)

    modes = list(scores.keys())
    raw = [scores[m] / temperature for m in modes]

    # Numerical stability: subtract max
    max_r = max(raw)
    exps = [math.exp(r - max_r) for r in raw]
    total = sum(exps)
    probs = [e / total for e in exps]

    return rng.choices(modes, weights=probs, k=1)[0]
