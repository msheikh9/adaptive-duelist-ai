"""Feature extraction for the sklearn prediction layer.

Converts a SemanticEvent (with surrounding context) into a fixed-length
numeric feature vector. The 35-feature layout:

  [0..13]  Game-state context (14 features)
  [14..28] One-hot history: last 3 commitments × 5 Phase1 names (15 features)
  [29..34] Profile aggregates (6 features)

No TickSnapshot data is used — only SemanticEvent fields and the
PlayerProfile. This avoids any coupling to the dense per-tick layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai.models.base_predictor import PHASE1_NAMES
from data.events import SemanticEvent
from game.combat.actions import CombatCommitment, FSMState, SpacingZone

if TYPE_CHECKING:
    from ai.profile.player_profile import PlayerProfile

NUM_PHASE1 = len(PHASE1_NAMES)  # 5
HISTORY_SLOTS = 3
NUM_FEATURES = 14 + (HISTORY_SLOTS * NUM_PHASE1) + 6  # 35

# Map Phase1 commitment names to one-hot indices
_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(PHASE1_NAMES)}

# Map SpacingZone to ordinal
_SPACING_ORD: dict[SpacingZone, float] = {
    SpacingZone.CLOSE: 0.0,
    SpacingZone.MID: 0.5,
    SpacingZone.FAR: 1.0,
}

# Map FSMState to ordinal bucket (0=free, 0.5=active, 1.0=recovery/locked)
_FSM_ORD: dict[FSMState, float] = {
    FSMState.IDLE: 0.0,
    FSMState.MOVING: 0.0,
    FSMState.ATTACK_STARTUP: 0.5,
    FSMState.ATTACK_ACTIVE: 0.5,
    FSMState.ATTACK_RECOVERY: 1.0,
    FSMState.DODGE_STARTUP: 0.5,
    FSMState.DODGING: 0.5,
    FSMState.DODGE_RECOVERY: 1.0,
    FSMState.BLOCKING: 0.5,
    FSMState.BLOCKSTUN: 1.0,
    FSMState.HITSTUN: 1.0,
    FSMState.PARRY_STUNNED: 1.0,
    FSMState.EXHAUSTED: 1.0,
    FSMState.KO: 1.0,
}


def extract_features(
    event: SemanticEvent,
    history: list[str],
    profile: PlayerProfile,
    max_hp: int,
    max_stamina: int,
    tick_rate: int,
) -> list[float]:
    """Build a 35-element feature vector from a single observation point.

    Parameters
    ----------
    event:       The COMMITMENT_START (or equivalent trigger) event.
    history:     Recent commitment names *before* this event (most recent last).
                 May be shorter than HISTORY_SLOTS.
    profile:     Current PlayerProfile snapshot.
    max_hp:      Max fighter HP from game config.
    max_stamina: Max fighter stamina from game config.
    tick_rate:   Simulation tick rate from game config.
    """
    vec: list[float] = []

    # --- Game-state context (14 features) ---

    # 0: actor_hp_frac
    vec.append(event.actor_hp / max_hp if max_hp > 0 else 0.0)
    # 1: opponent_hp_frac
    vec.append(event.opponent_hp / max_hp if max_hp > 0 else 0.0)
    # 2: actor_stamina_frac
    vec.append(event.actor_stamina / max_stamina if max_stamina > 0 else 0.0)
    # 3: opponent_stamina_frac
    vec.append(event.opponent_stamina / max_stamina if max_stamina > 0 else 0.0)
    # 4: spacing_zone ordinal
    vec.append(_SPACING_ORD.get(event.spacing_zone, 0.5) if event.spacing_zone else 0.5)
    # 5: opponent_fsm ordinal
    vec.append(_FSM_ORD.get(event.opponent_fsm_state, 0.0) if event.opponent_fsm_state else 0.0)
    # 6: hp_differential (player - ai, normalized)
    vec.append((event.actor_hp - event.opponent_hp) / max_hp if max_hp > 0 else 0.0)
    # 7: stamina_differential
    vec.append(
        (event.actor_stamina - event.opponent_stamina) / max_stamina
        if max_stamina > 0 else 0.0
    )
    # 8: is_low_hp (actor below 30%)
    vec.append(1.0 if event.actor_hp < max_hp * 0.30 else 0.0)
    # 9: is_opponent_low_hp
    vec.append(1.0 if event.opponent_hp < max_hp * 0.30 else 0.0)
    # 10: is_close_range
    vec.append(1.0 if event.spacing_zone == SpacingZone.CLOSE else 0.0)
    # 11: is_far_range
    vec.append(1.0 if event.spacing_zone == SpacingZone.FAR else 0.0)
    # 12: reaction_ticks normalized (capped at 2 seconds worth)
    max_react = tick_rate * 2  # 120 ticks at 60Hz
    rt = event.reaction_ticks if event.reaction_ticks and event.reaction_ticks > 0 else 0
    vec.append(min(rt / max_react, 1.0))
    # 13: tick_id fraction of typical match length (cap at 6000 = ~100s)
    vec.append(min(event.tick_id / 6000.0, 1.0))

    assert len(vec) == 14

    # --- One-hot history (15 features: 3 slots × 5 Phase1 names) ---
    # Pad history from the left if shorter than HISTORY_SLOTS
    padded = ([""] * max(0, HISTORY_SLOTS - len(history))
              + history[-HISTORY_SLOTS:])

    for slot_name in padded:
        one_hot = [0.0] * NUM_PHASE1
        idx = _NAME_TO_IDX.get(slot_name, -1)
        if idx >= 0:
            one_hot[idx] = 1.0
        vec.extend(one_hot)

    assert len(vec) == 14 + 15

    # --- Profile aggregates (6 features) ---
    # 29: aggression_index
    vec.append(profile.aggression_index)
    # 30: initiative_rate
    vec.append(profile.initiative_rate)
    # 31: dodge_frequency
    vec.append(profile.dodge_frequency)
    # 32: movement_direction_bias (re-scaled from [-1,1] to [0,1])
    vec.append((profile.movement_direction_bias + 1.0) / 2.0)
    # 33: avg_reaction_time_ms normalized (cap at 2000ms)
    vec.append(min(profile.avg_reaction_time_ms / 2000.0, 1.0))
    # 34: win_rate_vs_ai
    vec.append(profile.win_rate_vs_ai)

    assert len(vec) == NUM_FEATURES
    return vec
