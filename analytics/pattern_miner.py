"""Mines player behavioral patterns from profiles and semantic events.

Reads player_profiles + semantic_events to build a TacticalPatternSummary.
Works even when profile fields are sparse or the profile row is missing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analytics.explanation_types import TacticalPatternSummary

if TYPE_CHECKING:
    from data.db import Database

PLAYER_ID = "player_1"


def mine_patterns(
    db: Database,
    player_id: str = PLAYER_ID,
) -> TacticalPatternSummary:
    """Build a TacticalPatternSummary from player profile data."""
    profile_row = db.fetchone(
        "SELECT * FROM player_profiles WHERE player_id = ?;",
        (player_id,),
    )

    action_freqs = _safe_json(profile_row, "action_frequencies")
    bigrams = _safe_json(profile_row, "bigrams")
    trigrams = _safe_json(profile_row, "trigrams")
    spacing = _safe_json(profile_row, "spacing_distribution")
    aggression = _safe_float(profile_row, "aggression_index")
    dodge_freq = _safe_float(profile_row, "dodge_frequency")
    initiative = _safe_float(profile_row, "initiative_rate")
    move_bias = _safe_float(profile_row, "movement_direction_bias")

    total = sum(action_freqs.values()) if action_freqs else 0
    top_commits = sorted(action_freqs.items(), key=lambda x: -x[1])[:5]
    top_bi = _flatten_ngrams(bigrams, top_n=5)
    top_tri = _flatten_ngrams(trigrams, top_n=5)

    habits = _identify_exploitable_habits(
        action_freqs, total, bigrams, spacing, aggression, dodge_freq,
    )

    return TacticalPatternSummary(
        player_id=player_id,
        total_commitments=total,
        top_commitments=top_commits,
        top_bigrams=top_bi,
        top_trigrams=top_tri,
        spacing_tendencies=spacing or {},
        aggression_index=aggression,
        dodge_frequency=dodge_freq,
        initiative_rate=initiative,
        movement_bias=move_bias,
        exploitable_habits=habits,
    )


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _safe_json(row, key: str) -> dict:
    if row is None:
        return {}
    raw = row[key]
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_float(row, key: str) -> float:
    if row is None:
        return 0.0
    val = row[key]
    return float(val) if val is not None else 0.0


def _flatten_ngrams(ngrams: dict, top_n: int = 5) -> list[tuple[str, int]]:
    """Flatten nested {prefix: {next: count}} into sorted (label, count) list."""
    flat: dict[str, int] = {}
    for prefix, transitions in ngrams.items():
        if isinstance(transitions, dict):
            for next_action, count in transitions.items():
                key = f"{prefix} \u2192 {next_action}"
                flat[key] = flat.get(key, 0) + count
    return sorted(flat.items(), key=lambda x: -x[1])[:top_n]


def _identify_exploitable_habits(
    action_freqs: dict[str, int],
    total: int,
    bigrams: dict,
    spacing: dict[str, int],
    aggression: float,
    dodge_freq: float,
) -> list[str]:
    """Identify exploitable tendencies from profile data."""
    habits: list[str] = []
    if total == 0:
        return habits

    if aggression > 0.65:
        habits.append(
            f"Aggressive playstyle (aggression={aggression:.2f}); "
            "bait-and-punish or defensive counter effective."
        )

    if dodge_freq > 0.25:
        habits.append(
            f"Frequent dodger ({dodge_freq:.0%}); "
            "advance on dodge to maintain pressure."
        )

    if action_freqs:
        top_action, top_count = max(action_freqs.items(), key=lambda x: x[1])
        pct = top_count / total
        if pct > 0.35:
            habits.append(
                f"Over-relies on {top_action} ({pct:.0%} of actions); "
                "counter specifically."
            )

    for prefix, transitions in bigrams.items():
        if not isinstance(transitions, dict):
            continue
        total_from = sum(transitions.values())
        if total_from < 5:
            continue
        top_next, top_count = max(transitions.items(), key=lambda x: x[1])
        pct = top_count / total_from
        if pct > 0.55:
            habits.append(
                f"After {prefix}, tends to {top_next} ({pct:.0%}); "
                "exploit this transition."
            )

    total_spacing = sum(spacing.values()) if spacing else 0
    if total_spacing > 0:
        close_pct = spacing.get("CLOSE", 0) / total_spacing
        if close_pct > 0.5:
            habits.append(
                f"Prefers close range ({close_pct:.0%}); "
                "control spacing to deny comfort zone."
            )

    return habits
