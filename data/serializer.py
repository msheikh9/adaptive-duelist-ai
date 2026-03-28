"""Serialization helpers for converting game types to storable formats.

Handles Enum → string/int, dataclass → dict, and numpy type coercion.
Used by the logger, replay recorder, and analytics exporter.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any


def enum_to_str(value: Enum) -> str:
    """Convert an enum member to its name string."""
    return value.name


def enum_to_int(value: Enum) -> int:
    """Convert an enum member to its integer value."""
    return value.value


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-safe dict.

    Enums are converted to their name strings.
    Nested dataclasses are recursively converted.
    None values are preserved.
    """
    if not is_dataclass(obj):
        raise TypeError(f"Expected a dataclass, got {type(obj).__name__}")

    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        result[f.name] = _convert_value(value)
    return result


def _convert_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return to_dict(value)
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _convert_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)):
        return value
    # Handle numpy scalar types if present
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except ImportError:
        pass
    return str(value)


def semantic_event_to_row(event) -> tuple:
    """Convert a SemanticEvent to a tuple matching the SQLite insert order.

    Returns:
        (event_type, match_id, tick_id, actor, commitment,
         opponent_fsm_state, opponent_commitment, spacing_zone,
         actor_hp, opponent_hp, actor_stamina, opponent_stamina,
         damage_dealt, reaction_ticks)
    """
    return (
        event.event_type.name,
        event.match_id,
        event.tick_id,
        event.actor.name,
        event.commitment.name if event.commitment else None,
        event.opponent_fsm_state.name if event.opponent_fsm_state else None,
        event.opponent_commitment.name if event.opponent_commitment else None,
        event.spacing_zone.name if event.spacing_zone else None,
        event.actor_hp,
        event.opponent_hp,
        event.actor_stamina,
        event.opponent_stamina,
        event.damage_dealt,
        event.reaction_ticks,
    )
