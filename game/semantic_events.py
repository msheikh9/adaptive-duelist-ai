"""Builders for the SemanticEvent stream.

Split out of the engine so that anything driving a match can produce the same
event stream the live game does. That matters because the prediction ensemble is
driven entirely by these events: the batch evaluator used to skip them, which is
why prediction accuracy could never be measured off-line.

These functions are pure — they read state and return an event. Routing (to the
logger, the behaviour model, the prediction engine) is the caller's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from data.events import EventType, SemanticEvent
from game.arena import classify_spacing
from game.combat.actions import Actor

if TYPE_CHECKING:
    from config.config_loader import GameConfig
    from game.combat.actions import CombatCommitment
    from game.state import SimulationState


def _spacing(state: SimulationState, gcfg: GameConfig):
    return classify_spacing(
        state.distance_sub(),
        gcfg.spacing.close_max,
        gcfg.spacing.mid_max,
        gcfg.simulation.sub_pixel_scale,
    )


def _sides(state: SimulationState, actor: Actor):
    """Return (actor_fighter, opponent_fighter) for `actor`."""
    if actor == Actor.PLAYER:
        return state.player, state.ai
    return state.ai, state.player


def build_commitment_event(
    state: SimulationState,
    gcfg: GameConfig,
    match_id: str,
    actor: Actor,
    commitment: CombatCommitment,
    reaction_ticks: int,
) -> SemanticEvent:
    """A COMMITMENT_START event.

    `reaction_ticks` is ticks elapsed since the *opponent's* last commitment;
    the caller owns that bookkeeping because it depends on emission order.
    """
    fighter, opponent = _sides(state, actor)
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id=match_id,
        tick_id=state.tick_id,
        actor=actor,
        commitment=commitment,
        opponent_fsm_state=opponent.fsm_state,
        opponent_commitment=opponent.active_commitment,
        spacing_zone=_spacing(state, gcfg),
        actor_hp=fighter.hp,
        opponent_hp=opponent.hp,
        actor_stamina=fighter.stamina,
        opponent_stamina=opponent.stamina,
        reaction_ticks=reaction_ticks,
    )


def build_hit_event(
    state: SimulationState,
    gcfg: GameConfig,
    match_id: str,
    attacker_actor: Actor,
    hit,
) -> SemanticEvent:
    """A HIT_LANDED event, attributed to the attacker."""
    attacker, defender = _sides(state, attacker_actor)
    return SemanticEvent(
        event_type=EventType.HIT_LANDED,
        match_id=match_id,
        tick_id=state.tick_id,
        actor=attacker_actor,
        commitment=hit.attacker_commitment,
        opponent_fsm_state=defender.fsm_state,
        spacing_zone=_spacing(state, gcfg),
        actor_hp=attacker.hp,
        opponent_hp=defender.hp,
        actor_stamina=attacker.stamina,
        opponent_stamina=defender.stamina,
        damage_dealt=hit.damage,
    )


def build_simple_event(
    state: SimulationState,
    match_id: str,
    event_type: EventType,
    actor: Actor,
) -> SemanticEvent:
    """An event carrying no commitment or damage — COMMITMENT_END and friends."""
    fighter, opponent = _sides(state, actor)
    return SemanticEvent(
        event_type=event_type,
        match_id=match_id,
        tick_id=state.tick_id,
        actor=actor,
        actor_hp=fighter.hp,
        opponent_hp=opponent.hp,
        actor_stamina=fighter.stamina,
        opponent_stamina=opponent.stamina,
    )


class EventRouter:
    """Turns a TickEffects into the semantic event stream, in engine order.

    Both the live engine and the batch evaluator route through this, so an
    off-line evaluation sees exactly the event sequence a played match would.
    That equivalence is the whole reason prediction accuracy is measurable
    off-line at all: the ensemble only ever predicts in response to these
    events.

    `sinks` receive every event. `planner`, when present, additionally gets the
    two callbacks it needs to score its own predictions and outcomes.
    """

    def __init__(
        self,
        gcfg: GameConfig,
        sinks: "list[Callable[[SemanticEvent], None]]",
        planner=None,
    ) -> None:
        self._gcfg = gcfg
        self._sinks = sinks
        self._planner = planner

    def _emit(self, event: SemanticEvent) -> None:
        for sink in self._sinks:
            sink(event)

    def route(
        self,
        state: SimulationState,
        match_id: str,
        ctx,
        fx,
    ) -> None:
        """Emit every event implied by `fx`, in the engine's canonical order."""
        # COMMITMENT_START — player first, then AI. Reaction time is measured
        # against the opponent's previous commitment, so the bookkeeping update
        # has to happen per-emission and in this order.
        for actor, commitments in (
            (Actor.PLAYER, fx.player_commitments),
            (Actor.AI, fx.ai_commitments),
        ):
            for commitment in commitments:
                reaction = ctx.reaction_ticks(actor, state.tick_id)
                if actor == Actor.PLAYER:
                    ctx.last_player_commit_tick = state.tick_id
                else:
                    ctx.last_ai_commit_tick = state.tick_id
                self._emit(build_commitment_event(
                    state, self._gcfg, match_id, actor, commitment, reaction))
                if actor == Actor.PLAYER and self._planner is not None:
                    self._planner.on_player_commit(commitment, state.tick_id)

        # HIT_LANDED
        if fx.player_hit:
            self._emit(build_hit_event(
                state, self._gcfg, match_id, Actor.PLAYER, fx.player_hit))
        if fx.ai_hit:
            self._emit(build_hit_event(
                state, self._gcfg, match_id, Actor.AI, fx.ai_hit))

        # STAMINA_EXHAUSTED
        if fx.player_exhausted:
            self._emit(build_simple_event(
                state, match_id, EventType.STAMINA_EXHAUSTED, Actor.PLAYER))
        if fx.ai_exhausted:
            self._emit(build_simple_event(
                state, match_id, EventType.STAMINA_EXHAUSTED, Actor.AI))

        # COMMITMENT_END — this is the ensemble's primary inference trigger.
        if fx.player_commitment_end:
            self._emit(build_simple_event(
                state, match_id, EventType.COMMITMENT_END, Actor.PLAYER))
        if fx.ai_commitment_end:
            self._emit(build_simple_event(
                state, match_id, EventType.COMMITMENT_END, Actor.AI))
            if self._planner is not None:
                self._planner.on_ai_commit_end(
                    state.tick_id, state.ai.hp, state.player.hp)
