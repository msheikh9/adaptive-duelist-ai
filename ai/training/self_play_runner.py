"""Headless self-play match runner for training data generation.

Runs T2 AI vs ScriptedOpponent (player slot). Writes matches +
semantic_events + ai_decisions to a persistent DB. All matches
tagged source='self_play'.

No Pygame. No latency instrumentation. Deterministic by seed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai.training.scripted_opponent import ScriptedOpponent, ScriptedProfile

if TYPE_CHECKING:
    from config.config_loader import AIConfig, GameConfig

log = logging.getLogger(__name__)


@dataclass
class SelfPlayResult:
    matches_run: int
    profiles_used: list[str] = field(default_factory=list)
    semantic_events_inserted: int = 0
    match_ids: list[str] = field(default_factory=list)


def run_self_play(
    n_matches: int,
    profiles: list[ScriptedProfile],
    seed_start: int,
    db_path: Path,
    game_cfg: "GameConfig",
    ai_cfg: "AIConfig",
    max_ticks: int = 5000,
) -> SelfPlayResult:
    """Run N headless matches: T2 AI vs ScriptedOpponent.

    Profiles cycle: match i uses profiles[i % len(profiles)].
    Each match uses seed seed_start + i.
    Writes matches + semantic_events to db_path.
    """
    from ai.layers.behavior_model import BehaviorModel
    from ai.layers.prediction_engine import PredictionEngine
    from ai.layers.tactical_planner import TacticalPlanner, AITier
    from data.db import Database
    from data.migrations.migration_runner import run_migrations
    from game.combat.actions import FSMState, FREE_STATES
    from game.combat.collision import HitTracker, check_hit
    from game.combat.damage import apply_hit
    from game.combat.physics import (
        apply_dodge_velocity, apply_velocity, clamp_to_arena, update_facing,
    )
    from game.combat.stamina import tick_stamina
    from game.combat.state_machine import tick_fsm
    from game.state import (
        ArenaState, FighterState, MatchStatus, SimulationState, TickPhase,
    )
    from game.arena import classify_spacing

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    result = SelfPlayResult(matches_run=0)
    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    fighter_w_sub = game_cfg.fighter.width * scale

    for i in range(n_matches):
        seed = seed_start + i
        profile = profiles[i % len(profiles)]
        match_id = f"sp-{profile.value.lower()}-{i:06d}"

        bm = BehaviorModel(db, ai_cfg, game_cfg)
        bm.load_profile()
        pe = PredictionEngine(db, bm, ai_cfg, game_cfg)
        planner = TacticalPlanner(db, pe, ai_cfg, game_cfg, AITier.T2_FULL_ADAPTIVE)

        sim = SimulationState(
            tick_id=0, rng_seed=seed,
            player=FighterState(
                x=arena.width_sub // 3, y=arena.ground_y_sub,
                hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
                facing=1,
            ),
            ai=FighterState(
                x=(arena.width_sub * 2) // 3, y=arena.ground_y_sub,
                hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
                facing=-1,
            ),
            arena=arena, match_status=MatchStatus.ACTIVE,
        )

        db.execute_safe(
            """INSERT INTO matches
               (match_id, session_id, started_at, rng_seed, config_hash, source)
               VALUES (?, 'self_play', datetime('now'), ?, 'self_play', 'self_play')""",
            (match_id, seed),
        )

        bm.on_match_start(match_id)
        pe.on_match_start(match_id)
        planner.on_match_start(match_id, "self_play", seed)

        opponent = ScriptedOpponent(profile, seed=seed)
        hit_tracker = HitTracker()
        events_this_match = 0
        last_free_tick: int = 0

        for tick in range(max_ticks):
            if sim.match_status == MatchStatus.ENDED:
                break
            sim.tick_id = tick
            sim.set_phase(TickPhase.SIMULATE)

            # Track when player fighter enters free state
            if sim.player.fsm_state in FREE_STATES:
                last_free_tick = tick

            player_commit = opponent.decide(sim.player, sim, game_cfg)
            ai_commit = planner.decide(sim.ai, sim, game_cfg)

            # Write semantic event for scripted opponent commitment
            if player_commit is not None:
                spacing = classify_spacing(
                    sim.distance_sub(),
                    game_cfg.spacing.close_max,
                    game_cfg.spacing.mid_max,
                    scale,
                )
                reaction_ticks = tick - last_free_tick
                db.execute_safe(
                    """INSERT INTO semantic_events
                       (event_type, match_id, tick_id, actor, commitment,
                        opponent_fsm_state, spacing_zone,
                        actor_hp, opponent_hp, actor_stamina, opponent_stamina,
                        reaction_ticks, damage_dealt)
                       VALUES (?, ?, ?, 'PLAYER', ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        'COMMITMENT_START', match_id, tick,
                        player_commit.name,
                        sim.ai.fsm_state.name,
                        spacing.name if spacing else None,
                        sim.player.hp, sim.ai.hp,
                        sim.player.stamina, sim.ai.stamina,
                        reaction_ticks,
                    ),
                )
                events_this_match += 1

            apply_dodge_velocity(sim.player, game_cfg)
            apply_dodge_velocity(sim.ai, game_cfg)
            apply_velocity(sim.player)
            apply_velocity(sim.ai)
            clamp_to_arena(sim.player, sim.arena, fighter_w_sub)
            clamp_to_arena(sim.ai, sim.arena, fighter_w_sub)
            update_facing(sim.player, sim.ai)

            p_hit = check_hit(sim.player, sim.ai, "player", hit_tracker, game_cfg)
            a_hit = check_hit(sim.ai, sim.player, "ai", hit_tracker, game_cfg)
            if p_hit:
                apply_hit(sim.ai, p_hit)
            if a_hit:
                apply_hit(sim.player, a_hit)

            tick_stamina(sim.player, game_cfg)
            tick_stamina(sim.ai, game_cfg)

            if sim.player.is_free:
                hit_tracker.reset("player")
            if sim.ai.is_free:
                hit_tracker.reset("ai")

            tick_fsm(sim.player, game_cfg)
            tick_fsm(sim.ai, game_cfg)

            if sim.player.fsm_state == FSMState.KO:
                sim.match_status = MatchStatus.ENDED
                sim.winner = "AI"
            elif sim.ai.fsm_state == FSMState.KO:
                sim.match_status = MatchStatus.ENDED
                sim.winner = "PLAYER"

        planner.on_match_end()
        bm.on_match_end()

        # Update match row with outcome
        winner = sim.winner or "DRAW"
        db.execute_safe(
            """UPDATE matches SET ended_at=datetime('now'), total_ticks=?,
               winner=?, player_hp_final=?, ai_hp_final=?
               WHERE match_id=?""",
            (sim.tick_id, winner, sim.player.hp, sim.ai.hp, match_id),
        )

        result.matches_run += 1
        result.profiles_used.append(profile.value)
        result.semantic_events_inserted += events_this_match
        result.match_ids.append(match_id)

        log.info(
            "Self-play match %d/%d: profile=%s seed=%d winner=%s events=%d",
            i + 1, n_matches, profile.value, seed, winner, events_this_match,
        )

    db.close()
    return result
