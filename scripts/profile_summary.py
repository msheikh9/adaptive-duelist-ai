"""Run several headless matches and print the player profile after each.

Demonstrates that the profile evolves across matches as the behavior
model accumulates observations.

Usage: python3 scripts/profile_summary.py [--matches N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config.config_loader import load_config
from data.db import Database
from data.events import EventType, SemanticEvent
from data.migrations.migration_runner import run_migrations
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone
from game.state import MatchStatus
from ai.layers.behavior_model import BehaviorModel
from tests.fixtures.headless_engine import HeadlessMatch


def _make_player_commit_event(commitment: CombatCommitment, match_id: str,
                               tick_id: int, spacing: SpacingZone,
                               actor_hp: int, opponent_hp: int,
                               actor_stamina: int, opponent_stamina: int,
                               opponent_fsm: FSMState,
                               reaction_ticks: int) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.COMMITMENT_START,
        match_id=match_id,
        tick_id=tick_id,
        actor=Actor.PLAYER,
        commitment=commitment,
        opponent_fsm_state=opponent_fsm,
        spacing_zone=spacing,
        actor_hp=actor_hp,
        opponent_hp=opponent_hp,
        actor_stamina=actor_stamina,
        opponent_stamina=opponent_stamina,
        reaction_ticks=reaction_ticks,
    )


def _make_hit_event(match_id: str, tick_id: int, commitment: CombatCommitment,
                    actor: Actor, opponent_fsm: FSMState,
                    actor_hp: int, opponent_hp: int,
                    actor_stamina: int, opponent_stamina: int,
                    damage: int) -> SemanticEvent:
    return SemanticEvent(
        event_type=EventType.HIT_LANDED,
        match_id=match_id,
        tick_id=tick_id,
        actor=actor,
        commitment=commitment,
        opponent_fsm_state=opponent_fsm,
        actor_hp=actor_hp,
        opponent_hp=opponent_hp,
        actor_stamina=actor_stamina,
        opponent_stamina=opponent_stamina,
        damage_dealt=damage,
    )


def run_match_with_profile(model: BehaviorModel, match_id: str,
                            game_cfg, rng_seed: int) -> dict:
    """Run a headless match and feed synthetic player events into the model."""
    from game.combat.stamina import tick_stamina
    from game.combat.collision import HitTracker, check_hit
    from game.combat.damage import apply_hit
    from game.combat.physics import (
        apply_dodge_velocity, apply_velocity, clamp_to_arena, update_facing,
    )
    from game.combat.state_machine import tick_fsm
    from game.arena import classify_spacing
    from game.state import TickPhase
    from data.tick_snapshot import TickSnapshot

    match = HeadlessMatch(game_cfg, rng_seed=rng_seed)
    scale = game_cfg.simulation.sub_pixel_scale
    fighter_w_sub = game_cfg.fighter.width * scale

    state = match.state
    hit_tracker = HitTracker()
    last_ai_commit_tick = 0
    last_player_commit_tick = 0

    model.on_match_start(match_id)

    for _ in range(20000):
        if state.match_status == MatchStatus.ENDED:
            break

        state.set_phase(TickPhase.SIMULATE)

        # AI decision
        ai_commit = match.ai_ctrl.decide(state.ai, state, game_cfg)
        if ai_commit is not None:
            last_ai_commit_tick = state.tick_id

        # Physics
        apply_dodge_velocity(state.player, game_cfg)
        apply_dodge_velocity(state.ai, game_cfg)
        apply_velocity(state.player)
        apply_velocity(state.ai)
        clamp_to_arena(state.player, state.arena, fighter_w_sub)
        clamp_to_arena(state.ai, state.arena, fighter_w_sub)
        update_facing(state.player, state.ai)

        # Collision & damage
        player_hit = check_hit(state.player, state.ai, "player", hit_tracker, game_cfg)
        ai_hit = check_hit(state.ai, state.player, "ai", hit_tracker, game_cfg)
        if player_hit:
            apply_hit(state.ai, player_hit)
            event = _make_hit_event(
                match_id, state.tick_id, player_hit.attacker_commitment,
                Actor.PLAYER, state.ai.fsm_state,
                state.player.hp, state.ai.hp,
                state.player.stamina, state.ai.stamina,
                player_hit.damage,
            )
            model.on_event(event)
        if ai_hit:
            apply_hit(state.player, ai_hit)

        # Stamina
        tick_stamina(state.player, game_cfg)
        tick_stamina(state.ai, game_cfg)

        if state.player.is_free:
            hit_tracker.reset("player")
        if state.ai.is_free:
            hit_tracker.reset("ai")

        tick_fsm(state.player, game_cfg)
        tick_fsm(state.ai, game_cfg)

        # Simulate player commitments via baseline AI as proxy
        # (real player input not available in headless mode — use AI decision as stand-in)
        if ai_commit is not None:
            spacing = classify_spacing(
                state.distance_sub(),
                game_cfg.spacing.close_max,
                game_cfg.spacing.mid_max,
                scale,
            )
            reaction = state.tick_id - last_ai_commit_tick
            event = _make_player_commit_event(
                ai_commit, match_id, state.tick_id,
                spacing, state.player.hp, state.ai.hp,
                state.player.stamina, state.ai.stamina,
                state.ai.fsm_state, reaction,
            )
            model.on_event(event)
            last_player_commit_tick = state.tick_id

        # KO check
        if state.player.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "AI"
        elif state.ai.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "PLAYER"

        state.tick_id += 1

    model.on_match_end(state.winner, state.tick_id)
    return {
        "winner": state.winner or "DRAW",
        "ticks": state.tick_id,
        "player_hp": state.player.hp,
        "ai_hp": state.ai.hp,
    }


def print_profile_summary(model: BehaviorModel, match_num: int, result: dict) -> None:
    p = model.profile
    print(f"\n{'='*60}")
    print(f"After match {match_num:3d} | winner={result['winner']:6s} | "
          f"ticks={result['ticks']:5d} | P_hp={result['player_hp']:3d} AI_hp={result['ai_hp']:3d}")
    print(f"  {p.summary()}")
    print(f"  aggression={p.aggression_index:.2f}  "
          f"dodge_freq={p.dodge_frequency:.2f}  "
          f"move_bias={p.movement_direction_bias:+.2f}  "
          f"init_rate={p.initiative_rate:.2f}")
    print(f"  punish_conv={p.punish_conversion_rate:.2f}  "
          f"react_ms={p.avg_reaction_time_ms:.0f}±{p.reaction_time_stddev:.0f}")
    if p.action_frequencies:
        top = sorted(p.action_frequencies.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}={v}" for k, v in top)
        print(f"  top_actions: {top_str}")
    if p.spacing_distribution:
        sp = p.spacing_distribution
        print(f"  spacing: CLOSE={sp.get('CLOSE',0)} "
              f"MID={sp.get('MID',0)} FAR={sp.get('FAR',0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matches and show profile evolution")
    parser.add_argument("--matches", type=int, default=5, help="Number of matches")
    parser.add_argument("--seed", type=int, default=99, help="Starting RNG seed")
    parser.add_argument("--db", type=str, default=":memory:", help="DB path (:memory: or file)")
    args = parser.parse_args()

    game_cfg, ai_cfg, _ = load_config()

    if args.db == ":memory:":
        import tempfile
        db_path = Path(tempfile.mkdtemp()) / "profile_run.db"
    else:
        db_path = Path(args.db)

    db = Database(db_path)
    db.connect()
    run_migrations(db)

    model = BehaviorModel(db, ai_cfg, game_cfg)
    model.load_profile()

    start = time.perf_counter()
    print(f"Running {args.matches} match(es) with profile tracking...")

    for i in range(args.matches):
        seed = args.seed + i
        match_id = f"profile-run-{i:04d}"
        result = run_match_with_profile(model, match_id, game_cfg, seed)
        print_profile_summary(model, i + 1, result)

    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.2f}s")
    db.close()


if __name__ == "__main__":
    main()
