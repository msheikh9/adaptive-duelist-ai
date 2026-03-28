"""Headless engine fixture for testing.

Runs matches without pygame display, feeding scripted inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we can import project modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set SDL to dummy video driver before importing pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config.config_loader import GameConfig, AIConfig, DisplayConfig
from data.db import Database
from data.events import EventType
from data.logger import GameLogger
from data.tick_snapshot import TickSnapshot
from game.arena import classify_spacing
from game.combat.actions import Actor, CombatCommitment, FSMState
from game.combat.collision import HitTracker, check_hit
from game.combat.damage import apply_hit
from game.combat.physics import (
    apply_dodge_velocity,
    apply_velocity,
    clamp_to_arena,
    update_facing,
)
from game.combat.stamina import tick_stamina
from game.combat.state_machine import tick_fsm
from game.entities.ai_fighter import BaselineAIController
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    SimulationState,
    TickPhase,
)


class HeadlessMatch:
    """Run a match without pygame display for testing.

    Manually steps through ticks with optional player input injection.
    """

    def __init__(self, game_cfg: GameConfig, rng_seed: int = 42) -> None:
        self.cfg = game_cfg
        self.rng_seed = rng_seed
        scale = game_cfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            game_cfg.arena.width, game_cfg.arena.height,
            game_cfg.arena.ground_y, scale,
        )

        player_x = arena.width_sub // 3
        ai_x = (arena.width_sub * 2) // 3

        self.state = SimulationState(
            tick_id=0,
            rng_seed=rng_seed,
            player=FighterState(
                x=player_x, y=arena.ground_y_sub,
                hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
                facing=1,
            ),
            ai=FighterState(
                x=ai_x, y=arena.ground_y_sub,
                hp=game_cfg.fighter.max_hp, stamina=game_cfg.fighter.max_stamina,
                facing=-1,
            ),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )

        self.ai_ctrl = BaselineAIController(rng_seed)
        self.hit_tracker = HitTracker()
        self.events: list = []
        self.snapshots: list[TickSnapshot] = []

    def tick(self, player_commitment: CombatCommitment | None = None) -> None:
        """Advance simulation by one tick with optional player action."""
        state = self.state
        cfg = self.cfg
        scale = cfg.simulation.sub_pixel_scale
        fighter_w_sub = cfg.fighter.width * scale

        if state.match_status == MatchStatus.ENDED:
            return

        state.set_phase(TickPhase.SIMULATE)

        # Player commitment
        if player_commitment is not None:
            from game.entities.fighter import attempt_commitment
            attempt_commitment(state.player, player_commitment, cfg)

        # AI decision
        self.ai_ctrl.decide(state.ai, state, cfg)

        # Physics
        apply_dodge_velocity(state.player, cfg)
        apply_dodge_velocity(state.ai, cfg)
        apply_velocity(state.player)
        apply_velocity(state.ai)
        clamp_to_arena(state.player, state.arena, fighter_w_sub)
        clamp_to_arena(state.ai, state.arena, fighter_w_sub)
        update_facing(state.player, state.ai)

        # Collision
        player_hit = check_hit(state.player, state.ai, "player",
                                self.hit_tracker, cfg)
        ai_hit = check_hit(state.ai, state.player, "ai",
                            self.hit_tracker, cfg)

        if player_hit:
            apply_hit(state.ai, player_hit)
            self.events.append(("HIT", Actor.PLAYER, player_hit))
        if ai_hit:
            apply_hit(state.player, ai_hit)
            self.events.append(("HIT", Actor.AI, ai_hit))

        # Stamina
        tick_stamina(state.player, cfg)
        tick_stamina(state.ai, cfg)

        # Reset hit tracker for free fighters
        if state.player.is_free:
            self.hit_tracker.reset("player")
        if state.ai.is_free:
            self.hit_tracker.reset("ai")

        # FSM advance
        tick_fsm(state.player, cfg)
        tick_fsm(state.ai, cfg)

        # Snapshot
        state.set_phase(TickPhase.LOG)
        self.snapshots.append(TickSnapshot.from_state(state))

        # KO check
        if state.player.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "AI"
        elif state.ai.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "PLAYER"

        state.tick_id += 1

    def run_ticks(self, n: int, player_commitment: CombatCommitment | None = None) -> None:
        """Run n ticks, optionally injecting a commitment on the first tick only."""
        for i in range(n):
            self.tick(player_commitment if i == 0 else None)

    def run_until_end(self, max_ticks: int = 10000) -> None:
        """Run until match ends or max_ticks reached."""
        for _ in range(max_ticks):
            if self.state.match_status == MatchStatus.ENDED:
                return
            self.tick()
