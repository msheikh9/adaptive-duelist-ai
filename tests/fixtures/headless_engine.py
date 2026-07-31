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
from game.entities.ai_fighter import BaselineAIController
from game.entities.fighter import attempt_commitment
from game.simulation_step import (
    SimulationContext,
    TickEffects,
    step_simulation,
)
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
        self.sim_ctx = SimulationContext()
        self.events: list = []
        self.snapshots: list[TickSnapshot] = []
        self.last_effects: TickEffects | None = None

    @property
    def hit_tracker(self):
        """The shared hit tracker. Kept as a property for existing tests."""
        return self.sim_ctx.hit_tracker

    def tick(self, player_commitment: CombatCommitment | None = None) -> None:
        """Advance simulation by one tick with optional player action.

        Runs the same `step_simulation` as the live engine, so this fixture
        exercises gravity, guard regen, cooldown decay and projectiles rather
        than a reduced subset of them.
        """
        state = self.state
        cfg = self.cfg

        if state.match_status == MatchStatus.ENDED:
            return

        state.set_phase(TickPhase.SIMULATE)

        def decide_player() -> list[CombatCommitment]:
            if player_commitment is None:
                return []
            if attempt_commitment(state.player, player_commitment, cfg):
                return [player_commitment]
            return []

        def decide_ai() -> list[CombatCommitment]:
            entered = self.ai_ctrl.decide(state.ai, state, cfg)
            return [entered] if entered is not None else []

        fx = step_simulation(state, cfg, self.sim_ctx,
                             decide_player, decide_ai)
        self.last_effects = fx

        if fx.player_hit:
            self.events.append(("HIT", Actor.PLAYER, fx.player_hit))
        if fx.ai_hit:
            self.events.append(("HIT", Actor.AI, fx.ai_hit))

        # Snapshot
        state.set_phase(TickPhase.LOG)
        self.snapshots.append(TickSnapshot.from_state(state))

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
