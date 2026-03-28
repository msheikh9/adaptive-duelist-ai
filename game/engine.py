"""Game engine: main loop, tick orchestration, match lifecycle.

Owns the SimulationState. Executes the fixed-timestep loop with
phases: INPUT → SIMULATE → AI_OBSERVE → LOG → RENDER.

The engine delegates all logic to subsystems. It orchestrates but
owns no gameplay logic itself.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime, timezone

import pygame

from config.config_loader import GameConfig, AIConfig, DisplayConfig, config_hash, CONFIG_DIR
from data.db import Database
from data.events import EventType, SemanticEvent
from data.logger import GameLogger
from data.tick_snapshot import TickSnapshot
from game.arena import classify_spacing
from game.clock import GameClock
from game.combat.actions import Actor, CombatCommitment, FSMState, FREE_STATES
from game.combat.collision import HitTracker, check_hit
from game.combat.damage import apply_hit
from game.combat.physics import (
    apply_dodge_velocity,
    apply_velocity,
    clamp_to_arena,
    update_facing,
)
from game.combat.stamina import tick_stamina
from game.combat.state_machine import tick_fsm, stop_moving
from game.entities.ai_fighter import BaselineAIController
from game.entities.player_fighter import PlayerController
from game.input.input_handler import InputHandler
from replay.recorder import ReplayRecorder
from ai.layers.behavior_model import BehaviorModel
from ai.layers.prediction_engine import PredictionEngine
from ai.layers.tactical_planner import AITier, TacticalPlanner
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    SimulationState,
    TickPhase,
)
from rendering.renderer import Renderer

log = logging.getLogger(__name__)


class Engine:
    """Top-level game engine. Manages the main loop and match lifecycle."""

    def __init__(self, game_cfg: GameConfig, ai_cfg: AIConfig,
                 display_cfg: DisplayConfig, db: Database,
                 headless: bool = False,
                 ai_tier: AITier = AITier.T0_BASELINE) -> None:
        self._gcfg = game_cfg
        self._ai_cfg = ai_cfg
        self._dcfg = display_cfg
        self._db = db
        self._headless = headless
        self._ai_tier = ai_tier

        self._state = SimulationState()
        self._clock = GameClock(game_cfg.simulation.tick_rate)
        self._input_handler = InputHandler()
        self._player_ctrl = PlayerController()
        self._ai_ctrl: BaselineAIController | None = None
        self._renderer: Renderer | None = None
        self._game_logger = GameLogger(db)
        self._hit_tracker = HitTracker()

        self._session_id = str(uuid.uuid4())
        self._match_id = ""
        self._running = False
        self._match_end_tick: int | None = None
        self._recorder: ReplayRecorder | None = None

        # Track last commitment ticks for reaction time calculation
        self._last_player_commit_tick = 0
        self._last_ai_commit_tick = 0

        # Track previous FSM states to detect COMMITMENT_END transitions
        self._prev_player_fsm: FSMState = FSMState.IDLE
        self._prev_ai_fsm: FSMState = FSMState.IDLE

        # Behavior modeling layer
        self._behavior_model = BehaviorModel(db, ai_cfg, game_cfg)
        self._behavior_model.load_profile()

        # Prediction engine (ensemble: Markov + sklearn)
        self._prediction_engine = PredictionEngine(
            db, self._behavior_model, ai_cfg, game_cfg)
        self._prediction_engine.try_load_sklearn()

        # Tactical planner (T1/T2 tiers use this instead of baseline AI)
        self._tactical_planner: TacticalPlanner | None = None
        if ai_tier in (AITier.T1_MARKOV_ONLY, AITier.T2_FULL_ADAPTIVE):
            self._tactical_planner = TacticalPlanner(
                db, self._prediction_engine, ai_cfg, game_cfg, ai_tier)

    def run(self) -> None:
        """Initialize and run the main game loop."""
        pygame.init()

        if not self._headless:
            self._renderer = Renderer(self._gcfg, self._dcfg)
            self._renderer.init()

        self._start_match()
        self._clock.start()
        self._running = True

        try:
            while self._running:
                if self._headless:
                    self._run_headless_tick()
                else:
                    self._run_frame()
        finally:
            pygame.quit()

    def _run_frame(self) -> None:
        """Run one frame: process clock, tick simulation, render."""
        ticks = self._clock.update()

        for _ in range(ticks):
            if not self._running:
                break
            self._run_tick()

        # Render at display rate
        if self._renderer and self._running:
            self._state.set_phase(TickPhase.RENDER)
            self._renderer.render(self._state)

        # Cap frame rate
        pygame.time.Clock().tick(self._dcfg.window.fps_cap)

    def _run_headless_tick(self) -> None:
        """Run a single tick without rendering. For testing."""
        self._run_tick()

    def _run_tick(self) -> None:
        """Execute one full simulation tick through all phases."""
        state = self._state

        if state.match_status == MatchStatus.ENDED:
            self._handle_post_match_input()
            return

        # === PHASE 1: INPUT ===
        state.set_phase(TickPhase.INPUT)
        inputs = self._input_handler.poll()
        if self._input_handler.quit_requested:
            self._running = False
            self._end_match()
            return

        # === PHASE 2: SIMULATE ===
        state.set_phase(TickPhase.SIMULATE)
        self._simulate(inputs)

        # === PHASE 3: AI_OBSERVE ===
        state.set_phase(TickPhase.AI_OBSERVE)
        self._prediction_engine.on_tick(state.tick_id)

        # === PHASE 4: LOG ===
        state.set_phase(TickPhase.LOG)
        snapshot = TickSnapshot.from_state(state)
        self._game_logger.record_tick(snapshot)
        if self._recorder:
            self._recorder.record_checksum_if_due(state)

        state.tick_id += 1

    def _simulate(self, inputs: list) -> None:
        """Run all simulation subsystems for one tick."""
        state = self._state
        gcfg = self._gcfg
        scale = gcfg.simulation.sub_pixel_scale
        fighter_w_sub = gcfg.fighter.width * scale

        # --- Process player input ---
        player_commitment = self._player_ctrl.process_inputs(
            state.player, inputs, gcfg
        )
        if player_commitment is not None:
            self._emit_commitment_event(Actor.PLAYER, player_commitment)
            if self._recorder:
                self._recorder.record_commitment(
                    state.tick_id, Actor.PLAYER, player_commitment)

        # --- AI decision ---
        if self._tactical_planner is not None:
            ai_commitment = self._tactical_planner.decide(state.ai, state, gcfg)
        else:
            ai_commitment = self._ai_ctrl.decide(state.ai, state, gcfg)
        if ai_commitment is not None:
            self._emit_commitment_event(Actor.AI, ai_commitment)
            if self._recorder:
                self._recorder.record_commitment(
                    state.tick_id, Actor.AI, ai_commitment)

        # --- Apply dodge velocity (must be before general velocity) ---
        apply_dodge_velocity(state.player, gcfg)
        apply_dodge_velocity(state.ai, gcfg)

        # --- Apply velocity ---
        apply_velocity(state.player)
        apply_velocity(state.ai)

        # --- Clamp to arena ---
        clamp_to_arena(state.player, state.arena, fighter_w_sub)
        clamp_to_arena(state.ai, state.arena, fighter_w_sub)

        # --- Update facing ---
        update_facing(state.player, state.ai)

        # --- Collision detection ---
        player_hit = check_hit(
            state.player, state.ai, "player", self._hit_tracker, gcfg
        )
        ai_hit = check_hit(
            state.ai, state.player, "ai", self._hit_tracker, gcfg
        )

        # --- Apply damage ---
        if player_hit:
            apply_hit(state.ai, player_hit)
            self._emit_hit_event(Actor.PLAYER, player_hit)
        if ai_hit:
            apply_hit(state.player, ai_hit)
            self._emit_hit_event(Actor.AI, ai_hit)

        # --- Stamina ---
        player_exhausted = tick_stamina(state.player, gcfg)
        ai_exhausted = tick_stamina(state.ai, gcfg)
        if player_exhausted:
            self._emit_simple_event(EventType.STAMINA_EXHAUSTED, Actor.PLAYER)
        if ai_exhausted:
            self._emit_simple_event(EventType.STAMINA_EXHAUSTED, Actor.AI)

        # --- Advance FSMs ---
        # Reset hit tracker when fighters return to free state
        if state.player.is_free:
            self._hit_tracker.reset("player")
        if state.ai.is_free:
            self._hit_tracker.reset("ai")

        tick_fsm(state.player, gcfg)
        tick_fsm(state.ai, gcfg)

        # --- Detect COMMITMENT_END transitions ---
        if (self._prev_player_fsm not in FREE_STATES
                and state.player.fsm_state in FREE_STATES):
            self._emit_simple_event(EventType.COMMITMENT_END, Actor.PLAYER)
        if (self._prev_ai_fsm not in FREE_STATES
                and state.ai.fsm_state in FREE_STATES):
            self._emit_simple_event(EventType.COMMITMENT_END, Actor.AI)
            # Notify planner of AI commitment completion for outcome tracking
            if self._tactical_planner is not None:
                self._tactical_planner.on_ai_commit_end(
                    state.tick_id, state.ai.hp, state.player.hp)
        self._prev_player_fsm = state.player.fsm_state
        self._prev_ai_fsm = state.ai.fsm_state

        # --- Check KO ---
        if state.player.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "AI"
            self._end_match()
        elif state.ai.fsm_state == FSMState.KO:
            state.match_status = MatchStatus.ENDED
            state.winner = "PLAYER"
            self._end_match()

    def _start_match(self) -> None:
        """Initialize a new match."""
        gcfg = self._gcfg
        scale = gcfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            gcfg.arena.width, gcfg.arena.height,
            gcfg.arena.ground_y, scale,
        )

        rng_seed = random.randint(0, 2**32 - 1)
        self._match_id = str(uuid.uuid4())

        # Position fighters at 1/3 and 2/3 of arena width
        player_x = arena.width_sub // 3
        ai_x = (arena.width_sub * 2) // 3

        self._state = SimulationState(
            tick_id=0,
            rng_seed=rng_seed,
            player=FighterState(
                x=player_x, y=arena.ground_y_sub,
                hp=gcfg.fighter.max_hp, stamina=gcfg.fighter.max_stamina,
                facing=1,
            ),
            ai=FighterState(
                x=ai_x, y=arena.ground_y_sub,
                hp=gcfg.fighter.max_hp, stamina=gcfg.fighter.max_stamina,
                facing=-1,
            ),
            arena=arena,
            match_status=MatchStatus.ACTIVE,
        )

        self._ai_ctrl = BaselineAIController(rng_seed)
        self._hit_tracker.clear()
        self._player_ctrl.reset()
        self._game_logger.clear()
        self._last_player_commit_tick = 0
        self._last_ai_commit_tick = 0
        self._prev_player_fsm = FSMState.IDLE
        self._prev_ai_fsm = FSMState.IDLE
        self._match_end_tick = None

        # Initialize replay recorder
        self._recorder = ReplayRecorder(self._state, self._gcfg)

        # Record match in DB
        self._db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
            "VALUES (?, ?, ?, ?, ?);",
            (self._match_id, self._session_id,
             datetime.now(timezone.utc).isoformat(),
             rng_seed,
             config_hash(CONFIG_DIR / "game_config.yaml")),
        )

        # Behavior model: reset per-match state
        self._behavior_model.on_match_start(self._match_id)

        # Prediction engine: reset per-match state
        self._prediction_engine.on_match_start(self._match_id)

        # Tactical planner: reset per-match state
        if self._tactical_planner is not None:
            self._tactical_planner.on_match_start(
                self._match_id, self._session_id, rng_seed)

        # Emit match start event
        self._emit_simple_event(EventType.MATCH_START, Actor.PLAYER)

        log.info("Match started: %s (seed=%d)", self._match_id[:8], rng_seed)

    def _end_match(self) -> None:
        """Finalize the current match."""
        if self._match_end_tick is not None:
            return  # Already ended

        self._match_end_tick = self._state.tick_id
        self._emit_simple_event(EventType.MATCH_END, Actor.PLAYER)
        self._game_logger.flush_all()

        # Save replay file
        if self._recorder:
            snapshots = self._game_logger.drain_tick_buffer()
            replay_path = self._recorder.finalize(
                self._state, snapshots, self._match_id)
            if replay_path:
                log.info("Replay saved: %s", replay_path.name)

        # Update match record
        self._db.execute_safe(
            "UPDATE matches SET ended_at=?, total_ticks=?, winner=?, "
            "player_hp_final=?, ai_hp_final=? WHERE match_id=?;",
            (datetime.now(timezone.utc).isoformat(),
             self._state.tick_id,
             self._state.winner,
             self._state.player.hp,
             self._state.ai.hp,
             self._match_id),
        )

        # Finalize behavior model: update metrics + persist profile
        self._behavior_model.on_match_end(self._state.winner, self._state.tick_id)

        # Finalize prediction engine: check if sklearn should activate
        self._prediction_engine.on_match_end()

        # Finalize tactical planner
        if self._tactical_planner is not None:
            self._tactical_planner.on_match_end()

        log.info("Match ended: %s — Winner: %s (tick %d)",
                 self._match_id[:8], self._state.winner, self._state.tick_id)

    def _handle_post_match_input(self) -> None:
        """Handle input while match is ended (restart or quit)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_r:
                    self._start_match()
                    self._clock.start()

        if self._renderer:
            self._state.set_phase(TickPhase.RENDER)
            self._renderer.render(self._state)
            pygame.time.Clock().tick(30)

    # --- Event emission helpers ---

    def _emit_commitment_event(self, actor: Actor,
                               commitment: CombatCommitment) -> None:
        state = self._state
        if actor == Actor.PLAYER:
            fighter = state.player
            opponent = state.ai
            self._last_player_commit_tick = state.tick_id
            reaction = state.tick_id - self._last_ai_commit_tick
        else:
            fighter = state.ai
            opponent = state.player
            self._last_ai_commit_tick = state.tick_id
            reaction = state.tick_id - self._last_player_commit_tick

        spacing = classify_spacing(
            state.distance_sub(),
            self._gcfg.spacing.close_max,
            self._gcfg.spacing.mid_max,
            self._gcfg.simulation.sub_pixel_scale,
        )

        event = SemanticEvent(
            event_type=EventType.COMMITMENT_START,
            match_id=self._match_id,
            tick_id=state.tick_id,
            actor=actor,
            commitment=commitment,
            opponent_fsm_state=opponent.fsm_state,
            opponent_commitment=opponent.active_commitment,
            spacing_zone=spacing,
            actor_hp=fighter.hp,
            opponent_hp=opponent.hp,
            actor_stamina=fighter.stamina,
            opponent_stamina=opponent.stamina,
            reaction_ticks=reaction,
        )
        self._game_logger.record_event(event)
        self._behavior_model.on_event(event)
        self._prediction_engine.on_event(event)

        # Feed player commits to planner for prediction accuracy tracking
        if (actor == Actor.PLAYER
                and self._tactical_planner is not None
                and commitment is not None):
            self._tactical_planner.on_player_commit(commitment, state.tick_id)

    def _emit_hit_event(self, attacker_actor: Actor, hit) -> None:
        state = self._state
        if attacker_actor == Actor.PLAYER:
            attacker = state.player
            defender = state.ai
        else:
            attacker = state.ai
            defender = state.player

        spacing = classify_spacing(
            state.distance_sub(),
            self._gcfg.spacing.close_max,
            self._gcfg.spacing.mid_max,
            self._gcfg.simulation.sub_pixel_scale,
        )

        event = SemanticEvent(
            event_type=EventType.HIT_LANDED,
            match_id=self._match_id,
            tick_id=state.tick_id,
            actor=attacker_actor,
            commitment=hit.attacker_commitment,
            opponent_fsm_state=defender.fsm_state,
            spacing_zone=spacing,
            actor_hp=attacker.hp,
            opponent_hp=defender.hp,
            actor_stamina=attacker.stamina,
            opponent_stamina=defender.stamina,
            damage_dealt=hit.damage,
        )
        self._game_logger.record_event(event)
        self._behavior_model.on_event(event)
        self._prediction_engine.on_event(event)

    def _emit_simple_event(self, event_type: EventType, actor: Actor) -> None:
        state = self._state
        if actor == Actor.PLAYER:
            fighter = state.player
            opponent = state.ai
        else:
            fighter = state.ai
            opponent = state.player

        event = SemanticEvent(
            event_type=event_type,
            match_id=self._match_id,
            tick_id=state.tick_id,
            actor=actor,
            actor_hp=fighter.hp,
            opponent_hp=opponent.hp,
            actor_stamina=fighter.stamina,
            opponent_stamina=opponent.stamina,
        )
        self._game_logger.record_event(event)
        self._behavior_model.on_event(event)
        self._prediction_engine.on_event(event)

    # --- Public API for headless testing ---

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def match_id(self) -> str:
        return self._match_id

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
