"""Game engine: main loop, tick orchestration, match lifecycle.

Owns the SimulationState. Executes the fixed-timestep loop with
phases: INPUT → SIMULATE → AI_OBSERVE → LOG → RENDER.

The engine delegates all logic to subsystems. It orchestrates but
owns no gameplay logic itself.

Phase 13 additions:
  - _run_title_screen()  — pre-match help/controls screen shown at startup
  - _show_help flag      — pause simulation and show controls overlay (H key)
  - hit-flash counters   — brief white flash when a fighter is hit

Phase 15 additions:
  - Hitstop              — freeze simulation N frames on hit (purely display)
  - Screen shake         — brief camera shake on hit
  - Impact VFX           — particle spawn notifications to renderer
  - Sound hooks          — NullSoundManager wired at all combat events
  - Gravity / landing    — apply_gravity + handle_landing in simulate loop

Phase 16 additions:
  - Combo counter        — _player_combo / _ai_combo streak tracking; reset on
                           whiff or opponent HITSTUN recovery; passed to renderer
  - Landing dust         — "land" VFX appended to _pending_hit_vfx on touch-down
  - Exponential shake    — _shake_max_frames tracked; _compute_shake_offset uses
                           power-curve decay (** 1.5) instead of flat intensity
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
from game.combat.projectile import Projectile, ProjectileHitEffect
from game.combat.state_machine import stop_moving
from game.entities.fighter import attempt_commitment
from game.semantic_events import (
    EventRouter,
    build_commitment_event,
    build_hit_event,
    build_simple_event,
)
from game.simulation_step import (
    SimulationContext,
    TickEffects,
    step_simulation,
)
from game.entities.ai_fighter import BaselineAIController
from game.entities.player_fighter import PlayerController
from game.input.input_handler import InputHandler
from game.sound import NullSoundManager
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

# Number of simulation ticks a hit-flash lasts (Phase 17: increased from 5 → 8)
_HIT_FLASH_TICKS = 8

# Phase 15/17: hitstop (display frames, not simulation ticks)
# Phase 17: increased for clearer hit confirmation (light 4→6, heavy 8→12)
_HITSTOP_LIGHT  = 6
_HITSTOP_HEAVY  = 12

# Phase 15/17: screen shake (Phase 17: stronger for hit clarity)
_SHAKE_FRAMES_LIGHT    = 4
_SHAKE_FRAMES_HEAVY    = 7
_SHAKE_INTENSITY_LIGHT = 3   # pixels
_SHAKE_INTENSITY_HEAVY = 6   # pixels

# Phase 16/20: combo flash = persistence timer (~1 second at 60fps display rate).
# First 10 frames: scale-up animation. Last 15 frames: alpha fade-out.
_COMBO_FLASH_FRAMES = 60

# Phase 18: guard break flash duration (display frames)
_GUARD_BREAK_FLASH_FRAMES = 20


class Engine:
    """Top-level game engine. Manages the main loop and match lifecycle."""

    def __init__(self, game_cfg: GameConfig, ai_cfg: AIConfig,
                 display_cfg: DisplayConfig, db: Database,
                 headless: bool = False,
                 ai_tier: AITier = AITier.T2_FULL_ADAPTIVE) -> None:
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

        # Per-match simulation state (hit tracker, live projectiles, previous
        # FSM states, reaction-time bookkeeping). Shared with the replay
        # verifier, batch evaluator and test fixture via step_simulation.
        self._sim_ctx = SimulationContext()

        self._session_id = str(uuid.uuid4())
        self._match_id = ""
        self._running = False
        self._match_end_tick: int | None = None
        self._recorder: ReplayRecorder | None = None

        # Phase 13: UX state
        self._show_help = False
        self._player_hit_flash: int = 0  # ticks remaining for player hit flash
        self._ai_hit_flash: int = 0      # ticks remaining for AI hit flash

        # Phase 15: combat juice state
        self._hitstop_remaining: int = 0   # display frames to freeze on hit
        self._shake_remaining: int = 0     # display frames of screen shake
        self._shake_intensity: int = 0     # current shake magnitude (pixels)
        self._shake_max_frames: int = 0    # Phase 16: initial frame count for exponential decay
        # Pending VFX: (x_sub, y_sub, is_heavy, kind) — drained into renderer each frame
        # kind: "light" | "heavy" | "dodge" | "whiff" | "land"
        self._pending_hit_vfx: list[tuple[int, int, bool, str]] = []
        # Phase 17: attacker whiff flash (ticks remaining)
        self._player_whiff_flash: int = 0
        self._ai_whiff_flash: int = 0

        # Phase 16: combo streak counters (consecutive hits without opponent recovering)
        self._player_combo: int = 0
        self._ai_combo: int = 0
        # Display frames remaining for combo emphasis animation
        self._player_combo_flash: int = 0
        self._ai_combo_flash: int = 0

        # Phase 18: block/guard-break flash counters (display frames)
        self._player_block_flash: int = 0    # player was hit while blocking
        self._ai_block_flash: int = 0        # AI was hit while blocking
        self._player_guard_break_flash: int = 0  # player's guard was broken
        self._ai_guard_break_flash: int = 0      # AI's guard was broken

        # Phase 19: debug hitbox overlay toggle
        self._show_hitboxes: bool = False

        # Phase 19: pending text popups — (text, x_sub, y_sub, is_large)
        # Drained into renderer.spawn_text_popup() each display frame.
        self._pending_popups: list[tuple[str, int, int, bool]] = []

        # Phase 15: sound hooks
        self._sound = NullSoundManager()

        # Behavior modeling layer
        self._behavior_model = BehaviorModel(db, ai_cfg, game_cfg)
        self._behavior_model.load_profile()

        # Prediction engine (ensemble: Markov + sklearn)
        self._prediction_engine = PredictionEngine(
            db, self._behavior_model, ai_cfg, game_cfg)
        # Always attempt to load the active promoted model from the registry
        # (force=True bypasses the match-count gate so the trained model is
        # used from the very first match in live play).
        self._prediction_engine.try_load_sklearn(force=True)

        # Tactical planner (T1/T2 tiers use this instead of baseline AI)
        self._tactical_planner: TacticalPlanner | None = None
        if ai_tier in (AITier.T1_MARKOV_ONLY, AITier.T2_FULL_ADAPTIVE):
            self._tactical_planner = TacticalPlanner(
                db, self._prediction_engine, ai_cfg, game_cfg, ai_tier)

        # Semantic event fan-out. The batch evaluator builds the same router
        # over the same sinks, which is what keeps off-line evaluation and live
        # play driving the prediction ensemble identically.
        self._event_router = EventRouter(
            game_cfg,
            sinks=[
                self._game_logger.record_event,
                self._behavior_model.on_event,
                self._prediction_engine.on_event,
            ],
            planner=self._tactical_planner,
        )

    def run(self) -> None:
        """Initialize and run the main game loop."""
        pygame.init()

        if not self._headless:
            self._renderer = Renderer(self._gcfg, self._dcfg,
                                      ai_tier_name=self._ai_tier.name)
            self._renderer.init()

            # Show title / tier-selection / controls screen before the first match.
            # Returns the chosen tier, or None if the player quit.
            chosen_tier = self._run_title_screen()
            if chosen_tier is None:
                pygame.quit()
                return

            # Apply the tier chosen on the title screen (may differ from default)
            self._apply_tier(chosen_tier)

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

    # ------------------------------------------------------------------
    # Title screen (pre-match)
    # ------------------------------------------------------------------

    def _run_title_screen(self) -> "AITier | None":
        """Show the title / tier-selection / controls screen.

        Returns the selected AITier when the player starts the match,
        or None if they pressed ESC or quit.
        """
        if self._headless or self._renderer is None:
            return self._ai_tier

        _TIERS = (AITier.T2_FULL_ADAPTIVE, AITier.T1_MARKOV_ONLY, AITier.T0_BASELINE)
        idx = 0  # default: T2

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return _TIERS[idx]
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in (pygame.K_LEFT, pygame.K_UP):
                        idx = (idx - 1) % len(_TIERS)
                    if event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                        idx = (idx + 1) % len(_TIERS)

            self._renderer.render_title(selected_tier=_TIERS[idx].name)
            clock.tick(30)

    def _apply_tier(self, tier: AITier) -> None:
        """Switch the engine to the given AI tier.

        Called once after title-screen tier selection, before _start_match().
        Rebuilds the tactical planner if the tier has changed.
        """
        if tier == self._ai_tier:
            return

        self._ai_tier = tier

        # Update renderer badge
        if self._renderer is not None:
            self._renderer._ai_tier_name = tier.name

        # Rebuild tactical planner for the new tier
        if tier in (AITier.T1_MARKOV_ONLY, AITier.T2_FULL_ADAPTIVE):
            self._tactical_planner = TacticalPlanner(
                self._db, self._prediction_engine, self._ai_cfg,
                self._gcfg, tier,
            )
        else:
            # T0: no planner
            self._tactical_planner = None

    # ------------------------------------------------------------------
    # Frame + tick loops
    # ------------------------------------------------------------------

    def _run_frame(self) -> None:
        """Run one frame: process clock, tick simulation, render."""
        ticks = self._clock.update()

        # Phase 15: hitstop — freeze simulation for N display frames
        if self._hitstop_remaining > 0:
            self._hitstop_remaining -= 1
            ticks = 0  # skip simulation this frame

        for _ in range(ticks):
            if not self._running:
                break
            if self._show_help:
                # Simulation paused while help overlay is open
                self._handle_help_input()
            else:
                self._run_tick()

        # Drain pending VFX notifications into renderer
        if self._renderer and self._pending_hit_vfx:
            for entry in self._pending_hit_vfx:
                x_sub, y_sub, is_heavy, kind = entry
                self._renderer.spawn_hit_particles(x_sub, y_sub, is_heavy, kind=kind)
            self._pending_hit_vfx.clear()

        # Phase 19: drain pending text popups
        if self._renderer and self._pending_popups:
            for text, x_sub, y_sub, is_large in self._pending_popups:
                self._renderer.spawn_text_popup(text, x_sub, y_sub, is_large)
            self._pending_popups.clear()

        # Render at display rate (passes flash counters and help flag)
        if self._renderer and self._running:
            sx, sy = self._compute_shake_offset()
            self._state.set_phase(TickPhase.RENDER)
            self._renderer.render(
                self._state,
                show_help=self._show_help,
                player_flash=self._player_hit_flash,
                ai_flash=self._ai_hit_flash,
                player_whiff=self._player_whiff_flash,
                ai_whiff=self._ai_whiff_flash,
                player_dodge_cd=self._state.player.dodge_cooldown,
                ai_dodge_cd=self._state.ai.dodge_cooldown,
                player_heavy_cd=self._state.player.heavy_cooldown,
                ai_heavy_cd=self._state.ai.heavy_cooldown,
                shake_x=sx,
                shake_y=sy,
                player_combo=self._player_combo,
                ai_combo=self._ai_combo,
                player_combo_flash=self._player_combo_flash,
                ai_combo_flash=self._ai_combo_flash,
                player_guard=self._state.player.guard,
                ai_guard=self._state.ai.guard,
                player_block_flash=self._player_block_flash,
                ai_block_flash=self._ai_block_flash,
                player_guard_break_flash=self._player_guard_break_flash,
                ai_guard_break_flash=self._ai_guard_break_flash,
                show_hitboxes=self._show_hitboxes,
                projectiles=self._sim_ctx.projectiles,
            )

        # Decay counters at display rate
        if self._player_hit_flash > 0:
            self._player_hit_flash -= 1
        if self._ai_hit_flash > 0:
            self._ai_hit_flash -= 1
        if self._shake_remaining > 0:
            self._shake_remaining -= 1
        if self._player_combo_flash > 0:
            self._player_combo_flash -= 1
        if self._ai_combo_flash > 0:
            self._ai_combo_flash -= 1
        if self._player_block_flash > 0:
            self._player_block_flash -= 1
        if self._ai_block_flash > 0:
            self._ai_block_flash -= 1
        if self._player_guard_break_flash > 0:
            self._player_guard_break_flash -= 1
        if self._ai_guard_break_flash > 0:
            self._ai_guard_break_flash -= 1

        # Cap frame rate
        pygame.time.Clock().tick(self._dcfg.window.fps_cap)

    def _compute_shake_offset(self) -> tuple[int, int]:
        """Return (dx, dy) screen shake offset for this frame.

        Phase 16: exponential decay — intensity falls off as shake_remaining
        approaches zero, using a power curve (** 1.5) for a punchy start that
        smoothly fades rather than cutting off abruptly.
        """
        if self._shake_remaining <= 0:
            return 0, 0
        # Use max(remaining, max_frames) so frac ≤ 1 even if max_frames was never
        # set (e.g. direct attribute mutation in tests).
        max_f = max(1, self._shake_max_frames, self._shake_remaining)
        frac = (self._shake_remaining / max_f) ** 1.5
        intensity = max(1, round(self._shake_intensity * frac))
        phase = self._shake_remaining % 2
        return (intensity if phase else -intensity), (1 if phase else -1)

    def _handle_help_input(self) -> None:
        """Poll events while the help overlay is shown (simulation paused)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                self._show_help = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_h, pygame.K_ESCAPE):
                    # ESC while help open → close help (don't quit)
                    self._show_help = False

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

        # H key toggles help overlay (pauses simulation next frame)
        if self._input_handler.toggle_help_requested:
            self._show_help = not self._show_help

        # F1 key toggles hitbox debug overlay
        if self._input_handler.toggle_hitbox_requested:
            self._show_hitboxes = not self._show_hitboxes

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
        """Advance one tick via the shared step, then drain its effects.

        All simulation lives in `game.simulation_step`; everything here is
        policy (who decides what), presentation, replay recording, and event
        emission.
        """
        state = self._state
        gcfg = self._gcfg

        def decide_player() -> list[CombatCommitment]:
            entered = self._player_ctrl.process_inputs(
                state.player, inputs, gcfg)
            return [entered] if entered is not None else []

        def decide_ai() -> list[CombatCommitment]:
            entered: list[CombatCommitment] = []
            if self._tactical_planner is not None:
                c = self._tactical_planner.decide(state.ai, state, gcfg)
                # Extras (e.g. a guard release) were entered before `c`.
                entered.extend(self._tactical_planner.take_extra_commitments())
            else:
                c = self._ai_ctrl.decide(state.ai, state, gcfg)
            if c is not None:
                entered.append(c)

            # Phase 20: minimal AI shoot policy — an uncharged instant shot
            # roughly every 3 seconds. tick % 180 == 90 keeps matches shorter
            # than 90 ticks unaffected, which several tests rely on.
            # MOVING is a free state, so this can legitimately fire on the same
            # tick the planner entered a movement commitment and override it.
            if (state.tick_id % 180 == 90
                    and state.ai.is_free
                    and state.ai.shoot_cooldown == 0):
                if attempt_commitment(
                        state.ai, CombatCommitment.SHOOT_INSTANT, gcfg):
                    entered.append(CombatCommitment.SHOOT_INSTANT)
            return entered

        fx = step_simulation(state, gcfg, self._sim_ctx,
                             decide_player, decide_ai)
        self._drain_effects(fx)

    def _drain_effects(self, fx: TickEffects) -> None:
        """Translate a TickEffects into recording, events, and presentation."""
        state = self._state

        # --- Semantic event stream (shared with the batch evaluator) ---
        self._event_router.route(state, self._match_id, self._sim_ctx, fx)

        # --- Commitments: replay stream + sound ---
        for actor, commitments in (
            (Actor.PLAYER, fx.player_commitments),
            (Actor.AI, fx.ai_commitments),
        ):
            for commitment in commitments:
                if self._recorder:
                    self._recorder.record_commitment(
                        state.tick_id, actor, commitment)
                if commitment == CombatCommitment.JUMP:
                    self._sound.play_jump()
                elif commitment == CombatCommitment.DODGE_BACKWARD:
                    self._sound.play_dodge_start()

        # --- Muzzle flash for shots released this tick ---
        for proj in fx.projectiles_fired:
            self._pending_hit_vfx.append((proj.x, proj.y, False, "muzzle_flash"))

        # --- Landing ---
        if fx.player_landed:
            self._sound.play_land()
            self._pending_hit_vfx.append(
                (state.player.x, state.player.y, False, "land"))
        if fx.ai_landed:
            self._sound.play_land()
            self._pending_hit_vfx.append(
                (state.ai.x, state.ai.y, False, "land"))

        # --- Blocked hits ---
        if fx.ai_blocked:
            self._ai_block_flash = _HIT_FLASH_TICKS
            self._player_combo = 0  # blocked hit doesn't continue a combo
            if fx.ai_guard_broken:
                self._ai_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                self._sound.play_guard_break()
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, True, "guard_break"))
                self._pending_popups.append(
                    ("GUARD BREAK!", state.ai.x, state.ai.y, True))
            else:
                self._sound.play_block()
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, False, "block"))

        if fx.player_blocked:
            self._player_block_flash = _HIT_FLASH_TICKS
            self._ai_combo = 0
            if fx.player_guard_broken:
                self._player_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                self._sound.play_guard_break()
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, True, "guard_break"))
                self._pending_popups.append(
                    ("GUARD BREAK!", state.player.x, state.player.y, True))
            else:
                self._sound.play_block()
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, False, "block"))

        # --- Landed melee hits ---
        if fx.player_hit:
            self._ai_hit_flash = _HIT_FLASH_TICKS
            self._apply_hit_juice(fx.player_hit, state.ai.x, state.ai.y)
            self._player_combo += 1
            self._player_combo_flash = _COMBO_FLASH_FRAMES
            if self._player_combo >= 2:
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, False, "combo_ring"))
            if self._player_combo % 5 == 0:
                self._pending_popups.append(
                    (f"{self._player_combo} HIT!", state.ai.x, state.ai.y, True))

        if fx.ai_hit:
            self._player_hit_flash = _HIT_FLASH_TICKS
            self._apply_hit_juice(fx.ai_hit, state.player.x, state.player.y)
            self._ai_combo += 1
            self._ai_combo_flash = _COMBO_FLASH_FRAMES
            if self._ai_combo >= 2:
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, False, "combo_ring"))
            if self._ai_combo % 5 == 0:
                self._pending_popups.append(
                    (f"{self._ai_combo} HIT!",
                     state.player.x, state.player.y, True))

        # --- Dodge-avoided ---
        if fx.player_dodge_avoided:
            self._sound.play_dodge_avoid()
            self._pending_hit_vfx.append(
                (state.ai.x, state.ai.y, False, "dodge"))
        if fx.ai_dodge_avoided:
            self._sound.play_dodge_avoid()
            self._pending_hit_vfx.append(
                (state.player.x, state.player.y, False, "dodge"))

        # --- Projectile impacts ---
        for hit in fx.projectile_hits:
            self._drain_projectile_hit(hit)

        # --- Whiffs ---
        if fx.player_whiffed:
            self._sound.play_whiff()
            self._player_whiff_flash = _HIT_FLASH_TICKS
            self._pending_hit_vfx.append(
                (state.player.x, state.player.y, False, "whiff"))
            self._player_combo = 0
            self._pending_popups.append(
                ("MISS", state.player.x, state.player.y, False))
        if fx.ai_whiffed:
            self._sound.play_whiff()
            self._ai_whiff_flash = _HIT_FLASH_TICKS
            self._pending_hit_vfx.append(
                (state.ai.x, state.ai.y, False, "whiff"))
            self._ai_combo = 0
            self._pending_popups.append(
                ("MISS", state.ai.x, state.ai.y, False))

        # Decay whiff flash
        if self._player_whiff_flash > 0:
            self._player_whiff_flash -= 1
        if self._ai_whiff_flash > 0:
            self._ai_whiff_flash -= 1

        # --- Streaks broken by the opponent escaping hitstun ---
        if fx.player_combo_reset:
            self._player_combo = 0
        if fx.ai_combo_reset:
            self._ai_combo = 0

        # --- KO ---
        if fx.ko_winner is not None:
            self._end_match()

    def _drain_projectile_hit(self, hit: ProjectileHitEffect) -> None:
        """Presentation for one projectile impact."""
        target_is_ai = hit.target_actor == Actor.AI
        target = self._state.ai if target_is_ai else self._state.player

        if hit.blocked:
            self._pending_hit_vfx.append(
                (target.x, target.y, False, "block"))
            if hit.guard_broken:
                self._sound.play_guard_break()
                if target_is_ai:
                    self._ai_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                else:
                    self._player_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                self._pending_popups.append(
                    ("GUARD BREAK!", target.x, target.y, True))
            return

        if target_is_ai:
            self._ai_hit_flash = _HIT_FLASH_TICKS
        else:
            self._player_hit_flash = _HIT_FLASH_TICKS

        kind = "heavy" if hit.is_heavy else "light"
        self._pending_hit_vfx.append(
            (hit.x, hit.y, hit.is_heavy, "projectile_hit"))
        self._pending_hit_vfx.append((hit.x, hit.y, hit.is_heavy, kind))

        # Hitstop is lighter than melee.
        self._hitstop_remaining = max(
            self._hitstop_remaining,
            _HITSTOP_HEAVY if hit.is_heavy else _HITSTOP_LIGHT,
        )

        if hit.is_heavy:
            self._sound.play_hit_heavy()
        else:
            self._sound.play_hit_light()

    def _apply_hit_juice(self, hit, defender_x: int, defender_y: int) -> None:
        """Set hitstop, screen shake, VFX, and sound for a confirmed hit."""
        is_heavy = (hit.attacker_commitment == CombatCommitment.HEAVY_ATTACK)

        # Hitstop (take the maximum in case of simultaneous hits)
        frames = _HITSTOP_HEAVY if is_heavy else _HITSTOP_LIGHT
        self._hitstop_remaining = max(self._hitstop_remaining, frames)

        # Screen shake (Phase 16: track max_frames for exponential decay)
        if is_heavy:
            self._shake_remaining  = _SHAKE_FRAMES_HEAVY
            self._shake_max_frames = _SHAKE_FRAMES_HEAVY
            self._shake_intensity  = _SHAKE_INTENSITY_HEAVY
        else:
            new_frames = max(self._shake_remaining, _SHAKE_FRAMES_LIGHT)
            if new_frames > self._shake_remaining:
                self._shake_max_frames = _SHAKE_FRAMES_LIGHT
            self._shake_remaining  = new_frames
            self._shake_intensity  = max(self._shake_intensity,  _SHAKE_INTENSITY_LIGHT)

        # VFX (sub-pixel coords; renderer converts to screen coords)
        # kind: "heavy" | "light" distinguishes spark colour/count
        kind = "heavy" if is_heavy else "light"
        self._pending_hit_vfx.append((defender_x, defender_y, is_heavy, kind))

        # Sound
        if is_heavy:
            self._sound.play_hit_heavy()
        else:
            self._sound.play_hit_light()

    # --- Phase 20: projectile helpers ---

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
        # Fresh per-match simulation state: hit tracker, projectiles, previous
        # FSM states and reaction-time bookkeeping all reset together.
        self._sim_ctx = SimulationContext()
        self._player_ctrl.reset()
        self._game_logger.clear()
        self._match_end_tick = None

        # Reset all UX/juice state for the new match
        self._player_hit_flash = 0
        self._ai_hit_flash = 0
        self._player_whiff_flash = 0
        self._ai_whiff_flash = 0
        self._show_help = False
        self._hitstop_remaining = 0
        self._shake_remaining = 0
        self._shake_intensity = 0
        self._shake_max_frames = 0
        self._pending_hit_vfx.clear()
        # Phase 16: reset combo counters
        self._player_combo = 0
        self._ai_combo = 0
        self._player_combo_flash = 0
        self._ai_combo_flash = 0
        # Phase 18: reset block/guard-break flash counters
        self._player_block_flash = 0
        self._ai_block_flash = 0
        self._player_guard_break_flash = 0
        self._ai_guard_break_flash = 0
        # Phase 19: clear pending popups
        self._pending_popups.clear()

        self._recorder = ReplayRecorder(self._state, self._gcfg)

        self._db.execute_safe(
            "INSERT INTO matches (match_id, session_id, started_at, rng_seed, config_hash) "
            "VALUES (?, ?, ?, ?, ?);",
            (self._match_id, self._session_id,
             datetime.now(timezone.utc).isoformat(),
             rng_seed,
             config_hash(CONFIG_DIR / "game_config.yaml")),
        )

        self._behavior_model.on_match_start(self._match_id)
        self._prediction_engine.on_match_start(self._match_id)

        if self._tactical_planner is not None:
            self._tactical_planner.on_match_start(
                self._match_id, self._session_id, rng_seed)

        self._emit_simple_event(EventType.MATCH_START, Actor.PLAYER)

        log.info("Match started: %s (seed=%d)", self._match_id[:8], rng_seed)

    def _end_match(self) -> None:
        """Finalize the current match."""
        if self._match_end_tick is not None:
            return  # Already ended

        self._match_end_tick = self._state.tick_id
        self._emit_simple_event(EventType.MATCH_END, Actor.PLAYER)
        self._game_logger.flush_all()

        if self._recorder:
            snapshots = self._game_logger.drain_tick_buffer()
            replay_path = self._recorder.finalize(
                self._state, snapshots, self._match_id)
            if replay_path:
                log.info("Replay saved: %s", replay_path.name)

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

        self._behavior_model.on_match_end(self._state.winner, self._state.tick_id)
        self._prediction_engine.on_match_end()

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
                elif event.key == pygame.K_h:
                    self._show_help = not self._show_help

        if self._renderer:
            self._state.set_phase(TickPhase.RENDER)
            self._renderer.render(
                self._state,
                show_help=self._show_help,
                player_flash=self._player_hit_flash,
                ai_flash=self._ai_hit_flash,
                player_whiff=self._player_whiff_flash,
                ai_whiff=self._ai_whiff_flash,
                player_dodge_cd=self._state.player.dodge_cooldown,
                ai_dodge_cd=self._state.ai.dodge_cooldown,
                player_guard=self._state.player.guard,
                ai_guard=self._state.ai.guard,
                player_block_flash=self._player_block_flash,
                ai_block_flash=self._ai_block_flash,
                player_guard_break_flash=self._player_guard_break_flash,
                ai_guard_break_flash=self._ai_guard_break_flash,
                projectiles=self._sim_ctx.projectiles,
            )
            pygame.time.Clock().tick(30)

    # --- Event emission helpers ---

    def _emit_commitment_event(self, actor: Actor,
                               commitment: CombatCommitment) -> None:
        state = self._state
        ctx = self._sim_ctx
        reaction = ctx.reaction_ticks(actor, state.tick_id)
        if actor == Actor.PLAYER:
            ctx.last_player_commit_tick = state.tick_id
        else:
            ctx.last_ai_commit_tick = state.tick_id

        event = build_commitment_event(
            state, self._gcfg, self._match_id, actor, commitment, reaction)
        self._game_logger.record_event(event)
        self._behavior_model.on_event(event)
        self._prediction_engine.on_event(event)

        if (actor == Actor.PLAYER
                and self._tactical_planner is not None
                and commitment is not None):
            self._tactical_planner.on_player_commit(commitment, state.tick_id)

    def _emit_hit_event(self, attacker_actor: Actor, hit) -> None:
        event = build_hit_event(
            self._state, self._gcfg, self._match_id, attacker_actor, hit)
        self._game_logger.record_event(event)
        self._behavior_model.on_event(event)
        self._prediction_engine.on_event(event)

    def _emit_simple_event(self, event_type: EventType, actor: Actor) -> None:
        event = build_simple_event(
            self._state, self._match_id, event_type, actor
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
