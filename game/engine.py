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
from game.combat.collision import HitTracker, check_hit, was_dodge_avoided
from game.combat.damage import apply_hit
from game.combat.physics import (
    apply_dodge_velocity,
    apply_gravity,
    apply_velocity,
    clamp_to_arena,
    handle_landing,
    update_facing,
)
from game.combat.stamina import tick_stamina
from game.combat.guard import apply_block_response, tick_guard
from game.combat.projectile import Projectile
from game.combat.state_machine import (
    enter_landing, tick_fsm, stop_moving, tick_dodge_cooldown, tick_heavy_cooldown,
    tick_shoot_cooldown,
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
        self._hit_tracker = HitTracker()

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

        # Phase 20: live projectiles
        self._projectiles: list[Projectile] = []

        # Phase 15: sound hooks
        self._sound = NullSoundManager()

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
        # Always attempt to load the active promoted model from the registry
        # (force=True bypasses the match-count gate so the trained model is
        # used from the very first match in live play).
        self._prediction_engine.try_load_sklearn(force=True)

        # Tactical planner (T1/T2 tiers use this instead of baseline AI)
        self._tactical_planner: TacticalPlanner | None = None
        if ai_tier in (AITier.T1_MARKOV_ONLY, AITier.T2_FULL_ADAPTIVE):
            self._tactical_planner = TacticalPlanner(
                db, self._prediction_engine, ai_cfg, game_cfg, ai_tier)

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
                projectiles=self._projectiles,
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
            if player_commitment == CombatCommitment.JUMP:
                self._sound.play_jump()
            elif player_commitment == CombatCommitment.DODGE_BACKWARD:
                self._sound.play_dodge_start()

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
            if ai_commitment == CombatCommitment.JUMP:
                self._sound.play_jump()
            elif ai_commitment == CombatCommitment.DODGE_BACKWARD:
                self._sound.play_dodge_start()

        # --- Phase 20: AI minimal shoot trigger ---
        # Fires an uncharged instant shot roughly every 3 seconds.
        # Uses tick % 180 == 90 so existing tests (< 90 ticks) are unaffected.
        if (state.tick_id % 180 == 90
                and state.ai.is_free
                and state.ai.shoot_cooldown == 0):
            from game.entities.fighter import attempt_commitment as _ac
            _ac(state.ai, CombatCommitment.SHOOT_INSTANT, gcfg)

        # --- Phase 20: accumulate charge ticks while in CHARGING state ---
        if state.player.fsm_state == FSMState.CHARGING:
            state.player.charge_ticks = min(
                state.player.charge_ticks + 1,
                gcfg.actions.shoot.max_charge_frames,
            )
        if state.ai.fsm_state == FSMState.CHARGING:
            state.ai.charge_ticks = min(
                state.ai.charge_ticks + 1,
                gcfg.actions.shoot.max_charge_frames,
            )

        # --- Phase 20: fire pending shots ---
        if state.player.pending_shot:
            state.player.pending_shot = False
            self._fire_projectile(state.player, "PLAYER", gcfg)
        if state.ai.pending_shot:
            state.ai.pending_shot = False
            self._fire_projectile(state.ai, "AI", gcfg)

        # --- Phase 17: tick dodge cooldowns (every tick regardless of FSM state) ---
        tick_dodge_cooldown(state.player)
        tick_dodge_cooldown(state.ai)

        # --- Phase 17b: tick heavy attack cooldowns ---
        tick_heavy_cooldown(state.player)
        tick_heavy_cooldown(state.ai)

        # --- Phase 20: tick shoot cooldowns ---
        tick_shoot_cooldown(state.player)
        tick_shoot_cooldown(state.ai)

        # --- Phase 18: tick guard regen ---
        tick_guard(state.player, gcfg)
        tick_guard(state.ai, gcfg)

        # --- Phase 15: apply gravity (before velocity) ---
        apply_gravity(state.player, state.arena, gcfg)
        apply_gravity(state.ai, state.arena, gcfg)

        # --- Apply dodge velocity (must be before general velocity) ---
        apply_dodge_velocity(state.player, gcfg)
        apply_dodge_velocity(state.ai, gcfg)

        # --- Apply velocity (x and y) ---
        apply_velocity(state.player)
        apply_velocity(state.ai)

        # --- Clamp to arena (x-axis + ceiling) ---
        clamp_to_arena(state.player, state.arena, fighter_w_sub)
        clamp_to_arena(state.ai, state.arena, fighter_w_sub)

        # --- Phase 15: handle landing ---
        player_landed = handle_landing(state.player, state.arena)
        ai_landed = handle_landing(state.ai, state.arena)

        if player_landed and state.player.fsm_state == FSMState.AIRBORNE:
            enter_landing(state.player, gcfg.fighter.landing_recovery_frames)
            self._sound.play_land()
            # Phase 16: landing dust
            self._pending_hit_vfx.append((state.player.x, state.player.y, False, "land"))

        if ai_landed and state.ai.fsm_state == FSMState.AIRBORNE:
            enter_landing(state.ai, gcfg.fighter.landing_recovery_frames)
            self._sound.play_land()
            # Phase 16: landing dust
            self._pending_hit_vfx.append((state.ai.x, state.ai.y, False, "land"))

        # --- Update facing ---
        update_facing(state.player, state.ai)

        # --- Collision detection ---
        player_hit = check_hit(
            state.player, state.ai, "player", self._hit_tracker, gcfg
        )
        ai_hit = check_hit(
            state.ai, state.player, "ai", self._hit_tracker, gcfg
        )

        # --- Phase 17: detect dodge-avoided hits (before damage, attacker still active) ---
        player_dodge_avoided = was_dodge_avoided(
            state.player, state.ai, "player", self._hit_tracker, gcfg
        )
        ai_dodge_avoided = was_dodge_avoided(
            state.ai, state.player, "ai", self._hit_tracker, gcfg
        )

        # --- Apply damage + trigger hit flash + juice ---

        # Phase 18: check if defender was blocking — intercept hit before normal damage
        if player_hit and state.ai.fsm_state == FSMState.BLOCKING:
            guard_broken = apply_block_response(state.ai, player_hit, gcfg)
            self._ai_block_flash = _HIT_FLASH_TICKS
            self._player_combo = 0  # blocked hit doesn't continue a combo
            if guard_broken:
                self._ai_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                self._sound.play_guard_break()
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, True, "guard_break"))
                # Phase 19: guard break text popup
                self._pending_popups.append(
                    ("GUARD BREAK!", state.ai.x, state.ai.y, True))
            else:
                self._sound.play_block()
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, False, "block"))
            player_hit = None  # consumed by block

        if ai_hit and state.player.fsm_state == FSMState.BLOCKING:
            guard_broken = apply_block_response(state.player, ai_hit, gcfg)
            self._player_block_flash = _HIT_FLASH_TICKS
            self._ai_combo = 0  # blocked hit doesn't continue a combo
            if guard_broken:
                self._player_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                self._sound.play_guard_break()
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, True, "guard_break"))
                # Phase 19: guard break text popup
                self._pending_popups.append(
                    ("GUARD BREAK!", state.player.x, state.player.y, True))
            else:
                self._sound.play_block()
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, False, "block"))
            ai_hit = None  # consumed by block

        if player_hit:
            apply_hit(state.ai, player_hit)
            self._ai_hit_flash = _HIT_FLASH_TICKS
            self._emit_hit_event(Actor.PLAYER, player_hit)
            self._apply_hit_juice(player_hit, state.ai.x, state.ai.y)
            # Phase 16: increment player combo streak
            self._player_combo += 1
            self._player_combo_flash = _COMBO_FLASH_FRAMES
            # Phase 20: combo ring burst at attacker when streak active
            if self._player_combo >= 2:
                self._pending_hit_vfx.append(
                    (state.player.x, state.player.y, False, "combo_ring"))
            # Phase 19: milestone popup at every 5-hit threshold
            if self._player_combo % 5 == 0:
                self._pending_popups.append(
                    (f"{self._player_combo} HIT!", state.ai.x, state.ai.y, True))

        if ai_hit:
            apply_hit(state.player, ai_hit)
            self._player_hit_flash = _HIT_FLASH_TICKS
            self._emit_hit_event(Actor.AI, ai_hit)
            self._apply_hit_juice(ai_hit, state.player.x, state.player.y)
            # Phase 16: increment AI combo streak
            self._ai_combo += 1
            self._ai_combo_flash = _COMBO_FLASH_FRAMES
            # Phase 20: combo ring burst at attacker when streak active
            if self._ai_combo >= 2:
                self._pending_hit_vfx.append(
                    (state.ai.x, state.ai.y, False, "combo_ring"))
            # Phase 19: milestone popup at every 5-hit threshold
            if self._ai_combo % 5 == 0:
                self._pending_popups.append(
                    (f"{self._ai_combo} HIT!", state.player.x, state.player.y, True))

        # --- Phase 17: dodge-avoided VFX / sound ---
        if player_dodge_avoided:
            self._sound.play_dodge_avoid()
            self._pending_hit_vfx.append((state.ai.x, state.ai.y, False, "dodge"))
        if ai_dodge_avoided:
            self._sound.play_dodge_avoid()
            self._pending_hit_vfx.append((state.player.x, state.player.y, False, "dodge"))

        # --- Stamina ---
        player_exhausted = tick_stamina(state.player, gcfg)
        ai_exhausted = tick_stamina(state.ai, gcfg)
        if player_exhausted:
            self._emit_simple_event(EventType.STAMINA_EXHAUSTED, Actor.PLAYER)
        if ai_exhausted:
            self._emit_simple_event(EventType.STAMINA_EXHAUSTED, Actor.AI)

        # --- Advance FSMs ---
        if state.player.is_free:
            self._hit_tracker.reset("player")
        if state.ai.is_free:
            self._hit_tracker.reset("ai")

        tick_fsm(state.player, gcfg)
        tick_fsm(state.ai, gcfg)

        # --- Phase 20: move projectiles + collision ---
        self._update_projectiles(state, gcfg)

        # --- Phase 17: whiff detection (ATTACK_ACTIVE → ATTACK_RECOVERY without hit) ---
        if (self._prev_player_fsm == FSMState.ATTACK_ACTIVE
                and state.player.fsm_state == FSMState.ATTACK_RECOVERY
                and not self._hit_tracker.has_connected("player")):
            self._sound.play_whiff()
            self._player_whiff_flash = _HIT_FLASH_TICKS
            self._pending_hit_vfx.append(
                (state.player.x, state.player.y, False, "whiff"))
            self._player_combo = 0  # Phase 16: whiff breaks combo
            # Phase 19: "MISS" text popup
            self._pending_popups.append(
                ("MISS", state.player.x, state.player.y, False))

        if (self._prev_ai_fsm == FSMState.ATTACK_ACTIVE
                and state.ai.fsm_state == FSMState.ATTACK_RECOVERY
                and not self._hit_tracker.has_connected("ai")):
            self._sound.play_whiff()
            self._ai_whiff_flash = _HIT_FLASH_TICKS
            self._pending_hit_vfx.append(
                (state.ai.x, state.ai.y, False, "whiff"))
            self._ai_combo = 0  # Phase 16: whiff breaks combo
            # Phase 19: "MISS" text popup
            self._pending_popups.append(
                ("MISS", state.ai.x, state.ai.y, False))

        # Decay whiff flash
        if self._player_whiff_flash > 0:
            self._player_whiff_flash -= 1
        if self._ai_whiff_flash > 0:
            self._ai_whiff_flash -= 1

        # Phase 16: reset combo when opponent recovers from hitstun without being re-hit
        if (self._prev_ai_fsm == FSMState.HITSTUN
                and state.ai.fsm_state in FREE_STATES
                and not player_hit):
            self._player_combo = 0

        if (self._prev_player_fsm == FSMState.HITSTUN
                and state.player.fsm_state in FREE_STATES
                and not ai_hit):
            self._ai_combo = 0

        # --- Detect COMMITMENT_END transitions ---
        if (self._prev_player_fsm not in FREE_STATES
                and state.player.fsm_state in FREE_STATES):
            self._emit_simple_event(EventType.COMMITMENT_END, Actor.PLAYER)
        if (self._prev_ai_fsm not in FREE_STATES
                and state.ai.fsm_state in FREE_STATES):
            self._emit_simple_event(EventType.COMMITMENT_END, Actor.AI)
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

    def _fire_projectile(self, shooter: "FighterState", owner: str,
                         gcfg: GameConfig) -> None:
        """Create a projectile from the shooter's position and start cooldown."""
        shoot_cfg = gcfg.actions.shoot
        charge_frac = min(1.0, shooter.charge_ticks / max(1, shoot_cfg.max_charge_frames))
        damage = round(shoot_cfg.min_damage + charge_frac * (shoot_cfg.max_damage - shoot_cfg.min_damage))
        speed_sub = shoot_cfg.projectile_speed * gcfg.simulation.sub_pixel_scale
        velocity_x = speed_sub * shooter.facing

        proj = Projectile(
            x=shooter.x,
            y=shooter.y,
            velocity_x=velocity_x,
            damage=damage,
            owner=owner,
            charge_frac=charge_frac,
        )
        self._projectiles.append(proj)

        # Start cooldown on the shooter
        shooter.shoot_cooldown = shoot_cfg.cooldown_frames
        shooter.charge_ticks = 0

        # Muzzle flash VFX
        self._pending_hit_vfx.append((shooter.x, shooter.y, False, "muzzle_flash"))

    def _update_projectiles(self, state: "SimulationState",
                             gcfg: GameConfig) -> None:
        """Move all active projectiles and check collisions."""
        if not self._projectiles:
            return

        arena_w = state.arena.width_sub
        fighter_w = gcfg.fighter.width * gcfg.simulation.sub_pixel_scale
        fighter_h = gcfg.fighter.height * gcfg.simulation.sub_pixel_scale

        for proj in self._projectiles:
            if not proj.active:
                continue

            proj.x += proj.velocity_x

            # Deactivate if it leaves the arena
            if proj.x < 0 or proj.x > arena_w:
                proj.active = False
                continue

            # Collision with opponent fighter
            if proj.owner == "PLAYER":
                target = state.ai
                target_name = "AI"
            else:
                target = state.player
                target_name = "PLAYER"

            # Simple AABB: projectile is a 16x16 sub-pixel region (1/6 of fighter width)
            proj_r = fighter_w // 6
            t_left  = target.x - fighter_w // 2
            t_right = target.x + fighter_w // 2
            t_top   = target.y - fighter_h
            t_bot   = target.y

            if (t_left - proj_r <= proj.x <= t_right + proj_r
                    and t_top - proj_r <= proj.y <= t_bot + proj_r):
                self._handle_projectile_hit(proj, target, target_name, state, gcfg)
                proj.active = False

        # Remove spent projectiles
        self._projectiles = [p for p in self._projectiles if p.active]

    def _handle_projectile_hit(self, proj: Projectile, target: "FighterState",
                                target_name: str, state: "SimulationState",
                                gcfg: GameConfig) -> None:
        """Apply projectile hit damage + juice without melee HitTracker."""
        from game.combat.damage import apply_hit as _apply_hit
        from game.combat.state_machine import enter_hitstun as _enter_hitstun

        is_heavy = proj.charge_frac >= 0.5

        if target.fsm_state == FSMState.BLOCKING:
            # Treat as a chip-only block (use guard system)
            from game.combat.guard import apply_block_response as _abr
            # Build a minimal hit-like object
            class _FakeHit:
                damage = proj.damage
                attacker_commitment = CombatCommitment.LIGHT_ATTACK  # treat as light for guard cost

            guard_broken = _abr(target, _FakeHit(), gcfg)
            self._pending_hit_vfx.append((target.x, target.y, False, "block"))
            if guard_broken:
                self._sound.play_guard_break()
                if target_name == "AI":
                    self._ai_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                    self._pending_popups.append(("GUARD BREAK!", target.x, target.y, True))
                else:
                    self._player_guard_break_flash = _GUARD_BREAK_FLASH_FRAMES
                    self._pending_popups.append(("GUARD BREAK!", target.x, target.y, True))
            return

        if target.fsm_state == FSMState.KO:
            return

        # Apply damage + hitstun
        target.hp = max(0, target.hp - proj.damage)
        if target.hp == 0:
            from game.combat.state_machine import enter_ko as _enter_ko
            _enter_ko(target)
        else:
            hitstun = 8 if is_heavy else 5
            _enter_hitstun(target, hitstun)

        # Hit flash
        if target_name == "AI":
            self._ai_hit_flash = _HIT_FLASH_TICKS
        else:
            self._player_hit_flash = _HIT_FLASH_TICKS

        # VFX
        kind = "heavy" if is_heavy else "light"
        self._pending_hit_vfx.append((proj.x, proj.y, is_heavy, "projectile_hit"))
        self._pending_hit_vfx.append((proj.x, proj.y, is_heavy, kind))

        # Hitstop (lighter than melee)
        self._hitstop_remaining = max(self._hitstop_remaining,
                                      _HITSTOP_LIGHT if not is_heavy else _HITSTOP_HEAVY)

        # Sound
        if is_heavy:
            self._sound.play_hit_heavy()
        else:
            self._sound.play_hit_light()

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
        self._hit_tracker.clear()
        self._player_ctrl.reset()
        self._game_logger.clear()
        self._last_player_commit_tick = 0
        self._last_ai_commit_tick = 0
        self._prev_player_fsm = FSMState.IDLE
        self._prev_ai_fsm = FSMState.IDLE
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
        # Phase 20: clear projectiles
        self._projectiles.clear()

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
                projectiles=self._projectiles,
            )
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
