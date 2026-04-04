"""Configuration loading with strict schema validation and defaults."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


CONFIG_DIR = Path(__file__).parent


# --- Dataclasses for typed access ---


@dataclass(frozen=True)
class ArenaConfig:
    width: int = 1200
    height: int = 400
    ground_y: int = 300


@dataclass(frozen=True)
class FighterConfig:
    width: int = 60
    height: int = 100
    max_hp: int = 200
    max_stamina: int = 100
    stamina_regen_idle: float = 3.0
    stamina_regen_moving: float = 1.5
    stamina_regen_attacking: float = 0.0
    exhaustion_recovery_frames: int = 30
    move_speed: int = 5
    # Phase 15: jump / verticality
    jump_velocity: int = 15        # pixels/tick upward launch speed
    jump_gravity: int = 1          # pixels/tick² downward acceleration
    jump_startup_frames: int = 3   # frames before leaving ground
    landing_recovery_frames: int = 5  # frames on landing before IDLE


@dataclass(frozen=True)
class SpacingConfig:
    close_max: int = 150
    mid_max: int = 350


@dataclass(frozen=True)
class AttackActionConfig:
    startup_frames: int
    active_frames: int
    recovery_frames: int
    stamina_cost: int
    damage: int
    reach: int
    hitstun_frames: int
    knockback: int
    # Inter-attack cooldown: ticks from commitment start until next use is legal.
    # 0 = no cooldown (default for light attack). Set per-attack in game_config.yaml.
    cooldown_ticks: int = 0

    @property
    def total_frames(self) -> int:
        return self.startup_frames + self.active_frames + self.recovery_frames


@dataclass(frozen=True)
class DodgeActionConfig:
    startup_frames: int = 3
    active_frames: int = 5
    recovery_frames: int = 12
    stamina_cost: int = 25
    distance: int = 130
    # Phase 17: inter-dodge cooldown (ticks from dodge start until next dodge is legal)
    cooldown_frames: int = 45

    @property
    def total_frames(self) -> int:
        return self.startup_frames + self.active_frames + self.recovery_frames


@dataclass(frozen=True)
class BlockActionConfig:
    startup_frames: int = 3
    stamina_cost_per_tick: int = 2
    chip_damage_pct: float = 0.30
    parry_window_frames: int = 3
    blockstun_frames: int = 6
    parry_stun_frames: int = 12
    # Phase 18: guard meter
    guard_max: int = 100
    guard_cost_light: int = 25
    guard_cost_heavy: int = 60
    guard_regen_per_tick: int = 2
    guard_regen_delay_ticks: int = 60
    guard_break_stun_frames: int = 45


@dataclass(frozen=True)
class MoveActionConfig:
    stamina_cost_per_tick: float = 0.5


@dataclass(frozen=True)
class ActionsConfig:
    light_attack: AttackActionConfig = field(
        default_factory=lambda: AttackActionConfig(
            startup_frames=4, active_frames=3, recovery_frames=8,
            stamina_cost=15, damage=8, reach=80, hitstun_frames=6, knockback=30,
        )
    )
    heavy_attack: AttackActionConfig = field(
        default_factory=lambda: AttackActionConfig(
            startup_frames=12, active_frames=4, recovery_frames=18,
            stamina_cost=30, damage=22, reach=120, hitstun_frames=12, knockback=80,
        )
    )
    dodge_backward: DodgeActionConfig = field(default_factory=DodgeActionConfig)
    block: BlockActionConfig = field(default_factory=BlockActionConfig)
    move: MoveActionConfig = field(default_factory=MoveActionConfig)


@dataclass(frozen=True)
class SimulationConfig:
    tick_rate: int = 60
    sub_pixel_scale: int = 100


@dataclass(frozen=True)
class GameConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    fighter: FighterConfig = field(default_factory=FighterConfig)
    spacing: SpacingConfig = field(default_factory=SpacingConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# --- AI Config ---


@dataclass(frozen=True)
class PredictionConfig:
    window_ticks: int = 60
    window_extend_step: int = 15
    window_max_ticks: int = 90
    hold_extend_threshold: float = 0.40
    markov_order: int = 3
    min_confidence_for_display: float = 0.25
    inference_reeval_idle_ticks: int = 30


@dataclass(frozen=True)
class EnsembleConfig:
    initial_markov_weight: float = 1.0
    initial_sklearn_weight: float = 0.0
    sklearn_activation_matches: int = 3
    weight_update_ema_alpha: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    min_matches_to_train: int = 3
    min_samples_to_train: int = 300
    retrain_after_every_n_matches: int = 1
    random_forest_n_estimators: int = 100
    random_forest_max_depth: int = 8
    random_forest_min_samples_leaf: int = 10
    holdout_fraction: float = 0.15


@dataclass(frozen=True)
class ScoringWeights:
    prediction_alignment: float = 1.0
    hp_differential: float = 0.6
    stamina_suitability: float = 0.5
    historical_success: float = 0.8
    exploration_bonus: float = 0.7
    risk_penalty: float = 0.4
    staleness_penalty: float = 0.6
    consecutive_penalty: float = 0.5
    exploration_pressure: float = 0.9
    accuracy_trend: float = 0.4
    shift_response: float = 0.7
    mode_success_rate: float = 0.6
    # Phase 11: session-level adaptation factors
    session_success_rate: float = 0.25
    archetype_alignment: float = 0.25


@dataclass(frozen=True)
class StrategyConfig:
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    softmax_temperature: float = 0.5


@dataclass(frozen=True)
class PlannerMemoryConfig:
    recent_modes_capacity: int = 30
    mode_outcome_capacity: int = 50
    recent_predictions_capacity: int = 30
    exploit_staleness_threshold_ticks: int = 300
    exploration_budget_drain_rate: float = 0.03
    exploration_budget_regen_rate: float = 0.15
    exploration_budget_floor: float = 0.10
    max_consecutive_before_penalty: int = 5
    shift_detection_confidence_drop: float = 0.20
    shift_probe_duration_decisions: int = 10


@dataclass(frozen=True)
class ActionResolverConfig:
    max_commit_delay_ticks: int = 30


@dataclass(frozen=True)
class ReactionConfig:
    min_reaction_ticks: int = 6


@dataclass(frozen=True)
class ProfileConfig:
    recency_weight_multiplier: float = 3.0
    rolling_window_size: int = 100


@dataclass(frozen=True)
class SessionAdaptationConfig:
    """Configuration for cross-match in-session behavioral adaptation."""
    decay_factor: float = 0.8             # per-match recency decay on session memory
    min_matches_for_archetype: int = 2    # classify archetype only after this many matches
    archetype_alignment_weight: float = 0.25
    session_success_weight: float = 0.25
    min_session_samples: int = 5          # min weighted samples before session success activates


@dataclass(frozen=True)
class AIConfig:
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    planner_memory: PlannerMemoryConfig = field(default_factory=PlannerMemoryConfig)
    action_resolver: ActionResolverConfig = field(default_factory=ActionResolverConfig)
    reaction: ReactionConfig = field(default_factory=ReactionConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    session_adaptation: SessionAdaptationConfig = field(
        default_factory=SessionAdaptationConfig
    )
    # Alignment bonus table: {mode_name: {archetype_name: float}}
    # Passed through as raw dict; defaults to empty (no alignment bonuses).
    archetype_mode_alignment: dict = field(default_factory=dict)


# --- Display Config ---


@dataclass(frozen=True)
class WindowConfig:
    width: int = 1280
    height: int = 720
    title: str = "Adaptive Duelist"
    fps_cap: int = 60


@dataclass(frozen=True)
class ColorScheme:
    background: tuple[int, int, int] = (30, 30, 40)
    arena_floor: tuple[int, int, int] = (60, 60, 70)
    player: tuple[int, int, int] = (80, 160, 255)
    ai: tuple[int, int, int] = (255, 80, 80)
    hp_bar_fill: tuple[int, int, int] = (50, 200, 80)
    hp_bar_empty: tuple[int, int, int] = (80, 30, 30)
    stamina_bar_fill: tuple[int, int, int] = (220, 180, 40)
    stamina_bar_empty: tuple[int, int, int] = (60, 50, 20)
    text_primary: tuple[int, int, int] = (240, 240, 240)
    text_secondary: tuple[int, int, int] = (160, 160, 170)
    hud_background: tuple[int, int, int, int] = (20, 20, 30, 200)
    # Phase 13: UX polish additions
    accent: tuple[int, int, int] = (160, 120, 240)
    hit_flash: tuple[int, int, int] = (255, 255, 255)
    health_critical: tuple[int, int, int] = (255, 120, 30)
    stamina_low: tuple[int, int, int] = (255, 140, 30)
    panel_bg: tuple[int, int, int] = (18, 18, 28)
    tier_badge: tuple[int, int, int] = (80, 200, 140)
    border: tuple[int, int, int] = (60, 65, 90)


@dataclass(frozen=True)
class HUDConfig:
    hp_bar_width: int = 300
    hp_bar_height: int = 20
    stamina_bar_width: int = 300
    stamina_bar_height: int = 12
    ai_panel_width: int = 320
    ai_panel_height: int = 160
    ai_panel_x: int = 940
    ai_panel_y: int = 20
    font_size_primary: int = 18
    font_size_secondary: int = 14
    # Phase 13: UX polish additions
    font_size_large: int = 32


@dataclass(frozen=True)
class DisplayConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    colors: ColorScheme = field(default_factory=ColorScheme)
    hud: HUDConfig = field(default_factory=HUDConfig)


# --- Validation and loading ---


def _validate_positive(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    if value <= 0:
        raise ConfigError(f"{path}: must be positive, got {value}")


def _validate_non_negative(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    if value < 0:
        raise ConfigError(f"{path}: must be non-negative, got {value}")


def _validate_fraction(value: Any, path: str) -> None:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path}: must be between 0.0 and 1.0, got {value}")


def _validate_color(value: Any, path: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected a color list, got {type(value).__name__}")
    if len(value) not in (3, 4):
        raise ConfigError(f"{path}: color must have 3 or 4 components, got {len(value)}")
    for i, c in enumerate(value):
        if not isinstance(c, int) or not 0 <= c <= 255:
            raise ConfigError(f"{path}[{i}]: color component must be int 0-255, got {c}")
    return tuple(value)


def _has_required_fields(cls: type) -> bool:
    """Check if a dataclass has fields without defaults."""
    for f in cls.__dataclass_fields__.values():
        if (
            f.default is f.default_factory  # both MISSING
            and f.default_factory is f.default  # confirms MISSING
        ):
            return True
    return False


def _try_default(cls: type):
    """Try to construct a default instance. Returns None if the class has required fields."""
    try:
        return cls()
    except TypeError:
        return None


def _build_dataclass(cls: type, raw: dict | None, path: str):
    """Recursively build a frozen dataclass from a raw dict, using defaults for missing keys."""
    defaults = _try_default(cls)

    if raw is None:
        if defaults is None:
            raise ConfigError(
                f"{path}: all fields are required for {cls.__name__} but no values were provided"
            )
        return defaults

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    kwargs = {}
    hints = {k: v for k, v in cls.__dataclass_fields__.items()}

    for field_name, field_obj in hints.items():
        field_path = f"{path}.{field_name}"
        raw_value = raw.get(field_name)

        if raw_value is None:
            if defaults is not None:
                kwargs[field_name] = getattr(defaults, field_name)
            elif field_obj.default is not MISSING:
                # Field has its own default even though the class has required fields
                kwargs[field_name] = field_obj.default
            elif field_obj.default_factory is not MISSING:
                kwargs[field_name] = field_obj.default_factory()
            else:
                raise ConfigError(f"{field_path}: required field is missing")
            continue

        field_type = field_obj.type

        if isinstance(field_type, str):
            # Resolve forward references in the module scope
            field_type = eval(field_type, globals())

        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_name] = _build_dataclass(field_type, raw_value, field_path)
        elif field_type in (
            tuple[int, int, int],
            tuple[int, int, int, int],
            "tuple[int, int, int]",
            "tuple[int, int, int, int]",
        ):
            kwargs[field_name] = _validate_color(raw_value, field_path)
        else:
            kwargs[field_name] = raw_value

    return cls(**kwargs)


def _validate_game_config(cfg: GameConfig) -> None:
    _validate_positive(cfg.arena.width, "arena.width")
    _validate_positive(cfg.arena.height, "arena.height")
    _validate_positive(cfg.fighter.max_hp, "fighter.max_hp")
    _validate_positive(cfg.fighter.max_stamina, "fighter.max_stamina")
    _validate_non_negative(cfg.fighter.stamina_regen_idle, "fighter.stamina_regen_idle")
    _validate_positive(cfg.fighter.move_speed, "fighter.move_speed")
    _validate_positive(cfg.simulation.tick_rate, "simulation.tick_rate")
    _validate_positive(cfg.simulation.sub_pixel_scale, "simulation.sub_pixel_scale")

    if cfg.spacing.close_max >= cfg.spacing.mid_max:
        raise ConfigError(
            f"spacing.close_max ({cfg.spacing.close_max}) must be less than "
            f"spacing.mid_max ({cfg.spacing.mid_max})"
        )

    for name in ("light_attack", "heavy_attack"):
        atk = getattr(cfg.actions, name)
        _validate_positive(atk.startup_frames, f"actions.{name}.startup_frames")
        _validate_positive(atk.active_frames, f"actions.{name}.active_frames")
        _validate_positive(atk.recovery_frames, f"actions.{name}.recovery_frames")
        _validate_positive(atk.stamina_cost, f"actions.{name}.stamina_cost")
        _validate_positive(atk.damage, f"actions.{name}.damage")
        _validate_positive(atk.reach, f"actions.{name}.reach")

        if atk.stamina_cost > cfg.fighter.max_stamina:
            raise ConfigError(
                f"actions.{name}.stamina_cost ({atk.stamina_cost}) exceeds "
                f"fighter.max_stamina ({cfg.fighter.max_stamina})"
            )


def _validate_ai_config(cfg: AIConfig) -> None:
    # Prediction
    _validate_positive(cfg.prediction.window_ticks, "prediction.window_ticks")
    _validate_positive(cfg.prediction.window_max_ticks, "prediction.window_max_ticks")
    _validate_positive(cfg.prediction.window_extend_step, "prediction.window_extend_step")
    _validate_positive(cfg.prediction.markov_order, "prediction.markov_order")
    _validate_fraction(cfg.prediction.hold_extend_threshold, "prediction.hold_extend_threshold")
    _validate_fraction(cfg.prediction.min_confidence_for_display, "prediction.min_confidence_for_display")
    _validate_positive(cfg.prediction.inference_reeval_idle_ticks, "prediction.inference_reeval_idle_ticks")

    if cfg.prediction.window_ticks > cfg.prediction.window_max_ticks:
        raise ConfigError(
            f"prediction.window_ticks ({cfg.prediction.window_ticks}) exceeds "
            f"prediction.window_max_ticks ({cfg.prediction.window_max_ticks})"
        )

    # Ensemble
    _validate_fraction(cfg.ensemble.initial_markov_weight, "ensemble.initial_markov_weight")
    _validate_fraction(cfg.ensemble.initial_sklearn_weight, "ensemble.initial_sklearn_weight")
    _validate_positive(cfg.ensemble.sklearn_activation_matches, "ensemble.sklearn_activation_matches")
    _validate_fraction(cfg.ensemble.weight_update_ema_alpha, "ensemble.weight_update_ema_alpha")

    # Training
    _validate_positive(cfg.training.min_matches_to_train, "training.min_matches_to_train")
    _validate_positive(cfg.training.min_samples_to_train, "training.min_samples_to_train")
    _validate_positive(cfg.training.retrain_after_every_n_matches, "training.retrain_after_every_n_matches")
    _validate_positive(cfg.training.random_forest_n_estimators, "training.random_forest_n_estimators")
    _validate_positive(cfg.training.random_forest_max_depth, "training.random_forest_max_depth")
    _validate_positive(cfg.training.random_forest_min_samples_leaf, "training.random_forest_min_samples_leaf")
    _validate_fraction(cfg.training.holdout_fraction, "training.holdout_fraction")

    # Strategy
    _validate_positive(cfg.strategy.softmax_temperature, "strategy.softmax_temperature")

    # Planner memory
    pm = cfg.planner_memory
    _validate_positive(pm.recent_modes_capacity, "planner_memory.recent_modes_capacity")
    _validate_positive(pm.mode_outcome_capacity, "planner_memory.mode_outcome_capacity")
    _validate_positive(pm.recent_predictions_capacity, "planner_memory.recent_predictions_capacity")
    _validate_positive(pm.exploit_staleness_threshold_ticks, "planner_memory.exploit_staleness_threshold_ticks")
    _validate_fraction(pm.exploration_budget_drain_rate, "planner_memory.exploration_budget_drain_rate")
    _validate_fraction(pm.exploration_budget_regen_rate, "planner_memory.exploration_budget_regen_rate")
    _validate_fraction(pm.exploration_budget_floor, "planner_memory.exploration_budget_floor")
    _validate_positive(pm.max_consecutive_before_penalty, "planner_memory.max_consecutive_before_penalty")
    _validate_fraction(pm.shift_detection_confidence_drop, "planner_memory.shift_detection_confidence_drop")
    _validate_positive(pm.shift_probe_duration_decisions, "planner_memory.shift_probe_duration_decisions")

    if pm.exploration_budget_drain_rate >= pm.exploration_budget_regen_rate:
        raise ConfigError(
            f"planner_memory.exploration_budget_drain_rate "
            f"({pm.exploration_budget_drain_rate}) must be less than "
            f"exploration_budget_regen_rate ({pm.exploration_budget_regen_rate})"
        )

    # Action resolver
    _validate_positive(cfg.action_resolver.max_commit_delay_ticks, "action_resolver.max_commit_delay_ticks")

    # Reaction
    _validate_positive(cfg.reaction.min_reaction_ticks, "reaction.min_reaction_ticks")

    # Profile
    _validate_positive(cfg.profile.recency_weight_multiplier, "profile.recency_weight_multiplier")
    _validate_positive(cfg.profile.rolling_window_size, "profile.rolling_window_size")

    # Session adaptation
    sa = cfg.session_adaptation
    _validate_fraction(sa.decay_factor, "session_adaptation.decay_factor")
    _validate_positive(sa.min_matches_for_archetype, "session_adaptation.min_matches_for_archetype")
    _validate_non_negative(sa.archetype_alignment_weight, "session_adaptation.archetype_alignment_weight")
    _validate_non_negative(sa.session_success_weight, "session_adaptation.session_success_weight")
    _validate_positive(sa.min_session_samples, "session_adaptation.min_session_samples")


def _validate_display_config(cfg: DisplayConfig) -> None:
    _validate_positive(cfg.window.width, "window.width")
    _validate_positive(cfg.window.height, "window.height")
    _validate_positive(cfg.window.fps_cap, "window.fps_cap")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse {path.name}: {e}") from e


def load_config(
    config_dir: Path | None = None,
) -> tuple[GameConfig, AIConfig, DisplayConfig]:
    """Load and validate all configuration files.

    Returns (GameConfig, AIConfig, DisplayConfig).
    Missing files or missing keys fall back to defaults.
    """
    base = config_dir or CONFIG_DIR

    game_raw = _load_yaml(base / "game_config.yaml")
    ai_raw = _load_yaml(base / "ai_config.yaml")
    display_raw = _load_yaml(base / "display_config.yaml")

    game_cfg = _build_dataclass(GameConfig, game_raw, "game")
    ai_cfg = _build_dataclass(AIConfig, ai_raw, "ai")
    display_cfg = _build_dataclass(DisplayConfig, display_raw, "display")

    _validate_game_config(game_cfg)
    _validate_ai_config(ai_cfg)
    _validate_display_config(display_cfg)

    return game_cfg, ai_cfg, display_cfg


def config_hash(path: Path) -> str:
    """SHA-256 hash of a config file's contents, for replay headers."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
