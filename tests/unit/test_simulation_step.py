"""Per-subsystem coverage for `step_simulation`, the one authoritative tick.

Four callers share this function (engine, replay verifier, batch evaluator, test
fixture). Because they share it, a missing subsystem does *not* show up as a
replay mismatch — recorder and replayer would be wrong together and still agree.
The only thing that catches a dropped subsystem is asserting on it directly, so
that is what these tests do.

Each test here fails if its subsystem is removed from `step_simulation`. Before
these existed, deleting `tick_dodge_cooldown`, `tick_shoot_cooldown` or
`tick_guard` from the step left the entire suite green — which is exactly how the
evaluator ran for so long with cooldowns that never decremented, letting the AI
use heavy attack and dodge at most once per match.
"""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.simulation_step import (
    SimulationContext,
    step_simulation,
)
from game.state import ArenaState, FighterState, MatchStatus, SimulationState


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def sim(cfg):
    """Two idle fighters at mid distance, out of each other's reach."""
    scale = cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        cfg.arena.width, cfg.arena.height, cfg.arena.ground_y, scale)
    return SimulationState(
        tick_id=0, rng_seed=1,
        player=FighterState(
            x=arena.width_sub // 4, y=arena.ground_y_sub,
            hp=cfg.fighter.max_hp, stamina=cfg.fighter.max_stamina, facing=1),
        ai=FighterState(
            x=(arena.width_sub * 3) // 4, y=arena.ground_y_sub,
            hp=cfg.fighter.max_hp, stamina=cfg.fighter.max_stamina, facing=-1),
        arena=arena, match_status=MatchStatus.ACTIVE,
    )


def _idle():
    """A decider that never acts."""
    return []


def _advance(sim, cfg, ctx, n, decide_player=_idle, decide_ai=_idle):
    for i in range(n):
        sim.tick_id = i
        step_simulation(sim, cfg, ctx, decide_player, decide_ai)


# ---------------------------------------------------------------------------
# Cooldown timers — the three that had no coverage
# ---------------------------------------------------------------------------

class TestCooldownsDecrement:
    """Cooldowns are set on commit and decremented only by the step.

    If the step stops ticking them they stay pinned forever, and the action
    becomes usable exactly once per match.
    """

    def test_dodge_cooldown_decrements(self, sim, cfg):
        ctx = SimulationContext()
        sim.player.dodge_cooldown = 10
        _advance(sim, cfg, ctx, 1)
        assert sim.player.dodge_cooldown == 9

    def test_dodge_cooldown_reaches_zero(self, sim, cfg):
        ctx = SimulationContext()
        sim.player.dodge_cooldown = 5
        _advance(sim, cfg, ctx, 8)
        assert sim.player.dodge_cooldown == 0

    def test_dodge_becomes_reusable_after_cooldown(self, sim, cfg):
        """The behavioural consequence: dodge must work more than once."""
        from game.combat.state_machine import can_commit
        ctx = SimulationContext()
        cooldown = cfg.actions.dodge_backward.cooldown_frames

        def dodge_once():
            from game.entities.fighter import attempt_commitment
            if attempt_commitment(
                    sim.player, CombatCommitment.DODGE_BACKWARD, cfg):
                return [CombatCommitment.DODGE_BACKWARD]
            return []

        sim.tick_id = 0
        step_simulation(sim, cfg, ctx, dodge_once, _idle)
        assert sim.player.dodge_cooldown > 0

        # Run past the cooldown and the recovery it implies.
        _advance(sim, cfg, ctx, cooldown + 60)
        assert sim.player.dodge_cooldown == 0
        assert can_commit(sim.player, CombatCommitment.DODGE_BACKWARD, cfg), (
            "dodge never became available again — cooldown is not being ticked"
        )

    def test_heavy_cooldown_decrements(self, sim, cfg):
        ctx = SimulationContext()
        sim.ai.heavy_cooldown = 12
        _advance(sim, cfg, ctx, 1)
        assert sim.ai.heavy_cooldown == 11

    def test_heavy_becomes_reusable_after_cooldown(self, sim, cfg):
        from game.combat.state_machine import can_commit
        from game.entities.fighter import attempt_commitment
        ctx = SimulationContext()

        def heavy_once():
            if attempt_commitment(
                    sim.ai, CombatCommitment.HEAVY_ATTACK, cfg):
                return [CombatCommitment.HEAVY_ATTACK]
            return []

        sim.tick_id = 0
        step_simulation(sim, cfg, ctx, _idle, heavy_once)
        assert sim.ai.heavy_cooldown > 0

        _advance(sim, cfg, ctx, cfg.actions.heavy_attack.cooldown_ticks + 90)
        assert sim.ai.heavy_cooldown == 0
        assert can_commit(sim.ai, CombatCommitment.HEAVY_ATTACK, cfg)

    def test_shoot_cooldown_decrements(self, sim, cfg):
        ctx = SimulationContext()
        sim.ai.shoot_cooldown = 20
        _advance(sim, cfg, ctx, 1)
        assert sim.ai.shoot_cooldown == 19

    def test_shoot_cooldown_reaches_zero(self, sim, cfg):
        ctx = SimulationContext()
        sim.ai.shoot_cooldown = 4
        _advance(sim, cfg, ctx, 6)
        assert sim.ai.shoot_cooldown == 0


# ---------------------------------------------------------------------------
# Guard meter
# ---------------------------------------------------------------------------

class TestGuardTicks:
    def test_guard_regenerates_while_not_blocking(self, sim, cfg):
        """A drained guard meter must recover, which only tick_guard does."""
        ctx = SimulationContext()
        sim.player.guard = 0
        _advance(sim, cfg, ctx, 120)
        assert sim.player.guard > 0, (
            "guard meter never regenerated — tick_guard is not being called"
        )


# ---------------------------------------------------------------------------
# Gravity and projectiles
# ---------------------------------------------------------------------------

class TestGravityAndProjectiles:
    def test_airborne_fighter_falls_and_lands(self, sim, cfg):
        ctx = SimulationContext()
        from game.entities.fighter import attempt_commitment

        def jump():
            if attempt_commitment(sim.player, CombatCommitment.JUMP, cfg):
                return [CombatCommitment.JUMP]
            return []

        sim.tick_id = 0
        step_simulation(sim, cfg, ctx, jump, _idle)
        ground = sim.arena.ground_y_sub

        went_up = False
        for i in range(1, 400):
            sim.tick_id = i
            fx = step_simulation(sim, cfg, ctx, _idle, _idle)
            if sim.player.y < ground:
                went_up = True
            if went_up and fx.player_landed:
                break
        else:
            pytest.fail("fighter jumped but never landed — gravity is missing")

        assert went_up, "JUMP never left the ground"
        assert sim.player.y == ground

    def test_projectile_is_simulated_and_can_damage(self, sim, cfg):
        """A fired shot must travel and connect.

        The step owns projectile movement and collision. If it stops calling
        update_projectiles, shots are spawned and then frozen forever — which is
        how evaluated AI fired hundreds of shots that could not deal damage.
        """
        ctx = SimulationContext()
        from game.entities.fighter import attempt_commitment

        # Put the fighters close so the shot connects quickly.
        sim.ai.x = sim.player.x + cfg.fighter.width * cfg.simulation.sub_pixel_scale * 3
        start_hp = sim.ai.hp

        def shoot():
            if attempt_commitment(
                    sim.player, CombatCommitment.SHOOT_INSTANT, cfg):
                return [CombatCommitment.SHOOT_INSTANT]
            return []

        sim.tick_id = 0
        fx = step_simulation(sim, cfg, ctx, shoot, _idle)
        assert fx.projectiles_fired, "SHOOT_INSTANT spawned no projectile"

        for i in range(1, 200):
            sim.tick_id = i
            fx = step_simulation(sim, cfg, ctx, _idle, _idle)
            if fx.projectile_hits:
                break
        else:
            pytest.fail(
                "projectile never resolved — update_projectiles is not running")

        assert sim.ai.hp < start_hp


# ---------------------------------------------------------------------------
# The multi-commitment contract the replay stream depends on
# ---------------------------------------------------------------------------

class TestDeciderContract:
    def test_all_entered_commitments_are_reported(self, sim, cfg):
        """Every commitment a decider enters must come back in TickEffects.

        The replay recorder writes exactly what TickEffects reports, so anything
        dropped here silently disappears from the replay stream.
        """
        from game.entities.fighter import attempt_commitment
        ctx = SimulationContext()

        def two_commitments():
            entered = []
            # MOVING is a free state, so a second commitment can follow.
            if attempt_commitment(sim.ai, CombatCommitment.MOVE_LEFT, cfg):
                entered.append(CombatCommitment.MOVE_LEFT)
            if attempt_commitment(
                    sim.ai, CombatCommitment.SHOOT_INSTANT, cfg):
                entered.append(CombatCommitment.SHOOT_INSTANT)
            return entered

        sim.tick_id = 0
        fx = step_simulation(sim, cfg, ctx, _idle, two_commitments)
        assert fx.ai_commitments == [
            CombatCommitment.MOVE_LEFT, CombatCommitment.SHOOT_INSTANT,
        ]
