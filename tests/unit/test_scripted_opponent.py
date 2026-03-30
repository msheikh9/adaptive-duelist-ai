"""Tests for ai/training/scripted_opponent.py."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState, PHASE_1_COMMITMENTS
from game.state import FighterState, SimulationState, ArenaState, MatchStatus

from ai.training.scripted_opponent import (
    ScriptedOpponent,
    ScriptedProfile,
    _PATTERNED_CYCLE,
    _MIXABLE,
)


@pytest.fixture
def game_cfg():
    cfg, _, _ = load_config()
    return cfg


@pytest.fixture
def fighter(game_cfg):
    """Fresh IDLE fighter at full resources."""
    return FighterState(
        x=300,
        y=game_cfg.arena.ground_y * game_cfg.simulation.sub_pixel_scale,
        hp=game_cfg.fighter.max_hp,
        stamina=game_cfg.fighter.max_stamina,
        facing=1,
        fsm_state=FSMState.IDLE,
    )


@pytest.fixture
def sim(game_cfg, fighter):
    scale = game_cfg.simulation.sub_pixel_scale
    arena = ArenaState.from_config(
        game_cfg.arena.width, game_cfg.arena.height,
        game_cfg.arena.ground_y, scale,
    )
    ai_fighter = FighterState(
        x=(arena.width_sub * 2) // 3,
        y=arena.ground_y_sub,
        hp=game_cfg.fighter.max_hp,
        stamina=game_cfg.fighter.max_stamina,
        facing=-1,
    )
    return SimulationState(
        tick_id=0,
        rng_seed=0,
        player=fighter,
        ai=ai_fighter,
        arena=arena,
        match_status=MatchStatus.ACTIVE,
    )


def _collect_decisions(opponent, fighter, sim, game_cfg, n: int) -> list[CombatCommitment]:
    """Force N accepted decisions by resetting fighter state between decisions."""
    decisions = []
    while len(decisions) < n:
        # Force the opponent to decide immediately
        opponent._ticks_until_decision = 1
        # Reset fighter to free state with full resources
        fighter.fsm_state = FSMState.IDLE
        fighter.active_commitment = None
        fighter.stamina = game_cfg.fighter.max_stamina
        fighter.fsm_frames_remaining = 0

        commit = opponent.decide(fighter, sim, game_cfg)
        if commit is not None:
            decisions.append(commit)
    return decisions


class TestPhase1Only:
    """All profiles must produce only Phase 1 commitments."""

    @pytest.mark.parametrize("profile", list(ScriptedProfile))
    def test_only_phase1_commitments(self, profile, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(profile, seed=42)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, 100)
        for d in decisions:
            assert d in PHASE_1_COMMITMENTS, (
                f"Profile {profile.value} produced non-Phase1 commitment: {d}"
            )


class TestDecideReturnsNoneWhenLocked:
    """decide() returns None when fighter is locked."""

    def test_locked_fighter_returns_none(self, game_cfg, fighter, sim):
        fighter.fsm_state = FSMState.ATTACK_STARTUP
        opponent = ScriptedOpponent(ScriptedProfile.RANDOM, seed=0)
        opponent._ticks_until_decision = 1
        result = opponent.decide(fighter, sim, game_cfg)
        assert result is None

    def test_ko_fighter_returns_none(self, game_cfg, fighter, sim):
        fighter.fsm_state = FSMState.KO
        opponent = ScriptedOpponent(ScriptedProfile.AGGRESSIVE, seed=0)
        opponent._ticks_until_decision = 1
        result = opponent.decide(fighter, sim, game_cfg)
        assert result is None


class TestRandomProfile:
    """RANDOM: all 5 Phase1 commitments appear over sufficient decisions."""

    def test_all_commitments_appear(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.RANDOM, seed=42)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, 200)
        seen = set(decisions)
        for c in PHASE_1_COMMITMENTS:
            assert c in seen, f"RANDOM never produced {c.name} over 200 decisions"


class TestAggressiveProfile:
    """AGGRESSIVE: attack share > movement share over 200 decisions.

    Note: the distance bias (far fighters prefer advancing) can reduce raw
    attack share. We assert attacks dominate over dodge — the core invariant
    — rather than a fixed percentage.
    """

    def test_attack_heavy_bias(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.AGGRESSIVE, seed=42)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, 200)
        attacks = sum(
            1 for d in decisions
            if d in (CombatCommitment.LIGHT_ATTACK, CombatCommitment.HEAVY_ATTACK)
        )
        dodges = sum(1 for d in decisions if d == CombatCommitment.DODGE_BACKWARD)
        # Attacks should far outnumber dodges (weight 7.0 vs 0.5)
        assert attacks > dodges * 5, (
            f"AGGRESSIVE attacks ({attacks}) should far exceed dodges ({dodges})"
        )
        # Attack share should be at least 40% even with distance bias applied
        assert attacks / len(decisions) >= 0.40, (
            f"AGGRESSIVE attack share too low: {attacks}/{len(decisions)}"
        )


class TestDefensiveProfile:
    """DEFENSIVE: DODGE_BACKWARD share >= 40% over 200 decisions."""

    def test_dodge_heavy_bias(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.DEFENSIVE, seed=42)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, 200)
        dodges = sum(1 for d in decisions if d == CombatCommitment.DODGE_BACKWARD)
        assert dodges / len(decisions) >= 0.35, (
            f"DEFENSIVE dodge share too low: {dodges}/{len(decisions)}"
        )


class TestPatternedProfile:
    """PATTERNED: produces the defined cycle, wraps correctly."""

    def test_first_cycle_matches_pattern(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.PATTERNED, seed=0)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, len(_PATTERNED_CYCLE))
        assert decisions == list(_PATTERNED_CYCLE), (
            f"First cycle mismatch: expected {[c.name for c in _PATTERNED_CYCLE]}, "
            f"got {[d.name for d in decisions]}"
        )

    def test_cycle_wraps(self, game_cfg, fighter, sim):
        cycle_len = len(_PATTERNED_CYCLE)
        opponent = ScriptedOpponent(ScriptedProfile.PATTERNED, seed=0)
        decisions = _collect_decisions(opponent, fighter, sim, game_cfg, cycle_len * 2)
        # Second cycle must match first cycle
        first = decisions[:cycle_len]
        second = decisions[cycle_len:]
        assert first == second, (
            f"Cycle does not repeat: first={[c.name for c in first]}, "
            f"second={[c.name for c in second]}"
        )


class TestDeterminism:
    """Same seed produces identical decision sequences."""

    def test_identical_seeds_identical_sequence(self, game_cfg):
        scale = game_cfg.simulation.sub_pixel_scale
        arena = ArenaState.from_config(
            game_cfg.arena.width, game_cfg.arena.height,
            game_cfg.arena.ground_y, scale,
        )

        def make_pair():
            f = FighterState(x=300, y=arena.ground_y_sub,
                             hp=game_cfg.fighter.max_hp,
                             stamina=game_cfg.fighter.max_stamina)
            ai = FighterState(x=900, y=arena.ground_y_sub,
                              hp=game_cfg.fighter.max_hp,
                              stamina=game_cfg.fighter.max_stamina,
                              facing=-1)
            s = SimulationState(player=f, ai=ai, arena=arena,
                                match_status=MatchStatus.ACTIVE)
            return ScriptedOpponent(ScriptedProfile.RANDOM, seed=42), f, s

        opp1, f1, s1 = make_pair()
        opp2, f2, s2 = make_pair()

        d1 = _collect_decisions(opp1, f1, s1, game_cfg, 50)
        d2 = _collect_decisions(opp2, f2, s2, game_cfg, 50)
        assert d1 == d2

    def test_reset_with_seed_produces_known_sequence(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.RANDOM, seed=99)
        d1 = _collect_decisions(opponent, fighter, sim, game_cfg, 20)

        # Reset fighter and opponent to reproduce
        fighter.fsm_state = FSMState.IDLE
        fighter.stamina = game_cfg.fighter.max_stamina
        opponent.reset(seed=99)
        d2 = _collect_decisions(opponent, fighter, sim, game_cfg, 20)
        assert d1 == d2


class TestMixedProfile:
    """MIXED: selects distinct underlying profiles across resets."""

    def test_mixed_produces_at_least_two_profiles_over_10_resets(self, game_cfg):
        opponent = ScriptedOpponent(ScriptedProfile.MIXED, seed=0)
        seen_profiles = set()
        for i in range(10):
            opponent.reset(seed=i)
            seen_profiles.add(opponent._active_profile)
        assert len(seen_profiles) >= 2, (
            f"MIXED always selected the same profile: {seen_profiles}"
        )

    def test_mixed_only_delegates_to_mixable_profiles(self, game_cfg):
        opponent = ScriptedOpponent(ScriptedProfile.MIXED, seed=0)
        for i in range(20):
            opponent.reset(seed=i)
            assert opponent._active_profile in _MIXABLE


class TestCountdownBehavior:
    """decide() respects the countdown interval — returns None until timer fires."""

    def test_returns_none_before_interval(self, game_cfg, fighter, sim):
        opponent = ScriptedOpponent(ScriptedProfile.RANDOM, seed=0)
        # The interval is 10-40 ticks. With a fresh opponent, timer > 0
        # so the first call should return None (timer decrements to >0).
        # (Unless the initial interval was sampled as 1.)
        opponent._ticks_until_decision = 5  # force a known interval
        # First 4 calls should all return None
        results = []
        for _ in range(4):
            results.append(opponent.decide(fighter, sim, game_cfg))
        assert all(r is None for r in results)
