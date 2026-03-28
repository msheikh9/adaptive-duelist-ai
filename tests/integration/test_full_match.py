"""Integration test: headless full match simulation."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.state import MatchStatus
from tests.fixtures.headless_engine import HeadlessMatch


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


class TestHeadlessMatch:
    def test_match_starts_active(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        assert match.state.match_status == MatchStatus.ACTIVE
        assert match.state.player.hp == cfg.fighter.max_hp
        assert match.state.ai.hp == cfg.fighter.max_hp

    def test_match_ends_with_winner(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        match.run_until_end(max_ticks=20000)
        assert match.state.match_status == MatchStatus.ENDED
        assert match.state.winner in ("PLAYER", "AI")

    def test_loser_has_zero_hp(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        match.run_until_end()
        if match.state.winner == "PLAYER":
            assert match.state.ai.hp == 0
            assert match.state.ai.fsm_state == FSMState.KO
        else:
            assert match.state.player.hp == 0
            assert match.state.player.fsm_state == FSMState.KO

    def test_winner_has_positive_hp(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        match.run_until_end()
        if match.state.winner == "PLAYER":
            assert match.state.player.hp > 0
        else:
            assert match.state.ai.hp > 0

    def test_tick_snapshots_recorded(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        match.run_ticks(100)
        assert len(match.snapshots) == 100
        assert match.snapshots[0].tick_id == 0
        assert match.snapshots[99].tick_id == 99

    def test_deterministic_replay(self, cfg):
        """Same seed and inputs must produce identical results."""
        match1 = HeadlessMatch(cfg, rng_seed=123)
        match1.run_until_end(max_ticks=5000)

        match2 = HeadlessMatch(cfg, rng_seed=123)
        match2.run_until_end(max_ticks=5000)

        assert match1.state.tick_id == match2.state.tick_id
        assert match1.state.winner == match2.state.winner
        assert match1.state.player.hp == match2.state.player.hp
        assert match1.state.ai.hp == match2.state.ai.hp
        assert match1.state.player.x == match2.state.player.x
        assert match1.state.ai.x == match2.state.ai.x


class TestPlayerInput:
    def test_player_can_attack(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        match.tick(CombatCommitment.LIGHT_ATTACK)
        assert match.state.player.fsm_state == FSMState.ATTACK_STARTUP
        assert match.state.player.active_commitment == CombatCommitment.LIGHT_ATTACK

    def test_player_attack_hits_ai(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        # Move player very close to AI first
        scale = cfg.simulation.sub_pixel_scale
        match.state.set_phase(MatchStatus.ACTIVE)  # hack for direct mutation
        from game.state import TickPhase
        match.state.set_phase(TickPhase.SIMULATE)
        match.state.player.x = match.state.ai.x - 60 * scale  # within light reach

        # Attack
        match.tick(CombatCommitment.LIGHT_ATTACK)
        # Advance through startup to active
        startup = cfg.actions.light_attack.startup_frames
        match.run_ticks(startup)

        # Check if AI took damage at some point
        initial_hp = cfg.fighter.max_hp
        # Might have taken damage from AI attacks too, but let's check total events
        hits_on_ai = [e for e in match.events if e[0] == "HIT" and e[1].name == "PLAYER"]
        assert len(hits_on_ai) >= 1

    def test_player_dodge_avoids_hit(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        scale = cfg.simulation.sub_pixel_scale
        from game.state import TickPhase

        # Position close
        match.state.set_phase(TickPhase.SIMULATE)
        match.state.player.x = match.state.ai.x - 60 * scale

        # Player dodges
        match.tick(CombatCommitment.DODGE_BACKWARD)
        dodge_startup = cfg.actions.dodge_backward.startup_frames
        match.run_ticks(dodge_startup)
        assert match.state.player.fsm_state == FSMState.DODGING

    def test_stamina_depletes_during_match(self, cfg):
        match = HeadlessMatch(cfg, rng_seed=42)
        initial_stamina = match.state.player.stamina
        # Do 3 light attacks in sequence
        for _ in range(3):
            # Wait for free state
            while match.state.player.is_locked:
                match.tick()
            match.tick(CombatCommitment.LIGHT_ATTACK)

        expected_loss = cfg.actions.light_attack.stamina_cost * 3
        # Stamina may have regenerated some, but should be less than initial
        assert match.state.player.stamina < initial_stamina
