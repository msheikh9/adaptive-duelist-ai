"""Tests for SimulationState and phase lock mechanism."""

from __future__ import annotations

import pytest

from game.combat.actions import FSMState
from game.state import (
    ArenaState,
    FighterState,
    MatchStatus,
    PhaseLockError,
    SimulationState,
    TickPhase,
)


class TestFighterState:
    def test_default_is_idle_and_free(self):
        f = FighterState()
        assert f.fsm_state == FSMState.IDLE
        assert f.is_free
        assert not f.is_locked
        assert f.is_alive

    def test_locked_state(self):
        f = FighterState(fsm_state=FSMState.ATTACK_STARTUP)
        assert f.is_locked
        assert not f.is_free

    def test_ko_state(self):
        f = FighterState(fsm_state=FSMState.KO)
        assert not f.is_alive
        assert f.is_locked


class TestArenaState:
    def test_from_config(self):
        arena = ArenaState.from_config(
            arena_width=1200, arena_height=400, ground_y=300, sub_pixel_scale=100,
        )
        assert arena.width_sub == 120_000
        assert arena.height_sub == 40_000
        assert arena.ground_y_sub == 30_000


class TestSimulationState:
    def test_distance_sub(self):
        state = SimulationState()
        state.player.x = 10000
        state.ai.x = 50000
        assert state.distance_sub() == 40000

    def test_distance_px(self):
        state = SimulationState()
        state.player.x = 10000
        state.ai.x = 50000
        assert state.distance_px(100) == 400.0

    def test_default_match_status(self):
        state = SimulationState()
        assert state.match_status == MatchStatus.WAITING
        assert state.winner is None


class TestPhaseLock:
    def test_mutation_allowed_during_simulate(self):
        state = SimulationState()
        state.set_phase(TickPhase.SIMULATE)
        state.phase_lock.check_write()  # should not raise

    def test_mutation_blocked_during_render(self):
        state = SimulationState()
        state.set_phase(TickPhase.RENDER)
        with pytest.raises(PhaseLockError, match="RENDER phase"):
            state.phase_lock.check_write()

    def test_mutation_blocked_during_ai_observe(self):
        state = SimulationState()
        state.set_phase(TickPhase.AI_OBSERVE)
        with pytest.raises(PhaseLockError, match="AI_OBSERVE phase"):
            state.phase_lock.check_write()

    def test_mutation_blocked_during_log(self):
        state = SimulationState()
        state.set_phase(TickPhase.LOG)
        with pytest.raises(PhaseLockError, match="LOG phase"):
            state.phase_lock.check_write()

    def test_mutation_blocked_during_input(self):
        state = SimulationState()
        state.set_phase(TickPhase.INPUT)
        with pytest.raises(PhaseLockError, match="INPUT phase"):
            state.phase_lock.check_write()
