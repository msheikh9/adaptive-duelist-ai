"""Tests for hitbox detection and collision system."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from game.combat.actions import CombatCommitment, FSMState
from game.combat.collision import HitTracker, check_hit
from game.combat.hitbox import Hitbox, get_attack_hitbox, get_fighter_hurtbox
from game.combat.state_machine import enter_commitment, tick_fsm
from game.state import FighterState


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def scale(cfg):
    return cfg.simulation.sub_pixel_scale


class TestHitbox:
    def test_overlapping_boxes(self):
        a = Hitbox(0, 100, 0, 100)
        b = Hitbox(50, 150, 50, 150)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_non_overlapping_boxes(self):
        a = Hitbox(0, 100, 0, 100)
        b = Hitbox(200, 300, 200, 300)
        assert not a.overlaps(b)

    def test_adjacent_boxes_dont_overlap(self):
        a = Hitbox(0, 100, 0, 100)
        b = Hitbox(100, 200, 0, 100)
        assert not a.overlaps(b)


class TestGetAttackHitbox:
    def test_no_hitbox_during_startup(self, cfg):
        f = FighterState(x=50000, y=30000, facing=1)
        enter_commitment(f, CombatCommitment.LIGHT_ATTACK, cfg)
        assert f.fsm_state == FSMState.ATTACK_STARTUP
        assert get_attack_hitbox(f, cfg) is None

    def test_hitbox_during_active(self, cfg, scale):
        f = FighterState(x=50000, y=30000, facing=1)
        enter_commitment(f, CombatCommitment.LIGHT_ATTACK, cfg)
        # Advance to active phase
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(f, cfg)
        assert f.fsm_state == FSMState.ATTACK_ACTIVE
        hb = get_attack_hitbox(f, cfg)
        assert hb is not None
        # Hitbox extends in front (facing right)
        half_w = (cfg.fighter.width * scale) // 2
        assert hb.x_min == f.x + half_w
        assert hb.x_max == f.x + half_w + cfg.actions.light_attack.reach * scale

    def test_hitbox_facing_left(self, cfg, scale):
        f = FighterState(x=50000, y=30000, facing=-1)
        enter_commitment(f, CombatCommitment.HEAVY_ATTACK, cfg)
        for _ in range(cfg.actions.heavy_attack.startup_frames):
            tick_fsm(f, cfg)
        hb = get_attack_hitbox(f, cfg)
        assert hb is not None
        half_w = (cfg.fighter.width * scale) // 2
        assert hb.x_max == f.x - half_w
        assert hb.x_min == f.x - half_w - cfg.actions.heavy_attack.reach * scale

    def test_no_hitbox_during_recovery(self, cfg):
        f = FighterState(x=50000, y=30000, facing=1)
        enter_commitment(f, CombatCommitment.LIGHT_ATTACK, cfg)
        total_before_recovery = (cfg.actions.light_attack.startup_frames +
                                 cfg.actions.light_attack.active_frames)
        for _ in range(total_before_recovery):
            tick_fsm(f, cfg)
        assert f.fsm_state == FSMState.ATTACK_RECOVERY
        assert get_attack_hitbox(f, cfg) is None


class TestCollisionDetection:
    def _make_close_fighters(self, cfg):
        scale = cfg.simulation.sub_pixel_scale
        p = FighterState(x=50000, y=30000, hp=200, stamina=100, facing=1)
        ai = FighterState(x=50000 + 80 * scale, y=30000, hp=200, stamina=100, facing=-1)
        return p, ai

    def test_hit_lands_during_active_frames(self, cfg):
        p, ai = self._make_close_fighters(cfg)
        tracker = HitTracker()

        enter_commitment(p, CombatCommitment.LIGHT_ATTACK, cfg)
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(p, cfg)

        assert p.fsm_state == FSMState.ATTACK_ACTIVE
        hit = check_hit(p, ai, "player", tracker, cfg)
        assert hit is not None
        assert hit.damage == cfg.actions.light_attack.damage

    def test_one_hit_per_swing(self, cfg):
        p, ai = self._make_close_fighters(cfg)
        tracker = HitTracker()

        enter_commitment(p, CombatCommitment.LIGHT_ATTACK, cfg)
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(p, cfg)

        hit1 = check_hit(p, ai, "player", tracker, cfg)
        hit2 = check_hit(p, ai, "player", tracker, cfg)
        assert hit1 is not None
        assert hit2 is None  # Already connected

    def test_dodge_grants_invincibility(self, cfg):
        p, ai = self._make_close_fighters(cfg)
        tracker = HitTracker()

        # Player attacks
        enter_commitment(p, CombatCommitment.LIGHT_ATTACK, cfg)
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(p, cfg)

        # AI is dodging
        enter_commitment(ai, CombatCommitment.DODGE_BACKWARD, cfg)
        for _ in range(cfg.actions.dodge_backward.startup_frames):
            tick_fsm(ai, cfg)
        assert ai.fsm_state == FSMState.DODGING

        hit = check_hit(p, ai, "player", tracker, cfg)
        assert hit is None  # Dodging = invincible

    def test_no_hit_when_out_of_range(self, cfg):
        scale = cfg.simulation.sub_pixel_scale
        p = FighterState(x=10000, y=30000, facing=1)
        ai = FighterState(x=80000, y=30000, facing=-1)
        tracker = HitTracker()

        enter_commitment(p, CombatCommitment.LIGHT_ATTACK, cfg)
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(p, cfg)

        hit = check_hit(p, ai, "player", tracker, cfg)
        assert hit is None

    def test_hit_tracker_reset(self, cfg):
        p, ai = self._make_close_fighters(cfg)
        tracker = HitTracker()

        # First swing hits
        enter_commitment(p, CombatCommitment.LIGHT_ATTACK, cfg)
        for _ in range(cfg.actions.light_attack.startup_frames):
            tick_fsm(p, cfg)
        hit1 = check_hit(p, ai, "player", tracker, cfg)
        assert hit1 is not None

        # Reset tracker (simulating return to free state)
        tracker.reset("player")

        # Same situation should hit again
        hit2 = check_hit(p, ai, "player", tracker, cfg)
        assert hit2 is not None
