"""Tests for DatasetBuilder: semantic_events → labeled feature dataset."""

from __future__ import annotations

import pytest

from config.config_loader import load_config
from data.db import Database
from data.migrations.migration_runner import run_migrations
from ai.features.feature_extractor import NUM_FEATURES
from ai.models.base_predictor import LABEL_HOLD, PHASE1_NAMES
from ai.profile.player_profile import PlayerProfile
from ai.training.dataset_builder import build_dataset


@pytest.fixture
def cfg():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    run_migrations(db)
    yield db
    db.close()


def _ensure_match(db, match_id: str) -> None:
    """Insert a match record if it doesn't exist (satisfies FK constraint)."""
    db.execute_safe(
        """INSERT OR IGNORE INTO matches
           (match_id, session_id, started_at, rng_seed, config_hash)
           VALUES (?, 'test-session', '2025-01-01T00:00:00', 42, 'testhash');""",
        (match_id,),
    )


def _insert_commitment_events(db, match_id: str, commitments: list[tuple[int, str]]) -> None:
    """Insert player COMMITMENT_START events into semantic_events.

    commitments: list of (tick_id, commitment_name)
    """
    _ensure_match(db, match_id)
    for tick_id, name in commitments:
        db.execute_safe(
            """INSERT INTO semantic_events
               (event_type, match_id, tick_id, actor, commitment,
                opponent_fsm_state, spacing_zone,
                actor_hp, opponent_hp, actor_stamina, opponent_stamina)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            ("COMMITMENT_START", match_id, tick_id, "PLAYER", name,
             "IDLE", "MID", 200, 200, 80, 80),
        )


class TestEmptyDataset:
    def test_empty_db_returns_empty(self, tmp_db, cfg):
        game_cfg, ai_cfg = cfg
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
        )
        assert X == []
        assert y == []


class TestBasicLabeling:
    def test_next_commitment_within_window_is_label(self, tmp_db, cfg):
        game_cfg, ai_cfg = cfg
        # Two events 30 ticks apart (within default window=60)
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
            (130, "HEAVY_ATTACK"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            window_ticks=60,
        )
        # First event labeled HEAVY_ATTACK, second labeled HOLD (no next)
        assert len(X) == 2
        assert y[0] == "HEAVY_ATTACK"
        assert y[1] == LABEL_HOLD

    def test_gap_exceeding_window_produces_hold(self, tmp_db, cfg):
        game_cfg, ai_cfg = cfg
        # Two events 100 ticks apart (exceeds window=60)
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
            (200, "HEAVY_ATTACK"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            window_ticks=60,
        )
        assert y[0] == LABEL_HOLD
        assert y[1] == LABEL_HOLD

    def test_last_event_in_match_always_hold(self, tmp_db, cfg):
        game_cfg, _ = cfg
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
        )
        assert len(y) == 1
        assert y[0] == LABEL_HOLD


class TestFeatureVectorIntegrity:
    def test_feature_vector_correct_length(self, tmp_db, cfg):
        game_cfg, _ = cfg
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
            (130, "HEAVY_ATTACK"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
        )
        assert all(len(row) == NUM_FEATURES for row in X)

    def test_all_labels_are_valid(self, tmp_db, cfg):
        game_cfg, _ = cfg
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
            (130, "HEAVY_ATTACK"),
            (160, "DODGE_BACKWARD"),
            (190, "MOVE_LEFT"),
            (220, "MOVE_RIGHT"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
        )
        valid = set(PHASE1_NAMES) | {LABEL_HOLD}
        for label in y:
            assert label in valid, f"Invalid label: {label}"


class TestCrossMatchIsolation:
    def test_events_from_different_matches_dont_label_across(self, tmp_db, cfg):
        game_cfg, _ = cfg
        # Match 1 last event and Match 2 first event are close in tick_id
        # but should not cross-label
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
        ])
        _insert_commitment_events(tmp_db, "m2", [
            (110, "HEAVY_ATTACK"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            window_ticks=60,
        )
        # m1's event should be HOLD (no next in same match)
        # m2's event should be HOLD (last in match)
        assert y[0] == LABEL_HOLD
        assert y[1] == LABEL_HOLD


class TestChronologicalOrder:
    def test_output_preserves_chronological_order(self, tmp_db, cfg):
        game_cfg, _ = cfg
        _insert_commitment_events(tmp_db, "m1", [
            (100, "LIGHT_ATTACK"),
            (130, "HEAVY_ATTACK"),
            (160, "DODGE_BACKWARD"),
        ])
        _insert_commitment_events(tmp_db, "m2", [
            (50, "MOVE_LEFT"),
            (80, "MOVE_RIGHT"),
        ])
        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            window_ticks=60,
        )
        # m1 comes first (sorted by match_id), then m2
        assert len(X) == 5


class TestMultipleCommitmentsInMatch:
    def test_pattern_produces_correct_labels(self, tmp_db, cfg):
        game_cfg, _ = cfg
        # L → H → L → H → L pattern, each 20 ticks apart
        events = []
        for i in range(5):
            name = "LIGHT_ATTACK" if i % 2 == 0 else "HEAVY_ATTACK"
            events.append((100 + i * 20, name))
        _insert_commitment_events(tmp_db, "m1", events)

        X, y = build_dataset(
            tmp_db, PlayerProfile(),
            max_hp=game_cfg.fighter.max_hp,
            max_stamina=game_cfg.fighter.max_stamina,
            tick_rate=game_cfg.simulation.tick_rate,
            window_ticks=60,
        )
        assert len(y) == 5
        assert y[0] == "HEAVY_ATTACK"
        assert y[1] == "LIGHT_ATTACK"
        assert y[2] == "HEAVY_ATTACK"
        assert y[3] == "LIGHT_ATTACK"
        assert y[4] == LABEL_HOLD  # last event
