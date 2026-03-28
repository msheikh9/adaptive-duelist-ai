"""Tests for replay binary format definitions."""

from __future__ import annotations

import pytest

from game.combat.actions import Actor, CombatCommitment, FSMState
from game.state import ArenaState, FighterState, MatchStatus, SimulationState
from replay.format import (
    COMMITMENT_FORMAT,
    COMMITMENT_SIZE,
    CHECKSUM_FORMAT,
    CHECKSUM_SIZE,
    HEADER_SIZE,
    MAGIC,
    ChecksumRecord,
    CommitmentRecord,
    ReplayHeader,
    compute_frame_data_hash,
    compute_state_hash,
    deserialize_initial_state,
    serialize_initial_state,
)
from config.config_loader import load_config


@pytest.fixture
def cfg():
    game_cfg, _, _ = load_config()
    return game_cfg


@pytest.fixture
def sample_state():
    return SimulationState(
        tick_id=0,
        rng_seed=12345,
        match_status=MatchStatus.ACTIVE,
        player=FighterState(
            x=40000, y=30000, hp=200, stamina=100, facing=1,
        ),
        ai=FighterState(
            x=80000, y=30000, hp=200, stamina=100, facing=-1,
        ),
        arena=ArenaState(width_sub=120000, height_sub=40000, ground_y_sub=30000),
    )


class TestCommitmentRecord:
    def test_size_is_6_bytes(self):
        assert COMMITMENT_SIZE == 6

    def test_pack_unpack_roundtrip(self):
        rec = CommitmentRecord(
            tick_id=42, actor=Actor.PLAYER,
            commitment=CombatCommitment.LIGHT_ATTACK,
        )
        packed = rec.pack()
        assert len(packed) == COMMITMENT_SIZE
        restored = CommitmentRecord.unpack(packed)
        assert restored.tick_id == 42
        assert restored.actor == Actor.PLAYER
        assert restored.commitment == CombatCommitment.LIGHT_ATTACK

    def test_all_commitment_types(self):
        for commitment in [CombatCommitment.MOVE_LEFT, CombatCommitment.MOVE_RIGHT,
                           CombatCommitment.LIGHT_ATTACK, CombatCommitment.HEAVY_ATTACK,
                           CombatCommitment.DODGE_BACKWARD]:
            rec = CommitmentRecord(tick_id=100, actor=Actor.AI, commitment=commitment)
            restored = CommitmentRecord.unpack(rec.pack())
            assert restored.commitment == commitment

    def test_large_tick_id(self):
        rec = CommitmentRecord(
            tick_id=2**32 - 1, actor=Actor.PLAYER,
            commitment=CombatCommitment.MOVE_LEFT,
        )
        restored = CommitmentRecord.unpack(rec.pack())
        assert restored.tick_id == 2**32 - 1


class TestChecksumRecord:
    def test_size_is_20_bytes(self):
        assert CHECKSUM_SIZE == 20

    def test_pack_unpack_roundtrip(self):
        md5 = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
        rec = ChecksumRecord(tick_id=300, state_md5=md5)
        packed = rec.pack()
        assert len(packed) == CHECKSUM_SIZE
        restored = ChecksumRecord.unpack(packed)
        assert restored.tick_id == 300
        assert restored.state_md5 == md5


class TestReplayHeader:
    def test_header_is_256_bytes(self):
        header = ReplayHeader()
        packed = header.pack()
        assert len(packed) == HEADER_SIZE

    def test_pack_unpack_roundtrip(self):
        header = ReplayHeader(
            format_version=1,
            engine_version="0.2.0",
            config_hash="abc123",
            frame_data_hash="def456",
            rng_seed=99999,
            match_id="12345678-1234-1234-1234-123456789abc",
            total_ticks=5000,
            winner=1,
            checksum_interval=300,
        )
        packed = header.pack()
        restored = ReplayHeader.unpack(packed)

        assert restored.format_version == 1
        assert restored.engine_version == "0.2.0"
        assert restored.config_hash == "abc123"
        assert restored.frame_data_hash == "def456"
        assert restored.rng_seed == 99999
        assert restored.match_id == "12345678-1234-1234-1234-123456789abc"
        assert restored.total_ticks == 5000
        assert restored.winner == 1
        assert restored.checksum_interval == 300

    def test_header_too_short_raises(self):
        with pytest.raises(ValueError, match="Header too short"):
            ReplayHeader.unpack(b"\x00" * 100)

    def test_default_values(self):
        header = ReplayHeader()
        assert header.format_version == 1
        assert header.winner == 2  # draw/none
        assert header.total_ticks == 0


class TestStateSerialization:
    def test_serialize_deserialize_roundtrip(self, sample_state):
        data = serialize_initial_state(sample_state)
        restored = deserialize_initial_state(data)

        assert restored.tick_id == sample_state.tick_id
        assert restored.rng_seed == sample_state.rng_seed
        assert restored.match_status == sample_state.match_status
        assert restored.player.x == sample_state.player.x
        assert restored.player.y == sample_state.player.y
        assert restored.player.hp == sample_state.player.hp
        assert restored.player.stamina == sample_state.player.stamina
        assert restored.player.facing == sample_state.player.facing
        assert restored.ai.x == sample_state.ai.x
        assert restored.ai.hp == sample_state.ai.hp
        assert restored.ai.facing == sample_state.ai.facing
        assert restored.arena.width_sub == sample_state.arena.width_sub
        assert restored.arena.ground_y_sub == sample_state.arena.ground_y_sub

    def test_serialized_is_json_bytes(self, sample_state):
        data = serialize_initial_state(sample_state)
        import json
        parsed = json.loads(data.decode("utf-8"))
        assert "player" in parsed
        assert "ai" in parsed
        assert "arena" in parsed


class TestStateHash:
    def test_hash_is_16_bytes(self, sample_state):
        h = compute_state_hash(sample_state)
        assert len(h) == 16

    def test_same_state_same_hash(self, sample_state):
        h1 = compute_state_hash(sample_state)
        h2 = compute_state_hash(sample_state)
        assert h1 == h2

    def test_different_state_different_hash(self, sample_state):
        h1 = compute_state_hash(sample_state)
        sample_state.player.x += 100
        h2 = compute_state_hash(sample_state)
        assert h1 != h2

    def test_hp_change_changes_hash(self, sample_state):
        h1 = compute_state_hash(sample_state)
        sample_state.player.hp -= 10
        h2 = compute_state_hash(sample_state)
        assert h1 != h2


class TestFrameDataHash:
    def test_returns_32_char_hex(self, cfg):
        h = compute_frame_data_hash(cfg)
        assert len(h) == 32
        int(h, 16)  # valid hex

    def test_deterministic(self, cfg):
        h1 = compute_frame_data_hash(cfg)
        h2 = compute_frame_data_hash(cfg)
        assert h1 == h2
