"""Tests for TickSnapshot and SemanticEvent data contracts."""

from __future__ import annotations

from data.events import EventType, SemanticEvent
from data.serializer import semantic_event_to_row, to_dict
from data.tick_snapshot import TICK_SNAPSHOT_SIZE, TickSnapshot
from game.combat.actions import Actor, CombatCommitment, FSMState, SpacingZone
from game.state import SimulationState


class TestTickSnapshot:
    def test_struct_size_is_34_bytes(self):
        assert TICK_SNAPSHOT_SIZE == 34

    def test_pack_unpack_roundtrip(self):
        snap = TickSnapshot(
            tick_id=100,
            player_x=60000, player_y=30000,
            ai_x=90000, ai_y=30000,
            player_hp=180, ai_hp=200,
            player_stamina=75, ai_stamina=100,
            player_fsm=FSMState.IDLE.value, ai_fsm=FSMState.ATTACK_ACTIVE.value,
            player_fsm_frame=0, ai_fsm_frame=3,
            distance=30000,
        )
        packed = snap.pack()
        assert len(packed) == 34
        unpacked = TickSnapshot.unpack(packed)
        assert unpacked == snap

    def test_from_state(self):
        state = SimulationState()
        state.tick_id = 42
        state.player.x = 10000
        state.player.y = 30000
        state.player.hp = 190
        state.player.stamina = 80
        state.player.fsm_state = FSMState.MOVING
        state.player.fsm_frames_remaining = 0
        state.ai.x = 50000
        state.ai.y = 30000
        state.ai.hp = 200
        state.ai.stamina = 100
        state.ai.fsm_state = FSMState.IDLE
        state.ai.fsm_frames_remaining = 0

        snap = TickSnapshot.from_state(state)
        assert snap.tick_id == 42
        assert snap.player_x == 10000
        assert snap.ai_x == 50000
        assert snap.distance == 40000
        assert snap.player_hp == 190

    def test_fsm_state_recovery(self):
        snap = TickSnapshot(
            tick_id=0,
            player_x=0, player_y=0, ai_x=0, ai_y=0,
            player_hp=200, ai_hp=200,
            player_stamina=100, ai_stamina=100,
            player_fsm=FSMState.HITSTUN.value,
            ai_fsm=FSMState.ATTACK_RECOVERY.value,
            player_fsm_frame=5, ai_fsm_frame=10,
            distance=0,
        )
        assert snap.fsm_state_player() == FSMState.HITSTUN
        assert snap.fsm_state_ai() == FSMState.ATTACK_RECOVERY


class TestSemanticEvent:
    def test_creation(self):
        event = SemanticEvent(
            event_type=EventType.HIT_LANDED,
            match_id="m1",
            tick_id=100,
            actor=Actor.PLAYER,
            commitment=CombatCommitment.LIGHT_ATTACK,
            opponent_fsm_state=FSMState.IDLE,
            spacing_zone=SpacingZone.CLOSE,
            actor_hp=180,
            opponent_hp=200,
            actor_stamina=85,
            opponent_stamina=100,
            damage_dealt=8,
        )
        assert event.event_type == EventType.HIT_LANDED
        assert event.damage_dealt == 8

    def test_to_row(self):
        event = SemanticEvent(
            event_type=EventType.COMMITMENT_START,
            match_id="m1",
            tick_id=50,
            actor=Actor.AI,
            commitment=CombatCommitment.HEAVY_ATTACK,
            actor_hp=200,
            opponent_hp=180,
            actor_stamina=70,
            opponent_stamina=90,
        )
        row = semantic_event_to_row(event)
        assert row[0] == "COMMITMENT_START"
        assert row[1] == "m1"
        assert row[2] == 50
        assert row[3] == "AI"
        assert row[4] == "HEAVY_ATTACK"


class TestSerializer:
    def test_to_dict_with_enums(self):
        event = SemanticEvent(
            event_type=EventType.MATCH_START,
            match_id="m1",
            tick_id=0,
            actor=Actor.PLAYER,
            actor_hp=200,
            opponent_hp=200,
            actor_stamina=100,
            opponent_stamina=100,
        )
        d = to_dict(event)
        assert d["event_type"] == "MATCH_START"
        assert d["actor"] == "PLAYER"
        assert d["commitment"] is None
        assert d["tick_id"] == 0
