"""Record a real match, then verify it re-simulates bit-identically.

This is the regression test for the class of bug that kept `replay-audit` at
0/14: *state mutated outside the recorded commitment stream*. Any policy that
changes the simulation without going through a `CombatCommitment` will make the
replayed match drift from the recorded one, and these tests fail.

Concretely, each of the following makes at least one test here fail:

  - T2's guard release assigning `fsm_state = IDLE` directly instead of entering
    `BLOCK_RELEASE` (the replayed AI stays stuck in BLOCKING forever).
  - The engine's periodic AI auto-shoot not being recorded.
  - The replay driver iterating `range(total_ticks)` instead of
    `total_ticks + 1`, which skips the final tick and mismatches the final hash.
  - Any caller re-introducing its own partial copy of the tick loop.

Note what these tests deliberately do *not* cover: removing a subsystem from
`step_simulation` itself. Recorder and replayer share that function, so they
would be wrong together and still agree. Per-subsystem coverage lives in
tests/unit/test_simulation_step.py.
"""

from __future__ import annotations

import pytest

from ai.layers.tactical_planner import AITier
from config.config_loader import load_config
from evaluation.match_runner import _run_match
from data.db import Database
from data.migrations.migration_runner import run_migrations
from replay.replay_player import load_replay, verify_replay


@pytest.fixture
def cfgs():
    game_cfg, ai_cfg, _ = load_config()
    return game_cfg, ai_cfg


@pytest.fixture
def eval_db(tmp_path):
    db = Database(tmp_path / "roundtrip.db")
    db.connect()
    run_migrations(db)
    yield db
    db.close()


def _record_and_verify(cfgs, eval_db, tmp_path, tier, seed):
    """Run one match with recording on, then verify the replay it wrote."""
    game_cfg, ai_cfg = cfgs
    out = tmp_path / "replays"
    match_id = f"rt-{tier.name}-{seed}"

    result = _run_match(
        game_cfg, ai_cfg, eval_db, seed, tier,
        max_ticks=20000, match_id=match_id, record_to=out,
    )

    path = out / f"{match_id}.replay"
    assert path.exists(), "recording produced no replay file"

    replay = load_replay(path)
    verification = verify_replay(replay, game_cfg)
    return result, replay, verification


@pytest.mark.parametrize("tier", [
    AITier.T0_BASELINE,
    AITier.T1_MARKOV_ONLY,
    AITier.T2_FULL_ADAPTIVE,
])
def test_recorded_match_reverifies(cfgs, eval_db, tmp_path, tier):
    """A recorded match must re-simulate to identical checksums, every tier.

    T2 is the important case — it is the only tier with the frame-perfect layer,
    and therefore the only one that releases guard mid-match.
    """
    _, replay, v = _record_and_verify(cfgs, eval_db, tmp_path, tier, seed=0)

    assert v.error is None, f"verification raised: {v.error}"
    assert v.total_checksums > 0, "replay recorded no checksums to verify against"
    assert v.failed_checksums == 0, (
        f"{v.failed_checksums}/{v.total_checksums} checksums diverged; "
        f"first at tick "
        f"{v.checksum_failures[0][0] if v.checksum_failures else '?'}. "
        "Something mutated simulation state without recording a commitment."
    )
    assert v.final_state_match, (
        "final-state hash mismatched even though periodic checksums agreed — "
        "the replay driver is probably not simulating the last tick"
    )
    assert v.passed


def test_recorded_match_reproduces_the_outcome(cfgs, eval_db, tmp_path):
    """Replaying must reach the same winner, HP and tick count as the recording."""
    result, replay, v = _record_and_verify(
        cfgs, eval_db, tmp_path, AITier.T2_FULL_ADAPTIVE, seed=3)
    assert v.passed

    from replay.replay_player import replay_match
    final = replay_match(replay, cfgs[0])

    assert final.winner == result["winner"]
    assert final.player.hp == result["player_hp"]
    assert final.ai.hp == result["ai_hp"]


def test_t2_releases_guard_through_the_commitment_stream(cfgs, eval_db, tmp_path):
    """T2's guard release must appear in the replay as a BLOCK_RELEASE record.

    Guarding against a regression to direct `fsm_state` assignment, which is
    invisible to the recorder. Without this the only symptom is a checksum
    mismatch somewhere deep in a match, which is far harder to diagnose.
    """
    from game.combat.actions import Actor, CombatCommitment

    # Seed 0 is known to involve the AI blocking and then releasing.
    _, replay, v = _record_and_verify(
        cfgs, eval_db, tmp_path, AITier.T2_FULL_ADAPTIVE, seed=0)
    assert v.passed

    blocks = [c for c in replay.commitments
              if c.commitment == CombatCommitment.BLOCK_START]
    releases = [c for c in replay.commitments
                if c.commitment == CombatCommitment.BLOCK_RELEASE]

    assert blocks, "expected T2 to block at least once in this match"
    assert releases, (
        "T2 blocked but never recorded a BLOCK_RELEASE — guard release is "
        "bypassing the commitment stream"
    )
    assert all(c.actor == Actor.AI for c in releases)


def test_auto_shoot_is_recorded(cfgs, eval_db, tmp_path):
    """The periodic AI shoot policy must reach the commitment stream.

    It fires at tick % 180 == 90, mutating state via attempt_commitment. It was
    originally applied without a matching record_commitment call.
    """
    from game.combat.actions import Actor, CombatCommitment

    _, replay, v = _record_and_verify(
        cfgs, eval_db, tmp_path, AITier.T0_BASELINE, seed=0)
    assert v.passed
    assert replay.header.total_ticks > 90, "match too short to reach the trigger"

    shots = [c for c in replay.commitments
             if c.actor == Actor.AI
             and c.commitment == CombatCommitment.SHOOT_INSTANT]
    assert shots, "AI auto-shoot fired but was never recorded"
