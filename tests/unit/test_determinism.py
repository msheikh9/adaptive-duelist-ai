"""The same seed must produce the same match — including across processes.

This guards a bug that no in-process test can see. `PHASE_1_COMMITMENTS` is a
`frozenset` of enum members; `Enum.__hash__` hashes the member *name*, and
CPython randomises string hashing per process. So the frozenset's iteration order
genuinely differs between runs. Two policies built weighted candidate lists by
iterating it and handed them to an RNG, which made identical seeds diverge across
processes while every single-process test stayed green.

The subprocess test below is the only kind that catches that, because PYTHONHASHSEED
is fixed for the lifetime of an interpreter and cannot be changed from inside it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from game.combat.actions import (
    PHASE_1_COMMITMENTS,
    PHASE_1_COMMITMENTS_ORDERED,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Runs one match at a fixed seed and prints a compact fingerprint of the result.
_MATCH_FINGERPRINT = """
import os, sys
sys.path.insert(0, %r)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from config.config_loader import load_config
from tests.fixtures.headless_engine import HeadlessMatch
game_cfg, _, _ = load_config()
m = HeadlessMatch(game_cfg, rng_seed=11)
m.run_until_end(max_ticks=6000)
s = m.state
print("%%s|%%s|%%s|%%s" %% (s.tick_id, s.winner, s.player.hp, s.ai.hp))
""" % str(PROJECT_ROOT)


def _run_with_hash_seed(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    proc = subprocess.run(
        [sys.executable, "-c", _MATCH_FINGERPRINT],
        capture_output=True, text=True, env=env, timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    # Ignore the pygame banner; the fingerprint is the last non-empty line.
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return lines[-1]


class TestCommitmentOrdering:
    """Structural guard — cheap, and fails immediately on a regression."""

    def test_ordered_covers_the_same_set(self):
        assert set(PHASE_1_COMMITMENTS_ORDERED) == set(PHASE_1_COMMITMENTS)
        assert len(PHASE_1_COMMITMENTS_ORDERED) == len(PHASE_1_COMMITMENTS)

    def test_ordered_is_a_sequence_not_a_set(self):
        """A set would reintroduce per-process iteration order."""
        assert isinstance(PHASE_1_COMMITMENTS_ORDERED, (list, tuple))

    def test_ordered_is_sorted_by_value(self):
        values = [c.value for c in PHASE_1_COMMITMENTS_ORDERED]
        assert values == sorted(values), (
            "ordering must be derived from a stable key, not from set iteration"
        )

    def test_policies_do_not_iterate_the_frozenset(self):
        """Catches the regression at the source, without spawning a process.

        Iterating PHASE_1_COMMITMENTS to build an RNG candidate list is the exact
        defect; membership tests (`x in PHASE_1_COMMITMENTS`) are fine. Uses AST
        rather than text matching so comments and docstrings mentioning the name
        do not trip it.
        """
        import ast

        offenders: list[str] = []
        for rel in ("game/entities/ai_fighter.py",
                    "ai/training/scripted_opponent.py",
                    "game/simulation_step.py",
                    "ai/layers/tactical_planner.py"):
            tree = ast.parse((PROJECT_ROOT / rel).read_text())
            for node in ast.walk(tree):
                # Collect every "iterated over" expression: for-loops and the
                # generator clauses of comprehensions.
                iters = []
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    iters.append(node.iter)
                elif isinstance(node, (ast.ListComp, ast.SetComp,
                                       ast.DictComp, ast.GeneratorExp)):
                    iters.extend(gen.iter for gen in node.generators)
                for it in iters:
                    if (isinstance(it, ast.Name)
                            and it.id == "PHASE_1_COMMITMENTS"):
                        offenders.append(f"{rel}:{it.lineno}")
        assert not offenders, (
            "these iterate the frozenset, whose order varies per process; "
            "use PHASE_1_COMMITMENTS_ORDERED: " + ", ".join(offenders)
        )


class TestCrossProcessDeterminism:
    def test_same_seed_same_match_in_one_process(self):
        from config.config_loader import load_config
        from tests.fixtures.headless_engine import HeadlessMatch

        game_cfg, _, _ = load_config()
        runs = []
        for _ in range(2):
            m = HeadlessMatch(game_cfg, rng_seed=11)
            m.run_until_end(max_ticks=6000)
            runs.append((m.state.tick_id, m.state.winner,
                         m.state.player.hp, m.state.ai.hp))
        assert runs[0] == runs[1]

    @pytest.mark.parametrize("hash_seeds", [("0", "1"), ("0", "98765")])
    def test_same_seed_same_match_across_hash_seeds(self, hash_seeds):
        """Identical rng_seed must give an identical match under any hash seed."""
        a, b = (_run_with_hash_seed(hs) for hs in hash_seeds)
        assert a == b, (
            f"same rng_seed produced different matches under "
            f"PYTHONHASHSEED={hash_seeds[0]} vs {hash_seeds[1]}: {a} vs {b}. "
            "Something order-dependent is feeding the RNG."
        )
