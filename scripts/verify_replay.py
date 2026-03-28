"""Verify a replay file: re-simulate from Layer A and compare checksums.

Usage: python3 scripts/verify_replay.py <replay_file> [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from replay.format import compute_frame_data_hash
from replay.replay_player import load_replay, verify_replay, ReplayError


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a replay file")
    parser.add_argument("replay_file", type=Path, help="Path to .replay file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    args = parser.parse_args()

    if not args.replay_file.exists():
        print(f"ERROR: File not found: {args.replay_file}")
        sys.exit(1)

    game_cfg, _, _ = load_config()

    # Load replay
    try:
        replay = load_replay(args.replay_file)
    except ReplayError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to parse replay: {e}")
        sys.exit(1)

    header = replay.header
    print(f"Replay: {args.replay_file.name}")
    print(f"  Format version: {header.format_version}")
    print(f"  Engine version: {header.engine_version}")
    print(f"  Match ID:       {header.match_id[:8]}...")
    print(f"  Total ticks:    {header.total_ticks}")
    print(f"  RNG seed:       {header.rng_seed}")
    print(f"  Winner:         {['PLAYER', 'AI', 'DRAW'][header.winner]}")
    print(f"  Commitments:    {len(replay.commitments)}")
    print(f"  Checksums:      {len(replay.checksums)}")

    # Check frame data compatibility
    current_fd_hash = compute_frame_data_hash(game_cfg)
    if header.frame_data_hash and header.frame_data_hash != current_fd_hash:
        print(f"\nWARNING: Frame data hash mismatch!")
        print(f"  Replay:  {header.frame_data_hash}")
        print(f"  Current: {current_fd_hash}")
        print("  Replay may produce different results with current config.")

    # Verify
    print(f"\nRe-simulating {header.total_ticks} ticks...")
    result = verify_replay(replay, game_cfg)

    if result.error:
        print(f"\nERROR during verification: {result.error}")
        sys.exit(1)

    print(f"\nChecksum verification:")
    print(f"  Total checksums: {result.total_checksums}")
    print(f"  Failed:          {result.failed_checksums}")
    print(f"  Final state:     {'MATCH' if result.final_state_match else 'MISMATCH'}")

    if args.verbose and result.checksum_failures:
        print(f"\n  Failed checksum ticks:")
        for tick, expected, actual in result.checksum_failures:
            print(f"    Tick {tick}: expected {expected.hex()[:16]}... got {actual.hex()[:16]}...")

    if result.passed:
        print(f"\nRESULT: PASS")
        sys.exit(0)
    else:
        print(f"\nRESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
