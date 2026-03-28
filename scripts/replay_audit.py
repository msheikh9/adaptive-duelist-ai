"""Batch replay verification and integrity audit.

Scans a replay directory, verifies all replays, and reports
corrupt / incompatible / missing metadata files.

Usage: python3 scripts/replay_audit.py [--dir PATH] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config.config_loader import load_config
from replay.format import compute_frame_data_hash, ENGINE_VERSION
from replay.replay_player import load_replay, verify_replay


def audit_replay(path: Path, game_cfg) -> dict:
    """Audit a single replay file. Returns a result dict."""
    result: dict = {
        "file": str(path.name),
        "size_bytes": path.stat().st_size,
    }

    try:
        replay = load_replay(path)
    except Exception as e:
        result["status"] = "CORRUPT"
        result["error"] = str(e)
        return result

    header = replay.header
    result["match_id"] = header.match_id
    result["total_ticks"] = header.total_ticks
    result["winner"] = {0: "PLAYER", 1: "AI", 2: "DRAW"}.get(
        header.winner, "UNKNOWN")
    result["engine_version"] = header.engine_version
    result["rng_seed"] = header.rng_seed
    result["commitments"] = len(replay.commitments)
    result["checksums"] = len(replay.checksums)

    # Check engine version compatibility
    if header.engine_version != ENGINE_VERSION:
        result["version_warning"] = (
            f"Replay engine {header.engine_version} != "
            f"current {ENGINE_VERSION}"
        )

    # Check frame data hash
    current_hash = compute_frame_data_hash(game_cfg)
    if header.frame_data_hash != current_hash:
        result["frame_data_warning"] = "Frame data hash mismatch"

    # Check metadata
    has_metadata = replay.metadata_raw is not None and len(replay.metadata_raw) > 0
    result["has_metadata"] = has_metadata
    if has_metadata:
        try:
            json.loads(replay.metadata_raw)
            result["metadata_valid"] = True
        except (json.JSONDecodeError, TypeError):
            result["metadata_valid"] = False

    # Full verification
    try:
        vr = verify_replay(replay, game_cfg)
        result["verification_passed"] = vr.passed
        result["checksum_failures"] = vr.failed_checksums
        result["final_state_match"] = vr.final_state_match
        if vr.error:
            result["verification_error"] = vr.error
        result["status"] = "OK" if vr.passed else "FAILED"
    except Exception as e:
        result["status"] = "VERIFY_ERROR"
        result["verification_error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch replay verification audit")
    parser.add_argument("--dir", default="replays",
                        help="Replay directory to scan")
    parser.add_argument("--output", "-o", default=None,
                        help="Export results to JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-file details")
    args = parser.parse_args()

    replay_dir = Path(args.dir)
    if not replay_dir.is_dir():
        print(f"Error: {replay_dir} is not a directory")
        sys.exit(1)

    files = sorted(replay_dir.glob("*.replay"))
    if not files:
        print(f"No .replay files found in {replay_dir}")
        sys.exit(0)

    game_cfg, _, _ = load_config()
    results: list[dict] = []
    counts = {"OK": 0, "FAILED": 0, "CORRUPT": 0, "VERIFY_ERROR": 0}

    print(f"Auditing {len(files)} replay files in {replay_dir}...")
    t0 = time.perf_counter()

    for path in files:
        r = audit_replay(path, game_cfg)
        results.append(r)
        status = r.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

        if args.verbose:
            warnings = ""
            if "version_warning" in r:
                warnings += f" [{r['version_warning']}]"
            if "frame_data_warning" in r:
                warnings += f" [frame data mismatch]"
            print(f"  {r['file']:40s}  {status:12s}  "
                  f"ticks={r.get('total_ticks', '?'):>6}  "
                  f"winner={r.get('winner', '?'):>6}{warnings}")

    elapsed = time.perf_counter() - t0

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Replay Audit Summary")
    print(f"  Files scanned:  {len(files)}")
    print(f"  OK:             {counts.get('OK', 0)}")
    print(f"  Failed:         {counts.get('FAILED', 0)}")
    print(f"  Corrupt:        {counts.get('CORRUPT', 0)}")
    print(f"  Verify errors:  {counts.get('VERIFY_ERROR', 0)}")
    print(f"  Elapsed:        {elapsed:.2f}s")
    print(f"{'=' * 60}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2, default=str))
        print(f"Detailed results exported to {args.output}")

    # Exit with error if any failures
    if counts.get("FAILED", 0) + counts.get("CORRUPT", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
