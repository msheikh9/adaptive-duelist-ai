"""Unified CLI entrypoint for all Adaptive Duelist tools.

Usage:
  python3 scripts/cli.py play             — launch the game
  python3 scripts/cli.py simulate         — run headless matches
  python3 scripts/cli.py bulk             — large batch simulation
  python3 scripts/cli.py train            — train sklearn model
  python3 scripts/cli.py explain <id>     — explain a match
  python3 scripts/cli.py analyze          — analyze player tendencies
  python3 scripts/cli.py export <id>      — export match report to JSON
  python3 scripts/cli.py replay-audit     — verify all replay files
  python3 scripts/cli.py db <cmd>         — database maintenance
  python3 scripts/cli.py profile          — performance profiler
  python3 scripts/cli.py benchmark        — run benchmark suite
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PYTHON = sys.executable

COMMANDS: dict[str, tuple[str, str]] = {
    "play":         ("../main.py",           "Launch the game"),
    "simulate":     ("headless_match.py",    "Run headless matches"),
    "bulk":         ("bulk_simulate.py",     "Large batch simulation"),
    "train":        ("train_model.py",       "Train sklearn model"),
    "explain":      ("explain_match.py",     "Explain a match"),
    "analyze":      ("analyze_player.py",    "Analyze player tendencies"),
    "export":       ("export_match_report.py", "Export match report to JSON"),
    "replay-audit": ("replay_audit.py",      "Verify all replay files"),
    "db":           ("db_maintenance.py",    "Database maintenance"),
    "profile":      ("perf_profiler.py",     "Performance profiler"),
    "benchmark":    ("benchmark.py",         "Run benchmark suite"),
    "verify":       ("verify_replay.py",     "Verify a single replay"),
    "player-profile": ("profile_summary.py", "Show profile evolution"),
    "evaluate":     ("evaluate_ai.py",       "Run AI evaluation suite"),
    "check-regression": ("check_regression.py", "Check for AI regressions"),
    "create-baseline": ("create_baseline.py", "Create baseline artifact"),
}


def print_help() -> None:
    print("Adaptive Duelist AI — CLI Tools")
    print(f"{'=' * 50}")
    print(f"\nUsage: python3 scripts/cli.py <command> [args...]\n")
    print("Commands:")
    for cmd, (_, desc) in sorted(COMMANDS.items()):
        print(f"  {cmd:20s} {desc}")
    print(f"\nPass --help to any command for detailed usage.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Run 'python3 scripts/cli.py --help' for available commands.")
        sys.exit(1)

    script, _ = COMMANDS[cmd]
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        print(f"Error: script not found at {script_path}")
        sys.exit(1)

    # Forward remaining args to the target script
    result = subprocess.run(
        [PYTHON, str(script_path)] + sys.argv[2:],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
