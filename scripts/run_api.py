#!/usr/bin/env python3
"""Launch the Adaptive Duelist local API server.

Usage:
    python3 scripts/run_api.py [--host HOST] [--port PORT] [--reload]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Adaptive Duelist API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
