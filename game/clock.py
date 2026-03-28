"""Fixed-timestep game clock.

Manages the fixed simulation tick rate and tracks frame timing
for the render loop. Uses an accumulator pattern: wall time is
accumulated and consumed in fixed-size simulation steps.
"""

from __future__ import annotations

import time


class GameClock:
    """Fixed-timestep clock with accumulator."""

    def __init__(self, tick_rate: int) -> None:
        self._tick_rate = tick_rate
        self._tick_duration = 1.0 / tick_rate  # seconds per tick
        self._last_time = 0.0
        self._accumulator = 0.0
        self._started = False

    @property
    def tick_rate(self) -> int:
        return self._tick_rate

    @property
    def tick_duration_ms(self) -> float:
        return self._tick_duration * 1000.0

    def start(self) -> None:
        self._last_time = time.perf_counter()
        self._accumulator = 0.0
        self._started = True

    def update(self) -> int:
        """Advance the clock and return the number of simulation ticks to run.

        Call this once per frame. Returns how many fixed-timestep ticks
        should be simulated this frame (usually 0 or 1, occasionally 2
        if the frame took longer than expected).
        """
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now

        # Clamp to prevent spiral of death (e.g., after a debugger pause)
        max_frame_time = self._tick_duration * 5
        if elapsed > max_frame_time:
            elapsed = max_frame_time

        self._accumulator += elapsed

        ticks = 0
        while self._accumulator >= self._tick_duration:
            self._accumulator -= self._tick_duration
            ticks += 1

        return ticks
