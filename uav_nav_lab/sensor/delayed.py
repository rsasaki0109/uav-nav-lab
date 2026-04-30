"""Fixed-delay + Gaussian-noise sensor.

The buffer is sized in control steps. At t < delay we return the initial
position so the planner has something coherent to chew on. Set
`position_noise_std=0` for a pure-delay sensor.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("delayed")
class DelayedSensor(SensorModel):
    def __init__(self, delay: float = 0.1, dt: float = 0.05, noise_std: float = 0.0) -> None:
        self.delay = float(delay)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        self._buffer: deque[np.ndarray] = deque(maxlen=self._buffer_len)
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DelayedSensor":
        return cls(
            delay=float(cfg.get("delay", 0.1)),
            dt=float(cfg.get("dt", 0.05)),
            noise_std=float(cfg.get("position_noise_std", 0.0)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._buffer = deque(maxlen=self._buffer_len)

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        true_position = np.asarray(true_position, dtype=float).copy()
        self._buffer.append(true_position)
        # Return the oldest entry only once the buffer is full; before that,
        # echo the very first sample (i.e. effectively the start state).
        if len(self._buffer) >= self._buffer_len:
            obs = self._buffer[0].copy()
        else:
            obs = self._buffer[0].copy()  # always the oldest known sample
        if self.noise_std > 0.0:
            obs = obs + self._rng.normal(0.0, self.noise_std, size=obs.shape)
        return obs

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        # Ground-truth dynamics with no per-object delay yet — same convention
        # as the perfect sensor. Position-level delay is what this class models.
        return [dict(d) for d in dynamic_obstacles]
