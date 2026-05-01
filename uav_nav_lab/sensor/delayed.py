"""Fixed-delay + Gaussian-noise sensor, optionally with CV extrapolation.

The buffer is sized in control steps. At t < delay we return the initial
position so the planner has something coherent to chew on. Set
`position_noise_std=0` for a pure-delay sensor.

`extrapolate=True` adds a constant-velocity forward projection of the
stale measurement to estimate the current pose: ``obs = stale_pos +
stale_vel · delay`` where ``stale_vel`` is finite-differenced from the
oldest two samples in the delay buffer. This is what a real onboard
state estimator does — combining a slow / stale visual position with a
faster motion model — and lets the planner act on a fresh ego estimate
even when the camera pipeline is laggy. Without it, the persistent
delay × max_speed cliff is a fixed property of the perception stack.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("delayed")
class DelayedSensor(SensorModel):
    def __init__(
        self,
        delay: float = 0.1,
        dt: float = 0.05,
        noise_std: float = 0.0,
        extrapolate: bool = False,
    ) -> None:
        self.delay = float(delay)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.extrapolate = bool(extrapolate)
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        # Need one extra slot when extrapolating so we can finite-difference
        # the two oldest samples to recover the stale velocity.
        buf_size = self._buffer_len + (1 if self.extrapolate else 0)
        self._buffer: deque[np.ndarray] = deque(maxlen=buf_size)
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DelayedSensor":
        return cls(
            delay=float(cfg.get("delay", 0.1)),
            dt=float(cfg.get("dt", 0.05)),
            noise_std=float(cfg.get("position_noise_std", 0.0)),
            extrapolate=bool(cfg.get("extrapolate", False)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        buf_size = self._buffer_len + (1 if self.extrapolate else 0)
        self._buffer = deque(maxlen=buf_size)

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        true_position = np.asarray(true_position, dtype=float).copy()
        self._buffer.append(true_position)
        if self.extrapolate and len(self._buffer) >= 2:
            # finite-difference the two oldest samples to recover stale
            # velocity, then project the stale position forward by `delay`.
            stale_pos = self._buffer[0]
            stale_vel = (self._buffer[1] - self._buffer[0]) / self.dt
            obs = stale_pos + stale_vel * self.delay
        else:
            obs = self._buffer[0].copy()  # leftmost is the oldest known sample
        if self.noise_std > 0.0:
            obs = obs + self._rng.normal(0.0, self.noise_std, size=obs.shape)
        return obs

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        # Ground-truth dynamics with no per-object delay yet — same convention
        # as the perfect sensor. Position-level delay is what this class models.
        return [dict(d) for d in dynamic_obstacles]
