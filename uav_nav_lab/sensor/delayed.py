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
        velocity_window: int = 1,
    ) -> None:
        self.delay = float(delay)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.extrapolate = bool(extrapolate)
        # When >1, averages the finite-difference velocity over the last
        # `velocity_window` consecutive sample pairs to reduce acceleration
        # noise. Lengthens the velocity estimate's effective lag.
        self.velocity_window = max(1, int(velocity_window))
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        # Need extra slots when extrapolating: one to compute the basic FD
        # velocity, plus `velocity_window-1` more to average over a window.
        extra = (self.velocity_window if self.extrapolate else 0)
        buf_size = self._buffer_len + extra
        self._buffer: deque[np.ndarray] = deque(maxlen=buf_size)
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DelayedSensor":
        return cls(
            delay=float(cfg.get("delay", 0.1)),
            dt=float(cfg.get("dt", 0.05)),
            noise_std=float(cfg.get("position_noise_std", 0.0)),
            extrapolate=bool(cfg.get("extrapolate", False)),
            velocity_window=int(cfg.get("velocity_window", 1)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        extra = (self.velocity_window if self.extrapolate else 0)
        self._buffer = deque(maxlen=self._buffer_len + extra)

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        true_position = np.asarray(true_position, dtype=float).copy()
        self._buffer.append(true_position)
        if self.extrapolate and len(self._buffer) >= 2:
            # Average the finite-differenced velocity over up to
            # `velocity_window` consecutive sample pairs from the start of
            # the buffer to suppress acceleration noise. With window=1 this
            # reduces to the simple two-sample FD used previously.
            n_pairs = min(self.velocity_window, len(self._buffer) - 1)
            stale_pos = self._buffer[0]
            v_acc = np.zeros_like(stale_pos)
            for i in range(n_pairs):
                v_acc = v_acc + (self._buffer[i + 1] - self._buffer[i])
            stale_vel = v_acc / (n_pairs * self.dt)
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
