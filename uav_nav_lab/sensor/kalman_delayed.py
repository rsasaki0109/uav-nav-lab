"""Delayed sensor with a 1-D Kalman estimator on the ego stream.

The previous-commit `delayed` sensor with `extrapolate=true` and
`velocity_window=N` shows that simple moving-average velocity smoothing
already lifts most of the perception-latency cliff — but the optimum
window depends on the speed regime (low speed prefers no smoothing,
high speed needs ~5 samples). A Kalman filter with an explicit
constant-velocity dynamics model and tunable process / measurement noise
should provide adaptive smoothing across speed regimes without per-deployment
window tuning.

State: [pos, vel] (length 2·ndim). Process model: CV with random-walk
velocity (Q on velocity only, zero on position). Measurement model:
position only. Each call:

  1. add a (possibly noisy) sample to a delay buffer
  2. KF predict step over dt, update with the *stale* sample at the
     buffer head (delayed by `delay` s)
  3. forward-propagate the KF's smoothed state by `delay` to estimate
     the current ego pose.

Set ``process_noise_std`` low to trust the CV model (smoother at low
speed, slower to react to maneuvers). Set ``measurement_noise_std``
low to trust each measurement (less smoothing under noise).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("kalman_delayed")
class KalmanDelayedSensor(SensorModel):
    def __init__(
        self,
        delay: float = 0.1,
        dt: float = 0.05,
        noise_std: float = 0.0,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.05,
    ) -> None:
        self.delay = float(delay)
        self.dt = float(dt)
        self.noise_std = float(noise_std)  # additive output noise (sim only)
        self.q = float(process_noise_std)
        self.r = float(measurement_noise_std)
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        self._buffer: deque[np.ndarray] = deque(maxlen=self._buffer_len)
        self._rng = np.random.default_rng()
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._ndim: int | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "KalmanDelayedSensor":
        return cls(
            delay=float(cfg.get("delay", 0.1)),
            dt=float(cfg.get("dt", 0.05)),
            noise_std=float(cfg.get("position_noise_std", 0.0)),
            process_noise_std=float(cfg.get("process_noise_std", 0.5)),
            measurement_noise_std=float(cfg.get("measurement_noise_std", 0.05)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._buffer = deque(maxlen=self._buffer_len)
        self._x = None
        self._P = None
        self._ndim = None

    def _init_kf(self, z: np.ndarray) -> None:
        ndim = z.shape[0]
        self._ndim = ndim
        self._x = np.concatenate([z, np.zeros(ndim)])
        # initial covariance: tight on position (we just observed it),
        # loose on velocity (unknown)
        self._P = np.diag([self.r * self.r] * ndim + [self.q * self.q] * ndim)

    def _kf_predict(self) -> None:
        ndim = self._ndim
        F = np.eye(2 * ndim)
        F[:ndim, ndim:] = np.eye(ndim) * self.dt
        # Random-walk on velocity: Q only on the velocity block
        Q = np.zeros((2 * ndim, 2 * ndim))
        Q[ndim:, ndim:] = np.eye(ndim) * (self.q * self.q * self.dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def _kf_update(self, z: np.ndarray) -> None:
        ndim = self._ndim
        H = np.zeros((ndim, 2 * ndim))
        H[:, :ndim] = np.eye(ndim)
        R = np.eye(ndim) * (self.r * self.r)
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        I_KH = np.eye(2 * ndim) - K @ H
        self._P = I_KH @ self._P

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        true_position = np.asarray(true_position, dtype=float).copy()
        # Add measurement noise *once* per call and store in the delay
        # buffer, so the KF sees a self-consistent noisy stream rather
        # than re-randomized output noise on each access.
        if self.noise_std > 0.0:
            noisy = true_position + self._rng.normal(0.0, self.noise_std, size=true_position.shape)
        else:
            noisy = true_position
        self._buffer.append(noisy)

        # Stale measurement at the head of the buffer (delayed by ~delay s)
        stale_z = self._buffer[0]

        if self._x is None:
            self._init_kf(stale_z)
        else:
            self._kf_predict()
            self._kf_update(stale_z)

        # Forward-propagate the KF state by `delay` to estimate the
        # current ego pose using the smoothed velocity estimate.
        ndim = self._ndim
        return self._x[:ndim] + self._x[ndim:] * self.delay
