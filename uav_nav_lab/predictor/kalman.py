"""Constant-velocity Kalman-filter predictor.

Each tracked obstacle has state ``[pos, vel]`` (length 2·ndim). The process
model is constant velocity with Gaussian acceleration noise; the
measurement model observes *position only* — velocity is estimated from the
sequence of position observations rather than trusted from the obstacle
dict. This makes Kalman a real upgrade over the constant-velocity baseline
when the sensor's velocity field is noisy or, in real deployments, simply
unavailable.

Tracks are associated to incoming observations by greedy nearest-neighbor
matching within ``association_threshold`` (m). Unmatched observations spawn
new tracks; unmatched tracks are dropped (no aging logic — keeps the MVP
small). Bootstrap of the track's initial velocity uses the observation's
velocity field on the first detection, after which KF takes over.

Optional ``delay_compensation`` lets the predictor advance every output
sample by a fixed lead time on top of ``horizon_dts`` — useful when the
*known* sensor latency exceeds zero and you want to project to the current
true time. Setting it equal to the delayed-sensor's ``delay`` is the canonical
"Kalman with known latency" deployment.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PREDICTOR_REGISTRY, Predictor


@PREDICTOR_REGISTRY.register("kalman_velocity")
class KalmanVelocityPredictor(Predictor):
    def __init__(
        self,
        dt: float = 0.2,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.1,
        association_threshold: float = 3.0,
        delay_compensation: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self.q = float(process_noise_std)
        self.r = float(measurement_noise_std)
        self.assoc_thr = float(association_threshold)
        self.delay_comp = float(delay_compensation)
        self._tracks: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "KalmanVelocityPredictor":
        return cls(
            dt=float(cfg.get("dt", 0.2)),
            process_noise_std=float(cfg.get("process_noise_std", 0.5)),
            measurement_noise_std=float(cfg.get("measurement_noise_std", 0.1)),
            association_threshold=float(cfg.get("association_threshold", 3.0)),
            delay_compensation=float(cfg.get("delay_compensation", 0.0)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        self._tracks = []

    def _init_track(self, pos: np.ndarray, vel: np.ndarray, ndim: int) -> dict[str, Any]:
        x = np.concatenate([pos, vel])
        # Loose initial covariance: trust position from sensor, less so velocity
        P = np.diag([self.r * self.r] * ndim + [self.q * self.q] * ndim)
        return {"x": x, "P": P, "ndim": ndim}

    def _predict_step(self, track: dict[str, Any]) -> None:
        """Advance a track's state by self.dt under constant-velocity model."""
        ndim = track["ndim"]
        F = np.eye(2 * ndim)
        F[:ndim, ndim:] = np.eye(ndim) * self.dt
        # Process noise: zero on position, q²·dt on velocity (random walk on velocity)
        Q = np.zeros((2 * ndim, 2 * ndim))
        Q[ndim:, ndim:] = np.eye(ndim) * (self.q * self.q * self.dt)
        track["x"] = F @ track["x"]
        track["P"] = F @ track["P"] @ F.T + Q

    def _update_step(self, track: dict[str, Any], z: np.ndarray) -> None:
        """Update a track with a new position measurement."""
        ndim = track["ndim"]
        H = np.zeros((ndim, 2 * ndim))
        H[:, :ndim] = np.eye(ndim)
        R = np.eye(ndim) * (self.r * self.r)
        y = z - H @ track["x"]
        S = H @ track["P"] @ H.T + R
        K = track["P"] @ H.T @ np.linalg.inv(S)
        track["x"] = track["x"] + K @ y
        I_KH = np.eye(2 * ndim) - K @ H
        track["P"] = I_KH @ track["P"]

    def _associate(self, dyn_obs: list[dict], ndim: int) -> list[int | None]:
        """Greedy NN assignment of new observations to existing tracks."""
        used: set[int] = set()
        assignments: list[int | None] = []
        for obs in dyn_obs:
            p = np.asarray(obs["position"], dtype=float)[:ndim]
            best_i: int | None = None
            best_d2 = np.inf
            for i, t in enumerate(self._tracks):
                if i in used or t["ndim"] != ndim:
                    continue
                d2 = float(np.sum((t["x"][:ndim] - p) ** 2))
                if d2 < best_d2 and d2 <= self.assoc_thr * self.assoc_thr:
                    best_i = i
                    best_d2 = d2
            assignments.append(best_i)
            if best_i is not None:
                used.add(best_i)
        return assignments

    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros((0, len(horizon_dts), 0), dtype=float)
        ndim = len(dynamic_obstacles[0]["position"])

        # 1. KF predict step for all existing tracks (advance by self.dt)
        for t in self._tracks:
            self._predict_step(t)

        # 2. Associate fresh observations to predicted tracks (NN within threshold)
        assignments = self._associate(dynamic_obstacles, ndim)

        # 3. Update each track with its observation; spawn new tracks for
        # unmatched observations. Unmatched old tracks are dropped — the
        # output corresponds 1:1 to the incoming `dynamic_obstacles` list,
        # so the MPC does not need to know that internal track IDs changed.
        new_tracks: list[dict[str, Any]] = []
        for obs, track_idx in zip(dynamic_obstacles, assignments):
            p = np.asarray(obs["position"], dtype=float)[:ndim]
            v = np.asarray(obs["velocity"], dtype=float)[:ndim]
            if track_idx is not None:
                self._update_step(self._tracks[track_idx], p)
                new_tracks.append(self._tracks[track_idx])
            else:
                new_tracks.append(self._init_track(p, v, ndim))
        self._tracks = new_tracks

        # 4. Forward-propagate each track's state over horizon_dts (+ optional
        # delay compensation). Constant-velocity rollout at the KF's current
        # smoothed velocity estimate.
        out = np.empty((len(self._tracks), len(horizon_dts), ndim), dtype=float)
        dts_arr = np.asarray(horizon_dts, dtype=float) + self.delay_comp
        for k, t in enumerate(self._tracks):
            pos = t["x"][:ndim]
            vel = t["x"][ndim:]
            out[k] = pos[None, :] + dts_arr[:, None] * vel[None, :]
        return out
