"""Imperfect velocity-estimator predictor.

Models a downstream tracker (Kalman / optical-flow / radar) whose velocity
estimate has finite precision. For each replan we sample a velocity bias
~ N(0, σ²·I) per obstacle and propagate constant-velocity from there.

Useful as an ablation: as `velocity_noise_std` rises, the planner's
predicted obstacle trajectories diverge from reality and collisions go up.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PREDICTOR_REGISTRY, Predictor


@PREDICTOR_REGISTRY.register("noisy_velocity")
class NoisyVelocityPredictor(Predictor):
    def __init__(self, velocity_noise_std: float = 0.5) -> None:
        self.velocity_noise_std = float(velocity_noise_std)
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "NoisyVelocityPredictor":
        return cls(velocity_noise_std=float(cfg.get("velocity_noise_std", 0.5)))

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros((0, len(horizon_dts), 0), dtype=float)
        ndim = len(dynamic_obstacles[0]["position"])
        H = len(horizon_dts)
        out = np.empty((len(dynamic_obstacles), H, ndim), dtype=float)
        dts = np.asarray(horizon_dts, dtype=float)
        sigma = self.velocity_noise_std
        for k, d in enumerate(dynamic_obstacles):
            p0 = np.asarray(d["position"], dtype=float)[:ndim]
            v = np.asarray(d["velocity"], dtype=float)[:ndim]
            if sigma > 0.0:
                v = v + self._rng.normal(0.0, sigma, size=ndim)
            out[k] = p0[None, :] + dts[:, None] * v[None, :]
        return out
