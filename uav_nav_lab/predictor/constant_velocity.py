"""Constant-velocity predictor (the historical MPC default).

Each obstacle is propagated as `position + velocity * t` independently. No
acceleration model, no interaction effects. Cheap and good enough for short
horizons when the velocity estimate is clean.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PREDICTOR_REGISTRY, Predictor


@PREDICTOR_REGISTRY.register("constant_velocity")
class ConstantVelocityPredictor(Predictor):
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ConstantVelocityPredictor":
        return cls()

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
        for k, d in enumerate(dynamic_obstacles):
            p0 = np.asarray(d["position"], dtype=float)[:ndim]
            v = np.asarray(d["velocity"], dtype=float)[:ndim]
            out[k] = p0[None, :] + dts[:, None] * v[None, :]
        return out
