"""Identity sensor — returns the true state. Useful as a baseline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("perfect")
class PerfectSensor(SensorModel):
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PerfectSensor":
        return cls()

    def reset(self, *, seed: int | None = None) -> None:
        pass

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        return np.asarray(true_position, dtype=float).copy()
