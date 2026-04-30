"""Sensor model interface.

Sensors observe the *true* simulator state and produce what the planner
actually sees. Real-world phenomena to model: latency, additive noise,
dropouts, biased measurements, partial observability. Each backend handles
the subset relevant to the study.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from ..registry import Registry


class SensorModel(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SensorModel": ...

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None: ...

    @abstractmethod
    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray: ...

    def observe_map(
        self, t: float, true_position: np.ndarray, true_obstacle_map: np.ndarray
    ) -> np.ndarray:
        """Return the perceived obstacle map.

        Default: passthrough — the planner sees the full map. Range-limited
        sensors (lidar / depth) override this to return what the sensor has
        actually observed up to time t.
        """
        return true_obstacle_map


SENSOR_REGISTRY: Registry[SensorModel] = Registry("sensor")
