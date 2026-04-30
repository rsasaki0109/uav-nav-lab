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


SENSOR_REGISTRY: Registry[SensorModel] = Registry("sensor")
