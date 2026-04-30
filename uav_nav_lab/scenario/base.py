"""Scenario interface — owns environment geometry, start, and goal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from ..registry import Registry


class Scenario(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Scenario": ...

    @abstractmethod
    def reseed(self, seed: int) -> None: ...

    @abstractmethod
    def is_collision(self, position: np.ndarray, radius: float) -> bool: ...

    @property
    @abstractmethod
    def start(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def goal(self) -> np.ndarray: ...

    @property
    def ndim(self) -> int:
        """Dimension of the scenario (2 for grid_world, 3 for voxel_world)."""
        return int(self.start.shape[0])


SCENARIO_REGISTRY: Registry[Scenario] = Registry("scenario")
