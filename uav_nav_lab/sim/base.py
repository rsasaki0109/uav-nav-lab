"""Simulator interface.

Anything that can be driven by a velocity command and reports collisions /
goal-reach is a valid backend. Real bridges (AirSim, PX4-SITL via mavsdk,
ROS2) plug in by implementing this same surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..registry import Registry


@dataclass
class SimState:
    t: float
    position: np.ndarray  # shape (D,) — D = 2 for grid_world MVP, 3 for AirSim
    velocity: np.ndarray  # shape (D,)
    extra: dict = field(default_factory=dict)

    def copy(self) -> "SimState":
        return SimState(
            t=self.t,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            extra=dict(self.extra),
        )


@dataclass
class SimStepInfo:
    collision: bool
    goal_reached: bool
    truncated: bool = False
    extra: dict = field(default_factory=dict)


class SimInterface(ABC):
    """Headless, step-driven simulator surface."""

    dt: float

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "SimInterface": ...

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> SimState: ...

    @abstractmethod
    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        """Apply a control command for `dt` seconds; return new state + info."""

    @property
    @abstractmethod
    def state(self) -> SimState: ...

    @property
    @abstractmethod
    def goal(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def obstacle_map(self) -> Any:
        """Return a representation usable by planners. The planner and the
        scenario must agree on the shape (e.g. 2D occupancy grid)."""


SIM_REGISTRY: Registry[SimInterface] = Registry("simulator")
