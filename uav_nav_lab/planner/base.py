"""Planner interface.

A Planner consumes the (possibly noisy / delayed) observed state plus the
goal and obstacle map, and returns a `Plan` — a sequence of waypoints.
The runner converts the plan to a velocity setpoint each control step.

Adding a new planner is dropping a new file in this package and decorating
the class with `@PLANNER_REGISTRY.register("my_name")`. Nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..registry import Registry


@dataclass
class Plan:
    waypoints: np.ndarray  # shape (N, D) — used by pure-pursuit follower
    target_velocity: np.ndarray | None = None  # bypass follower if set
    meta: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.waypoints.shape[0] == 0


class Planner(ABC):
    """Pure function from (observation, goal, map) to a plan.

    Planners may keep internal state (e.g. last plan, cached graph) but they
    must not mutate the simulator. The runner owns timing.
    """

    max_speed: float

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Planner": ...

    @abstractmethod
    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
    ) -> Plan: ...

    def reset(self) -> None:
        """Called once at the start of each episode. Default: no-op."""


PLANNER_REGISTRY: Registry[Planner] = Registry("planner")
