"""Straight-line planner.

Baseline that ignores obstacles and points directly at the goal. Useful for
isolating the effect of replanning vs. environment difficulty.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner


@PLANNER_REGISTRY.register("straight")
class StraightLinePlanner(Planner):
    def __init__(self, max_speed: float = 10.0, samples: int = 8) -> None:
        self.max_speed = float(max_speed)
        self.samples = int(samples)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "StraightLinePlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            samples=int(cfg.get("samples", 8)),
        )

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        ndim = int(np.asarray(obstacle_map).ndim) if obstacle_map is not None else 2
        start = np.asarray(observation, dtype=float)[:ndim]
        end = np.asarray(goal, dtype=float)[:ndim]
        ts = np.linspace(0.0, 1.0, self.samples + 1)[1:]
        wps = start[None, :] + (end - start)[None, :] * ts[:, None]
        return Plan(waypoints=wps, meta={"planner": "straight"})
