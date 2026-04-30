"""Multi-drone 2D grid scenario.

Static obstacles + (optional) scenario-level dynamic obstacles are inherited
from `GridWorldScenario`. The new piece is a `drones: list[{start, goal,
radius}]` block. Each drone gets its own start / goal; cross-drone collision
attribution and peer perception live in the runner.

The scenario itself does *not* know about peer drones — a drone is not an
"obstacle" in the static map. The runner builds a per-drone `peers` view at
each step and passes it through the configured sensor before feeding the
planner. That keeps this scenario a thin extension of the single-drone case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SCENARIO_REGISTRY
from .grid_world import GridWorldScenario, _DynamicObstacle, _ObstacleSpec


@dataclass
class DroneSpec:
    start: np.ndarray
    goal: np.ndarray
    radius: float = 0.4
    name: str = ""


@SCENARIO_REGISTRY.register("multi_drone_grid")
class MultiDroneGridScenario(GridWorldScenario):
    """Same static + dynamic-obstacle world as `grid_world`, with N drones."""

    def __init__(
        self,
        size: tuple[int, int],
        drones: list[DroneSpec],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
        dynamic_obstacles: list[_DynamicObstacle] | None = None,
    ) -> None:
        if len(drones) < 1:
            raise ValueError("multi_drone_grid needs at least one drone")
        self.drones: list[DroneSpec] = drones
        # Use drone 0's start/goal for the parent class so single-drone APIs
        # (start, goal, is_collision) still respond sensibly when something
        # accidentally treats this as a single-drone scenario.
        super().__init__(
            size=size,
            start=tuple(drones[0].start),
            goal=tuple(drones[0].goal),
            obstacles=obstacles,
            resolution=resolution,
            dynamic_obstacles=dynamic_obstacles,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MultiDroneGridScenario":
        size = tuple(cfg.get("size", (50, 50)))
        if len(size) != 2:
            raise ValueError("multi_drone_grid.size must be 2D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "random")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
        )
        drone_specs = cfg.get("drones") or []
        if not drone_specs:
            raise ValueError("multi_drone_grid requires a non-empty `drones` list")
        drones = [
            DroneSpec(
                start=np.asarray(d["start"], dtype=float),
                goal=np.asarray(d["goal"], dtype=float),
                radius=float(d.get("radius", 0.4)),
                name=str(d.get("name", f"d{i}")),
            )
            for i, d in enumerate(drone_specs)
        ]
        dynamic_specs = cfg.get("dynamic_obstacles", []) or []
        dynamic = [
            _DynamicObstacle(
                pos0=np.asarray(d["start"], dtype=float),
                velocity=np.asarray(d["velocity"], dtype=float),
                reflect=bool(d.get("reflect", True)),
                radius=float(d.get("radius", 0.5)),
            )
            for d in dynamic_specs
        ]
        return cls(
            size=(int(size[0]), int(size[1])),
            drones=drones,
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
            dynamic_obstacles=dynamic,
        )

    @property
    def n_drones(self) -> int:
        return len(self.drones)
