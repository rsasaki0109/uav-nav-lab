"""Multi-drone 3D voxel scenario.

3D analogue of multi_drone_grid: same drone-list contract (start, goal,
radius, name), but the underlying world is a `voxel_world`. The runner
already iterates over `scenario.n_drones` for any scenario that exposes
it, so no runner change is needed.

Why a separate class instead of a `ndim` switch on multi_drone_grid:
the static-obstacle handling, dynamic-obstacle objects, and is_collision
math all differ between the 2D and 3D bases, and `multi_drone_grid`
already inherits from `GridWorldScenario`. Mirroring the inheritance
keeps each scenario thin (~100 lines of glue) instead of forking either
parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SCENARIO_REGISTRY
from .voxel_world import VoxelWorldScenario, _DynamicObstacle3D, _ObstacleSpec


@dataclass
class DroneSpec3D:
    start: np.ndarray
    goal: np.ndarray
    radius: float = 0.4
    name: str = ""


@SCENARIO_REGISTRY.register("multi_drone_voxel")
class MultiDroneVoxelScenario(VoxelWorldScenario):
    """Same static + dynamic-obstacle voxel world as `voxel_world`, with N drones."""

    def __init__(
        self,
        size: tuple[int, int, int],
        drones: list[DroneSpec3D],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
        dynamic_obstacles: list[_DynamicObstacle3D] | None = None,
    ) -> None:
        if len(drones) < 1:
            raise ValueError("multi_drone_voxel needs at least one drone")
        self.drones: list[DroneSpec3D] = drones
        super().__init__(
            size=size,
            start=tuple(drones[0].start),
            goal=tuple(drones[0].goal),
            obstacles=obstacles,
            resolution=resolution,
            dynamic_obstacles=dynamic_obstacles,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MultiDroneVoxelScenario":
        size = tuple(cfg.get("size", (40, 40, 12)))
        if len(size) != 3:
            raise ValueError("multi_drone_voxel.size must be 3D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "random")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
        )
        drone_specs = cfg.get("drones") or []
        if not drone_specs:
            raise ValueError("multi_drone_voxel requires a non-empty `drones` list")
        drones = [
            DroneSpec3D(
                start=np.asarray(d["start"], dtype=float),
                goal=np.asarray(d["goal"], dtype=float),
                radius=float(d.get("radius", 0.4)),
                name=str(d.get("name", f"d{i}")),
            )
            for i, d in enumerate(drone_specs)
        ]
        for d in drones:
            if d.start.shape != (3,) or d.goal.shape != (3,):
                raise ValueError(
                    f"multi_drone_voxel drone {d.name!r}: start/goal must be 3D"
                )
        dynamic_specs = cfg.get("dynamic_obstacles", []) or []
        dynamic = [
            _DynamicObstacle3D(
                pos0=np.asarray(d["start"], dtype=float),
                velocity=np.asarray(d["velocity"], dtype=float),
                reflect=bool(d.get("reflect", True)),
                radius=float(d.get("radius", 0.5)),
            )
            for d in dynamic_specs
        ]
        return cls(
            size=(int(size[0]), int(size[1]), int(size[2])),
            drones=drones,
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
            dynamic_obstacles=dynamic,
        )

    @property
    def n_drones(self) -> int:
        return len(self.drones)
