"""Point-cloud → occupancy-grid sensor.

Consumes raw LiDAR point clouds the simulator backend exposes via
`state.extra["lidar_points"]` (currently produced by `AirSimBridge` when
its `lidars: [...]` config is non-empty) and rasterizes them into the
same occupancy-grid shape the rest of the framework's planners consume.

Why this is its own sensor (not a flag on `lidar`):
  - The existing `lidar` sensor is a *synthetic* range-limited view of
    the scenario's known occupancy. It models partial observability for
    studies that already have a ground-truth grid (`grid_world`,
    `voxel_world`).
  - This sensor is the inverse direction: there is no scenario-side
    ground truth (AirSim's world is a 3D mesh), only sensor returns,
    and we build occupancy *from* the returns.
  - Keeping them separate avoids overloading one config block with
    two mutually exclusive data sources.

Coordinate convention:
  - Point clouds arrive in vehicle-local ENU (the AirSim bridge does
    the NED → ENU flip). World position = `true_position + local_point`.
  - Cell index = `floor(world_pos / resolution)`. Cells outside the
    grid are clipped (silently).
  - 2D scenarios drop the z component; 3D scenarios use all three.

`memory: true` (default) accumulates seen-occupied cells across the
episode — what most onboard SLAM-fed planners get. `memory: false`
returns only the cells hit by the latest sweep.

`inflate` adds a per-axis cell-radius dilation around each hit, useful
for making the planner's safety margin geometric rather than purely
cost-based. Default 0 (no inflation).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("pointcloud_occupancy")
class PointcloudOccupancySensor(SensorModel):
    def __init__(
        self,
        resolution: float = 1.0,
        memory: bool = True,
        inflate: int = 0,
        range_m: float | None = None,
        lidars: list[str] | None = None,
    ) -> None:
        self.resolution = float(resolution)
        self.memory = bool(memory)
        self.inflate = max(0, int(inflate))
        # Optional max-range filter (drops points farther than this from
        # the drone). `None` keeps every point the simulator delivered.
        self.range_m = None if range_m is None else float(range_m)
        # Restrict to a subset of lidar names; `None` consumes whatever
        # the simulator produced.
        self.lidars: list[str] | None = (
            None if lidars is None else [str(name) for name in lidars]
        )
        self._seen: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PointcloudOccupancySensor":
        lidars_cfg = cfg.get("lidars")
        return cls(
            resolution=float(cfg.get("resolution", 1.0)),
            memory=bool(cfg.get("memory", True)),
            inflate=int(cfg.get("inflate", 0)),
            range_m=None if cfg.get("range_m") is None else float(cfg["range_m"]),
            lidars=None if lidars_cfg is None else list(lidars_cfg),
        )

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002
        # No randomness in this sensor; seed is accepted for interface parity.
        self._seen = None

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:  # noqa: ARG002
        # Position observation is pass-through (perfect). Use composition
        # with `delayed` / `kalman_delayed` upstream if you want noise.
        return np.asarray(true_position, dtype=float).copy()

    def observe_map(
        self,
        t: float,
        true_position: np.ndarray,
        true_obstacle_map: np.ndarray,
        sim_extra: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        occ_shape = np.asarray(true_obstacle_map).shape
        ndim = len(occ_shape)
        if self._seen is None or self._seen.shape != occ_shape:
            self._seen = np.zeros(occ_shape, dtype=bool)
        if not self.memory:
            self._seen.fill(False)

        clouds = self._collect_clouds(sim_extra)
        if not clouds:
            return self._seen

        pos = np.asarray(true_position, dtype=float)
        pos3 = np.zeros(3)
        pos3[: min(3, pos.size)] = pos[:3]

        for local in clouds:
            if local.size == 0:
                continue
            world = local + pos3  # vehicle-local ENU → world ENU
            if self.range_m is not None:
                d2 = np.sum((world - pos3) ** 2, axis=1)
                world = world[d2 <= self.range_m * self.range_m]
                if world.size == 0:
                    continue
            self._mark(world, ndim, occ_shape)

        if self.inflate > 0:
            self._dilate(occ_shape)
        return self._seen

    def _collect_clouds(
        self, sim_extra: Mapping[str, Any] | None
    ) -> list[np.ndarray]:
        if not sim_extra:
            return []
        bag = sim_extra.get("lidar_points")
        if not isinstance(bag, dict):
            return []
        names = self.lidars if self.lidars is not None else list(bag.keys())
        clouds: list[np.ndarray] = []
        for name in names:
            pc = bag.get(name)
            if pc is None:
                continue
            arr = np.asarray(pc, dtype=float)
            # Expect shape (N, 3); reshape forgivingly if flat.
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            if arr.ndim != 2 or arr.shape[1] != 3:
                continue
            clouds.append(arr)
        return clouds

    def _mark(self, world: np.ndarray, ndim: int, occ_shape: tuple[int, ...]) -> None:
        cells = np.floor(world[:, :ndim] / self.resolution).astype(int)
        # Discard cells outside the grid (silent clip — the cloud may extend
        # past the scenario's bounding box even though those points are real).
        in_bounds = np.ones(cells.shape[0], dtype=bool)
        for i in range(ndim):
            in_bounds &= (cells[:, i] >= 0) & (cells[:, i] < occ_shape[i])
        cells = cells[in_bounds]
        if cells.size == 0:
            return
        # tuple-of-arrays indexing — works for both 2D and 3D occ_shape.
        idx = tuple(cells[:, i] for i in range(ndim))
        self._seen[idx] = True

    def _dilate(self, occ_shape: tuple[int, ...]) -> None:
        # Cheap separable dilation: shift the seen mask by ±inflate cells in
        # every axis and OR the results in. Works in 2D and 3D, no scipy dep.
        out = self._seen.copy()
        for axis in range(len(occ_shape)):
            for shift in range(1, self.inflate + 1):
                out |= np.roll(self._seen, shift, axis=axis)
                out |= np.roll(self._seen, -shift, axis=axis)
        self._seen = out

    def observe_dynamics(
        self,
        t: float,  # noqa: ARG002
        true_position: np.ndarray,  # noqa: ARG002
        dynamic_obstacles: list[dict],  # noqa: ARG002
    ) -> list[dict]:
        # Point clouds do not segment moving threats — that is downstream
        # (clustering / tracking). Return empty so the planner does not
        # silently get free ground-truth dyn obstacles when this sensor
        # is the only one configured.
        return []
