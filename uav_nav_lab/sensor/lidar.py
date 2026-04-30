"""Range-limited perception sensor (a stand-in for lidar / depth camera).

Models partial observability: only obstacle cells whose centers are within
`range_m` of the drone are visible. Unseen cells are reported as free —
this is the optimistic / "explore first" treatment, which is what most
onboard mapping pipelines feed downstream planners.

`memory: true` accumulates seen cells across the episode (a SLAM-like map
built up over time). `memory: false` reports only the current view.

Position observation supports the same fixed-delay + Gaussian noise model
as the `delayed` sensor, so a lidar deployment can model perception lag in
a single block.

Works in 2D and 3D — cell iteration is over the bounding box of the
drone's visible sphere, so it stays cheap even on large maps.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("lidar")
class LidarSensor(SensorModel):
    def __init__(
        self,
        range_m: float = 8.0,
        delay: float = 0.0,
        dt: float = 0.05,
        position_noise_std: float = 0.0,
        resolution: float = 1.0,
        memory: bool = True,
    ) -> None:
        self.range_m = float(range_m)
        self.delay = float(delay)
        self.dt = float(dt)
        self.noise_std = float(position_noise_std)
        self.resolution = float(resolution)
        self.memory = bool(memory)
        self._buffer_len = max(1, int(round(self.delay / self.dt)))
        self._buffer: deque[np.ndarray] = deque(maxlen=self._buffer_len)
        self._rng = np.random.default_rng()
        self._seen: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LidarSensor":
        return cls(
            range_m=float(cfg.get("range", cfg.get("range_m", 8.0))),
            delay=float(cfg.get("delay", 0.0)),
            dt=float(cfg.get("dt", 0.05)),
            position_noise_std=float(cfg.get("position_noise_std", 0.0)),
            resolution=float(cfg.get("resolution", 1.0)),
            memory=bool(cfg.get("memory", True)),
        )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._buffer = deque(maxlen=self._buffer_len)
        self._seen = None

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:
        true_position = np.asarray(true_position, dtype=float).copy()
        self._buffer.append(true_position)
        obs = self._buffer[0].copy()
        if self.noise_std > 0.0:
            obs = obs + self._rng.normal(0.0, self.noise_std, size=obs.shape)
        return obs

    def observe_map(
        self, t: float, true_position: np.ndarray, true_obstacle_map: np.ndarray
    ) -> np.ndarray:
        occ = np.asarray(true_obstacle_map, dtype=bool)
        ndim = occ.ndim
        if not self.memory or self._seen is None or self._seen.shape != occ.shape:
            self._seen = np.zeros_like(occ, dtype=bool)

        pos = np.asarray(true_position, dtype=float)[:ndim]
        # bounding box of cells potentially within range
        cells_radius = max(1, int(np.ceil(self.range_m / self.resolution)))
        center = (pos / self.resolution).astype(int)
        r2 = (self.range_m / self.resolution) ** 2

        # iterate over the bounding box and OR in newly visible obstacles
        slices = []
        for i in range(ndim):
            lo = max(0, center[i] - cells_radius)
            hi = min(occ.shape[i], center[i] + cells_radius + 1)
            slices.append(slice(lo, hi))
        sub_occ = occ[tuple(slices)]
        # build coordinate grid for this sub-region (cell centers)
        ranges = [np.arange(slices[i].start, slices[i].stop) + 0.5 for i in range(ndim)]
        meshes = np.meshgrid(*ranges, indexing="ij")
        sq = np.zeros_like(meshes[0], dtype=float)
        for i in range(ndim):
            sq += (meshes[i] - center[i] - 0.5) ** 2
        within = sq <= r2
        # Overwrite (not OR) so dynamic obstacles that have moved away are
        # cleared from memory. Cells outside the visible sphere keep their
        # previous belief — that is what `memory=True` actually means.
        sub_seen = self._seen[tuple(slices)]
        sub_seen[within] = sub_occ[within]
        self._seen[tuple(slices)] = sub_seen
        return self._seen

    def observe_dynamics(
        self, t: float, true_position: np.ndarray, dynamic_obstacles: list[dict]
    ) -> list[dict]:
        pos = np.asarray(true_position, dtype=float)
        ndim = pos.shape[0]
        out = []
        r2 = self.range_m * self.range_m
        for d in dynamic_obstacles:
            obs_pos = np.asarray(d["position"], dtype=float)[:ndim]
            sep2 = float(np.sum((obs_pos - pos[:ndim]) ** 2))
            if sep2 <= r2:
                out.append(dict(d))
        return out
