"""3D voxel-world scenario.

The 3D analogue of grid_world: integer-cell axis-aligned voxels at unit
resolution (overridable). Obstacles are 1x1x1 voxels. Start and goal carry
a halo so the drone is not spawned next to a voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SCENARIO_REGISTRY, Scenario


@dataclass
class _ObstacleSpec:
    type: str = "random"
    count: int = 0
    seed: int = 0
    cells: list[tuple[int, int, int]] | None = None


@SCENARIO_REGISTRY.register("voxel_world")
class VoxelWorldScenario(Scenario):
    def __init__(
        self,
        size: tuple[int, int, int],
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
    ) -> None:
        self.size = size
        self.resolution = float(resolution)
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self._obs_spec = obstacles
        self.occupancy = np.zeros(size, dtype=bool)
        self._rng = np.random.default_rng(obstacles.seed)
        self._populate()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VoxelWorldScenario":
        size = tuple(cfg.get("size", (30, 30, 10)))
        if len(size) != 3:
            raise ValueError("voxel_world.size must be 3D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "random")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
        )
        default_goal = (size[0] - 2, size[1] - 2, size[2] // 2)
        return cls(
            size=(int(size[0]), int(size[1]), int(size[2])),
            start=tuple(cfg.get("start", (1.0, 1.0, 1.0))),
            goal=tuple(cfg.get("goal", default_goal)),
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
        )

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed ^ self._obs_spec.seed)
        self.occupancy[:] = False
        self._populate()

    def _cell(self, p: np.ndarray | tuple[float, ...]) -> tuple[int, int, int]:
        p = np.asarray(p, dtype=float)
        return tuple(
            int(np.clip(p[i] / self.resolution, 0, self.size[i] - 1)) for i in range(3)
        )  # type: ignore[return-value]

    def _populate(self) -> None:
        if self._obs_spec.cells is not None:
            for ix, iy, iz in self._obs_spec.cells:
                if all(0 <= c < s for c, s in zip((ix, iy, iz), self.size)):
                    self.occupancy[ix, iy, iz] = True
            return
        if self._obs_spec.type == "none":
            return
        if self._obs_spec.type != "random":
            raise ValueError(f"unknown obstacle type: {self._obs_spec.type}")

        n = self._obs_spec.count
        forbidden: set[tuple[int, int, int]] = set()
        for anchor in (self._cell(self._start), self._cell(self._goal)):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        forbidden.add((anchor[0] + dx, anchor[1] + dy, anchor[2] + dz))
        placed = 0
        tries = 0
        while placed < n and tries < n * 30:
            ix = int(self._rng.integers(0, self.size[0]))
            iy = int(self._rng.integers(0, self.size[1]))
            iz = int(self._rng.integers(0, self.size[2]))
            if (ix, iy, iz) in forbidden or self.occupancy[ix, iy, iz]:
                tries += 1
                continue
            self.occupancy[ix, iy, iz] = True
            placed += 1
            tries += 1

    def is_collision(self, position: np.ndarray, radius: float) -> bool:
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        if x < 0 or y < 0 or z < 0:
            return True
        if (
            x > self.size[0] * self.resolution
            or y > self.size[1] * self.resolution
            or z > self.size[2] * self.resolution
        ):
            return True
        cx = int(x / self.resolution)
        cy = int(y / self.resolution)
        cz = int(z / self.resolution)
        cells_to_check = max(1, int(np.ceil(radius / self.resolution)))
        r2 = radius * radius
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                for dz in range(-cells_to_check, cells_to_check + 1):
                    ix, iy, iz = cx + dx, cy + dy, cz + dz
                    if not (
                        0 <= ix < self.size[0]
                        and 0 <= iy < self.size[1]
                        and 0 <= iz < self.size[2]
                    ):
                        continue
                    if not self.occupancy[ix, iy, iz]:
                        continue
                    cell_cx = (ix + 0.5) * self.resolution
                    cell_cy = (iy + 0.5) * self.resolution
                    cell_cz = (iz + 0.5) * self.resolution
                    ddx = max(abs(x - cell_cx) - self.resolution / 2, 0.0)
                    ddy = max(abs(y - cell_cy) - self.resolution / 2, 0.0)
                    ddz = max(abs(z - cell_cz) - self.resolution / 2, 0.0)
                    if ddx * ddx + ddy * ddy + ddz * ddz <= r2:
                        return True
        return False

    @property
    def start(self) -> np.ndarray:
        return self._start.copy()

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()
