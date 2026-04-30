"""2D grid-world scenario.

Coordinates are in meters; the underlying occupancy grid is integer-cell.
Each obstacle is a 1x1 cell. `world_resolution` lets you scale meters-per-cell
if you want a denser grid (default 1.0).
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
    cells: list[tuple[int, int]] | None = None  # explicit list overrides random


@SCENARIO_REGISTRY.register("grid_world")
class GridWorldScenario(Scenario):
    def __init__(
        self,
        size: tuple[int, int],
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
    ) -> None:
        self.size = size
        self.resolution = float(resolution)
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self._obs_spec = obstacles
        self._rng = np.random.default_rng(obstacles.seed)
        self.occupancy = np.zeros((size[0], size[1]), dtype=bool)
        self._populate()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GridWorldScenario":
        size = tuple(cfg.get("size", (50, 50)))
        if len(size) != 2:
            raise ValueError("scenario.size must be 2D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "random")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
        )
        return cls(
            size=(int(size[0]), int(size[1])),
            start=tuple(cfg.get("start", (1.0, 1.0))),
            goal=tuple(cfg.get("goal", (size[0] - 2, size[1] - 2))),
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
        )

    def reseed(self, seed: int) -> None:
        # Mix run-level seed with scenario-level obstacle seed so multiple
        # episodes inside one run get different layouts deterministically.
        self._rng = np.random.default_rng(seed ^ self._obs_spec.seed)
        self.occupancy[:] = False
        self._populate()

    def _populate(self) -> None:
        if self._obs_spec.cells is not None:
            for ix, iy in self._obs_spec.cells:
                if 0 <= ix < self.size[0] and 0 <= iy < self.size[1]:
                    self.occupancy[ix, iy] = True
            return
        if self._obs_spec.type == "random":
            n = self._obs_spec.count
            tries = 0
            placed = 0
            # keep a halo around start and goal so the drone (with radius)
            # is not spawned immediately next to an obstacle.
            forbidden: set[tuple[int, int]] = set()
            for anchor in (self._cell(self._start), self._cell(self._goal)):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        forbidden.add((anchor[0] + dx, anchor[1] + dy))
            while placed < n and tries < n * 20:
                ix = int(self._rng.integers(0, self.size[0]))
                iy = int(self._rng.integers(0, self.size[1]))
                if (ix, iy) in forbidden or self.occupancy[ix, iy]:
                    tries += 1
                    continue
                self.occupancy[ix, iy] = True
                placed += 1
                tries += 1
        elif self._obs_spec.type == "none":
            pass
        else:
            raise ValueError(f"unknown obstacle type: {self._obs_spec.type}")

    def _cell(self, p: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        p = np.asarray(p, dtype=float)
        ix = int(np.clip(p[0] / self.resolution, 0, self.size[0] - 1))
        iy = int(np.clip(p[1] / self.resolution, 0, self.size[1] - 1))
        return ix, iy

    def is_collision(self, position: np.ndarray, radius: float) -> bool:
        # Out-of-bounds counts as collision (drone left the world).
        x, y = float(position[0]), float(position[1])
        if x < 0 or y < 0:
            return True
        if x > self.size[0] * self.resolution or y > self.size[1] * self.resolution:
            return True
        # Check the cell under the drone and its 8 neighbours within `radius`.
        cx = int(x / self.resolution)
        cy = int(y / self.resolution)
        cells_to_check = max(1, int(np.ceil(radius / self.resolution)))
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                ix, iy = cx + dx, cy + dy
                if not (0 <= ix < self.size[0] and 0 <= iy < self.size[1]):
                    continue
                if not self.occupancy[ix, iy]:
                    continue
                # closest point on the cell to the drone
                cell_cx = (ix + 0.5) * self.resolution
                cell_cy = (iy + 0.5) * self.resolution
                ddx = max(abs(x - cell_cx) - self.resolution / 2, 0.0)
                ddy = max(abs(y - cell_cy) - self.resolution / 2, 0.0)
                if ddx * ddx + ddy * ddy <= radius * radius:
                    return True
        return False

    @property
    def start(self) -> np.ndarray:
        return self._start.copy()

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()
