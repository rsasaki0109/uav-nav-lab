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


@dataclass
class _DynamicObstacle:
    """Linear-motion obstacle (with optional reflection at world bounds)."""
    pos0: np.ndarray
    velocity: np.ndarray
    reflect: bool = True
    radius: float = 0.5
    pos: np.ndarray = None  # type: ignore[assignment]
    vel: np.ndarray = None  # type: ignore[assignment]

    def reset(self) -> None:
        self.pos = self.pos0.copy()
        self.vel = self.velocity.copy()

    def step(self, dt: float, world_size: tuple[float, ...]) -> None:
        self.pos = self.pos + self.vel * dt
        if not self.reflect:
            return
        for i in range(len(self.pos)):
            upper = world_size[i]
            if self.pos[i] < 0:
                self.pos[i] = -self.pos[i]
                self.vel[i] = -self.vel[i]
            elif self.pos[i] > upper:
                self.pos[i] = 2 * upper - self.pos[i]
                self.vel[i] = -self.vel[i]


@SCENARIO_REGISTRY.register("grid_world")
class GridWorldScenario(Scenario):
    def __init__(
        self,
        size: tuple[int, int],
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
        dynamic_obstacles: list[_DynamicObstacle] | None = None,
    ) -> None:
        self.size = size
        self.resolution = float(resolution)
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self._obs_spec = obstacles
        self._rng = np.random.default_rng(obstacles.seed)
        self._static_occ = np.zeros((size[0], size[1]), dtype=bool)
        self.occupancy = self._static_occ  # alias until advance() rebuilds
        self._dynamic: list[_DynamicObstacle] = list(dynamic_obstacles or [])
        self._populate()
        for d in self._dynamic:
            d.reset()
        self._refresh_occupancy()

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
            start=tuple(cfg.get("start", (1.0, 1.0))),
            goal=tuple(cfg.get("goal", (size[0] - 2, size[1] - 2))),
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
            dynamic_obstacles=dynamic,
        )

    def reseed(self, seed: int) -> None:
        # Mix run-level seed with scenario-level obstacle seed so multiple
        # episodes inside one run get different layouts deterministically.
        self._rng = np.random.default_rng(seed ^ self._obs_spec.seed)
        self._static_occ[:] = False
        self._populate()
        for d in self._dynamic:
            d.reset()
        self._refresh_occupancy()

    def advance(self, dt: float) -> None:
        """Move dynamic obstacles forward by `dt` and refresh occupancy."""
        if not self._dynamic:
            return
        world = (self.size[0] * self.resolution, self.size[1] * self.resolution)
        for d in self._dynamic:
            d.step(dt, world)
        self._refresh_occupancy()

    def _refresh_occupancy(self) -> None:
        if not self._dynamic:
            self.occupancy = self._static_occ
            return
        grid = self._static_occ.copy()
        for d in self._dynamic:
            ix = int(d.pos[0] / self.resolution)
            iy = int(d.pos[1] / self.resolution)
            cells = max(1, int(np.ceil(d.radius / self.resolution)))
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < self.size[0] and 0 <= py < self.size[1]:
                        grid[px, py] = True
        self.occupancy = grid

    def _populate(self) -> None:
        if self._obs_spec.cells is not None:
            for ix, iy in self._obs_spec.cells:
                if 0 <= ix < self.size[0] and 0 <= iy < self.size[1]:
                    self._static_occ[ix, iy] = True
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
                if (ix, iy) in forbidden or self._static_occ[ix, iy]:
                    tries += 1
                    continue
                self._static_occ[ix, iy] = True
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
        # static cells under or near the drone
        cx = int(x / self.resolution)
        cy = int(y / self.resolution)
        cells_to_check = max(1, int(np.ceil(radius / self.resolution)))
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                ix, iy = cx + dx, cy + dy
                if not (0 <= ix < self.size[0] and 0 <= iy < self.size[1]):
                    continue
                if not self._static_occ[ix, iy]:
                    continue
                cell_cx = (ix + 0.5) * self.resolution
                cell_cy = (iy + 0.5) * self.resolution
                ddx = max(abs(x - cell_cx) - self.resolution / 2, 0.0)
                ddy = max(abs(y - cell_cy) - self.resolution / 2, 0.0)
                if ddx * ddx + ddy * ddy <= radius * radius:
                    return True
        # dynamic obstacles: simple sphere-sphere distance test on true positions
        for d in self._dynamic:
            sep = (d.pos[0] - x) ** 2 + (d.pos[1] - y) ** 2
            r = d.radius + radius
            if sep <= r * r:
                return True
        return False

    @property
    def start(self) -> np.ndarray:
        return self._start.copy()

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @property
    def dynamic_obstacles(self) -> list[dict]:
        return [
            {
                "position": [float(v) for v in d.pos],
                "velocity": [float(v) for v in d.vel],
                "radius": float(d.radius),
            }
            for d in self._dynamic
        ]
