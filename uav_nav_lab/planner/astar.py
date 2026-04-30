"""2D A* planner over an occupancy grid.

8-connected grid, octile heuristic. The output waypoints are in world meters
(cell-center positions), so the runner can follow them directly.
"""

from __future__ import annotations

import heapq
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_NEIGHBOURS = [
    (-1, -1, np.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, np.sqrt(2.0)),
]


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2.0) - 2.0) * min(dx, dy)


def _astar_grid(
    occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    if occ[start] or occ[goal]:
        return None
    nx, ny = occ.shape
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], float] = {start: 0.0}
    counter = 1
    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        cur_g = gscore[cur]
        for dx, dy, w in _NEIGHBOURS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if occ[nb]:
                continue
            # disallow corner-cutting through diagonally-adjacent obstacles
            if dx != 0 and dy != 0:
                if occ[cur[0] + dx, cur[1]] or occ[cur[0], cur[1] + dy]:
                    continue
            tentative = cur_g + w
            if tentative < gscore.get(nb, np.inf):
                gscore[nb] = tentative
                came_from[nb] = cur
                f = tentative + _octile(nb, goal)
                heapq.heappush(open_heap, (f, counter, nb))
                counter += 1
    return None


@PLANNER_REGISTRY.register("astar")
class AStarPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        resolution: float = 1.0,
        inflate: int = 0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.resolution = float(resolution)
        self.inflate = int(inflate)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "AStarPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 0)),
        )

    def _world_to_cell(self, p: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
        ix = int(np.clip(p[0] / self.resolution, 0, shape[0] - 1))
        iy = int(np.clip(p[1] / self.resolution, 0, shape[1] - 1))
        return ix, iy

    def _inflated(self, occ: np.ndarray) -> np.ndarray:
        if self.inflate <= 0:
            return occ
        out = occ.copy()
        for _ in range(self.inflate):
            shifted = np.zeros_like(out)
            shifted[1:, :] |= out[:-1, :]
            shifted[:-1, :] |= out[1:, :]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            out |= shifted
        return out

    def plan(self, observation: np.ndarray, goal: np.ndarray, obstacle_map: Any) -> Plan:
        occ = np.asarray(obstacle_map, dtype=bool)
        occ = self._inflated(occ)
        start_cell = self._world_to_cell(np.asarray(observation, dtype=float), occ.shape)
        goal_cell = self._world_to_cell(np.asarray(goal, dtype=float), occ.shape)
        # If start/goal landed on an inflated obstacle, fall back to raw map
        # so we still produce *some* plan — the runner can still detect collisions.
        if occ[start_cell] or occ[goal_cell]:
            occ = np.asarray(obstacle_map, dtype=bool)
        path_cells = _astar_grid(occ, start_cell, goal_cell)
        if path_cells is None:
            # No path — fall back to a one-waypoint plan toward the goal so
            # the controller still moves; the recorder will mark planner failure.
            return Plan(
                waypoints=np.asarray([goal[:2]], dtype=float),
                meta={"planner": "astar", "status": "no_path"},
            )
        wps = np.asarray(
            [((ix + 0.5) * self.resolution, (iy + 0.5) * self.resolution) for ix, iy in path_cells],
            dtype=float,
        )
        return Plan(waypoints=wps, meta={"planner": "astar", "status": "ok"})
