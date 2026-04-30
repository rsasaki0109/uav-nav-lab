"""N-D A* planner over an occupancy grid.

Detects dimension from the occupancy map's ndim:
  - 2D → 8-connected
  - 3D → 26-connected
The output waypoints are in world meters (cell-center positions).
"""

from __future__ import annotations

import heapq
from typing import Any, Mapping

import numpy as np

from ._grid import build_neighbours, inflate_obstacles
from .base import PLANNER_REGISTRY, Plan, Planner


def _heuristic(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))


def _astar(
    occ: np.ndarray, start: tuple[int, ...], goal: tuple[int, ...]
) -> list[tuple[int, ...]] | None:
    if occ[start] or occ[goal]:
        return None
    ndim = occ.ndim
    neighbours = build_neighbours(ndim)
    open_heap: list[tuple[float, int, tuple[int, ...]]] = []
    heapq.heappush(open_heap, (0.0, 0, start))
    came_from: dict[tuple[int, ...], tuple[int, ...]] = {}
    gscore: dict[tuple[int, ...], float] = {start: 0.0}
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
        for delta, w in neighbours:
            nb = tuple(cur[i] + delta[i] for i in range(ndim))
            if any(not (0 <= nb[i] < occ.shape[i]) for i in range(ndim)):
                continue
            if occ[nb]:
                continue
            nz = sum(1 for d in delta if d != 0)
            if nz > 1:
                blocked = False
                for i in range(ndim):
                    if delta[i] == 0:
                        continue
                    probe = list(cur)
                    probe[i] += delta[i]
                    if occ[tuple(probe)]:
                        blocked = True
                        break
                if blocked:
                    continue
            tentative = cur_g + w
            if tentative < gscore.get(nb, np.inf):
                gscore[nb] = tentative
                came_from[nb] = cur
                f = tentative + _heuristic(nb, goal)
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

    def _world_to_cell(self, p: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(np.clip(p[i] / self.resolution, 0, shape[i] - 1)) for i in range(len(shape)))

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        occ_raw = np.asarray(obstacle_map, dtype=bool)
        ndim = occ_raw.ndim
        occ = inflate_obstacles(occ_raw, self.inflate)
        start_cell = self._world_to_cell(np.asarray(observation, dtype=float)[:ndim], occ.shape)
        goal_cell = self._world_to_cell(np.asarray(goal, dtype=float)[:ndim], occ.shape)
        if occ[start_cell] or occ[goal_cell]:
            occ = occ_raw
        path_cells = _astar(occ, start_cell, goal_cell)
        if path_cells is None:
            return Plan(
                waypoints=np.asarray([np.asarray(goal, dtype=float)[:ndim]], dtype=float),
                meta={"planner": "astar", "status": "no_path"},
            )
        wps = np.asarray(
            [tuple((c + 0.5) * self.resolution for c in cell) for cell in path_cells],
            dtype=float,
        )
        return Plan(waypoints=wps, meta={"planner": "astar", "status": "ok"})
