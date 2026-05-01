"""RRT* — RRT with neighborhood rewiring for asymptotic optimality.

Same continuous-space sampling as `rrt`, with two extensions on each
new sample:

  1. **Best-parent selection**: instead of attaching the new node to
     the nearest existing node, scan all tree nodes within
     `rewire_radius` and pick the one that yields the lowest
     cost-from-start (collision-free). The classical heuristic uses
     a `radius = gamma * (log(n)/n)^(1/d)` schedule; we expose
     `rewire_radius` as a fixed knob (good enough for our scenarios
     and a single configurable to ablate).
  2. **Rewiring**: after attaching, scan neighbors again and rewire
     any node whose path through the new node would be cheaper than
     its current parent's path.

Compared to plain `rrt`, the produced path is closer to optimal
(shortest collision-free) at the cost of more per-sample work.
For the framework's head-to-head, this lets us isolate "sampling vs
grid" (rrt vs astar) from "sampling-optimal vs sampling-greedy"
(rrt_star vs rrt).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._grid import inflate_obstacles
from .base import PLANNER_REGISTRY, Plan, Planner


@PLANNER_REGISTRY.register("rrt_star")
class RRTStarPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        replan_period: float = 0.5,
        max_samples: int = 1000,
        step_size: float = 2.0,
        rewire_radius: float = 4.0,
        goal_tolerance: float = 1.5,
        goal_bias: float = 0.1,
        resolution: float = 1.0,
        inflate: int = 0,
        seed: int = 0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.replan_period = float(replan_period)
        self.max_samples = int(max_samples)
        self.step_size = float(step_size)
        self.rewire_radius = float(rewire_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_bias = float(goal_bias)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RRTStarPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            replan_period=float(cfg.get("replan_period", 0.5)),
            max_samples=int(cfg.get("max_samples", 1000)),
            step_size=float(cfg.get("step_size", 2.0)),
            rewire_radius=float(cfg.get("rewire_radius", 4.0)),
            goal_tolerance=float(cfg.get("goal_tolerance", 1.5)),
            goal_bias=float(cfg.get("goal_bias", 0.1)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 0)),
            seed=int(cfg.get("seed", 0)),
        )

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def _cell(self, p: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(np.clip(p[i] / self.resolution, 0, shape[i] - 1)) for i in range(len(shape))
        )

    def _is_free(self, occ: np.ndarray, p: np.ndarray) -> bool:
        ndim = occ.ndim
        for i in range(ndim):
            ci = int(p[i] / self.resolution)
            if not (0 <= ci < occ.shape[i]):
                return False
        return not bool(occ[self._cell(p, occ.shape)])

    def _edge_free(self, occ: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        diff = b - a
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return self._is_free(occ, a)
        n_steps = max(1, int(np.ceil(dist / (self.resolution * 0.5))))
        for k in range(n_steps + 1):
            p = a + diff * (k / n_steps)
            if not self._is_free(occ, p):
                return False
        return True

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
        start = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        if occ[self._cell(start, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        nodes: list[np.ndarray] = [start.copy()]
        parents: list[int] = [-1]
        costs: list[float] = [0.0]  # cost-from-start
        world_lo = np.zeros(ndim)
        world_hi = np.array(occ.shape, dtype=float) * self.resolution

        goal_idx = -1
        for _ in range(self.max_samples):
            if self._rng.random() < self.goal_bias:
                target = gl
            else:
                target = self._rng.uniform(world_lo, world_hi)

            arr = np.asarray(nodes)
            dists_to_target = np.linalg.norm(arr - target, axis=1)
            near_idx = int(np.argmin(dists_to_target))
            near = nodes[near_idx]
            direction = target - near
            d = float(np.linalg.norm(direction))
            if d < 1e-9:
                continue
            new = near + direction / d * min(self.step_size, d)
            if not self._edge_free(occ, near, new):
                continue

            # RRT* step 1: pick best parent within rewire_radius
            dists_to_new = np.linalg.norm(arr - new, axis=1)
            neighbor_mask = dists_to_new <= self.rewire_radius
            best_parent = near_idx
            best_cost = costs[near_idx] + float(np.linalg.norm(new - near))
            for ni in np.where(neighbor_mask)[0]:
                ni = int(ni)
                cand_cost = costs[ni] + float(np.linalg.norm(new - nodes[ni]))
                if cand_cost < best_cost and self._edge_free(occ, nodes[ni], new):
                    best_parent = ni
                    best_cost = cand_cost

            nodes.append(new)
            parents.append(best_parent)
            costs.append(best_cost)
            new_idx = len(nodes) - 1

            # RRT* step 2: rewire neighbors that would benefit from going
            # through the new node.
            for ni in np.where(neighbor_mask)[0]:
                ni = int(ni)
                if ni == new_idx or ni == best_parent:
                    continue
                edge = float(np.linalg.norm(nodes[ni] - new))
                cand_cost = best_cost + edge
                if cand_cost < costs[ni] and self._edge_free(occ, new, nodes[ni]):
                    parents[ni] = new_idx
                    costs[ni] = cand_cost

            if float(np.linalg.norm(new - gl)) <= self.goal_tolerance:
                # Don't break immediately — keep refining the tree to push
                # the goal cost down via rewiring. But once we have a goal
                # node, only refine for a fixed extra budget (1/4 of the
                # remaining sample budget) to avoid going too long.
                if goal_idx < 0:
                    goal_idx = new_idx
                # Otherwise: this might be a better goal-region node; keep
                # the lower-cost one.
                elif costs[new_idx] < costs[goal_idx]:
                    goal_idx = new_idx

        if goal_idx < 0:
            return Plan(
                waypoints=np.asarray([gl], dtype=float),
                meta={"planner": "rrt_star", "status": "no_path", "tree_size": len(nodes)},
            )

        path: list[np.ndarray] = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        wps = np.asarray(path, dtype=float)
        return Plan(
            waypoints=wps,
            meta={
                "planner": "rrt_star",
                "status": "ok",
                "tree_size": len(nodes),
                "path_cost": float(costs[goal_idx]),
            },
        )
