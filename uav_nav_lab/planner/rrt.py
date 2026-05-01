"""Continuous-space RRT planner — N-D.

Single-tree Rapidly-exploring Random Tree over an occupancy grid.
Sampling in *world* coordinates (not cells), with collision checks
that walk a discrete step along the candidate edge against the
inflated occupancy. Returns waypoints in world meters.

Compared to A* (grid-discrete, deterministic, optimal on the grid):
  - RRT samples in continuous space, so paths are not constrained to
    the 8-/26-connected grid lattice — short diagonals across open
    space are natural.
  - RRT is probabilistically complete but *not* optimal. It returns
    the first path found, which is typically zigzaggy. Set
    `goal_bias` > 0 to bias samples toward the goal and reduce the
    expected number of iterations to find a path.
  - For the framework's CV peer-prediction MPC, RRT is the obvious
    sampling-based comparison point: same probabilistic-coverage
    spirit, no rollout / prediction model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._grid import inflate_obstacles
from .base import PLANNER_REGISTRY, Plan, Planner


@PLANNER_REGISTRY.register("rrt")
class RRTPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        replan_period: float = 0.5,
        max_samples: int = 1000,
        step_size: float = 2.0,
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
        self.goal_tolerance = float(goal_tolerance)
        self.goal_bias = float(goal_bias)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RRTPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            replan_period=float(cfg.get("replan_period", 0.5)),
            max_samples=int(cfg.get("max_samples", 1000)),
            step_size=float(cfg.get("step_size", 2.0)),
            goal_tolerance=float(cfg.get("goal_tolerance", 1.5)),
            goal_bias=float(cfg.get("goal_bias", 0.1)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 0)),
            seed=int(cfg.get("seed", 0)),
        )

    def reset(self) -> None:
        # Re-seed each episode so different episodes use different sample
        # streams; deterministic given (planner_seed, episode_index).
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
            if occ[(*[int(x) for x in [int(p[j] / self.resolution) for j in range(ndim)]],)]:
                return False
        return not bool(occ[self._cell(p, occ.shape)])

    def _edge_free(self, occ: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        # Walk from a to b at sub-cell resolution, check each step.
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

        # If the start or goal cell is occupied (e.g. the drone has clipped
        # an obstacle's inflation halo), fall back to the un-inflated map so
        # the planner can still produce a path.
        if occ[self._cell(start, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        nodes: list[np.ndarray] = [start.copy()]
        parents: list[int] = [-1]
        world_lo = np.zeros(ndim)
        world_hi = np.array(occ.shape, dtype=float) * self.resolution

        goal_idx = -1
        for _ in range(self.max_samples):
            if self._rng.random() < self.goal_bias:
                target = gl
            else:
                target = self._rng.uniform(world_lo, world_hi)
            # nearest node in tree
            arr = np.asarray(nodes)
            dists = np.linalg.norm(arr - target, axis=1)
            near_idx = int(np.argmin(dists))
            near = nodes[near_idx]
            # step toward target by step_size
            direction = target - near
            d = float(np.linalg.norm(direction))
            if d < 1e-9:
                continue
            new = near + direction / d * min(self.step_size, d)
            if not self._edge_free(occ, near, new):
                continue
            nodes.append(new)
            parents.append(near_idx)
            if float(np.linalg.norm(new - gl)) <= self.goal_tolerance:
                goal_idx = len(nodes) - 1
                break

        if goal_idx < 0:
            return Plan(
                waypoints=np.asarray([gl], dtype=float),
                meta={"planner": "rrt", "status": "no_path", "tree_size": len(nodes)},
            )

        # Trace path from goal back to start.
        path: list[np.ndarray] = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        wps = np.asarray(path, dtype=float)
        return Plan(
            waypoints=wps,
            meta={"planner": "rrt", "status": "ok", "tree_size": len(nodes)},
        )
