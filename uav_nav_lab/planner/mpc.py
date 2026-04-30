"""Sampling-based MPC planner with an A* cost-to-go heuristic.

Each replan we

  1. run Dijkstra from the goal cell over the (inflated) occupancy grid to
     get a global cost-to-go map — this is the obstacle-aware "distance to
     goal" the local rollouts are scored against.
  2. sample `n_samples` constant-velocity candidates around the goal direction
  3. roll each out for `horizon` steps
  4. score by  w_goal * cost_to_go(end_cell)
            + w_obs  * occupied_cells_traversed
            + w_smooth * |action - prev_action|

and return the winning rollout (with `target_velocity` set, so the runner
applies it directly without pure-pursuit aliasing).

The hybrid avoids the local-minima trap of a pure local MPC — without
pulling in a full optimization stack.
"""

from __future__ import annotations

import heapq
from typing import Any, Mapping

import numpy as np

from .base import PLANNER_REGISTRY, Plan, Planner

_NEIGH = [
    (-1, -1, np.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, np.sqrt(2.0)),
]


def _cost_to_go(occ: np.ndarray, goal_cell: tuple[int, int]) -> np.ndarray:
    """Dijkstra from goal across the free cells; obstacles get +inf."""
    nx, ny = occ.shape
    dist = np.full(occ.shape, np.inf, dtype=float)
    if occ[goal_cell]:
        return dist
    dist[goal_cell] = 0.0
    heap: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, goal_cell)]
    counter = 1
    while heap:
        d, _, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        for dx, dy, w in _NEIGH:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if occ[nb]:
                continue
            if dx != 0 and dy != 0:
                if occ[cur[0] + dx, cur[1]] or occ[cur[0], cur[1] + dy]:
                    continue
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(heap, (nd, counter, nb))
                counter += 1
    return dist


@PLANNER_REGISTRY.register("mpc")
class SamplingMPCPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        horizon: int = 60,
        dt_plan: float = 0.05,
        n_samples: int = 32,
        resolution: float = 1.0,
        inflate: int = 1,
        goal_radius: float = 1.5,
        w_goal: float = 1.0,
        w_obs: float = 100.0,
        w_smooth: float = 0.05,
    ) -> None:
        self.max_speed = float(max_speed)
        self.horizon = int(horizon)
        self.dt_plan = float(dt_plan)
        self.n_samples = int(n_samples)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self.goal_radius = float(goal_radius)
        self.w_goal = float(w_goal)
        self.w_obs = float(w_obs)
        self.w_smooth = float(w_smooth)
        self._prev_action: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SamplingMPCPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            horizon=int(cfg.get("horizon", 60)),
            dt_plan=float(cfg.get("dt_plan", 0.05)),
            n_samples=int(cfg.get("n_samples", 32)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 1)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            w_goal=float(cfg.get("w_goal", 1.0)),
            w_obs=float(cfg.get("w_obs", 100.0)),
            w_smooth=float(cfg.get("w_smooth", 0.05)),
        )

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

    def reset(self) -> None:
        self._prev_action = None

    def _occupied(self, occ: np.ndarray, p: np.ndarray) -> bool:
        ix = int(p[0] / self.resolution)
        iy = int(p[1] / self.resolution)
        if not (0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]):
            return True  # out-of-bounds counts as obstacle
        return bool(occ[ix, iy])

    def _cell(self, p: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
        ix = int(np.clip(p[0] / self.resolution, 0, shape[0] - 1))
        iy = int(np.clip(p[1] / self.resolution, 0, shape[1] - 1))
        return ix, iy

    def plan(self, observation: np.ndarray, goal: np.ndarray, obstacle_map: Any) -> Plan:
        occ_raw = np.asarray(obstacle_map, dtype=bool)
        occ = self._inflated(occ_raw)
        obs = np.asarray(observation, dtype=float)[:2]
        gl = np.asarray(goal, dtype=float)[:2]

        to_goal = gl - obs
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-6:
            return Plan(waypoints=np.asarray([gl], dtype=float), meta={"planner": "mpc"})

        # if start/goal landed in inflated obstacle, fall back to raw map
        if occ[self._cell(obs, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        ctg = _cost_to_go(occ, self._cell(gl, occ.shape))
        max_finite = float(np.max(ctg[np.isfinite(ctg)])) if np.any(np.isfinite(ctg)) else 1e6
        unreachable_penalty = max_finite + 100.0  # used when rollout end is in an isolated cell

        # Full 360° fan around goal direction (goal direction first so it's
        # tried with zero smoothing penalty in the very first plan).
        base = to_goal / dist_goal
        angles = np.linspace(-np.pi, np.pi, self.n_samples, endpoint=False)
        actions: list[np.ndarray] = []
        for ang in angles:
            ca, sa = np.cos(ang), np.sin(ang)
            v = np.array([ca * base[0] - sa * base[1], sa * base[0] + ca * base[1]])
            actions.append(v * self.max_speed)

        best_cost = np.inf
        best_rollout: np.ndarray | None = None
        best_action: np.ndarray | None = None
        gr2 = self.goal_radius * self.goal_radius
        for v in actions:
            rollout = np.empty((self.horizon + 1, 2), dtype=float)
            rollout[0] = obs
            collision_pen = 0
            ctg_min = np.inf
            ctg_sum_until = 0.0
            steps_until = 0
            reaches_goal = False
            for h in range(1, self.horizon + 1):
                rollout[h] = rollout[h - 1] + v * self.dt_plan
                # stop scoring once we touch the goal — a rollout that reaches
                # the goal mid-horizon should not be penalized for whatever
                # comes after.
                d2 = float(np.sum((rollout[h] - gl) ** 2))
                if d2 <= gr2:
                    reaches_goal = True
                    break
                if self._occupied(occ, rollout[h]):
                    collision_pen += 1
                cell_h = self._cell(rollout[h], occ.shape)
                ctg_h = float(ctg[cell_h]) if np.isfinite(ctg[cell_h]) else unreachable_penalty
                ctg_sum_until += ctg_h
                if ctg_h < ctg_min:
                    ctg_min = ctg_h
                steps_until = h
            smooth_pen = 0.0
            if self._prev_action is not None:
                smooth_pen = float(np.linalg.norm(v - self._prev_action))
            if reaches_goal:
                # large negative offset so any goal-reaching rollout dominates
                cost = -1e6 + self.w_smooth * smooth_pen
            else:
                ctg_avg = ctg_sum_until / max(1, steps_until)
                cost = (
                    self.w_goal * (0.5 * ctg_avg + 0.5 * ctg_min)
                    + self.w_obs * collision_pen
                    + self.w_smooth * smooth_pen
                )
            if cost < best_cost:
                best_cost = cost
                best_rollout = rollout[: steps_until + 1] if steps_until > 0 else rollout
                best_action = v

        assert best_rollout is not None and best_action is not None
        self._prev_action = best_action
        return Plan(
            waypoints=best_rollout[1:],
            target_velocity=best_action,
            meta={"planner": "mpc", "cost": float(best_cost)},
        )
