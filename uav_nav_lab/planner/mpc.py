"""Sampling-based MPC planner with an A* cost-to-go heuristic — N-D.

Each replan we

  1. run Dijkstra from the goal cell over the (inflated) occupancy grid to
     get a global cost-to-go map — the obstacle-aware "distance to goal".
  2. sample `n_samples` constant-velocity candidates covering all directions
     (full circle in 2D, Fibonacci sphere in 3D); the goal direction is
     always the first sample.
  3. roll each out for `horizon` steps and score by
        w_goal * (cost_to_go avg + cost_to_go min along rollout)
        + w_obs  * occupied_cells_traversed
        + w_smooth * |action - prev_action|
     A rollout that touches the goal radius mid-horizon dominates.

Returned `Plan.target_velocity` is applied directly by the runner — no
pure-pursuit aliasing under noisy observations.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..predictor import Predictor, build_predictor
from ._grid import dijkstra_cost_to_go, inflate_obstacles, sample_unit_directions
from .base import PLANNER_REGISTRY, Plan, Planner


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
        safety_margin: float = 0.4,
        use_prediction: bool = True,
        wind: tuple[float, ...] = (),
        w_goal: float = 1.0,
        w_obs: float = 100.0,
        w_smooth: float = 0.05,
        predictor: Predictor | None = None,
    ) -> None:
        self.max_speed = float(max_speed)
        self.horizon = int(horizon)
        self.dt_plan = float(dt_plan)
        self.n_samples = int(n_samples)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self.goal_radius = float(goal_radius)
        self.safety_margin = float(safety_margin)
        self.use_prediction = bool(use_prediction)
        self._wind = np.asarray(wind, dtype=float) if wind else None
        self.w_goal = float(w_goal)
        self.w_obs = float(w_obs)
        self.w_smooth = float(w_smooth)
        self._predictor: Predictor = predictor if predictor is not None else build_predictor(None)
        self._prev_action: np.ndarray | None = None
        # Per-episode caches: ctg / static-occ depend only on the static
        # obstacle layout + goal cell, both of which are stable within an
        # episode. The Dijkstra cost-to-go dominates 3D plan_dt, so caching
        # it makes 3D sweeps tractable. Cleared by reset().
        self._static_occ_inflated: np.ndarray | None = None
        self._ctg_cache: np.ndarray | None = None
        self._ctg_cache_goal: tuple[int, ...] | None = None

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
            safety_margin=float(cfg.get("safety_margin", 0.4)),
            use_prediction=bool(cfg.get("use_prediction", True)),
            wind=tuple(cfg.get("wind", ())),
            w_goal=float(cfg.get("w_goal", 1.0)),
            w_obs=float(cfg.get("w_obs", 100.0)),
            w_smooth=float(cfg.get("w_smooth", 0.05)),
            predictor=build_predictor(cfg.get("predictor")),
        )

    def reset(self) -> None:
        self._prev_action = None
        self._predictor.reset()
        self._static_occ_inflated = None
        self._ctg_cache = None
        self._ctg_cache_goal = None

    def _cell(self, p: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(np.clip(p[i] / self.resolution, 0, shape[i] - 1)) for i in range(len(shape))
        )

    def _mask_dynamic_cells(self, occ_raw: np.ndarray, d: Mapping[str, Any]) -> None:
        """Zero out cells inside a dynamic obstacle's footprint (in-place).

        The heuristic ignores movers — the rollout's sphere-sphere distance
        check is what enforces dynamic-obstacle avoidance. Mask is applied
        once per episode against the raw (un-inflated) occupancy.
        """
        pos = np.asarray(d.get("position", ()), dtype=float)
        if pos.size == 0:
            return
        radius = float(d.get("radius", 0.5))
        cells = max(1, int(np.ceil(radius / self.resolution)))
        ndim = occ_raw.ndim
        center = self._cell(pos[:ndim], occ_raw.shape)
        if ndim == 2:
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    cx, cy = center[0] + dx, center[1] + dy
                    if 0 <= cx < occ_raw.shape[0] and 0 <= cy < occ_raw.shape[1]:
                        occ_raw[cx, cy] = False
        else:  # 3D
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    for dz in range(-cells + 1, cells):
                        cx, cy, cz = center[0] + dx, center[1] + dy, center[2] + dz
                        if (
                            0 <= cx < occ_raw.shape[0]
                            and 0 <= cy < occ_raw.shape[1]
                            and 0 <= cz < occ_raw.shape[2]
                        ):
                            occ_raw[cx, cy, cz] = False

    def _occupied(self, occ: np.ndarray, p: np.ndarray) -> bool:
        ndim = occ.ndim
        coords = []
        for i in range(ndim):
            ci = int(p[i] / self.resolution)
            if not (0 <= ci < occ.shape[i]):
                return True  # OOB counts as obstacle
            coords.append(ci)
        return bool(occ[tuple(coords)])

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
        # Pre-compute predicted dynamic-obstacle trajectories once per replan
        # (rather than per-sample × per-step) via the configured predictor.
        # Shape: [n_obs, horizon, ndim]; r2_arr: [n_obs] of squared radii.
        if self.use_prediction and dynamic_obstacles:
            horizon_dts = np.arange(1, self.horizon + 1, dtype=float) * self.dt_plan
            pred_traj = self._predictor.predict(
                dynamic_obstacles, horizon_dts
            )[:, :, :ndim]
            r2_arr = np.array(
                [(float(d.get("radius", 0.5)) + self.safety_margin) ** 2 for d in dynamic_obstacles],
                dtype=float,
            )
        else:
            pred_traj = None
            r2_arr = None
        occ = inflate_obstacles(occ_raw, self.inflate)
        obs = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        to_goal = gl - obs
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < 1e-6:
            return Plan(waypoints=np.asarray([gl], dtype=float), meta={"planner": "mpc"})

        if occ[self._cell(obs, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        # Heuristic ctg lives on a cached static-only occupancy: dynamic
        # obstacles are masked out the first time we see them, since the
        # rollout already does proper sphere-distance avoidance. This keeps
        # the heuristic stable across replans (so we can cache once per
        # episode) and avoids the Dijkstra cost dominating 3D plan_dt.
        if self._static_occ_inflated is None or self._static_occ_inflated.shape != occ.shape:
            static_raw = occ_raw.copy()
            if dynamic_obstacles:
                for d in dynamic_obstacles:
                    self._mask_dynamic_cells(static_raw, d)
            self._static_occ_inflated = inflate_obstacles(static_raw, self.inflate)
            self._ctg_cache = None
            self._ctg_cache_goal = None

        goal_cell = self._cell(gl, self._static_occ_inflated.shape)
        if self._ctg_cache is None or self._ctg_cache_goal != goal_cell:
            self._ctg_cache = dijkstra_cost_to_go(self._static_occ_inflated, goal_cell)
            self._ctg_cache_goal = goal_cell
        ctg = self._ctg_cache
        max_finite = float(np.max(ctg[np.isfinite(ctg)])) if np.any(np.isfinite(ctg)) else 1e6
        unreachable_penalty = max_finite + 100.0

        base = to_goal / dist_goal
        directions = sample_unit_directions(ndim, self.n_samples, base)
        actions = directions * self.max_speed
        # Per-step displacement from external wind (if known to the planner).
        # Truncate / pad to scenario ndim so YAML can stay 2D-friendly.
        if self._wind is not None and self._wind.size > 0:
            wind_step = np.zeros(ndim)
            n = min(self._wind.size, ndim)
            wind_step[:n] = self._wind[:n]
        else:
            wind_step = None

        best_cost = np.inf
        best_rollout: np.ndarray | None = None
        best_action: np.ndarray | None = None
        gr2 = self.goal_radius * self.goal_radius
        for v in actions:
            rollout = np.empty((self.horizon + 1, ndim), dtype=float)
            rollout[0] = obs
            collision_pen = 0
            ctg_min = np.inf
            ctg_sum_until = 0.0
            steps_until = 0
            reaches_goal = False
            for h in range(1, self.horizon + 1):
                step = v * self.dt_plan
                if wind_step is not None:
                    step = step + wind_step * self.dt_plan
                rollout[h] = rollout[h - 1] + step
                d2 = float(np.sum((rollout[h] - gl) ** 2))
                if d2 <= gr2:
                    reaches_goal = True
                    break
                if self._occupied(occ, rollout[h]):
                    collision_pen += 1
                # predicted dynamic obstacles (precomputed by self._predictor)
                if pred_traj is not None:
                    diffs = pred_traj[:, h - 1, :] - rollout[h]
                    sep2 = np.sum(diffs * diffs, axis=1)
                    if np.any(sep2 <= r2_arr):
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
            if reaches_goal and collision_pen == 0:
                # Clean reach: huge bonus.
                cost = -1e6 + self.w_smooth * smooth_pen
            elif reaches_goal:
                # Reaches goal but with collisions on the way — the bonus is
                # withheld so a non-reaching, non-colliding alternative wins.
                ctg_avg = ctg_sum_until / max(1, steps_until)
                cost = (
                    self.w_goal * ctg_avg
                    + self.w_obs * collision_pen
                    + self.w_smooth * smooth_pen
                )
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
