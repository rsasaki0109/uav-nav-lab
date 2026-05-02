"""MPC + CHOMP smoothing — MPC picks the rollout, CHOMP polishes it.

Plain sampling MPC commits to a constant velocity for the whole horizon
and applies it via `target_velocity` (no pure-pursuit). The rollout is
piecewise straight, so any direction switch between replans shows up as
a corner the controller chases. This wrapper

  1. delegates to a `SamplingMPCPlanner` for the rollout (full goal-aware
     scoring, dynamic-obstacle prediction — unchanged),
  2. prepends the current observation and runs a few CHOMP smoothing
     iterations over the full polyline with both endpoints clamped,
  3. clears `target_velocity` so the runner pure-pursues the smoothed
     path instead of the constant rollout velocity.

The point isn't to find a different *route* — that's MPC's job and CHOMP
local-only descent can't beat it. The point is to file off the corner at
each replan boundary so the velocity profile is gentler.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._grid import inflate_obstacles
from .base import PLANNER_REGISTRY, Plan, Planner
from .chomp import _distance_field, _obstacle_cost_and_grad, _smoothness_hessian
from .mpc import SamplingMPCPlanner


@PLANNER_REGISTRY.register("mpc_chomp")
class MPCChompPlanner(Planner):
    def __init__(
        self,
        mpc: SamplingMPCPlanner,
        n_smooth_iters: int = 15,
        learning_rate: float = 0.05,
        max_step_norm: float = 1.0,
        w_smooth: float = 1.0,
        w_obs: float = 5.0,
        epsilon: float = 2.0,
        smooth_resolution: float = 1.0,
        smooth_inflate: int = 0,
    ) -> None:
        self._mpc = mpc
        self.max_speed = mpc.max_speed
        self.n_smooth_iters = int(n_smooth_iters)
        self.learning_rate = float(learning_rate)
        self.max_step_norm = float(max_step_norm)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.epsilon = float(epsilon)
        self.smooth_resolution = float(smooth_resolution)
        self.smooth_inflate = int(smooth_inflate)
        # Hessian cache keyed by trajectory length; MPC rollouts vary in
        # length when a sample reaches the goal mid-horizon.
        self._K_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MPCChompPlanner":
        mpc_cfg = dict(cfg.get("mpc", {}))
        # YAML-friendly: MPC params default to the wrapper's max_speed if the
        # user only specified one set of speed/dt.
        mpc_cfg.setdefault("max_speed", float(cfg.get("max_speed", 10.0)))
        mpc = SamplingMPCPlanner.from_config(mpc_cfg)
        return cls(
            mpc=mpc,
            n_smooth_iters=int(cfg.get("n_smooth_iters", 15)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_step_norm=float(cfg.get("max_step_norm", 1.0)),
            w_smooth=float(cfg.get("w_smooth", 1.0)),
            w_obs=float(cfg.get("w_obs", 5.0)),
            epsilon=float(cfg.get("epsilon", 2.0)),
            smooth_resolution=float(cfg.get("smooth_resolution", 1.0)),
            smooth_inflate=int(cfg.get("smooth_inflate", 0)),
        )

    def reset(self) -> None:
        self._mpc.reset()

    def _hessians(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._K_cache.get(n)
        if cached is not None:
            return cached
        K = _smoothness_hessian(n)
        k_int = K[1:-1, 1:-1]
        K_int_inv = np.linalg.inv(k_int + 1e-6 * np.eye(n - 2))
        K_endpts = K[1:-1][:, [0, -1]]
        out = (K, K_int_inv, K_endpts)
        self._K_cache[n] = out
        return out

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,
    ) -> Plan:
        base = self._mpc.plan(
            observation, goal, obstacle_map, dynamic_obstacles=dynamic_obstacles
        )
        wps = np.asarray(base.waypoints, dtype=float)
        if wps.shape[0] < 3:
            # Too few points to smooth meaningfully — fall through unchanged.
            meta = dict(base.meta)
            meta["planner"] = "mpc_chomp"
            meta["smoothed"] = False
            return Plan(waypoints=wps, target_velocity=base.target_velocity, meta=meta)

        ndim = wps.shape[1]
        obs = np.asarray(observation, dtype=float)[:ndim]
        x = np.vstack([obs[None, :], wps])  # (n, ndim) with start clamped
        n = x.shape[0]

        K, K_int_inv, K_endpts = self._hessians(n)
        k_int = K[1:-1, 1:-1]
        endpts = np.stack([x[0], x[-1]])

        occ_raw = np.asarray(obstacle_map, dtype=bool)
        occ = inflate_obstacles(occ_raw, self.smooth_inflate)
        dist = _distance_field(occ, self.smooth_resolution, cap=2.0 * self.epsilon)

        for _ in range(self.n_smooth_iters):
            _c, grad_obs = _obstacle_cost_and_grad(
                x, dist, self.epsilon, self.smooth_resolution
            )
            grad_smooth_int = k_int @ x[1:-1] + K_endpts @ endpts
            grad_int = (
                self.w_smooth * grad_smooth_int + self.w_obs * grad_obs[1:-1]
            )
            step = self.learning_rate * (K_int_inv @ grad_int)
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            scale = np.minimum(
                1.0, self.max_step_norm / np.maximum(step_norms, 1e-12)
            )
            x[1:-1] = x[1:-1] - step * scale

        # Drop the prepended observation. Clearing target_velocity forces the
        # runner's pure-pursuit follower onto the smoothed waypoints; if we
        # left target_velocity set the smoothing would be cosmetic.
        smoothed = x[1:]
        meta = dict(base.meta)
        meta["planner"] = "mpc_chomp"
        meta["smoothed"] = True
        meta["n_smooth_iters"] = self.n_smooth_iters
        return Plan(waypoints=smoothed, target_velocity=None, meta=meta)
