"""MPC + CHOMP smoothing — MPC picks the rollout, CHOMP polishes it.

Plain sampling MPC commits to a constant velocity for the whole horizon
and applies it via `target_velocity` (no pure-pursuit). The rollout is
piecewise straight, so any direction switch between replans shows up as
a corner the controller chases. This wrapper

  1. delegates to a `SamplingMPCPlanner` for the rollout (full goal-aware
     scoring, dynamic-obstacle prediction — unchanged),
  2. prepends the current observation and runs a few CHOMP smoothing
     iterations over the full polyline with both endpoints clamped,
  3. emits the smoothed trajectory either as waypoints (pure-pursuit) or
     as a time-indexed velocity profile (controller tracks v(t) directly).

Output modes:
  - `output: "waypoints"` — clears target_velocity, pure-pursuit on the
    smoothed waypoints. Original PR #21 mode; the head-to-head there
    found this *worse* on per-step |Δcmd| than plain MPC because
    pure-pursuit re-aims every control step (smoothness lives in the
    constant-velocity bypass, not in the path).
  - `output: "velocity_profile"` — derives per-step velocities from the
    smoothed path (finite-difference / dt_plan) and emits a time-indexed
    profile. The runner's velocity-tracking mode applies them in order,
    so the controller follows the *smoothed velocity* directly. This is
    the architectural fix the PR #21 finding pointed at.
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
        w_action_jump: float = 0.0,
        epsilon: float = 2.0,
        smooth_resolution: float = 1.0,
        smooth_inflate: int = 0,
        output: str = "waypoints",
    ) -> None:
        self._mpc = mpc
        self.max_speed = mpc.max_speed
        self.n_smooth_iters = int(n_smooth_iters)
        self.learning_rate = float(learning_rate)
        self.max_step_norm = float(max_step_norm)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        # Penalises (vel[0] - prev_emitted_velocity)² inside CHOMP descent,
        # where vel[0] = (x[1] - x[0]) / dt_plan. Active only when the
        # planner has a previous emission to reference (so the first replan
        # of an episode is unconstrained). Designed for `output:
        # velocity_profile` mode; for `output: waypoints` it just makes the
        # smoothed first segment match the previous initial velocity, which
        # is mostly cosmetic since pure-pursuit ignores the velocity. PR
        # #21 / #22's null result identified the replan-boundary jump as
        # the candidate load-bearing factor — this knob is the direct
        # ablation of that hypothesis.
        self.w_action_jump = float(w_action_jump)
        self.epsilon = float(epsilon)
        self.smooth_resolution = float(smooth_resolution)
        self.smooth_inflate = int(smooth_inflate)
        if output not in ("waypoints", "velocity_profile"):
            raise ValueError(
                f"output must be 'waypoints' or 'velocity_profile'; got {output!r}"
            )
        self.output = output
        # Last emitted vel[0] — used as the reference for w_action_jump.
        # Reset to None at episode start. Approximation: assumes the
        # controller is still on the previous profile's first sample at
        # replan time; for replan_period ≈ 1-2 dt_plan this is close, for
        # larger gaps the controller has already moved ahead but vel[0]
        # is still the cleanest reference signal the planner has access to.
        self._prev_emitted_velocity: np.ndarray | None = None
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
            w_action_jump=float(cfg.get("w_action_jump", 0.0)),
            epsilon=float(cfg.get("epsilon", 2.0)),
            smooth_resolution=float(cfg.get("smooth_resolution", 1.0)),
            smooth_inflate=int(cfg.get("smooth_inflate", 0)),
            output=str(cfg.get("output", "waypoints")),
        )

    def reset(self) -> None:
        self._mpc.reset()
        self._prev_emitted_velocity = None

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

        # Action-jump cost contribution. Cost = w_jump · ||(x[1] - x[0])/dt
        # - prev_emitted||²; gradient w.r.t. x[1] is
        #   2 · w_jump · ((x[1] - x[0])/dt² - prev_emitted/dt).
        # x[1] is the *first* interior waypoint, so this term enters
        # grad_int[0]. Active only when prev_emitted is known and the
        # weight is non-zero.
        dt_plan = float(self._mpc.dt_plan)
        use_jump_cost = (
            self.w_action_jump > 0.0
            and self._prev_emitted_velocity is not None
            and n >= 3
        )
        prev_v = (
            np.asarray(self._prev_emitted_velocity, dtype=float)[:ndim]
            if use_jump_cost else None
        )

        for _ in range(self.n_smooth_iters):
            _c, grad_obs = _obstacle_cost_and_grad(
                x, dist, self.epsilon, self.smooth_resolution
            )
            grad_smooth_int = k_int @ x[1:-1] + K_endpts @ endpts
            grad_int = (
                self.w_smooth * grad_smooth_int + self.w_obs * grad_obs[1:-1]
            )
            if use_jump_cost:
                vel0 = (x[1] - x[0]) / dt_plan
                grad_int[0] = grad_int[0] + (2.0 * self.w_action_jump / dt_plan) * (
                    vel0 - prev_v
                )
            step = self.learning_rate * (K_int_inv @ grad_int)
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            scale = np.minimum(
                1.0, self.max_step_norm / np.maximum(step_norms, 1e-12)
            )
            x[1:-1] = x[1:-1] - step * scale

        # Drop the prepended observation.
        smoothed = x[1:]
        meta = dict(base.meta)
        meta["planner"] = "mpc_chomp"
        meta["smoothed"] = True
        meta["n_smooth_iters"] = self.n_smooth_iters
        meta["output"] = self.output

        if self.output == "velocity_profile":
            # Per-step velocity from the smoothed path (forward differences
            # over dt_plan, matching MPC's rollout time grid). Length is
            # n - 1 = original waypoint count; the last entry repeats the
            # final velocity so the controller has something to apply if
            # replan is late. Clearing target_velocity is essential — if it
            # were set, the runner would prefer it over the profile.
            dt = float(self._mpc.dt_plan)
            full = np.vstack([np.atleast_2d(x[0]), smoothed])  # (n, ndim)
            vel = np.diff(full, axis=0) / dt                    # (n-1, ndim)
            # Cache for the next replan's action-jump cost reference.
            self._prev_emitted_velocity = vel[0].copy()
            meta["w_action_jump"] = self.w_action_jump
            return Plan(
                waypoints=smoothed,
                target_velocity=None,
                velocity_profile=vel,
                profile_dt=dt,
                meta=meta,
            )
        # `waypoints` mode: clear target_velocity so the runner falls back
        # to pure-pursuit on the smoothed path. (PR #21 baseline mode.)
        return Plan(waypoints=smoothed, target_velocity=None, meta=meta)
