"""CHOMP — gradient-based trajectory smoothing planner.

CHOMP (Covariant Hamiltonian Optimization for Motion Planning, Zucker
et al. 2013) optimises a fixed-length sequence of waypoints between
start and goal under

    U(x) = w_smooth · ||A x||² / 2  +  w_obs · sum_i c(x_i)

where `A` is the second-difference (acceleration) matrix and `c(·)` is
the standard CHOMP obstacle potential built from a distance field:

    c(x) = -d(x) + ε/2                    if d(x) < 0   (inside obstacle)
           (1/(2ε)) (d(x) - ε)²            if 0 ≤ d(x) ≤ ε
           0                               otherwise

Updates use the M⁻¹-preconditioned step from the original paper, where
M is the interior block of the smoothness Hessian K = AᵀA. This is the
key reason CHOMP is well-behaved at non-trivial trajectory lengths —
plain gradient descent on AᵀA x diverges once n is large enough that
the largest eigenvalue exceeds 2/lr (≈ n=20 with lr=0.5), but
preconditioned descent stays stable for lr ≤ 1 regardless. Endpoints
stay clamped by optimising only the interior block and folding the
endpoint contribution into the gradient as a constant offset.

Init strategy: straight line between observation and goal. Pair with
`replan_period` ≪ episode duration so the planner can escape local
minima the way RRT does — by re-initialising from a fresher state each
replan.

Tradeoff vs the other planners:
  - vs RRT/RRT*: produces *smoother* paths (continuous gradient on
    trajectory smoothness) at fixed cost-per-replan, but is local —
    if the straight-line init crosses too thick an obstacle the
    optimiser cannot tunnel out and `status` returns `local_minimum`.
  - vs MPC (the framework's Pareto-saturated default): MPC samples
    velocities directly and re-decides every replan; CHOMP optimises a
    *path* and follows it pure-pursuit. CHOMP wins on smoothness;
    MPC wins on dynamic-obstacle reactivity.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._grid import inflate_obstacles
from .base import PLANNER_REGISTRY, Plan, Planner
from .rrt import RRTPlanner


def _distance_field(
    occ: np.ndarray, resolution: float, cap: float
) -> np.ndarray:
    """Per-cell Euclidean distance (in world units) to the nearest occupied
    cell, capped at `cap`. Brute-force vectorised — O(M·K) memory where
    M = #cells and K = #obstacles. Fine for the framework's typical grids
    (50² up to 40×40×12); large 3D grids should pre-inflate to keep K
    tractable.

    The cap keeps obstacle-free cells finite so centred-difference gradients
    elsewhere don't underflow to +inf − +inf = NaN. Anything beyond the
    CHOMP `epsilon` already contributes zero cost, so any cap > ε is sound."""
    obstacle_idx = np.argwhere(occ)
    if obstacle_idx.shape[0] == 0:
        return np.full(occ.shape, float(cap), dtype=np.float64)
    grid = np.indices(occ.shape).reshape(occ.ndim, -1).T  # (M, ndim)
    diff = grid[:, None, :] - obstacle_idx[None, :, :]    # (M, K, ndim)
    d_cells = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
    out = np.minimum(d_cells * float(resolution), float(cap))
    return out.reshape(occ.shape)


def _smoothness_hessian(n: int) -> np.ndarray:
    """A^T A for the (n-2, n) second-difference matrix A. Same Hessian for
    every spatial dimension since smoothness is separable across axes."""
    if n < 3:
        return np.zeros((n, n), dtype=np.float64)
    a = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        a[i, i] = 1.0
        a[i, i + 1] = -2.0
        a[i, i + 2] = 1.0
    return a.T @ a


def _resample_polyline(wps: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly `n` points along its arc length.

    The RRT/RRT* planners return variable-length, unevenly-spaced
    waypoint sequences. CHOMP needs a fixed n with reasonably uniform
    spacing so the smoothness Hessian's per-waypoint scale is consistent.
    Linear arc-length parameterisation does both at once.

    Returns shape (n, ndim). For n ≤ 1 or single-point polylines the
    output is just the (repeated) start point."""
    wps = np.asarray(wps, dtype=float)
    if wps.shape[0] <= 1 or n <= 1:
        return np.repeat(wps[:1], max(n, 1), axis=0)
    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.repeat(wps[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, wps.shape[1]), dtype=float)
    for k in range(wps.shape[1]):
        out[:, k] = np.interp(targets, cum, wps[:, k])
    return out


def _obstacle_cost_and_grad(
    x: np.ndarray, dist_field: np.ndarray, epsilon: float, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-waypoint CHOMP obstacle potential + its spatial gradient.

    Gradient is taken via centred finite differences on the precomputed
    cost field — cheap and numpy-only. Out-of-bounds queries clamp; we
    rely on the runner to keep the drone inside the world."""
    n, ndim = x.shape
    cells = np.clip(
        np.round(x / resolution).astype(int),
        0, np.array(dist_field.shape, dtype=int) - 1,
    )
    d = dist_field[tuple(cells.T)]                       # (n,)
    eps = float(epsilon)
    # CHOMP standard potential.
    c = np.where(
        d < 0.0,
        -d + eps / 2.0,
        np.where(d <= eps, (d - eps) ** 2 / (2.0 * eps), 0.0),
    )
    # Gradient via finite differences on `c` along each axis: ∂c/∂x_k
    # estimated from neighbouring cells of `cells` along axis k.
    grad = np.zeros_like(x)
    for k in range(ndim):
        plus = cells.copy()
        minus = cells.copy()
        plus[:, k] = np.clip(plus[:, k] + 1, 0, dist_field.shape[k] - 1)
        minus[:, k] = np.clip(minus[:, k] - 1, 0, dist_field.shape[k] - 1)
        d_plus = dist_field[tuple(plus.T)]
        d_minus = dist_field[tuple(minus.T)]
        # ∂c/∂d at this point (same case-split as above).
        dcdd = np.where(
            d < 0.0,
            -1.0,
            np.where(d <= eps, (d - eps) / eps, 0.0),
        )
        # ∂d/∂x_k via centred difference; resolution converts cell→world.
        ddx = (d_plus - d_minus) / (2.0 * resolution)
        grad[:, k] = dcdd * ddx
    return c, grad


@PLANNER_REGISTRY.register("chomp")
class ChompPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        replan_period: float = 0.5,
        n_waypoints: int = 30,
        n_iters: int = 100,
        learning_rate: float = 0.05,
        max_step_norm: float = 1.0,
        w_smooth: float = 1.0,
        w_obs: float = 5.0,
        epsilon: float = 2.0,
        resolution: float = 1.0,
        inflate: int = 0,
        goal_tolerance: float = 1.5,
        init: str = "straight",
        rrt_max_samples: int = 1000,
        rrt_step_size: float = 2.0,
        rrt_goal_tolerance: float = 1.5,
        rrt_goal_bias: float = 0.1,
        rrt_seed: int = 0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.replan_period = float(replan_period)
        self.n_waypoints = int(n_waypoints)
        self.n_iters = int(n_iters)
        self.learning_rate = float(learning_rate)
        self.max_step_norm = float(max_step_norm)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.epsilon = float(epsilon)
        self.resolution = float(resolution)
        self.inflate = int(inflate)
        self.goal_tolerance = float(goal_tolerance)
        # Init strategy. "straight" (default): linear interpolation from
        # observation to goal. "rrt": run an RRT sampling planner first,
        # resample its variable-length path to n_waypoints, then smooth.
        # RRT init lifts CHOMP out of the local-minimum trap on cluttered
        # scenarios at the cost of one RRT replan worth of compute.
        if init not in ("straight", "rrt"):
            raise ValueError(f"init must be 'straight' or 'rrt'; got {init!r}")
        self.init = init
        self._rrt: RRTPlanner | None = None
        if init == "rrt":
            self._rrt = RRTPlanner(
                max_speed=self.max_speed,
                max_samples=int(rrt_max_samples),
                step_size=float(rrt_step_size),
                goal_tolerance=float(rrt_goal_tolerance),
                goal_bias=float(rrt_goal_bias),
                resolution=self.resolution,
                inflate=self.inflate,
                seed=int(rrt_seed),
            )
        # Hessians only depend on n_waypoints — cache across replans. The
        # M⁻¹ block is the interior-only inverse used to precondition each
        # step (avoiding the divergence that plain GD on `K` exhibits for
        # n ≳ 20). Tiny λ keeps `K_int` numerically invertible.
        self._K: np.ndarray | None = None
        self._K_int_inv: np.ndarray | None = None
        self._K_endpts: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ChompPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            replan_period=float(cfg.get("replan_period", 0.5)),
            n_waypoints=int(cfg.get("n_waypoints", 30)),
            n_iters=int(cfg.get("n_iters", 100)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_step_norm=float(cfg.get("max_step_norm", 1.0)),
            w_smooth=float(cfg.get("w_smooth", 1.0)),
            w_obs=float(cfg.get("w_obs", 5.0)),
            epsilon=float(cfg.get("epsilon", 2.0)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 0)),
            goal_tolerance=float(cfg.get("goal_tolerance", 1.5)),
            init=str(cfg.get("init", "straight")),
            rrt_max_samples=int(cfg.get("rrt_max_samples", 1000)),
            rrt_step_size=float(cfg.get("rrt_step_size", 2.0)),
            rrt_goal_tolerance=float(cfg.get("rrt_goal_tolerance", 1.5)),
            rrt_goal_bias=float(cfg.get("rrt_goal_bias", 0.1)),
            rrt_seed=int(cfg.get("rrt_seed", 0)),
        )

    def reset(self) -> None:
        # Forward to the inner RRT (re-seeds its sample stream per episode).
        if self._rrt is not None:
            self._rrt.reset()

    def plan(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        obstacle_map: Any,
        *,
        dynamic_obstacles: list[dict] | None = None,  # noqa: ARG002
    ) -> Plan:
        occ_raw = np.asarray(obstacle_map, dtype=bool)
        ndim = occ_raw.ndim
        occ = inflate_obstacles(occ_raw, self.inflate)
        start = np.asarray(observation, dtype=float)[:ndim]
        gl = np.asarray(goal, dtype=float)[:ndim]

        n = max(4, self.n_waypoints)  # need n≥4 so the interior block is ≥2×2
        # Cache the smoothness Hessian and its interior inverse. K factors as
        #   ∇U_smooth = K @ x      (n, ndim)
        # Splitting into interior (i = 1..n-2) and endpoint contributions:
        #   ∇U_smooth[1:-1] = K[1:-1, 1:-1] @ x[1:-1]
        #                     + K[1:-1, [0, -1]] @ [x_0, x_{-1}]
        if self._K is None or self._K.shape[0] != n:
            self._K = _smoothness_hessian(n)
            k_int = self._K[1:-1, 1:-1]
            # Tiny ridge keeps `K_int` well-conditioned at small n.
            self._K_int_inv = np.linalg.inv(k_int + 1e-6 * np.eye(n - 2))
            self._K_endpts = self._K[1:-1][:, [0, -1]]            # (n-2, 2)

        # Init: straight-line by default; optional RRT path resampled to n
        # waypoints. RRT init is what lets CHOMP escape local minima the
        # straight-line init falls into (cluttered scenarios, wraparound
        # detours). On RRT failure (no path within max_samples), fall back
        # to straight-line so the optimiser still produces something.
        init_used = self.init
        if self._rrt is not None:
            rrt_plan = self._rrt.plan(start, gl, occ_raw)
            if rrt_plan.meta.get("status") == "ok" and rrt_plan.waypoints.shape[0] >= 2:
                x = _resample_polyline(rrt_plan.waypoints, n)
            else:
                init_used = "rrt_fallback_straight"
                ts = np.linspace(0.0, 1.0, n)
                x = start[None, :] + (gl - start)[None, :] * ts[:, None]
        else:
            ts = np.linspace(0.0, 1.0, n)
            x = start[None, :] + (gl - start)[None, :] * ts[:, None]   # (n, ndim)

        # Cap at 2×ε; anything farther contributes zero gradient anyway.
        dist = _distance_field(occ, self.resolution, cap=2.0 * self.epsilon)

        endpts = np.stack([x[0], x[-1]])                           # (2, ndim)
        k_int = self._K[1:-1, 1:-1]
        for _ in range(self.n_iters):
            _c, grad_obs = _obstacle_cost_and_grad(
                x, dist, self.epsilon, self.resolution
            )
            # Smoothness gradient at interior points only (endpoints clamped).
            grad_smooth_int = k_int @ x[1:-1] + self._K_endpts @ endpts
            grad_int = (
                self.w_smooth * grad_smooth_int
                + self.w_obs * grad_obs[1:-1]
            )
            # M⁻¹-preconditioned step. The preconditioner makes the smoothness
            # part naturally stable, but it amplifies low-frequency components
            # of the obstacle gradient — without per-step clipping the
            # trajectory can swing wildly outside the world the first few
            # iterations. Clipping per-waypoint step norm to `max_step_norm`
            # keeps the optimiser monotone in practice.
            step = self.learning_rate * (self._K_int_inv @ grad_int)
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.max_step_norm / np.maximum(step_norms, 1e-12))
            x[1:-1] = x[1:-1] - step * scale

        # Status: ok if no waypoint is inside an obstacle (using the inflated
        # map for the safety check, raw occ as the hard floor).
        cells = np.clip(
            np.round(x / self.resolution).astype(int),
            0, np.array(occ_raw.shape, dtype=int) - 1,
        )
        in_obstacle = bool(occ_raw[tuple(cells.T)].any())
        status = "local_minimum" if in_obstacle else "ok"

        return Plan(
            waypoints=x.astype(float),
            meta={
                "planner": "chomp",
                "status": status,
                "n_waypoints": n,
                "n_iters": self.n_iters,
                "init": init_used,
            },
        )
