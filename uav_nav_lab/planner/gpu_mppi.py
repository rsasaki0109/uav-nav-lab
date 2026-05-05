"""GPU MPPI — batched PyTorch rollout with softmax-weighted action selection.

Same interface as `MPPIPlanner` but the per-sample rollout loop is replaced
by a single batched tensor operation on GPU.  For n_samples=128/256 this
brings plan_dt from O(1000 ms) to O(10 ms), unlocking the rightward shift
of the compute Pareto curve.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ..predictor import Predictor, build_predictor
from ._grid import dijkstra_cost_to_go, inflate_obstacles, sample_unit_directions
from .base import PLANNER_REGISTRY, Plan, Planner


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@PLANNER_REGISTRY.register("gpu_mppi")
class GPUMPPIPlanner(Planner):
    def __init__(
        self,
        max_speed: float = 10.0,
        horizon: int = 60,
        dt_plan: float = 0.05,
        n_samples: int = 128,
        resolution: float = 1.0,
        inflate: int = 1,
        goal_radius: float = 1.5,
        safety_margin: float = 0.4,
        use_prediction: bool = True,
        wind: tuple[float, ...] = (),
        w_goal: float = 1.0,
        w_obs: float = 100.0,
        w_smooth: float = 0.05,
        temperature: float = 1.0,
        predictor: Predictor | None = None,
        device: str = "cuda",
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
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature!r}")
        self.temperature = float(temperature)
        self._predictor: Predictor = (
            predictor if predictor is not None else build_predictor(None)
        )
        self._prev_action: np.ndarray | None = None
        self._static_occ_inflated: np.ndarray | None = None
        self._ctg_cache: np.ndarray | None = None
        self._ctg_cache_goal: tuple[int, ...] | None = None
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GPUMPPIPlanner":
        return cls(
            max_speed=float(cfg.get("max_speed", 10.0)),
            horizon=int(cfg.get("horizon", 60)),
            dt_plan=float(cfg.get("dt_plan", 0.05)),
            n_samples=int(cfg.get("n_samples", 128)),
            resolution=float(cfg.get("resolution", 1.0)),
            inflate=int(cfg.get("inflate", 1)),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            safety_margin=float(cfg.get("safety_margin", 0.4)),
            use_prediction=bool(cfg.get("use_prediction", True)),
            wind=tuple(cfg.get("wind", ())),
            w_goal=float(cfg.get("w_goal", 1.0)),
            w_obs=float(cfg.get("w_obs", 100.0)),
            w_smooth=float(cfg.get("w_smooth", 0.05)),
            temperature=float(cfg.get("temperature", 1.0)),
            predictor=build_predictor(cfg.get("predictor")),
            device=str(cfg.get("device", "cuda")),
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
        else:
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
        if self.use_prediction and dynamic_obstacles:
            horizon_dts = np.arange(1, self.horizon + 1, dtype=float) * self.dt_plan
            pred_traj = self._predictor.predict(dynamic_obstacles, horizon_dts)[:, :, :ndim]
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
            return Plan(waypoints=np.asarray([gl], dtype=float), meta={"planner": "gpu_mppi"})

        if occ[self._cell(obs, occ.shape)] or occ[self._cell(gl, occ.shape)]:
            occ = occ_raw

        # Dijkstra cost-to-go — computed on CPU, cached per episode (minor cost)
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
        ctg_np = self._ctg_cache

        base = to_goal / dist_goal
        directions = sample_unit_directions(ndim, self.n_samples, base)
        actions_np = directions * self.max_speed

        wind_step = None
        if self._wind is not None and self._wind.size > 0:
            wind_step = np.zeros(ndim)
            nw = min(self._wind.size, ndim)
            wind_step[:nw] = self._wind[:nw]

        # --- GPU batched rollout ---
        device = self._device
        occ_t = _to_tensor(occ.astype(np.float32), device)
        ctg_t = _to_tensor(ctg_np, device)
        obs_t = _to_tensor(obs, device)
        gl_t = _to_tensor(gl, device)
        actions_t = _to_tensor(actions_np, device)
        gr2 = self.goal_radius ** 2
        max_finite = float(ctg_np[ctg_np < np.inf].max()) if np.any(ctg_np < np.inf) else 1e6
        unreachable_penalty = max_finite + 100.0

        # Rollout: obs + v * dt * h  for all samples and horizon steps
        # Shape: [n_samples, horizon, ndim]
        dt = self.dt_plan
        h = torch.arange(1, self.horizon + 1, dtype=torch.float32, device=device) * dt  # [H]
        rollouts = obs_t[None, None, :] + actions_t[:, None, :] * h[None, :, None]  # [S, H, D]
        if wind_step is not None:
            ws_t = _to_tensor(wind_step, device)
            rollouts = rollouts + ws_t[None, None, :] * h[None, :, None]

        # Cell indices for occupancy check
        shape_t = torch.tensor(list(occ.shape), dtype=torch.long, device=device)
        cell_indices_float = rollouts / self.resolution  # [S, H, D]
        cell_indices_raw = cell_indices_float.long()
        # Out-of-bounds detection: OOB counts as obstacle
        oob = (
            (cell_indices_raw < 0)
            | (cell_indices_raw >= shape_t[None, None, :])
        ).any(dim=-1)  # [S, H]
        cell_indices = cell_indices_raw.clamp(
            torch.zeros_like(shape_t), shape_t - 1
        )  # [S, H, D]

        # Collision: gather occupancy values at cell indices + OOB
        if ndim == 2:
            occ_collision = occ_t[cell_indices[:, :, 0], cell_indices[:, :, 1]].float()  # [S, H]
        else:
            occ_collision = occ_t[cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]].float()
        collision_mask = occ_collision + oob.float()  # OOB = collision

        collision_pen = collision_mask.sum(dim=1)  # [S]

        # CTG: gather cost-to-go values at each rollout position
        if ndim == 2:
            ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1]]  # [S, H]
        else:
            ctg_roll = ctg_t[cell_indices[:, :, 0], cell_indices[:, :, 1], cell_indices[:, :, 2]]

        ctg_roll = torch.where(torch.isfinite(ctg_roll), ctg_roll, torch.tensor(unreachable_penalty, device=device))
        ctg_min = ctg_roll.min(dim=1).values  # [S]
        ctg_avg = ctg_roll.mean(dim=1)  # [S]

        # Goal reach: check if any step is within goal_radius
        dist2 = ((rollouts - gl_t[None, None, :]) ** 2).sum(dim=-1)  # [S, H]
        reaches_goal_any = (dist2 <= gr2).any(dim=1)  # [S]
        first_goal_h = torch.where(reaches_goal_any, (dist2 <= gr2).float().argmax(dim=1), torch.tensor(self.horizon, device=device))

        # Cost computation (matches CPU mppi logic)
        smooth_pen = torch.zeros(self.n_samples, device=device)
        if self._prev_action is not None:
            prev_t = _to_tensor(self._prev_action, device)
            smooth_pen = torch.norm(actions_t - prev_t[None, :], dim=1)

        no_coll = collision_pen == 0
        clean_reach = reaches_goal_any & no_coll
        dirty_reach = reaches_goal_any & ~no_coll
        neither = ~reaches_goal_any

        costs = torch.empty(self.n_samples, device=device)
        costs[clean_reach] = -1e6 + self.w_smooth * smooth_pen[clean_reach]
        costs[dirty_reach] = (
            self.w_goal * ctg_avg[dirty_reach]
            + self.w_obs * collision_pen[dirty_reach]
            + self.w_smooth * smooth_pen[dirty_reach]
        )
        costs[neither] = (
            self.w_goal * (0.5 * ctg_avg[neither] + 0.5 * ctg_min[neither])
            + self.w_obs * collision_pen[neither]
            + self.w_smooth * smooth_pen[neither]
        )

        # MPPI softmax-weighted average
        cost_min = costs.min()
        weights = torch.exp(-(costs - cost_min) / self.temperature)
        weights = weights / weights.sum()
        chosen_action_t = (weights[:, None] * actions_t).sum(dim=0)
        speed = torch.norm(chosen_action_t)
        if speed > self.max_speed:
            chosen_action_t = chosen_action_t * (self.max_speed / speed)
        best_k = int(weights.argmax().item())

        # Build best rollout for visualisation (on CPU)
        best_rollout = rollouts[best_k].cpu().numpy()
        best_full = np.concatenate([obs.reshape(1, -1), best_rollout], axis=0)
        if reaches_goal_any[best_k]:
            best_full = best_full[: first_goal_h[best_k].item() + 2]

        chosen_action = chosen_action_t.cpu().numpy()
        self._prev_action = chosen_action

        return Plan(
            waypoints=best_full[1:],
            target_velocity=chosen_action,
            meta={
                "planner": "gpu_mppi",
                "cost_min": float(cost_min.item()),
                "weight_max": float(weights.max().item()),
                "weight_entropy": float((-weights * torch.log(weights + 1e-12)).sum().item()),
                "n_samples": self.n_samples,
                "device": str(device),
            },
        )
