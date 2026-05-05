"""Gymnasium environments wrapping the framework's voxel_world / grid_world.

Observation space:  [ego_x, ego_y, goal_x, goal_y, local_occ_flat...]
Action space:       continuous [vx, vy], bounded by max_speed
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..scenario import SCENARIO_REGISTRY
from ..sim import SIM_REGISTRY


class _NavEnv(gym.Env):
    """Shared base for grid/voxel navigation environments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_cfg: dict,
        max_speed: float = 5.0,
        dt: float = 0.05,
        max_steps: int = 600,
        goal_radius: float = 1.5,
        drone_radius: float = 0.4,
        inflate: int = 1,
        safety_margin: float = 0.5,
        local_occ_size: int = 5,
        success_reward: float = 10.0,
        collision_reward: float = -10.0,
        step_reward: float = -0.01,
        goal_bonus_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self._scenario_cfg = dict(scenario_cfg)
        self._max_speed = float(max_speed)
        self._dt = float(dt)
        self._max_steps = int(max_steps)
        self._goal_radius = float(goal_radius)
        self._drone_radius = float(drone_radius)
        self._inflate = int(inflate)
        self._safety_margin = float(safety_margin)
        self._local_size = int(local_occ_size)
        self._success_reward = float(success_reward)
        self._collision_reward = float(collision_reward)
        self._step_reward = float(step_reward)
        self._goal_bonus = float(goal_bonus_factor)

        self._scenario: Any = None
        self._sim: Any = None
        self._goal_pos: np.ndarray | None = None
        self._step_count: int = 0
        self._ndim: int = 2

        # Observation: ego(2) + goal(2) + local_occ(5x5=25) = 29
        obs_dim = 4 + local_occ_size * local_occ_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: continuous velocity [vx, vy]
        self.action_space = spaces.Box(
            low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32
        )

    def _build_scenario_and_sim(self) -> None:
        scenario_cls = SCENARIO_REGISTRY.get(self._scenario_cfg.get("type", "grid_world"))
        self._scenario = scenario_cls.from_config(self._scenario_cfg)
        self._ndim = self._scenario.ndim

        sim_cfg = {
            "type": "dummy_2d" if self._ndim == 2 else "dummy_3d",
            "dt": self._dt,
            "max_steps": self._max_steps,
            "goal_radius": self._goal_radius,
            "drone_radius": self._drone_radius,
            "max_accel": 50.0,
        }
        sim_cls = SIM_REGISTRY.get(sim_cfg["type"])
        self._sim = sim_cls.from_config(sim_cfg, self._scenario)

    def _get_obs(self, state: Any) -> np.ndarray:
        pos = state.position[:2]
        goal = self._goal_pos[:2]
        occ = self._sim.obstacle_map
        # Extract local occupancy around ego position
        half = self._local_size // 2
        local = np.zeros((self._local_size, self._local_size), dtype=np.float32)
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                cx = int(pos[0] + dx)
                cy = int(pos[1] + dy)
                lx = dx + half
                ly = dy + half
                if (0 <= cx < occ.shape[0] and 0 <= cy < occ.shape[1]
                        and 0 <= lx < self._local_size and 0 <= ly < self._local_size):
                    local[lx, ly] = 1.0 if occ[cx, cy] else 0.0
        obs = np.concatenate([pos, goal, local.flatten()]).astype(np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._build_scenario_and_sim()
        self._goal_pos = np.asarray(self._scenario.goal, dtype=float)
        state = self._sim.reset(seed=seed, initial_position=self._scenario.start)
        self._step_count = 0
        return self._get_obs(state), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        v = np.asarray(action, dtype=float)[: self._ndim]
        speed = float(np.linalg.norm(v))
        if speed > self._max_speed:
            v = v / speed * self._max_speed
        state, info = self._sim.step(v)
        self._step_count += 1

        goal_dist = float(np.linalg.norm(state.position[: self._ndim] - self._goal_pos[: self._ndim]))

        terminated = False
        truncated = False
        reward = self._step_reward

        if info.collision:
            reward += self._collision_reward
            terminated = True
        elif info.goal_reached or goal_dist <= self._goal_radius:
            reward += self._success_reward
            terminated = True
        else:
            # Shaping reward: progress toward goal
            prev_dist = float(np.linalg.norm(
                state.position[: self._ndim] - self._goal_pos[: self._ndim]
            ))
            reward += self._goal_bonus * (1.0 / max(prev_dist, 0.1))

        if self._step_count >= self._max_steps:
            truncated = True

        return self._get_obs(state), reward, terminated, truncated, {
            "goal_dist": goal_dist,
            "collision": info.collision,
            "goal_reached": info.goal_reached,
        }


class GridNavEnv(_NavEnv):
    """2D grid_world navigation environment."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scenario_cfg", {
            "type": "grid_world",
            "size": [50, 50],
            "start": [2.0, 2.0],
            "goal": [45.0, 45.0],
            "resolution": 1.0,
            "obstacles": {"type": "random", "count": 30, "seed": 7},
        })
        super().__init__(**kwargs)


class VoxelNavEnv(_NavEnv):
    """3D voxel_world navigation environment (2D action space)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scenario_cfg", {
            "type": "voxel_world",
            "size": [40, 40, 12],
            "start": [2.0, 2.0, 6.0],
            "goal": [37.0, 37.0, 6.0],
            "resolution": 1.0,
            "obstacles": {"type": "random", "count": 60, "seed": 7},
        })
        super().__init__(**kwargs)
