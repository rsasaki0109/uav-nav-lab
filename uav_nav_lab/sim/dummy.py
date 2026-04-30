"""2D point-mass dummy simulator.

Simple Euler-integrated kinematics with a 2D occupancy grid for collisions.
Used to validate the framework end-to-end without spinning up AirSim or PX4.
The same `SimInterface` surface is what the AirSim/ROS2 backends will fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo


@dataclass
class _DummyParams:
    dt: float = 0.05
    max_steps: int = 2000
    max_accel: float = 50.0  # m/s^2 — keep loose so high-speed runs aren't bottlenecked
    goal_radius: float = 1.0
    drone_radius: float = 0.4


@SIM_REGISTRY.register("dummy_2d")
class Dummy2DSim(SimInterface):
    """2D headless point-mass over an occupancy grid.

    Command convention: 2-D velocity setpoint. The integrator clamps the
    instantaneous accel to `max_accel` so step responses stay realistic.
    """

    def __init__(self, params: _DummyParams, scenario: Any) -> None:
        self.p = params
        self.dt = params.dt
        self.scenario = scenario  # GridWorldScenario
        self._state: SimState | None = None
        self._step_count = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "Dummy2DSim":
        params = _DummyParams(
            dt=float(cfg.get("dt", 0.05)),
            max_steps=int(cfg.get("max_steps", 2000)),
            max_accel=float(cfg.get("max_accel", 50.0)),
            goal_radius=float(cfg.get("goal_radius", 1.0)),
            drone_radius=float(cfg.get("drone_radius", 0.4)),
        )
        return cls(params, scenario)

    def reset(self, *, seed: int | None = None) -> SimState:
        if seed is not None:
            self.scenario.reseed(seed)
        start = np.asarray(self.scenario.start, dtype=float)
        self._state = SimState(t=0.0, position=start.copy(), velocity=np.zeros(2))
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        cmd = np.asarray(command, dtype=float).reshape(2)

        # accel-limited tracking of velocity setpoint
        dv = cmd - self._state.velocity
        max_dv = self.p.max_accel * self.dt
        norm = float(np.linalg.norm(dv))
        if norm > max_dv:
            dv *= max_dv / norm
        self._state.velocity = self._state.velocity + dv
        self._state.position = self._state.position + self._state.velocity * self.dt
        self._state.t += self.dt
        self._step_count += 1

        collision = self.scenario.is_collision(self._state.position, self.p.drone_radius)
        goal_reached = bool(
            np.linalg.norm(self._state.position - self.scenario.goal) <= self.p.goal_radius
        )
        truncated = self._step_count >= self.p.max_steps
        return self._state.copy(), SimStepInfo(
            collision=collision, goal_reached=goal_reached, truncated=truncated
        )

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.scenario.goal, dtype=float)

    @property
    def obstacle_map(self) -> np.ndarray:
        return self.scenario.occupancy
