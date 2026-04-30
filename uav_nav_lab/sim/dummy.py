"""N-D point-mass dummy simulator.

Simple Euler-integrated kinematics with a 2D / 3D occupancy grid for
collisions. The same class backs both `dummy_2d` and `dummy_3d`; dimension
is inferred from the scenario's `ndim` so YAML stays clean.
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
    max_accel: float = 50.0
    goal_radius: float = 1.0
    drone_radius: float = 0.4
    # Constant wind (m/s) added to drone velocity each step, plus optional
    # Gaussian gust on top. Ndim is detected from the scenario at construction.
    wind: tuple[float, ...] = ()
    gust_std: float = 0.0


class DummySim(SimInterface):
    """N-D headless point-mass over an occupancy grid (2D or 3D).

    Command convention: N-D velocity setpoint. The integrator clamps the
    instantaneous accel to `max_accel` so step responses stay realistic.
    """

    def __init__(
        self,
        params: _DummyParams,
        scenario: Any,
        *,
        advance_scenario: bool = True,
    ) -> None:
        self.p = params
        self.dt = params.dt
        self.scenario = scenario
        self._advance_scenario = bool(advance_scenario)
        self._ndim = scenario.ndim
        self._state: SimState | None = None
        self._step_count = 0
        # Per-drone goal override (set by the multi-drone runner). When unset
        # we delegate to the scenario's single goal — preserving single-drone
        # behavior unchanged.
        self._goal_override: np.ndarray | None = None
        wind = np.asarray(params.wind, dtype=float) if params.wind else np.zeros(self._ndim)
        if wind.shape[0] < self._ndim:
            wind = np.concatenate([wind, np.zeros(self._ndim - wind.shape[0])])
        self._wind = wind[: self._ndim]
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "DummySim":
        dist = dict(cfg.get("disturbance", {}))
        params = _DummyParams(
            dt=float(cfg.get("dt", 0.05)),
            max_steps=int(cfg.get("max_steps", 2000)),
            max_accel=float(cfg.get("max_accel", 50.0)),
            goal_radius=float(cfg.get("goal_radius", 1.0)),
            drone_radius=float(cfg.get("drone_radius", 0.4)),
            wind=tuple(dist.get("wind", ())),
            gust_std=float(dist.get("gust_std", 0.0)),
        )
        return cls(params, scenario)

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        if seed is not None:
            self.scenario.reseed(seed)
            self._rng = np.random.default_rng(seed)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float).reshape(self._ndim)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        self._state = SimState(t=0.0, position=start.copy(), velocity=np.zeros(self._ndim))
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        cmd = np.asarray(command, dtype=float).reshape(self._ndim)

        dv = cmd - self._state.velocity
        max_dv = self.p.max_accel * self.dt
        norm = float(np.linalg.norm(dv))
        if norm > max_dv:
            dv *= max_dv / norm
        self._state.velocity = self._state.velocity + dv
        # external disturbance: wind + gust. Affects position but does not
        # alter the controller's velocity tracking — the drone is "blown".
        disturbance = self._wind.copy()
        if self.p.gust_std > 0.0:
            disturbance = disturbance + self._rng.normal(
                0.0, self.p.gust_std, size=self._ndim
            )
        self._state.position = self._state.position + (
            self._state.velocity + disturbance
        ) * self.dt
        self._state.t += self.dt
        self._step_count += 1
        if self._advance_scenario:
            self.scenario.advance(self.dt)  # no-op for static-only scenarios

        collision = self.scenario.is_collision(self._state.position, self.p.drone_radius)
        goal_pos = (
            self._goal_override if self._goal_override is not None else self.scenario.goal
        )
        goal_reached = bool(
            np.linalg.norm(self._state.position - goal_pos) <= self.p.goal_radius
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
        if self._goal_override is not None:
            return np.asarray(self._goal_override, dtype=float)
        return np.asarray(self.scenario.goal, dtype=float)

    def set_goal(self, goal: np.ndarray) -> None:
        """Override the goal used by the goal-reached check (multi-drone)."""
        self._goal_override = np.asarray(goal, dtype=float).reshape(self._ndim)

    @property
    def obstacle_map(self) -> np.ndarray:
        return self.scenario.occupancy


# Register the same class under both names so the YAML stays explicit about
# the expected dimension. Mismatch (e.g. dummy_3d + grid_world) is caught at
# build time by the scenario's ndim.
SIM_REGISTRY.register("dummy_2d")(DummySim)
SIM_REGISTRY.register("dummy_3d")(DummySim)
SIM_REGISTRY.register("dummy")(DummySim)
