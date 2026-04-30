"""AirSim bridge — sketch implementation.

This is a structural stub demonstrating that the `SimInterface` abstraction
maps cleanly onto the AirSim Python client. It is *not* exercised in CI
(would need an AirSim server). Run a real AirSim instance, then:

    pip install airsim
    uav-nav run examples/exp_airsim.yaml

The contract:

  - reset(seed)        → arms / disarms / teleports the drone back to start
  - step(velocity_cmd) → moveByVelocityAsync(...) for `dt` seconds
  - state              → current pose + velocity from the AirSim API
  - obstacle_map       → comes from a static occupancy map you load on the
                         scenario side; AirSim itself does not give you a
                         grid, so the scenario is the source of truth.

Real-world pitfalls (left as TODO comments below):
  - Time synchronization: AirSim runs in real time; the experiment runner
    expects fast-forward. Use `client.simPause(True)` + `simContinueForTime`.
  - Coordinate frames: AirSim uses NED; the framework uses ENU.
  - Async commands: moveByVelocityAsync returns a future — block on join().
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo


@SIM_REGISTRY.register("airsim")
class AirSimBridge(SimInterface):
    def __init__(self, dt: float, scenario: Any, host: str, port: int, vehicle: str) -> None:
        self.dt = dt
        self.scenario = scenario
        self.host = host
        self.port = port
        self.vehicle = vehicle
        self._client: Any = None
        self._state: SimState | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "AirSimBridge":
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 41451)),
            vehicle=str(cfg.get("vehicle", "Drone1")),
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import airsim  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover
                raise SystemExit(
                    "airsim package is not installed. Install with `pip install airsim` "
                    "and start an AirSim server before running this experiment."
                ) from e
            self._client = airsim.MultirotorClient(ip=self.host, port=self.port)
            self._client.confirmConnection()
            self._client.enableApiControl(True, self.vehicle)
            self._client.armDisarm(True, self.vehicle)
        return self._client

    def reset(self, *, seed: int | None = None) -> SimState:  # pragma: no cover
        client = self._ensure_client()
        if seed is not None:
            self.scenario.reseed(seed)
        client.reset()
        client.enableApiControl(True, self.vehicle)
        client.armDisarm(True, self.vehicle)
        # TODO: teleport to scenario.start (NED conversion)
        # TODO: takeoffAsync().join()
        start = np.asarray(self.scenario.start, dtype=float)
        self._state = SimState(t=0.0, position=start.copy(), velocity=np.zeros(start.shape[0]))
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:  # pragma: no cover
        client = self._ensure_client()
        v = np.asarray(command, dtype=float)
        # AirSim velocity command — assumes 3D vehicle with vz=0 for 2D scenarios
        vx, vy = float(v[0]), float(v[1])
        vz = float(v[2]) if v.shape[0] >= 3 else 0.0
        client.moveByVelocityAsync(vx, vy, vz, self.dt, vehicle_name=self.vehicle).join()
        # Read state back
        kin = client.getMultirotorState(vehicle_name=self.vehicle).kinematics_estimated
        pos = np.asarray([kin.position.x_val, kin.position.y_val, kin.position.z_val])
        vel = np.asarray([kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val])
        ndim = self.scenario.ndim
        assert self._state is not None
        self._state.position = pos[:ndim]
        self._state.velocity = vel[:ndim]
        self._state.t += self.dt
        collision = bool(client.simGetCollisionInfo(vehicle_name=self.vehicle).has_collided)
        goal_reached = bool(np.linalg.norm(self._state.position - self.scenario.goal) <= 1.5)
        return self._state.copy(), SimStepInfo(
            collision=collision, goal_reached=goal_reached, truncated=False
        )

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.scenario.goal, dtype=float)

    @property
    def obstacle_map(self) -> Any:
        # AirSim has no built-in occupancy grid; the scenario remains the
        # source of truth for what the planner sees.
        return self.scenario.occupancy
