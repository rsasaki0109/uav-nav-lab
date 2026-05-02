"""AirSim bridge — wires the framework's `SimInterface` to Microsoft AirSim.

Not exercised in CI (would need an AirSim server) but the AirSim Python
client is mockable, so the logic that converts between AirSim's NED
convention and the framework's ENU is unit-tested via the
`AirSimBridge._to_airsim_velocity` / `._from_airsim_position` helpers.

Run a real AirSim instance, then:

    pip install airsim
    uav-nav run examples/exp_airsim.yaml

Contract:
  - reset(seed)        → resets the AirSim world, teleports to start (in NED)
  - step(velocity_cmd) → simPause → moveByVelocity for `dt` → simContinueForTime
                          → read kinematics back. The pause/continue dance
                          is what gives the experiment runner deterministic
                          fast-forward instead of real-time wall clock.
  - state              → ENU pose / velocity converted from NED kinematics
  - obstacle_map       → comes from the scenario; AirSim has no occupancy grid

Coordinate frames:
  - Framework: ENU (east-north-up, +z up).
  - AirSim:    NED (north-east-down, +z down).
  - We map (x, y, z)_ENU = (y, x, -z)_NED.

Async commands: AirSim's moveByVelocityAsync returns a future; we
join() before reading kinematics to avoid race-y "command in flight"
states. Combined with simPause / simContinueForTime, the step is
synchronous from the runner's perspective.

LiDAR sensors:
  - Configure on the AirSim side via settings.json (one entry per
    sensor with a unique name).
  - List the names in the bridge config (`simulator.lidars: [Lidar1, …]`)
    to have the bridge poll `getLidarData(name)` after each step and
    stash the converted (N, 3) ENU point cloud at
    `state.extra["lidar_points"][name]`. Empty list = no polling.
  - Downstream rasterization / fusion (point cloud → occupancy /
    distance field) is intentionally left to consumer code; this
    bridge does not pin a perception architecture.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo


def _enu_to_ned(p: np.ndarray) -> np.ndarray:
    """(x, y, z)_ENU → (y, x, -z)_NED. Pads / truncates to 3D."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(3)
    out[: p.size] = p[:3]
    return np.array([out[1], out[0], -out[2]])


def _ned_to_enu(p: np.ndarray) -> np.ndarray:
    """(y, x, -z)_NED → (x, y, z)_ENU."""
    p = np.asarray(p, dtype=float)
    return np.array([p[1], p[0], -p[2]])


def _ned_pointcloud_to_enu(point_cloud_flat: Any) -> np.ndarray:
    """AirSim's `LidarData.point_cloud` is a flat list of NED triples
    (x, y, z) in vehicle-local frame. Reshape to (N, 3) and convert to
    ENU with the same (x, y, z) → (y, x, -z) flip used for poses.

    Returns shape (N, 3) — empty (0, 3) array if the readout is empty
    or malformed (e.g. lidar not yet populated)."""
    arr = np.asarray(list(point_cloud_flat), dtype=float)
    if arr.size == 0 or arr.size % 3 != 0:
        return np.zeros((0, 3))
    ned = arr.reshape(-1, 3)
    enu = np.empty_like(ned)
    enu[:, 0] = ned[:, 1]
    enu[:, 1] = ned[:, 0]
    enu[:, 2] = -ned[:, 2]
    return enu


@SIM_REGISTRY.register("airsim")
class AirSimBridge(SimInterface):
    def __init__(
        self,
        dt: float,
        scenario: Any,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle: str = "Drone1",
        goal_radius: float = 1.5,
        max_steps: int = 2000,
        lidars: list[str] | None = None,
        client: Any = None,
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.host = host
        self.port = port
        self.vehicle = vehicle
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        # Names of LiDAR sensors configured on the AirSim side (in
        # settings.json). Each step we pull `getLidarData(name)` and stash
        # the converted (N, 3) ENU point cloud at
        # state.extra["lidar_points"][name]. Empty / unset = no polling.
        self.lidars: list[str] = list(lidars or [])
        # `client` lets tests inject a fake airsim client; in production
        # the real client is created lazily on first reset/step.
        self._client: Any = client
        self._state: SimState | None = None
        self._step_count = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "AirSimBridge":
        lidars_cfg = cfg.get("lidars", []) or []
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 41451)),
            vehicle=str(cfg.get("vehicle", "Drone1")),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            max_steps=int(cfg.get("max_steps", 2000)),
            lidars=[str(name) for name in lidars_cfg],
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
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

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        client = self._ensure_client()
        if seed is not None:
            self.scenario.reseed(seed)
        client.reset()
        client.enableApiControl(True, self.vehicle)
        client.armDisarm(True, self.vehicle)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        # Teleport via simSetVehiclePose, NED. Hover at the start
        # altitude implied by the ENU z-component.
        ned_start = _enu_to_ned(start)
        if hasattr(client, "simSetVehiclePose"):
            try:
                import airsim  # type: ignore[import-not-found]
                pose = airsim.Pose(
                    airsim.Vector3r(float(ned_start[0]), float(ned_start[1]), float(ned_start[2])),
                    airsim.to_quaternion(0.0, 0.0, 0.0),
                )
                client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.vehicle)
            except ImportError:  # pragma: no cover
                pass
        ndim = self.scenario.ndim
        self._state = SimState(t=0.0, position=start[:ndim].copy(), velocity=np.zeros(ndim))
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        client = self._ensure_client()
        # ENU velocity → NED. 2D scenarios pad vz=0.
        v = np.asarray(command, dtype=float)
        v3 = np.zeros(3)
        v3[: min(3, v.size)] = v[:3]
        v_ned = _enu_to_ned(v3)
        # Pause sim, run command for exactly dt, continue. This makes the
        # experiment runner's wall-clock-independent loop work against a
        # real-time engine. simPause + simContinueForTime exists in AirSim
        # >= 1.4; older versions need a manual moveByVelocity timeout.
        if hasattr(client, "simPause"):
            client.simPause(False)
        future = client.moveByVelocityAsync(
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2]),
            self.dt,
            vehicle_name=self.vehicle,
        )
        if hasattr(client, "simContinueForTime"):
            client.simContinueForTime(self.dt)
        else:  # pragma: no cover
            future.join()
        if hasattr(client, "simPause"):
            client.simPause(True)
        # Read kinematics back, NED → ENU.
        kin = client.getMultirotorState(vehicle_name=self.vehicle).kinematics_estimated
        pos_ned = np.array([kin.position.x_val, kin.position.y_val, kin.position.z_val])
        vel_ned = np.array(
            [kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val]
        )
        pos_enu = _ned_to_enu(pos_ned)
        vel_enu = _ned_to_enu(vel_ned)
        ndim = self.scenario.ndim
        self._state.position = pos_enu[:ndim]
        self._state.velocity = vel_enu[:ndim]
        self._state.t += self.dt
        self._step_count += 1
        # Poll any configured LiDAR sensors. We expose raw (N, 3) ENU
        # point clouds via state.extra["lidar_points"][name]; downstream
        # rasterization / fusion is the consumer's job — keeps this
        # bridge from committing to a particular perception architecture.
        if self.lidars:
            self._state.extra["lidar_points"] = {
                name: _ned_pointcloud_to_enu(
                    client.getLidarData(name, vehicle_name=self.vehicle).point_cloud
                )
                for name in self.lidars
            }
        # AirSim's collision flag is the source of truth in physics-simulated
        # worlds; the scenario's occupancy is only used for planner input.
        collision = bool(client.simGetCollisionInfo(vehicle_name=self.vehicle).has_collided)
        goal = (
            self.scenario.goal
            if not hasattr(self, "_goal_override") or self._goal_override is None
            else self._goal_override
        )
        goal_reached = bool(np.linalg.norm(self._state.position - goal[:ndim]) <= self.goal_radius)
        truncated = self._step_count >= self.max_steps
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
    def obstacle_map(self) -> Any:
        return self.scenario.occupancy
