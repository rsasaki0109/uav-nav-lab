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

Cameras:
  - Configure with `simulator.cameras: [{name, image_type}, …]`
    where `image_type` is one of `scene` (default), `depth_vis`,
    `depth_perspective`, `depth_planar`, `segmentation`,
    `surface_normals`, `infrared`. Compressed PNG bytes land at
    `state.extra["camera_images"][name]` after each step. Combined
    with `output.save_camera_frames: true` and the `uav-nav video`
    CLI verb, this gives a one-line path from sim to MP4 demo reels.
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
        cameras: list[Mapping[str, Any]] | None = None,
        depths: list[Mapping[str, Any]] | None = None,
        settle_after_reset: float = 0.0,
        settle_after_teleport: float = 0.0,
        client: Any = None,
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.host = host
        self.port = port
        self.vehicle = vehicle
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        self.wind = tuple(float(v) for v in wind)
        # Names of LiDAR sensors configured on the AirSim side (in
        # settings.json). Each step we pull `getLidarData(name)` and stash
        # the converted (N, 3) ENU point cloud at
        # state.extra["lidar_points"][name]. Empty / unset = no polling.
        self.lidars: list[str] = list(lidars or [])
        # Camera specs: each {name, image_type}. Compressed PNG bytes per
        # camera land at state.extra["camera_images"][name] after each step.
        # Empty / unset = no polling.
        self.cameras: list[dict[str, str]] = [
            {"name": str(c["name"]), "image_type": str(c.get("image_type", "scene"))}
            for c in (cameras or [])
        ]
        # Depth-camera specs: each {name, image_type=depth_planar by default,
        # fov_deg, width, height}. Per step we pull depth_planar with
        # pixels_as_float=True and stash a {depth, intrinsics} payload at
        # state.extra["depth_images"][name] for the depth_image_occupancy
        # sensor to consume. Width/height are required because AirSim's
        # ImageResponse carries them but we need the values *before* the
        # call to compute intrinsics from fov.
        self.depths: list[dict[str, Any]] = [
            {
                "name": str(d["name"]),
                "image_type": str(d.get("image_type", "depth_planar")),
                "fov_deg": float(d.get("fov_deg", 90.0)),
                "width": int(d.get("width", 256)),
                "height": int(d.get("height", 144)),
            }
            for d in (depths or [])
        ]
        # Sleep windows after `client.reset()` and after the teleport
        # to the scenario start, in seconds. Without these, real
        # AirSim sometimes carries a transient collision flag from
        # the just-cleared physics state into the first step(). Both
        # default to 0 so unit tests stay fast; real-server runs
        # should set settle_after_reset≈1.0 and settle_after_teleport
        # ≈0.3 in YAML (`simulator.settle_after_reset: 1.0`).
        self.settle_after_reset = float(settle_after_reset)
        self.settle_after_teleport = float(settle_after_teleport)
        # `client` lets tests inject a fake airsim client; in production
        # the real client is created lazily on first reset/step.
        self._client: Any = client
        self._state: SimState | None = None
        self._step_count = 0
        # Multi-drone runner sets this to False on every bridge except
        # sim 0. AirSim has a single shared physics clock — only one
        # bridge per global tick should call simContinueForTime, or the
        # world advances N×dt instead of dt. Sim 0 owns the time
        # advance; sims 1..N-1 just queue moveByVelocityAsync (which
        # AirSim holds while paused) and read kinematics post-tick.
        self._advance_scenario: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "AirSimBridge":
        lidars_cfg = cfg.get("lidars", []) or []
        cameras_cfg = cfg.get("cameras", []) or []
        depths_cfg = cfg.get("depths", []) or []
        # Wind vector in ENU (m/s). Converted to NED and pushed to AirSim
        # via simSetWind on reset so the physical wind matches the planner's
        # wind_belief sweep target.  Default (0,0,0) = no wind.
        wind_raw = cfg.get("wind", ()) or ()
        wind_tuple = (float(wind_raw[0]), float(wind_raw[1]), float(wind_raw[2])) if len(wind_raw) >= 2 else (0.0, 0.0, 0.0)
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 41451)),
            vehicle=str(cfg.get("vehicle", "Drone1")),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            max_steps=int(cfg.get("max_steps", 2000)),
            lidars=[str(name) for name in lidars_cfg],
            cameras=list(cameras_cfg),
            depths=list(depths_cfg),
            settle_after_reset=float(cfg.get("settle_after_reset", 0.0)),
            settle_after_teleport=float(cfg.get("settle_after_teleport", 0.0)),
            wind=wind_tuple,
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

    def _build_image_requests(self) -> list[Any]:
        """Translate the bridge's camera spec list to airsim.ImageRequest
        objects. Lazy-imports airsim so the bridge module imports cleanly
        in environments where airsim is not installed (CI, mock tests
        that inject a fake `airsim` into sys.modules)."""
        import airsim  # type: ignore[import-not-found]

        type_map = {
            "scene": airsim.ImageType.Scene,
            "depth_vis": airsim.ImageType.DepthVis,
            "depth_perspective": airsim.ImageType.DepthPerspective,
            "depth_planar": airsim.ImageType.DepthPlanar,
            "segmentation": airsim.ImageType.Segmentation,
            "surface_normals": airsim.ImageType.SurfaceNormals,
            "infrared": airsim.ImageType.Infrared,
        }
        return [
            # pixels_as_float=False, compress=True → response.image_data_uint8
            # is a PNG byte string ready to be written to disk.
            airsim.ImageRequest(
                spec["name"],
                type_map.get(spec["image_type"], airsim.ImageType.Scene),
                False,
                True,
            )
            for spec in self.cameras
        ]

    def _build_depth_requests(self) -> list[Any]:
        """ImageRequests for the depth-camera specs.

        Uses `pixels_as_float=True` and `compress=False` so the response
        carries `image_data_float` (a flat list of metres) rather than
        the colour-mapped PNG that the `cameras: [...]` path produces.
        Image type defaults to depth_planar — depth_perspective is also
        valid; depth_vis is *not* (it's the visualisation, not raw)."""
        import airsim  # type: ignore[import-not-found]

        type_map = {
            "depth_planar": airsim.ImageType.DepthPlanar,
            "depth_perspective": airsim.ImageType.DepthPerspective,
        }
        return [
            airsim.ImageRequest(
                spec["name"],
                type_map.get(spec["image_type"], airsim.ImageType.DepthPlanar),
                True,    # pixels_as_float
                False,   # compress
            )
            for spec in self.depths
        ]

    @staticmethod
    def _intrinsics_from_fov(fov_deg: float, width: int, height: int) -> dict[str, float]:
        """AirSim cameras use a horizontal fov; AirSim's pixels are square so
        fy = fx. Optical centre = image centre."""
        fov_rad = float(fov_deg) * np.pi / 180.0
        fx = (width / 2.0) / float(np.tan(fov_rad / 2.0))
        return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        client = self._ensure_client()
        if seed is not None:
            self.scenario.reseed(seed)
        # `client.reset()` is global in AirSim — it wipes every vehicle.
        # In multi-drone runs only the master bridge (sim 0) should
        # call it; the passive sims would otherwise clobber sim 0's
        # already-teleported drone every time they reset themselves.
        if self._advance_scenario:
            client.reset()
            # Set global wind via API (settings.json Wind not supported
            # by all AirSim builds).  ENU → NED.
            try:
                import airsim  # type: ignore[import-not-found]
                w = self.wind
                wind_ned = airsim.Vector3r(float(w[1]), float(w[0]), float(-w[2]) if len(w) > 2 else 0.0)
                client.simSetWind(wind_ned)
            except Exception:
                pass
            # Let the world settle after reset() before issuing API
            # calls. Without this, AirSim sometimes carries a transient
            # collision flag from the just-cleared state into the
            # first step().
            if hasattr(client, "simPause"):
                import time as _time
                _time.sleep(self.settle_after_reset)
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
                # Same reason: give the engine a tick to register the
                # new pose before the runner starts step()ing it.
                if hasattr(client, "simPause"):
                    import time as _time
                    _time.sleep(self.settle_after_teleport)
            except ImportError:  # pragma: no cover
                pass
        # Pause AirSim before returning so the drone holds its
        # teleported pose during the (possibly multi-second) first
        # replan. step() will simPause(False) → moveByVelocity →
        # simContinueForTime(dt) → simPause(True), so we hand control
        # back to the engine for exactly `dt` per call. Without this
        # pause, an armed multirotor at altitude can drift / fall
        # during long planner waits and trigger a t=0 collision.
        if hasattr(client, "simPause"):
            client.simPause(True)
        ndim = self.scenario.ndim
        self._state = SimState(t=0.0, position=start[:ndim].copy(), velocity=np.zeros(ndim))
        self._step_count = 0
        return self._state.copy()

    def step_command(self, command: np.ndarray) -> None:
        """Queue a velocity command.  If master, also handle
        simPause(False) → moveByVelocityAsync → simContinueForTime(dt)
        → simPause(True).  State is NOT read back — call
        :meth:`step_readback` after *all* bridges have issued their
        commands so readbacks see the fully-advanced physics tick.

        For multi-drone runs the runner calls this in passive-first
        order (passive sims queue while paused, then the master
        unpauses / continues / re-pauses), then calls step_readback
        on every bridge to pull the fresh kinematics."""
        assert self._state is not None, "call reset() first"
        client = self._ensure_client()
        v = np.asarray(command, dtype=float)
        v3 = np.zeros(3)
        v3[: min(3, v.size)] = v[:3]
        v_ned = _enu_to_ned(v3)
        if self._advance_scenario and hasattr(client, "simPause"):
            client.simPause(False)
        _future = client.moveByVelocityAsync(
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2]),
            self.dt,
            vehicle_name=self.vehicle,
        )
        if self._advance_scenario:
            if hasattr(client, "simContinueForTime"):
                client.simContinueForTime(self.dt)
            elif hasattr(client, "simPause"):  # pragma: no cover
                client.simPause(True)
            if hasattr(client, "simPause"):
                client.simPause(True)

    def step_readback(self) -> tuple[SimState, SimStepInfo]:
        """Read kinematics, sensors and collision after the physics tick."""
        client = self._ensure_client()
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
        if self.lidars:
            self._state.extra["lidar_points"] = {
                name: _ned_pointcloud_to_enu(
                    client.getLidarData(name, vehicle_name=self.vehicle).point_cloud
                )
                for name in self.lidars
            }
        if self.cameras:
            requests = self._build_image_requests()
            responses = client.simGetImages(requests, vehicle_name=self.vehicle)
            self._state.extra["camera_images"] = {
                spec["name"]: bytes(getattr(resp, "image_data_uint8", b"") or b"")
                for spec, resp in zip(self.cameras, responses)
            }
        if self.depths:
            depth_requests = self._build_depth_requests()
            depth_responses = client.simGetImages(depth_requests, vehicle_name=self.vehicle)
            depth_bag: dict[str, dict[str, Any]] = {}
            for spec, resp in zip(self.depths, depth_responses):
                floats = getattr(resp, "image_data_float", None)
                if not floats:
                    continue
                arr = np.asarray(list(floats), dtype=np.float32).reshape(
                    spec["height"], spec["width"]
                )
                depth_bag[spec["name"]] = {
                    "depth": arr,
                    "intrinsics": self._intrinsics_from_fov(
                        spec["fov_deg"], spec["width"], spec["height"]
                    ),
                }
            if depth_bag:
                self._state.extra["depth_images"] = depth_bag
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

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        self.step_command(command)
        return self.step_readback()

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        if getattr(self, "_goal_override", None) is not None:
            return np.asarray(self._goal_override, dtype=float)
        return np.asarray(self.scenario.goal, dtype=float)

    def set_goal(self, goal: np.ndarray) -> None:
        """Override the goal used by the goal-reached check (multi-drone)."""
        self._goal_override = np.asarray(goal, dtype=float).reshape(self.scenario.ndim)

    @property
    def obstacle_map(self) -> Any:
        return self.scenario.occupancy
