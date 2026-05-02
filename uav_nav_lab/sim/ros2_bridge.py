"""ROS 2 bridge — wires the framework's `SimInterface` to a ROS 2 stack.

Not exercised in CI (would need rclpy + a sim like Gazebo / Ignition /
PX4-SITL via MAVROS) but the adapter is mockable, so the publish-spin-
read plumbing is unit-tested via an injected fake adapter.

Run a real ROS 2 + sim stack publishing /odom and accepting /cmd_vel,
then:

    source /opt/ros/jazzy/setup.bash
    uav-nav run examples/exp_ros2.yaml

Contract:
  - reset(seed)        → re-seeds the scenario, optionally teleports via
                          adapter.teleport(...), spins once so the first
                          /odom message lands. Falls back to scenario.start
                          if no odom arrives within the dt window.
  - step(velocity_cmd) → publish a Twist on cmd_topic, spin for `dt` so
                          the latest /odom is consumed, read pose/velocity
                          back. Collision flag comes from /collision (or
                          False if the topic is unconfigured). Optional
                          LiDAR / camera readouts populate `state.extra`
                          mirroring the AirSim bridge — see below.
  - state              → ENU pose / velocity from the latest odom message.
  - obstacle_map       → comes from the scenario; this bridge does not
                          ingest a ROS occupancy grid.

Coordinate frames:
  - Framework: ENU (east-north-up, +z up).
  - ROS 2:     ENU per REP-103 for outdoor robotics — pass-through here.
    PX4-NED users should convert in their MAVROS / setpoint layer; this
    bridge does not flip frames.

Topology:
  - publish:   cmd_topic        geometry_msgs/Twist    (ENU velocity setpoint)
  - subscribe: odom_topic       nav_msgs/Odometry      (true pose+velocity)
  - subscribe: collision_topic  std_msgs/Bool          (optional)
  - subscribe: lidars[*]        sensor_msgs/PointCloud2 (optional)
  - subscribe: cameras[*]       sensor_msgs/Image       (optional)

LiDAR / camera readouts mirror the AirSim bridge so the same
`pointcloud_occupancy` sensor and `uav-nav video` CLI work transparently
across backends:
  - state.extra["lidar_points"][topic]  = (N, 3) ENU point cloud
  - state.extra["camera_images"][topic] = compressed PNG bytes

The adapter abstraction (`_Ros2Adapter` duck-typed surface) keeps the
bridge testable without rclpy installed. `_RclpyAdapter` is the
production implementation; tests inject a fake adapter that records
publishes and returns canned odom / lidar / camera data.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo


@SIM_REGISTRY.register("ros2")
class Ros2Bridge(SimInterface):
    def __init__(
        self,
        dt: float,
        scenario: Any,
        cmd_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        collision_topic: str | None = None,
        goal_radius: float = 1.5,
        max_steps: int = 2000,
        lidars: list[str] | None = None,
        cameras: list[str] | None = None,
        adapter: Any = None,
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.collision_topic = collision_topic
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        # PointCloud2 topics to subscribe to. Each step's latest message
        # lands at state.extra["lidar_points"][topic] as (N, 3) ENU points.
        self.lidars: list[str] = list(lidars or [])
        # sensor_msgs/Image topics. Each step's latest frame is encoded
        # to PNG bytes and stashed at state.extra["camera_images"][topic].
        self.cameras: list[str] = list(cameras or [])
        # `adapter` lets tests inject a fake; in production the real
        # rclpy-backed adapter is created lazily on first reset/step.
        self._adapter: Any = adapter
        self._state: SimState | None = None
        self._step_count = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "Ros2Bridge":
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            cmd_topic=str(cfg.get("cmd_topic", "/cmd_vel")),
            odom_topic=str(cfg.get("odom_topic", "/odom")),
            collision_topic=cfg.get("collision_topic"),
            goal_radius=float(cfg.get("goal_radius", 1.5)),
            max_steps=int(cfg.get("max_steps", 2000)),
            lidars=[str(t) for t in (cfg.get("lidars") or [])],
            cameras=[str(t) for t in (cfg.get("cameras") or [])],
        )

    def _ensure_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        self._adapter = _RclpyAdapter(
            cmd_topic=self.cmd_topic,
            odom_topic=self.odom_topic,
            collision_topic=self.collision_topic,
            lidar_topics=self.lidars,
            camera_topics=self.cameras,
        )
        return self._adapter

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        adapter = self._ensure_adapter()
        if seed is not None:
            self.scenario.reseed(seed)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        ndim = self.scenario.ndim
        # Optional teleport (Gazebo set_entity_state, Ignition /world/.../set_pose).
        # Silently no-op if the adapter does not implement it.
        teleport = getattr(adapter, "teleport", None)
        if callable(teleport):
            pos3 = np.zeros(3)
            pos3[:ndim] = start[:ndim]
            teleport(pos3)
        # Spin briefly so the first odom arrives. Fall back to scenario.start
        # if nothing lands within the dt window — the run can still progress.
        adapter.tick(self.dt)
        latest = adapter.latest_pose_velocity()
        if latest is not None:
            pos_enu, vel_enu = latest
            self._state = SimState(
                t=0.0,
                position=np.asarray(pos_enu, dtype=float)[:ndim].copy(),
                velocity=np.asarray(vel_enu, dtype=float)[:ndim].copy(),
            )
        else:
            self._state = SimState(
                t=0.0, position=start[:ndim].copy(), velocity=np.zeros(ndim)
            )
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        adapter = self._ensure_adapter()
        v = np.asarray(command, dtype=float)
        v3 = np.zeros(3)
        v3[: min(3, v.size)] = v[:3]  # 2D scenarios pad vz=0
        adapter.publish_velocity(float(v3[0]), float(v3[1]), float(v3[2]))
        adapter.tick(self.dt)
        latest = adapter.latest_pose_velocity()
        ndim = self.scenario.ndim
        if latest is not None:
            pos_enu, vel_enu = latest
            self._state.position = np.asarray(pos_enu, dtype=float)[:ndim].copy()
            self._state.velocity = np.asarray(vel_enu, dtype=float)[:ndim].copy()
        # else: keep previous state (sensor dropout); planner replan handles it.
        self._state.t += self.dt
        self._step_count += 1
        # Mirror AirSim bridge: surface latest sensor side-channel data
        # under the same state.extra keys so consumers (pointcloud_occupancy
        # sensor, uav-nav video CLI) work transparently across backends.
        if self.lidars:
            clouds = adapter.latest_lidar_clouds() if hasattr(adapter, "latest_lidar_clouds") else {}
            if clouds:
                self._state.extra["lidar_points"] = dict(clouds)
        if self.cameras:
            imgs = adapter.latest_camera_images() if hasattr(adapter, "latest_camera_images") else {}
            if imgs:
                self._state.extra["camera_images"] = dict(imgs)
        collision = bool(adapter.latest_collision())
        goal = np.asarray(self.scenario.goal, dtype=float)
        goal_reached = bool(
            np.linalg.norm(self._state.position - goal[:ndim]) <= self.goal_radius
        )
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


class _RclpyAdapter:  # pragma: no cover
    """Production adapter — owns an rclpy node, a Twist publisher, and
    Odometry / Bool / PointCloud2 / Image subscriptions. Lazy-imports
    rclpy + PIL so the bridge module imports cleanly without ROS sourced
    or PIL installed."""

    def __init__(
        self,
        cmd_topic: str,
        odom_topic: str,
        collision_topic: str | None,
        lidar_topics: list[str] | None = None,
        camera_topics: list[str] | None = None,
    ) -> None:
        try:
            import rclpy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
            from nav_msgs.msg import Odometry  # type: ignore[import-not-found]
            from std_msgs.msg import Bool  # type: ignore[import-not-found]
            from sensor_msgs.msg import Image, PointCloud2  # type: ignore[import-not-found]
            from rclpy.qos import QoSPresetProfiles  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit(
                "rclpy is not on PYTHONPATH. Source ROS 2 (e.g. "
                "`source /opt/ros/jazzy/setup.bash`) before running this experiment."
            ) from e

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._Twist = Twist
        self._node = rclpy.create_node("uav_nav_lab_ros2_bridge")
        self._cmd_pub = self._node.create_publisher(Twist, cmd_topic, 10)
        self._latest_odom: Any = None
        self._latest_collision: bool = False
        self._latest_clouds: dict[str, np.ndarray] = {}
        self._latest_images: dict[str, bytes] = {}
        # SENSOR_DATA QoS: best-effort + small depth, matches typical odom /
        # lidar / camera publishers (Gazebo, ardupilot_gz, PX4-SITL via MAVROS).
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self._node.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        if collision_topic is not None:
            self._node.create_subscription(Bool, collision_topic, self._on_collision, 10)
        for topic in (lidar_topics or []):
            self._node.create_subscription(
                PointCloud2, topic,
                lambda msg, t=topic: self._on_pointcloud(t, msg),
                sensor_qos,
            )
        for topic in (camera_topics or []):
            self._node.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._on_image(t, msg),
                sensor_qos,
            )

    def _on_odom(self, msg: Any) -> None:
        self._latest_odom = msg

    def _on_collision(self, msg: Any) -> None:
        self._latest_collision = bool(msg.data)

    def _on_pointcloud(self, topic: str, msg: Any) -> None:
        self._latest_clouds[topic] = _decode_pointcloud2(msg)

    def _on_image(self, topic: str, msg: Any) -> None:
        self._latest_images[topic] = _encode_image_to_png(msg)

    def publish_velocity(self, vx: float, vy: float, vz: float) -> None:
        twist = self._Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = float(vz)
        self._cmd_pub.publish(twist)

    def latest_pose_velocity(self) -> tuple[np.ndarray, np.ndarray] | None:
        msg = self._latest_odom
        if msg is None:
            return None
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        return (
            np.array([p.x, p.y, p.z]),
            np.array([v.x, v.y, v.z]),
        )

    def latest_collision(self) -> bool:
        return self._latest_collision

    def latest_lidar_clouds(self) -> dict[str, np.ndarray]:
        return dict(self._latest_clouds)

    def latest_camera_images(self) -> dict[str, bytes]:
        return dict(self._latest_images)

    def tick(self, timeout_s: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout_s))

    def teleport(self, pos_enu: np.ndarray) -> None:  # noqa: ARG002
        # Sim-specific service call (Gazebo set_entity_state, Ignition
        # /world/.../set_pose). Left as a no-op in the generic adapter;
        # subclass and override if you need it.
        return None


def _decode_pointcloud2(msg: Any) -> np.ndarray:  # pragma: no cover
    """Extract (N, 3) float32 ENU points from a `sensor_msgs/PointCloud2`.

    Reads `x` / `y` / `z` fields by name+offset from `msg.fields`, so it
    works against the standard PointCloud2 layout regardless of whether
    extra fields (intensity, rgb, …) are present in between."""
    import struct

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fmt_pref = ">" if getattr(msg, "is_bigendian", False) else "<"
    fmt = f"{fmt_pref}f"
    point_step = int(msg.point_step)
    data = bytes(msg.data)
    x_off = int(fields["x"].offset)
    y_off = int(fields["y"].offset)
    z_off = int(fields["z"].offset)
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * point_step
        out[i, 0] = struct.unpack_from(fmt, data, base + x_off)[0]
        out[i, 1] = struct.unpack_from(fmt, data, base + y_off)[0]
        out[i, 2] = struct.unpack_from(fmt, data, base + z_off)[0]
    return out


def _encode_image_to_png(msg: Any) -> bytes:  # pragma: no cover
    """Convert a `sensor_msgs/Image` to PNG bytes via PIL.

    Supports the most common encodings (rgb8, bgr8, mono8, rgba8). For
    less common encodings (depth16, yuyv, …) returns empty bytes so the
    runner can still write the rest of the step's data without crashing."""
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        raise SystemExit(
            "PIL/Pillow is required to encode ROS Image messages. "
            "Install with `pip install pillow` (already a `[viz]` extra)."
        ) from e
    import io

    enc = str(msg.encoding)
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)
    if enc == "rgb8":
        img = PILImage.frombytes("RGB", (w, h), raw)
    elif enc == "bgr8":
        img = PILImage.frombytes("RGB", (w, h), raw)
        b, g, r = img.split()
        img = PILImage.merge("RGB", (r, g, b))
    elif enc == "rgba8":
        img = PILImage.frombytes("RGBA", (w, h), raw)
    elif enc == "mono8":
        img = PILImage.frombytes("L", (w, h), raw)
    else:
        return b""  # unsupported encoding — silently skip this frame
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
