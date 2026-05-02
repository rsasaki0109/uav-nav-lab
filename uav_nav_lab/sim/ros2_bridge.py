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
                          False if the topic is unconfigured).
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

The adapter abstraction (`_Ros2Adapter` duck-typed surface) keeps the
bridge testable without rclpy installed. `_RclpyAdapter` is the
production implementation; tests inject a fake adapter that records
publishes and returns canned odom.
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
        adapter: Any = None,
    ) -> None:
        self.dt = float(dt)
        self.scenario = scenario
        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.collision_topic = collision_topic
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
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
        )

    def _ensure_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        self._adapter = _RclpyAdapter(
            cmd_topic=self.cmd_topic,
            odom_topic=self.odom_topic,
            collision_topic=self.collision_topic,
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
    Odometry / Bool subscriptions. Lazy-imports rclpy so the bridge
    module can be imported without ROS sourced."""

    def __init__(
        self, cmd_topic: str, odom_topic: str, collision_topic: str | None
    ) -> None:
        try:
            import rclpy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
            from nav_msgs.msg import Odometry  # type: ignore[import-not-found]
            from std_msgs.msg import Bool  # type: ignore[import-not-found]
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
        # SENSOR_DATA QoS: best-effort + small depth, matches typical odom
        # publishers (Gazebo, ardupilot_gz, PX4-SITL via MAVROS).
        self._node.create_subscription(
            Odometry, odom_topic, self._on_odom, QoSPresetProfiles.SENSOR_DATA.value
        )
        if collision_topic is not None:
            self._node.create_subscription(Bool, collision_topic, self._on_collision, 10)

    def _on_odom(self, msg: Any) -> None:
        self._latest_odom = msg

    def _on_collision(self, msg: Any) -> None:
        self._latest_collision = bool(msg.data)

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

    def tick(self, timeout_s: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout_s))

    def teleport(self, pos_enu: np.ndarray) -> None:  # noqa: ARG002
        # Sim-specific service call (Gazebo set_entity_state, Ignition
        # /world/.../set_pose). Left as a no-op in the generic adapter;
        # subclass and override if you need it.
        return None
