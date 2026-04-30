"""ROS 2 bridge — sketch implementation.

Same intent as `airsim_bridge.py`: prove the SimInterface contract maps
onto a real backend without forcing the dep into CI.

Topology (ROS 2 side):
  - publish: /cmd_vel   geometry_msgs/Twist  (drone velocity setpoint)
  - subscribe: /odom    nav_msgs/Odometry    (true pose+velocity)
  - subscribe: /collision std_msgs/Bool      (or use /tf + geometry checks)

Real-world pitfalls:
  - rclpy must spin to receive messages — use a SingleThreadedExecutor here
    and tick it inside `step` so the latest odom is consumed every dt.
  - QoS: odom should be best-effort/sensor-data; collision flag should be
    reliable.
  - Sim-time vs ros-time: PX4 SITL ticks via /clock; bridge must respect
    `use_sim_time` if the experiment runner is in fast-forward.
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
        cmd_topic: str,
        odom_topic: str,
        collision_topic: str | None,
    ) -> None:
        self.dt = dt
        self.scenario = scenario
        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.collision_topic = collision_topic
        self._node: Any = None
        self._state: SimState | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "Ros2Bridge":
        return cls(
            dt=float(cfg.get("dt", 0.05)),
            scenario=scenario,
            cmd_topic=str(cfg.get("cmd_topic", "/cmd_vel")),
            odom_topic=str(cfg.get("odom_topic", "/odom")),
            collision_topic=cfg.get("collision_topic"),
        )

    def _ensure_node(self) -> Any:  # pragma: no cover
        if self._node is not None:
            return self._node
        try:
            import rclpy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]  # noqa: F401
            from nav_msgs.msg import Odometry  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "rclpy is not on PYTHONPATH. Source ROS 2 (e.g. "
                "`source /opt/ros/jazzy/setup.bash`) before running this experiment."
            ) from e

        if not rclpy.ok():
            rclpy.init()
        # Real implementation creates a node, publishers, subscriptions, and a
        # buffer for the latest odom — left as TODO so this stub stays small.
        raise NotImplementedError(
            "Ros2Bridge is currently a structural stub. Wire publishers and "
            "odom subscriber here, then drop this exception."
        )

    def reset(self, *, seed: int | None = None) -> SimState:  # pragma: no cover
        self._ensure_node()
        if seed is not None:
            self.scenario.reseed(seed)
        start = np.asarray(self.scenario.start, dtype=float)
        self._state = SimState(t=0.0, position=start.copy(), velocity=np.zeros(start.shape[0]))
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:  # pragma: no cover
        self._ensure_node()
        # publish Twist; spin once so /odom is consumed; read latest pose
        raise NotImplementedError("Ros2Bridge.step: see _ensure_node TODO")

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
