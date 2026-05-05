#!/usr/bin/env python3
"""Minimal ROS 2 node that acts as a dummy_3d physics backend.

Publishes /odom (nav_msgs/Odometry) and subscribes to /cmd_vel
(geometry_msgs/Twist) so the framework's ros2_bridge can drive it
exactly like AirSim or Gazebo.  The integrator mirrors the dummy_3d
simulator: perfect velocity tracking subject to max_accel.

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 scripts/ros2_dummy_sim.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Header
import numpy as np


class DummySimNode(Node):
    def __init__(self) -> None:
        super().__init__("dummy_sim")

        self.declare_parameter("dt", 0.05)
        self.declare_parameter("max_accel", 50.0)
        self.declare_parameter("drone_radius", 0.4)
        self.declare_parameter("start_x", 2.0)
        self.declare_parameter("start_y", 2.0)
        self.declare_parameter("goal_x", 40.0)
        self.declare_parameter("goal_y", 40.0)
        self.declare_parameter("goal_radius", 1.5)

        self.dt = self.get_parameter("dt").value
        self.max_accel = self.get_parameter("max_accel").value

        self.pos = np.array([
            self.get_parameter("start_x").value,
            self.get_parameter("start_y").value,
        ], dtype=float)
        self.vel = np.zeros(2)
        self.goal = np.array([
            self.get_parameter("goal_x").value,
            self.get_parameter("goal_y").value,
        ], dtype=float)
        self.goal_radius = self.get_parameter("goal_radius").value
        self.t = 0.0
        self._cmd = np.zeros(2)
        self._collision = False

        self._odom_pub = self.create_publisher(
            Odometry, "/odom", qos_profile=QoSPresetProfiles.SENSOR_DATA.value
        )
        self._collision_pub = self.create_publisher(Bool, "/collision", 10)
        self._cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_cb, 10
        )
        self._timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"DummySim ready — pos=({self.pos[0]:.0f},{self.pos[1]:.0f}) "
            f"goal=({self.goal[0]:.0f},{self.goal[1]:.0f})"
        )

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd = np.array([msg.linear.x, msg.linear.y], dtype=float)

    def _tick(self) -> None:
        # Integrate with acceleration limit (mirrors dummy_3d)
        cmd = self._cmd
        dv = cmd - self.vel
        dvmag = float(np.linalg.norm(dv))
        if dvmag > self.max_accel * self.dt:
            dv = dv / dvmag * self.max_accel * self.dt
        self.vel = self.vel + dv
        self.pos = self.pos + self.vel * self.dt
        self.t += self.dt

        # Collision detection (none in this simple sim)
        if self._collision:
            return

        # Publish odometry
        now = self.get_clock().now()
        odom = Odometry()
        odom.header = Header(stamp=now.to_msg(), frame_id="odom")
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(self.pos[0])
        odom.pose.pose.position.y = float(self.pos[1])
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = float(self.vel[0])
        odom.twist.twist.linear.y = float(self.vel[1])
        self._odom_pub.publish(odom)

        # Publish collision
        col = Bool(data=False)
        self._collision_pub.publish(col)

        # Goal check
        if np.linalg.norm(self.pos - self.goal) <= self.goal_radius:
            self.get_logger().info(f"Goal reached at t={self.t:.1f}s")
            self._timer.cancel()


def main() -> None:
    rclpy.init()
    node = DummySimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
