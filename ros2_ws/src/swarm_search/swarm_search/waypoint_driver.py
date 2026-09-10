"""Milestone 6a node: drive one kinematic drone through a list of waypoints.

Subscribes to the drone's ground-truth odometry, and publishes a ``cmd_vel``
Twist that steers it toward the current waypoint with a simple clamped
proportional controller. When it arrives (within ``tolerance``), it advances to
the next waypoint; after the last, it holds position.

This is the *first* Phase 6 milestone — proving one drone spawns and flies to a
commanded point in Gazebo — before any exploration logic is wired in. The drone
keeps a fixed heading (yaw 0), so commanding world-frame vx/vy is unambiguous.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _parse_waypoints(spec: str) -> list[tuple[float, float]]:
    """Parse ``"x1,y1; x2,y2; ..."`` into a list of (x, y) waypoints."""
    pts = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    return pts


class WaypointDriver(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_driver")
        self.declare_parameter("waypoints", "3,0; 3,3; 0,3; 0,0")
        self.declare_parameter("cmd_topic", "/model/drone1/cmd_vel")
        self.declare_parameter("odom_topic", "/model/drone1/odometry")
        self.declare_parameter("max_speed", 1.0)      # m/s cap per axis
        self.declare_parameter("gain", 1.2)           # proportional gain
        self.declare_parameter("tolerance", 0.2)      # m: "arrived" radius

        self.waypoints = _parse_waypoints(self.get_parameter("waypoints").value)
        self.vmax = float(self.get_parameter("max_speed").value)
        self.gain = float(self.get_parameter("gain").value)
        self.tol = float(self.get_parameter("tolerance").value)
        self.idx = 0

        self.pub = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.sub = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10
        )
        self.get_logger().info(f"driving {len(self.waypoints)} waypoints: {self.waypoints}")

    def _clamp(self, v: float) -> float:
        return max(-self.vmax, min(self.vmax, v))

    def _on_odom(self, msg: Odometry) -> None:
        cmd = Twist()
        if self.idx >= len(self.waypoints):
            self.pub.publish(cmd)  # zero velocity: hold position
            return

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        tx, ty = self.waypoints[self.idx]
        ex, ey = tx - px, ty - py
        dist = (ex * ex + ey * ey) ** 0.5

        if dist <= self.tol:
            self.get_logger().info(
                f"reached waypoint {self.idx + 1}/{len(self.waypoints)} "
                f"({tx:.1f}, {ty:.1f}) at ({px:.2f}, {py:.2f})"
            )
            self.idx += 1
            self.pub.publish(cmd)  # brief stop before next leg
            return

        cmd.linear.x = self._clamp(self.gain * ex)
        cmd.linear.y = self._clamp(self.gain * ey)
        self.pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = WaypointDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
