"""Milestone 6b node: build a shared occupancy grid from Gazebo LiDAR.

Subscribes to the drone's ``LaserScan`` (bridged from Gazebo's gpu_lidar) and its
ground-truth odometry, folds each scan into an ``OccupancyGrid`` using the project
mapping code (``src/mapping``, imported unchanged), and republishes the belief map
as a ``nav_msgs/OccupancyGrid`` on ``/map`` for RViz.

This is the I/O boundary only: the actual mapping algorithm is the same
``integrate_scan`` used and unit-tested in the pure-Python suite.
"""

from __future__ import annotations

import math
import os
import sys


def _add_repo_to_path() -> None:
    """Put the project root (which holds ``src/``) on ``sys.path``.

    ``src/`` is deliberately kept outside the ROS package (CLAUDE.md §6.3), so we
    locate the repo root by walking up from a few candidate bases until we find
    the directory that contains both ``src/mapping`` and ``CLAUDE.md``."""
    bases = [os.environ.get("AISWARM_REPO"), os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for base in bases:
        p = base
        while p and p != os.path.dirname(p):
            if os.path.isdir(os.path.join(p, "src", "mapping")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
                if p not in sys.path:
                    sys.path.insert(0, p)
                return
            p = os.path.dirname(p)


_add_repo_to_path()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

from src.constants import FREE, OCCUPIED  # noqa: E402
from src.mapping.occupancy_grid import OccupancyGrid  # noqa: E402
from src.mapping.scan import integrate_scan  # noqa: E402


def _yaw_from_quat(q) -> float:
    """Yaw (rotation about z) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class MappingNode(Node):
    def __init__(self) -> None:
        super().__init__("mapping_node")
        self.declare_parameter("scan_topic", "/model/drone1/scan")
        self.declare_parameter("odom_topic", "/model/drone1/odometry")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("resolution", 0.25)
        self.declare_parameter("size", 240)          # cells per side
        self.declare_parameter("origin_x", -30.0)    # world coord of cell (0,0)
        self.declare_parameter("origin_y", -30.0)
        self.declare_parameter("agent_id", 0)
        self.declare_parameter("publish_period", 0.5)

        self.res = float(self.get_parameter("resolution").value)
        self.size = int(self.get_parameter("size").value)
        self.origin = (float(self.get_parameter("origin_x").value),
                       float(self.get_parameter("origin_y").value))
        self.agent_id = int(self.get_parameter("agent_id").value)

        self.grid = OccupancyGrid(self.size, self.size, resolution=self.res)
        self.pose: tuple[float, float, float] | None = None

        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, 10)
        # Latched map so RViz gets the current grid the moment it subscribes.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGridMsg, self.get_parameter("map_topic").value, latched)
        self.create_timer(float(self.get_parameter("publish_period").value), self._publish_map)
        self.get_logger().info("mapping_node up: folding LiDAR scans into the shared grid")

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))

    def _on_scan(self, msg: LaserScan) -> None:
        if self.pose is None:
            return  # wait for the first pose
        integrate_scan(
            self.grid, self.pose, list(msg.ranges),
            angle_min=msg.angle_min, angle_increment=msg.angle_increment,
            range_max=msg.range_max, origin=self.origin, agent_id=self.agent_id,
            occupied_sticky=True,  # don't let grazing beams punch holes in walls
        )

    def _publish_map(self) -> None:
        msg = OccupancyGridMsg()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.res
        msg.info.width = self.size
        msg.info.height = self.size
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0
        # nav_msgs OccupancyGrid: -1 unknown, 0..100 occupied probability, row-major.
        # FREE -> 0, OCCUPIED -> 100; leave UNKNOWN at -1.
        out = np.full(self.grid.grid.shape, -1, dtype=np.int8)
        out[self.grid.grid == FREE] = 0
        out[self.grid.grid == OCCUPIED] = 100
        msg.data = out.ravel().tolist()
        self.map_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
