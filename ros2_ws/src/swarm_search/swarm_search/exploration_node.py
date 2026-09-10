"""Milestone 6c node: autonomous single-drone exploration in Gazebo.

This is the Phase 2 exploration loop, ported to ROS with **no algorithm change**.
It subscribes to the shared ``/map`` (built by ``mapping_node`` from real LiDAR)
and the drone odometry, then on a timer:

    frontiers -> cluster -> nearest centroid -> A* -> follow path via cmd_vel

exactly as ``src/exploration/explorer.explore`` does in the pure-Python sim. The
only differences are I/O: the grid arrives as a ``nav_msgs/OccupancyGrid`` instead
of being sensed in-process, and motion is commanded over ``cmd_vel`` instead of
teleporting along the path. Frontier detection, clustering and A* are imported
unchanged from ``src/``.
"""

from __future__ import annotations

import math
import os
import sys
import time


def _add_repo_to_path() -> None:
    bases = [os.environ.get("AISWARM_REPO"), os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for base in bases:
        p = base
        while p and p != os.path.dirname(p):
            if os.path.isdir(os.path.join(p, "src", "planning")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
                if p not in sys.path:
                    sys.path.insert(0, p)
                return
            p = os.path.dirname(p)


_add_repo_to_path()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402

from src.constants import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from src.frontier.clustering import cluster_frontiers  # noqa: E402
from src.frontier.detection import find_frontiers  # noqa: E402
from src.planning.astar import astar  # noqa: E402


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _msg_to_grid(msg: OccupancyGridMsg) -> np.ndarray:
    """nav_msgs/OccupancyGrid (-1/0/100) -> our int8 grid (UNKNOWN/FREE/OCCUPIED)."""
    arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    g = np.full(arr.shape, UNKNOWN, dtype=np.int8)
    g[arr == 0] = FREE
    g[arr == 100] = OCCUPIED
    return g


class ExplorationNode(Node):
    def __init__(self) -> None:
        super().__init__("exploration_node")
        self.declare_parameter("cmd_topic", "/model/drone1/cmd_vel")
        self.declare_parameter("odom_topic", "/model/drone1/odometry")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("resolution", 0.25)
        self.declare_parameter("origin_x", -30.0)
        self.declare_parameter("origin_y", -30.0)
        self.declare_parameter("min_cluster_size", 3)
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("gain", 1.5)
        self.declare_parameter("waypoint_tol", 0.25)   # m: advance to next path cell
        self.declare_parameter("replan_period", 3.0)   # s: re-plan cadence
        self.declare_parameter("control_period", 0.1)  # s: cmd_vel rate
        self.declare_parameter("blacklist_after", 3)

        self.res = float(self.get_parameter("resolution").value)
        self.origin = (float(self.get_parameter("origin_x").value),
                       float(self.get_parameter("origin_y").value))
        self.min_cluster = int(self.get_parameter("min_cluster_size").value)
        self.vmax = float(self.get_parameter("max_speed").value)
        self.gain = float(self.get_parameter("gain").value)
        self.wp_tol = float(self.get_parameter("waypoint_tol").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.blacklist_after = int(self.get_parameter("blacklist_after").value)

        self.grid: np.ndarray | None = None
        self.pose: tuple[float, float, float] | None = None
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0
        self.last_plan = 0.0
        self.blacklist: set[tuple[int, int]] = set()
        self.failures: dict[tuple[int, int], int] = {}
        self.done = False

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGridMsg, self.get_parameter("map_topic").value, self._on_map, latched)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.create_timer(float(self.get_parameter("control_period").value), self._control)
        self.get_logger().info("exploration_node up: frontier-driven autonomous exploration")

    # ------------------------------------------------------------------ #
    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self.grid = _msg_to_grid(msg)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.origin[1]) / self.res)), int(round((x - self.origin[0]) / self.res))

    def _world(self, r: int, c: int) -> tuple[float, float]:
        return self.origin[0] + c * self.res, self.origin[1] + r * self.res

    def _stop(self) -> None:
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------ #
    def _plan(self) -> None:
        """Pick the nearest frontier and A*-plan a path to it (Phase 2 logic)."""
        start = self._cell(self.pose[0], self.pose[1])
        mask = find_frontiers(self.grid)
        clusters = cluster_frontiers(mask, self.min_cluster)
        goals = [c.representative for c in clusters if c.representative not in self.blacklist]
        if not goals:  # thin-corridor fallback, as in explore()
            clusters = cluster_frontiers(mask, min_cluster_size=1)
            goals = [c.representative for c in clusters if c.representative not in self.blacklist]
        if not goals:
            if not self.done:
                self.get_logger().info("exploration complete: no frontiers remain")
            self.done = True
            self.path = []
            return

        goals.sort(key=lambda g: (g[0] - start[0]) ** 2 + (g[1] - start[1]) ** 2)
        for goal in goals:
            plan = astar(self.grid, start, goal, True)
            if plan.path is not None and len(plan.path) >= 2:
                self.path = plan.path
                self.path_idx = 1
                self.done = False
                return
            self.failures[goal] = self.failures.get(goal, 0) + 1
            if self.failures[goal] >= self.blacklist_after:
                self.blacklist.add(goal)
        self.path = []  # nothing plannable this cycle

    def _control(self) -> None:
        if self.grid is None or self.pose is None:
            return
        now = time.monotonic()

        need_replan = (
            not self.path
            or self.path_idx >= len(self.path)
            or (now - self.last_plan) > self.replan_period
        )
        if need_replan:
            self.last_plan = now
            self._plan()

        if self.done or not self.path or self.path_idx >= len(self.path):
            self._stop()
            return

        # If the next cell became a wall as the map sharpened, force a re-plan.
        nr, nc = self.path[self.path_idx]
        if self.grid[nr, nc] == OCCUPIED:
            self.path = []
            self._stop()
            return

        # Drive toward the current path cell with a clamped P-controller.
        tx, ty = self._world(nr, nc)
        ex, ey = tx - self.pose[0], ty - self.pose[1]
        if math.hypot(ex, ey) <= self.wp_tol:
            self.path_idx += 1
            return
        cmd = Twist()
        cmd.linear.x = max(-self.vmax, min(self.vmax, self.gain * ex))
        cmd.linear.y = max(-self.vmax, min(self.vmax, self.gain * ey))
        self.cmd_pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
