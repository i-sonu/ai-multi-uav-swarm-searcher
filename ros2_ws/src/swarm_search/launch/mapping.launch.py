"""Milestone 6b launch: one drone patrols while its LiDAR builds the /map.

Brings up Gazebo, the ROS<->GZ bridge (cmd_vel, odometry, and now the LaserScan),
the waypoint_driver (so the drone moves and sees new area), and the mapping_node
that folds scans into a shared occupancy grid published on /map.

Usage:
  ros2 launch swarm_search mapping.launch.py            # Gazebo GUI
  ros2 launch swarm_search mapping.launch.py gui:=false # headless

View the map:  rviz2  -> add a Map display on topic /map (fixed frame "map").
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_repo_root() -> str:
    """Walk up from cwd to the project root (holds src/ and CLAUDE.md)."""
    p = os.getcwd()
    while p and p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "src", "mapping")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
            return p
        p = os.path.dirname(p)
    return os.getcwd()


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("swarm_search")
    world = os.path.join(share, "worlds", "one_drone.sdf")
    gui = LaunchConfiguration("gui")

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        # Make the project's src/ importable by the mapping node.
        SetEnvironmentVariable("AISWARM_REPO", _find_repo_root()),

        ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen", condition=IfCondition(gui)),
        ExecuteProcess(cmd=["gz", "sim", "-r", "-s", world], output="screen", condition=UnlessCondition(gui)),

        Node(
            package="ros_gz_bridge", executable="parameter_bridge", output="screen",
            arguments=[
                "/model/drone1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                "/model/drone1/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                "/model/drone1/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            ],
        ),
        Node(package="swarm_search", executable="waypoint_driver", output="screen"),
        Node(package="swarm_search", executable="mapping_node", output="screen"),
    ])
