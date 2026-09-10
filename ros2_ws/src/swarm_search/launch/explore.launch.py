"""Milestone 6c launch: one drone autonomously explores the bounded room.

Gazebo (room world) + ROS<->GZ bridge (cmd_vel, odometry, scan) + mapping_node
(builds /map from LiDAR) + exploration_node (frontier + A* drives the drone).
No waypoint_driver — the exploration node commands cmd_vel itself.

Usage:
  ros2 launch swarm_search explore.launch.py            # Gazebo GUI
  ros2 launch swarm_search explore.launch.py gui:=false # headless

View the map building live:  rviz2 -> Map display on /map (fixed frame "map").
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
    p = os.getcwd()
    while p and p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "src", "planning")) and os.path.isfile(os.path.join(p, "CLAUDE.md")):
            return p
        p = os.path.dirname(p)
    return os.getcwd()


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("swarm_search")
    world = os.path.join(share, "worlds", "room.sdf")
    gui = LaunchConfiguration("gui")

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
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
        Node(package="swarm_search", executable="mapping_node", output="screen"),
        Node(package="swarm_search", executable="exploration_node", output="screen"),
    ])
