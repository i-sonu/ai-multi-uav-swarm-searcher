"""Milestone 6a launch: Gazebo world + ROS<->GZ bridge + waypoint driver.

Brings up:
  - `gz sim` running the one_drone world (GUI by default; headless with gui:=false),
  - a ros_gz parameter_bridge translating cmd_vel (ROS->GZ) and odometry (GZ->ROS),
  - the waypoint_driver node that flies the drone through its waypoints.

Usage:
  ros2 launch swarm_search one_drone.launch.py
  ros2 launch swarm_search one_drone.launch.py gui:=false   # headless (CI / video capture)
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("swarm_search")
    world = os.path.join(share, "worlds", "one_drone.sdf")
    gui = LaunchConfiguration("gui")

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="run the Gazebo GUI (false = headless server only)"),

        # Gazebo — GUI (server+client) or headless (server only, -s).
        ExecuteProcess(
            cmd=["gz", "sim", "-r", world], output="screen",
            condition=IfCondition(gui),
        ),
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "-s", world], output="screen",
            condition=UnlessCondition(gui),
        ),

        # ROS <-> Gazebo bridge.
        Node(
            package="ros_gz_bridge", executable="parameter_bridge", output="screen",
            arguments=[
                "/model/drone1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                "/model/drone1/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            ],
        ),

        # Waypoint driver (the actual milestone-6a behaviour).
        Node(
            package="swarm_search", executable="waypoint_driver", output="screen",
        ),
    ])
