#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to Tiago URDF
    urdf_file = '/opt/pal/alum/share/tiago_description/robots/tiago.urdf.xacro'

    # Process xacro to get URDF
    import xacro
    robot_description = xacro.process_file(urdf_file).toxml()

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 50.0
        }]
    )

    return LaunchDescription([
        robot_state_publisher
    ])
