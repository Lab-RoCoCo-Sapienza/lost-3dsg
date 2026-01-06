#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to rosbag file

    # Get package share directory
    pkg_share = get_package_share_directory('lost3dsg')

    # Path to RViz config
    rviz_config_path = os.path.join(pkg_share, '..', '..', '..', '..', 'rviz', 'config.rviz')

    # 1. Include tiago_state_only launch file
    tiago_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'tiago_state_only.launch.py')
        )
    )

    # 3. RViz with config
    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config_path],
        output='screen',
        shell=False
    )

    return LaunchDescription([
        tiago_state_launch,
        rviz,
    ])
