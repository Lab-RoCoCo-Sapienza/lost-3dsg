#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to rosbag file
    rosbag_path = '/home/michele/exchange/ultima/ultima_0.db3'

    # Path to RViz config
    rviz_config_path = '/home/michele/exchange/config_rviz.rviz'

    # Get package share directory
    pkg_share = get_package_share_directory('tiago_project')

    # 1. Include tiago_state_only launch file
    tiago_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'tiago_state_only.launch.py')
        )
    )

    # 2. Rosbag playback with clock and rate 0.1
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--clock', '--rate', '0.1'],
        output='screen',
        shell=False
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
