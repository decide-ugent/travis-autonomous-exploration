"""
Standalone RViz launch for the TRAVIS exploration visualisation.

Decoupled from exploration.launch.py so RViz can be started:
  - by exploration.launch.py (which includes this file), or
  - on its own, e.g. for a manual baseline run where the exploration node is
    replaced by manual_exploration.py but you still drive via RViz Nav2 goals:

        ros2 launch exploration rviz.launch.py

Optional overrides:
    ros2 launch exploration rviz.launch.py rviz_config:=/abs/path/config.rviz
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    exploration_dir = Path(get_package_share_directory('exploration'))
    default_rviz_cfg = str(exploration_dir / 'rviz' / 'rviz2_explo_config.rviz')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_cfg,
        description='Path to the RViz config file.',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
    )

    return LaunchDescription([
        rviz_config_arg,
        rviz_node,
    ])
