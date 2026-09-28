import yaml
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description() -> LaunchDescription:
    share_dir = Path(get_package_share_directory('perception'))
    perception_cfg = share_dir / 'config' / 'perception_system_parameters.yaml'

    with open(perception_cfg) as f:
        lidar_cfg = yaml.safe_load(f)['lidar']

    return LaunchDescription([
        DeclareLaunchArgument('sync_slop', default_value='0.1'),
        # Sim time so scan timestamps match slam_toolbox/nav2 (which run on the
        # sim clock). Wall-clock stamps land ~1.78e9 s in SLAM's future, so every
        # scan is dropped on TF lookup and the map never grows. Real robot: false.
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='perception',
            executable='panoramic_laser_scan.py',
            name='panoramic_laser_scan',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'output_frame': lidar_cfg['base_frame'],
                'output_topic': lidar_cfg['scan_topic'],
                'scan_topic_1': lidar_cfg['scan_topic_1'],
                'scan_topic_2': lidar_cfg['scan_topic_2'],
                'num_rays': lidar_cfg['num_rays'],
                'sync_slop': LaunchConfiguration('sync_slop'),
            }],
        ),
    ])
