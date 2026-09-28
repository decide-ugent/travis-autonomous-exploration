"""
Launch file for the TRAVIS exploration node.

Loads parameters from:
  - exploration_system_parameters.yaml   (exploration + nav2 sections, ROS2 params format)
  - perception_system_parameters.yaml    (camera + lidar sections, read manually)

Both files are resolved from the installed share directory so they work
after `colcon build` without specifying absolute paths.

Usage:
    ros2 launch exploration exploration.launch.py

Optional overrides:
    ros2 launch exploration exploration.launch.py \
        map_dir:=/abs/path/to/map_folder use_rviz:=false

RViz is included from launch/rviz.launch.py (use_rviz:=true by default); it
can also be launched standalone, e.g. for manual baseline runs:
    ros2 launch exploration rviz.launch.py
"""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_map_paths(map_dir: str) -> tuple[str, str]:
    """Return (pgm_path, yaml_path) found inside map_dir, or ('', '') if empty."""
    if not map_dir:
        return '', ''
    d = Path(map_dir)
    pgm_files  = list(d.glob('*.pgm'))
    yaml_files = list(d.glob('*.yaml'))
    if not pgm_files or not yaml_files:
        raise RuntimeError(f"map_dir '{map_dir}' must contain at least one .pgm and one .yaml file.")
    return str(pgm_files[0]), str(yaml_files[0])


def _build_node_params(perception_cfg: str, nav2_params_path: str,
                       map_file_path: str, map_yaml_path: str) -> dict:
    """Assemble the exploration node's runtime params sourced from other YAMLs.

    Reads each config file once here so all cross-package parameter forwarding
    lives in a single place:
      - perception YAML  → camera.* / lidar.*
      - nav2 params YAML → nav2.costmap_inflation_radius (nav2's global costmap
        inflation is the single source of truth; forwarding it keeps the
        exploration planner's navigable_mask consistent with what nav2 can
        navigate, so waypoints never land where nav2 would treat them as costly)
      - launch args      → exploration.map_file_path / map_yaml_path
    """
    with open(perception_cfg) as f:
        p = yaml.safe_load(f)
    with open(nav2_params_path) as f:
        nav2 = yaml.safe_load(f)

    global_costmap = nav2['global_costmap']['global_costmap']['ros__parameters']

    return {
        'camera.max_detection_range': float(p['camera']['max_detection_range']),
        'camera.fov_horizontal':      float(p['camera']['fov_horizontal']),
        'camera.fov_vertical':        float(p['camera']['fov_vertical']),
        'lidar.num_rays':             p['lidar']['num_rays'],
        'lidar.base_frame':           p['lidar']['base_frame'],
        # Override the exploration YAML's fallback inflation with nav2's value.
        'nav2.costmap_inflation_radius': float(global_costmap['inflation_layer']['inflation_radius']),
        'exploration.map_file_path': map_file_path,
        'exploration.map_yaml_path': map_yaml_path,
    }


def _make_exploration_actions(context, *args, **kwargs):
    exploration_dir = Path(get_package_share_directory('exploration'))
    perception_dir  = Path(get_package_share_directory('perception'))
    exploration_cfg = str(exploration_dir / 'config' / 'exploration_system_parameters.yaml')
    perception_cfg  = str(perception_dir / 'config' / 'perception_system_parameters.yaml')

    map_file_path, map_yaml_path = _resolve_map_paths(
        LaunchConfiguration('map_dir').perform(context)
    )
    nav2_params_path = LaunchConfiguration('nav2_params').perform(context)

    node_params = _build_node_params(
        perception_cfg, nav2_params_path, map_file_path, map_yaml_path,
    )

    exploration_node = Node(
        package='exploration',
        executable='ros2_exploration_node.py',
        name='exploration_node',
        parameters=[
            exploration_cfg,
            node_params,
            # Sim clock: this node does TF pose lookups and stamps Nav2 goals.
            # On wall time against a sim-time TF tree, lookups land ~1.78e9 s off
            # and fail. Override with use_sim_time:=false on a real robot.
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        output='screen',
        emulate_tty=True,
    )

    # RViz lives in its own launch file (launch/rviz.launch.py) so it can also
    # be started standalone (e.g. manual baseline runs where the exploration
    # node is replaced but RViz is still used to send Nav2 goals). Disable it
    # here with use_rviz:=false for headless runs.
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(exploration_dir / 'launch' / 'rviz.launch.py')),
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return [exploration_node, rviz_launch]


def generate_launch_description() -> LaunchDescription:
    nav2_share = Path(get_package_share_directory('nav2'))

    map_dir_arg = DeclareLaunchArgument(
        'map_dir',
        default_value='',
        description='Absolute path to a folder containing a .pgm and .yaml map file. Leave empty to use /map topic.',
    )

    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params',
        default_value=str(nav2_share / 'config' / 'nav2_mir_jazzy_params.yaml'),
        description='Path to the nav2 params file; its global costmap inflation_radius is '
                    'forwarded to the exploration node so both stay consistent.',
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Start RViz (via launch/rviz.launch.py). Set false for headless runs.',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use the simulation clock. Must match the rest of the sim '
                    'stack (slam_toolbox, nav2). Set false on a real robot.',
    )

    return LaunchDescription([
        map_dir_arg,
        nav2_params_arg,
        use_rviz_arg,
        use_sim_time_arg,
        OpaqueFunction(function=_make_exploration_actions),
    ])
