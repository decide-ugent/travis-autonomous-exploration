"""
Launch file for the manual-baseline coverage node (tests/system layer).

Drop-in replacement for exploration.launch.py with the STRATEGY OFF: it starts
manual_exploration.py with exactly the same assembled parameters (camera.* /
lidar.* from the perception YAML, nav2 inflation radius from the nav2 params,
map paths from map_dir) so a human-driven reference run measures coverage
identically to an autonomous run. Reuses exploration.launch.py's helpers so
the parameter forwarding cannot drift between the two.

RViz is included (use_rviz:=true by default) because the baseline is driven by
clicking Nav2 "2D Goal Pose" in RViz; Nav2 and the simulator are launched
separately, same as for an autonomous run.

Usage:
    # known-map mode (mode is deduced from map_dir being set):
    ros2 launch exploration manual_exploration.launch.py \
        map_dir:=/abs/path/to/map_folder

    # SLAM mode (omit map_dir; the node reads /map):
    ros2 launch exploration manual_exploration.launch.py

A non-empty map_dir that does not exist fails loudly (typo protection) rather
than silently falling back to SLAM.
"""
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_exploration_launch_helpers():
    """Import _resolve_map_paths/_build_node_params from exploration.launch.py
    (launch files are not importable as a package, so load by path)."""
    path = Path(__file__).resolve().parent / 'exploration.launch.py'
    spec = importlib.util.spec_from_file_location('exploration_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._resolve_map_paths, mod._build_node_params


def _make_actions(context, *args, **kwargs):
    resolve_map_paths, build_node_params = _load_exploration_launch_helpers()

    exploration_dir = Path(get_package_share_directory('exploration'))
    perception_dir  = Path(get_package_share_directory('perception'))
    perception_cfg  = str(perception_dir / 'config' / 'perception_system_parameters.yaml')

    # Mode is DEDUCED from map_dir: a map folder -> known_map, empty -> slam
    # (the node then reads /map). No separate mode flag to contradict it.
    # Failsafe: a non-empty map_dir that does not exist is almost certainly a
    # typo, not an intent to SLAM -> fail loudly instead of silently slamming.
    map_dir = LaunchConfiguration('map_dir').perform(context)
    if map_dir and not Path(map_dir).is_dir():
        raise RuntimeError(
            f"map_dir '{map_dir}' does not exist. If you meant to run in SLAM "
            "mode (live /map), omit map_dir entirely.")
    map_file_path, map_yaml_path = resolve_map_paths(map_dir)
    mode = 'known_map' if map_file_path else 'slam'
    print(f"[manual_exploration.launch] mode={mode} "
          + (f"(static map: {map_file_path})" if map_file_path
             else "(no map_dir given -> SLAM, reading /map)"))

    nav2_params_path = LaunchConfiguration('nav2_params').perform(context)

    # Same assembled params as the autonomous node (single source of truth).
    node_params = build_node_params(
        perception_cfg, nav2_params_path, map_file_path, map_yaml_path,
    )
    node_params['mode'] = mode
    node_params['use_sim_time'] = (
        LaunchConfiguration('use_sim_time').perform(context).lower() == 'true')

    manual_node = Node(
        package='exploration',
        executable='manual_exploration.py',
        name='manual_exploration',
        parameters=[node_params],
        output='screen',
        emulate_tty=True,
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(exploration_dir / 'launch' / 'rviz.launch.py')),
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return [manual_node, rviz_launch]


def generate_launch_description() -> LaunchDescription:
    nav2_share = Path(get_package_share_directory('nav2'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_dir', default_value='',
            description='Folder with the .pgm and .yaml map (known-map mode). '
                        'Leave empty to read /map (SLAM mode).'),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=str(nav2_share / 'config' / 'nav2_mir_jazzy_params.yaml'),
            description='nav2 params file; its global costmap inflation_radius '
                        'is forwarded, same as exploration.launch.py.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the simulator /clock (required for system tests).'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz (needed to click Nav2 goals).'),
        OpaqueFunction(function=_make_actions),
    ])
