"""
Combined launch file: Nav2 + TRAVIS exploration node.

Two modes selected by whether `map_dir` is provided:

  Known-map mode (map_dir given):
    Starts Nav2 with map_server + AMCL, then the exploration node.
    map_dir must contain exactly one .pgm and one .yaml file.
    The .yaml is forwarded to Nav2; both files are forwarded to the exploration node.

  SLAM mode (map_dir omitted or empty):
    Starts Nav2 with slam_toolbox online_async (map built live), then the
    exploration node which subscribes to the /map topic.

Optional arguments:
    map_dir:      absolute path to a folder with a .pgm and .yaml map (leave empty for SLAM)
    use_sim_time: true/false (default: false)
    nav2_params:  path to nav2_params.yaml
                  (default: share/nav2/config/nav2_params.yaml)

Usage:
    # Known map
    ros2 launch exploration exploration_nav2.launch.py \
        map_dir:=/abs/path/to/map_folder

    # SLAM — build the map live
    ros2 launch exploration exploration_nav2.launch.py
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _resolve_map_yaml(map_dir: str) -> str:
    """Return the .yaml path inside map_dir, or '' if map_dir is empty."""
    if not map_dir:
        return ''
    yaml_files = list(Path(map_dir).glob('*.yaml'))
    if not yaml_files:
        raise RuntimeError(f"map_dir '{map_dir}' contains no .yaml file.")
    return str(yaml_files[0])


def _make_actions(context, *args, **kwargs):
    nav2_share        = Path(get_package_share_directory('nav2'))
    exploration_share = Path(get_package_share_directory('exploration'))
    nav2_launch_file        = str(nav2_share / 'launch' / 'nav2.launch.py')
    exploration_launch_file = str(exploration_share / 'launch' / 'exploration.launch.py')

    map_dir      = LaunchConfiguration('map_dir').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    nav2_params  = LaunchConfiguration('nav2_params').perform(context)

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            'map':          _resolve_map_yaml(map_dir),
            'use_sim_time': use_sim_time,
            'params_file':  nav2_params,
        }.items(),
    )

    exploration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(exploration_launch_file),
        launch_arguments={
            'map_dir':      map_dir,
            'use_sim_time': use_sim_time,
            'nav2_params':  nav2_params,
        }.items(),
    )

    return [nav2, exploration]


def generate_launch_description() -> LaunchDescription:
    nav2_share = Path(get_package_share_directory('nav2'))

    map_dir_arg = DeclareLaunchArgument(
        'map_dir',
        default_value='',
        description='Absolute path to a folder containing a .pgm and .yaml map file. Leave empty for SLAM mode.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true.',
    )
    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params',
        default_value=str(nav2_share / 'config' / 'nav2_mir_jazzy_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes',
    )

    return LaunchDescription([
        map_dir_arg,
        use_sim_time_arg,
        nav2_params_arg,
        OpaqueFunction(function=_make_actions),
    ])
