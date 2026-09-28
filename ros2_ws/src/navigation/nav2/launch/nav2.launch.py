"""
Nav2 launch file for TRAVIS.

Two modes selected by whether `map` is provided:

  Known-map mode (map given):
    Starts the full Nav2 stack. Localisation is AMCL by default, or
    slam_toolbox in localization mode with localization:=slam_toolbox.

  SLAM mode (map omitted or empty):
    Starts Nav2 without map_server/AMCL and launches slam_toolbox in
    online_async mode so the map is built live.

Optional arguments:
    map           : absolute path to the map YAML file (leave empty for SLAM)
    use_sim_time  : true/false (default: true)
    robot         : which per-robot params to load, mir | turtle3 (default: mir)
    params_file   : absolute path to a params file; overrides `robot` when set
    localization  : amcl | slam_toolbox (default: amcl), known-map mode only
    serialized_map: pose-graph stem with NO extension, required by slam_toolbox
    base_frame    : slam_toolbox base frame (default: base_footprint)
    scan_topic    : slam_toolbox scan topic (default: /panoramic/scan)

Usage:
    # SLAM with the default MiR params
    ros2 launch nav2 nav2.launch.py

    # TurtleBot3 params, known map, AMCL localisation
    ros2 launch nav2 nav2.launch.py robot:=turtle3 map:=/abs/path/to/map.yaml

    # Known map, localising with slam_toolbox against a saved pose-graph.
    # The graph comes from a previous SLAM run, see scripts/save_map.sh.
    ros2 launch nav2 nav2.launch.py robot:=turtle3 \
        map:=/abs/path/to/map.yaml \
        localization:=slam_toolbox \
        serialized_map:=/abs/path/to/map

    # Same, on the ROSbot XL (different base frame and scan topic)
    ros2 launch nav2 nav2.launch.py map:=/abs/map.yaml \
        localization:=slam_toolbox serialized_map:=/abs/map \
        base_frame:=base_link scan_topic:=/scan

    # Explicit params file (overrides robot)
    ros2 launch nav2 nav2.launch.py params_file:=/abs/path/to/params.yaml

Why slam_toolbox localization is offered at all: AMCL matches the live scan
against a rasterized occupancy grid with a particle filter, while slam_toolbox
scan-matches against the stored scans of the pose-graph. It publishes map -> odom
exactly as AMCL does, so nav2 and the exploration node need no change.
"""
import subprocess
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

perception_share_dir   = Path(get_package_share_directory('perception'))
Lidar_launch_file = str(perception_share_dir / 'launch' / 'panoramic_laser_scan.launch.py')

def _make_nav_actions(context, *args, **kwargs):
    nav2_bringup_dir = Path(get_package_share_directory('nav2_bringup'))
    nav2_share_dir   = Path(get_package_share_directory('nav2'))
    
    slam_toolbox_dir = Path(get_package_share_directory('slam_toolbox'))

    

    map_path       = LaunchConfiguration('map').perform(context)
    use_sim_time   = LaunchConfiguration('use_sim_time').perform(context)
    params_file    = LaunchConfiguration('params_file').perform(context)
    robot          = LaunchConfiguration('robot').perform(context)
    localization   = LaunchConfiguration('localization').perform(context)
    serialized_map = LaunchConfiguration('serialized_map').perform(context)
    base_frame     = LaunchConfiguration('base_frame').perform(context)
    scan_topic     = LaunchConfiguration('scan_topic').perform(context)

    # Resolve the params file. An explicit params_file:=... always wins; else
    # pick the per-robot file by the `robot` argument (mir | turtle3).
    if not params_file:
        robot_params = {
            'mir':     'nav2_mir_jazzy_params.yaml',
            'turtle3': 'nav2_turtle3_jazzy_params.yaml',
        }
        if robot not in robot_params:
            raise RuntimeError(
                f"robot:={robot!r} not recognised; use one of "
                f"{sorted(robot_params)} or pass params_file:=<abs path>.")
        params_file = str(nav2_share_dir / 'config' / robot_params[robot])
    # Validate here rather than letting an unrecognised value fall through to the AMCL branch, where a typo like localization:=slamtoolbox would silently start AMCL and look like it worked.
    if localization not in ('amcl', 'slam_toolbox'):
        raise RuntimeError(
            f"localization:={localization!r} not recognised; use amcl | slam_toolbox.")
    print(f"[nav2.launch] robot={robot} params_file={params_file} "
          f"localization={localization if map_path else 'n/a (SLAM mode)'}")

    # Kill orphaned nav2 processes from any previous launch. OpaqueFunction runs
    # synchronously before any nodes are spawned, so pkill cannot hit new processes.
    for exe in [
        'controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
        'waypoint_follower', 'velocity_smoother', 'collision_monitor', 'lifecycle_manager',
    ]:
        subprocess.run(['pkill', '-f', exe], check=False)
    time.sleep(1.0)

    actions = []

    if map_path and localization == 'slam_toolbox':
        # Known-map mode, localizing with slam_toolbox instead of AMCL. slam_toolbox in localization mode publishes BOTH /map and map -> odom, so map_server and AMCL are not started at all and the plain navigation bringup is used, exactly as in SLAM mode below.
        if not serialized_map:
            raise RuntimeError(
                "localization:=slam_toolbox needs serialized_map:=<path to the "
                "pose-graph stem, no extension>. Produce one with "
                "`ros2 run nav2 save_map.sh <out_dir>` during a SLAM run.")
        if not Path(serialized_map + '.posegraph').is_file():
            raise RuntimeError(
                f"No pose-graph at '{serialized_map}.posegraph'. serialized_map "
                "must be the stem WITHOUT an extension, e.g. /path/to/map "
                "for /path/to/map.posegraph.")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(nav2_share_dir / 'launch' / 'navigation_jazzy_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file':  params_file,
            }.items(),
        ))
        # The slam_toolbox node is declared here rather than via online_async_launch.py because map_file_name, base_frame and scan_topic must be injected per run, and that include only forwards a params FILE.
        loc_params = str(nav2_share_dir / 'config' / 'slam_toolbox_localization_params.yaml')
        actions.append(Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                loc_params,
                {
                    'use_sim_time':  use_sim_time.lower() == 'true',
                    'map_file_name': serialized_map,
                    'base_frame':    base_frame,
                    'scan_topic':    scan_topic,
                },
            ],
        ))
    elif map_path:
        # Known-map mode: map_server + AMCL + full nav stack, no docking server
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(nav2_share_dir / 'launch' / 'localization_jazzy_launch.py')
            ),
            launch_arguments={
                'map':          map_path,
                'use_sim_time': use_sim_time,
                'params_file':  params_file,
            }.items(),
        ))
    else:
        # SLAM mode: nav2_bringup without map_server/AMCL + slam_toolbox online_async
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(nav2_share_dir / 'launch' / 'navigation_jazzy_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file':  params_file,
            }.items(),
        ))
        slam_params = str(nav2_share_dir / 'config' / 'slam_toolbox_params.yaml')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(slam_toolbox_dir / 'launch' / 'online_async_launch.py')
            ),
            launch_arguments={
                'use_sim_time':     use_sim_time,
                'slam_params_file': slam_params,
            }.items(),
        ))

    return actions


def generate_launch_description() -> LaunchDescription:
    perception_double_lidars = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(Lidar_launch_file),
        # Forward the sim clock: without this the panoramic scan node defaults to
        # wall time while slam_toolbox runs on sim time, so scans are stamped
        # ~1.78e9 s in SLAM's future and dropped — the map never grows.
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )
    nav2_share_dir = Path(get_package_share_directory('nav2'))

    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Absolute path to the map YAML file. Leave empty to use SLAM mode.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true.',
    )
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='mir',
        description='Which robot params to load when params_file is empty: '
                    'mir | turtle3.',
    )
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='',
        description='Full path to a ROS2 params file. Overrides `robot` when '
                    'set; leave empty to select by robot.',
    )
    localization_arg = DeclareLaunchArgument(
        'localization',
        default_value='amcl',
        description='Known-map localization backend: amcl | slam_toolbox. '
                    'Ignored in SLAM mode (no map given).',
    )
    serialized_map_arg = DeclareLaunchArgument(
        'serialized_map',
        default_value='',
        description='Absolute path to a slam_toolbox pose-graph WITHOUT its '
                    'extension. Required when localization:=slam_toolbox.',
    )
    # base_frame and scan_topic only reach slam_toolbox; AMCL takes both from its own params file. They are arguments because they differ per robot: the sim uses base_footprint and /panoramic/scan, the ROSbot XL uses base_link and /scan.
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_footprint',
        description='Robot base frame for slam_toolbox. Use base_link on the '
                    'ROSbot XL.',
    )
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/panoramic/scan',
        description='Laser scan topic for slam_toolbox. Use /scan on the '
                    'ROSbot XL.',
    )


    return LaunchDescription([
        # perception_double_lidars,
        map_arg,
        use_sim_time_arg,
        robot_arg,
        params_file_arg,
        localization_arg,
        serialized_map_arg,
        base_frame_arg,
        scan_topic_arg,
        OpaqueFunction(function=_make_nav_actions),
    ])
