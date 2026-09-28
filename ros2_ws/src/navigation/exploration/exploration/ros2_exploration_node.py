#!/usr/bin/env python3
"""
ROS2 exploration node.

Wraps ExplorationSession in a ROS2 node that:
  - Receives the occupancy grid from /map (live) or a static PGM file.
  - Plans waypoints via plan_waypoints_raw() from explore_costmap_map.py.
  - Navigates the robot to each waypoint via the Nav2 NavigateToPose action.
  - Rotates at each waypoint using get_headings() / RotationState.
  - Publishes coverage progress and exploration status.

State machine (timer-driven at 1 Hz):
    WAITING_FOR_MAP → PLANNING → NAVIGATING → TRAVELING → ROTATING → PLANNING
                                             ↓ (complete)            (one Spin action per heading;
                                           COMPLETE                   covered_mask updated on result)

"""
from __future__ import annotations

import json
import math
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
# Aliased deliberately: builtin_interfaces.msg.Duration below is the MESSAGE type used to fill Spin goals, while this is the rclpy duration object that tf2_ros.Buffer(cache_time=...) requires. Importing both unaliased would shadow one with the other and fail at runtime.
from rclpy.duration import Duration as RclpyDuration

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from exploration.explore_costmap_map import (
    build_map_data,
    compute_headings_for_waypoint,
    compute_visibility,
    load_map,
    pixel_to_world,
    reproject_covered_mask,
    update_covered_mask,
    world_to_pixel,
)
from exploration.execution_strategy import ExplorationSession
from exploration.rotation_strategy import get_headings, RotationState
from exploration import rviz_visualisation as rviz


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

# Consecutive fully-failed plans (every waypoint inaccessible) before giving up.
# 2 -> first recovery Spin; this cap -> stop. Bounds the stuck-pose recovery loop so a genuinely wedged robot doesn't Spin/re-plan forever.
_MAX_FAILED_PLAN_STREAK = 5

# TF buffer history, in seconds. Longer than tf2's 10 s default so a momentary gap in the dynamic /tf stream (SLAM loop closure, a busy tick) cannot empty the buffer on its own. This is deliberately NOT the fix for a missing /tf_static link, which no cache length can help: latched static samples are delivered once on subscription match, so recovering one needs a NEW subscription, see _check_tf_alive.
_TF_CACHE_S = 30.0

# Consecutive nav2 watchdog trips (no accepted goal or arrival in between) before the action clients are destroyed and recreated. 2 so a single ordinary nav2 restart is handled by re-discovery alone and only a persistent binding problem pays for a rebuild.
_MAX_NAV2_TRIPS_BEFORE_REBUILD = 2


class _State(Enum):
    WAITING_FOR_MAP = auto()
    PLANNING        = auto()
    NAVIGATING      = auto()
    TRAVELING       = auto()
    # Nav2 gave up on the current waypoint, but the robot may still be at (or be driven to) the goal. Hold here and verify arrival from the TF pose instead of trusting nav2's verdict, see _do_verify_check. No goal is in flight in this state.
    VERIFYING       = auto()
    ROTATING        = auto()
    COMPLETE        = auto()


# ---------------------------------------------------------------------------
# Helper: heading degrees → quaternion (yaw only, in the map plane)
# ---------------------------------------------------------------------------

def _yaw_to_quaternion(yaw_deg: float) -> Quaternion:
    yaw = math.radians(yaw_deg)
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ExplorationNode(Node):

    def __init__(self) -> None:
        super().__init__('exploration_node')

        # ── Declare parameters ────────────────────────────────────────────
        self.declare_parameter('exploration.sampling_step_m', 3.0)
        self.declare_parameter('exploration.observation_rotation_increment', 30.0)
        self.declare_parameter('exploration.frontier_weight', 1.0)
        self.declare_parameter('exploration.coverage_weight', 1.0)
        self.declare_parameter('exploration.travel_cost_weight', 1.0)
        self.declare_parameter('exploration.exploration_completion_threshold', 0.90)
        self.declare_parameter('exploration.planner_coverage_warning_threshold', 0.90)
        self.declare_parameter('exploration.max_waypoints_per_plan', -1)  # -1 = auto-derive
        self.declare_parameter('exploration.replan_every_n_step', -1)  # <=0 = drain whole plan
        self.declare_parameter('exploration.mid_path_replan_ratio', 0.75)
        self.declare_parameter('exploration.map_file_path', '')
        self.declare_parameter('exploration.map_yaml_path', '')
        # See-while-moving: mark the camera FOV continuously along the path and aim
        # the Nav2 goal yaw at the most-uncovered area, INSTEAD of driving heading-
        # blind then spinning in place at the goal. True = new default behaviour;
        # False = legacy turn-at-goal (Spin sequence per waypoint).
        # SLAM stop flexibility. min_frontier_cells: treat "no frontiers" as total
        # visible frontier cells <= this (0 = strict, must be exactly none). no_progress_*:
        # also stop after N consecutive arrived waypoints each adding < eps new coverage
        # (0 streak = disabled), which catches persistent UNREACHABLE frontiers the
        # frontier tolerance alone cannot. Both gated to live SLAM in _finish_waypoint.
        # How long to keep re-planning while no waypoints can be produced (candidate pool starved / robot wedged) before giving up. The timer is 1 Hz so this is both a tick count and a duration in seconds. 0 = never give up (loop forever).
        self.declare_parameter('exploration.plan_timeout_s', 300.0)
        self.declare_parameter('exploration.min_frontier_cells', 0)
        # Blacklist a waypoint after this many consecutive Nav2 ABORTs so a persistently-unreachable frontier (in a wall/inflation) stops being re-planned as waypoint 0 forever. <=1 blacklists on the first abort.
        self.declare_parameter('exploration.abort_blacklist_after', 3)
        # Arrival failsafe (see _do_verify_check). Nav2 reporting failure does not prove the robot is not at the goal: it may have stopped just outside its own goal checker, or a human may drive it the last stretch (the teleop rescue described in the package README). On failure the node therefore holds in VERIFYING and decides from the TF pose instead. arrival_tolerance_m is 2x nav2's xy_goal_tolerance (0.2 in the live params files) and orientation is ignored entirely, so any yaw counts as arrived.
        self.declare_parameter('exploration.arrival_tolerance_m', 0.4)
        # Seconds the robot must stay STILL (not a wall-clock cap) before the waypoint is finally declared failed. Motion resets this, so a human actively driving keeps the window open. 0 disables the failsafe and restores the immediate-abort behaviour.
        self.declare_parameter('exploration.arrival_verify_timeout_s', 30.0)
        # Pose delta per tick that counts as real motion rather than TF/SLAM jitter. Without it, estimator noise alone would look like movement and hold the window open forever on a parked robot.
        self.declare_parameter('exploration.arrival_motion_eps_m', 0.05)
        # Nav2 silent-death watchdog (see _check_nav2_alive). rclpy never fails a pending result future when its action server disappears, so a nav2 that dies or is restarted mid-goal leaves this node waiting in TRAVELING forever: no SUCCEEDED, no ABORTED, no result at all. The watchdog bounds that wait so the waypoint can be handed to the arrival failsafe and the next goal send can re-link to the restarted server.
        self.declare_parameter('nav2.watchdog_enabled', True)
        # Seconds without NavigateToPose feedback before a nav2 that is STILL advertising its action server is presumed wedged. Fires long before nav2's own action_server_result_timeout of 900.0, so the node reacts in seconds rather than a quarter hour. 0 disables this half only; a server that vanishes from the graph is still detected immediately.
        self.declare_parameter('nav2.watchdog_stall_timeout_s', 30.0)
        self.declare_parameter('exploration.no_progress_streak', 0)
        self.declare_parameter('exploration.no_progress_eps', 0.005)
        self.declare_parameter('exploration.see_while_moving', True)
        # How far the robot travels (m) between successive along-path FOV marks when
        # see_while_moving is on. Small enough to tile the path, large enough to keep
        # the mark cost and RViz traffic bounded.
        self.declare_parameter('exploration.observe_step_m', 0.5)
        self.declare_parameter('nav2.costmap_inflation_radius', 0.4)
        self.declare_parameter('nav2.spin_time_allowance', 10.0)
        self.declare_parameter('camera.max_detection_range', 6.0)
        self.declare_parameter('camera.fov_horizontal', 87.0)
        self.declare_parameter('lidar.num_rays', 360)
        # Fallback only; the real value comes from perception_system_parameters.yaml
        # via the launch file. base_link matches Husarion/ROSbot XL, whose TF tree
        # has no base_footprint (a wrong frame here makes every TF pose lookup fail).
        self.declare_parameter('lidar.base_frame', 'base_link')

        # ── Build planner config dict ─────────────────────────────────────
        gp = self.get_parameter
        self._cfg: dict = {
            'sampling_step_m':                    gp('exploration.sampling_step_m').value,
            'observation_rotation_increment':     gp('exploration.observation_rotation_increment').value,
            'frontier_weight':                    gp('exploration.frontier_weight').value,
            'coverage_weight':                    gp('exploration.coverage_weight').value,
            'travel_cost_weight':                 gp('exploration.travel_cost_weight').value,
            'exploration_completion_threshold':   gp('exploration.exploration_completion_threshold').value,
            'planner_coverage_warning_threshold': gp('exploration.planner_coverage_warning_threshold').value,
            'max_detection_range':                gp('camera.max_detection_range').value,
            'fov_horizontal':                     gp('camera.fov_horizontal').value,
            'num_rays':                           gp('lidar.num_rays').value,
            # -1 means auto-derive; plan_waypoints expects None for the same behaviour.
            'max_waypoints_per_plan':             (
                gp('exploration.max_waypoints_per_plan').value
                if gp('exploration.max_waypoints_per_plan').value > 0 else None
            ),
            'mid_path_replan_ratio':              gp('exploration.mid_path_replan_ratio').value,
            'min_frontier_cells':                 gp('exploration.min_frontier_cells').value,
            'abort_blacklist_after':              gp('exploration.abort_blacklist_after').value,
        }
        self._inflation_m:    float = gp('nav2.costmap_inflation_radius').value
        self._spin_allowance: float = gp('nav2.spin_time_allowance').value
        self._increment:      float = self._cfg['observation_rotation_increment']
        self._base_frame:     str   = gp('lidar.base_frame').value
        self._see_while_moving: bool = gp('exploration.see_while_moving').value
        self._observe_step_m:  float = gp('exploration.observe_step_m').value
        # No-progress stop guard (live SLAM only). Streak of consecutive arrivals that
        # each add < eps new coverage; when it reaches the threshold, terminate.
        self._no_progress_streak: int = gp('exploration.no_progress_streak').value
        self._no_progress_eps:  float = gp('exploration.no_progress_eps').value
        self._stale_streak:       int = 0
        self._cov_at_last_arrival: float = 0.0
        # Empty-plan escalation (see _do_planning). Timer is 1 Hz => streak == seconds.
        self._plan_timeout_s: float = gp('exploration.plan_timeout_s').value
        self._empty_plan_streak:    int   = 0
        # Arrival failsafe tuning, see the declarations above and _do_verify_check.
        self._arrival_tolerance_m:    float = gp('exploration.arrival_tolerance_m').value
        self._arrival_verify_timeout_s: float = gp('exploration.arrival_verify_timeout_s').value
        self._arrival_motion_eps_m:   float = gp('exploration.arrival_motion_eps_m').value
        # Nav2 silent-death watchdog tuning, see the declarations above and _check_nav2_alive.
        self._nav2_watchdog_enabled:   bool = gp('nav2.watchdog_enabled').value
        self._nav2_stall_timeout_s:   float = gp('nav2.watchdog_stall_timeout_s').value
        # Pixel position of the last along-path FOV mark, so travel marking is
        # throttled by distance rather than firing every timer tick.
        self._last_mark_px: tuple[int, int] | None = None
        # Replan cadence: under live SLAM, return to PLANNING every N arrived
        # waypoints instead of draining the whole (increasingly stale) plan. <=0
        # disables (drain). Only applied when exploring a live /map (self._is_slam);
        # a given static map can't change, so draining is always correct there.
        self._replan_every_n_step: int = gp('exploration.replan_every_n_step').value

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_status    = self.create_publisher(String,  '/exploration/status',    10)
        self._pub_coverage  = self.create_publisher(Float32, '/exploration/coverage',  10)

        # RViz visualisation publishers
        self._pub_covered_mask = self.create_publisher(
            OccupancyGrid, '/exploration/covered_mask',   1)
        self._pub_nav_mask     = self.create_publisher(
            OccupancyGrid, '/exploration/navigable_mask', 1)
        self._pub_wp_markers   = self.create_publisher(
            MarkerArray,   '/exploration/waypoints',      1)
        self._pub_goal_marker  = self.create_publisher(
            Marker,        '/exploration/current_goal',   1)
        self._pub_fov          = self.create_publisher(
            LaserScan,     '/exploration/robot_fov',      1)

        # ── Nav2 action clients ───────────────────────────────────────────
        self._nav_client  = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._spin_client = ActionClient(self, Spin,           '/spin')

        # ── TF buffer for robot pose ──────────────────────────────────────
        # TF health tracking for the listener watchdog (see _check_tf_alive). Set BEFORE the first buffer exists because _init_session() can run a pose lookup during construction, and that lookup writes _tf_fail_since.
        # Time of the first failed map -> base_frame lookup in the current run of failures, None while TF is healthy.
        self._tf_fail_since: float | None = None
        # How many times the listener has been rebuilt, so repeated recovery attempts are obvious in the log and in /exploration/status.
        self._tf_rebuilds: int = 0
        self._tf_buffer   = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=_TF_CACHE_S))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Exploration state ─────────────────────────────────────────────
        self._state:    _State                  = _State.WAITING_FOR_MAP
        self._md                                = None   # MapData
        self._session:  ExplorationSession | None = None
        self._waypoints: list                   = []
        self._wp_index:  int                    = 0
        # Safety net for inaccessible waypoints (nav2 abort/reject). On failure we try the NEXT waypoint in the current plan rather than re-planning to the same one; the failed spot is left UNVISITED (reachable from a later pose). _plan_all_failed guards the "start occupied" case (robot's own pose is the problem): if a whole plan fails end-to-end this many times in a row, run a nav2 recovery (Spin) to move the robot out of the inflation band.
        # The state machine itself is the concurrency guard: during the recovery Spin the state is NAVIGATING and the 1 Hz timer no-ops, so no separate flag is needed.
        self._plan_all_failed_streak: int       = 0
        # Wall-clock (sim-clock) time of the FIRST fully-failed plan in the current streak. The streak count alone is not a duration: when the robot sits inside the inflation band Nav2 cannot plan FROM that start and rejects every goal in ~30 ms, so a 20-waypoint plan exhausts in <1 s and the 5-plan cap is reached in a couple of seconds. The stop now also requires exploration.plan_timeout_s to have elapsed.
        self._plan_all_failed_since: float | None = None
        # Separate clock for "robot parked in the inflation band". Distinct from the plan-exhaustion clock above because _send_nav_goal (which owns this one) runs for every waypoint of an abort cascade and must not reset that one.
        self._inflated_since: float | None = None
        self._coverage:  float                  = 0.0
        self._heading:   float                  = 0.0   # current robot heading (degrees)
        self._goal_handle                       = None  # NavigateToPose goal handle
        # Arrival-failsafe state (VERIFYING), see _do_verify_check.
        # World coords of the goal currently/last sent to nav2. _send_nav_goal computes these to fill the goal message but nothing kept them, and the failsafe must measure against exactly the pose that was sent (not the waypoint's stored col/row, which SLAM can make stale). None means no goal has been sent yet, which disables the check.
        self._current_goal_xy: tuple[float, float] | None = None
        # Start of the current STILL period. Reset on every detected motion, so the timeout measures how long the robot has been stationary, not how long the window has been open.
        self._verify_since: float | None = None
        # Robot world pose at the previous verify tick, for the motion delta.
        self._verify_last_pos: tuple[float, float] | None = None
        # Time of the last sign of life from nav2 on the in-flight goal (armed at send, then refreshed by acceptance and every feedback). None means no goal is being tracked, which is what disarms the watchdog when nothing is in flight.
        self._last_nav_progress_t: float | None = None
        # The action client owning the in-flight goal, so the liveness probe checks the right server (Spin during a rotation, NavigateToPose otherwise).
        self._active_client = None
        # Consecutive watchdog trips with no accepted goal or arrival in between, and how many times the clients have been rebuilt as a result.
        self._nav2_trips: int = 0
        self._nav2_rebuilds: int = 0
        # Operator teleop mode (see _on_teleop_enabled): while true the planner keeps running but no nav2 goals are sent, and reaching a waypoint by driving counts as arriving.
        self._teleop_enabled: bool = False
        # Latch so one drive credits one waypoint: cleared on acceptance, set again once the driver leaves the tolerance circle. Without it, advancing to a waypoint that is already within tolerance of the robot credits the rest of the plan from a single spot.
        self._teleop_left_last_goal: bool = True

        # ── Map source: static PGM or live /map ───────────────────────────
        # self._is_slam gates the replan cadence: True only when the map is live
        # (subscribed to /map) and can therefore grow/change under the robot. A
        # given static map is loaded once and never re-subscribes, so it can't go
        # stale, draining the whole plan is always correct there.
        map_path = gp('exploration.map_file_path').value
        self._is_slam: bool = not bool(map_path)
        # The planner needs the map source too: frontiers are unknown cells, which only a live /map can ever resolve. On a static map they are permanent voids, so plan_waypoints must ignore frontier gain entirely (scoring AND the stop test) or exploration can never complete. Injected before _init_session() below so the session is built with it.
        self._cfg['is_slam'] = self._is_slam
        if map_path:
            yaml_path = gp('exploration.map_yaml_path').value
            self.get_logger().info(f'Loading static map from {map_path}')
            self._md = load_map(map_path, yaml_path, self._inflation_m)
            # TF is typically not populated yet at construction time, so session
            # init may fail here. Stay in WAITING_FOR_MAP and let the timer retry
            # init once TF is up, mirroring the SLAM path. The map itself is already loaded, so _on_map is never wired: the timer drives the retry.
            if self._init_session():
                self._state = _State.PLANNING
            else:
                self._state = _State.WAITING_FOR_MAP
        else:
            self.get_logger().info('Waiting for /map topic...')
            self._map_sub = self.create_subscription(
                OccupancyGrid, '/map', self._on_map, 10)

        # ── Operator teleop switch ────────────────────────────────────────
        # Deliberately VOLATILE (the default sensor-data-free profile), NOT transient-local. A TRANSIENT_LOCAL subscription is the one durability combination that refuses to match a VOLATILE publisher, and `ros2 topic pub` is VOLATILE unless told otherwise, so latching here made the documented one-liner sit at "Waiting for at least 1 matching subscription(s)" forever. VOLATILE matches both a plain `ros2 topic pub` and a transient-local one. Latching bought nothing anyway: the mode is held in _teleop_enabled, and this node is the subscriber rather than the publisher.
        self._teleop_sub = self.create_subscription(
            Bool, '/exploration/teleop_enabled', self._on_teleop_enabled, 10)

        # ── Main timer (1 Hz) ─────────────────────────────────────────────
        self._timer = self.create_timer(1.0, self._exploration_timer)

    # ── Map callbacks ─────────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Decode OccupancyGrid and (re)build MapData. Preserves covered_mask."""
        new_H = msg.info.height
        new_W = msg.info.width
        data = np.array(msg.data, dtype=np.float64).reshape(new_H, new_W)
        data = np.flipud(data)                          # ROS row 0 = world bottom
        p_occ = np.where(data < 0, 0.5, data / 100.0)  # -1 (unknown) → 0.5

        new_origin_x = msg.info.origin.position.x
        new_origin_y = msg.info.origin.position.y
        res = msg.info.resolution

        covered = None
        if self._md is not None:
            old_covered = self._md.covered_mask
            # Map resized or origin shifted: reproject covered_mask into the
            # new pixel space by world position (shared helper, also used by
            # the manual baseline node so the behaviour cannot drift).
            covered = reproject_covered_mask(
                old_covered,
                self._md.resolution, self._md.origin_x, self._md.origin_y,
                (new_H, new_W), res, new_origin_x, new_origin_y,
            )
            # A geometry change must never lose coverage (SLAM maps only
            # grow/shift). A big shrink means the mask was corrupted before
            # reprojection, make it loud instead of silent.
            n_old, n_new = int(old_covered.sum()), int(covered.sum())
            if n_new < 0.95 * n_old:
                self.get_logger().error(
                    f'covered_mask lost cells on map change: {n_old} -> {n_new} '
                    f'({old_covered.shape[1]}x{old_covered.shape[0]} -> {new_W}x{new_H}, '
                    f'origin ({self._md.origin_x:.3f},{self._md.origin_y:.3f}) -> '
                    f'({new_origin_x:.3f},{new_origin_y:.3f}))')

        self._md = build_map_data(
            p_occ,
            resolution=res,
            origin_x=new_origin_x,
            origin_y=new_origin_y,
            inflation_radius_m=self._inflation_m,
            covered_mask=covered,
        )

        # Keep the session's map reference in sync so plan_waypoints_raw uses
        # the current covered_mask and navigable_mask, not the one from session init.
        # set_map also regenerates the session's candidate list from the new
        # navigable_mask (newly revealed SLAM cells become candidates).
        if self._session is not None:
            self._session.set_map(self._md)

        if self._state == _State.WAITING_FOR_MAP:
            if self._init_session():
                self.get_logger().info('Map received, starting exploration.')
                self._state = _State.PLANNING
            # else: pose not ready, stay WAITING_FOR_MAP, retry on next map callback.

    def _init_session(self) -> bool:
        """Initialise the session, seeding the start from the robot's TF pose.

        Returns False if the robot pose is not yet available from TF (TF tree not
        populated right after startup); the caller stays in WAITING_FOR_MAP so the
        next map callback retries. Without a pose there is nothing meaningful to do
       , the whole plan is seeded from where the robot is.
        """
        pos = self._get_robot_world_pos()
        if pos is None:
            self.get_logger().warning(
                'Robot pose unavailable from TF; waiting for TF before starting exploration.')
            return False
        robot_x, robot_y = pos

        self._session = ExplorationSession(self._md, self._cfg)
        H = self._md.navigable_mask.shape[0]
        cx, cy = world_to_pixel(robot_x, robot_y, self._md.resolution,
                                self._md.origin_x, self._md.origin_y, H)
        self._session.nearest_start(cx, cy)
        # Seed _heading from TF so the first get_headings() sort and the first
        # spin delta are computed relative to the robot's actual initial heading,
        # not the hardcoded 0.0 (East) default.
        actual_heading = self._get_robot_heading_deg()
        if actual_heading is not None:
            self._heading = actual_heading
        else:
            self.get_logger().warning(
                'TF heading unavailable at session init - defaulting to 0° (East).')
        self.get_logger().info(
            f'ExplorationSession initialised. Initial heading: {self._heading:.1f}°.')
        return True

    # ── Robot pose helpers ────────────────────────────────────────────────

    def _lookup_robot_transform(self):
        """Look up map -> base_frame, or return None, tracking how long it has failed.

        The single place the robot pose enters this node, so the health bookkeeping the
        TF watchdog needs (see _check_tf_alive) lives here once instead of being
        duplicated in every caller. Returning None rather than raising keeps every
        existing call site behaving exactly as before: they already treat None as
        "TF not ready, wait and retry".
        """
        try:
            t = self._tf_buffer.lookup_transform('map', self._base_frame, Time())
        except Exception:
            # Stamp only the FIRST failure of a run: the watchdog measures how long TF has been broken, so later failures must not keep pushing the clock forward.
            if self._tf_fail_since is None:
                self._tf_fail_since = self.get_clock().now().nanoseconds / 1e9
            return None
        self._tf_fail_since = None   # a successful lookup means TF is healthy again
        return t

    def _get_robot_world_pos(self) -> tuple[float, float] | None:
        """Return (x, y) in world metres from TF, or None on TF failure.

        None (not a fabricated (0,0), which is a valid coordinate) so callers can
        distinguish 'TF not ready' from a genuine pose and wait/skip accordingly.
        """
        t = self._lookup_robot_transform()
        if t is None:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _get_robot_heading_deg(self) -> float | None:
        """Return robot yaw in degrees (0=East, CCW positive) from TF.

        Extracts yaw from the map→base_link quaternion using the standard
        atan2(2(wz + xy), 1 - 2(y² + z²)) formula. Returns None on TF failure
        so callers can decide whether to fall back or abort.
        """
        t = self._lookup_robot_transform()
        if t is None:
            return None
        q = t.transform.rotation
        # Yaw from quaternion (rotation about Z axis only)
        yaw_rad = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return math.degrees(yaw_rad)

    def _get_robot_pixel_pos(self) -> tuple[int, int] | None:
        """Robot (col, row) from TF, or None if the pose is unavailable."""
        pos = self._get_robot_world_pos()
        if pos is None:
            return None
        x, y = pos
        H = self._md.navigable_mask.shape[0]
        return world_to_pixel(x, y, self._md.resolution,
                              self._md.origin_x, self._md.origin_y, H)

    def _max_range_px(self) -> int:
        """Sensor range in pixels for the current map (recomputed live: resolution
        changes when SLAM rebuilds the map)."""
        return max(1, int(self._cfg['max_detection_range'] / self._md.resolution))

    # ── Main timer ────────────────────────────────────────────────────────

    def _exploration_timer(self) -> None:
        self._publish_status()

        # Before the state dispatch, and deliberately before the WAITING_FOR_MAP early return below: TF health is state-independent, and a static-map run whose TF breaks sits in WAITING_FOR_MAP retrying _init_session forever, so a check inside the per-state branches would never fire in exactly one of the cases that needs it.
        self._check_tf_alive()

        if self._state == _State.WAITING_FOR_MAP:
            # Static map: the map is already loaded but session init failed at
            # startup because TF wasn't ready. There is no /map callback to drive
            # a retry, so retry init here each tick until TF is available.
            if not self._is_slam and self._session is None and self._init_session():
                self.get_logger().info('TF ready, starting exploration.')
                self._state = _State.PLANNING
            return

        # Always refresh mask overlays when the map is available
        self._publish_mask_overlays()

        # Teleop runs alongside the normal dispatch below: the planner keeps producing waypoints (so the operator can see the target and coverage keeps growing) while this credits the waypoint as soon as they drive to it.
        if self._teleop_enabled:
            self._do_teleop_check()
            self._publish_fov()

        if self._state == _State.PLANNING:
            self._do_planning()

        elif self._state == _State.NAVIGATING:
            # Waiting for the goal-accepted callback, which also never arrives if nav2 died between the send and the response.
            self._check_nav2_alive()

        elif self._state == _State.TRAVELING:
            self._check_nav2_alive()
            if self._state != _State.TRAVELING:
                return  # the watchdog tripped and moved us on, do not drive a stale goal
            self._do_travel_check()
            self._publish_fov()

        elif self._state == _State.VERIFYING:
            self._do_verify_check()
            self._publish_fov()

        elif self._state == _State.ROTATING:
            # A Spin goal is in flight here, so this state is exposed to exactly the same silent-death hang as TRAVELING and was previously uncovered.
            self._check_nav2_alive()
            if self._state != _State.ROTATING:
                return  # the watchdog tripped and moved us on, do not drive a stale spin
            self._do_rotation()

        elif self._state == _State.COMPLETE:
            self.get_logger().info(
                f'Exploration complete. Final coverage: {self._coverage:.1%}',
                once=True,
            )
            self._timer.cancel()

    # ── State handlers ────────────────────────────────────────────────────

    def _do_planning(self) -> None:
        pos = self._get_robot_world_pos()
        if pos is None:
            self.get_logger().warning('Robot pose unavailable from TF; skipping plan cycle.')
            return  # stay in PLANNING, retry on next timer tick
        robot_x, robot_y = pos
        waypoints, ratio, no_frontiers, _ = self._session.plan_waypoints_raw(robot_x, robot_y)
        self._coverage = ratio
        threshold = self._cfg['exploration_completion_threshold']

        # Only declare complete when we have confirmed no frontiers AND coverage is
        # above threshold.  An empty waypoint list without no_frontiers means the map
        # is still mostly unknown (SLAM startup), stay in PLANNING and retry.
        if no_frontiers and ratio >= threshold:
            self.get_logger().info(
                f'Exploration complete, coverage {ratio:.1%}.')
            self._clear_rviz_markers()
            self._state = _State.COMPLETE
            return

        if not waypoints:
            self._handle_empty_plan(ratio, no_frontiers)
            return

        self._empty_plan_streak = 0   # a real plan proves the pool is healthy again
        self._waypoints  = waypoints
        self._wp_index   = 0
        # Note: _plan_all_failed_streak is NOT reset here, a fresh plan alone doesn't prove the robot is unstuck (the same unreachable set can recur). It resets only on a real SUCCEEDED arrival (in _on_nav_result), which proves progress.
        # Start the no-progress streak fresh for this plan/phase so it measures gain within the current exploration, not across a forced replan boundary.
        self._stale_streak = 0
        self._cov_at_last_arrival = self._coverage
        self._publish_waypoint_markers()
        self.get_logger().info(
            f'Plan: {len(waypoints)} waypoints, coverage so far {ratio:.1%}.')
        self._send_nav_goal(waypoints[0])

    # Escalation thresholds for a starved planner (timer is 1 Hz → ticks == seconds).
    _EMPTY_PLAN_RESTORE_AFTER_S = 5     # restore the unreachable set at this point
    _EMPTY_PLAN_WARN_EVERY_S    = 30    # keep the situation visible in the log

    def _handle_empty_plan(self, ratio: float, no_frontiers: bool) -> None:
        """No waypoints could be planned. Retry, but loudly and with escalation.

        Retrying is deliberate and useful: it gives the robot (or an operator) time to
        get out of a bad spot, and transient blockages clear on their own. What must not
        happen is the old behaviour, retrying silently forever with no recovery, which
        hid a starved candidate pool for ~20 minutes at 56.6 % coverage.

        Escalation (streak is in seconds at the 1 Hz timer):
          <5 s  : quiet retry (transients resolve themselves)
          5 s   : WARN + restore every waypoint parked as unreachable, so the next tick
                  re-plans against a full pool
          >5 s  : keep retrying, WARN every 30 s (window to free the robot manually)
          >=timeout: ERROR and stop, so a genuinely dead run is not left spinning
                  (exploration.plan_timeout_s; 0 disables the stop entirely)
        """
        self._empty_plan_streak += 1
        s = self._empty_plan_streak

        if s == self._EMPTY_PLAN_RESTORE_AFTER_S:
            restored = self._session.clear_unreachable() if self._session else 0
            self.get_logger().warning(
                f'Candidate pool starved for {s}s (coverage {ratio:.1%}, '
                f'frontiers={not no_frontiers}): restored {restored} waypoint(s) '
                f'previously abandoned as unreachable; re-planning.')
            return

        if (self._plan_timeout_s
                and s >= self._plan_timeout_s):
            self.get_logger().error(
                f'No waypoints could be planned for {s}s (coverage {ratio:.1%}). '
                'Candidate pool is exhausted and could not be recovered, stopping '
                'exploration. Raise exploration.plan_timeout_s (0 = never give up) '
                'to keep retrying longer.')
            self._clear_rviz_markers()
            self._state = _State.COMPLETE
            return

        if s % self._EMPTY_PLAN_WARN_EVERY_S == 0:
            self.get_logger().warning(
                f'Still no waypoints after {s}s (coverage {ratio:.1%}, '
                f'frontiers={not no_frontiers}). Robot may be wedged, move it to clear '
                'space if possible; still retrying.')
        elif s < self._EMPTY_PLAN_RESTORE_AFTER_S:
            self.get_logger().info(
                f'No waypoints planned yet (coverage {ratio:.1%}, '
                f'frontiers={not no_frontiers}), retrying next cycle.')

    def _send_nav_goal(self, wp) -> None:
        # Single choke point for every navigation goal, so suppressing here is enough to keep nav2 and the human operator from fighting over cmd_vel. The planner keeps running and the target is still published, so the driver can see where exploration wanted to go.
        if self._teleop_enabled:
            # The target MUST still be recorded even though no goal is sent: _do_teleop_check measures the driver against _current_goal_xy, and leaving it pinned to the previously sent goal made every subsequent waypoint "reached" from the same spot, crediting the whole plan in a few ticks. Taken from the waypoint's own world coords rather than re-deriving them from the map, which may not be loaded yet.
            self._current_goal_xy = (float(wp.x), float(wp.y))
            self.get_logger().info(
                f'Teleop enabled: not sending waypoint {self._wp_index} to nav2. '
                'Drive there manually, or publish teleop_enabled false to resume.',
                throttle_duration_sec=10.0)
            self._publish_current_goal(wp)
            # TRAVELING, not left as-is: the waypoint is now being pursued (by a human), and _do_teleop_check only credits arrivals in a pursuing state.
            self._state = _State.TRAVELING
            return
        # Nav2 cannot plan FROM a start pose it considers occupied/inflated: it rejects every goal in ~30 ms regardless of where the goal is. Because an abort immediately sends the NEXT waypoint (callback cascade, not tick-limited), a whole plan burns through in <1 s and the 1 Hz replan repeats it, thousands of doomed goals while the robot is wedged. Sending nothing until the pose is plannable again collapses that to ~1 check per second and leaves Nav2 alone.
        pos = self._get_robot_pixel_pos()
        if pos is not None and not self._pose_is_navigable(pos):
            # Suppressing the send also means no abort callback fires, so the plan-exhaustion streak never advances, this path must therefore carry its OWN time bound, or a wedged robot would loop at 1 Hz forever. Reuse the same stuck clock and plan_timeout_s as _on_plan_exhausted_failed.
            now = self.get_clock().now().nanoseconds / 1e9
            if self._inflated_since is None:
                self._inflated_since = now
            stuck_for = now - self._inflated_since
            if self._plan_timeout_s and stuck_for >= self._plan_timeout_s:
                self.get_logger().error(
                    f'Robot has been inside the inflation band for {stuck_for:.0f}s '
                    f'(>= plan_timeout_s {self._plan_timeout_s:.0f}s) and nav2 cannot '
                    'plan from there. Stopping exploration.')
                self._clear_rviz_markers()
                self._state = _State.COMPLETE
                return
            self.get_logger().warning(
                f'Robot is inside the inflation band (nav2 cannot plan from here); '
                f'not sending waypoint {self._wp_index}. Stuck for {stuck_for:.0f}s, '
                'move the robot to free space.',
                throttle_duration_sec=10.0)
            self._state = _State.PLANNING   # re-check on the next 1 Hz tick
            return

        # Pose is plannable again: clear only the INFLATION clock, so a brief excursion
        # does not count toward a later episode. Deliberately NOT the plan-exhaustion
        # clock (_plan_all_failed_since): this method runs for every waypoint of an abort
        # cascade, and clearing it here would stop `stuck_for` ever accumulating in the
        # "navigable pose but all goals unreachable" case, silently disabling that stop.
        self._inflated_since = None

        # server_is_ready() is a non-blocking graph query; wait_for_server() sleeps in a loop and this node runs on a single-threaded rclpy.spin, so waiting here froze EVERYTHING (TF, /map, the 1 Hz timer, status publishing) for up to 5s per attempt, exactly while nav2 was unhealthy and visibility mattered most. Returning to PLANNING instead lets the next tick retry, which is also what re-links to a nav2 that has restarted under us.
        if not self._nav_client.server_is_ready():
            self.get_logger().warning(
                'Nav2 action server not available, retrying next tick.',
                throttle_duration_sec=10.0)
            self._state = _State.PLANNING
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp    = self.get_clock().now().to_msg()
        wx, wy = pixel_to_world(wp.col, wp.row, self._md.resolution,
                                self._md.origin_x, self._md.origin_y,
                                self._md.navigable_mask.shape[0])
        goal.pose.pose.position.x = float(wx)
        goal.pose.pose.position.y = float(wy)
        # Remember the goal in world coords so the arrival failsafe can measure the robot against the exact pose nav2 was asked for, see _do_verify_check.
        self._current_goal_xy = (float(wx), float(wy))
        # See-while-moving: aim the goal yaw at the most-uncovered area so the single arrival look (no spin) sees the most. Nav2 drives the robot to end facing this heading (within yaw_goal_tolerance). Legacy path leaves 0° and spins.
        goal_yaw = self._best_uncovered_heading(wp) if self._see_while_moving else 0.0
        goal.pose.pose.orientation = _yaw_to_quaternion(goal_yaw)

        self._last_mark_px = None   # fresh travel leg -> restart the observe-distance throttle
        self._state = _State.NAVIGATING
        # Arm BEFORE sending, not on acceptance: send_goal_async's future never fails when the server vanishes (rclpy has no set_exception), so a nav2 destroyed between the send and the acceptance callback left the clock at None and the watchdog permanently disarmed. That is the observed 10-minute hang.
        self._arm_action_watchdog(self._nav_client)
        send_future = self._nav_client.send_goal_async(
            goal,
            feedback_callback=self._on_nav_feedback,
        )
        send_future.add_done_callback(self._on_goal_response)
        self._publish_current_goal(wp)
        self.get_logger().info(
            f'Navigating to waypoint {self._wp_index} '
            f'(col={wp.col}, row={wp.row}).')

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Nav2 rejected goal, trying next waypoint.')
            self._try_next_waypoint()
            return
        self._goal_handle = goal_handle
        self._state = _State.TRAVELING
        # Refresh the clock (already armed at send) and clear the trip streak: a server that accepts a goal is demonstrably alive, so past trips must not accumulate toward a client rebuild across unrelated episodes.
        self._arm_action_watchdog(self._nav_client)
        self._nav2_trips = 0
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_feedback(self, feedback_msg) -> None:
        # Feedback doubles as nav2's heartbeat for the silent-death watchdog (see _check_nav2_alive): rclpy never fails a pending result future when the action server disappears, so a nav2 that dies mid-goal would otherwise leave this node waiting in TRAVELING forever with no result of any kind. distance_remaining is still available at feedback_msg.feedback.distance_remaining if it is ever needed.
        self._last_nav_progress_t = self.get_clock().now().nanoseconds / 1e9

    def _on_nav_result(self, future) -> None:
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._accept_arrival()
        elif status == GoalStatus.STATUS_CANCELED:
            # A CANCELED result is a DELIBERATE interruption we requested (mid-path replan cancels the current goal, see _do_travel_check), NOT an inaccessible waypoint. The code that issued the cancel already set the next state (PLANNING), so do nothing here: must not advance _wp_index, count toward the stuck-streak, or trigger recovery. Ignoring it also closes the race where the stale cancel result lands after a new goal has already moved the state back to NAVIGATING.
            self.get_logger().debug(f'Nav2 goal canceled (status {status}), ignoring.')
        else:
            # ABORTED (or other failure). Nav2 giving up does NOT prove the waypoint was not reached: it may have stopped just outside its own goal checker, or a human may drive the robot the last stretch (the teleop rescue in the README). Rather than blacklisting on nav2's word, hold in VERIFYING and let the TF pose decide, see _do_verify_check.
            if self._state in (_State.TRAVELING, _State.NAVIGATING):
                # The goal is already terminal, so nothing is in flight to cancel and the handle must not be reused.
                self._goal_handle = None
                if self._begin_arrival_verification(
                        f'Nav2 goal ended with status {status}'):
                    return
                # Failsafe disabled (timeout <= 0) or no goal pose recorded: keep the original immediate-abort behaviour.
                self._reject_waypoint(status)

    def _begin_arrival_verification(self, reason: str) -> bool:
        """Enter the VERIFYING window, or report that the failsafe cannot run.

        Shared by the two ways a goal can stop being driveable: nav2 reporting a
        failure (_on_nav_result) and nav2 going silent (_check_nav2_alive), so both
        paths open the window identically and cannot drift apart.

        Returns True when the window was entered, False when the caller must fall
        back to rejecting the waypoint outright.
        """
        if self._arrival_verify_timeout_s <= 0 or self._current_goal_xy is None:
            return False
        self.get_logger().warning(
            f'{reason}; verifying arrival from the robot pose for up to '
            f'{self._arrival_verify_timeout_s:.0f}s of no motion before giving up on '
            f'waypoint {self._wp_index}.')
        self._verify_since = self.get_clock().now().nanoseconds / 1e9
        self._verify_last_pos = self._get_robot_world_pos()
        self._disarm_action_watchdog()   # no goal in flight any more
        self._state = _State.VERIFYING
        return True

    def _accept_arrival(self) -> None:
        """Treat the current waypoint as reached and run the arrival observation.

        Shared by the nav2 SUCCEEDED path and the arrival failsafe, so a
        pose-verified arrival is indistinguishable downstream from one nav2
        reported itself and the two cannot drift apart.

        Only acts from TRAVELING or VERIFYING, the two states in which a goal's
        outcome is still undecided. Both are left before this returns, so a late or
        duplicate result (a SUCCEEDED from the OLD nav2 landing after the watchdog
        already handed the waypoint to the failsafe, or after the failsafe accepted
        it) is silently ignored instead of marking the same waypoint twice and
        double-advancing the plan.
        """
        if self._state not in (_State.TRAVELING, _State.VERIFYING):
            self.get_logger().debug(
                f'Ignoring a late arrival for waypoint {self._wp_index}: '
                f'already resolved (state {self._state.name}).')
            return
        # A real arrival proves the robot is not stuck: clear the failed-plan streak and its stuck-since clock, so the time gate measures the CURRENT episode.
        self._plan_all_failed_streak = 0
        self._plan_all_failed_since = None
        self._inflated_since = None
        # The goal is resolved either way, so stop tracking it: leaves nothing for the watchdog to time out during the arrival observation, which is not a period nav2 sends feedback in.
        self._disarm_action_watchdog()
        self._nav2_trips = 0   # a real arrival proves the clients are working
        if self._session is not None and self._wp_index < len(self._waypoints):
            # Reachable after all: drop THIS waypoint's transient aborts. The parked unreachable set is deliberately left alone here, restoring it on every arrival livelocks on genuinely unreachable waypoints (restored -> instantly re-selected as highest-gain -> aborts -> re-parked, forever, with plans never empty so no timeout could catch it). It is restored only when the pool is actually starved, in _handle_empty_plan.
            self._session.on_arrive_clear_aborts(self._waypoints[self._wp_index])
        # Read actual robot heading from TF so heading sort and delta computation are correct regardless of approach direction.
        actual_heading = self._get_robot_heading_deg()
        if actual_heading is not None:
            self._heading = actual_heading
        else:
            self.get_logger().warning(
                'TF heading unavailable at waypoint arrival, '
                'using last known heading.')
        if self._see_while_moving:
            # No spin: Nav2 already ended the robot facing the best-uncovered heading (goal yaw). Mark that single arrival look, then finish.
            self._arrive_no_spin()
            return
        self._state        = _State.ROTATING
        self._rot_state    = RotationState(stop_after=3)
        # Recompute headings NOW from the current covered_mask, not the plan-time set.
        wp = self._waypoints[self._wp_index]
        self._refresh_waypoint_headings(wp)
        self._rot_headings = get_headings(wp, self._increment, self._heading)
        self._rot_idx = 0
        self._send_spin_goal()  # send first physical rotation

    def _reject_waypoint(self, status) -> None:
        """Give up on the current waypoint after a nav2 failure the failsafe could not rescue.

        This is the original pre-failsafe abort behaviour, extracted so the
        immediate path (failsafe disabled) and the post-window path share one
        implementation.
        """
        # The goal is finished with, so stop tracking it for the watchdog.
        self._disarm_action_watchdog()
        # Count the abort; a waypoint Nav2 aborts repeatedly is  PERSISTENTLY unreachable (in a wall/inflation) and gets  blacklisted so re-plans stop regenerating it as waypoint 0 (the observed livelock: same col/row aborted every cycle).
        if self._session is not None and self._wp_index < len(self._waypoints):
            wp = self._waypoints[self._wp_index]
            if self._session.on_nav_aborted(wp):
                self.get_logger().warning(
                    f'Waypoint {self._wp_index} (col={wp.col}, row={wp.row}) '
                    f'aborted repeatedly; blacklisting it.')
        self.get_logger().warning(
            f'Nav2 goal ended with status {status}, '
            f'waypoint {self._wp_index} inaccessible.')
        self._try_next_waypoint()

    def _check_tf_alive(self) -> None:
        """Rebuild the TF listener when map -> base_frame stays unresolvable.

        Observed failure: nav2 is restarted while exploration keeps running, and from
        then on every pose lookup fails permanently, while a FRESH tf2_echo in the same
        container resolves map -> base_link continuously. A new listener works, the
        long-lived one does not, which places the fault in this node's own
        TransformListener rather than in the TF data or the network.

        The mechanism is /tf_static. TransformListener subscribes to /tf as VOLATILE and
        to /tf_static as TRANSIENT_LOCAL, and a latched static sample is delivered only
        once, when a subscription MATCHES a publisher. Restarting the other container
        destroyed and recreated every static-TF publisher with new participant GUIDs, so
        each replacement's latched sample went only to subscriptions that matched
        afterwards. A new tf2_echo gets the full static set; this node's already-matched
        listener never receives the replacement, and static transforms are never
        republished periodically, so it cannot self-heal.

        The buffer therefore holds a complete dynamic tree but is missing a static link,
        and no cache length or clear() can fix that: the sample will never be re-sent to
        an already-matched subscription. Only a NEW subscription can obtain it, which is
        what this rebuild creates.
        """
        if not self._nav2_watchdog_enabled or self._tf_fail_since is None:
            return  # disabled, or TF is healthy so there is nothing to recover

        broken_for = self.get_clock().now().nanoseconds / 1e9 - self._tf_fail_since
        if broken_for < self._nav2_stall_timeout_s:
            return  # transient gap, every caller already handles a None pose by retrying

        # Diagnostics FIRST, because their absence is why the original failure was invisible: both pose helpers swallowed the exception, so the log never said which link was missing. Guarded because a broken buffer must never prevent the recovery below.
        try:
            # Duration() default = non-blocking. A timeout here would freeze the single-threaded executor, the same hazard already removed from wait_for_server.
            debug = self._tf_buffer.can_transform(
                'map', self._base_frame, Time(), return_debug_tuple=True)
            frames = self._tf_buffer.all_frames_as_yaml()
        except Exception as exc:
            debug = f'<can_transform raised: {exc}>'
            frames = '<unavailable>'
        self.get_logger().warning(
            f'TF has been unable to resolve map -> {self._base_frame} for '
            f'{broken_for:.0f}s (>= watchdog_stall_timeout_s '
            f'{self._nav2_stall_timeout_s:.0f}s). This is usually a /tf_static publisher '
            'that was replaced (nav2 or the robot bringup restarted), whose latched '
            'sample a listener matched before the restart never receives. Rebuilding the '
            f'TF listener (rebuild #{self._tf_rebuilds + 1}). can_transform: {debug}. '
            f'Frames currently in the buffer:\n{frames}')

        # A NEW buffer, not clear(): set_transform_static entries never expire, so a stale link from the dead publisher would otherwise survive and conflict with its replacement.
        self._tf_listener.unregister()
        self._tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=_TF_CACHE_S))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_rebuilds += 1
        # Re-arm rather than latch: if the rebuild did not help, this trips again after another timeout and logs the missing link again, which is strictly better than the previous behaviour of failing silently forever.
        self._tf_fail_since = None
        self.get_logger().info(
            'TF listener rebuilt with fresh /tf and /tf_static subscriptions; '
            'the buffer refills over the next few ticks.')

    def _arm_action_watchdog(self, client) -> None:
        """Start the silent-death clock for a goal about to be sent on `client`.

        Called immediately BEFORE every send_goal_async. Arming at send rather than at
        acceptance is the whole point: send_goal_async's future never fails when the
        server disappears, so a nav2 destroyed inside the send-to-accept window used to
        leave the clock unset and the watchdog disarmed forever.

        `client` is recorded so the liveness probe checks the server that actually owns
        the in-flight goal, which is the Spin server during a rotation.
        """
        self._active_client = client
        self._last_nav_progress_t = self.get_clock().now().nanoseconds / 1e9

    def _disarm_action_watchdog(self) -> None:
        """Stop tracking: no goal is in flight, so nav2 silence is expected, not a fault."""
        self._last_nav_progress_t = None
        self._active_client = None

    def _rebuild_action_clients(self) -> None:
        """Destroy and recreate both action clients after repeated watchdog trips.

        Mirrors the TF listener rebuild in _check_tf_alive, for the case where graph
        re-discovery alone does not restore a binding severed by a server restart.
        Destroying also drops the client's pending futures, so the stale goal and result
        futures that never resolve are released rather than accumulating.
        """
        self._nav_client.destroy()
        self._spin_client.destroy()
        self._nav_client  = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._spin_client = ActionClient(self, Spin,           '/spin')
        self._active_client = None
        self._nav2_rebuilds += 1
        self._nav2_trips = 0   # fresh clients, so start the escalation over
        self.get_logger().warning(
            f'Rebuilt the nav2 action clients after {_MAX_NAV2_TRIPS_BEFORE_REBUILD} '
            f'consecutive watchdog trips (rebuild #{self._nav2_rebuilds}).')

    def _check_nav2_alive(self) -> None:
        """Catch a nav2 that died or wedged without ever returning a result.

        rclpy never fails a pending result future when its action server disappears
        (there is no set_exception anywhere in ActionClient), so if nav2 dies or is
        restarted mid-goal, _on_nav_result is never called and the node would wait in
        TRAVELING forever: no SUCCEEDED, no ABORTED, nothing. Nothing else bounds that
        wait, because plan_timeout_s only guards the empty-plan and plan-exhaustion
        paths, and the mid-path replanner only fires once the robot has MOVED, which a
        robot with a dead nav2 does not do.

        Runs once per 1 Hz tick while a goal is in flight. Trips on either the action
        server vanishing from the graph (nav2 gone) or feedback drying up while it is
        still advertised (nav2 alive but wedged), then hands the waypoint to the
        arrival failsafe so a robot that did reach the goal still gets credit.
        """
        if not self._nav2_watchdog_enabled or self._last_nav_progress_t is None:
            return  # disabled, or no goal is being tracked so there is nothing to time out
        if self._teleop_enabled:
            return  # the human is driving and no goal was sent, so nav2 silence is expected

        now = self.get_clock().now().nanoseconds / 1e9
        silent_for = now - self._last_nav_progress_t
        # Probe the client that owns the IN-FLIGHT goal (the Spin client during a rotation), not always the nav client, or a spin-phase death would go undetected.
        client = self._active_client or self._nav_client
        if not client.server_is_ready():
            reason = ('Nav2 action server has disappeared (died or is restarting) '
                      f'after {silent_for:.0f}s on this goal')
        elif (self._nav2_stall_timeout_s > 0
              and silent_for >= self._nav2_stall_timeout_s):
            reason = (f'Nav2 has sent no feedback for {silent_for:.0f}s '
                      f'(>= watchdog_stall_timeout_s {self._nav2_stall_timeout_s:.0f}s) '
                      'while still advertising its action server, so it is presumed wedged')
        else:
            return  # healthy

        # Abandon the goal WITHOUT cancelling it: cancel_goal_async on a server that is gone returns another future that never resolves, which is the exact hang this watchdog exists to break. The stale handle is dropped instead, and a late result from the old server is ignored by the state guards in _on_nav_result.
        self._goal_handle = None
        self._disarm_action_watchdog()
        # Escalate only on CONSECUTIVE trips (any accepted goal or arrival resets this), so one ordinary nav2 restart takes the cheap path and only a persistent binding problem pays for a client rebuild.
        self._nav2_trips += 1
        if self._nav2_trips >= _MAX_NAV2_TRIPS_BEFORE_REBUILD:
            self._rebuild_action_clients()
        if not self._begin_arrival_verification(reason):
            # Failsafe unavailable (disabled, or no goal pose recorded): still must not stay wedged, so give the waypoint up and let the normal failure escalation run.
            self.get_logger().warning(f'{reason}; giving up on waypoint {self._wp_index}.')
            self._reject_waypoint(GoalStatus.STATUS_ABORTED)

    def _on_teleop_enabled(self, msg: Bool) -> None:
        """Hand the robot to a human operator, or take it back.

        Edge-triggered on purpose: the value is state, not a heartbeat. Publishing once
        is enough and the mode then persists with no republishing and no timeout, while
        a repeat of the value already held is ignored (no second cancel, no repeated
        log). That also makes the latched QoS safe, since a late subscriber replaying
        the retained sample cannot re-fire the transition.
        """
        enabled = bool(msg.data)
        if enabled == self._teleop_enabled:
            return  # same value as before, nothing to do
        self._teleop_enabled = enabled

        if enabled:
            self.get_logger().info(
                'Teleop enabled: exploration keeps planning and marking coverage, but '
                'sends no nav2 goals. Drive to within '
                f'{self._arrival_tolerance_m:.2f}m of the current waypoint and it counts '
                'as reached. Publish false to replan from wherever you stop.')
            # Hand back control immediately. Unlike a watchdog trip this cancel IS safe, because the server is alive and will answer; the CANCELED result is then ignored by the existing guard in _on_nav_result.
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                self._goal_handle = None
            self._disarm_action_watchdog()
            # Whatever the robot is standing on when the operator takes over does not count as a fresh arrival; require a deliberate drive first.
            self._teleop_left_last_goal = False
            return

        self.get_logger().info(
            'Teleop disabled: replanning from the robot current position.')
        # force_replan is the EXISTING "abandon the rest of this plan and go to PLANNING" path, and _do_planning already seeds from _get_robot_world_pos(), so replanning from wherever the operator parked needs no new code.
        self._advance_waypoint(force_replan=True)

    def _within_arrival_tolerance(self, pos: tuple[float, float]) -> float | None:
        """Distance to the current goal when inside arrival_tolerance_m, else None.

        Shared by the arrival failsafe and teleop mode so the two can never disagree
        about what counts as reaching a waypoint. Orientation is ignored entirely.
        """
        if self._current_goal_xy is None:
            return None
        gx, gy = self._current_goal_xy
        dist = math.hypot(pos[0] - gx, pos[1] - gy)
        return dist if dist <= self._arrival_tolerance_m else None

    def _do_teleop_check(self) -> None:
        """While a human drives, credit the waypoint once they reach it.

        Same rule as the arrival failsafe (arrival_tolerance_m, orientation ignored),
        but with no stillness timeout: in teleop the operator decides when to move on,
        so sitting still away from the goal is not a failure.
        """
        pos = self._get_robot_world_pos()
        if pos is None:
            return  # TF not ready, retry next tick
        # Keep crediting what the camera sees while the human drives, exactly as during autonomous travel.
        if self._see_while_moving:
            px = self._get_robot_pixel_pos()
            if px is not None:
                self._mark_moving_coverage(px)
        if self._state not in (_State.NAVIGATING, _State.TRAVELING):
            return  # no waypoint is being pursued right now, so there is nothing to reach
        dist = self._within_arrival_tolerance(pos)
        if dist is None:
            self._teleop_left_last_goal = True   # driver moved away, a new arrival can count again
            return
        # One arrival per drive. Accepting a waypoint advances the plan, and the NEXT waypoint can easily be within tolerance of where the robot already stands, which would credit the whole plan from one spot at 1 Hz (observed: every marker turning green after reaching waypoint 0). Require the driver to leave the tolerance circle before the next waypoint can be claimed.
        if not self._teleop_left_last_goal:
            return
        self._teleop_left_last_goal = False
        self.get_logger().info(
            f'Teleop: robot driven to within {dist:.2f}m of waypoint '
            f'{self._wp_index} (tolerance {self._arrival_tolerance_m:.2f}m); '
            'counting it as reached.')
        self._accept_arrival()

    def _do_verify_check(self) -> None:
        """Decide whether a nav2-failed waypoint was actually reached, from the TF pose.

        Runs once per 1 Hz tick while in VERIFYING. Accepts the waypoint as soon as
        the robot is within arrival_tolerance_m of the goal (orientation ignored
        entirely), so a human who drives the robot there gets an immediate response.
        Motion resets the stillness clock, which keeps the window open while someone
        is actively driving; only after arrival_verify_timeout_s of the robot sitting
        still is the waypoint finally declared failed.
        """
        pos = self._get_robot_world_pos()
        if pos is None:
            return  # TF momentarily unavailable, retry next tick. Deliberately counts as neither motion nor stillness, so a TF outage can neither hold the window open nor time it out.

        # Position check first: being at the goal ends the window immediately, whatever the motion state.
        dist = self._within_arrival_tolerance(pos)
        if dist is not None:
            self.get_logger().info(
                f'Arrival failsafe: robot is {dist:.2f}m from the goal '
                f'(tolerance {self._arrival_tolerance_m:.2f}m) despite the nav2 '
                f'failure; accepting waypoint {self._wp_index} as reached.')
            self._verify_since = None
            self._verify_last_pos = None
            self._accept_arrival()
            return

        now = self.get_clock().now().nanoseconds / 1e9
        # Any real movement means someone (or something) is still working on reaching the goal, so restart the stillness clock rather than counting down through it.
        if self._verify_last_pos is not None:
            moved = math.hypot(pos[0] - self._verify_last_pos[0],
                               pos[1] - self._verify_last_pos[1])
            if moved > self._arrival_motion_eps_m:
                self._verify_since = now
        self._verify_last_pos = pos

        # Keep recording what the camera sees while waiting: if a human drives the robot around during the window, that observation is real coverage and must not be thrown away. Already distance-throttled by observe_step_m.
        if self._see_while_moving:
            px = self._get_robot_pixel_pos()
            if px is not None:
                self._mark_moving_coverage(px)

        if self._verify_since is None:
            self._verify_since = now
            return
        still_for = now - self._verify_since
        if still_for >= self._arrival_verify_timeout_s:
            self.get_logger().warning(
                f'Arrival failsafe: robot stationary for {still_for:.0f}s without '
                f'reaching waypoint {self._wp_index}; treating it as failed.')
            self._verify_since = None
            self._verify_last_pos = None
            self._reject_waypoint(GoalStatus.STATUS_ABORTED)

    def _advance_waypoint(self, force_replan: bool = False,
                          on_exhausted=None) -> None:
        """Advance _wp_index and either send the next waypoint or hand off.

        Single shared "what to do after this waypoint" path used by BOTH the success
        path (_finish_waypoint) and the failure path (_try_next_waypoint), so index
        advance and the send-next-vs-stop decision cannot diverge between them.

        Args:
            force_replan: skip the rest of the current plan and go straight to
                          PLANNING even if waypoints remain (SLAM replan cadence).
            on_exhausted: called instead of the default PLANNING transition when the
                          plan runs out (the failure path uses it for streak/recovery).
                          If None, exhaustion just returns to PLANNING.
        """
        self._wp_index += 1
        if not force_replan and self._wp_index < len(self._waypoints):
            self._send_nav_goal(self._waypoints[self._wp_index])
            return
        if force_replan or on_exhausted is None:
            self._state = _State.PLANNING
        else:
            on_exhausted()

    def _try_next_waypoint(self) -> None:
        """Advance to the next waypoint in the CURRENT plan after a nav2 failure.

        The failed waypoint is deliberately left UNVISITED (not added to the
        session's visited_candidates): it may be reachable from a different pose in
        a future plan, so it must stay a live candidate. Only when every waypoint in
        the current plan has failed do we escalate (see _on_plan_exhausted_failed).
        """
        self._advance_waypoint(on_exhausted=self._on_plan_exhausted_failed)

    def _on_plan_exhausted_failed(self) -> None:
        """Every waypoint in the plan was inaccessible from the current pose.

        A re-plan would yield the same unreachable set (the robot's own pose is the
        problem, e.g. nav2 "Start occupied", parked in the inflation band), so
        escalate: after a streak, Spin to regain clearance; if that keeps failing,
        stop instead of Spin/re-plan looping forever.
        """
        self._plan_all_failed_streak += 1
        now = self.get_clock().now().nanoseconds / 1e9
        if self._plan_all_failed_since is None:
            self._plan_all_failed_since = now
        stuck_for = now - self._plan_all_failed_since
        self.get_logger().warning(
            f'All {len(self._waypoints)} waypoints inaccessible from current pose '
            f'(streak {self._plan_all_failed_streak}, stuck for {stuck_for:.0f}s).')

        # Give up ONLY when the robot has failed enough plans AND has been failing for
        # long enough. The time gate is what makes this survivable: a blocked start pose
        # (robot in the inflation band) makes Nav2 reject every goal in ~30 ms, so the
        # streak alone would fire within seconds and kill the run before the robot could
        # be freed. plan_timeout_s = 0 disables the stop entirely (retry forever).
        timed_out = (self._plan_timeout_s
                     and stuck_for >= self._plan_timeout_s)
        if self._plan_all_failed_streak >= _MAX_FAILED_PLAN_STREAK and timed_out:
            # Recovery Spin(s) failed to free the robot: genuinely wedged, cannot progress. Stop instead of Spin/re-plan looping forever.
            self.get_logger().error(
                f'Robot stuck: {self._plan_all_failed_streak} consecutive fully-failed '
                f'plans over {stuck_for:.0f}s (>= plan_timeout_s '
                f'{self._plan_timeout_s:.0f}s). Stopping exploration.')
            self._clear_rviz_markers()
            self._state = _State.COMPLETE
        elif self._plan_all_failed_streak >= 2:
            # Likely stuck (start occupied). Spin to regain clearance, then re-plan. _plan_all_failed_streak resets on the next successful arrival.
            self._send_recovery_spin()
        else:
            self._state = _State.PLANNING

    def _send_recovery_spin(self) -> None:
        """Nav2 Spin to un-stick a pose where every waypoint is unreachable.

        A rotation nudges the robot (and refreshes local costmap clearing) so it can
        leave the inflation band it parked in; on completion we return to PLANNING.
        """
        # Non-blocking probe, see the note in _send_nav_goal: a blocking wait here froze the whole node for 5s. Behaviour on failure is unchanged.
        if not self._spin_client.server_is_ready():
            self.get_logger().warning(
                'Spin server unavailable for recovery; re-planning anyway.')
            self._state = _State.PLANNING
            return
        self._state = _State.NAVIGATING  # not TRAVELING: no waypoint in flight
        goal = Spin.Goal()
        goal.target_yaw     = math.pi          # half turn
        allowance_sec       = int(self._spin_allowance)
        allowance_nsec      = int((self._spin_allowance - allowance_sec) * 1e9)
        goal.time_allowance = Duration(sec=allowance_sec, nanosec=allowance_nsec)
        self.get_logger().info('Recovery Spin: robot stuck, rotating to regain clearance.')
        # Same arming rule as _send_nav_goal, and against the SPIN client: this goal is the one in flight, so it is the server whose disappearance must be detected.
        self._arm_action_watchdog(self._spin_client)
        send_future = self._spin_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_recovery_spin_response)

    def _on_recovery_spin_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Recovery Spin rejected; re-planning.')
            self._state = _State.PLANNING
            return
        goal_handle.get_result_async().add_done_callback(self._on_recovery_spin_result)

    def _on_recovery_spin_result(self, future) -> None:
        # Regardless of Spin outcome, re-plan from the (hopefully) freed pose.
        self._state = _State.PLANNING

    def _do_travel_check(self) -> None:
        """Poll robot position, mark coverage while moving, and check for replan."""
        pos = self._get_robot_pixel_pos()
        if pos is None:
            return  # TF momentarily unavailable, skip this poll, retry next tick

        # See-while-moving: mark the forward camera FOV at the current pose, throttled
        # by distance travelled since the last mark (observe_step_m). This is the
        # "observe continuously along the path" behaviour; legacy mode marks only at
        # the goal spin.
        if self._see_while_moving:
            self._mark_moving_coverage(pos)

        wp = self._waypoints[self._wp_index]
        should_replan = self._session.on_step(
            pos, wp, self._waypoints)
        if should_replan:
            # Do NOT cancel the current goal to replan while the robot is standing in the inflation band: Nav2 cannot plan FROM a start pose it considers occupied, so every waypoint of the new plan is rejected in ~30 ms and the robot is left with no goal at all, worse than letting the in-flight goal continue driving it back into free space. Observed on the hospital run: a mid-path replan fired while inflated, then all 20 waypoints aborted instantly and the run gave up. Keep travelling and re-check next tick.
            if not self._pose_is_navigable(pos):
                self.get_logger().info(
                    'Mid-path replan wanted, but robot is inside the inflation band '
                    '(nav2 cannot plan from here); keeping the current goal.')
                return
            self.get_logger().info('Mid-path replan triggered.')
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                self._goal_handle = None
            self._state = _State.PLANNING

    def _pose_is_navigable(self, pos: tuple[int, int]) -> bool:
        """True when the robot's own cell is navigable (outside the inflation band).

        Nav2 refuses to plan from a start pose it treats as occupied/inflated, so this
        gates actions that depend on a fresh plan being possible.
        """
        col, row = pos
        mask = self._md.navigable_mask
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            return False
        return bool(mask[row, col])

    def _mark_moving_coverage(self, pos: tuple[int, int]) -> None:
        """Mark the forward camera FOV at the current pose while travelling.

        Throttled by distance: only marks once the robot has moved observe_step_m
        since the last mark, so mark cost and RViz traffic stay bounded regardless
        of timer rate. Uses the measured TF heading so covered_mask records what the
        camera actually faced. No-op until the robot has moved far enough.
        """
        if self._last_mark_px is not None:
            step_px = max(1.0, self._observe_step_m / self._md.resolution)
            dc = pos[0] - self._last_mark_px[0]
            dr = pos[1] - self._last_mark_px[1]
            if (dc * dc + dr * dr) < step_px * step_px:
                return  # not far enough since last mark
        heading = self._get_robot_heading_deg()
        if heading is None:
            return  # need a real facing to know what the camera saw
        self._heading = heading
        ratio = update_covered_mask(
            self._md, pos[0], pos[1], self._heading,
            self._cfg['fov_horizontal'], self._max_range_px(),
            self._cfg['num_rays'],
        )
        self._last_mark_px = pos
        self._coverage = ratio
        self._pub_coverage.publish(Float32(data=float(ratio)))
        self._publish_mask_overlays()

    def _refresh_waypoint_headings(self, wp) -> None:
        """Recompute wp.headings from the CURRENT covered_mask, at arrival.

        Same logic as the planner (compute_visibility → compute_headings_for_waypoint) but on the
        up-to-date mask, so headings only target still-uncovered cells. Sets wp.headings in place;
        get_headings then sorts them (and falls back to a full sweep if the list is empty).
        """
        H = self._md.navigable_mask.shape[0]
        col, row = world_to_pixel(wp.x, wp.y, self._md.resolution,
                                  self._md.origin_x, self._md.origin_y, H)
        max_range_px = self._max_range_px()
        cov_cells, _ = compute_visibility(
            (col, row), self._md, max_range_px, self._cfg['num_rays'])
        wp.headings = compute_headings_for_waypoint(
            col, row, cov_cells,
            fov_deg=self._cfg['fov_horizontal'],
            increment_deg=self._increment,
        )

    def _best_uncovered_heading(self, wp) -> float:
        """The single camera heading (deg) that observes the most still-uncovered
        cells from wp, for the see-while-moving goal yaw. compute_headings_for_waypoint
        is greedy, so its first element is the highest-gain heading. Falls back to the
        robot's current heading when the viewpoint has nothing left to reveal (already
        covered / boxed in) so we don't request a needless rotation."""
        self._refresh_waypoint_headings(wp)   # recompute wp.headings from live mask
        if wp.headings:
            return float(wp.headings[0])
        cur = self._get_robot_heading_deg()
        return cur if cur is not None else self._heading

    def _do_rotation(self) -> None:
        """Timer tick during ROTATING, publishes FOV overlay while waiting for Spin result.

        Actual heading progression is driven by _send_spin_goal / _on_spin_result.
        """
        self._publish_fov()

    def _send_spin_goal(self) -> None:
        """Send one Nav2 Spin action goal for the next heading in _rot_headings.

        Computes the shortest signed angular delta from the robot's current heading
        to the target heading and sends it to /spin. The covered_mask is updated
        in _on_spin_result once the physical rotation completes.
        """
        if self._rot_idx >= len(self._rot_headings):
            self._finish_waypoint()
            return

        target_deg = self._rot_headings[self._rot_idx]
        # Shortest signed delta in [-180, 180] degrees, then convert to radians.
        delta_deg = (target_deg - self._heading + 180.0) % 360.0 - 180.0
        target_yaw_rad = math.radians(delta_deg)

        # Non-blocking probe, see the note in _send_nav_goal: a blocking wait here froze the whole node for 5s per heading. Behaviour on failure is unchanged.
        if not self._spin_client.server_is_ready():
            self.get_logger().warning('Spin action server not available, skipping heading.')
            self._rot_idx += 1
            self._send_spin_goal()
            return

        goal = Spin.Goal()
        goal.target_yaw       = target_yaw_rad
        allowance_sec         = int(self._spin_allowance)
        allowance_nsec        = int((self._spin_allowance - allowance_sec) * 1e9)
        goal.time_allowance   = Duration(sec=allowance_sec, nanosec=allowance_nsec)

        self.get_logger().info(
            f'Spin goal: heading {target_deg:.0f}° '
            f'(delta {delta_deg:+.1f}°, {math.degrees(target_yaw_rad):+.1f}° rad→deg check).')
        # Armed per spin goal, not once per rotation sequence: Spin feedback is a different action type so _on_nav_feedback never fires here, meaning the clock is only refreshed at each send. That is safe because spin_time_allowance (10s) is well under the 30s threshold, but only if every individual spin re-arms.
        self._arm_action_watchdog(self._spin_client)
        send_future = self._spin_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_spin_response)

    def _on_spin_response(self, future) -> None:
        """Called when the Spin action server accepts or rejects the goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning(
                f'Spin goal rejected, skipping heading {self._rot_headings[self._rot_idx]:.0f}°.')
            self._rot_idx += 1
            self._send_spin_goal()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_spin_result)

    def _on_spin_result(self, future) -> None:
        """Called when a Spin action finishes.

        The robot has physically rotated to the target heading. Update heading,
        mark covered_mask, publish coverage, then send the next spin goal.
        """
        if self._state != _State.ROTATING:
            return  # state changed (e.g. mid-path replan cancelled rotation)

        result = future.result()
        if result.status not in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED):
            # Cancelled externally, exit without advancing
            return

        # Commit the heading now that the robot has physically turned. Prefer the
        # measured TF yaw so covered_mask records what the camera actually saw, not
        # the planned angle (the robot may settle slightly off target). Targets are
        # absolute, so the next spin delta self-corrects, no drift accumulates.
        # Fall back to the planned heading only if TF is momentarily unavailable.
        measured_heading = self._get_robot_heading_deg()
        self._heading = (
            measured_heading if measured_heading is not None
            else float(self._rot_headings[self._rot_idx])
        )
        self._rot_idx += 1

        wp = self._waypoints[self._wp_index]
        # Recompute pixel position from world coords in case SLAM resized the map
        # since the waypoint was planned (origin shift makes stored col/row stale).
        H = self._md.navigable_mask.shape[0]
        wp_col, wp_row = world_to_pixel(
            wp.x, wp.y,
            self._md.resolution, self._md.origin_x, self._md.origin_y, H,
        )
        max_range_px = self._max_range_px()
        ratio = update_covered_mask(
            self._md,
            wp_col,
            wp_row,
            self._heading,
            self._cfg['fov_horizontal'],
            max_range_px,
            self._cfg['num_rays'],
        )
        self._coverage = ratio
        self._pub_coverage.publish(Float32(data=float(ratio)))
        self._publish_mask_overlays()   # covered_mask has changed, push to RViz immediately
        self._publish_fov()
        self.get_logger().info(
            f'Heading {self._heading:.0f}° observed. Coverage: {ratio:.1%}.')

        if not self._rot_state.update(ratio):
            self.get_logger().debug(
                f'Rotation saturated at heading {self._heading:.0f}° '
                f'(coverage {ratio:.1%}), stopping early.')
            self._finish_waypoint()
            return

        self._send_spin_goal()  # proceed to next heading

    def _arrive_no_spin(self) -> None:
        """See-while-moving arrival: no spin. The robot ended facing the best-
        uncovered heading (set as the goal yaw), so record that single camera look
        at the arrival pose, then finish the waypoint. Coverage along the path was
        already marked in _do_travel_check."""
        wp = self._waypoints[self._wp_index]
        H = self._md.navigable_mask.shape[0]
        wp_col, wp_row = world_to_pixel(
            wp.x, wp.y,
            self._md.resolution, self._md.origin_x, self._md.origin_y, H,
        )
        ratio = update_covered_mask(
            self._md, wp_col, wp_row, self._heading,
            self._cfg['fov_horizontal'], self._max_range_px(),
            self._cfg['num_rays'],
        )
        self._coverage = ratio
        self._pub_coverage.publish(Float32(data=float(ratio)))
        self._publish_mask_overlays()
        self._publish_fov()
        self.get_logger().info(
            f'Arrived facing {self._heading:.0f}°. Coverage: {ratio:.1%}.')
        self._last_mark_px = None   # reset travel-mark throttle for the next leg
        self._finish_waypoint()

    def _finish_waypoint(self) -> None:
        wp = self._waypoints[self._wp_index]
        # Refresh pixel coords from world coords in case SLAM resized the map
        # since the waypoint was planned (origin shift makes stored col/row stale).
        H = self._md.navigable_mask.shape[0]
        wp.col, wp.row = world_to_pixel(
            wp.x, wp.y,
            self._md.resolution, self._md.origin_x, self._md.origin_y, H,
        )
        self._session.on_arrive(wp)
        self.get_logger().info(
            f'Waypoint {self._wp_index} done. Coverage: {self._coverage:.1%}.')

        # No-progress stop guard (live SLAM only): count consecutive arrivals that each
        # add < eps new coverage. K such arrivals means the robot is churning over
        # already-seen ground (typically chasing phantom / unreachable frontiers), so
        # terminate instead of dithering. On a static known map "no progress" is the
        # expected steady state once covered, so the guard is SLAM-gated.
        if self._is_slam and self._no_progress_streak > 0:
            if (self._coverage - self._cov_at_last_arrival) < self._no_progress_eps:
                self._stale_streak += 1
            else:
                self._stale_streak = 0
            self._cov_at_last_arrival = self._coverage
            if self._stale_streak >= self._no_progress_streak:
                self.get_logger().info(
                    f'No-progress stop: {self._stale_streak} consecutive waypoints added '
                    f'< {self._no_progress_eps:.1%} coverage. Coverage {self._coverage:.1%}.')
                self._clear_rviz_markers()
                self._state = _State.COMPLETE
                return

        # Under live SLAM, return to PLANNING every N arrived waypoints so the next waypoints are chosen from the freshly revealed map instead of a plan that is now N arrivals stale. Draining the whole plan is kept for a static map (self._is_slam False) and when the cadence is disabled (<=0).
        replan_now = (
            self._is_slam
            and self._replan_every_n_step > 0
            and self._wp_index + 1 >= self._replan_every_n_step
        )
        self._advance_waypoint(force_replan=replan_now)

    # ── RViz publishers ───────────────────────────────────────────────────

    def _publish_mask_overlays(self) -> None:
        """Publish covered_mask and navigable_mask as OccupancyGrids for RViz overlay."""
        stamp = self.get_clock().now().to_msg()
        self._pub_covered_mask.publish(
            rviz.mask_to_occupancy_grid(self._md.covered_mask, self._md, stamp))
        # Publish the inverse of navigable_mask so free navigable cells show as 0 (transparent in RViz) and non-navigable cells show as 100 (coloured).
        self._pub_nav_mask.publish(
            rviz.mask_to_occupancy_grid(~self._md.navigable_mask, self._md, stamp))

    def _publish_waypoint_markers(self) -> None:
        """Publish the ordered waypoint plan as a MarkerArray.

        Contains three marker types:
          - LINE_STRIP , polyline connecting all waypoints in visit order.
          - SPHERE     , one per waypoint, colour-coded green→red by index.
          - TEXT_VIEW_FACING, rank number floating above each sphere.
        """
        if not self._waypoints:
            return
        stamp = self.get_clock().now().to_msg()
        self._pub_wp_markers.publish(
            rviz.build_waypoint_markers(self._waypoints, self._md, stamp))

    def _publish_current_goal(self, wp) -> None:
        """Publish a distinct cyan sphere at the waypoint being navigated to."""
        stamp = self.get_clock().now().to_msg()
        self._pub_goal_marker.publish(rviz.build_goal_marker(wp, self._md, stamp))

    def _publish_fov(self) -> None:
        """Publish the camera FOV as a LaserScan in the robot body frame; RViz
        places and rotates it via TF."""
        self._pub_fov.publish(rviz.build_fov_laserscan(
            self._cfg['fov_horizontal'],
            self._cfg['max_detection_range'],
            self.get_clock().now().to_msg(),
            self._base_frame,
        ))

    def _clear_rviz_markers(self) -> None:
        """Delete all waypoint and goal markers when exploration finishes."""
        stamp = self.get_clock().now().to_msg()
        wp_array, goal_marker = rviz.build_clear_markers(stamp)
        self._pub_wp_markers.publish(wp_array)
        self._pub_goal_marker.publish(goal_marker)

    # ── Standard publishers ───────────────────────────────────────────────

    def _publish_status(self) -> None:
        # covered/free cell counts let downstream tools plot coverage against
        # the live SLAM map size and tell mask loss apart from denominator
        # growth (ratio alone conflates the two).
        covered_cells = free_cells = 0
        if self._md is not None:
            covered_cells = int(self._md.covered_mask.sum())
            free_cells    = int(self._md.free_mask.sum())
        payload = {
            'state':    self._state.name,
            'coverage': round(self._coverage, 4),
            'waypoint': self._wp_index,
            'total':    len(self._waypoints),
            'covered_cells': covered_cells,
            'free_cells':    free_cells,
            # Non-zero means TF broke and the listener had to be rebuilt (see _check_tf_alive), which is worth seeing in a run's recorded status rather than only in the console log.
            'tf_rebuilds':   self._tf_rebuilds,
            # Same idea for nav2: how many times the action clients had to be rebuilt after repeated silent-death trips, and whether a human was driving.
            'nav2_rebuilds': self._nav2_rebuilds,
            'teleop':        self._teleop_enabled,
        }
        self._pub_status.publish(String(data=json.dumps(payload)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # On Ctrl-C rclpy's signal handler already shuts the context down, so an
        # unconditional shutdown() raises "rcl_shutdown already called". Guard it.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
