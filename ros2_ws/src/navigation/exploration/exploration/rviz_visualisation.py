#!/usr/bin/env python3
"""
RViz visualisation builders for the exploration node.

Pure, stateless functions that turn exploration data (masks, waypoints, config)
into ROS message / marker types for RViz. No node state, no publishing — the
node owns the publishers and TF clock and calls these to build what it sends.
Keeping them here isolates "how it's drawn" from the node's planning / Nav2 /
state-machine logic.
"""
from __future__ import annotations

import math

import numpy as np

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from exploration.explore_costmap_map import pixel_to_world


# ---------------------------------------------------------------------------
# RViz visualisation constants
# ---------------------------------------------------------------------------

MARKER_LIFETIME    = Duration(sec=0, nanosec=0)  # 0 = persistent until replaced
WP_SPHERE_SCALE    = 0.25   # metres, waypoint sphere diameter
GOAL_SPHERE_SCALE  = 0.40   # metres, current-goal sphere diameter
LINE_WIDTH         = 0.05   # metres, connecting polyline width
FOV_POINT_STEP_DEG = 1.0    # angular resolution of the FOV fan (degrees)


# ---------------------------------------------------------------------------
# Message / marker builders
# ---------------------------------------------------------------------------

def mask_to_occupancy_grid(
    mask: np.ndarray,
    md,
    stamp,
    frame: str = 'map',
    occupied_val: int = 100,
) -> OccupancyGrid:
    """Convert a boolean numpy mask to a nav_msgs/OccupancyGrid.

    True cells → occupied_val, False cells → 0.
    Grid is flipped vertically before packing because ROS OccupancyGrid
    row 0 is the world bottom while numpy/image row 0 is the world top.
    """
    H, W = mask.shape
    grid = OccupancyGrid()
    grid.header = Header(frame_id=frame, stamp=stamp)
    grid.info = MapMetaData(
        map_load_time=stamp,
        resolution=md.resolution,
        width=W,
        height=H,
        origin=Pose(),
    )
    grid.info.origin.position.x = md.origin_x
    grid.info.origin.position.y = md.origin_y
    grid.info.origin.orientation.w = 1.0
    flipped = np.flipud(mask)  # ROS row 0 = world bottom
    grid.data = [int(occupied_val) if v else 0 for v in flipped.flatten()]
    return grid


def build_fov_laserscan(fov_deg: float, range_m: float, stamp, frame: str) -> LaserScan:
    """Build a LaserScan representing the camera FOV, in the robot body frame.

    The scan is a constant-range arc spanning ±fov/2 about the frame's forward
    (x) axis. Publishing it in the robot frame lets RViz place and rotate it via
    TF, so no world position or heading maths is needed here — the wedge always
    sits at the robot and points along its current heading.
    """
    half_rad  = math.radians(fov_deg / 2.0)
    n_arc     = max(2, int(fov_deg / FOV_POINT_STEP_DEG) + 1)
    increment = (2.0 * half_rad) / (n_arc - 1)

    scan = LaserScan()
    scan.header          = Header(frame_id=frame, stamp=stamp)
    scan.angle_min       = -half_rad
    scan.angle_max       = half_rad
    scan.angle_increment = increment
    scan.range_min       = 0.0
    scan.range_max       = range_m
    scan.ranges          = [range_m] * n_arc
    return scan


def lerp_color(t: float) -> ColorRGBA:
    """Green (t=0) → yellow (t=0.5) → red (t=1) gradient for visit-order colouring."""
    return ColorRGBA(
        r=float(min(1.0, 2.0 * t)),
        g=float(min(1.0, 2.0 * (1.0 - t))),
        b=0.0,
        a=1.0,
    )


def build_waypoint_markers(waypoints, md, stamp) -> MarkerArray:
    """Build the ordered waypoint plan as a MarkerArray.

    Three marker types:
      - LINE_STRIP        , polyline connecting all waypoints in visit order.
      - SPHERE            , one per waypoint, colour-coded green→red by index.
      - TEXT_VIEW_FACING  , rank number floating above each sphere.
    """
    n     = len(waypoints)
    array = MarkerArray()
    H     = md.navigable_mask.shape[0]

    line          = Marker()
    line.header   = Header(frame_id='map', stamp=stamp)
    line.ns       = 'exploration_plan'
    line.id       = 0
    line.type     = Marker.LINE_STRIP
    line.action   = Marker.ADD
    line.scale.x  = LINE_WIDTH
    line.color    = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
    line.lifetime = MARKER_LIFETIME

    for i, wp in enumerate(waypoints):
        px, py = pixel_to_world(wp.col, wp.row, md.resolution,
                                md.origin_x, md.origin_y, H)
        wx = float(px)
        wy = float(py)
        t  = i / max(n - 1, 1)

        line.points.append(Point(x=wx, y=wy, z=0.05))

        sphere           = Marker()
        sphere.header    = line.header
        sphere.ns        = 'exploration_plan'
        sphere.id        = i + 1
        sphere.type      = Marker.SPHERE
        sphere.action    = Marker.ADD
        sphere.pose.position = Point(x=wx, y=wy, z=0.05)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x   = sphere.scale.y = sphere.scale.z = WP_SPHERE_SCALE
        sphere.color     = lerp_color(t)
        sphere.lifetime  = MARKER_LIFETIME
        array.markers.append(sphere)

        label          = Marker()
        label.header   = line.header
        label.ns       = 'exploration_plan_labels'
        label.id       = i
        label.type     = Marker.TEXT_VIEW_FACING
        label.action   = Marker.ADD
        label.pose.position = Point(x=wx, y=wy + 0.25, z=0.2)
        label.pose.orientation.w = 1.0
        label.scale.z  = 0.20
        label.color    = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text     = str(i)
        label.lifetime = MARKER_LIFETIME
        array.markers.append(label)

    array.markers.insert(0, line)
    return array


def build_goal_marker(wp, md, stamp) -> Marker:
    """Build the distinct cyan sphere at the waypoint being navigated to."""
    H     = md.navigable_mask.shape[0]
    px, py = pixel_to_world(wp.col, wp.row, md.resolution,
                            md.origin_x, md.origin_y, H)

    marker          = Marker()
    marker.header   = Header(frame_id='map', stamp=stamp)
    marker.ns       = 'exploration_goal'
    marker.id       = 0
    marker.type     = Marker.SPHERE
    marker.action   = Marker.ADD
    marker.pose.position = Point(x=float(px), y=float(py), z=0.1)
    marker.pose.orientation.w = 1.0
    marker.scale.x  = marker.scale.y = marker.scale.z = GOAL_SPHERE_SCALE
    marker.color    = ColorRGBA(r=0.0, g=0.6, b=1.0, a=1.0)  # cyan
    marker.lifetime = MARKER_LIFETIME
    return marker


def build_clear_markers(stamp) -> tuple[MarkerArray, Marker]:
    """Build the delete-all markers used when exploration finishes.

    Returns (waypoint_array, goal_marker): a MarkerArray that DELETEALLs the plan
    namespaces, and a single Marker that DELETEs the current-goal sphere.
    """
    array = MarkerArray()
    for ns in ('exploration_plan', 'exploration_plan_labels'):
        m          = Marker()
        m.header   = Header(frame_id='map', stamp=stamp)
        m.ns       = ns
        m.id       = 0
        m.action   = Marker.DELETEALL
        array.markers.append(m)

    gone          = Marker()
    gone.header   = Header(frame_id='map', stamp=stamp)
    gone.ns       = 'exploration_goal'
    gone.id       = 0
    gone.action   = Marker.DELETE
    return array, gone
