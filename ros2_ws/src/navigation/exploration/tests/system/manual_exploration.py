#!/usr/bin/env python3
"""
Manual exploration: the human-driven reference run (system-test baseline).

This node runs the SAME coverage measurement pipeline as the autonomous
exploration node, but WITH THE EXPLORATION STRATEGY OFF. There is no waypoint
planner and no automatic viewpoint selection. A human decides where the robot
goes; how they drive it is up to them and does not matter to this node:

    - teleop (teleop_twist / joystick publishing /cmd_vel), or
    - manually issued Nav2 goals (clicking "2D Goal Pose" in RViz, or sending
      NavigateToPose by hand).

Either way this node only observes the robot pose from TF and grows the
covered_mask exactly as a real run would. Because Nav2 may be used to drive,
recorder.py still captures those nav_goals/paths, so the reference run is a
faithful apples-to-apples comparison to an autonomous run.

A baseline is taken per (scene, mode): run this once in known_map mode and once
in slam mode for each environment. Recorded alongside recorder.py, this produces
the same per-run CSV folder as a model run, so evaluate_run.py --save-baseline
can summarise it into the scene file's <mode>.baseline block.

It is a drop-in stand-in for ros2_exploration_node.py, so it takes the SAME ROS
parameters (camera.*, lidar.*, nav2.costmap_inflation_radius, exploration.map_*)
the autonomous node takes, supplied by the same launch. Nothing is re-read from
config files here; the launch is the single source of truth, exactly as for the
autonomous node.

Publishes the topics the recorder listens to:
    /exploration/coverage      (Float32)        live coverage ratio
    /exploration/covered_mask  (OccupancyGrid)  cumulative observed cells
    /exploration/status        (String, JSON)   {state: "MANUAL", mode, coverage}

plus the autonomous node's RViz progress visuals (same topics, so the
exploration RViz config displays a manual run identically):
    /exploration/robot_fov       (LaserScan)      live FOV wedge at the robot
    /exploration/navigable_mask  (OccupancyGrid)  non-navigable cells overlay

Coverage progress is also logged to the terminal at every whole percent.
"""
from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

import tf2_ros

from exploration import rviz_visualisation as rviz
from exploration.explore_costmap_map import (
    build_map_data,
    load_map,
    reproject_covered_mask,
    update_covered_mask,
    world_to_pixel,
)


class ManualExploration(Node):
    """Grows covered_mask from the robot pose; no planner. Drive via teleop OR Nav2."""

    def __init__(self) -> None:
        super().__init__("manual_exploration")

        # Same ROS parameters the autonomous node takes (supplied by the launch).
        gp = self.get_parameter
        self.declare_parameter("camera.max_detection_range", 6.0)
        self.declare_parameter("camera.fov_horizontal", 87.0)
        self.declare_parameter("lidar.num_rays", 360)
        self.declare_parameter("lidar.base_frame", "base_footprint")
        self.declare_parameter("nav2.costmap_inflation_radius", 0.4)
        self.declare_parameter("exploration.map_file_path", "")
        self.declare_parameter("exploration.map_yaml_path", "")
        # Heartbeat period for the terminal coverage line (seconds). 0 disables
        # the heartbeat and restores "log only on a new whole-percent high".
        self.declare_parameter("exploration.coverage_log_period_s", 15.0)
        # Tag only (known_map | slam); does not change the measurement.
        self.declare_parameter("mode", "known_map")
        self.declare_parameter("update_rate_hz", 5.0)

        self._mode = gp("mode").value
        self._max_range_m = float(gp("camera.max_detection_range").value)
        self._fov = float(gp("camera.fov_horizontal").value)
        self._num_rays = int(gp("lidar.num_rays").value)
        self._base_frame = gp("lidar.base_frame").value
        self._inflation_m = float(gp("nav2.costmap_inflation_radius").value)

        self._pub_coverage = self.create_publisher(Float32, "/exploration/coverage", 10)
        self._pub_covered_mask = self.create_publisher(
            OccupancyGrid, "/exploration/covered_mask", 1)
        self._pub_status = self.create_publisher(String, "/exploration/status", 10)
        # Same progress visuals as the autonomous node (same topics, so the
        # exploration RViz config shows them unchanged): the navigable-mask
        # overlay and the live FOV wedge marking what the sweep is crediting.
        self._pub_nav_mask = self.create_publisher(
            OccupancyGrid, "/exploration/navigable_mask", 1)
        self._pub_fov = self.create_publisher(
            LaserScan, "/exploration/robot_fov", 1)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._md = None
        self._last_grid_info = None  # (res, origin_x, origin_y, frame)
        self._last_logged_pct = -1   # highest whole-percent coverage logged
        self._last_log_t = None      # wall/sim time of the last coverage line
        # Heartbeat: log coverage at least this often even when the ratio is not
        # setting a new high. Under SLAM the ratio is covered/navigable and the
        # DENOMINATOR grows as new area is discovered, so the ratio routinely
        # falls back; a "new high only" rule then goes silent for long stretches
        # (measured: 86% of a 3138 s hospital run, incl. one 22-minute gap) —
        # exactly when the human driving needs to know where they stand.
        self._log_period_s = float(gp("exploration.coverage_log_period_s").value)

        # Map source: static file (known-map) or live /map (SLAM), as the node.
        map_path = gp("exploration.map_file_path").value
        if map_path:
            self._md = load_map(
                map_path, gp("exploration.map_yaml_path").value, self._inflation_m)
            self.get_logger().info(f"Loaded static map from {map_path}")
        else:
            self.create_subscription(OccupancyGrid, "/map", self._on_map, 1)
            self.get_logger().info("Waiting for /map topic...")

        rate = float(gp("update_rate_hz").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"Manual exploration active (strategy OFF, mode={self._mode}). "
            f"Drive via teleop or Nav2 goals.")

    # Map
    def _on_map(self, msg: OccupancyGrid) -> None:
        new_H, new_W = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.float64).reshape(new_H, new_W)
        data = np.flipud(data)                          # ROS row 0 = world bottom
        p_occ = np.where(data < 0, 0.5, data / 100.0)  # -1 (unknown) -> 0.5

        # Carry the accumulated coverage across SLAM map growth by world
        # position — same reprojection as the autonomous node (shared helper),
        # otherwise every map resize resets the mask.
        covered = None
        if self._md is not None:
            covered = reproject_covered_mask(
                self._md.covered_mask,
                self._md.resolution, self._md.origin_x, self._md.origin_y,
                (new_H, new_W),
                msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y,
            )
            n_old = int(self._md.covered_mask.sum())
            n_new = int(covered.sum())
            if n_new < 0.95 * n_old:
                self.get_logger().error(
                    f"covered_mask lost cells on map change: {n_old} -> {n_new}")
        self._md = build_map_data(
            p_occ,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            inflation_radius_m=self._inflation_m,
            covered_mask=covered,
        )
        self._last_grid_info = (
            msg.info.resolution, msg.info.origin.position.x,
            msg.info.origin.position.y, msg.header.frame_id or "map",
        )

    # Per-tick observation
    def _tick(self) -> None:
        if self._md is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        x, y, heading_deg = pose

        H = self._md.navigable_mask.shape[0]
        col, row = world_to_pixel(
            x, y, self._md.resolution, self._md.origin_x, self._md.origin_y, H)
        if not (0 <= col < self._md.navigable_mask.shape[1] and 0 <= row < H):
            return

        max_range_px = int(self._max_range_m / self._md.resolution)
        ratio = update_covered_mask(
            self._md, col, row, heading_deg, self._fov, max_range_px, self._num_rays)

        self._pub_coverage.publish(Float32(data=float(ratio)))
        self._publish_status(ratio)
        self._publish_covered_mask()
        self._publish_progress_visuals(ratio)

    def _robot_pose(self):
        """(x_m, y_m, yaw_deg) from TF, or None on failure."""
        try:
            t = self._tf_buffer.lookup_transform("map", self._base_frame, Time())
        except Exception:
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return x, y, math.degrees(yaw)

    # Publishing
    def _publish_progress_visuals(self, ratio: float) -> None:
        """RViz progress: FOV wedge + navigable-mask overlay (same topics and
        encodings as the autonomous node), and a coverage log every whole
        percent gained so the terminal shows the sweep progressing."""
        stamp = self.get_clock().now().to_msg()
        self._pub_fov.publish(rviz.build_fov_laserscan(
            self._fov, self._max_range_m, stamp, self._base_frame))
        # Inverse of navigable_mask: free navigable cells -> 0 (transparent in
        # RViz), non-navigable -> 100 (coloured), as the autonomous node does.
        self._pub_nav_mask.publish(rviz.mask_to_occupancy_grid(
            ~self._md.navigable_mask, self._md, stamp))
        # Log on EITHER a new whole-percent high OR the heartbeat period, so the
        # driver always has a current number even while the SLAM denominator is
        # growing and the ratio is below its previous peak.
        now_s = self.get_clock().now().nanoseconds / 1e9
        if self._last_log_t is None:
            self._last_log_t = now_s
        pct = int(ratio * 100)
        new_high = pct > self._last_logged_pct
        due = (self._log_period_s > 0.0
               and (now_s - self._last_log_t) >= self._log_period_s)
        if new_high or due:
            self._last_logged_pct = max(self._last_logged_pct, pct)
            self._last_log_t = now_s
            # Absolute observed area alongside the ratio: under SLAM the ratio
            # can fall when new free space is discovered, while the area only
            # grows — so the area is the honest progress signal for the driver.
            area_m2 = float(self._md.covered_mask.sum()) * self._md.resolution ** 2
            self.get_logger().info(
                f"Coverage: {ratio:.1%}  ({area_m2:.1f} m² observed)")

    def _publish_status(self, ratio: float) -> None:
        self._pub_status.publish(String(data=json.dumps({
            "state": "MANUAL",
            "mode": self._mode,
            "coverage": round(ratio, 4),
            "waypoint": 0,
            "total": 0,
        })))

    def _publish_covered_mask(self) -> None:
        if self._last_grid_info is None and self._md is not None:
            self._last_grid_info = (
                self._md.resolution, self._md.origin_x, self._md.origin_y, "map")
        if self._last_grid_info is None:
            return
        res, ox, oy, frame = self._last_grid_info
        covered = self._md.covered_mask
        H, W = covered.shape
        grid = OccupancyGrid()
        grid.header.frame_id = frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = res
        grid.info.width = W
        grid.info.height = H
        grid.info.origin.position.x = ox
        grid.info.origin.position.y = oy
        # Flip back to ROS row order (row 0 = world bottom) and 0/100 encoding.
        flat = np.flipud(covered).astype(np.int8) * 100
        grid.data = flat.flatten().tolist()
        self._pub_covered_mask.publish(grid)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManualExploration()
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


if __name__ == "__main__":
    main()
