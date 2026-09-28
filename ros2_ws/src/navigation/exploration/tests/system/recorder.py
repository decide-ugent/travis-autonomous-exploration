#!/usr/bin/env python3
"""
Exploration run recorder (system-test layer, L3).

A passive ROS2 observer node. It subscribes to everything the exploration loop
produces and writes one self-contained per-run folder so the run can be scored
(evaluate_run.py) and replayed/visualised (visualize_run.py) offline.

Recording starts at t=0: launch this BEFORE unpausing exploration so the
coverage curve, state timeline, and time-to-first-waypoint are all captured.

Time base: all timestamp_s values come from the node clock. Launch with
``use_sim_time:=true`` (Gazebo /clock, Isaac Sim /clock) so 1 s in the CSVs is
1 s of SIMULATED time regardless of the real-time factor. The clock source is
recorded into meta.yaml (time_source: sim|wall) so evaluate_run.py can tell.

Robot pose: taken from TF ``map -> <base_frame>`` (same frame as the map,
waypoints and Nav2 paths), so every position in every CSV is in the map frame.
/odom is still bagged but no longer used as the pose source (it drifts under
SLAM and lives in a different frame).

Motion sampling: a fixed-rate timer (``sample_rate_hz``, default 5 Hz) writes
one motion.csv row per tick with the CURRENT pose and the LAST-KNOWN coverage.
Every row is complete (no blank cells): rows are only written once both a pose
and a coverage value have been seen at least once.

Output layout (one folder per run, not per result type):

    <out_dir>/<scene>_<mode>_run<N>_<timestamp>/
        plans.csv  motion.csv                     (exploration telemetry)
        nav_goals.csv                             (NavigateToPose lifecycle)
        published_waypoints.csv                   (strategy waypoints as published)
        nav2_paths.csv                            (Nav2 planned paths)
        covered_mask_final.npy                    (final real coverage)
        covered_mask_meta.yaml                    (resolution/origin/frame of the mask)
        map_final.npy + map_meta.yaml             (last /map received, for the visualiser)
        maps_completion/                          (per-plan-cycle covered_mask + /map
                                                   snapshots as .npy + meta, for the
                                                   offline coverage-growth video)
        meta.yaml                                 (scene/mode/params/start pose)
        bag/                                      (ros2 bag of the whole session)

On a --mode slam run, shutdown also serializes the live slam_toolbox map into
the nav2 package's map store (navigation/nav2/maps/<scene>_<stamp>/) via nav2's
save_map.sh, giving a .posegraph/.data pair for slam_toolbox localization and a
.pgm/.yaml pair for the exploration planner. The map is NOT written into this
run folder: it is reused across many later runs, so it belongs with nav2. The
run's meta.yaml records saved_map_dir, and the map folder's source.txt names the
run that produced it.

Run configuration comes from the scene's combined config+baseline YAML (not from
ordering-sensitive ROS parameters): the recorder cannot guess the scene or the
mode (the mode depends on whether Nav2 is running SLAM). The same file also holds
the evaluation gates/tolerances/baseline used later by evaluate_run.py, so one
hand-editable file fully describes a scene. The file is split into known_map:
and slam: sections (--mode picks one); the recorder reads only that section's
run: block:

    baselines/baseline_<scene>.yaml
    -------------------------------
    known_map:
      run: {scene, mode, out_dir, record_bag, start_pose, ...}
      gates: {...}          # used by evaluate_run.py
      tolerances: {...}     # used by evaluate_run.py
      baseline: {...}       # written by evaluate_run.py
    slam:
      run: {...}
      ...

The run index is derived automatically by counting existing
<scene>_<mode>_run* folders in out_dir, so reruns never collide.

The lidar scan topic is read the same way the exploration node reads it: as the
ROS parameter ``lidar.scan_topic`` (the launch passes the value from
perception_system_parameters.yaml). The exploration topics are fixed constants
published by ros2_exploration_node.py and live as class attributes.

Run (alongside the exploration stack, sharing the same launch params):

    ros2 run exploration recorder.py \
        --config baselines/baseline_warehouse.yaml --mode slam \
        --ros-args -p use_sim_time:=true -p lidar.scan_topic:=/panoramic/scan
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.node import Node

from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, OccupancyGrid, Path as NavPath
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray


def _default_scene_doc(scene: str) -> dict:
    """Fresh scene YAML (both mode sections, empty baseline) for a new world.

    Mirrors baselines/baseline_lab05.yaml; gates start at the standard values
    and time_to_complete_max_s must be tuned once the baseline exists.
    """
    def section(mode: str, budget_s: int) -> dict:
        return {
            "run": {
                "scene": scene,
                "mode": mode,
                "simulator": "",          # gazebo | isaac | jazzy_docker (fill in)
                # Absolute path inside the mounted src volume so runs persist
                # on the host (cwd-relative ./runs lands in the container-only
                # filesystem and dies with the container).
                "out_dir": "/ros2_ws/src/navigation/exploration/tests/system/runs",
                "record_bag": True,
                "sample_rate_hz": 5.0,
                "start_pose": [0.0, 0.0, 0.0],
            },
            "gates": {
                "final_coverage_min": 0.90,
                "must_terminate": True,
                "time_to_complete_max_s": budget_s,
                "nav_aborted_max": 0,
            },
            "tolerances": {
                "path_length_tolerance_pct": 25,
                "waypoints_tolerance_pct": 25,
                "coverage_tolerance_abs": 0.02,
            },
            "baseline": {"source": "", "run_folder": ""},
        }
    return {"known_map": section("known_map", 1200),
            "slam": section("slam", 1800)}


def _resolve_config_path(path_str: str) -> Path:
    """Resolve --config: as given, else relative to the installed baselines/.

    Returns the path where the file IS or SHOULD BE created (preferring the
    location as typed, so a new scene file lands where the user pointed)."""
    p = Path(path_str)
    if p.is_file():
        return p
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("exploration"))
        candidate = share / "baselines" / p.name
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    return p


def _load_config(path_str: str, mode: str) -> tuple[dict, Path]:
    """Read <mode>.run from the scene's combined config+baseline YAML.

    The file is split into known_map: and slam: sections; `mode` selects one.
    A flat file (no mode sections) is tolerated for convenience.

    If the file does not exist (first run in a new world), it is CREATED from
    the default template — scene name derived from the filename
    (baseline_<scene>.yaml) — so a new simulation world never blocks a run.
    The generated gates/budgets must be reviewed once the baseline is taken.
    """
    path = _resolve_config_path(path_str)
    if not path.is_file():
        stem = path.stem
        scene = stem[len("baseline_"):] if stem.startswith("baseline_") else stem
        if not scene:
            raise ValueError(f"Cannot derive a scene name from '{path.name}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_default_scene_doc(scene),
                                       sort_keys=False))
        print(f"[recorder] No scene config at '{path_str}'. Created a fresh "
              f"one for scene '{scene}' at {path.resolve()} — review its "
              "gates and set run.simulator.")
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    section = doc.get(mode, doc)
    run = section.get("run", section)
    if "scene" not in run:
        raise ValueError(f"{path} [{mode}] must set run.scene")
    run.setdefault("mode", mode)
    run.setdefault("out_dir", str(Path.cwd() / "runs"))
    run.setdefault("record_bag", True)
    run.setdefault("start_pose", [0.0, 0.0, 0.0])
    run.setdefault("sample_rate_hz", 5.0)
    return run, path


def _next_run_index(out_dir: Path, scene: str, mode: str) -> int:
    """One past the highest existing run index for this scene+mode."""
    prefix = f"{scene}_{mode}_run"
    indices = []
    if out_dir.is_dir():
        for d in out_dir.glob(f"{prefix}*"):
            tail = d.name[len(prefix):].split("_", 1)[0]
            if tail.isdigit():
                indices.append(int(tail))
    return (max(indices) + 1) if indices else 1


def _grid_to_meta(msg: OccupancyGrid) -> dict:
    return {
        "resolution": float(msg.info.resolution),
        "origin_x": float(msg.info.origin.position.x),
        "origin_y": float(msg.info.origin.position.y),
        "width": int(msg.info.width),
        "height": int(msg.info.height),
        "frame_id": msg.header.frame_id or "map",
    }


class ExplorationRecorder(Node):
    """Subscribes to the exploration stack and logs a per-run folder."""

    # Fixed topics published by ros2_exploration_node.py (constants there too).
    STATUS_TOPIC = "/exploration/status"
    COVERAGE_TOPIC = "/exploration/coverage"
    COVERED_MASK_TOPIC = "/exploration/covered_mask"
    WAYPOINTS_TOPIC = "/exploration/waypoints"
    CURRENT_GOAL_TOPIC = "/exploration/current_goal"
    MAP_TOPIC = "/map"
    ODOM_TOPIC = "/odometry/filtered"
    NAV2_PLAN_TOPIC = "/plan"
    NAV_STATUS_TOPIC = "/navigate_to_pose/_action/status"
    NAV_FEEDBACK_TOPIC = "/navigate_to_pose/_action/feedback"
    # RealSense D435 (realsense2_camera launched with camera_namespace=camera,
    # camera_name=camera -> /camera/camera/...).
    #
    # COLOUR IS BAGGED COMPRESSED (sensor_msgs/CompressedImage, JPEG q95), depth
    # is bagged RAW (sensor_msgs/Image). This is not a size optimisation - raw
    # colour DOES NOT SURVIVE THE NETWORK to this laptop. Measured against the
    # robot over Ethernet: raw is 2.76 MB/frame (~16.6 MB/s at 6 fps) and arrives
    # at ~3 fps and falling out of 6 published, i.e. ~50% of frames lost, with
    # 924 kernel receive-buffer errors logged on the Jetson. The same stream
    # compressed is 0.25 MB/frame (1.5 MB/s) and arrives at 5.99 fps with jitter
    # down from 0.119 s to 0.0014 s. Bagging image_raw here would silently record
    # a half-empty colour stream.
    #
    # JPEG q95 is set publisher-side in realsense_params.yaml. It is LOSSY: fine
    # for the semantic/object work these bags feed, but if a task ever needs
    # bit-exact colour, switch to .../image_raw/theora or raise quality there.
    #
    # DEPTH STAYS RAW AND LOSSLESS - it is 16-bit millimetres used to place
    # objects in 3D, and JPEG would corrupt the geometry. Depth is also only
    # 0.8 MB/frame, so it is not the bandwidth problem. If depth ever needs
    # shrinking, use .../compressedDepth (PNG, lossless), never /compressed.
    CAMERA_TOPIC = "/camera/camera/color/image_raw/compressed"
    CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
    DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
    DEPTH_INFO_TOPIC = "/camera/camera/depth/camera_info"
    # Keep 1 of every N camera frames in the bag (1 = no decimation).
    # Camera is 6 fps, so 3 -> ~2 Hz of bagged imagery.
    CAMERA_KEEP_EVERY = 3
    DECIMATED_NS = "/rec"

    # TEMPORARY (2026-08-10): set False to bag NO camera topics at all - no
    # colour, no depth, no camera_info, and no topic_tools decimators spawned.
    # Purpose: guarantee the exploration-node topics are captured. In run16 the
    # camera occupied most of a 100 Mb/s link (12.5 MB/s ceiling) while
    # /exploration/covered_mask recorded ZERO messages, so this trades imagery
    # for certainty on the exploration data.
    # SET BACK TO True once covered_mask is confirmed arriving - the semantic
    # work needs the imagery, and compressed colour only costs 1.5 MB/s.
    RECORD_CAMERAS = False

    MAP_FRAME = "map"

    def __init__(self, cfg: dict) -> None:
        super().__init__("exploration_recorder")

        # Read the scan topic the same way the exploration node does:
        # as the lidar.scan_topic ROS parameter (the launch passes the value).
        self.declare_parameter("lidar.scan_topic", "/scan")
        self.declare_parameter("lidar.base_frame", "base_footprint")
        self.scan_topic = self.get_parameter("lidar.scan_topic").value
        self._base_frame = self.get_parameter("lidar.base_frame").value

        scene = cfg["scene"]
        mode = cfg["mode"]
        out_root = Path(cfg["out_dir"])
        run_index = _next_run_index(out_root, scene, mode)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = out_root / f"{scene}_{mode}_run{run_index}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Per-plan-cycle snapshots of the covered_mask and /map, so the offline
        # visualiser can show coverage growing cycle by cycle (raw .npy is cheap
        # to write in the live loop; rendering to png/video is done offline).
        self.completion_dir = self.run_dir / "maps_completion"
        self.completion_dir.mkdir(parents=True, exist_ok=True)

        self._cfg = cfg
        self._record_bag = bool(cfg["record_bag"])

        # Time base: node clock. With use_sim_time this is /clock (sim time),
        # so timestamp_s is simulation seconds, comparable across RT factors.
        self._use_sim_time = bool(
            self.get_parameter("use_sim_time").value)
        self._t0 = None  # set lazily on first non-zero clock sample

        self._latest_coverage = None            # float, last /exploration/coverage
        self._latest_covered_mask = None        # (np.ndarray, meta dict)
        self._latest_map = None                 # (np.ndarray, meta dict) from /map
        self._plan_id = 0                       # increments on each PLANNING entry
        self._last_state = None
        self._open_goals: dict[bytes, dict] = {}  # goal_id -> open goal info
        self._latest_goal_xy = None             # (x, y) from current_goal marker
        # goal_id -> latest action feedback (this Nav2 returns no error code, so
        # feedback is the only "why" signal: recoveries + distance_remaining).
        self._goal_feedback: dict[bytes, dict] = {}

        # TF: robot pose in the map frame (same frame as waypoints/paths).
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Set by _save_slam_map on a successful SLAM save, then appended to meta.yaml at shutdown so the run records which map it produced (source.txt in the map folder is the same link in the other direction).
        self._saved_map_dir: str = ""

        self._drops: list[subprocess.Popen] = []   # topic_tools drop children
        self._files: dict[str, object] = {}
        self._writers: dict[str, csv.DictWriter] = {}

        self._write_meta(scene, mode, run_index, stamp)
        self._open_csvs()
        self._make_subscriptions()
        self._maybe_start_bag()

        # Fixed-rate motion sampler: one complete row per tick.
        rate = float(cfg["sample_rate_hz"])
        self.create_timer(1.0 / rate, self._sample_motion)

        self.get_logger().info(
            f"Recording to {self.run_dir} "
            f"(time_source={'sim' if self._use_sim_time else 'wall'}); "
            f"scan_topic={self.scan_topic!r} base_frame={self._base_frame!r}")

    # CSV plumbing
    def _csv(self, name: str, fieldnames: list[str]) -> csv.DictWriter:
        f = open(self.run_dir / f"{name}.csv", "w", newline="")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        self._files[name] = f
        self._writers[name] = w
        return w

    def _open_csvs(self) -> None:
        self._csv("plans", [
            "plan_id", "timestamp_s", "state", "coverage", "waypoint", "total",
            "robot_x_m", "robot_y_m", "frame_id",
        ])
        self._csv("motion", [
            "timestamp_s", "plan_id", "x_m", "y_m", "yaw_deg",
            "coverage", "covered_cells", "map_free_cells", "coverage_vs_free",
            "frame_id",
        ])
        self._csv("nav_goals", [
            "plan_id", "timestamp_s", "t_accept", "t_result", "nav_time",
            "status", "recoveries", "distance_remaining", "reason",
            "goal_x_m", "goal_y_m", "frame_id",
        ])
        self._csv("published_waypoints", [
            "timestamp_s", "plan_id", "kind", "rank", "x_m", "y_m", "frame_id",
        ])
        self._csv("nav2_paths", [
            "timestamp_s", "plan_id", "seq", "x_m", "y_m", "frame_id",
        ])

    def _row(self, name: str, **row) -> None:
        self._writers[name].writerow(row)
        self._files[name].flush()

    def _now(self) -> float:
        """Seconds since recorder start, on the node clock (sim time when
        use_sim_time is set). t0 latches on the first non-zero clock sample so
        a sim clock that starts publishing late does not skew the origin."""
        t = self.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            if t == 0.0:
                return 0.0
            self._t0 = t
        return round(t - self._t0, 3)

    # Subscriptions
    def _make_subscriptions(self) -> None:
        self.create_subscription(String, self.STATUS_TOPIC, self._on_status, 10)
        self.create_subscription(Float32, self.COVERAGE_TOPIC, self._on_coverage, 10)
        # Publishers use default (VOLATILE) QoS -> subscribe compatibly; the
        # masks/map are republished every cycle so nothing is missed.
        #
        # DEPTH 10, NOT 1, ON THE MASK. In run16 covered_mask recorded ZERO
        # messages while /exploration/coverage - published in the SAME function
        # call (_publish_mask_overlays is invoked right beside
        # _pub_coverage.publish) - recorded 231. So the masks were definitely
        # sent and were lost in transit, not never produced. The mask is ~137 KB
        # (it fragments over the 100 Mb/s link) AND was depth-1: a large message
        # with no queue slack is dropped outright the moment the callback is
        # even slightly late. /map is the same size but survives because
        # slam_toolbox publishes it TRANSIENT_LOCAL (latched/retried); these
        # masks are plain VOLATILE, so the queue is the only buffer they get.
        self.create_subscription(
            OccupancyGrid, self.COVERED_MASK_TOPIC, self._on_covered_mask, 10)
        self.create_subscription(OccupancyGrid, self.MAP_TOPIC, self._on_map, 10)
        self.create_subscription(MarkerArray, self.WAYPOINTS_TOPIC, self._on_waypoints, 10)
        self.create_subscription(Marker, self.CURRENT_GOAL_TOPIC, self._on_current_goal, 10)
        self.create_subscription(NavPath, self.NAV2_PLAN_TOPIC, self._on_nav2_plan, 10)
        self.create_subscription(
            GoalStatusArray, self.NAV_STATUS_TOPIC, self._on_nav_status, 10)
        # Action feedback carries the only "why" signal Nav2 gives here.
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage, self.NAV_FEEDBACK_TOPIC,
            self._on_nav_feedback, 10)

    # Callbacks
    def _on_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        state = data.get("state")
        # A transition into PLANNING marks a new planning cycle.
        if state == "PLANNING" and self._last_state != "PLANNING":
            self._plan_id += 1
            self._snapshot_completion(self._plan_id)
        self._last_state = state

        pose = self._map_pose()
        rx, ry = (round(pose[0], 3), round(pose[1], 3)) if pose else ("", "")
        self._row("plans",
                  plan_id=self._plan_id, timestamp_s=self._now(), state=state,
                  coverage=data.get("coverage"), waypoint=data.get("waypoint"),
                  total=data.get("total"), robot_x_m=rx, robot_y_m=ry,
                  frame_id=self.MAP_FRAME)

    def _on_coverage(self, msg: Float32) -> None:
        # Only cache; motion.csv rows come from the fixed-rate sampler so the
        # curve is dense and every row is complete.
        self._latest_coverage = float(msg.data)

    def _sample_motion(self) -> None:
        """Fixed-rate motion sample: pose (TF map frame) + last-known coverage.

        Skips the tick until BOTH a pose and a coverage value exist, so the
        core columns are always filled. The cell-count columns
        (covered_cells / map_free_cells / coverage_vs_free) are blank until the
        mask and map grids have arrived — downstream parsing treats blank as
        None (_f in evaluate_run)."""
        pose = self._map_pose()
        if pose is None or self._latest_coverage is None:
            return
        x, y, yaw_deg = pose
        covered_cells, free_cells, cov_vs_free = self._coverage_vs_free()
        self._row("motion",
                  timestamp_s=self._now(), plan_id=self._plan_id,
                  x_m=round(x, 3), y_m=round(y, 3), yaw_deg=round(yaw_deg, 1),
                  coverage=round(self._latest_coverage, 4),
                  covered_cells=covered_cells, map_free_cells=free_cells,
                  coverage_vs_free=cov_vs_free,
                  frame_id=self.MAP_FRAME)

    def _coverage_vs_free(self) -> tuple:
        """(covered_cells, map_free_cells, covered/free ratio) from the latest
        cached grids. Raw absolute counts make mask resets directly visible
        (covered_cells must be monotonic); coverage_vs_free is the progression
        against the live SLAM map, so denominator growth (map discovery) can be
        told apart from mask loss. Blank until both grids have arrived."""
        if self._latest_covered_mask is None or self._latest_map is None:
            return "", "", ""
        covered_cells = int((self._latest_covered_mask[0] > 0).sum())
        map_arr = self._latest_map[0]
        free_cells = int(((map_arr >= 0) & (map_arr < 50)).sum())
        ratio = round(covered_cells / free_cells, 4) if free_cells else ""
        return covered_cells, free_cells, ratio

    def _snapshot_completion(self, plan_id: int) -> None:
        """Dump the cached covered_mask + /map for this plan cycle as raw .npy.

        Called on each PLANNING transition. Raw arrays only (no rendering) so the
        cost in the live loop is a couple of np.save calls; the offline visualiser
        turns these into the per-cycle coverage-growth frames. Skips whichever grid
        has not arrived yet (early cycles may have no mask/map)."""
        d = self.completion_dir
        if self._latest_covered_mask is not None:
            arr, meta = self._latest_covered_mask
            np.save(d / f"covered_mask_{plan_id:03d}.npy", arr)
            (d / f"covered_mask_{plan_id:03d}_meta.yaml").write_text(
                yaml.safe_dump(meta, sort_keys=False))
        if self._latest_map is not None:
            arr, meta = self._latest_map
            np.save(d / f"map_{plan_id:03d}.npy", arr)
            (d / f"map_{plan_id:03d}_meta.yaml").write_text(
                yaml.safe_dump(meta, sort_keys=False))

    def _on_covered_mask(self, msg: OccupancyGrid) -> None:
        arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._latest_covered_mask = (arr, _grid_to_meta(msg))

    def _on_map(self, msg: OccupancyGrid) -> None:
        arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._latest_map = (arr, _grid_to_meta(msg))

    def _on_waypoints(self, msg: MarkerArray) -> None:
        # The plan is published as a MarkerArray with THREE overlapping marker
        # types per waypoint: a LINE_STRIP holding every waypoint in visit order,
        # plus a per-waypoint SPHERE (glyph) and a TEXT label offset +0.25 m in y.
        # Recording all of them triple-counts each waypoint and mixes in the
        # label offset (the source of the "two paths" artifact). The LINE_STRIP is
        # the authoritative ordered plan, so record only its points, once each.
        line = next((m for m in msg.markers
                     if m.type == Marker.LINE_STRIP and m.points), None)
        if line is None:
            return
        for rank, p in enumerate(line.points):
            self._row("published_waypoints",
                      timestamp_s=self._now(), plan_id=self._plan_id,
                      kind="waypoint", rank=rank,
                      x_m=round(p.x, 3), y_m=round(p.y, 3),
                      frame_id=line.header.frame_id)

    def _on_current_goal(self, msg: Marker) -> None:
        p = msg.pose.position
        self._latest_goal_xy = (round(p.x, 3), round(p.y, 3))
        self._row("published_waypoints",
                  timestamp_s=self._now(), plan_id=self._plan_id,
                  kind="current_goal", rank=-1,
                  x_m=self._latest_goal_xy[0], y_m=self._latest_goal_xy[1],
                  frame_id=msg.header.frame_id)

    def _on_nav2_plan(self, msg: NavPath) -> None:
        for seq, ps in enumerate(msg.poses):
            self._row("nav2_paths",
                      timestamp_s=self._now(), plan_id=self._plan_id, seq=seq,
                      x_m=round(ps.pose.position.x, 3),
                      y_m=round(ps.pose.position.y, 3),
                      frame_id=msg.header.frame_id)

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        """Track every Nav2 goal individually by its goal_id UUID.

        GoalStatusArray carries the full recent goal history; keying on the
        UUID keeps preempted/parallel goals correctly paired even when a
        mid-path replan cancels one goal and immediately sends another."""
        # GoalStatus: 1=ACCEPTED 2=EXECUTING 4=SUCCEEDED 5=CANCELED 6=ABORTED
        terminal = {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
        for gs in msg.status_list:
            gid = bytes(gs.goal_info.goal_id.uuid)
            code = gs.status
            if code in (1, 2):
                if gid not in self._open_goals:
                    # Goal position = strategy's current_goal marker (the goal
                    # itself), NOT the robot's position at accept time.
                    gx, gy = self._latest_goal_xy or ("", "")
                    self._open_goals[gid] = {
                        "plan_id": self._plan_id, "t_accept": self._now(),
                        "goal_x_m": gx, "goal_y_m": gy,
                    }
            elif code in terminal and gid in self._open_goals:
                g = self._open_goals.pop(gid)
                fb = self._goal_feedback.pop(gid, {})
                t_res = self._now()
                status = terminal[code]
                recoveries = fb.get("recoveries", "")
                dist_rem = fb.get("distance_remaining", "")
                self._row("nav_goals",
                          plan_id=g["plan_id"], timestamp_s=t_res,
                          t_accept=g["t_accept"], t_result=t_res,
                          nav_time=round(t_res - g["t_accept"], 3),
                          status=status,
                          recoveries=recoveries,
                          distance_remaining=dist_rem,
                          reason=self._failure_reason(status, recoveries, dist_rem),
                          goal_x_m=g["goal_x_m"], goal_y_m=g["goal_y_m"],
                          frame_id=self.MAP_FRAME)

    def _on_nav_feedback(self, msg) -> None:
        """Cache the latest action feedback per goal_id.

        This Nav2 (nav2_msgs 1.1.20) returns std_msgs/Empty as the action result —
        no error code. Feedback is the only progress signal, so we keep the last
        recoveries + distance_remaining seen for each goal and attach them to the
        goal's nav_goals row when it terminates (see _failure_reason)."""
        gid = bytes(msg.goal_id.uuid)
        self._goal_feedback[gid] = {
            "recoveries": int(msg.feedback.number_of_recoveries),
            "distance_remaining": round(float(msg.feedback.distance_remaining), 3),
        }

    @staticmethod
    def _failure_reason(status, recoveries, dist_rem) -> str:
        """Best-effort 'why' label for a terminal Nav2 goal.

        Nav2 gives no error code here, so we classify from the feedback signal:
          - SUCCEEDED / CANCELED  -> "" (not a failure; CANCELED is our own replan).
          - ABORTED, no feedback seen -> 'no_valid_path' (died before making progress,
            e.g. planner found no path from the start).
          - ABORTED, recoveries>0 and still far from goal -> 'stuck_no_progress'
            (Nav2 fought with recovery behaviours but could not advance).
          - ABORTED, close to goal -> 'failed_near_goal' (e.g. goal in an obstacle
            inflation, controller could not settle).
          - ABORTED otherwise -> 'aborted_unknown'.
        Thresholds are heuristic; the raw recoveries + distance_remaining columns
        are kept alongside so the label can always be second-guessed."""
        if status != "ABORTED":
            return ""
        if recoveries == "" and dist_rem == "":
            return "no_valid_path"
        try:
            rec = int(recoveries) if recoveries != "" else 0
            dr = float(dist_rem) if dist_rem != "" else 0.0
        except (TypeError, ValueError):
            return "aborted_unknown"
        if rec > 0 and dr > 0.5:
            return "stuck_no_progress"
        if dr <= 0.5:
            return "failed_near_goal"
        return "aborted_unknown"

    # Helpers
    def _map_pose(self):
        """(x_m, y_m, yaw_deg) of the robot in the MAP frame from TF, or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                self.MAP_FRAME, self._base_frame, rclpy.time.Time())
        except Exception:
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return x, y, float(np.degrees(yaw))

    def _write_meta(self, scene, mode, run_index, stamp) -> None:
        start = list(self._cfg["start_pose"])
        meta = {
            "scene": scene,
            "mode": mode,
            "run_index": run_index,
            "timestamp": stamp,
            "time_source": "sim" if self._use_sim_time else "wall",
            "sample_rate_hz": float(self._cfg["sample_rate_hz"]),
            "start_pose": {"x_m": start[0], "y_m": start[1], "yaw_rad": start[2]},
            "scan_topic": self.scan_topic,
            "base_frame": self._base_frame,
            "out_dir": self._cfg["out_dir"],
            "record_bag": self._cfg["record_bag"],
            # Empty when RECORD_CAMERAS is False, so a reader can tell
            # "cameras deliberately excluded" from "imagery was lost".
            "record_cameras": self.RECORD_CAMERAS,
            "camera_topics": ([
                self.CAMERA_TOPIC, self.CAMERA_INFO_TOPIC,
                self.DEPTH_TOPIC, self.DEPTH_INFO_TOPIC,
            ] if self.RECORD_CAMERAS else []),
            # Images land in the bag under DECIMATED_NS at 1/CAMERA_KEEP_EVERY
            # of the camera rate; record both so the bag can be read back.
            "camera_keep_every": self.CAMERA_KEEP_EVERY,
            "camera_bag_prefix": (
                self.DECIMATED_NS if self.CAMERA_KEEP_EVERY > 1 else ""),
            # Message types differ per stream and CANNOT be inferred from the
            # topic names alone - a reader must know colour is CompressedImage
            # (JPEG, lossy) while depth is a raw 16-bit Image (lossless).
            "color_encoding": "jpeg",
            "color_msg_type": "sensor_msgs/msg/CompressedImage",
            "depth_msg_type": "sensor_msgs/msg/Image",
        }
        (self.run_dir / "meta.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False))

    def _append_meta(self, extra: dict) -> None:
        """Merge extra keys into the already-written meta.yaml.

        meta.yaml is written in __init__ so a crashed run still records what it
        was, but a few facts (the saved map path) are only known at shutdown.
        Re-reading and rewriting keeps that single file authoritative rather
        than scattering shutdown-time facts into a second file."""
        path = self.run_dir / "meta.yaml"
        try:
            meta = yaml.safe_load(path.read_text()) or {}
            meta.update(extra)
            path.write_text(yaml.safe_dump(meta, sort_keys=False))
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().warn(f"Could not update meta.yaml: {e}")

    def _start_camera_decimation(self) -> list[str]:
        """Spawn topic_tools drop nodes: keep 1 of every CAMERA_KEEP_EVERY frames.

        The D435 runs at 6 fps (its lowest hardware profile — see
        realsense_params.yaml); decimating by 3 puts the bagged imagery at ~2 Hz,
        which is plenty for offline review and cuts the camera's share of the bag
        to a third. `drop X Y` DROPS X of every Y messages (verified empirically:
        `drop 1 3` keeps 2/3, `drop 2 3` keeps 1/3), so keeping 1 of every N means
        dropping N-1 of every N. This is a message-count decimation, not a rate
        limiter, so it stays correct if the camera fps is changed later.

        camera_info is decimated alongside its image stream so each kept frame
        keeps a matching intrinsics message (it is tiny, but pairing keeps the
        bag self-consistent).

        Returns the topic names to bag. On failure we fall back to the raw
        topics: a bigger bag beats a bag with no imagery in it."""
        if not self.RECORD_CAMERAS:
            # No camera topics bagged and no decimators spawned (see
            # RECORD_CAMERAS). Returning [] keeps every camera topic out of the
            # `ros2 bag record` argument list entirely.
            self.get_logger().warn(
                "RECORD_CAMERAS=False - bagging NO camera topics "
                "(no colour, no depth, no camera_info)")
            return []
        n = self.CAMERA_KEEP_EVERY
        raw = [self.CAMERA_TOPIC, self.CAMERA_INFO_TOPIC,
               self.DEPTH_TOPIC, self.DEPTH_INFO_TOPIC]
        if n <= 1:
            return raw
        out = []
        for topic in raw:
            dst = f"{self.DECIMATED_NS}{topic}"
            try:
                self._drops.append(subprocess.Popen(
                    ["ros2", "run", "topic_tools", "drop",
                     topic, str(n - 1), str(n), dst],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            except FileNotFoundError:
                self.get_logger().warn(
                    "topic_tools not found; bagging camera at full rate")
                for p in self._drops:
                    p.terminate()
                self._drops.clear()
                return raw
            out.append(dst)
        self.get_logger().info(
            f"Camera decimation: keeping 1 of every {n} frames -> {self.DECIMATED_NS}/*")
        return out

    def _maybe_start_bag(self) -> None:
        if not self._record_bag:
            self._bag = None
            return
        topics = [
            self.STATUS_TOPIC, self.COVERAGE_TOPIC, self.COVERED_MASK_TOPIC,
            self.WAYPOINTS_TOPIC, self.CURRENT_GOAL_TOPIC, self.MAP_TOPIC,
            self.scan_topic, self.ODOM_TOPIC, self.NAV2_PLAN_TOPIC,
            self.NAV_STATUS_TOPIC, self.NAV_FEEDBACK_TOPIC, "/tf", "/tf_static",
        ]
        # camera_info rides along so the images stay usable (intrinsics,
        # and depth->color extrinsics come from tf_static above).
        topics += self._start_camera_decimation()
        try:
            self._bag = subprocess.Popen(
                ["ros2", "bag", "record", "-o", str(self.run_dir / "bag"), *topics],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.get_logger().warn("ros2 bag not found; skipping bag recording")
            self._bag = None

    # Shutdown
    def finalize(self) -> None:
        """Write the final covered_mask + map (with their grid meta) and close."""
        if self._latest_covered_mask is not None:
            arr, meta = self._latest_covered_mask
            np.save(self.run_dir / "covered_mask_final.npy", arr)
            (self.run_dir / "covered_mask_meta.yaml").write_text(
                yaml.safe_dump(meta, sort_keys=False))
        if self._latest_map is not None:
            arr, meta = self._latest_map
            np.save(self.run_dir / "map_final.npy", arr)
            (self.run_dir / "map_meta.yaml").write_text(
                yaml.safe_dump(meta, sort_keys=False))
        for f in self._files.values():
            f.close()
        if getattr(self, "_bag", None) is not None:
            self._bag.terminate()
            try:
                self._bag.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._bag.kill()
        # Stop the decimators only AFTER the bag is closed, so the recorder never
        # outlives its own publishers while it is still writing.
        for p in getattr(self, "_drops", []):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self._save_slam_map()
        self._adopt_staged_logs()
        self.get_logger().info(f"Run saved: {self.run_dir}")

    # Where reusable maps are kept. Not under runs/: a map is a nav2 artifact reused across many runs, whereas a run folder is a record of one session. The default is the container path (the same convention as run.out_dir in the scene configs, so maps persist on the host through the bind mount); MAP_STORE in the environment overrides it for a host-side or differently-mounted run.
    MAP_STORE_DEFAULT = "/ros2_ws/src/navigation/nav2/maps"

    def _save_slam_map(self) -> None:
        """Persist the live slam_toolbox map, on SLAM runs only, so the run is reusable.

        Until now a SLAM run's map died with the process: only map_final.npy was
        written, which is a raw numpy dump of the last /map and not a loadable
        nav2 map. This calls nav2's save_map.sh while slam_toolbox is still
        alive, producing BOTH a .posegraph/.data pair (what slam_toolbox
        localization scan-matches against) and a .pgm/.yaml pair (what the
        exploration node plans on). Both are needed, see save_map.sh's header.

        Shelling out to the script keeps ONE implementation of the save: the
        operator running it by hand and the recorder running it automatically go
        through exactly the same code path, so they cannot drift apart.

        Never raises. A failed map save must not lose the run's CSVs and bag,
        which are already written by the time this is called."""
        if self._cfg.get("mode") != "slam":
            return  # known_map runs have no slam_toolbox to serialize
        scene = self._cfg.get("scene", "unknown")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        store = os.environ.get("MAP_STORE", self.MAP_STORE_DEFAULT)
        out_dir = f"{store}/{scene}_{stamp}"
        # get_package_prefix, not get_package_share_directory: CMake installs the script into lib/nav2 (via install(PROGRAMS ...)), which is a sibling of share/nav2 rather than inside it.
        try:
            from ament_index_python.packages import get_package_prefix
            script = Path(get_package_prefix("nav2")) / "lib" / "nav2" / "save_map.sh"
        except Exception as e:
            self.get_logger().warn(f"Could not locate nav2's save_map.sh: {e}")
            return
        if not script.is_file():
            self.get_logger().warn(f"save_map.sh not found at {script}; was nav2 rebuilt?")
            return
        try:
            r = subprocess.run([str(script), out_dir, "map", str(self.run_dir)],
                               capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.get_logger().warn(f"Map save failed to run: {e}")
            return
        if r.returncode == 0:
            self._saved_map_dir = out_dir
            self._append_meta({"saved_map_dir": out_dir})
            self.get_logger().info(f"Saved reusable map: {out_dir}")
        elif r.returncode == 2:
            # slam_toolbox absent. Expected when nav2 was stopped before the recorder, so this is information rather than a warning.
            self.get_logger().info("No slam_toolbox running; no map saved.")
        else:
            self.get_logger().warn(
                f"Map save failed (rc={r.returncode}): {r.stderr.strip()}")

    # Default staging dir shared with run_with_log.sh (keep the two in sync).
    LOG_STAGE_DEFAULT = "/tmp/exprun_stage"

    def _adopt_staged_logs(self) -> None:
        """Copy the run_with_log.sh staging dir into runs/<run>/logs, then clear it.

        nav2/exploration start in their own terminals before this recorder, so
        their logs are captured to a fixed staging dir (RUN_LOG_STAGE, default
        /tmp/exprun_stage — no per-terminal export needed). At shutdown we own the
        run folder, so we pull that dir in and then empty the staging dir so the
        next run starts clean (old rcl node-logs don't leak into it).

        We COPY (not move) into the run folder: nav2/exploration are often still
        running when the recorder stops, so their console tee and rcl .log files
        are still open. The copy captures everything logged up to this moment; we
        then clear the staging dir's *contents* (leaving the dir itself) so the
        still-running processes' open file handles keep writing to fresh files."""
        stage = os.environ.get("RUN_LOG_STAGE", self.LOG_STAGE_DEFAULT)
        stage_dir = Path(stage)
        if not stage_dir.is_dir() or not any(stage_dir.iterdir()):
            return
        dest = self.run_dir / "logs"
        try:
            # dirs_exist_ok so a pre-existing logs/ (e.g. recorder wrapped too) merges.
            shutil.copytree(stage_dir, dest, dirs_exist_ok=True)
            self.get_logger().info(f"Copied staged logs into {dest}")
        except OSError as e:
            self.get_logger().warn(f"Could not copy staged logs from {stage_dir}: {e}")
            return
        # Clear staging so the next run does not inherit these logs.
        for item in stage_dir.iterdir():
            try:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except OSError:
                pass  # a live process may hold a file open; harmless, next run overwrites


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Exploration run recorder")
    parser.add_argument("--config", required=True,
                        help="Path to baselines/baseline_<scene>.yaml")
    parser.add_argument("--mode", required=True, choices=["known_map", "slam"],
                        help="Which section of the config to use")
    # Let ROS args (e.g. -p use_sim_time:=true) pass through untouched.
    known, ros_args = parser.parse_known_args(argv)

    cfg, config_path = _load_config(known.config, known.mode)
    print(f"[recorder] Using scene config: {config_path.resolve()}")
    rclpy.init(args=ros_args)
    node = ExplorationRecorder(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        # On Ctrl-C rclpy's signal handler already shuts the context down, so an
        # unconditional shutdown() raises "rcl_shutdown already called". Guard it.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
