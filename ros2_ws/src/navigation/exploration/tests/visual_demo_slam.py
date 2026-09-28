"""
Demo 2 — SLAM exploration with a partially unknown map.

A user-configurable fraction of the right side of the lab_ghent map starts
as unknown (grey). As the robot navigates toward and through the hidden zone,
the map is revealed in real time — mimicking how a SLAM system discovers new
areas. The exploration planner automatically generates frontier waypoints
toward unknown regions and switches to pure visual coverage once the map is
fully revealed.

Run:
    python tests/navigation/visual_demo_slam.py

Controls:
    Close the window to end the demo early.

Configurable constants below.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── sys.path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent.parent  # /ros2_ws/src/
_TESTS_DIR = Path(__file__).parent       # tests/
_EXPLORATION_PKG = _ROOT / "navigation" / "exploration"
sys.path.insert(0, str(_TESTS_DIR))  # for conftest constants
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "navigation"))
sys.path.insert(0, str(_ROOT / "travis_brain"))
sys.path.insert(0, str(_EXPLORATION_PKG))

from navigation.exploration.exploration.explore_costmap_map import (
    compute_headings_for_waypoint,
    compute_visibility,
    load_map,
    pixel_to_world,
    update_covered_mask,
)
from navigation.exploration.visualisation.semantic_exploration_visualiser import SemanticExplorationVisualiser
from demo_robot import (
    find_path, reveal_cells, build_demo_config,
    FOV_HORIZONTAL, OBSERVATION_INCREMENT, NUM_RAYS, INFLATION_M,
)
from navigation.exploration.exploration.rotation_strategy import get_headings, RotationState
from navigation.exploration.exploration.execution_strategy import ExplorationSession
from conftest import REFERENCE_MAP_PGM, REFERENCE_MAP_YAML

# ── Configurable ──────────────────────────────────────────────────────────
UNKNOWN_FRACTION = 0.40   # fraction of map width (right side) that starts unknown
PAUSE_TRAVEL     = 0.01   # seconds per pixel step along path
PAUSE_ROTATE     = 0.04   # seconds per heading rotation at waypoint
PLOT_SAVING_TIME = 15.0   # seconds between auto-saves of the plot
# ─────────────────────────────────────────────────────────────────────────

MAP_PGM  = REFERENCE_MAP_PGM
MAP_YAML = REFERENCE_MAP_YAML
TEST_DIR_SAVE = _TESTS_DIR / "visual_demo_slam"
TEST_DIR_SAVE.mkdir(exist_ok=True, parents=True)


def _get_unique_log_dir(base_name: str, directory: Path) -> Path:
    target = directory / base_name
    if not target.exists():
        return target
    counter = 1
    while True:
        target = directory / f"{base_name}_{counter}"
        if not target.exists():
            return target
        counter += 1


def _travel_heading(prev: tuple[int, int], cur: tuple[int, int]) -> float:
    dc = cur[0] - prev[0]
    dr = -(cur[1] - prev[1])
    return math.degrees(math.atan2(dr, dc)) % 360.0



def _save_log(
    plans: list[dict],
    waypoints: list[dict],
    motion: list[dict],
    events: list[dict],
    candidates: list[dict],
    log_dir: Path,
) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    def _write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = log_dir / f"{name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    _write("plans", plans)
    _write("waypoints", waypoints)
    _write("robot_motion", motion)
    _write("events", events)
    _write("candidates", candidates)
    print(f"Log saved → {log_dir}/")


def main() -> None:
    # ── Load map ──────────────────────────────────────────────────────────
    md = load_map(MAP_PGM, MAP_YAML, INFLATION_M)
    H, W = md.pgm_array.shape
    cfg = build_demo_config()
    max_range_px = max(1, int(cfg["max_detection_range"] / md.resolution))
    _inflation_px = INFLATION_M / md.resolution

    # ── Save original masks before masking ────────────────────────────────
    original_free = md.free_mask.copy()
    original_occ  = md.occupied_mask.copy()

    # ── Mask right UNKNOWN_FRACTION of map as unknown ─────────────────────
    col_threshold = int(W * (1.0 - UNKNOWN_FRACTION))
    unknown_cols = slice(col_threshold, W)
    n_initial_unknown = int(np.sum(md.free_mask[:, unknown_cols]))

    md.unknown_mask[:, unknown_cols] |= md.free_mask[:, unknown_cols]
    md.free_mask[:, unknown_cols]     = False
    md.occupied_mask[:, unknown_cols] = False
    md.navigable_mask[:, unknown_cols] = False

    # ── Visualiser (no objects in this demo) ──────────────────────────────
    vis = SemanticExplorationVisualiser(md, objects=[], show_3d=False)
    vis.show_nonblocking()

    # ── Exploration session + robot start ─────────────────────────────────
    session = ExplorationSession(md, cfg)
    cx, cy = col_threshold // 2, H // 2
    try:
        robot_col, robot_row = session.nearest_start(cx, cy)
    except RuntimeError as e:
        print(e)
        return
    robot_x, robot_y = pixel_to_world(robot_col, robot_row, md.resolution,
                                      md.origin_x, md.origin_y, H)

    trajectory: list[tuple[int, int]] = [(robot_col, robot_row)]
    heading = 0.0
    ratio = 0.0

    # ── Log setup ─────────────────────────────────────────────────────────
    start_time = time.time()
    log_dir = _get_unique_log_dir("slam_demo_log", TEST_DIR_SAVE)
    log_plans: list[dict] = []
    log_waypoints: list[dict] = []
    log_motion: list[dict] = []
    log_events: list[dict] = []
    log_candidates: list[dict] = []
    plan_id = 0
    step_id = 0
    total_distance_m = 0.0
    _last_plot_save = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        waypoints, ratio, no_frontiers, candidate_records = session.plan_waypoints_raw(robot_x, robot_y)
        if (no_frontiers and ratio >= cfg["exploration_completion_threshold"]) or not waypoints:
            break

        plan_id += 1
        t = round(time.time() - start_time, 2)
        n_unknown_remaining = int(np.sum(md.unknown_mask))

        log_plans.append({
            "plan_id":            plan_id,
            "robot_col":          robot_col,
            "robot_row":          robot_row,
            "robot_x_m":          round(robot_x, 3),
            "robot_y_m":          round(robot_y, 3),
            "coverage_pct":       round(ratio * 100, 2),
            "n_waypoints":        len(waypoints),
            "unknown_cells_left": n_unknown_remaining,
            "timestamp_s":        t,
        })

        for rank, wp in enumerate(waypoints):
            log_waypoints.append({
                "plan_id":        plan_id,
                "rank":           rank,
                "col":            wp.col,
                "row":            wp.row,
                "x_m":            round(wp.x, 3),
                "y_m":            round(wp.y, 3),
                "headings":       ",".join(str(int(h)) for h in wp.headings),
                "frontier_gain":  wp.frontier_gain,
                "coverage_gain":  wp.coverage_gain,
                "geodesic_dist_px": round(wp.geodesic_dist_px, 1) if wp.geodesic_dist_px is not None else "",
                "score":          round(wp.score, 4) if wp.score is not None else "",
                "timestamp_s":    t,
            })

        for rec in candidate_records:
            x_m, y_m = pixel_to_world(rec["col"], rec["row"], md.resolution,
                                      md.origin_x, md.origin_y, H)
            log_candidates.append({
                "plan_id":        plan_id,
                "col":            rec["col"],
                "row":            rec["row"],
                "x_m":            round(x_m, 3),
                "y_m":            round(y_m, 3),
                "frontier_gain":  rec["frontier_gain"],
                "coverage_gain":  rec["coverage_gain"],
                "geodesic_dist_px": round(rec["geodesic_dist_px"], 1) if rec["geodesic_dist_px"] is not None else "",
                "score":          round(rec["score"], 4),
                "selected":       rec["selection_rank"] is not None,
                "selection_rank": rec["selection_rank"] if rec["selection_rank"] is not None else "",
                "timestamp_s":    t,
            })

        # Execute the WHOLE plan in order before replanning, exactly like the node
        # (ExplorationNode._finish_waypoint drains waypoints, replans only when exhausted).
        # No waypoints[0] skip: the node always goes to waypoints[0].
        for wp in waypoints:
            path = find_path(md.navigable_mask, (robot_col, robot_row), (wp.col, wp.row))

            euclid_px = math.sqrt((wp.col - robot_col) ** 2 + (wp.row - robot_row) ** 2)
            bfs_px = len(path) - 1
            if session.on_unreachable(wp, path):
                log_events.append({
                    "plan_id":     plan_id,
                    "type":        "unreachable_waypoint",
                    "wp_col":      wp.col,
                    "wp_row":      wp.row,
                    "wp_x_m":      round(wp.x, 3),
                    "wp_y_m":      round(wp.y, 3),
                    "detail":      "find_path returned only start; no navigable path exists",
                    "timestamp_s": round(time.time() - start_time, 2),
                })
                continue
            elif euclid_px > 0 and bfs_px / euclid_px > 1.5:
                log_events.append({
                    "plan_id":     plan_id,
                    "type":        "wall_detour",
                    "wp_col":      wp.col,
                    "wp_row":      wp.row,
                    "wp_x_m":      round(wp.x, 3),
                    "wp_y_m":      round(wp.y, 3),
                    "detail":      f"BFS {bfs_px} px vs Euclidean {euclid_px:.1f} px (×{bfs_px/euclid_px:.2f})",
                    "timestamp_s": round(time.time() - start_time, 2),
                })

            # Travel along path. reveal_cells stays (simulates SLAM revealing the map as the robot
            # drives — real SLAM runs continuously), but coverage (update_covered_mask) is NOT
            # marked in transit: the node marks coverage only at rotation (arrival-only).
            _mid_replan = False
            prev = path[0]
            for step in path[1:]:
                step_id += 1
                heading = _travel_heading(prev, step)
                step_dist_m = math.hypot(step[0] - prev[0], step[1] - prev[1]) * md.resolution
                total_distance_m += step_dist_m
                reveal_cells(
                    md.free_mask, md.occupied_mask, md.unknown_mask, md.navigable_mask,
                    original_free, original_occ,
                    step[0], step[1], max_range_px, NUM_RAYS,
                    inflation_px=_inflation_px,
                )
                sx, sy = pixel_to_world(step[0], step[1], md.resolution,
                                        md.origin_x, md.origin_y, H)
                log_motion.append({
                    "step_id":              step_id,
                    "plan_id":              plan_id,
                    "type":                 "travel",
                    "col":                  step[0],
                    "row":                  step[1],
                    "x_m":                  round(sx, 3),
                    "y_m":                  round(sy, 3),
                    "heading_deg":          round(heading, 1),
                    "coverage_pct":         round(ratio * 100, 2),
                    "travelled_distance_m": round(total_distance_m, 3),
                    "unknown_cells_left":   int(np.sum(md.unknown_mask)),
                    "timestamp_s":          round(time.time() - start_time, 2),
                })
                trajectory.append(step)
                vis.update(waypoints, step[0], step[1], heading, trajectory, ratio)
                plt.pause(PAUSE_TRAVEL)
                if session.on_step(step, wp, waypoints):
                    robot_col, robot_row = step
                    robot_x, robot_y = pixel_to_world(step[0], step[1], md.resolution,
                                                      md.origin_x, md.origin_y, H)
                    log_events.append({
                        "plan_id":     plan_id,
                        "type":        "mid_path_replan",
                        "wp_col":      step[0],
                        "wp_row":      step[1],
                        "wp_x_m":      round(robot_x, 3),
                        "wp_y_m":      round(robot_y, 3),
                        "detail":      f"cheaper waypoint found; aborting path to ({wp.col},{wp.row})",
                        "timestamp_s": round(time.time() - start_time, 2),
                    })
                    _mid_replan = True
                    break
                if not plt.get_fignums():
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"slam_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    return
                if time.time() - _last_plot_save >= PLOT_SAVING_TIME:
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"slam_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    _last_plot_save = time.time()
                prev = step

            # Mid-path replan aborts the whole plan (like the node cancelling the goal → PLANNING).
            if _mid_replan:
                break

            # Rotate at waypoint. reveal_cells before update_covered_mask so newly revealed cells
            # count immediately. Early stop disabled (stop_after=999): rotating reveals new map area.
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y
            _rot_state = RotationState(stop_after=999)
            # Recompute headings from the current covered_mask at arrival (matches the node).
            _cov_now, _ = compute_visibility((wp.col, wp.row), md, max_range_px, NUM_RAYS)
            wp.headings = compute_headings_for_waypoint(
                wp.col, wp.row, _cov_now, fov_deg=FOV_HORIZONTAL, increment_deg=OBSERVATION_INCREMENT)
            for h in get_headings(wp, OBSERVATION_INCREMENT, heading):
                step_id += 1
                heading = float(h)
                reveal_cells(
                    md.free_mask, md.occupied_mask, md.unknown_mask, md.navigable_mask,
                    original_free, original_occ,
                    wp.col, wp.row, max_range_px, NUM_RAYS,
                    inflation_px=_inflation_px,
                )
                ratio = update_covered_mask(md, wp.col, wp.row, heading, FOV_HORIZONTAL,
                                            max_range_px, NUM_RAYS)
                log_motion.append({
                    "step_id":              step_id,
                    "plan_id":              plan_id,
                    "type":                 "rotate",
                    "col":                  wp.col,
                    "row":                  wp.row,
                    "x_m":                  round(wp.x, 3),
                    "y_m":                  round(wp.y, 3),
                    "heading_deg":          round(heading, 1),
                    "coverage_pct":         round(ratio * 100, 2),
                    "travelled_distance_m": round(total_distance_m, 3),
                    "unknown_cells_left":   int(np.sum(md.unknown_mask)),
                    "timestamp_s":          round(time.time() - start_time, 2),
                })
                vis.update(waypoints, wp.col, wp.row, heading, trajectory, ratio)
                plt.pause(PAUSE_ROTATE)
                if not plt.get_fignums():
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"slam_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    return
                if time.time() - _last_plot_save >= PLOT_SAVING_TIME:
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"slam_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    _last_plot_save = time.time()
                if not _rot_state.update(ratio):
                    break

            session.on_arrive(wp)
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y

    # Final frame
    vis.update([], robot_col, robot_row, heading, trajectory, ratio)
    print(f"Demo 2 complete — visual coverage: {ratio:.1%}, distance: {total_distance_m:.1f} m, "
          f"unknown revealed: {n_initial_unknown - int(np.sum(md.unknown_mask))} cells")
    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
    plt.savefig(str(log_dir / f"slam_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
    vis.show()


if __name__ == "__main__":
    main()
