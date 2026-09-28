"""
Demo 1 — Visual coverage on a fully known map.

The robot navigates the lab_ghent map according to the exploration planner,
moving pixel-by-pixel along a collision-free path to each waypoint. The
coverage mask (green) accumulates in real time as the robot travels and rotates.

10 labelled objects are scattered across the map as coloured markers.
They are reference points only (not detected by the camera in this demo —
see visual_demo_semantic.py for detection).

Run:
    python tests/visual_demo_coverage.py

Controls:
    Close the window to end the demo early.

Configurable constants below.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import csv

import numpy as np

# ── sys.path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent.parent  # /ros2_ws/src/
_TESTS_DIR = Path(__file__).parent                  # tests/
_EXPLORATION_PKG = _ROOT / "navigation" / "exploration"
sys.path.insert(0, str(_TESTS_DIR))  # for conftest constants
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "navigation"))
# sys.path.insert(0, str(_ROOT / "travis_brain"))
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
    place_objects, find_path, check_detections, build_demo_config,
    FOV_HORIZONTAL, OBSERVATION_INCREMENT, NUM_RAYS, INFLATION_M,
)
from navigation.exploration.exploration.rotation_strategy import get_headings, RotationState
from navigation.exploration.exploration.execution_strategy import ExplorationSession
from conftest import REFERENCE_MAP_PGM, REFERENCE_MAP_YAML

# ── Configurable ──────────────────────────────────────────────────────────
N_OBJECTS    = 10
SEED         = 42
PAUSE_TRAVEL = 0.002    # seconds per pixel step along path
PAUSE_ROTATE = 0.007    # seconds per heading rotation at waypoint
# 
#pure visualisation timer
PLOT_SAVING_TIME = 10.0 #─────────────────────────────────────────────────────────────────────────

MAP_PGM  = REFERENCE_MAP_PGM
MAP_YAML = REFERENCE_MAP_YAML

TEST_DIR_SAVE = _TESTS_DIR / "visual_demo_coverage"
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
    """Heading of a single travel step (degrees, 0=East, 90=North, CCW+)."""
    dc = cur[0] - prev[0]
    dr = -(cur[1] - prev[1])   # row↓ in image = y↑ in world
    return math.degrees(math.atan2(dr, dc)) % 360.0



def main() -> None:
    # ── Load map ──────────────────────────────────────────────────────────
    md = load_map(MAP_PGM, MAP_YAML, INFLATION_M)
    H, W = md.pgm_array.shape
    cfg = build_demo_config()
    max_range_px = max(1, int(cfg["max_detection_range"] / md.resolution))

    # ── Place objects ─────────────────────────────────────────────────────
    objects = place_objects(
        md.navigable_mask, md.resolution, md.origin_x, md.origin_y,
        n=N_OBJECTS, seed=SEED,
    )

    # ── Visualiser ────────────────────────────────────────────────────────
    vis = SemanticExplorationVisualiser(md, objects, show_3d=False, all_objects_visible=True)
    vis.show_nonblocking()

    # ── Exploration session + robot start ─────────────────────────────────
    session = ExplorationSession(md, cfg)
    cx, cy = W // 2, H // 2
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
    total_distance_m = 0.0
    _last_plot_save = time.time()

    # ── Log setup ─────────────────────────────────────────────────────────
    start_time = time.time()
    log_dir = _get_unique_log_dir("coverage_demo_log", TEST_DIR_SAVE)
    log_plans: list[dict] = []
    log_waypoints: list[dict] = []
    log_motion: list[dict] = []
    log_events: list[dict] = []
    log_candidates: list[dict] = []
    plan_id = 0
    step_id = 0
    n_detected = 0

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        waypoints, ratio, no_frontiers, candidate_records = session.plan_waypoints_raw(robot_x, robot_y)
        if (no_frontiers and ratio >= cfg["exploration_completion_threshold"]) or not waypoints:
            break

        plan_id += 1
        t = round(time.time() - start_time, 2)

        log_plans = _append_logs(log_plans, plan_id, robot_col,
                                 robot_row, robot_x, robot_y, ratio, waypoints, start_time, n_detected)

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
                    "plan_id":      plan_id,
                    "type":         "unreachable_waypoint",
                    "wp_col":       wp.col,
                    "wp_row":       wp.row,
                    "wp_x_m":       round(wp.x, 3),
                    "wp_y_m":       round(wp.y, 3),
                    "detail":       "find_path returned only start; no navigable path exists",
                    "timestamp_s":  round(time.time() - start_time, 2),
                })
                continue
            elif euclid_px > 0:
                detour = bfs_px / euclid_px
                if detour > 1.5:
                    log_events.append({
                        "plan_id":      plan_id,
                        "type":         "wall_detour",
                        "wp_col":       wp.col,
                        "wp_row":       wp.row,
                        "wp_x_m":       round(wp.x, 3),
                        "wp_y_m":       round(wp.y, 3),
                        "detail":       (f"BFS {bfs_px} px vs Euclidean {euclid_px:.1f} px "
                                        f"(detour ×{detour:.2f})"),
                        "timestamp_s":  round(time.time() - start_time, 2),
                    })

            # Travel along path. Coverage is NOT updated in transit — the node marks coverage
            # only at rotation (arrival-only); mirror that here so the demo reflects real behaviour.
            _mid_replan = False
            prev = path[0]
            for step in path[1:]:
                step_id += 1
                heading = _travel_heading(prev, step)
                total_distance_m += math.hypot(step[0] - prev[0], step[1] - prev[1]) * md.resolution
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
                    "n_detected":           n_detected,
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
                    plt.savefig(str(log_dir / f"coverage_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    return
                if time.time() - _last_plot_save >= PLOT_SAVING_TIME:
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"coverage_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    _last_plot_save = time.time()
                prev = step

            # Mid-path replan aborts the whole plan (like the node cancelling the goal → PLANNING).
            if _mid_replan:
                break

            # Rotate at waypoint — headings recomputed from the current covered_mask at arrival.
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y
            _rot_state = RotationState(stop_after=3)
            _cov_now, _ = compute_visibility((wp.col, wp.row), md, max_range_px, NUM_RAYS)
            wp.headings = compute_headings_for_waypoint(
                wp.col, wp.row, _cov_now, fov_deg=FOV_HORIZONTAL, increment_deg=OBSERVATION_INCREMENT)
            for h in get_headings(wp, OBSERVATION_INCREMENT, heading):
                step_id += 1
                heading = float(h)
                ratio = update_covered_mask(md, wp.col, wp.row, heading, FOV_HORIZONTAL,
                                            max_range_px, NUM_RAYS)
                newly = check_detections(
                    md.occupied_mask, md.unknown_mask, md.resolution,
                    wp.col, wp.row, heading, FOV_HORIZONTAL, objects,
                )
                n_detected += len(newly)
                for obj in newly:
                    log_events.append({
                        "plan_id":     plan_id,
                        "type":        "object_detected",
                        "wp_col":      wp.col,
                        "wp_row":      wp.row,
                        "wp_x_m":      round(obj.x, 3),
                        "wp_y_m":      round(obj.y, 3),
                        "detail":      f"label={obj.label} at ({obj.col},{obj.row})",
                        "timestamp_s": round(time.time() - start_time, 2),
                    })
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
                    "n_detected":           n_detected,
                    "timestamp_s":          round(time.time() - start_time, 2),
                })
                vis.update(waypoints, wp.col, wp.row, heading, trajectory, ratio)
                plt.pause(PAUSE_ROTATE)
                if not plt.get_fignums():
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"coverage_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    return
                if time.time() - _last_plot_save >= PLOT_SAVING_TIME:
                    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
                    plt.savefig(str(log_dir / f"coverage_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
                    _last_plot_save = time.time()
                if not _rot_state.update(ratio):
                    break

            session.on_arrive(wp)
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y

    # Final frame
    vis.update([], robot_col, robot_row, heading, trajectory, ratio)
    print(f"Demo 1 complete — final visual coverage: {ratio:.1%}, distance: {total_distance_m:.1f} m")
    _save_log(log_plans, log_waypoints, log_motion, log_events, log_candidates, log_dir)
    plt.savefig(str(log_dir / f"coverage_demo_plot_{time.time():.0f}.png"), dpi=150, bbox_inches="tight")
    vis.show()

def _append_logs(
        log_plans: list[dict],
        plan_id: int,
        robot_col: int,
        robot_row: int,
        robot_x: float,
        robot_y: float,
        ratio: float,
        waypoints: int,
        start_time: float,
        n_detected: int = 0,
):
    # ── Log this planning event ───────────────────────────────────────
    log_plans.append({
            "plan_id":       plan_id,
            "robot_col":     robot_col,
            "robot_row":     robot_row,
            "robot_x_m":     round(robot_x, 3),
            "robot_y_m":     round(robot_y, 3),
            "coverage_pct":  round(ratio * 100, 2),
            "n_waypoints":   len(waypoints),
            "n_detected":    n_detected,
            "timestamp_s":   round(time.time() - start_time, 2),
        })
    return log_plans

def _save_log(
    plans: list[dict],
    waypoints: list[dict],
    motion: list[dict],
    events: list[dict],
    candidates: list[dict],
    log_dir: Path | None = None,
) -> None:
    """Write all collected log data to CSV files inside a directory."""
    if log_dir is None:
        log_dir = _TESTS_DIR / "visual_demo_coverage/coverage_demo_log"
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


if __name__ == "__main__":
    main()
