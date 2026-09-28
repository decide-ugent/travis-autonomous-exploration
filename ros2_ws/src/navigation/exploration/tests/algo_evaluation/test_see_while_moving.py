#!/usr/bin/env python3
"""
See-while-moving vs turn-at-goal , exploration-strategy A/B/C bench.

QUESTION
--------
TRAVIS's autonomous node observes the environment ONLY at goal points: it travels
to a waypoint without crediting any coverage in transit
(ros2_exploration_node._do_travel_check does NOT call update_covered_mask), then
turns in place (a sequence of Nav2 Spin actions) and marks coverage once per
completed heading (_on_spin_result -> update_covered_mask). The manual_exploration
baseline node instead credits coverage CONTINUOUSLY while moving (a 5 Hz timer
marks the live TF pose every tick, no spin).

Should we change production to observe while moving? This bench compares three
strategies on the metrics needed to decide, plus map/trajectory visuals. The ONLY
structural difference between the arms is WHEN update_covered_mask is called , the
lever under study. Planner, camera model, and locomotion are identical across arms.

ARMS
----
  turn_at_goal              : production today , no marking during travel; spin at each goal.
  see_while_moving          : mark continuously along the path (heading = travel direction),
                              sampled every OBSERVE_STEP_M of travel; NO spin at goal, arrival
                              heading uncontrolled (mirrors manual_exploration.py).
  see_while_moving_oriented : like see_while_moving, but at each goal turn ONCE to the best FOV
                              heading (most still-uncovered cells) and take one camera view.
                              Since there is no spin, that arrival heading is the only look the
                              camera gets at that waypoint , so aim it deliberately.
  both                      : mark along the path AND spin at each goal.

WHAT IS SIMULATED (mirrors the canonical offline loop visual_demo_coverage.py,
which itself mirrors the current ros2_exploration_node)
-------------------------------------------------------------------------------
  plan_waypoints_raw -> per waypoint: find_path (Nav2 stand-in) travel ->
  [observe while moving, if arm] -> [arrival-time heading refresh + spin, if arm] ->
  on_arrive; drain whole plan (known_map) / progressive reveal (slam). Mid-path
  replan via session.on_step, exactly like the node.

TWO SETTINGS
------------
  known_map : full static map known up front (no unknown cells, no frontiers).
              covered_mask starts empty; each plan drained. Deterministic.
  slam      : progressive reveal , SUT starts blind (unknown=0.5) with a seed
              disc; each observation event ray-casts the TRUE world to reveal
              cells, which are spliced into the SUT p_occ and rebound via
              session.set_map (mirroring the node _on_map). The reveal is driven
              by the arm's OWN observation events, so the map grows DIFFERENTLY per
              arm , the core question: does seeing while moving reveal sooner?
              The SLAM reveal + scoring layer is an INDEPENDENT ray-cast model
              (no exploration.* import in the scorer), reused from
              test_FOV_visited_candidates.py.

TWO SENSORS (per the real robot , see _observe)
-----------------------------------------------
  * COVERAGE (covered_mask, the scored metric): 87° forward CAMERA only. While
    moving, the robot credits ONLY what the depth camera sees in the travel
    direction , it never "sees behind" itself without a spin.
  * SLAM MAP GROWTH (reveal, slam mode only): 360° panoramic LIDAR. The lidar
    sees all around continuously, so the MAP reveals a full circle at each pose
    even with no spin , but that reveal does NOT count as visual coverage.
  turn_at_goal/both additionally get the camera FOV per spin heading at the goal.

METRICS (per map x mode x arm) , the decision table
---------------------------------------------------
  compute time : total_plan_time_s, mean_plan_time_ms, max_plan_time_ms,
                 mark_time_s (update_covered_mask wall time , see_while_moving
                 calls it far more often, a real cost).
  explore time : est_exploration_time_s = path/LINEAR_SPEED + turn/ROT_SPEED
                 + spins*SPIN_OVERHEAD (the headline "faster overall?" number).
  coverage     : final_coverage, reached_threshold, coverage_holes_frac.
  angular turns: total_angular_travel_deg, total_spins, zero_gain_spins.
  path         : path_length_m, coverage_per_m, coverage_auc,
                 path_to_50pct_m, path_to_90pct_m.
  other        : n_plans, n_waypoints, redundant_coverage_frac,
                 n_direction_reversals, total_turning_angle_deg,
                 path_efficiency_vs_nn.

VISUALS (test_see_while_moving_out/)
------------------------------------
  <map>_<mode>_tours.png          : side-by-side per arm , map bg, final covered
                                    overlay, executed TRAJECTORY polyline, numbered
                                    WAYPOINTS reached, start marker.
  <map>_<mode>_coverage_curve.png : coverage-vs-distance, one line per arm, 0.90 hline.
  <map>_<mode>_metric_bars.png    : headline metrics, %-vs-turn_at_goal labels.

Usage:
  python test_see_while_moving.py --maps lab_ghent --mode known_map   # stage-1 smoke
  python test_see_while_moving.py                                     # full matrix
  python test_see_while_moving.py --maps lab_ghent lab_05 --mode slam
  python test_see_while_moving.py --no-viz
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# ── Repo-root walk (depth-independent) ──────────────────────────────────────
_ROOT = Path(__file__).resolve()
while not (_ROOT / ".git").exists() and not (_ROOT / "docker-compose.yml").is_file():
    if _ROOT.parent == _ROOT:
        raise RuntimeError("repo root not found")
    _ROOT = _ROOT.parent
ASSETS_DIR = _ROOT / "assets"
OUT_DIR = Path(__file__).resolve().parent / "test_see_while_moving_out"

DEFAULT_MAPS = ["lab_ghent", "lab_05", "warehouse_amazon", "house_amazon"]
COMPLETION_THRESHOLD = 0.90     # cross-checked against config below; used for hlines
FREE_THRESH_DEFAULT = 0.196
OCC_THRESH_DEFAULT = 0.65

# ── Report-time model constants (NOT planner params , no production YAML home) ──
LINEAR_SPEED_MPS = 0.26         # TB3/MiR sustained forward speed (nav2 vx_max)
ROT_SPEED_DPS = 45.0            # sustained Nav2 Spin angular speed, deg/s
SPIN_OVERHEAD_S = 1.5           # per-Spin action overhead (goal round-trip + settle)
OBSERVE_STEP_M = 0.5            # see-while-moving marking interval along the path

# SLAM stop-condition flexibility (mirrors the production params of the same names).
# Overridable via --min-frontier-cells / --no-progress in main().
MIN_FRONTIER_CELLS = 20         # treat "no frontiers" as union frontier cells <= this (0=strict)
NO_PROGRESS_STREAK = 3          # slam: stop after N arrivals each adding < eps coverage (0=off)
NO_PROGRESS_EPS = 0.005         # min coverage-fraction gain per arrival to count as progress


# ===========================================================================
# REFERENCE / SCORER LAYER , NO exploration.* IMPORTS ALLOWED BELOW THIS LINE
# (independent PGM loader + ray-cast FOV model, reused from
#  test_FOV_visited_candidates.py so ground truth is not the code under test)
# ===========================================================================

@dataclass
class TrueWorld:
    """Ground-truth occupancy, known only to the scorer. p_occ in [0,1]:
    0=free, 1=occupied (no 0.5 , this is the full truth)."""
    p_occ: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    free_thresh: float
    occ_thresh: float

    @property
    def free(self) -> np.ndarray:
        return self.p_occ < self.free_thresh

    @property
    def occupied(self) -> np.ndarray:
        return self.p_occ > self.occ_thresh


def load_true_world(name: str) -> TrueWorld:
    """Independent PGM+YAML loader (PIL+numpy+yaml). Shares NO code with
    exploration.load_map."""
    folder = ASSETS_DIR / name
    with open(folder / "map.yaml") as f:
        meta = yaml.safe_load(f)
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    free_thresh = float(meta.get("free_thresh", FREE_THRESH_DEFAULT))
    occ_thresh = float(meta.get("occupied_thresh", OCC_THRESH_DEFAULT))
    negate = int(meta.get("negate", 0))
    arr = np.array(Image.open(folder / meta["image"]), dtype=np.uint8)
    pf = arr.astype(np.float64)
    p_occ = pf / 255.0 if negate else 1.0 - pf / 255.0
    return TrueWorld(p_occ, resolution, float(origin[0]), float(origin[1]),
                     free_thresh, occ_thresh)


def _cast_fov(world: TrueWorld, col0, row0, heading_deg, fov_deg, max_r, n_rays):
    """Boolean mask of TRUE-WORLD cells the camera at (col0,row0) with the given
    heading+FOV actually observes (stops at occupied/OOB). Independent
    reimplementation used both to REVEAL the SLAM map and to SCORE coverage.
    Convention: col grows right, row grows down, world-y up (hence -sin)."""
    H, W = world.p_occ.shape
    occ = world.occupied
    step = 360.0 / n_rays
    len_deg = step / 2.0
    k0 = math.ceil((heading_deg - fov_deg / 2.0 - len_deg) / step)
    k1 = math.floor((heading_deg + fov_deg / 2.0 + len_deg) / step)
    ks = np.arange(k0, k1 + 1)
    ang = np.deg2rad(ks * step)
    r = np.arange(1, max_r + 1)
    cols = np.floor(col0 + np.outer(np.cos(ang), r) + 0.5).astype(int)
    rows = np.floor(row0 - np.outer(np.sin(ang), r) + 0.5).astype(int)
    inb = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    cc = np.clip(cols, 0, W - 1)
    rr = np.clip(rows, 0, H - 1)
    blocked = (~inb) | occ[rr, cc]
    cum = np.cumsum(blocked, axis=1)
    alive = ((cum - blocked) == 0) & (~blocked)
    seen = np.zeros((H, W), dtype=bool)
    seen[rr[alive], cc[alive]] = True
    first = (cum == 1) & blocked & inb
    seen[rr[first], cc[first]] = True
    return seen


def _select_headings(world, col, row, already_covered, max_r, n_rays, fov, incr):
    """Independent realistic heading selection for one waypoint SPIN , rotate only
    to the headings needed to observe cells not yet covered (same principle as the
    planner's compute_headings_for_waypoint, reimplemented with no exploration.*
    import)."""
    los = _cast_fov(world, col, row, 0.0, 360.0, max_r, n_rays)
    los &= world.free & ~already_covered
    ys, xs = np.where(los)
    if len(xs) == 0:
        return []
    bearings = np.degrees(np.arctan2(-(ys - row), xs - col)) % 360.0
    half = fov / 2.0
    options = [i * incr for i in range(int(round(360.0 / incr)))]
    remaining = np.ones(len(bearings), dtype=bool)
    selected = []
    while remaining.any():
        rb = bearings[remaining]
        best = max(options, key=lambda h: int((np.abs(((rb - h + 180) % 360) - 180) <= half).sum()))
        hit = np.abs(((bearings - best + 180) % 360) - 180) <= half
        newly = remaining & hit
        if not newly.any():
            break
        selected.append(best)
        remaining &= ~hit
    return selected


def _wrapped_step(frm: float, to: float) -> float:
    """|shortest signed delta| in degrees , the node's Spin cost for one heading."""
    return abs((to - frm + 180.0) % 360.0 - 180.0)


# ===========================================================================
# SYSTEM-UNDER-TEST LAYER  (imports exploration.* , production as black box)
# ===========================================================================
_EXPLORATION_PKG = _ROOT / "ros2_ws" / "src" / "navigation" / "exploration"
_TESTS_DIR = _EXPLORATION_PKG / "tests"
sys.path.insert(0, str(_EXPLORATION_PKG))
sys.path.insert(0, str(_TESTS_DIR))

from exploration.explore_costmap_map import (  # noqa: E402
    build_map_data, compute_visibility, compute_headings_for_waypoint,
    update_covered_mask, pixel_to_world, navigable_distance_map,
)
from exploration.execution_strategy import ExplorationSession  # noqa: E402
from exploration.rotation_strategy import get_headings, RotationState  # noqa: E402
from demo_robot import (  # noqa: E402  (real shared helpers , the code the node mirrors)
    build_demo_config, find_path,
    MAX_DETECTION_M, FOV_HORIZONTAL, NUM_RAYS, OBSERVATION_INCREMENT,
)

# Costmap inflation is ROBOT-DEPENDENT: the deployed nav2 config differs per base
# (MiR vs TurtleBot3). Read the SAME inflation_radius the chosen robot's nav2
# costmap uses so the navigable mask (hence candidate generation) matches
# production. INFLATION_M is set from --robot in main(); this module-level default
# keeps direct simulate() callers working.
_NAV2_CONFIG_DIR = _EXPLORATION_PKG.parent / "nav2" / "config"
_ROBOT_NAV2_PARAMS = {
    "mir": "nav2_mir_jazzy_params.yaml",
    "turtle3": "nav2_turtle3_jazzy_params.yaml",
}


def _inflation_radius_for(robot: str) -> float:
    """Read global_costmap inflation_radius (m) from the robot's nav2 params YAML."""
    path = _NAV2_CONFIG_DIR / _ROBOT_NAV2_PARAMS[robot]
    with open(path) as f:
        nav2 = yaml.safe_load(f)
    params = nav2["global_costmap"]["global_costmap"]["ros__parameters"]
    return float(params["inflation_layer"]["inflation_radius"])


def _yaw_goal_tolerance_deg_for(robot: str) -> float:
    """Read the controller_server goal-checker yaw_goal_tolerance (rad -> deg) from
    the robot's nav2 params YAML. This is how far off the REQUESTED heading Nav2
    may leave the robot when it declares the goal reached. It bounds how precisely
    the no-spin 'oriented' arm can actually aim its single camera look."""
    path = _NAV2_CONFIG_DIR / _ROBOT_NAV2_PARAMS[robot]
    with open(path) as f:
        nav2 = yaml.safe_load(f)
    cs = nav2["controller_server"]["ros__parameters"]
    checker = cs.get("general_goal_checker") or cs.get("goal_checker") or {}
    return math.degrees(float(checker["yaw_goal_tolerance"]))


INFLATION_M = _inflation_radius_for("mir")            # default; overridden by --robot in main()
YAW_GOAL_TOL_DEG = _yaw_goal_tolerance_deg_for("mir")  # default; overridden by --robot in main()


def _travel_heading(prev, cur) -> float:
    """Heading of a single travel step (deg, 0=East, 90=North, CCW+). Copied from
    visual_demo_coverage._travel_heading , row-down in image = y-up in world."""
    dc = cur[0] - prev[0]
    dr = -(cur[1] - prev[1])
    return math.degrees(math.atan2(dr, dc)) % 360.0


# The strategy lever: flags decide WHEN/HOW the camera observes.
#   observe_travel : mark the 87° camera along the path (see while moving).
#   spin_at_goal   : full rotation at the goal (turn-at-goal / production behaviour).
#   orient_at_goal : no spin, but turn ONCE to the single best FOV heading (the one
#                    that observes the most still-uncovered cells) and take one
#                    camera view there. Cheap "aim the last look" for the no-spin
#                    robot, since without a spin the arrival heading is the only
#                    view it gets at that waypoint.
ARMS = {
    "turn_at_goal":            {"observe_travel": False, "spin_at_goal": True,  "orient_at_goal": False},  # production
    "see_while_moving":        {"observe_travel": True,  "spin_at_goal": False, "orient_at_goal": False},
    "see_while_moving_oriented": {"observe_travel": True, "spin_at_goal": False, "orient_at_goal": True},
    "both":                    {"observe_travel": True,  "spin_at_goal": True,  "orient_at_goal": False},
}


@dataclass
class Trace:
    """Scoring + trajectory accumulator for one arm on one map/mode."""
    resolution: float
    observe_count: np.ndarray                   # per-cell observation count (true world)
    covered: np.ndarray                         # bool, ground-truth coverage (scorer)
    path_length_m: float = 0.0
    angular_travel_deg: float = 0.0
    n_spins: int = 0
    zero_gain_spins: int = 0
    trajectory_px: list = field(default_factory=list)     # every walked (col,row)
    visited_px: list = field(default_factory=list)        # waypoints reached
    visited_world: list = field(default_factory=list)
    coverage_curve: list = field(default_factory=list)    # (dist_m, coverage_frac)
    n_plans: int = 0
    n_visited: int = 0
    completed: bool = False
    hit_plan_cap: bool = False       # ran out at max_plans -> dithering; metrics unreliable
    final_coverage: float = 0.0
    plan_time_s: float = 0.0
    mark_time_s: float = 0.0


def _make_mapdata(p_occ, world: TrueWorld, covered_mask):
    return build_map_data(
        p_occ=p_occ, resolution=world.resolution,
        origin_x=world.origin_x, origin_y=world.origin_y,
        inflation_radius_m=INFLATION_M,
        free_thresh=world.free_thresh, occ_thresh=world.occ_thresh,
        covered_mask=covered_mask,
        pgm_array=(p_occ * 255).astype(np.uint8),
    )


def _reveal_disc(p_occ, world, col, row, radius_px):
    H, W = world.p_occ.shape
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - col) ** 2 + (yy - row) ** 2 <= radius_px ** 2
    p_occ[disc] = world.p_occ[disc]


def _coverage_frac(trace: Trace, world: TrueWorld) -> float:
    free = world.free
    denom = int(free.sum())
    return 0.0 if denom == 0 else float((trace.covered & free).sum()) / denom


def simulate(name: str, arm: str, mode: str, max_plans: int = 400) -> tuple[Trace, "MapData", TrueWorld]:
    """Drive one arm end-to-end on one map in one mode. Mirrors the canonical
    visual_demo_coverage loop, generalised over the arm's observe flags.

    Returns (trace, final MapData, world) , the MapData/world are used by the
    renderers for the map background."""
    spec = ARMS[arm]
    observe_travel = spec["observe_travel"]
    spin_at_goal = spec["spin_at_goal"]
    orient_at_goal = spec.get("orient_at_goal", False)

    world = load_true_world(name)
    cfg = build_demo_config()
    cfg["min_frontier_cells"] = MIN_FRONTIER_CELLS   # frontier-cell stop tolerance (prod key)
    res = world.resolution
    H, W = world.p_occ.shape
    max_range_px = max(1, int(cfg["max_detection_range"] / res))
    fov = cfg["fov_horizontal"]
    nrays = cfg["num_rays"]
    incr = cfg["observation_rotation_increment"]
    thresh = cfg["exploration_completion_threshold"]

    # Map source: full truth (known_map) vs blind+reveal (slam).
    if mode == "known_map":
        sut_p_occ = world.p_occ.copy()
    else:
        sut_p_occ = np.full((H, W), 0.5, dtype=np.float64)
    covered = np.zeros((H, W), dtype=bool)

    trace = Trace(resolution=res,
                  observe_count=np.zeros((H, W), dtype=np.int32),
                  covered=np.zeros((H, W), dtype=bool))

    # SLAM: seed a revealed disc around the physical start so there is navigable
    # ground to stand on before the first plan.
    sc, sr = W // 2, H // 2
    if mode == "slam":
        if not world.free[sr, sc]:
            fy, fx = np.where(world.free)
            j = int(np.argmin((fx - sc) ** 2 + (fy - sr) ** 2))
            sc, sr = int(fx[j]), int(fy[j])
        _reveal_disc(sut_p_occ, world, sc, sr, max_range_px)

    md = _make_mapdata(sut_p_occ, world, covered)
    session = ExplorationSession(md, cfg)
    robot_col, robot_row = session.nearest_start(W // 2, H // 2)
    robot_x, robot_y = pixel_to_world(robot_col, robot_row, res, world.origin_x, world.origin_y, H)
    heading = 0.0
    trace.trajectory_px.append((robot_col, robot_row))
    dist_since_observe = 0.0

    def _observe(col, row, hdg):
        """Independent observation at one pose. TWO sensors, per the real robot:

          * SLAM map growth (reveal) uses the 360° panoramic LIDAR — it sees all
            around even without spinning, so the map reveals a full circle at this
            pose regardless of heading.
          * Coverage (covered_mask) uses the 87° forward CAMERA at `hdg` — it only
            credits what the depth camera actually observes in the travel/facing
            direction. It NEVER sees behind or beside the robot without a spin.

        Returns cells newly ADDED to coverage (camera), for zero-gain detection."""
        # LIDAR 360° reveal -> grows the SLAM map (unknown -> true value).
        if mode == "slam":
            lidar = _cast_fov(world, col, row, 0.0, 360.0, max_range_px, nrays)
            newly = lidar & (sut_p_occ == 0.5)
            sut_p_occ[newly] = world.p_occ[newly]
        # CAMERA 87° forward -> grows the coverage mask (what was visually observed).
        cam = _cast_fov(world, col, row, hdg, fov, max_range_px, nrays)
        seen_free = cam & world.free
        gained = int((seen_free & ~trace.covered).sum())
        trace.observe_count[seen_free] += 1
        trace.covered |= seen_free
        return gained

    def _rebuild_map():
        """SLAM only: rebuild MapData from the revealed p_occ and rebind the
        session, carrying covered_mask forward (mirrors node _on_map)."""
        nonlocal md
        md = _make_mapdata(sut_p_occ, world, md.covered_mask)
        session.set_map(md)

    stop = False
    stale_streak = 0            # no-progress guard (slam only), mirrors the node
    cov_at_last_arrival = 0.0
    while trace.n_plans < max_plans and not stop:
        t0 = time.perf_counter()
        waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(robot_x, robot_y)
        trace.plan_time_s += time.perf_counter() - t0
        stale_streak = 0        # fresh plan/phase: restart the streak (as the node does)
        cov_at_last_arrival = _coverage_frac(trace, world)

        if not waypoints or (no_frontiers and ratio >= thresh):
            # Terminate as the node does: empty plan, or no frontiers left and the
            # SUT's own coverage ratio has reached the completion threshold.
            trace.completed = bool(ratio >= thresh) or bool(no_frontiers)
            break

        trace.n_plans += 1
        mid_replan = False
        for wp in waypoints:
            path = find_path(md.navigable_mask, (robot_col, robot_row), (wp.col, wp.row))
            if session.on_unreachable(wp, path):
                continue

            # ── Travel leg ─────────────────────────────────────────────────
            prev = path[0]
            for step in path[1:]:
                seg_m = math.hypot(step[0] - prev[0], step[1] - prev[1]) * res
                trace.path_length_m += seg_m
                step_heading = _travel_heading(prev, step)
                # angular travel of the moving camera = heading change along path
                trace.angular_travel_deg += _wrapped_step(heading, step_heading)
                heading = step_heading
                trace.trajectory_px.append(step)

                if observe_travel:
                    dist_since_observe += seg_m
                    if dist_since_observe >= OBSERVE_STEP_M:
                        dist_since_observe = 0.0
                        _observe(step[0], step[1], step_heading)
                        tm = time.perf_counter()
                        # SUT covered_mask (planner's view) = 87° camera forward,
                        # consistent with the scorer above (no spin => no 360° credit).
                        ratio = update_covered_mask(md, step[0], step[1], step_heading,
                                                    fov, max_range_px, nrays)
                        trace.mark_time_s += time.perf_counter() - tm
                        if mode == "slam":
                            _rebuild_map()
                        trace.coverage_curve.append((trace.path_length_m,
                                                     _coverage_frac(trace, world)))

                if session.on_step(step, wp, waypoints):
                    robot_col, robot_row = step
                    robot_x, robot_y = pixel_to_world(step[0], step[1], res,
                                                      world.origin_x, world.origin_y, H)
                    mid_replan = True
                    break
                prev = step
            if mid_replan:
                break

            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y

            # ── Goal spin (arrival-time refresh + RotationState) ───────────
            if spin_at_goal:
                cov_now, _ = compute_visibility((wp.col, wp.row), md, max_range_px, nrays)
                wp.headings = compute_headings_for_waypoint(
                    wp.col, wp.row, cov_now, fov_deg=fov, increment_deg=incr)
                rot = RotationState(stop_after=3)
                for h in get_headings(wp, incr, heading):
                    trace.angular_travel_deg += _wrapped_step(heading, float(h))
                    heading = float(h)
                    gained = _observe(wp.col, wp.row, heading)
                    trace.n_spins += 1
                    if gained == 0:
                        trace.zero_gain_spins += 1
                    tm = time.perf_counter()
                    ratio = update_covered_mask(md, wp.col, wp.row, heading, fov,
                                                max_range_px, nrays)
                    trace.mark_time_s += time.perf_counter() - tm
                    if mode == "slam":
                        _rebuild_map()
                    trace.coverage_curve.append((trace.path_length_m,
                                                 _coverage_frac(trace, world)))
                    if not rot.update(ratio):
                        break

            # ── Goal ORIENTATION (no spin): aim the single last look ───────
            # Without a spin, the arrival heading is the ONLY view the camera gets here. Turn once to the best FOV heading (the one covering the most still-uncovered cells) and take one 87° camera observation, instead of leaving the heading at the uncontrolled travel direction.
            if orient_at_goal:
                cov_now, _ = compute_visibility((wp.col, wp.row), md, max_range_px, nrays)
                best = compute_headings_for_waypoint(
                    wp.col, wp.row, cov_now, fov_deg=fov, increment_deg=incr)
                if best:
                    ideal = float(best[0])   # top pick = most uncovered coverage
                    # Nav2 only guarantees the achieved yaw is within
                    # YAW_GOAL_TOL_DEG of the request. Model the WORST-CASE miss inside that band: of {ideal-tol, ideal, ideal+tol}, take the heading that the camera would cover LEAST from here. This is a conservative lower bound on the oriented arm's benefit at the configured tolerance (tighten yaw_goal_tolerance in nav2 to recover the difference).
                    cand = [ideal]
                    if YAW_GOAL_TOL_DEG > 0:
                        cand = [(ideal - YAW_GOAL_TOL_DEG) % 360.0, ideal,
                                (ideal + YAW_GOAL_TOL_DEG) % 360.0]
                    def _cam_gain(h):
                        seen = _cast_fov(world, wp.col, wp.row, h, fov, max_range_px, nrays)
                        return int((seen & world.free & ~trace.covered).sum())
                    achieved = min(cand, key=_cam_gain)   # worst-case within tolerance
                    trace.angular_travel_deg += _wrapped_step(heading, achieved)
                    heading = achieved
                    _observe(wp.col, wp.row, heading)
                    tm = time.perf_counter()
                    update_covered_mask(md, wp.col, wp.row, heading, fov,
                                        max_range_px, nrays)
                    trace.mark_time_s += time.perf_counter() - tm
                    if mode == "slam":
                        _rebuild_map()
                    trace.coverage_curve.append((trace.path_length_m,
                                                 _coverage_frac(trace, world)))

            session.on_arrive(wp)
            trace.visited_px.append((wp.col, wp.row))
            trace.visited_world.append((wp.x, wp.y))
            trace.n_visited += 1
            # curve sample at every arrival (guarantees a point even if no observe)
            cov_now = _coverage_frac(trace, world)
            trace.coverage_curve.append((trace.path_length_m, cov_now))

            # No-progress guard (slam only, mirrors the node): stop after NO_PROGRESS_STREAK consecutive arrivals each adding < eps coverage.
            if mode == "slam" and NO_PROGRESS_STREAK > 0:
                if (cov_now - cov_at_last_arrival) < NO_PROGRESS_EPS:
                    stale_streak += 1
                else:
                    stale_streak = 0
                cov_at_last_arrival = cov_now
                if stale_streak >= NO_PROGRESS_STREAK:
                    trace.completed = True
                    stop = True
                    break

    trace.final_coverage = _coverage_frac(trace, world)
    # Hit the plan cap without terminating = the arm dithered (replanning without reaching waypoints). Its time/plan metrics are dominated by the cap, not by real exploration cost, so flag the run as unreliable rather than reporting distorted numbers as if they were a fair comparison.
    trace.hit_plan_cap = (trace.n_plans >= max_plans) and not trace.completed
    return trace, md, world


# ===========================================================================
# Metrics
# ===========================================================================

def _coverage_auc(curve, total_m):
    if len(curve) < 2 or total_m <= 0:
        return curve[-1][1] if curve else 0.0
    xs = [0.0] + [d for d, _ in curve]
    ys = [0.0] + [c for _, c in curve]
    area = sum(0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]) for i in range(1, len(xs)))
    return area / total_m


def _milestone(curve, target):
    for d, c in curve:
        if c >= target:
            return round(d, 2)
    return None


def _turning(pts):
    """(#direction reversals, total absolute turning angle deg) over the visited-
    waypoint polyline (port from benchmark_waypoint_scoring._turning)."""
    if len(pts) < 3:
        return 0, 0.0
    reversals, total = 0, 0.0
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(cosang))
        total += ang
        if ang > 120.0:
            reversals += 1
    return reversals, total


def _nn_lower_bound_m(md, visited_px):
    """Nearest-neighbour tour length over visited waypoints using production navigable_distance_map (geodesic px), in metres. Cheap reference for path_efficiency = executed / this."""
    if len(visited_px) < 2:
        return 0.0
    remaining = list(visited_px[1:])
    cur = visited_px[0]
    total_px = 0.0
    while remaining:
        dm = navigable_distance_map(md.navigable_mask, cur[0], cur[1], md.inflation_px)
        reach = [w for w in remaining if dm[w[1], w[0]] != np.inf]
        if not reach:
            break
        nxt = min(reach, key=lambda w: dm[w[1], w[0]])
        total_px += dm[nxt[1], nxt[0]]
        cur = nxt
        remaining.remove(nxt)
    return total_px * md.resolution


def compute_metrics(name, mode, arm, trace: Trace, md, world) -> dict:
    free = world.free
    denom = int(free.sum())
    observed = trace.observe_count > 0
    redundant = trace.observe_count > 1
    cov = trace.final_coverage
    holes = int((free & ~observed).sum())
    auc = _coverage_auc(trace.coverage_curve, trace.path_length_m)
    nn_lb = _nn_lower_bound_m(md, trace.visited_px)
    eff = (trace.path_length_m / nn_lb) if nn_lb > 0 else None
    reversals, turning = _turning(trace.visited_px)
    est_t = (trace.path_length_m / LINEAR_SPEED_MPS
             + trace.angular_travel_deg / ROT_SPEED_DPS
             + trace.n_spins * SPIN_OVERHEAD_S)
    return {
        "map": name,
        "mode": mode,
        "arm": arm,
        "hit_plan_cap": bool(trace.hit_plan_cap),   # True => metrics UNRELIABLE (dithered)
        # compute time
        "total_plan_time_s": round(trace.plan_time_s, 3),
        "mean_plan_time_ms": round(1000 * trace.plan_time_s / max(1, trace.n_plans), 2),
        "mark_time_s": round(trace.mark_time_s, 3),
        # exploration time (est.)
        "est_exploration_time_s": round(est_t, 1),
        # coverage
        "final_coverage": round(cov, 4),
        "reached_threshold": bool(cov >= COMPLETION_THRESHOLD),
        "coverage_holes_frac": round(holes / denom, 4) if denom else 0.0,
        # angular turns
        "total_angular_travel_deg": round(trace.angular_travel_deg, 1),
        "total_spins": trace.n_spins,
        "zero_gain_spins": trace.zero_gain_spins,
        # path
        "path_length_m": round(trace.path_length_m, 2),
        "coverage_per_m": round(cov / trace.path_length_m, 5) if trace.path_length_m else 0.0,
        "coverage_auc": round(auc, 4),
        "path_to_50pct_m": _milestone(trace.coverage_curve, 0.50),
        "path_to_90pct_m": _milestone(trace.coverage_curve, 0.90),
        # other
        "n_plans": trace.n_plans,
        "n_waypoints": trace.n_visited,
        "redundant_coverage_frac": round(int(redundant.sum()) / max(1, int(observed.sum())), 4),
        "n_direction_reversals": reversals,
        "total_turning_angle_deg": round(turning, 1),
        "path_efficiency_vs_nn": round(eff, 3) if eff is not None else None,
    }


# +1 = higher is better, -1 = lower is better, None = neutral/info.
_METRIC_BETTER = {
    "est_exploration_time_s": -1, "final_coverage": +1, "coverage_holes_frac": -1,
    "total_angular_travel_deg": -1, "total_spins": -1, "zero_gain_spins": -1,
    "path_length_m": -1, "coverage_per_m": +1, "coverage_auc": +1,
    "path_to_50pct_m": -1, "path_to_90pct_m": -1, "redundant_coverage_frac": -1,
    "n_direction_reversals": -1, "total_turning_angle_deg": -1,
    "path_efficiency_vs_nn": -1, "total_plan_time_s": -1, "mark_time_s": -1,
}


# ===========================================================================
# Report tables
# ===========================================================================

_TABLE_KEYS = [
    "est_exploration_time_s", "final_coverage", "coverage_holes_frac",
    "path_length_m", "total_angular_travel_deg", "total_spins", "zero_gain_spins",
    "coverage_auc", "path_to_90pct_m", "redundant_coverage_frac",
    "total_plan_time_s", "mark_time_s", "n_waypoints",
]


def print_table(rows, name, mode):
    print(f"\n=== {name} / {mode} ===")
    hdr = f"{'metric':<26s}" + "".join(f"{r['arm']:>18s}" for r in rows)
    print(hdr)
    print("-" * len(hdr))
    capped = [r["arm"] for r in rows if r.get("hit_plan_cap")]
    if capped:
        print(f"  !! UNRELIABLE (hit plan cap / dithered): {', '.join(capped)} "
              f"— time/plan metrics distorted, do not compare")
    base = next((r for r in rows if r["arm"] == "turn_at_goal"), rows[0])
    for k in _TABLE_KEYS:
        line = f"{k:<26s}"
        for r in rows:
            v = r[k]
            cell = "None" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            if r["arm"] != base["arm"] and _METRIC_BETTER.get(k) and base[k] not in (None, 0) \
                    and v is not None and isinstance(v, (int, float)):
                d = (v - base[k]) / abs(base[k]) * 100.0
                mark = "+" if d * _METRIC_BETTER[k] > 0 else ("." if d == 0 else "-")
                cell += f"({d:+.0f}%{mark})"
            line += f"{cell:>18s}"
        print(line)


# ===========================================================================
# Visuals
# ===========================================================================

def _bg_image(md):
    return np.where(md.occupied_mask, 0.0, np.where(md.free_mask, 1.0, 0.5))


def render_tours(tag, name, mode, results, out_dir):
    """Side-by-side per arm: map bg, final covered overlay, executed trajectory polyline, numbered waypoints reached, start marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(results)
    fig, axes = plt.subplots(1, len(arms), figsize=(6.5 * len(arms), 6.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        trace, md, world = results[arm]
        ax.imshow(_bg_image(md), cmap="gray", origin="upper")
        cov = np.ma.masked_where(~(trace.covered & world.free), trace.covered.astype(float))
        ax.imshow(cov, cmap="autumn", alpha=0.5, origin="upper", vmin=0, vmax=1)
        if trace.trajectory_px:
            xs = [c for c, _ in trace.trajectory_px]
            ys = [r for _, r in trace.trajectory_px]
            ax.plot(xs, ys, "-", color="deepskyblue", lw=1.1, alpha=0.9)
            ax.plot(xs[0], ys[0], "s", color="lime", ms=9, label="start")
        for i, (c, r) in enumerate(trace.visited_px, start=1):
            ax.plot(c, r, "o", color="navy", ms=4)
            ax.annotate(str(i), (c, r), fontsize=6, color="white",
                        ha="center", va="center")
        m = compute_metrics(name, mode, arm, trace, md, world)
        ax.set_title(f"{arm}\ncov={m['final_coverage']:.1%}  path={m['path_length_m']:.0f}m  "
                     f"turn={m['total_angular_travel_deg']:.0f}°  spins={m['total_spins']}\n"
                     f"est_time={m['est_exploration_time_s']:.0f}s",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name} / {mode} — trajectory (blue), coverage (orange), "
                 f"waypoints (numbered)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_tours.png", dpi=120)
    plt.close(fig)


def render_coverage_curve(tag, name, mode, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, (trace, _md, _w) in results.items():
        if not trace.coverage_curve:
            continue
        xs = [d for d, _ in trace.coverage_curve]
        ys = [c for _, c in trace.coverage_curve]
        ax.plot(xs, ys, "-", label=arm, lw=1.6)
    ax.axhline(COMPLETION_THRESHOLD, color="k", ls="--", alpha=0.5,
               label=f"threshold {COMPLETION_THRESHOLD:.0%}")
    ax.set_xlabel("path distance travelled (m)")
    ax.set_ylabel("coverage fraction")
    ax.set_title(f"{name} / {mode} — coverage vs distance (front-loaded = better)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_coverage_curve.png", dpi=120)
    plt.close(fig)


def render_metric_bars(tag, name, mode, metric_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["est_exploration_time_s", "path_length_m", "total_angular_travel_deg",
            "final_coverage", "total_plan_time_s"]
    titles = ["est. exploration time (s)", "path length (m)",
              "total angular travel (°)", "final coverage", "total plan time (s)"]
    arms = [r["arm"] for r in metric_rows]
    base = next((r for r in metric_rows if r["arm"] == "turn_at_goal"), metric_rows[0])
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.5))
    for ax, key, title in zip(axes, keys, titles):
        vals = [r[key] if r[key] is not None else 0 for r in metric_rows]
        colors = ["#888888" if a == "turn_at_goal" else "#2b8cbe" for a in arms]
        ax.bar(range(len(arms)), vals, color=colors)
        for i, v in enumerate(vals):
            lbl = f"{v:.2f}" if isinstance(v, float) and v < 10 else f"{v:.0f}"
            if arms[i] != base["arm"] and base[key] not in (None, 0):
                d = (v - base[key]) / abs(base[key]) * 100.0
                lbl += f"\n{d:+.0f}%"
            ax.text(i, v, lbl, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{name} / {mode} — headline metrics (% vs turn_at_goal)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_metric_bars.png", dpi=120)
    plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================

def main():
    global INFLATION_M, YAW_GOAL_TOL_DEG, MIN_FRONTIER_CELLS, NO_PROGRESS_STREAK
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    ap.add_argument("--mode", choices=["known_map", "slam", "both"], default="both")
    ap.add_argument("--robot", choices=list(_ROBOT_NAV2_PARAMS), default="mir",
                    help="selects the nav2 params YAML whose costmap inflation_radius "
                         "the navigable mask is built with (mir=0.6, turtle3=0.2)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help=f"subset of {list(ARMS)}")
    ap.add_argument("--max-plans", type=int, default=400,
                    help="safety cap on planning cycles; a run that hits it is flagged "
                         "hit_plan_cap (dithered, metrics unreliable)")
    ap.add_argument("--yaw-tol-deg", type=float, default=None,
                    help="override the nav2 yaw_goal_tolerance (deg) used to model the "
                         "worst-case aim error of the oriented arm; default reads it from "
                         "the robot's nav2 params YAML. Use 0 for perfect aim (exact).")
    ap.add_argument("--min-frontier-cells", type=int, default=None,
                    help="SLAM stop tolerance: treat 'no frontiers' as union frontier "
                         f"cells <= this (default {MIN_FRONTIER_CELLS}; 0 = strict).")
    ap.add_argument("--no-progress", type=int, default=None,
                    help="SLAM no-progress guard: stop after N consecutive arrivals each "
                         f"adding < {NO_PROGRESS_EPS:.1%} coverage (default {NO_PROGRESS_STREAK}; 0 = off).")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    # Robot-dependent costmap inflation + goal-yaw tolerance from the real nav2 config.
    INFLATION_M = _inflation_radius_for(args.robot)
    YAW_GOAL_TOL_DEG = (args.yaw_tol_deg if args.yaw_tol_deg is not None
                        else _yaw_goal_tolerance_deg_for(args.robot))
    if args.min_frontier_cells is not None:
        MIN_FRONTIER_CELLS = args.min_frontier_cells
    if args.no_progress is not None:
        NO_PROGRESS_STREAK = args.no_progress
    print(f"robot={args.robot}  inflation_radius={INFLATION_M} m  "
          f"yaw_goal_tolerance={YAW_GOAL_TOL_DEG:.1f}°  "
          f"min_frontier_cells={MIN_FRONTIER_CELLS}  no_progress_streak={NO_PROGRESS_STREAK}")

    modes = ["known_map", "slam"] if args.mode == "both" else [args.mode]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        all_metrics = []
        for name in args.maps:
            results = {}      # arm -> (trace, md, world)
            metric_rows = []
            for arm in args.arms:
                trace, md, world = simulate(name, arm, mode, max_plans=args.max_plans)
                results[arm] = (trace, md, world)
                m = compute_metrics(name, mode, arm, trace, md, world)
                metric_rows.append(m)
                all_metrics.append(m)
            for m in metric_rows:
                m["robot"] = args.robot
                m["min_frontier_cells"] = MIN_FRONTIER_CELLS
                m["no_progress_streak"] = NO_PROGRESS_STREAK
            print_table(metric_rows, name, mode)
            if not args.no_viz:
                tag = f"{name}_{args.robot}_{mode}"
                render_tours(tag, name, mode, results, OUT_DIR)
                render_coverage_curve(tag, name, mode, results, OUT_DIR)
                render_metric_bars(tag, name, mode, metric_rows, OUT_DIR)
                print(f"  visuals -> {OUT_DIR}")

        stem = f"metrics_{args.robot}_{mode}"
        with open(OUT_DIR / f"{stem}.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        with open(OUT_DIR / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
            w.writeheader()
            w.writerows(all_metrics)
        print(f"\nmetrics -> {OUT_DIR / stem}.json / .csv")


if __name__ == "__main__":
    main()
