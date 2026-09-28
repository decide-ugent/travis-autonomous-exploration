"""
Waypoint-scoring benchmark.

Purpose
-------
Quantify the exploration planner's behaviour so changes to the scoring logic
(greedy_set_cover) can be compared before/after on a fixed set of maps.

It drives a HEADLESS version of the exploration session loop (the same call
sequence as tests/visual_demo_coverage.py, minus matplotlib / object detection):

    session.plan_waypoints_raw -> for each wp: find_path -> travel (update
    covered_mask at arrival) -> on_arrive -> replan when the plan is drained.

For every (map, run) it records a MetricSet (see compute_metrics) covering
coverage quality, path efficiency, the signatures of the three known scoring
problems (A myopic distance, B weak far-penalty, C objective/metric mismatch),
and plan compute cost.

Maps
----
Synthetic (fully known -> frontier_gain == 0 -> pure coverage+distance scoring):
    - large_rectangle : open hall, tests raw coverage tiling
    - L_shape         : concave corner, tests around-the-corner routing
    - donut           : ring with a central obstacle, the classic case where
                        Euclidean-near != navigable-near (problem A signature)
Asset maps (loaded from assets/): lab_05, lab_ghent, warehouse_amazon (real occupancy grids).

Usage
-----
    python tests/algo_evaluation/benchmark_waypoint_scoring.py            # table
    python tests/algo_evaluation/benchmark_waypoint_scoring.py --variant baseline
    python tests/algo_evaluation/benchmark_waypoint_scoring.py --viz lab_05

The module is import-safe: `run_all()` returns the results so a pytest or a
before/after driver can call it without the CLI.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# ── Repo-root walk (depth-independent, matches the algo_evaluation notebooks) ─
_ROOT = Path(__file__).resolve()
while not (_ROOT / ".git").exists() and not (_ROOT / "docker-compose.yml").is_file():
    if _ROOT.parent == _ROOT:
        raise RuntimeError("repo root not found")
    _ROOT = _ROOT.parent

ASSETS_DIR = _ROOT / "assets"
_EXPLORATION_PKG = _ROOT / "ros2_ws" / "src" / "navigation" / "exploration"
_TESTS_DIR = _EXPLORATION_PKG / "tests"
sys.path.insert(0, str(_EXPLORATION_PKG))    # -> import exploration.*
sys.path.insert(0, str(_TESTS_DIR))          # -> import demo_robot

from exploration.explore_costmap_map import (  # noqa: E402
    MapData,
    build_map_data,
    load_map,
    pixel_to_world,
    update_covered_mask,
    compute_visibility,
    compute_headings_for_waypoint,
    navigable_distance_map,
)
from scoring_variants import VariantSession  # noqa: E402
from demo_robot import find_path, build_demo_config, INFLATION_M  # noqa: E402

# Perception constants used by the coverage model. Read once from config so the
# benchmark tracks the same FOV / range / ray count the planner is tuned for.
_CFG = build_demo_config()
FOV_HORIZONTAL: float = _CFG["fov_horizontal"]
NUM_RAYS: int = _CFG["num_rays"]
OBSERVATION_INCREMENT: float = _CFG["observation_rotation_increment"]


# ==========================================================================
# Scoring variants
# ==========================================================================
# Each variant is a set of config overrides layered on the base demo config, so
# every ablation stage is a reproducible run (python ... --variant <name>) and
# no variant overwrites another in code. These overrides are consumed by the
# self-contained scoring path (scoring_variants.plan_waypoints_variant /
# greedy_set_cover_variant, via VariantSession) — NOT production, which now
# hardcodes expdecay and carries no such knobs. The "B_expdecay" variant
# (expdecay, gamma=1, no sequential, no floor) reproduces production scoring, a
# cross-check that this test copy has not drifted from prod.
#
#   baseline               : linear distance penalty (production default), gamma=1
#   A_sequential           : tour-aware re-seeded distance (fix A) — underperforms
#   B_expdecay             : exponential distance decay (fix B), gamma=1
#   AB_sequential_expdecay : both fixes combined
#
# gamma sweep on expdecay (fix-B tuning): expdecay at gamma=1 wins on path but
# selects more, tighter waypoints -> +25..58% plan time. Lower gamma = gentler
# decay = fewer near-picks, aiming to keep the path win at less compute.
#   B_expdecay_g069 : gamma=0.69 -> factor x0.50 at max-range (matches the old
#                     linear 'halved at max_range' feel).
#   B_expdecay_g050 : gamma=0.50 -> even gentler.
#   B_gaussian      : half-Gaussian shape (flat near robot, hard far cutoff) at
#                     gamma=1, testing whether the shape alone curbs near-picks.
VARIANTS: dict[str, dict] = {
    "baseline":               {"distance_model": "linear",   "scoring_sequential": False},
    "A_sequential":           {"distance_model": "linear",   "scoring_sequential": True},
    "B_expdecay":             {"distance_model": "expdecay",  "scoring_sequential": False},
    "AB_sequential_expdecay": {"distance_model": "expdecay",  "scoring_sequential": True},
    "B_expdecay_g069":        {"distance_model": "expdecay",  "scoring_sequential": False,
                               "travel_cost_weight": 0.69},
    "B_expdecay_g050":        {"distance_model": "expdecay",  "scoring_sequential": False,
                               "travel_cost_weight": 0.50},
    "B_gaussian":             {"distance_model": "gaussian",  "scoring_sequential": False},
    # Lever 4 (structural, not a user knob): expdecay@g1 + marginal-gain floor.
    # The floor drops low-value tail waypoints that inflate expdecay's plan-cycle
    # cost, aiming to keep the path win at lower compute. Tested on lab_05 and
    # lab_ghent per request. 0.15 = stop once a pick's marginal gain < 15% of the
    # first pick's.
    "B_expdecay_floor":       {"distance_model": "expdecay",  "scoring_sequential": False,
                               "gain_floor": 0.15},
}


def build_variant_config(variant: str) -> dict:
    """Base demo config with the named variant's scoring overrides applied."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    cfg = build_demo_config()
    cfg.update(VARIANTS[variant])
    return cfg


# ==========================================================================
# Synthetic map builders
# ==========================================================================
# build_map_data expects p_occ in [0, 1]: 0 = free, 1 = occupied, 0.5 = unknown.
# Fully-known synthetic maps use only 0.0 (free) and 1.0 (occupied) so
# unknown_mask is empty and frontier_gain is 0 everywhere -> pure coverage.

_SYNTH_RESOLUTION = 0.05   # m/px, matches the asset maps
_SYNTH_ORIGIN = (0.0, 0.0)


def _finalise(p_occ: np.ndarray) -> MapData:
    return build_map_data(
        p_occ=p_occ,
        resolution=_SYNTH_RESOLUTION,
        origin_x=_SYNTH_ORIGIN[0],
        origin_y=_SYNTH_ORIGIN[1],
        inflation_radius_m=INFLATION_M,
    )


def make_large_rectangle(w_m: float = 20.0, h_m: float = 12.0) -> MapData:
    """Open rectangular hall with a 1-cell occupied border wall."""
    H = int(h_m / _SYNTH_RESOLUTION)
    W = int(w_m / _SYNTH_RESOLUTION)
    p = np.zeros((H, W), dtype=np.float64)
    p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 1.0
    return _finalise(p)


def make_L_shape(arm_m: float = 14.0, width_m: float = 6.0) -> MapData:
    """L-shaped corridor: a big square with one quadrant filled solid."""
    side = int(arm_m / _SYNTH_RESOLUTION)
    band = int(width_m / _SYNTH_RESOLUTION)
    p = np.zeros((side, side), dtype=np.float64)
    # Fill the top-right block solid, leaving an L of free space.
    p[:side - band, band:] = 1.0
    # Border wall.
    p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 1.0
    return _finalise(p)


def make_donut(outer_m: float = 16.0, hole_m: float = 6.0) -> MapData:
    """Square hall with a solid central obstacle -> a navigable ring.

    The classic problem-A case: cells across the hole are Euclidean-near but
    navigable-far, so any distance term must use BFS distance to route sanely.
    """
    side = int(outer_m / _SYNTH_RESOLUTION)
    p = np.zeros((side, side), dtype=np.float64)
    lo = (side - int(hole_m / _SYNTH_RESOLUTION)) // 2
    hi = lo + int(hole_m / _SYNTH_RESOLUTION)
    p[lo:hi, lo:hi] = 1.0
    p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 1.0
    return _finalise(p)


def load_asset(name: str) -> MapData:
    folder = ASSETS_DIR / name
    return load_map(folder / "map.pgm", folder / "map.yaml", INFLATION_M)


SYNTHETIC_BUILDERS = {
    "large_rectangle": make_large_rectangle,
    "L_shape": make_L_shape,
    "donut": make_donut,
}
ASSET_MAPS = ["lab_05", "lab_ghent", "warehouse_amazon"]


# ==========================================================================
# Headless session runner
# ==========================================================================

@dataclass
class RunTrace:
    """Everything the metrics need, collected while driving the session."""
    resolution: float
    # coverage curve: (cumulative_path_m, coverage_ratio) sampled at each arrival
    coverage_curve: list[tuple[float, float]] = field(default_factory=list)
    visited_wp_px: list[tuple[int, int]] = field(default_factory=list)  # visit order
    visited_wp_world: list[tuple[float, float]] = field(default_factory=list)
    path_length_m: float = 0.0
    n_plans: int = 0
    n_waypoints: int = 0            # total waypoints across all plans (planned)
    n_visited: int = 0             # waypoints actually arrived at
    plan_times_s: list[float] = field(default_factory=list)
    final_coverage: float = 0.0
    completed: bool = False
    observe_count: np.ndarray | None = None   # per free-cell observation tally
    selected_geodesic_px: list[float] = field(default_factory=list)


def _observe_at(md: MapData, col: int, row: int, max_range_px: int,
                observe_count: np.ndarray) -> float:
    """Rotate in place: cover all headings the planner would use, tally per-cell
    observation counts. Returns the coverage ratio after the update.

    Mirrors the demo's arrival behaviour (headings from current covered_mask),
    sweeping every selected heading so coverage is order-independent and the
    benchmark is deterministic.
    """
    cov_now, _ = compute_visibility((col, row), md, max_range_px, NUM_RAYS)
    headings = compute_headings_for_waypoint(
        col, row, cov_now, fov_deg=FOV_HORIZONTAL, increment_deg=OBSERVATION_INCREMENT)
    if not headings:
        headings = [0.0]

    # Measure THIS stop's raw camera footprint on a scratch mask so the tally
    # reflects only cells this stop illuminates (update_covered_mask on the real
    # mask is cumulative and would make every already-covered cell look
    # re-observed). +1 per free cell this stop sees -> observe_count>1 == a cell
    # genuinely observed by more than one stop (true redundancy).
    real_mask = md.covered_mask
    scratch = np.zeros_like(real_mask)
    md.covered_mask = scratch
    for h in headings:
        update_covered_mask(md, col, row, h, FOV_HORIZONTAL, max_range_px, NUM_RAYS)
    md.covered_mask = real_mask
    observe_count[scratch & md.free_mask] += 1

    # Apply this stop's footprint to the real cumulative mask and return ratio.
    ratio = 0.0
    for h in headings:
        ratio = update_covered_mask(md, col, row, h, FOV_HORIZONTAL, max_range_px, NUM_RAYS)
    return ratio


def run_session(md: MapData, cfg: dict, start_px: tuple[int, int] | None = None,
                max_plans: int = 400) -> RunTrace:
    """Drive the exploration session headlessly to completion (or max_plans)."""
    H, W = md.pgm_array.shape
    resolution = md.resolution
    max_range_px = max(1, int(cfg["max_detection_range"] / resolution))

    session = VariantSession(md, cfg)
    if start_px is None:
        start_px = (W // 2, H // 2)
    robot_col, robot_row = session.nearest_start(*start_px)
    robot_x, robot_y = pixel_to_world(robot_col, robot_row, resolution,
                                      md.origin_x, md.origin_y, H)

    trace = RunTrace(resolution=resolution)
    trace.observe_count = np.zeros(md.free_mask.shape, dtype=np.int32)
    ratio = 0.0

    while trace.n_plans < max_plans:
        t0 = time.perf_counter()
        waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(robot_x, robot_y)
        trace.plan_times_s.append(time.perf_counter() - t0)

        if (no_frontiers and ratio >= cfg["exploration_completion_threshold"]) or not waypoints:
            trace.completed = True
            break

        trace.n_plans += 1
        trace.n_waypoints += len(waypoints)
        for wp in waypoints:
            if wp.geodesic_dist_px is not None:
                trace.selected_geodesic_px.append(wp.geodesic_dist_px)

        mid_replan = False
        for wp in waypoints:
            path = find_path(md.navigable_mask, (robot_col, robot_row), (wp.col, wp.row))
            if session.on_unreachable(wp, path):
                continue

            prev = path[0]
            for step in path[1:]:
                trace.path_length_m += math.hypot(step[0] - prev[0],
                                                  step[1] - prev[1]) * resolution
                if session.on_step(step, wp, waypoints):
                    robot_col, robot_row = step
                    robot_x, robot_y = pixel_to_world(step[0], step[1], resolution,
                                                      md.origin_x, md.origin_y, H)
                    mid_replan = True
                    break
                prev = step
            if mid_replan:
                break

            # Arrived: observe (rotate) and mark visited.
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = wp.x, wp.y
            ratio = _observe_at(md, wp.col, wp.row, max_range_px, trace.observe_count)
            session.on_arrive(wp)
            trace.visited_wp_px.append((wp.col, wp.row))
            trace.visited_wp_world.append((wp.x, wp.y))
            trace.n_visited += 1
            trace.coverage_curve.append((trace.path_length_m, ratio))

    trace.final_coverage = ratio
    return trace


# ==========================================================================
# Metrics
# ==========================================================================

@dataclass
class MetricSet:
    map_name: str
    # endpoint
    final_coverage: float
    completed: bool
    n_plans: int
    n_waypoints: int
    n_visited: int
    # efficiency (headline)
    path_length_m: float
    coverage_per_metre: float
    coverage_auc: float                 # area under coverage-vs-distance, normalised
    path_to_50pct_m: float | None
    path_to_90pct_m: float | None
    path_efficiency_ratio: float | None  # executed / NN-tour lower bound
    redundant_coverage_frac: float
    mean_leg_length_m: float
    max_leg_length_m: float
    # diagnostics (problem signatures)
    n_direction_reversals: int
    total_turning_angle_deg: float
    mean_geodesic_dist_selected_px: float | None
    # compute cost
    total_plan_time_s: float
    mean_plan_time_ms: float
    max_plan_time_ms: float


def _milestone(curve: list[tuple[float, float]], target: float) -> float | None:
    for path_m, cov in curve:
        if cov >= target:
            return path_m
    return None


def _coverage_auc(curve: list[tuple[float, float]]) -> float:
    """Area under coverage(distance), normalised by total distance -> [0,1].

    Higher means coverage is reached earlier (front-loaded). Trapezoidal over
    the (path_m, coverage) samples, divided by final path length so it is a
    dimensionless 'how soon' score comparable across maps of different size.
    """
    if len(curve) < 2:
        return curve[0][1] if curve else 0.0
    xs = [0.0] + [p for p, _ in curve]
    ys = [0.0] + [c for _, c in curve]
    total = xs[-1]
    if total <= 0:
        return ys[-1]
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return area / total


def _leg_lengths_m(wp_world: list[tuple[float, float]]) -> list[float]:
    return [math.hypot(wp_world[i][0] - wp_world[i - 1][0],
                       wp_world[i][1] - wp_world[i - 1][1])
            for i in range(1, len(wp_world))]


def _turning(wp_world: list[tuple[float, float]]) -> tuple[int, float]:
    """(#direction reversals, total absolute turning angle deg) over the
    visited-waypoint polyline. A criss-crossing (problem-A) tour turns a lot."""
    if len(wp_world) < 3:
        return 0, 0.0
    reversals = 0
    total = 0.0
    for i in range(1, len(wp_world) - 1):
        a, b, c = wp_world[i - 1], wp_world[i], wp_world[i + 1]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(cosang))
        total += ang
        if ang > 120.0:   # near-U-turn = backtracking
            reversals += 1
    return reversals, total


def _nn_tour_lower_bound_m(md: MapData, start_px: tuple[int, int],
                           wp_px: list[tuple[int, int]]) -> float | None:
    """Nearest-neighbour tour length over visited waypoints using BFS
    (navigable) distances, in metres. A cheap reference for
    path_efficiency_ratio = executed_path / this: > 1 means the executed run
    travelled further than an idealised NN tour (extra travel from mid-path
    replans, detours, re-approaches)."""
    if not wp_px:
        return None
    # Seed from the first visited waypoint (guaranteed navigable). The raw
    # start_px is the map centre, which on some maps (e.g. L_shape) lands in an
    # occupied cell -> all-inf BFS -> no reference. Anchoring on wp_px[0] keeps
    # the tour navigable and the ratio meaningful.
    remaining = list(wp_px[1:])
    cur = wp_px[0]
    total = 0.0
    while remaining:
        dm = navigable_distance_map(md.navigable_mask, cur[0], cur[1])
        reach = [w for w in remaining if dm[w[1], w[0]] != np.inf]
        if not reach:
            break
        nxt = min(reach, key=lambda w: dm[w[1], w[0]])
        total += float(dm[nxt[1], nxt[0]]) * md.resolution
        cur = nxt
        remaining.remove(nxt)
    return total if total > 0 else None


def compute_metrics(map_name: str, trace: RunTrace, md: MapData,
                    start_px: tuple[int, int]) -> MetricSet:
    curve = trace.coverage_curve
    legs = _leg_lengths_m(trace.visited_wp_world)
    reversals, turning = _turning(trace.visited_wp_world)

    free_total = int(np.sum(md.free_mask))
    observed = trace.observe_count
    redundant = 0.0
    if observed is not None and free_total > 0:
        n_observed_cells = int(np.sum(observed > 0))
        n_redundant = int(np.sum(observed > 1))
        redundant = (n_redundant / n_observed_cells) if n_observed_cells else 0.0

    nn_lb = _nn_tour_lower_bound_m(md, start_px, trace.visited_wp_px)
    eff = (trace.path_length_m / nn_lb) if nn_lb else None
    plan_ms = [t * 1000.0 for t in trace.plan_times_s]

    return MetricSet(
        map_name=map_name,
        final_coverage=round(trace.final_coverage, 4),
        completed=trace.completed,
        n_plans=trace.n_plans,
        n_waypoints=trace.n_waypoints,
        n_visited=trace.n_visited,
        path_length_m=round(trace.path_length_m, 2),
        coverage_per_metre=round(trace.final_coverage / trace.path_length_m, 5)
        if trace.path_length_m > 0 else 0.0,
        coverage_auc=round(_coverage_auc(curve), 4),
        path_to_50pct_m=_round(_milestone(curve, 0.50)),
        path_to_90pct_m=_round(_milestone(curve, 0.90)),
        path_efficiency_ratio=round(eff, 3) if eff else None,
        redundant_coverage_frac=round(redundant, 4),
        mean_leg_length_m=round(sum(legs) / len(legs), 2) if legs else 0.0,
        max_leg_length_m=round(max(legs), 2) if legs else 0.0,
        n_direction_reversals=reversals,
        total_turning_angle_deg=round(turning, 1),
        mean_geodesic_dist_selected_px=round(
            sum(trace.selected_geodesic_px) / len(trace.selected_geodesic_px), 1)
        if trace.selected_geodesic_px else None,
        total_plan_time_s=round(sum(trace.plan_times_s), 3),
        mean_plan_time_ms=round(sum(plan_ms) / len(plan_ms), 2) if plan_ms else 0.0,
        max_plan_time_ms=round(max(plan_ms), 2) if plan_ms else 0.0,
    )


def _round(v, nd=2):
    return round(v, nd) if v is not None else None


# ==========================================================================
# Driver
# ==========================================================================

def _start_px(md: MapData) -> tuple[int, int]:
    H, W = md.pgm_array.shape
    return (W // 2, H // 2)


def run_all(maps: list[str] | None = None,
            variant: str = "baseline") -> dict[str, MetricSet]:
    """Run the benchmark on the given maps (default: all) with the named scoring
    variant. Returns {map_name: MetricSet}. Import-safe entry point for pytest /
    before-after."""
    cfg = build_variant_config(variant)
    names = maps or (list(SYNTHETIC_BUILDERS) + ASSET_MAPS)
    results: dict[str, MetricSet] = {}
    for name in names:
        md = SYNTHETIC_BUILDERS[name]() if name in SYNTHETIC_BUILDERS else load_asset(name)
        start = _start_px(md)
        trace = run_session(md, cfg, start_px=start)
        results[name] = compute_metrics(name, trace, md, start)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────

# Every captured metric, grouped for readability. (group_label, [(attr, label)]).
# Covers all fields on MetricSet so the table is complete — nothing captured is
# silently dropped from the comparison.
_TABLE_GROUPS = [
    ("endpoint", [
        ("final_coverage", "cov"),
        ("completed", "done"),
        ("n_plans", "n_plans"),
        ("n_waypoints", "n_wp_plan"),
        ("n_visited", "n_wp_seen"),
    ]),
    ("efficiency", [
        ("path_length_m", "path_m"),
        ("coverage_per_metre", "cov/m"),
        ("coverage_auc", "auc"),
        ("path_to_50pct_m", "path@50%"),
        ("path_to_90pct_m", "path@90%"),
        ("path_efficiency_ratio", "eff"),
        ("redundant_coverage_frac", "redund"),
    ]),
    ("routing diagnostics", [
        ("mean_leg_length_m", "leg_avg"),
        ("max_leg_length_m", "leg_max"),
        ("n_direction_reversals", "revers"),
        ("total_turning_angle_deg", "turn"),
        ("mean_geodesic_dist_selected_px", "geo_sel"),
    ]),
    ("compute cost", [
        ("mean_plan_time_ms", "plan_ms_avg"),
        ("max_plan_time_ms", "plan_ms_max"),
        ("total_plan_time_s", "plan_s_tot"),
    ]),
]

# Flat view for the single-run table.
_TABLE_ROWS = [row for _, rows in _TABLE_GROUPS for row in rows]


def print_table(results: dict[str, MetricSet]) -> None:
    names = list(results)
    w = max(12, max(len(n) for n in names) + 1)
    header = f"{'metric':<26}" + "".join(f"{n:>{w}}" for n in names)
    print(header)
    print("-" * len(header))
    for attr, label in _TABLE_ROWS:
        cells = []
        for n in names:
            v = getattr(results[n], attr)
            cells.append(f"{'' if v is None else v:>{w}}")
        print(f"{label:<26}" + "".join(cells))


# Direction each metric should move for the run to be "better", used to mark
# improvement/regression in the comparison. None = neutral (context only).
_METRIC_BETTER = {
    "final_coverage": +1, "completed": None, "n_plans": None, "n_waypoints": None,
    "n_visited": None,
    "path_length_m": -1, "coverage_per_metre": +1, "coverage_auc": +1,
    "path_to_50pct_m": -1, "path_to_90pct_m": -1, "path_efficiency_ratio": -1,
    "redundant_coverage_frac": -1,
    "mean_leg_length_m": -1, "max_leg_length_m": -1, "n_direction_reversals": -1,
    "total_turning_angle_deg": -1, "mean_geodesic_dist_selected_px": None,
    "mean_plan_time_ms": -1, "max_plan_time_ms": -1, "total_plan_time_s": -1,
}


def print_comparison(out_dir: Path, variant_order: list[str] | None = None) -> None:
    """Load every metrics_<variant>.json in out_dir and print, per map, a table
    with variants as columns and metrics as rows. The first variant (baseline if
    present) is the reference; other columns show the % change vs it and a
    +/-/. marker for better/worse/neutral. This is the full ablation reader."""
    files = sorted(out_dir.glob("metrics_*.json"))
    if not files:
        print(f"no metrics_*.json found in {out_dir}")
        return
    loaded = {f.stem.replace("metrics_", ""): json.loads(f.read_text()) for f in files}

    # Order: requested order first (those present), then any extras alphabetically.
    order = [v for v in (variant_order or []) if v in loaded]
    order += [v for v in sorted(loaded) if v not in order]
    # Put baseline first as the reference if it exists.
    if "baseline" in order:
        order.remove("baseline")
        order.insert(0, "baseline")
    ref = order[0]

    maps = list(loaded[ref])
    for m in maps:
        print(f"\n================ {m}  (ref: {ref}) ================")
        colw = 22
        hdr = f"{'metric':<24}" + "".join(f"{v:>{colw}}" for v in order)
        print(hdr)
        print("-" * len(hdr))
        for group_label, rows in _TABLE_GROUPS:
            print(f"-- {group_label} " + "-" * (len(hdr) - len(group_label) - 4))
            for attr, label in rows:
                cells = []
                base_v = loaded[ref].get(m, {}).get(attr)
                for v in order:
                    val = loaded[v].get(m, {}).get(attr)
                    if val is None:
                        cells.append(f"{'—':>{colw}}")
                        continue
                    if v == ref or not isinstance(val, (int, float)) \
                            or not isinstance(base_v, (int, float)) or base_v == 0:
                        cells.append(f"{val:>{colw}.4g}" if isinstance(val, (int, float))
                                     else f"{str(val):>{colw}}")
                        continue
                    pct = (val - base_v) / abs(base_v) * 100.0
                    better = _METRIC_BETTER.get(attr)
                    mark = "." if better is None else (
                        "+" if (val - base_v) * better > 0 else
                        ("-" if (val - base_v) * better < 0 else "."))
                    cells.append(f"{val:>10.4g} {pct:+6.1f}%{mark}")
                print(f"{label:<24}" + "".join(cells))
    print("\nmarks: + better than ref, - worse, . neutral/context")


def visualise_waypoints(map_name: str, out_path: Path, variant: str = "baseline") -> None:
    """Render the final visited-waypoint positions and visit order on one map.

    variant labels the scoring logic in force (e.g. 'baseline', 'A_sequential')
    so the title and file name identify which version produced the tour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = build_variant_config(variant)
    md = SYNTHETIC_BUILDERS[map_name]() if map_name in SYNTHETIC_BUILDERS else load_asset(map_name)
    start = _start_px(md)
    trace = run_session(md, cfg, start_px=start)

    fig, ax = plt.subplots(figsize=(11, 8))
    H, W = md.free_mask.shape
    img = np.ones((H, W, 3))
    img[md.occupied_mask] = (0.15, 0.15, 0.15)
    img[md.unknown_mask] = (0.6, 0.6, 0.6)
    img[md.covered_mask & md.free_mask] = (0.75, 0.92, 0.75)
    ax.imshow(img, origin="upper")

    wp = trace.visited_wp_px
    if wp:
        ax.plot([c for c, _ in wp], [r for _, r in wp],
                "-", color="#1f6feb", lw=1.4, alpha=0.8, zorder=2)
        for i in range(1, len(wp)):
            ax.annotate("", xy=wp[i], xytext=wp[i - 1],
                        arrowprops=dict(arrowstyle="->", color="#1f6feb", lw=1.2),
                        zorder=2)
        for i, (c, r) in enumerate(wp):
            ax.plot(c, r, "o", color="#1f6feb", ms=9, zorder=3)
            ax.annotate(str(i + 1), (c, r), color="white", fontsize=7,
                        ha="center", va="center", zorder=4)
    ax.plot(*start, "*", color="crimson", ms=18, zorder=5, label="start")
    ax.set_title(f"{map_name} [{variant}]: visited waypoints in order "
                 f"(n={len(wp)}, path={trace.path_length_m:.1f} m, "
                 f"cov={trace.final_coverage:.1%})")
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"waypoint visualisation -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="*", default=None,
                    help="subset of maps to run (default: all)")
    ap.add_argument("--variant", type=str, default="baseline",
                    choices=sorted(VARIANTS),
                    help="scoring variant to run; selects distance_model / "
                         "scoring_sequential overrides and tags the output file "
                         "names and visualisation title so each ablation stage is "
                         "self-identifying.")
    ap.add_argument("--viz", type=str, default="lab_05",
                    help="map to render waypoint order for (default: lab_05; "
                         "pass 'none' to skip)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output directory (default: benchmark_waypoint_scoring_out/)")
    ap.add_argument("--compare", action="store_true",
                    help="don't run: load every metrics_<variant>.json in out-dir "
                         "and print the full per-map variant comparison, then exit.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / "benchmark_waypoint_scoring_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        print_comparison(out_dir, variant_order=list(VARIANTS))
        return

    t0 = time.perf_counter()
    results = run_all(args.maps, variant=args.variant)
    print_table(results)
    print(f"\ntotal benchmark wall time: {time.perf_counter() - t0:.1f} s")

    # Merge into any existing metrics file rather than overwrite: a partial run
    # (--maps subset) must not clobber other maps' results already saved for this
    # variant. Only the maps just run are updated.
    json_path = out_dir / f"metrics_{args.variant}.json"
    merged: dict = {}
    if json_path.exists():
        try:
            merged = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update({n: asdict(m) for n, m in results.items()})
    json_path.write_text(json.dumps(merged, indent=2))
    print(f"results JSON -> {json_path}  (maps: {', '.join(results)})")

    if args.viz and args.viz.lower() != "none":
        out = out_dir / f"waypoints_{args.viz}_{args.variant}.png"
        visualise_waypoints(args.viz, out, variant=args.variant)


if __name__ == "__main__":
    main()
