#!/usr/bin/env python3
"""
A/B harness , `visited_candidates` disc removal (issues A + C).

Validates that removing the wall-blind Euclidean `visited_candidates` disc (which
sterilises the candidate pool and causes premature termination at ~13.6% coverage
on lab_ghent) and relying solely on the FOV/occlusion-aware `covered_mask` lets
exploration run to the completion threshold WITHOUT regressing path length,
redundancy, or compute time.

=============================================================================
TWO STRICTLY SEPARATED LAYERS  (read before editing)
=============================================================================
1. SYSTEM-UNDER-TEST layer  (may import exploration.*)
   Runs the *production* planner as a black box in two arms:
     - Arm A "baseline": stock ExplorationSession (Euclidean disc active).
     - Arm B "patched" : _PatchedSession, covered_mask-only, reproducing the
       exact behaviour the real code change will have, WITHOUT editing
       exploration/. It overrides on_arrive to a no-op and calls plan_waypoints
       with an empty visited set, plus a safety-net probe for empty-plan-with-
       frontiers. This lets us prove the fix behaviourally before touching prod.
   Production update_covered_mask / build_map_data advance the SUT's OWN internal
   state (that is the SUT's machinery), they are NOT used to score.

2. REFERENCE / SCORER layer  (MUST NOT import exploration.*)
   Independent ground truth: own PGM/YAML loader, own ray-cast FOV coverage
   model, own path-length + nearest-neighbour lower bound, own metrics. The
   verdict is computed by code that is NOT the code under test.

=============================================================================
SLAM / PROGRESSIVE-REVEAL  (required, a static map has no frontiers)
=============================================================================
A fully-known map has an empty unknown_mask -> zero frontier gain -> the
premature-stop bug and the safety net are never exercised. So the harness
simulates SLAM the way the node does (ros2_exploration_node._on_map: unknown
cells -> p_occ 0.5):
  - The reference layer holds the FULL ground-truth occupancy grid (true world).
  - The SUT is fed a PARTIALLY-REVEALED p_occ: unobserved cells = 0.5 (unknown),
    seeded with a small revealed disc around the start.
  - On each arrival+spin, the independent model ray-casts the TRUE world to reveal
    the cells the camera actually observes; those flip unknown->free/occupied, are
    spliced into the SUT p_occ, MapData is rebuilt and bound via session.set_map()
    (mirroring _on_map). covered_mask carries forward.
  - Termination mirrors the node: done on (no_frontiers and ratio>=threshold); an
    empty plan while frontiers remain is the PREMATURE-STOP / safety-net event we
    count (Arm A is expected to hit it; Arm B must not).

Usage:
  python test_FOV_visited_candidates.py --maps lab_ghent lab_05 warehouse_amazon
  python test_FOV_visited_candidates.py --maps lab_ghent --no-viz
"""
from __future__ import annotations

import argparse
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
OUT_DIR = Path(__file__).resolve().parent / "test_FOV_visited_candidates_out"

# ---------------------------------------------------------------------------
# Planner parameters (kept in ONE place; must match config so the SUT is driven
# the same way production is). Deliberately hard-copied here rather than read via
# demo_robot so the reference layer never transitively imports exploration.*.
# ---------------------------------------------------------------------------
MAX_DETECTION_M = 6.0
FOV_HORIZONTAL = 87.0
NUM_RAYS = 360
OBSERVATION_INCREMENT = 30.0
SAMPLING_STEP_M = 3.0
FRONTIER_WEIGHT = 1.0
COVERAGE_WEIGHT = 1.0
COMPLETION_THRESHOLD = 0.90
INFLATION_M = 0.4
FREE_THRESH_DEFAULT = 0.196
OCC_THRESH_DEFAULT = 0.65


def planner_config() -> dict:
    return {
        "max_detection_range": MAX_DETECTION_M,
        "fov_horizontal": FOV_HORIZONTAL,
        "observation_rotation_increment": OBSERVATION_INCREMENT,
        "sampling_step_m": SAMPLING_STEP_M,
        "num_rays": NUM_RAYS,
        "frontier_weight": FRONTIER_WEIGHT,
        "coverage_weight": COVERAGE_WEIGHT,
        "exploration_completion_threshold": COMPLETION_THRESHOLD,
        "planner_coverage_warning_threshold": COMPLETION_THRESHOLD,
    }


# ===========================================================================
# REFERENCE / SCORER LAYER , NO exploration.* IMPORTS ALLOWED BELOW THIS LINE
# ===========================================================================

@dataclass
class TrueWorld:
    """Ground-truth occupancy, known only to the scorer. p_occ in [0,1]:
    0=free, 1=occupied. (No 0.5 here , this is the full truth.)"""
    p_occ: np.ndarray          # (H, W) float in [0,1]
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
    """Independent PGM+YAML loader (PIL+numpy+yaml). Mirrors the ROS map_server
    convention but shares NO code with exploration.load_map."""
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
    """Return a boolean mask of TRUE-WORLD cells the camera at (col0,row0) with the
    given heading+FOV actually observes (stops at occupied/OOB). Independent
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
    # alive = cells strictly before the first block along each ray
    cum = np.cumsum(blocked, axis=1)
    alive = ((cum - blocked) == 0) & (~blocked)
    seen = np.zeros((H, W), dtype=bool)
    seen[rr[alive], cc[alive]] = True
    # also credit the first blocking cell as "observed" (we saw the wall/edge)
    first = (cum == 1) & blocked & inb
    seen[rr[first], cc[first]] = True
    return seen


def _angular_diff(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _select_headings(world, col, row, already_covered, max_r, n_rays, fov, incr):
    """Independent, realistic heading selection for one waypoint stop.

    A real robot does NOT spin a full blind 360° at every stop , it rotates only
    to the headings needed to observe cells it has not yet covered (the same
    principle as the planner's compute_headings_for_waypoint, reimplemented here
    with no exploration.* import). Casting only these headings is what keeps the
    reference coverage model consistent with a physical FOV-limited spin, and
    avoids the phantom 'full-360 reveal' that double-counts coverage.

    Returns the list of heading angles (deg) the robot rotates through.
    """
    # Cells this waypoint could newly observe (360° line-of-sight, minus what is
    # already covered) , the bearings we must point the camera at.
    los = _cast_fov(world, col, row, 0.0, 360.0, max_r, n_rays)  # full LOS disc
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


def _observe_spin(world, col, row, already_covered, max_r, n_rays, fov, incr):
    """Union of camera footprints over the REALISTIC spin (only the headings the
    robot actually rotates to, per _select_headings). Independent ground-truth
    observation for one waypoint stop."""
    seen = np.zeros(world.p_occ.shape, dtype=bool)
    for h in _select_headings(world, col, row, already_covered, max_r, n_rays, fov, incr):
        seen |= _cast_fov(world, col, row, h, fov, max_r, n_rays)
    return seen


@dataclass
class RefTrace:
    """Independent scoring accumulator for one arm on one map."""
    resolution: float
    observe_count: np.ndarray = None           # per-cell observation count (true world)
    covered: np.ndarray = None                 # bool, ref ground-truth coverage
    path_length_m: float = 0.0
    visited_world: list = field(default_factory=list)
    visited_px: list = field(default_factory=list)
    coverage_curve: list = field(default_factory=list)   # (dist_m, coverage_frac)
    frames: list = field(default_factory=list)           # covered-mask snapshots
    n_plans: int = 0
    n_visited: int = 0
    premature_stops: int = 0        # empty plan while ref frontiers remain
    pool_exhausted_events: int = 0  # SUT returned empty candidate pool
    completed: bool = False
    final_coverage: float = 0.0
    plan_time_s: float = 0.0


def _coverage_frac(trace: RefTrace, world: TrueWorld) -> float:
    free = world.free
    denom = int(free.sum())
    return 0.0 if denom == 0 else float((trace.covered & free).sum()) / denom


def _nn_lower_bound_m(points_world: list) -> float:
    """Independent nearest-neighbour tour length (Euclidean) over visited
    waypoints , a cheap lower-bound reference for path efficiency."""
    if len(points_world) < 2:
        return 0.0
    pts = list(points_world)
    cur = pts.pop(0)
    total = 0.0
    while pts:
        j = min(range(len(pts)),
                key=lambda i: (pts[i][0] - cur[0]) ** 2 + (pts[i][1] - cur[1]) ** 2)
        total += math.hypot(pts[j][0] - cur[0], pts[j][1] - cur[1])
        cur = pts.pop(j)
    return total


def compute_metrics(name: str, arm: str, trace: RefTrace, world: TrueWorld) -> dict:
    free = world.free
    denom = int(free.sum())
    observed = trace.observe_count > 0
    redundant = trace.observe_count > 1
    coverage = trace.final_coverage
    holes = int((free & ~observed).sum())          # free cells never observed
    auc = 0.0
    if len(trace.coverage_curve) >= 2 and trace.path_length_m > 0:
        xs = [d for d, _ in trace.coverage_curve]
        ys = [c for _, c in trace.coverage_curve]
        for i in range(1, len(xs)):
            auc += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
        auc /= trace.path_length_m
    nn_lb = _nn_lower_bound_m(trace.visited_world)
    eff = (trace.path_length_m / nn_lb) if nn_lb > 0 else float("nan")
    return {
        "map": name,
        "arm": arm,
        "completed": bool(trace.completed),
        "final_coverage": round(coverage, 4),
        "reached_threshold": bool(coverage >= COMPLETION_THRESHOLD),
        "path_length_m": round(trace.path_length_m, 2),
        "coverage_per_m": round(coverage / trace.path_length_m, 5) if trace.path_length_m else 0.0,
        "coverage_auc": round(auc, 4),
        "n_plans": trace.n_plans,
        "n_visited": trace.n_visited,
        "premature_stops": trace.premature_stops,
        "pool_exhausted_events": trace.pool_exhausted_events,
        "coverage_holes_cells": holes,
        "coverage_holes_frac": round(holes / denom, 4) if denom else 0.0,
        "redundant_coverage_frac": round(int(redundant.sum()) / max(1, int(observed.sum())), 4),
        "path_efficiency_vs_nn": round(eff, 3) if not math.isnan(eff) else None,
        "total_plan_time_s": round(trace.plan_time_s, 3),
        "mean_plan_time_ms": round(1000 * trace.plan_time_s / max(1, trace.n_plans), 2),
    }


# ===========================================================================
# SYSTEM-UNDER-TEST LAYER  (imports exploration.* , production as black box)
# ===========================================================================
_EXPLORATION_PKG = _ROOT / "ros2_ws" / "src" / "navigation" / "exploration"
_TESTS_DIR = _EXPLORATION_PKG / "tests"
sys.path.insert(0, str(_EXPLORATION_PKG))
sys.path.insert(0, str(_TESTS_DIR))

from exploration.explore_costmap_map import (  # noqa: E402
    build_map_data, plan_waypoints, pixel_to_world,
)
from exploration.execution_strategy import ExplorationSession, MidPathReplanner  # noqa: E402
from demo_robot import find_path  # noqa: E402  (pure BFS pathfinder, no scoring)


class _PatchedSession(ExplorationSession):
    """Arm B / D black box: covered_mask-only (no Euclidean disc). on_arrive is a
    no-op and planning never passes a visited set. Arm D adds execute-1-replan
    cadence at the driver level (see run_arm), not here."""

    def on_arrive(self, wp) -> None:  # noqa: D401
        return  # covered_mask (grown by the spin) is the only exclusion mechanism

    def plan_waypoints_raw(self, robot_x: float, robot_y: float):
        result = plan_waypoints(self._md, self._config, robot_x, robot_y,
                                visited_candidates=None)  # empty: no disc filter
        self._replanner = MidPathReplanner(self.visit_radius_px)
        return result


# FOV-aware mark threshold: a candidate is "visited" once this fraction of its
# full visible footprint is already covered (relative, tolerates small residual
# gaps behind occluders). Swept across 3 maps: below 0.95 the mark gets too eager
# on mostly-known maps (lab_05) and REINTRODUCES the premature-stop bug (drops to
# ~69% cov, premature=1); 0.95 and 0.99 are both safe. 0.95 is the loosest safe
# value (tolerates residual gaps without eager sterilisation).
OBSERVED_FRAC = 0.95


class _FovGatedSession(ExplorationSession):
    """Arm C black box: keep visited_candidates as a re-visit suppressor, but mark
    a candidate visited ONLY when its own coverage cells are already observed
    (in covered_mask) , an FOV/occlusion-aware mark, not a blind Euclidean disc.
    This retains the anti-overlap benefit the disc actually provided while
    removing the wall-blind premature-stop failure. Reproduces the 'option 2'
    fix behaviourally without editing exploration/."""

    def _mark(self, pos):
        # Override the Euclidean-disc _mark with an FOV/occlusion-aware test: mark
        # a candidate once the camera has observed >= OBSERVED_FRAC of the cells it
        # could ever see (relative, so a tiny residual sliver behind a shelf edge
        # doesn't keep a near-saturated viewpoint alive forever). compute_visibility
        # returns only the STILL-UNCOVERED cells; the full footprint (denominator)
        # is the same cast with covered_mask ignored.
        import numpy as _np
        from exploration.explore_costmap_map import compute_visibility
        max_range_px = max(1, int(self._config["max_detection_range"] / self._md.resolution))
        nr = self._config.get("num_rays", 360)
        real_covered = self._md.covered_mask
        empty = _np.zeros_like(real_covered)
        for c in self._candidates:
            remaining, _ = compute_visibility(c, self._md, max_range_px, nr)
            self._md.covered_mask = empty          # full footprint (ignore covered)
            full, _ = compute_visibility(c, self._md, max_range_px, nr)
            self._md.covered_mask = real_covered
            if not full:
                self.visited_candidates.add(c)      # boxed in / nothing to see
                continue
            observed_frac = 1.0 - len(remaining) / len(full)
            if observed_frac >= OBSERVED_FRAC:
                self.visited_candidates.add(c)


def _make_sut_mapdata(sut_p_occ, world: TrueWorld, covered_mask):
    """Build a production MapData from the SUT's partially-revealed p_occ."""
    return build_map_data(
        p_occ=sut_p_occ,
        resolution=world.resolution,
        origin_x=world.origin_x,
        origin_y=world.origin_y,
        inflation_radius_m=INFLATION_M,
        free_thresh=world.free_thresh,
        occ_thresh=world.occ_thresh,
        covered_mask=covered_mask,
        pgm_array=(sut_p_occ * 255).astype(np.uint8),
    )


def _reveal_disc(sut_p_occ, world, col, row, radius_px):
    """Seed reveal: copy the true world into an initial disc around the start."""
    H, W = world.p_occ.shape
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - col) ** 2 + (yy - row) ** 2 <= radius_px ** 2
    sut_p_occ[disc] = world.p_occ[disc]


def _advance_sut_covered(md, col, row, max_range_px):
    """Grow the SUT's OWN covered_mask via the production spin , this is the SUT's
    internal machinery, NOT scoring. Uses the production caster on purpose so Arm B
    behaves exactly as the shipped fix will."""
    from exploration.explore_costmap_map import (
        compute_visibility, compute_headings_for_waypoint, update_covered_mask,
    )
    cov_now, _ = compute_visibility((col, row), md, max_range_px, NUM_RAYS)
    headings = compute_headings_for_waypoint(
        col, row, cov_now, fov_deg=FOV_HORIZONTAL, increment_deg=OBSERVATION_INCREMENT)
    if not headings:
        headings = [0.0]
    for h in headings:
        update_covered_mask(md, col, row, h, FOV_HORIZONTAL, max_range_px, NUM_RAYS)


# Arm registry. Each arm is a dict of composable knobs:
#   session      : factory for the ExplorationSession variant (disc behaviour)
#   replan_k     : drain this many waypoints per plan then replan (None = whole
#                  plan; 1 = execute-1-replan; K = replan-every-K cadence, TODO §11)
#   no_progress  : stop after this many consecutive waypoints add < NO_PROGRESS_EPS
#                  new coverage (None = disabled); the 'stop at the knee' lever
#   step_m       : sampling_step_m override (None = config default 3.0)
NO_PROGRESS_EPS = 0.005          # 0.5% of free cells
_DISC = lambda md, cfg: ExplorationSession(md, cfg)          # noqa: E731
_NODISC = lambda md, cfg: _PatchedSession(md, cfg)           # noqa: E731
_FOV = lambda md, cfg: _FovGatedSession(md, cfg)             # noqa: E731
ARMS = {
    # originals
    "baseline": {"session": _DISC},                          # production today
    "naive":    {"session": _NODISC},                        # disc removed
    "fovgated": {"session": _FOV},                           # FOV-aware disc
    "replan1":  {"session": _NODISC, "replan_k": 1},         # execute-1-replan
    # single levers
    "stop":     {"session": _DISC, "no_progress": 3},        # no-progress stop only
    "step45":   {"session": _DISC, "step_m": 4.5},           # bigger sampling step
    "replan3":  {"session": _FOV, "replan_k": 3},            # replan-every-3 + FOV
    # combined candidate: FOV-mark + no-progress stop + replan-every-3
    "combo":    {"session": _FOV, "replan_k": 3, "no_progress": 3},
    "combo_stop": {"session": _FOV, "no_progress": 3},       # FOV + stop, whole plan
    # CHOSEN production candidate: FOV-aware mark + bigger sampling step.
    "chosen":   {"session": _FOV, "step_m": 4.5},
}


def run_arm(name: str, world: TrueWorld, arm: str, capture_frames: bool,
            max_plans: int = 400, known_map: bool = False) -> RefTrace:
    """Drive one arm end-to-end. Default: SLAM/progressive-reveal (SUT sees only
    revealed cells). known_map=True: SUT starts with the FULL map known (no
    unknown cells, no frontiers) , a control that validates this harness against
    the production planner (byte-identical to benchmark B_expdecay)."""
    spec = ARMS[arm]
    make_session = spec["session"]
    replan_k = spec.get("replan_k")
    no_progress = spec.get("no_progress")
    H, W = world.p_occ.shape
    res = world.resolution
    max_range_px = max(1, int(MAX_DETECTION_M / res))
    cfg = planner_config()
    if spec.get("step_m"):
        cfg["sampling_step_m"] = spec["step_m"]

    # SLAM: start blind (unknown=0.5) except a seed disc. known_map: full truth.
    if known_map:
        sut_p_occ = world.p_occ.copy()
    else:
        sut_p_occ = np.full((H, W), 0.5, dtype=np.float64)
    covered = np.zeros((H, W), dtype=bool)

    trace = RefTrace(resolution=res)
    trace.observe_count = np.zeros((H, W), dtype=np.int32)
    trace.covered = np.zeros((H, W), dtype=bool)

    # SLAM: seed a revealed disc around the physical start (raw centre projected
    # to nearest free true-world cell) BEFORE the first plan, so there is
    # navigable ground to stand on. Known-map: full truth already present.
    if not known_map:
        sc, sr = W // 2, H // 2
        if not world.free[sr, sc]:
            fy, fx = np.where(world.free)
            j = int(np.argmin((fx - sc) ** 2 + (fy - sr) ** 2))
            sc, sr = int(fx[j]), int(fy[j])
        _reveal_disc(sut_p_occ, world, sc, sr, max_range_px)

    # Start as the reference does: nearest_start(W//2, H//2) on the raw centre.
    md = _make_sut_mapdata(sut_p_occ, world, covered)
    session = make_session(md, cfg)
    robot_col, robot_row = session.nearest_start(W // 2, H // 2)
    robot_x, robot_y = pixel_to_world(robot_col, robot_row, res, world.origin_x, world.origin_y, H)

    def _true_frontiers_remain() -> bool:
        """Independent frontier test: any true-free cell still unknown to the SUT."""
        known = sut_p_occ != 0.5
        return bool((world.free & ~known).any())

    def _observe(col, row):
        """Independent arrival observation: reveal + score coverage from the
        REALISTIC FOV-limited spin (only the headings the robot rotates through,
        chosen against cells the reference has not yet covered)."""
        seen = _observe_spin(world, col, row, trace.covered, max_range_px, NUM_RAYS,
                             FOV_HORIZONTAL, OBSERVATION_INCREMENT)
        newly = seen & (sut_p_occ == 0.5)         # reveal unknown -> true value
        sut_p_occ[newly] = world.p_occ[newly]
        seen_free = seen & world.free             # score coverage independently
        trace.observe_count[seen_free] += 1
        trace.covered |= seen_free

    stale_streak = 0                      # consecutive low-gain arrivals (no-progress stop)
    stop_requested = False
    while trace.n_plans < max_plans and not stop_requested:
        t0 = time.perf_counter()
        waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(robot_x, robot_y)
        trace.plan_time_s += time.perf_counter() - t0

        ref_frontiers = _true_frontiers_remain()
        ref_cov = _coverage_frac(trace, world)

        if not waypoints:
            # Empty plan. A stop is PREMATURE only if the planner gave up while it
            # still judged itself incomplete: the SUT's own coverage ratio (which
            # it uses for the completion decision) is below threshold AND real
            # frontiers remain. Comparing the independent all-free coverage to the
            # achievable-based threshold would be an unfair denominator mismatch
            # (achievable < all-free), so we gate on the SUT `ratio` here.
            if ref_frontiers and ratio < COMPLETION_THRESHOLD:
                trace.premature_stops += 1
                trace.pool_exhausted_events += 1
            trace.completed = ratio >= COMPLETION_THRESHOLD or not ref_frontiers
            break

        if no_frontiers and ratio >= COMPLETION_THRESHOLD:
            trace.completed = True
            break

        trace.n_plans += 1
        mid_replan = False
        wps_this_plan = 0
        for wp in waypoints:
            path = find_path(md.navigable_mask, (robot_col, robot_row), (wp.col, wp.row))
            if session.on_unreachable(wp, path):
                continue
            prev = path[0]
            for step in path[1:]:
                trace.path_length_m += math.hypot(step[0] - prev[0],
                                                  step[1] - prev[1]) * res
                if session.on_step(step, wp, waypoints):
                    robot_col, robot_row = step
                    robot_x, robot_y = pixel_to_world(step[0], step[1], res,
                                                      world.origin_x, world.origin_y, H)
                    mid_replan = True
                    break
                prev = step
            if mid_replan:
                break

            # Arrived: independent observation (reveals SLAM + scores coverage),
            # then advance the SUT's own covered_mask via the production spin.
            robot_col, robot_row = wp.col, wp.row
            robot_x, robot_y = pixel_to_world(wp.col, wp.row, res,
                                              world.origin_x, world.origin_y, H)
            cov_before = _coverage_frac(trace, world)
            _observe(wp.col, wp.row)
            _advance_sut_covered(md, wp.col, wp.row, max_range_px)
            session.on_arrive(wp)
            cov_after = _coverage_frac(trace, world)
            trace.visited_px.append((wp.col, wp.row))
            trace.visited_world.append((robot_x, robot_y))
            trace.n_visited += 1
            wps_this_plan += 1
            trace.coverage_curve.append((trace.path_length_m, cov_after))
            if capture_frames:
                trace.frames.append(trace.covered.copy())

            # No-progress stop lever: count consecutive arrivals that add < eps
            # new coverage; stop once the streak reaches the threshold. K
            # consecutive near-zero-gain arrivals means the robot is churning
            # over already-seen ground (the flat tail / phantom-frontier chase),
            # so we cut it regardless of nominal frontier flags.
            if no_progress is not None:
                if cov_after - cov_before < NO_PROGRESS_EPS:
                    stale_streak += 1
                else:
                    stale_streak = 0
                if stale_streak >= no_progress:
                    trace.completed = True
                    stop_requested = True
                    break

            # replan_k cadence: stop draining after K arrivals so the next plan is
            # scored against the freshly-grown covered_mask (K=1 -> execute-1).
            if replan_k is not None and wps_this_plan >= replan_k:
                break

        # SLAM update: rebuild MapData from the newly-revealed p_occ, bind via
        # set_map (mirrors node _on_map) ONLY when the revealed map actually
        # changed. In known-map mode nothing is ever revealed, so , like the
        # production node, which rebuilds only on a new /map , we must NOT rebind
        # every cycle (that regenerates candidates and perturbs the plan). We keep
        # the same md and only refresh its covered_mask reference.
        if not known_map:
            md = _make_sut_mapdata(sut_p_occ, world, md.covered_mask)
            session.set_map(md)

    trace.final_coverage = _coverage_frac(trace, world)
    return trace


# ===========================================================================
# Visuals (matplotlib) , independent covered mask + tours + curves
# ===========================================================================

def render(name: str, world: TrueWorld, traces: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bg = np.where(world.occupied, 0.0, np.where(world.free, 1.0, 0.5))
    # Tag output files with the arm-set so different comparison runs don't clobber
    # each other's reference images (e.g. a 4-arm sweep vs a baseline/chosen pair).
    tag = "_" + "-".join(sorted(traces))

    # 1. tours side by side
    fig, axes = plt.subplots(1, len(traces), figsize=(7 * len(traces), 6))
    if len(traces) == 1:
        axes = [axes]
    for ax, (arm, tr) in zip(axes, traces.items()):
        ax.imshow(bg, cmap="gray", origin="upper")
        cov = np.ma.masked_where(~tr.covered, tr.covered)
        ax.imshow(cov, cmap="autumn", alpha=0.45, origin="upper")
        if tr.visited_px:
            xs = [p[0] for p in tr.visited_px]
            ys = [p[1] for p in tr.visited_px]
            ax.plot(xs, ys, "-o", color="deepskyblue", ms=4, lw=1)
            ax.plot(xs[0], ys[0], "s", color="lime", ms=9, label="start")
        ax.set_title(f"{name} , {arm}\ncov={tr.final_coverage:.1%}  "
                     f"wps={tr.n_visited}  premature={tr.premature_stops}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_tours{tag}.png", dpi=110)
    plt.close(fig)

    # 2. coverage-vs-distance curve, both arms
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, tr in traces.items():
        if tr.coverage_curve:
            xs = [d for d, _ in tr.coverage_curve]
            ys = [c for _, c in tr.coverage_curve]
            ax.plot(xs, ys, "-o", ms=3, label=f"{arm} (final {tr.final_coverage:.1%})")
    ax.axhline(COMPLETION_THRESHOLD, ls="--", color="gray", label="completion threshold")
    ax.set_xlabel("path length (m)")
    ax.set_ylabel("coverage fraction")
    ax.set_title(f"{name} , coverage vs distance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_coverage_curve{tag}.png", dpi=110)
    plt.close(fig)

    # 3. coverage-progression small multiples (per arm)
    for arm, tr in traces.items():
        if not tr.frames:
            continue
        n = len(tr.frames)
        idxs = sorted(set(np.linspace(0, n - 1, min(n, 8)).astype(int)))
        fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.2))
        if len(idxs) == 1:
            axes = [axes]
        for ax, k in zip(axes, idxs):
            ax.imshow(bg, cmap="gray", origin="upper")
            cov = np.ma.masked_where(~tr.frames[k], tr.frames[k])
            ax.imshow(cov, cmap="autumn", alpha=0.5, origin="upper")
            # overlay the tour driven up to and including this frame's waypoint
            pts = tr.visited_px[:k + 1]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "-", color="deepskyblue", lw=0.8)
                ax.plot(xs, ys, "o", color="deepskyblue", ms=2.5)
                ax.plot(xs[0], ys[0], "s", color="lime", ms=6)      # start
                ax.plot(xs[-1], ys[-1], "*", color="magenta", ms=10)  # current
            frac = (tr.frames[k] & world.free).sum() / max(1, world.free.sum())
            ax.set_title(f"wp {k + 1}\n{frac:.0%}")
            ax.axis("off")
        fig.suptitle(f"{name} , {arm} coverage progression")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}_{arm}_progression.png", dpi=100)
        plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=["lab_ghent", "lab_05", "warehouse_amazon"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help=f"subset of {list(ARMS)}")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--max-plans", type=int, default=400)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name in args.maps:
        world = load_true_world(name)
        print(f"\n=== {name}  ({world.p_occ.shape[1]}x{world.p_occ.shape[0]} px, "
              f"res {world.resolution}) ===")
        traces = {}
        for arm in args.arms:
            tr = run_arm(name, world, arm=arm,
                         capture_frames=not args.no_viz, max_plans=args.max_plans)
            traces[arm] = tr
            m = compute_metrics(name, arm, tr, world)
            all_metrics.append(m)
            print(f"  [{arm:9s}] cov={m['final_coverage']:.1%}  "
                  f"reached_thr={m['reached_threshold']}  wps={m['n_visited']}  "
                  f"path={m['path_length_m']}m  cov/m={m['coverage_per_m']:.4f}  "
                  f"redund={m['redundant_coverage_frac']:.0%}  "
                  f"premature={m['premature_stops']}  "
                  f"holes={m['coverage_holes_frac']:.1%}  "
                  f"plan_t={m['total_plan_time_s']}s")
        if not args.no_viz:
            render(name, world, traces)
            print(f"  visuals -> {OUT_DIR}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nmetrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
