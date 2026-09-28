#!/usr/bin/env python3
"""
Heading-selection angular-travel sweep, exploration refactor

QUESTION
--------
`compute_headings_for_waypoint` greedily minimises the NUMBER of camera headings
needed to cover all uncovered bearings, but ignores the TOTAL ANGULAR TRAVEL the
robot physically turns (the real cost per the heading-timing study: time + odometry
drift). Two independent losses exist today:

  L1. TIE-BREAK LOSS (selection): on equal bearing-coverage gain, `max()` picks the
      first heading in 0°,30°,60°,… order — i.e. an arbitrary heading that may sit
      far from the others, spreading the selected set wider than needed.
  L2. ORDERING LOSS (sweep): `get_headings` sorts the selected set CLOCKWISE from the
      arrival heading. When the headings cluster just *behind* the robot (e.g. robot
      at 10°, headings {350°, 330°}), the clockwise sort forces a ~320°+ forward wrap
      instead of a cheap -20°,-40° backward sweep.

VARIANTS (each targets one loss; names say what they do)
--------------------------------------------------------
  production_today          : stock selection + clockwise-from-arrival sort (baseline).
  cluster_tiebreak          : selection prefers, ON GAIN TIES ONLY, the heading
                              angularly closest to the arrival heading / already-
                              selected set (fixes L1). Sweep order unchanged (CW sort).
  optimal_sweep_order       : stock selection; sweep order replaced by the exact
                              minimum-total-travel visiting order (brute force over
                              ≤6! permutations with wrapped shortest deltas — the
                              node's Spin goals use shortest signed delta, so travel
                              = sum of wrapped |Δ| along the order). Fixes L2.
                              Coverage-invariant by construction (same heading SET).
  cluster_and_optimal_order : both fixes combined (L1 + L2).

WHAT IS SIMULATED (realigned to the CURRENT node — do not copy from the old
heading_timing_impact.ipynb sim, which predates two production changes)
-----------------------------------------------------------------------
  * Arrival-time heading refresh: node `_refresh_waypoint_headings` recomputes
    wp.headings from the live covered_mask at every arrival (shipped after the
    notebook's study). The sim does the same for ALL variants.
  * Replan cadence: node replans every `replan_every_n_step=2` arrived waypoints
    under SLAM; static-map runs drain the plan. This sim runs the STATIC-map mode
    (known map, drain) as the primary arena — heading selection is identical logic
    in both modes and static mode is deterministic/noise-free — plus a `--cadence 2`
    flag to confirm on the SLAM-like cadence that conclusions hold.
  * Arrival heading = bearing of the last path step (the node reads the real yaw
    from TF at arrival; direction-of-travel is the faithful headless equivalent —
    the old sim's carried-over rotation heading is NOT what TF reports).
  * RotationState(stop_after=3) early-stop per waypoint, exactly as the node.
  * Spin cost per heading = wrapped shortest |Δ| from current heading (node
    `_send_spin_goal` sends shortest signed delta).

METRICS (per map x start x variant)
-----------------------------------
  primary : total_angular_travel_deg  (down = better; the real cost)
  guards  : final_coverage identical to production_today (hard, selection variants)
            total_spins not increased
  info    : zero_gain_spins, near_redundant_spins (<5% of peak single-spin gain),
            est_rotation_time_s  (travel / ROT_SPEED_DPS + spins * SPIN_OVERHEAD_S)

VISUAL PROOF (written to test_heading_travel_out/)
--------------------------------------------------
  <map>_travel_bars.png        : angular travel per variant, grouped bars.
  <map>_wp_panels_<var>.png    : per-waypoint local crops — covered-before (green),
                                 arrival heading (black arrow), numbered sweep arrows
                                 in execution order (blue=productive, orange=near-
                                 redundant, red=zero-gain), per-panel °-turned.
  <map>_sweep_<varA>_vs_<varB>.gif : side-by-side animation, one frame per spin,
                                 FOV wedge + growing coverage + running travel counter.
  metrics.json / metrics.csv   : full table.

DECISION RULE (agreed up front): adopt a variant only if angular travel drops ≥10%
on at least one real map, spins do not increase, and coverage is held on all maps.

Usage:
  python test_heading_travel.py                          # all maps, all variants
  python test_heading_travel.py --maps lab_ghent
  python test_heading_travel.py --cadence 2              # SLAM-like replan-every-2
  python test_heading_travel.py --no-gif                 # skip the slow animations
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import deque
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
OUT_DIR = Path(__file__).resolve().parent / "test_heading_travel_out"

_EXPLORATION_PKG = _ROOT / "ros2_ws" / "src" / "navigation" / "exploration"
sys.path.insert(0, str(_EXPLORATION_PKG))

from exploration.explore_costmap_map import (  # noqa: E402
    build_map_data, compute_visibility, compute_headings_for_waypoint,
    update_covered_mask, pixel_to_world, angular_diff,
)
from exploration.execution_strategy import ExplorationSession  # noqa: E402
from exploration.rotation_strategy import get_headings, RotationState  # noqa: E402

# Config mirror (matches node defaults / other harnesses in this folder).
MAX_DETECTION_M = 6.0
FOV_HORIZONTAL = 87.0
NUM_RAYS = 360
OBSERVATION_INCREMENT = 30.0
SAMPLING_STEP_M = 4.5          
COMPLETION_THRESHOLD = 0.90
INFLATION_M = 0.4

# Rotation-time model for the est_rotation_time_s info metric (not a decision
# metric — just to translate degrees into seconds for the report).
ROT_SPEED_DPS = 45.0           # sustained Nav2 Spin angular speed, deg/s
SPIN_OVERHEAD_S = 1.5          # per-Spin action overhead (goal round-trip + settle)

NEAR_REDUNDANT_FRAC = 0.05     # <5% of peak single-spin gain = "barely paid for the turn"


def planner_config() -> dict:
    return {
        "max_detection_range": MAX_DETECTION_M,
        "fov_horizontal": FOV_HORIZONTAL,
        "observation_rotation_increment": OBSERVATION_INCREMENT,
        "sampling_step_m": SAMPLING_STEP_M,
        "num_rays": NUM_RAYS,
        "frontier_weight": 1.0,
        "coverage_weight": 1.0,
        "exploration_completion_threshold": COMPLETION_THRESHOLD,
        "planner_coverage_warning_threshold": COMPLETION_THRESHOLD,
    }


# ===========================================================================
# Variant implementations (harness-local — production is NOT edited)
# ===========================================================================

def _wrapped_step(frm: float, to: float) -> float:
    """|shortest signed delta| in degrees — the node's Spin cost for one heading."""
    return abs((to - frm + 180.0) % 360.0 - 180.0)


def select_headings_stock(col, row, uncovered_cells, arrival_heading):
    """Production selection: compute_headings_for_waypoint verbatim (tie-break =
    first heading in 0°,30°,… option order)."""
    return compute_headings_for_waypoint(
        col, row, uncovered_cells,
        fov_deg=FOV_HORIZONTAL, increment_deg=OBSERVATION_INCREMENT)


def select_headings_cluster_tiebreak(col, row, uncovered_cells, arrival_heading):
    """Same greedy as production but, ON EQUAL bearing-coverage gain, prefer the
    heading angularly closest to what the robot already faces or has selected —
    clusters the set so the eventual sweep is tighter. Non-tie picks are identical
    to production (same gain ordering), so the covered-bearing guarantee holds.
    Mirrors compute_headings_for_waypoint's loop structure on purpose."""
    if not uncovered_cells:
        return []
    bearings = [
        math.degrees(math.atan2(-(r - row), c - col)) % 360.0
        for c, r in uncovered_cells
    ]
    half_fov = FOV_HORIZONTAL / 2.0
    options = [i * OBSERVATION_INCREMENT
               for i in range(int(round(360.0 / OBSERVATION_INCREMENT)))]
    remaining = list(range(len(bearings)))
    selected: list[float] = []
    while remaining:
        anchors = selected if selected else [arrival_heading]

        def key(h):
            gain = sum(1 for i in remaining if angular_diff(bearings[i], h) <= half_fov)
            prox = min(_wrapped_step(a, h) for a in anchors)
            return (gain, -prox)          # max gain first; ties → closest to anchors

        best = max(options, key=key)
        newly = [i for i in remaining if angular_diff(bearings[i], best) <= half_fov]
        if not newly:
            break
        selected.append(best)
        newly_set = set(newly)
        remaining = [i for i in remaining if i not in newly_set]
    return selected


def order_headings_clockwise(headings, arrival_heading):
    """Production ordering: get_headings' clockwise-from-arrival sort."""
    return sorted((float(h) for h in headings),
                  key=lambda h: (h - arrival_heading) % 360.0)


def order_headings_min_travel(headings, arrival_heading):
    """Exact minimum-total-angular-travel visiting order, starting from the arrival
    heading, where each move costs the wrapped shortest |Δ| (matching the node's
    Spin goals). Brute force: heading sets are ≤ 360/inc = 12 but in practice ≤6
    (planner minimum-set), so ≤720 permutations — exact and instant."""
    hs = [float(h) for h in headings]
    if len(hs) <= 1:
        return hs
    best_order, best_cost = None, float("inf")
    for perm in itertools.permutations(hs):
        cost, cur = 0.0, arrival_heading
        for h in perm:
            cost += _wrapped_step(cur, h)
            cur = h
            if cost >= best_cost:
                break
        if cost < best_cost:
            best_cost, best_order = cost, list(perm)
    return best_order


# Variant registry: name → (selection_fn, ordering_fn). Names are the labels used
# everywhere in output — they say what the variant DOES, not a letter.
VARIANTS = {
    "production_today":          (select_headings_stock,             order_headings_clockwise),
    "cluster_tiebreak":          (select_headings_cluster_tiebreak,  order_headings_clockwise),
    "optimal_sweep_order":       (select_headings_stock,             order_headings_min_travel),
    "cluster_and_optimal_order": (select_headings_cluster_tiebreak,  order_headings_min_travel),
}


# ===========================================================================
# Headless explore sim — realigned to the CURRENT node loop
# ===========================================================================

def find_path(navigable_mask, start, goal):
    """BFS 8-connected path (col,row); [start] if unreachable. (Sim locomotion
    only — not part of the system under test.)"""
    if start == goal:
        return [start]
    H, W = navigable_mask.shape
    q = deque([start])
    came = {start: None}
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        col, row = q.popleft()
        if (col, row) == goal:
            break
        for dc, dr in nb:
            nc, nr = col + dc, row + dr
            if 0 <= nc < W and 0 <= nr < H and navigable_mask[nr, nc] and (nc, nr) not in came:
                came[(nc, nr)] = (col, row)
                q.append((nc, nr))
    if goal not in came:
        return [start]
    path = [goal]
    while path[-1] != start:
        path.append(came[path[-1]])
    return path[::-1]


def load_map(name: str):
    d = ASSETS_DIR / name
    meta = yaml.safe_load(open(d / "map.yaml"))
    arr = np.array(Image.open(d / meta["image"]), dtype=np.uint8)
    res = float(meta["resolution"])
    negate = int(meta.get("negate", 0))
    p_occ = arr / 255.0 if negate else 1.0 - arr / 255.0
    return build_map_data(p_occ, res, float(meta["origin"][0]),
                          float(meta["origin"][1]), inflation_radius_m=INFLATION_M)


def _arrival_heading_from_path(path, fallback: float) -> float:
    """Robot yaw at arrival ≈ bearing of the final path step (what TF reports after
    Nav2 drives in). Row grows down, world-y up → -Δrow."""
    if len(path) < 2:
        return fallback
    (c0, r0), (c1, r1) = path[-2], path[-1]
    return math.degrees(math.atan2(-(r1 - r0), c1 - c0)) % 360.0


def simulate(md, cfg, variant: str, start_cx: int, start_cy: int,
             replan_every: int | None = None, max_iters: int = 300,
             capture_frames: bool = False):
    """Headless explore loop mirroring the CURRENT ros2_exploration_node:

      plan → per waypoint: BFS-walk → arrival heading from travel direction →
      arrival-time heading REFRESH (compute_visibility on live covered_mask →
      variant's selection fn) → variant's ordering fn → spin through headings
      with RotationState(stop_after=3) → on_arrive; replan after `replan_every`
      arrivals (None = drain whole plan, the static-map mode).

    The ONLY things that differ between variants are the two functions in
    VARIANTS[variant]. Everything else is byte-identical.
    """
    select_fn, order_fn = VARIANTS[variant]
    md.covered_mask[:] = False
    H = md.pgm_array.shape[0]
    res = md.resolution
    max_range_px = max(1, int(cfg["max_detection_range"] / res))
    fov, nrays = cfg["fov_horizontal"], cfg["num_rays"]
    inc = cfg["observation_rotation_increment"]

    session = ExplorationSession(md, cfg)
    rc, rr = session.nearest_start(start_cx, start_cy)
    rx, ry = pixel_to_world(rc, rr, res, md.origin_x, md.origin_y, H)
    heading = 0.0

    spins = []          # per-spin records
    wp_snapshots = []   # (wp_idx, col, row, arrival_heading, covered_before)
    frames = []         # per-spin (col,row,heading,new_cells,covered_after) for GIF
    wp_idx = 0
    final_ratio = 0.0

    for _ in range(max_iters):
        waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(rx, ry)
        final_ratio = ratio
        if (no_frontiers and ratio >= cfg["exploration_completion_threshold"]) or not waypoints:
            break
        arrived_this_plan = 0
        for wp in waypoints:
            path = find_path(md.navigable_mask, (rc, rr), (wp.col, wp.row))
            if session.on_unreachable(wp, path):
                continue
            rc, rr = wp.col, wp.row
            rx, ry = wp.x, wp.y
            heading = _arrival_heading_from_path(path, heading)   # TF yaw equivalent

            # ── arrival-time refresh (node _refresh_waypoint_headings) ──────
            cov_cells, _ = compute_visibility((wp.col, wp.row), md, max_range_px, nrays)
            selected = select_fn(wp.col, wp.row, cov_cells, heading)
            if not selected:
                # node: get_headings falls back to full 360° sweep on empty set
                selected = [float(h) for h in range(0, 360, int(inc))]
            rot_headings = order_fn(selected, heading)

            wp_snapshots.append((wp_idx, wp.col, wp.row, heading, md.covered_mask.copy()))

            rot = RotationState(stop_after=3)
            for h in rot_headings:
                ang_step = _wrapped_step(heading, float(h))
                heading = float(h)
                before = int(np.sum(md.covered_mask & md.free_mask))
                ratio = update_covered_mask(md, wp.col, wp.row, heading, fov,
                                            max_range_px, nrays)
                new_cells = int(np.sum(md.covered_mask & md.free_mask)) - before
                spins.append({"wp_idx": wp_idx, "col": wp.col, "row": wp.row,
                              "heading": heading, "ang_step": ang_step,
                              "new_cells": new_cells})
                if capture_frames:
                    frames.append((wp.col, wp.row, heading, new_cells,
                                   md.covered_mask.copy()))
                final_ratio = ratio
                if not rot.update(ratio):
                    break
            session.on_arrive(wp)
            wp_idx += 1
            arrived_this_plan += 1
            if replan_every is not None and arrived_this_plan >= replan_every:
                break   # node _finish_waypoint: back to PLANNING every N arrivals

    return {"spins": spins, "snapshots": wp_snapshots, "frames": frames,
            "final_coverage": round(final_ratio, 4), "waypoints": wp_idx,
            "max_range_px": max_range_px}


# ===========================================================================
# Metrics + report
# ===========================================================================

def _near_redundant_ids(spins):
    if not spins:
        return set()
    peak = max(s["new_cells"] for s in spins) or 1
    return {id(s) for s in spins if s["new_cells"] < NEAR_REDUNDANT_FRAC * peak}


def compute_metrics(map_name, variant, run):
    spins = run["spins"]
    ang = sum(s["ang_step"] for s in spins)
    zero = sum(1 for s in spins if s["new_cells"] == 0)
    nr = len(_near_redundant_ids(spins))
    return {
        "map": map_name,
        "variant": variant,
        "total_angular_travel_deg": round(ang, 1),
        "total_spins": len(spins),
        "zero_gain_spins": zero,
        "near_redundant_spins": nr,
        "final_coverage": run["final_coverage"],
        "waypoints_visited": run["waypoints"],
        "est_rotation_time_s": round(ang / ROT_SPEED_DPS + len(spins) * SPIN_OVERHEAD_S, 1),
        "cells_per_deg": round(sum(s["new_cells"] for s in spins) / ang, 2) if ang else 0.0,
    }


# ===========================================================================
# Visual proof
# ===========================================================================

def render_travel_bars(map_name, metrics_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [m for m in metrics_rows if m["map"] == map_name]
    names = [m["variant"] for m in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(
            axes,
            ["total_angular_travel_deg", "total_spins", "est_rotation_time_s"],
            ["total angular travel (deg) — PRIMARY, lower is better",
             "spin count (guard: must not increase)",
             f"est. rotation time (s) @ {ROT_SPEED_DPS:.0f}°/s + {SPIN_OVERHEAD_S}s/spin"]):
        vals = [m[key] for m in rows]
        colors = ["#888888" if n == "production_today" else "#2b8cbe" for n in names]
        ax.bar(range(len(rows)), vals, color=colors)
        base = vals[names.index("production_today")]
        for i, v in enumerate(vals):
            d = (v - base) / base * 100 if base else 0.0
            lbl = f"{v:.0f}" + ("" if names[i] == "production_today" else f"\n{d:+.0f}%")
            ax.text(i, v, lbl, ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    cov = ", ".join(f"{m['variant']}={m['final_coverage']:.1%}" for m in rows)
    fig.suptitle(f"{map_name} — heading-selection variants (coverage guard: {cov})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{map_name}_travel_bars.png", dpi=120)
    plt.close(fig)


def render_wp_panels(map_name, variant, md, run, out_dir, ncols=5, max_panels=25):
    """Per-waypoint local crops: covered-before (green), arrival heading (black,
    dashed), numbered sweep arrows in execution order (blue=productive,
    orange=near-redundant <5%, red=zero-gain), title = degrees turned there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spins, snaps = run["spins"], run["snapshots"][:max_panels]
    if not snaps:
        return
    nr_ids = _near_redundant_ids(spins)
    n = len(snaps)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.3 * nrows),
                             squeeze=False)
    crop = int(run["max_range_px"] * 1.3)
    ray = run["max_range_px"] * 0.75
    for k, (wp_idx, col, row, arr_h, cov_before) in enumerate(snaps):
        ax = axes[k // ncols][k % ncols]
        r0, r1 = max(0, row - crop), min(md.free_mask.shape[0], row + crop)
        c0, c1 = max(0, col - crop), min(md.free_mask.shape[1], col + crop)
        ax.imshow(md.free_mask[r0:r1, c0:c1], cmap="Greys", origin="upper",
                  alpha=0.25, extent=[c0, c1, r1, r0])
        cov = np.ma.masked_where(~cov_before, cov_before)
        ax.imshow(cov[r0:r1, c0:c1], cmap="Greens", origin="upper", alpha=0.55,
                  vmin=0, vmax=1, extent=[c0, c1, r1, r0])
        # arrival heading: black dashed arrow
        th = np.radians(arr_h)
        ax.annotate("", xy=(col + ray * 0.6 * np.cos(th), row - ray * 0.6 * np.sin(th)),
                    xytext=(col, row),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.4, ls="--", alpha=0.8))
        wp_spins = [s for s in spins if s["wp_idx"] == wp_idx]
        for j, s in enumerate(wp_spins, start=1):
            th = np.radians(s["heading"])
            dx, dy = ray * np.cos(th), -ray * np.sin(th)
            color = ("red" if s["new_cells"] == 0 else
                     "orange" if id(s) in nr_ids else "tab:blue")
            ax.annotate("", xy=(col + dx, row + dy), xytext=(col, row),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.9))
            ax.text(col + dx * 1.08, row + dy * 1.08, str(j), color=color,
                    fontsize=8, ha="center", va="center", fontweight="bold")
        ax.scatter([col], [row], s=40, c="k", marker="o", zorder=3)
        turned = sum(s["ang_step"] for s in wp_spins)
        ax.set_title(f"wp{wp_idx}: {len(wp_spins)} spins, {turned:.0f}° turned",
                     fontsize=8)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(
        f"{map_name} — {variant}\n"
        "black dashed = arrival heading; numbered arrows = sweep order "
        "(blue=productive, orange=near-redundant <5%, red=zero-gain)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{map_name}_wp_panels_{variant}.png", dpi=110)
    plt.close(fig)


def render_gif(map_name, md, runs: dict, out_dir, fps=2):
    """Side-by-side animation, one frame per spin: robot, FOV wedge (red if that
    spin gained zero cells), growing coverage, running angular-travel counter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from matplotlib.patches import Wedge

    names = list(runs)
    nmax = max(len(runs[n]["frames"]) for n in names)
    if nmax == 0:
        return
    bg = np.where(md.occupied_mask, 0.0, np.where(md.free_mask, 1.0, 0.5))
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    ims, wedges, titles, travel = [], [], [], {n: 0.0 for n in names}
    for ax, n in zip(axes, names):
        ax.imshow(bg, cmap="gray", origin="upper")
        im = ax.imshow(np.ma.masked_all(bg.shape), cmap="Greens", alpha=0.5,
                       origin="upper", vmin=0, vmax=1)
        ims.append(im)
        w = Wedge((0, 0), 0, 0, 0, alpha=0.35)
        ax.add_patch(w)
        wedges.append(w)
        titles.append(ax.set_title(n, fontsize=10))
        ax.axis("off")
    mrp = max(runs[n]["max_range_px"] for n in names)
    spins_by = {n: runs[n]["spins"] for n in names}

    def update(f):
        artists = []
        for i, n in enumerate(names):
            frames = runs[n]["frames"]
            k = min(f, len(frames) - 1)
            col, row, hdg, new_cells, cov = frames[k]
            m = np.ma.masked_where(~cov, cov.astype(float))
            ims[i].set_data(m)
            wedges[i].set_center((col, row))
            wedges[i].set_radius(mrp)
            # matplotlib angle CCW from +x with y DOWN in image coords → use -hdg
            wedges[i].set_theta1(-hdg - FOV_HORIZONTAL / 2)
            wedges[i].set_theta2(-hdg + FOV_HORIZONTAL / 2)
            wedges[i].set_color("red" if new_cells == 0 else "tab:blue")
            if f < len(frames):
                travel[n] = sum(s["ang_step"] for s in spins_by[n][:f + 1])
            titles[i].set_text(
                f"{n} — spin {min(f + 1, len(frames))}/{len(frames)}, "
                f"travel {travel[n]:.0f}°"
                f"{'  [RED = zero-gain spin]' if new_cells == 0 else ''}")
            artists += [ims[i], wedges[i], titles[i]]
        return artists

    ani = manim.FuncAnimation(fig, update, frames=nmax, blit=False)
    path = out_dir / f"{map_name}_sweep_{'_vs_'.join(names)}.gif"
    ani.save(path, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+",
                    default=["lab_ghent", "lab_05", "warehouse_amazon"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--cadence", type=int, default=None,
                    help="replan every N arrivals (SLAM-like); default = drain plan "
                         "(static-map mode, deterministic)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name in args.maps:
        md = load_map(name)
        H, W = md.pgm_array.shape
        tag = f" (replan every {args.cadence})" if args.cadence else " (drain plan)"
        print(f"\n=== {name}{tag} ===")
        runs = {}
        for variant in args.variants:
            run = simulate(md, planner_config(), variant, W // 2, H // 2,
                           replan_every=args.cadence,
                           capture_frames=not (args.no_gif or args.no_viz))
            runs[variant] = run
            m = compute_metrics(name, variant, run)
            all_metrics.append(m)
            print(f"  [{variant:26s}] travel={m['total_angular_travel_deg']:7.0f}°  "
                  f"spins={m['total_spins']:3d}  zero-gain={m['zero_gain_spins']:2d}  "
                  f"near-red={m['near_redundant_spins']:2d}  "
                  f"cov={m['final_coverage']:.1%}  wps={m['waypoints_visited']:3d}  "
                  f"est_rot_t={m['est_rotation_time_s']:6.1f}s")
        if not args.no_viz:
            render_travel_bars(name, all_metrics, OUT_DIR)
            for variant, run in runs.items():
                render_wp_panels(name, variant, md, run, OUT_DIR)
            if not args.no_gif:
                pair = {v: runs[v] for v in
                        ("production_today", "cluster_and_optimal_order") if v in runs}
                if len(pair) == 2:
                    render_gif(name, md, pair, OUT_DIR)
            print(f"  visuals -> {OUT_DIR}")

    # Tag output tables by cadence so a drain run and a cadence run don't clobber
    # each other (both belong in the evidence record).
    stem = f"metrics_replan{args.cadence}" if args.cadence else "metrics_drain"
    with open(OUT_DIR / f"{stem}.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(OUT_DIR / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\nmetrics -> {OUT_DIR / stem}.json / .csv")


if __name__ == "__main__":
    main()
