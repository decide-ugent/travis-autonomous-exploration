"""
Step-by-step visualisation of the waypoint-selection pipeline.

Runs the *real* planner once on a map, captures the interesting intermediate
state at each stage, and renders it as a sequence of labelled PNG panels plus an
animation (GIF, and MP4 when ffmpeg is available). The point is to *explain the
logic*: how an occupancy grid becomes an ordered list of waypoints, and above
all WHY the greedy set-cover picks what it picks.

Pipeline stages, one panel group each:
    1. Masks              free / occupied / unknown / navigable
    2. Candidates         the sampled viewpoint grid
    3. Score field        every candidate coloured by its score (what greedy maximises)
    4. Greedy picks       per round: score -> pick (+360deg planning disc, arrow)
                          -> re-score; the chosen pick highlighted each round
    5. TSP ordering       geodesic nearest-neighbour tour vs the naive Euclidean
    6. Final plan         the ordered numbered waypoints the node would drive

The greedy score at each iteration is re-derived here (not read from the log)
because production greedy_set_cover only returns final-state residual scores.
The reimplemented loop is asserted equal to greedy_set_cover's selection order,
and the ordered result equal to plan_waypoints', so the picture cannot silently
drift from production.

Run:
    python3 visualisation/visualise_system_steps.py                 # reference lab map
    python3 visualisation/visualise_system_steps.py --pgm m.pgm --yaml m.yaml
    python3 visualisation/visualise_system_steps.py --no-anim       # PNGs + CSV only

Output: visualisation/visualise_system_steps/<run>/ with step_*.png,
exploration_steps.mp4 (imageio; GIF fallback), and steps.csv (per-iteration scores).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed, we only save files
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np

# -- sys.path setup (mirrors visual_demo_coverage.py) ------------------------
_ROOT = Path(__file__).parent.parent.parent.parent   # /ros2_ws/src/
_HERE = Path(__file__).parent                         # visualisation/
_EXPLORATION_PKG = _ROOT / "navigation" / "exploration"
_TESTS_DIR = _EXPLORATION_PKG / "tests"               # conftest + demo_robot live here
sys.path.insert(0, str(_TESTS_DIR))  # for conftest constants + demo_robot
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "navigation"))
sys.path.insert(0, str(_EXPLORATION_PKG))

from navigation.exploration.exploration.explore_costmap_map import (
    compute_achievable_cells,
    compute_all_visibility,
    coverage_ratio,
    generate_candidates,
    greedy_set_cover,
    nearest_neighbor_order,
    navigable_distance_map,
    plan_waypoints,
    world_to_pixel,
)
from demo_robot import build_demo_config, INFLATION_M
from conftest import REFERENCE_MAP_PGM, REFERENCE_MAP_YAML
from navigation.exploration.exploration.explore_costmap_map import load_map

# -- colours (mirror exploration_visualiser.py) ------------------------------
C_FREE = 1.0
C_UNKNOWN = 0.65
C_WALL = 0.0
GREEN = (0.0, 0.75, 0.0)
STEELBLUE = "steelblue"
ROBOT_RED = "red"

# Every animated panel (steps 3-6) is rendered at this exact figure size and DPI,
# with a FIXED layout (no bbox_inches="tight"), so all frames share identical
# pixel dimensions - required for a clean MP4 with no per-frame padding.
PANEL_FIGSIZE = (13, 8)
PANEL_DPI = 130


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------

def _base_display(md) -> np.ndarray:
    """Occupancy RGB image: white free, black wall, grey unknown."""
    H, W = md.pgm_array.shape
    disp = np.full((H, W, 3), C_UNKNOWN)
    disp[md.free_mask] = C_FREE
    disp[md.occupied_mask] = C_WALL
    return disp


def _new_ax(md, title: str):
    """Create an animated-panel figure with FIXED axes rectangles: the map axes and
    a colorbar axes both sit at constant positions, so every panel (step 3-6) is the
    same pixel size whether or not it actually draws a colorbar. Returns (fig, ax, cax).
    Panels with a score field colour into `cax`; panels without one hide it."""
    H, W = md.pgm_array.shape
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    ax = fig.add_axes((0.08, 0.08, 0.80, 0.80))    # map: constant rectangle
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.72))   # colorbar: constant rectangle
    ax.imshow(_base_display(md), origin="upper", zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return fig, ax, cax


def _fill_colorbar(fig, cax, mappable):
    """Draw the score colourbar into the reserved cax, or blank it when a panel has
    no score field (keeps the axes rectangle identical either way)."""
    if mappable is None:
        cax.axis("off")
        return
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("greedy score", fontsize=11)


def _overlay_cells(ax, cells, rgb, alpha, H, W, zorder=1):
    """Paint a set/iterable of (col,row) cells as a translucent colour layer."""
    if not cells:
        return
    layer = np.zeros((H, W, 4), dtype=np.float32)
    cols = np.fromiter((c for c, _ in cells), dtype=np.intp)
    rows = np.fromiter((r for _, r in cells), dtype=np.intp)
    layer[rows, cols] = (*rgb, alpha)
    ax.imshow(layer, origin="upper", zorder=zorder)


def _save(fig, out_dir: Path, name: str, *, fixed_size: bool = True) -> Path:
    """Save a figure. Animated panels use fixed_size=True: a constant DPI and NO
    bbox='tight' crop, so every frame is byte-identical in dimensions. The masks
    overview (not animated) passes fixed_size=False and may crop to content."""
    path = out_dir / name
    if fixed_size:
        # Fixed add_axes rectangles + no bbox crop => the saved PNG is exactly
        # PANEL_FIGSIZE * PANEL_DPI for every animated panel.
        fig.savefig(path, dpi=PANEL_DPI)
    else:
        fig.savefig(path, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _get_unique_run_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.parent / f"{base.name}_{i}"
        if not cand.exists():
            return cand
        i += 1


# ---------------------------------------------------------------------------
# The instrumented greedy loop (mirrors greedy_set_cover, snapshots each pick)
# ---------------------------------------------------------------------------

def greedy_with_snapshots(vis, alpha, beta, dist_from_robot, gamma, normaliser,
                          max_waypoints):
    """Re-run the production greedy score/select loop, recording per-iteration
    marginal scores for every remaining candidate.

    Returns (selected, snapshots) where snapshots[i] is a dict for the i-th pick:
        {"chosen": (col,row),
         "scores": {cand: marginal_score, ...over remaining before this pick},
         "covered_fraction": running covered fraction after the pick}
    The scoring is byte-for-byte the same formula as greedy_set_cover._score.
    """
    achievable = set().union(*(cov for cov, _ in vis.values())) if vis else set()
    remaining_coverage = set(achievable)
    remaining_frontiers = set().union(*(fr for _, fr in vis.values())) if vis else set()

    if dist_from_robot is not None:
        remaining = [c for c in vis if dist_from_robot.get(c, np.inf) < np.inf]
    else:
        remaining = list(vis.keys())

    use_dist = dist_from_robot is not None and gamma > 0.0 and normaliser > 0.0

    def score(c):
        gain = (alpha * len(vis[c][1] & remaining_frontiers)
                + beta * len(vis[c][0] & remaining_coverage))
        if not use_dist:
            return gain
        dist = dist_from_robot.get(c, 0.0)
        if dist == np.inf:
            dist = normaliser * 1e6
        return gain * math.exp(-gamma * dist / normaliser)

    selected = []
    snapshots = []
    while remaining:
        if max_waypoints is not None and len(selected) >= max_waypoints:
            break
        scores = {c: score(c) for c in remaining}  # snapshot BEFORE the pick
        best = max(remaining, key=lambda c: scores[c])
        if (not (vis[best][0] & remaining_coverage)
                and not (vis[best][1] & remaining_frontiers)):
            break
        selected.append(best)
        remaining_coverage -= vis[best][0]
        remaining_frontiers -= vis[best][1]
        remaining.remove(best)
        covered_fraction = (
            1.0 - len(remaining_coverage) / len(achievable) if achievable else 1.0)
        snapshots.append({
            "chosen": best,
            "scores": scores,
            "covered_fraction": covered_fraction,
        })
    return selected, snapshots


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_masks(md, out_dir):
    """Each mask painted opaquely on a plain white background, so occupied and
    unknown are unmistakable (they are sparse on a converted SLAM map and vanish
    if drawn faintly over the base occupancy image)."""
    H, W = md.pgm_array.shape
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    specs = [
        (md.free_mask, "free_mask", (0.30, 0.55, 1.0)),
        (md.occupied_mask, "occupied_mask (walls)", (0.0, 0.0, 0.0)),
        (md.unknown_mask, "unknown_mask (not yet mapped)", (0.55, 0.35, 0.75)),
        (md.navigable_mask, "navigable_mask (robot can stand here)", (0.0, 0.65, 0.0)),
    ]
    for ax, (mask, title, colour) in zip(axes.ravel(), specs):
        img = np.ones((H, W, 3), dtype=np.float32)   # plain white background
        img[mask] = colour                            # opaque, no blending
        ax.imshow(img, origin="upper", zorder=0)
        ax.set_title(f"{title}  ({int(mask.sum())} px)", fontsize=11)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Step 1 - Occupancy masks from the map", fontsize=14)
    fig.tight_layout()
    return _save(fig, out_dir, "step_1_masks.png", fixed_size=False)


def _overlay_cells_from_mask(ax, mask, rgb, alpha, H, W, zorder=1):
    layer = np.zeros((H, W, 4), dtype=np.float32)
    layer[mask] = (*rgb, alpha)
    ax.imshow(layer, origin="upper", zorder=zorder)


def panel_candidates(md, candidates, robot_px, out_dir):
    H, W = md.pgm_array.shape
    fig, ax, cax = _new_ax(md, f"Step 2 - Candidate viewpoints ({len(candidates)} "
                               f"sampled on the navigable grid)")
    _fill_colorbar(fig, cax, None)
    _overlay_cells_from_mask(ax, md.navigable_mask, (0.0, 0.7, 0.0), 0.15, H, W)
    if candidates:
        cc = [c for c, _ in candidates]
        rr = [r for _, r in candidates]
        ax.plot(cc, rr, "o", color=STEELBLUE, markersize=4, zorder=3)
    ax.plot(*robot_px, "s", color=ROBOT_RED, markersize=10, zorder=5, label="robot")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "step_2_candidates.png")


def panel_score_field(initial_scores, md, robot_px, out_dir):
    """Step 3: every candidate coloured by its score (what greedy maximises).
    score = alpha*frontier_gain + beta*coverage_gain, scaled by a distance decay."""
    fig, ax, cax = _new_ax(
        md, "Step 3 - Score of every candidate (greedy maximises this)\n"
            "score = alpha*frontier_gain + beta*coverage_gain, "
            "scaled by exp(-gamma * geodesic_dist / range)")
    # same fixed colour scale as the step-4 greedy frames (round-1 max)
    vmax = max(initial_scores.values()) if initial_scores else 1.0
    mappable = None
    if initial_scores:
        cols = np.array([c for c, _ in initial_scores])
        rows = np.array([r for _, r in initial_scores])
        vals = np.array([initial_scores[(c, r)] for c, r in zip(cols, rows)], float)
        mappable = ax.scatter(cols, rows, c=vals, cmap="plasma", s=300,
                              vmin=0.0, vmax=vmax,
                              edgecolors="k", linewidths=0.8, zorder=3)
    _fill_colorbar(fig, cax, mappable)
    ax.plot(*robot_px, "s", color=ROBOT_RED, markersize=12, zorder=5, label="robot")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "step_3_score_field.png")


# -- Step 5: greedy set cover, shown as a sequence of sub-frames per round ----

def _draw_greedy_frame(md, robot_px, max_range_px, title, out_dir, name, *,
                       remaining_scores, vmax, selected_prev, faint_cells,
                       covered_cells, winner=None, prev_winner=None,
                       draw_fov=False):
    """Render ONE greedy sub-frame.

    remaining_scores : {cand: score} for candidates still in the running (coloured)
    vmax             : fixed colour-scale max, so darkening is comparable across frames
    selected_prev    : already-picked candidates -> drawn solid GREY
    faint_cells      : never-picked candidates not in remaining_scores -> faint dots
    covered_cells    : cells claimed by picks so far -> green overlay
    winner           : the candidate chosen this round -> red ring (+ 360deg FOV disc if draw_fov)
    prev_winner      : previous pick -> draw an arrow prev_winner -> winner
    """
    H, W = md.pgm_array.shape
    fig, ax, cax = _new_ax(md, title)

    _overlay_cells(ax, covered_cells, GREEN, 0.30, H, W, zorder=1)

    # non-selected, non-remaining candidates: kept visible but very faint
    if faint_cells:
        fc = [c for c, _ in faint_cells]
        fr = [r for _, r in faint_cells]
        ax.plot(fc, fr, "o", color="0.5", markersize=6, alpha=0.20, zorder=2)

    # already-selected candidates: solid grey
    if selected_prev:
        sc_ = [c for c, _ in selected_prev]
        sr_ = [r for _, r in selected_prev]
        ax.plot(sc_, sr_, "o", color="0.35", markersize=11, zorder=3)

    # remaining candidates: coloured by current score, fixed scale. A colorbar is
    # ALWAYS drawn (even with no remaining candidates) so every animated frame keeps
    # the same axes geometry as step 3 -> identical pixel dimensions across the MP4.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if remaining_scores:
        cols = np.array([c for c, _ in remaining_scores])
        rows = np.array([r for _, r in remaining_scores])
        vals = np.array([remaining_scores[(c, r)] for c, r in zip(cols, rows)], float)
        mappable = ax.scatter(cols, rows, c=vals, cmap="plasma", s=260,
                              vmin=0.0, vmax=vmax,
                              edgecolors="k", linewidths=0.7, zorder=4)
    else:
        mappable = ScalarMappable(norm=Normalize(0.0, vmax), cmap="plasma")
    _fill_colorbar(fig, cax, mappable)

    # arrow from the previous pick to this one, making the 1->2 hand-off explicit
    if prev_winner is not None and winner is not None:
        ax.annotate("", xy=winner, xytext=prev_winner,
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2.2,
                                    shrinkA=8, shrinkB=8), zorder=6)

    # the chosen candidate: red ring, and (on the pick frame) its 360deg planning disc
    if winner is not None:
        if draw_fov:
            disc = plt.Circle(winner, max_range_px, color=(1.0, 0.85, 0.0),
                              alpha=0.25, zorder=2)
            ax.add_patch(disc)
            ax.add_patch(plt.Circle(winner, max_range_px, fill=False,
                                    edgecolor=(0.9, 0.6, 0.0), lw=1.5, zorder=6))
        ax.plot(*winner, "o", color="red", markersize=18, markerfacecolor="none",
                markeredgewidth=2.8, zorder=7)

    ax.plot(*robot_px, "s", color=ROBOT_RED, markersize=11, zorder=8)
    return _save(fig, out_dir, name)


def panel_greedy_picks(snapshots, md, robot_px, vis, max_range_px, out_dir):
    """Step 4: three sub-frames per greedy round so the mechanism is legible:

      a) SCORE   - remaining candidates coloured by their current score; the
                   highest is about to be chosen.
      b) PICK    - that candidate is ringed and its 360deg PLANNING disc is drawn
                   (this is the rotation-potential the score is based on, NOT the
                   87deg see-while-moving camera); an arrow ties it to the prior pick.
      c) RESCORE - the pick's cells become 'covered' (green); the remaining
                   candidates near it re-score DOWNWARD, which is why the next
                   winner is somewhere else.

    Previously-selected candidates stay on every frame in solid grey; never-picked
    candidates stay faint. A single fixed colour scale (max over round 1) makes the
    darkening comparable frame to frame.
    """
    # fixed colour scale = the biggest score seen in the first round
    vmax = max(snapshots[0]["scores"].values()) if snapshots and snapshots[0]["scores"] else 1.0

    # Only the candidates the greedy loop actually considers (reachable ones): this
    # is exactly round 1's scored set, and it matches what step 3 draws. Using
    # vis.keys() here would add unreachable candidates that step 3 never shows.
    all_cands = set(snapshots[0]["scores"].keys()) if snapshots else set()
    covered = set()
    selected_prev: list = []
    prev_winner = None
    paths = []
    n = len(snapshots)

    for i, snap in enumerate(snapshots, start=1):
        winner = snap["chosen"]
        remaining = snap["scores"]                    # cand -> score before this pick
        remaining_set = set(remaining.keys())
        # faint = everything not yet selected and not currently a scored remaining
        faint = all_cands - remaining_set - set(selected_prev)

        base = (f"Step 4 - Greedy set cover, round {i}/{n}  "
                f"(covered {snap['covered_fraction']:.0%})")

        # (a) score frame
        paths.append(_draw_greedy_frame(
            md, robot_px, max_range_px,
            base + "\n(a) score every remaining candidate - highest wins",
            out_dir, f"step_4_{i:02d}a_score.png",
            remaining_scores=remaining, vmax=vmax, selected_prev=list(selected_prev),
            faint_cells=faint, covered_cells=covered,
            winner=winner, prev_winner=None, draw_fov=False))

        # (b) pick frame: ring + 360 planning disc + arrow from previous pick
        paths.append(_draw_greedy_frame(
            md, robot_px, max_range_px,
            base + "\n(b) pick it: its 360deg PLANNING disc (rotation potential) "
                   "is what the score measured",
            out_dir, f"step_4_{i:02d}b_pick.png",
            remaining_scores=remaining, vmax=vmax, selected_prev=list(selected_prev),
            faint_cells=faint, covered_cells=covered,
            winner=winner, prev_winner=prev_winner, draw_fov=True))

        # claim this pick's coverage
        covered = covered | vis[winner][0]
        selected_prev.append(winner)

        # (c) rescore frame: winner now grey, its area green, neighbours re-scored
        next_remaining = snapshots[i]["scores"] if i < n else {}
        paths.append(_draw_greedy_frame(
            md, robot_px, max_range_px,
            base + "\n(c) its cells are now covered -> nearby candidates re-score "
                   "DOWNWARD",
            out_dir, f"step_4_{i:02d}c_rescore.png",
            remaining_scores=next_remaining, vmax=vmax,
            selected_prev=list(selected_prev), faint_cells=all_cands - set(next_remaining) - set(selected_prev),
            covered_cells=covered, winner=None, prev_winner=None, draw_fov=False))

        prev_winner = winner

    return paths


def panel_tsp(selected_px, ordered_wps, md, robot_px, robot_world, out_dir):
    """Geodesic nearest-neighbour tour vs the naive Euclidean order."""
    H, W = md.pgm_array.shape
    fig, ax, cax = _new_ax(
        md, "Step 5 - Tour ordering: geodesic nearest-neighbour (blue) vs "
            "naive Euclidean (grey dashed)")
    _fill_colorbar(fig, cax, None)

    # Euclidean nearest-neighbour order from robot, for contrast
    eucl = _euclidean_nn_order(selected_px, robot_px)
    ex = [robot_px[0]] + [p[0] for p in eucl]
    ey = [robot_px[1]] + [p[1] for p in eucl]
    ax.plot(ex, ey, "--", color="0.5", linewidth=1.2, alpha=0.8, zorder=2,
            label="Euclidean order")

    gx = [robot_px[0]] + [wp.col for wp in ordered_wps]
    gy = [robot_px[1]] + [wp.row for wp in ordered_wps]
    ax.plot(gx, gy, "-", color=STEELBLUE, linewidth=2.0, zorder=3,
            label="geodesic order (driven)")
    for i, wp in enumerate(ordered_wps, start=1):
        ax.plot(wp.col, wp.row, "o", color=STEELBLUE, markersize=8, zorder=4)
        ax.text(wp.col + 4, wp.row - 4, str(i), color=STEELBLUE, fontsize=9,
                fontweight="bold", zorder=5)
    ax.plot(*robot_px, "s", color=ROBOT_RED, markersize=10, zorder=6, label="robot")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "step_5_tsp_ordering.png")


def _euclidean_nn_order(pts, start_px):
    remaining = list(pts)
    order = []
    cur = start_px
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def panel_final(ordered_wps, md, robot_px, ratio, out_dir):
    H, W = md.pgm_array.shape
    fig, ax, cax = _new_ax(
        md, f"Step 6 - Final plan: {len(ordered_wps)} ordered waypoints "
            f"(coverage so far {ratio:.1%})")
    _fill_colorbar(fig, cax, None)
    gx = [robot_px[0]] + [wp.col for wp in ordered_wps]
    gy = [robot_px[1]] + [wp.row for wp in ordered_wps]
    ax.plot(gx, gy, "--", color=STEELBLUE, linewidth=1.0, alpha=0.6, zorder=2)
    for i, wp in enumerate(ordered_wps, start=1):
        ax.plot(wp.col, wp.row, "o", color=STEELBLUE, markersize=9, zorder=3)
        ax.text(wp.col + 4, wp.row - 4, str(i), color=STEELBLUE, fontsize=9,
                fontweight="bold", zorder=4)
    ax.plot(*robot_px, "s", color=ROBOT_RED, markersize=11, zorder=5)
    return _save(fig, out_dir, "step_6_final_plan.png")


# ---------------------------------------------------------------------------
# Animation: stitch the ordered PNGs into an MP4 (imageio + bundled ffmpeg),
# the same mechanism tests/system/visualize_run.py uses. Falls back to a GIF
# (Pillow, always available) only when imageio-ffmpeg is missing.
# ---------------------------------------------------------------------------

def build_animation(png_paths, out_dir, fps=1):
    frame_paths = [str(p) for p in png_paths]
    try:
        import imageio.v2 as imageio
        images = [imageio.imread(fp) for fp in frame_paths]
        # pad every frame to the largest frame size (panels differ in pixel
        # dimensions), then crop to even H/W as H.264 requires.
        max_h = max(im.shape[0] for im in images)
        max_w = max(im.shape[1] for im in images)

        def _pad(im):
            h, w = im.shape[:2]
            c = im.shape[2] if im.ndim == 3 else 1
            canvas = np.full((max_h, max_w, c), 255, dtype=im.dtype)
            src = im if im.ndim == 3 else im[:, :, None]
            canvas[:h, :w, :src.shape[2]] = src
            return canvas

        images = [_pad(im) for im in images]
        images = [im[: im.shape[0] & ~1, : im.shape[1] & ~1] for im in images]
        mp4_path = out_dir / "exploration_steps.mp4"
        # macro_block_size=None avoids a second silent resize on top of our crop.
        imageio.mimsave(mp4_path, images, fps=fps,
                        codec="libx264", macro_block_size=None)
        print(f"  wrote {mp4_path.name} ({len(images)} frames @ {fps} fps)")
        return
    except ImportError:
        print("  imageio / imageio-ffmpeg not available - falling back to GIF")

    # GIF fallback (Pillow ships with matplotlib, no ffmpeg needed).
    import matplotlib.image as mpimg
    frames = [mpimg.imread(fp) for fp in frame_paths]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])
    anim = manimation.FuncAnimation(
        fig, lambda k: (im.set_data(frames[k]) or im,),
        frames=len(frames), interval=1000 / fps, blit=True)
    gif_path = out_dir / "exploration_steps.gif"
    anim.save(str(gif_path), writer=manimation.PillowWriter(fps=fps))
    print(f"  wrote {gif_path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# steps.csv
# ---------------------------------------------------------------------------

def write_steps_csv(snapshots, vis, dist_from_robot, out_dir):
    rows = []
    for i, snap in enumerate(snapshots, start=1):
        for (col, row), sc in snap["scores"].items():
            rows.append({
                "greedy_iteration": i,
                "col": col,
                "row": row,
                "frontier_gain": len(vis[(col, row)][1]),
                "coverage_gain": len(vis[(col, row)][0]),
                "geodesic_dist_px": (round(dist_from_robot[(col, row)], 1)
                                     if dist_from_robot else ""),
                "marginal_score": round(sc, 4),
                "is_chosen": (col, row) == snap["chosen"],
                "covered_fraction_after": round(snap["covered_fraction"], 4),
            })
    path = out_dir / "steps.csv"
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgm", type=Path, default=REFERENCE_MAP_PGM)
    ap.add_argument("--yaml", type=Path, default=REFERENCE_MAP_YAML)
    ap.add_argument("--no-anim", action="store_true", help="skip GIF/MP4")
    args = ap.parse_args()

    md = load_map(args.pgm, args.yaml, INFLATION_M)
    H, W = md.pgm_array.shape
    cfg = build_demo_config()

    # -- replicate plan_waypoints' parameter derivation exactly --------------
    max_range_m = cfg.get("max_detection_range", 6.0)
    sampling_step_m = cfg.get("sampling_step_m", 3.0)
    num_rays = cfg.get("num_rays", 360)
    alpha = cfg.get("frontier_weight", 1.0)
    beta = cfg.get("coverage_weight", 1.0)
    gamma = cfg.get("travel_cost_weight", 1.0)
    is_slam = bool(cfg.get("is_slam", True))
    effective_alpha = alpha if is_slam else 0.0
    sampling_step_px = max(1, int(sampling_step_m / md.resolution))
    max_range_px = max(1, int(max_range_m / md.resolution))

    # robot starts at map centre (same convention as visual_demo_coverage)
    robot_col, robot_row = W // 2, H // 2
    # snap onto navigable if needed via the distance map's start-snap
    from navigation.exploration.exploration.explore_costmap_map import pixel_to_world
    robot_x, robot_y = pixel_to_world(robot_col, robot_row, md.resolution,
                                      md.origin_x, md.origin_y, H)

    candidates = generate_candidates(md.navigable_mask, sampling_step_px, md.resolution)
    achievable = compute_achievable_cells(candidates, md, max_range_m, num_rays)

    r_col, r_row = world_to_pixel(robot_x, robot_y, md.resolution,
                                  md.origin_x, md.origin_y, H)
    dist_map = navigable_distance_map(md.navigable_mask, r_col, r_row, md.inflation_px)
    dist_from_robot = {c: float(dist_map[c[1], c[0]]) for c in candidates}

    vis = compute_all_visibility(candidates, md, max_range_m, num_rays)
    max_waypoints = cfg.get("max_waypoints_per_plan", None)
    if max_waypoints is None:
        overlap_sq = max(1.0, (max_range_m / sampling_step_m) ** 2)
        max_waypoints = max(5, int(np.ceil(len(candidates) / overlap_sq)))

    # -- instrumented greedy + parity check against production ---------------
    selected, snapshots = greedy_with_snapshots(
        vis, effective_alpha, beta, dist_from_robot, gamma,
        float(max_range_px), max_waypoints)
    prod_selected, _, _, _ = greedy_set_cover(
        vis, effective_alpha, beta, dist_from_robot=dist_from_robot,
        gamma=gamma, normaliser=float(max_range_px), max_waypoints=max_waypoints)
    assert selected == prod_selected, (
        "instrumented greedy diverged from greedy_set_cover:\n"
        f"  ours: {selected}\n  prod: {prod_selected}")

    ordered_wps = nearest_neighbor_order(selected, md, vis, robot_x, robot_y)
    prod_wps, ratio, _, _ = plan_waypoints(md, cfg, robot_x, robot_y)
    assert [(w.col, w.row) for w in ordered_wps] == [(w.col, w.row) for w in prod_wps], (
        "ordered waypoints diverged from plan_waypoints")
    print(f"parity OK: {len(selected)} selected, {len(ordered_wps)} ordered; "
          f"coverage {ratio:.1%}")

    # -- render --------------------------------------------------------------
    out_dir = _get_unique_run_dir(_HERE / "visualise_system_steps" / "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_px = (r_col, r_row)

    initial_scores = snapshots[0]["scores"] if snapshots else {}

    pngs = []
    pngs.append(panel_masks(md, out_dir))
    pngs.append(panel_candidates(md, candidates, robot_px, out_dir))
    pngs.append(panel_score_field(initial_scores, md, robot_px, out_dir))
    pngs += panel_greedy_picks(snapshots, md, robot_px, vis, max_range_px, out_dir)
    pngs.append(panel_tsp(selected, ordered_wps, md, robot_px, (robot_x, robot_y), out_dir))
    pngs.append(panel_final(ordered_wps, md, robot_px, ratio, out_dir))
    print(f"wrote {len(pngs)} panels -> {out_dir}/")

    write_steps_csv(snapshots, vis, dist_from_robot, out_dir)

    if not args.no_anim:
        # animate stages 3->7 (skip the mask/candidate setup frames)
        build_animation(pngs[2:], out_dir)

    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
