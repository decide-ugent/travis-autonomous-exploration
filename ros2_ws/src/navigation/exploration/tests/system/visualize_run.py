#!/usr/bin/env python3
"""
Visualise an exploration run over time on the map (system-test layer, L3).

Pure Python + matplotlib, no ROS. Reads a per-run folder produced by
recorder.py and renders, per planning cycle (and as a combined overview):

  - the scene map as a background underlay (see "Map source" below),
  - the covered_mask (observed cells) as a translucent overlay on the map,
  - the executed robot trajectory (from motion.csv, TF map frame),
  - the strategy's published waypoints and current goal (published_waypoints.csv),
  - the path Nav2 actually planned to each goal (nav2_paths.csv),
  - coverage%-vs-path, coverage%-vs-time, and absolute-covered-area-vs-time
    (the last one is monotonic and disambiguates the SLAM coverage% sawtooth,
    whose denominator grows as free space is discovered).

All recorded positions are in the map frame (recorder.py takes the robot pose
from TF map->base), so trajectory, waypoints, Nav2 paths and the map underlay
share one frame with no reconciliation needed.

Map source (first match wins):
  1. <run_dir>/map_final.npy + map_meta.yaml  — the last /map the recorder saw
     (present whenever /map was published, i.e. SLAM or a map_server);
  2. --map-yaml <map.yaml>                    — a standard map_server YAML
     (image/resolution/origin), for known-map runs where /map was not published.

Together this is the visual counterpart of evaluate_run.py: it lets you SEE why
the planner chose what it did and how that evolved cycle to cycle, and where the
executed path deviated from the Nav2-planned path.

Outputs PNG frames into <run_dir>/frames/ (one per plan cycle) plus
overview.png. If --gif is given and imageio is available, also assembles a
timelapse.

Usage:
    visualize_run.py runs/lab05_known_map_run1                  # frames + overview
    visualize_run.py runs/lab05_known_map_run1 --gif            # also a timelapse
    visualize_run.py runs/... --map-yaml maps/lab05.yaml        # static map underlay
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.colors  # noqa: E402  (ListedColormap for the covered overlay)
import matplotlib.patches  # noqa: E402  (legend proxy for the covered overlay)
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _safe_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _xy(rows, upto_plan=None):
    xs, ys = [], []
    for r in rows:
        if upto_plan is not None:
            try:
                if int(float(r.get("plan_id", 0))) > upto_plan:
                    continue
            except (ValueError, TypeError):
                pass
        x, y = _f(r, "x_m"), _f(r, "y_m")
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


# Map underlay
def load_map_underlay(run_dir: Path, map_yaml: str | None):
    """(image_array, extent) for imshow, or (None, None).

    image_array is grayscale with occupied dark; extent = (xmin, xmax, ymin,
    ymax) in map-frame metres so plots in metres overlay directly.
    """
    # 1) The recorder's snapshot of /map (SLAM or map_server-published).
    # OccupancyGrid row 0 = world bottom; imshow origin='lower' handles it.
    img, extent = _load_map_npy(run_dir / "map_final.npy",
                                run_dir / "map_meta.yaml")
    if img is not None:
        return img, extent

    # 2) A standard map_server YAML (image/resolution/origin).
    if map_yaml:
        ypath = Path(map_yaml)
        meta = yaml.safe_load(ypath.read_text()) or {}
        img_path = Path(meta["image"])
        if not img_path.is_absolute():
            img_path = ypath.parent / img_path
        img = plt.imread(str(img_path))
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        img = np.flipud(img)  # image row 0 = top; flip so row 0 = world bottom
        if img.max() > 1.0:
            img = img / 255.0
        res = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        h, w = img.shape
        extent = (ox, ox + w * res, oy, oy + h * res)
        return img, extent

    return None, None


def _load_mask_npy(npy: Path, meta_path: Path):
    """(bool_mask, extent) from a covered_mask .npy + meta, or (None, None)."""
    if not (npy.is_file() and meta_path.is_file()):
        return None, None
    grid = np.load(npy)
    meta = yaml.safe_load(meta_path.read_text()) or {}
    res = float(meta["resolution"])
    ox, oy = float(meta["origin_x"]), float(meta["origin_y"])
    h, w = grid.shape
    mask = grid > 0                                  # observed cells
    extent = (ox, ox + w * res, oy, oy + h * res)
    return mask, extent


def _load_map_npy(npy: Path, meta_path: Path):
    """(gray_img, extent) from a /map .npy + meta, or (None, None)."""
    if not (npy.is_file() and meta_path.is_file()):
        return None, None
    grid = np.load(npy)  # OccupancyGrid values: -1 unknown, 0 free, 100 occ
    meta = yaml.safe_load(meta_path.read_text()) or {}
    res = float(meta["resolution"])
    ox, oy = float(meta["origin_x"]), float(meta["origin_y"])
    h, w = grid.shape
    img = np.where(grid < 0, 0.5, 1.0 - grid / 100.0)  # free→1, occ→0, unk→0.5
    extent = (ox, ox + w * res, oy, oy + h * res)
    return img, extent


def load_covered_mask(run_dir: Path):
    """(bool_mask, extent) of observed cells from covered_mask_final.npy."""
    return _load_mask_npy(run_dir / "covered_mask_final.npy",
                          run_dir / "covered_mask_meta.yaml")


def load_cycle_snapshots(run_dir: Path):
    """{plan_id: {'map': (img, extent) | None, 'covered': (mask, extent) | None}}.

    Reads the recorder's per-plan-cycle snapshots from maps_completion/ so the
    visualiser can show coverage growing cycle by cycle. Empty dict when the run
    predates this feature (callers then fall back to the single final snapshot)."""
    d = run_dir / "maps_completion"
    if not d.is_dir():
        return {}
    out: dict[int, dict] = {}
    for npy in sorted(d.glob("covered_mask_*.npy")):
        pid = _safe_int(npy.stem.rsplit("_", 1)[-1])
        if pid is None:
            continue
        out.setdefault(pid, {})["covered"] = _load_mask_npy(
            npy, d / f"covered_mask_{pid:03d}_meta.yaml")
    for npy in sorted(d.glob("map_*.npy")):
        pid = _safe_int(npy.stem.rsplit("_", 1)[-1])
        if pid is None:
            continue
        out.setdefault(pid, {})["map"] = _load_map_npy(
            npy, d / f"map_{pid:03d}_meta.yaml")
    return out


def _draw_covered(ax, covered_mask, covered_extent, rotate=False) -> None:
    """Translucent overlay of the observed (covered) cells."""
    if covered_mask is None or not covered_mask.any():
        return
    # Masked array so only observed cells are painted (rest fully transparent).
    overlay = np.ma.masked_where(~covered_mask, covered_mask.astype(float))
    ext = covered_extent
    if rotate:
        overlay = overlay.T
        if covered_extent is not None:
            ext = (covered_extent[2], covered_extent[3],
                   covered_extent[0], covered_extent[1])
    ax.imshow(overlay, cmap=matplotlib.colors.ListedColormap(["tab:cyan"]),
              extent=ext, origin="lower", zorder=1,
              alpha=0.45, interpolation="nearest")


def _draw_cycle(ax, motion, waypoints, nav_paths, plan_id,
                map_img=None, map_extent=None,
                covered_mask=None, covered_extent=None,
                fixed_xlim=None, fixed_ylim=None,
                nav_goals=None, rotate=False) -> None:
    # rotate=True swaps world X and Y so a tall/narrow map (e.g. hospital) is
    # drawn landscape: world-Y runs along the horizontal axis, world-X vertical.
    # Every coordinate goes through XY(); the map image is transposed too.
    def XY(x, y):
        return (y, x) if rotate else (x, y)

    ax.set_title(f"plan cycle {plan_id}")
    # adjustable='box' keeps the axes limits we pin below (datalim would let
    # matplotlib expand them to fit content, drifting the frame between cycles).
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]" if rotate else "x [m]")
    ax.set_ylabel("x [m]" if rotate else "y [m]")

    # Map underlay (same map frame as every recorded position).
    if map_img is not None:
        img = map_img.T if rotate else map_img
        ext = map_extent
        if rotate and map_extent is not None:
            # (xmin,xmax,ymin,ymax) -> (ymin,ymax,xmin,xmax)
            ext = (map_extent[2], map_extent[3], map_extent[0], map_extent[1])
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0,
                  extent=ext, origin="lower", zorder=0,
                  interpolation="nearest")

    # Covered (observed) cells as a translucent overlay on top of the map.
    _draw_covered(ax, covered_mask, covered_extent, rotate=rotate)

    # Executed trajectory up to and including this cycle.
    tx, ty = _xy(motion, upto_plan=plan_id)
    if tx:
        hx, hy = (ty, tx) if rotate else (tx, ty)
        ax.plot(hx, hy, "-", color="0.4", lw=1.2, label="executed (map frame)")
        ax.plot(*XY(tx[-1], ty[-1]), "o", color="tab:blue", ms=7, label="robot")

    # Nav2 planned paths for this cycle. A run publishes one /plan per goal;
    # they all share plan_id in a manual run (single planning cycle), so draw
    # each goal's path as its OWN segment (split when seq resets to 0) instead
    # of one giant polyline that connects every goal end-to-start.
    np_rows = [r for r in nav_paths
               if _safe_int(r.get("plan_id")) == plan_id]
    if np_rows:
        labelled = False
        seg_x, seg_y = [], []
        prev_seq = None
        for r in np_rows:
            seq = _safe_int(r.get("seq"))
            if seq is not None and prev_seq is not None and seq <= prev_seq:
                # new plan starts: flush the accumulated segment
                gx, gy = (seg_y, seg_x) if rotate else (seg_x, seg_y)
                ax.plot(gx, gy, "-", color="tab:green", lw=0.6, alpha=0.35,
                        label=None if labelled else "Nav2 planned paths")
                labelled = True
                seg_x, seg_y = [], []
            seg_x.append(_f(r, "x_m"))
            seg_y.append(_f(r, "y_m"))
            prev_seq = seq
        if seg_x:
            gx, gy = (seg_y, seg_x) if rotate else (seg_x, seg_y)
            ax.plot(gx, gy, "-", color="tab:green", lw=0.6, alpha=0.35,
                    label=None if labelled else "Nav2 planned paths")

    # Strategy waypoints / current goal for this cycle, in intended visit order
    # (the 'rank' column is the order the strategy plans to reach them).
    wp = [r for r in waypoints
          if _safe_int(r.get("plan_id")) == plan_id and r.get("kind") == "waypoint"]
    wp.sort(key=lambda r: (_safe_int(r.get("rank")) if _safe_int(r.get("rank"))
                           is not None else 1_000_000))
    if wp:
        wx = [_f(r, "x_m") for r in wp]
        wy = [_f(r, "y_m") for r in wp]
        # Dashed line linking the waypoints in the order we intend to reach them,
        # anchored at the robot's current position so it reads as the planned route.
        route_x = ([tx[-1]] + wx) if tx else wx
        route_y = ([ty[-1]] + wy) if ty else wy
        rox, roy = (route_y, route_x) if rotate else (route_x, route_y)
        ax.plot(rox, roy, "--", color="tab:orange", lw=1.0, alpha=0.7,
                zorder=4, label="intended order")
        sx, sy = (wy, wx) if rotate else (wx, wy)
        ax.scatter(sx, sy, c="tab:orange", s=40, marker="^",
                   label="waypoints", zorder=5)
        # Order number as text just beside each point.
        for i, (x, y) in enumerate(zip(wx, wy)):
            ax.annotate(str(i), XY(x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color="darkorange",
                        fontweight="bold", zorder=7)
    goal = [r for r in waypoints
            if _safe_int(r.get("plan_id")) == plan_id and r.get("kind") == "current_goal"]
    if goal:
        g = goal[-1]
        gxv, gyv = XY(_f(g, "x_m"), _f(g, "y_m"))
        ax.scatter([gxv], [gyv], c="tab:red", s=90,
                   marker="*", label="current goal", zorder=6)

    # Points the robot ACTUALLY reached (SUCCEEDED nav goals), cumulative up to
    # this cycle, in the order it visited them. This is the realised tour, which
    # can differ from the plan (the orange markers above are only the CURRENT
    # cycle's intended waypoints — mid-path replans mean many are never reached).
    if nav_goals:
        reached = [r for r in nav_goals
                   if r.get("status") == "SUCCEEDED"
                   and _safe_int(r.get("plan_id")) is not None
                   and _safe_int(r.get("plan_id")) <= plan_id]
        # nav_goals.csv is written in result order, i.e. visit order already.
        rx = [_f(r, "goal_x_m") for r in reached]
        ry = [_f(r, "goal_y_m") for r in reached]
        rx = [x for x in rx if x is not None]
        ry = [y for y in ry if y is not None]
        if rx:
            hrx, hry = (ry, rx) if rotate else (rx, ry)
            ax.plot(hrx, hry, "-", color="tab:purple", lw=1.3, alpha=0.8,
                    zorder=5, label="reached (actual order)")
            ax.scatter(hrx, hry, c="tab:purple", s=28, marker="o", zorder=6)
            for i, (x, y) in enumerate(zip(rx, ry)):
                ax.annotate(str(i), XY(x, y), textcoords="offset points",
                            xytext=(4, -8), fontsize=6, color="purple",
                            zorder=7)

    handles, labels = ax.get_legend_handles_labels()
    if covered_mask is not None and covered_mask.any():
        # imshow adds no legend entry; add a proxy patch for the overlay.
        handles.append(matplotlib.patches.Patch(
            facecolor="tab:cyan", alpha=0.45, label="covered (observed)"))
        labels.append("covered (observed)")
    ax.legend(handles, labels, loc="upper right", fontsize=7)

    # Pin the view to the final-map extent so the frame (and its centre) stays
    # fixed across cycles — the coverage/map grow inside a stable frame instead
    # of the axes rescaling every cycle.
    xlim, ylim = fixed_xlim, fixed_ylim
    if rotate:
        xlim, ylim = fixed_ylim, fixed_xlim   # axes are swapped
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _coverage_points(motion, cell_area_m2=None):
    """[(time_s, path_m, coverage_pct, covered_area_m2_or_None), ...].

    covered_area is absolute observed free area (covered_cells * cell_area) —
    monotonic even under SLAM, unlike the coverage RATIO whose denominator
    (map_free_cells) grows as new space is discovered."""
    pts = []
    total = 0.0
    prev = None
    for r in motion:
        t = _f(r, "timestamp_s")
        x, y, cov = _f(r, "x_m"), _f(r, "y_m"), _f(r, "coverage")
        cc = _f(r, "covered_cells")
        if x is not None and y is not None:
            if prev is not None:
                total += ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5
            prev = (x, y)
        if t is not None and cov is not None:
            area = cc * cell_area_m2 if (cc is not None and cell_area_m2) else None
            pts.append((t, total, cov, area))
    return pts


def _coverage_curves(ax_path, ax_time, ax_area, motion, time_source="?",
                     cell_area_m2=None) -> None:
    """Coverage% vs path, coverage% vs time, and absolute covered area vs time.

    The third panel exists because under SLAM the coverage RATIO can dip (the
    free-space denominator grows), which reads as a sawtooth; absolute covered
    area only ever grows, so it shows the true progress unambiguously."""
    pts = _coverage_points(motion, cell_area_m2)
    if pts:
        ax_path.plot([p[1] for p in pts], [100 * p[2] for p in pts],
                     "-", color="tab:blue")
        ax_time.plot([p[0] for p in pts], [100 * p[2] for p in pts],
                     "-", color="tab:purple")
        areas = [(p[0], p[3]) for p in pts if p[3] is not None]
        if areas:
            ax_area.plot([a[0] for a in areas], [a[1] for a in areas],
                         "-", color="tab:green")

    ax_path.set_title("coverage vs path")
    ax_path.set_xlabel("path travelled [m]")
    ax_path.set_ylabel("coverage [%]")
    ax_path.set_ylim(0, 100)

    ax_time.set_title(f"coverage vs time ({time_source} time)")
    ax_time.set_xlabel("time [s]")
    ax_time.set_ylabel("coverage [%]")
    ax_time.set_ylim(0, 100)

    ax_area.set_title(f"covered area vs time ({time_source} time)")
    ax_area.set_xlabel("time [s]")
    ax_area.set_ylabel("covered area [m²]")

    for ax in (ax_path, ax_time, ax_area):
        ax.grid(True, alpha=0.3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visualise an exploration run over time")
    parser.add_argument("run_dir", help="Path to a per-run folder from recorder.py")
    parser.add_argument("--map-yaml", default=None,
                        help="map_server YAML for the static map underlay "
                             "(fallback when the run has no map_final.npy)")
    parser.add_argument("--video", action="store_true",
                        help="Assemble a timelapse mp4 (pausable/seekable)")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="Timelapse mp4 frame rate (frames per second; default 2)")
    parser.add_argument("--rotate", choices=["auto", "on", "off"], default="auto",
                        help="Rotate the map 90° (world-Y on x-axis) so a tall/narrow "
                             "map reads landscape. 'auto' rotates when the map is "
                             "clearly taller than wide (default).")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    motion = _read_csv(run_dir / "motion.csv")
    waypoints = _read_csv(run_dir / "published_waypoints.csv")
    nav_paths = _read_csv(run_dir / "nav2_paths.csv")
    nav_goals = _read_csv(run_dir / "nav_goals.csv")
    meta_path = run_dir / "meta.yaml"
    meta = (yaml.safe_load(meta_path.read_text()) or {}) if meta_path.is_file() else {}

    map_img, map_extent = load_map_underlay(run_dir, args.map_yaml)
    if map_img is None:
        print("No map underlay (no map_final.npy in the run and no --map-yaml); "
              "plotting on a blank background.")

    # Cell area for the absolute covered-area curve (from the run's own mask
    # meta resolution; None -> that panel is left empty).
    mask_meta_path = run_dir / "covered_mask_meta.yaml"
    mask_meta = (yaml.safe_load(mask_meta_path.read_text()) or {}) \
        if mask_meta_path.is_file() else {}
    res = mask_meta.get("resolution")
    cell_area_m2 = float(res) ** 2 if res else None

    # Final covered (observed) cells — fallback overlay when a cycle has no
    # per-cycle snapshot (e.g. runs recorded before maps_completion/ existed).
    covered_mask, covered_extent = load_covered_mask(run_dir)

    # Per-plan-cycle snapshots (coverage growth). Empty for pre-feature runs.
    cycle_snaps = load_cycle_snapshots(run_dir)

    # Fixed frame: pin every cycle to the FINAL map extent so the centre never
    # shifts as the SLAM map (and coverage) grow. Fall back to the covered-mask
    # extent, then to None (auto) if the run has no map at all.
    fixed_extent = map_extent or covered_extent
    fixed_xlim = (fixed_extent[0], fixed_extent[1]) if fixed_extent else None
    fixed_ylim = (fixed_extent[2], fixed_extent[3]) if fixed_extent else None

    # Decide rotation: 'auto' rotates when the map is clearly taller than wide
    # (height > 1.3x width), so a tall map like hospital reads landscape.
    if args.rotate == "on":
        rotate = True
    elif args.rotate == "off":
        rotate = False
    elif fixed_extent:
        w = fixed_extent[1] - fixed_extent[0]
        h = fixed_extent[3] - fixed_extent[2]
        rotate = h > 1.3 * w
    else:
        rotate = False
    # Landscape frame when rotated so the now-horizontal long axis has room.
    frame_size = (10, 6) if rotate else (7, 7)

    plan_ids = sorted({_safe_int(r.get("plan_id")) for r in motion} - {None})
    if not plan_ids:
        plan_ids = [0]

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths = []
    for pid in plan_ids:
        # Prefer this cycle's own map/coverage snapshot; fall back to the final
        # ones so pre-feature runs still render (just without the growth).
        snap = cycle_snaps.get(pid, {})
        snap_map = snap.get("map")
        if snap_map and snap_map[0] is not None:
            c_map_img, c_map_extent = snap_map
        else:
            c_map_img, c_map_extent = map_img, map_extent
        snap_cov = snap.get("covered")
        if snap_cov and snap_cov[0] is not None:
            c_cov_mask, c_cov_extent = snap_cov
        elif not cycle_snaps:
            c_cov_mask, c_cov_extent = covered_mask, covered_extent
        else:
            c_cov_mask, c_cov_extent = None, None
        fig, ax = plt.subplots(figsize=frame_size)
        _draw_cycle(ax, motion, waypoints, nav_paths, pid,
                    c_map_img, c_map_extent, c_cov_mask, c_cov_extent,
                    fixed_xlim=fixed_xlim, fixed_ylim=fixed_ylim,
                    nav_goals=nav_goals, rotate=rotate)
        fp = frames_dir / f"cycle_{pid:03d}.png"
        # No bbox_inches='tight': a per-frame tight crop would resize frames
        # differently and defeat the fixed frame / stable mp4 dimensions.
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        frame_paths.append(fp)

    # Overview: final map state (top, wide) + three curves (bottom row):
    # coverage% vs path, coverage% vs time, absolute covered area vs time.
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])
    ax_map = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    ax_a = fig.add_subplot(gs[1, 2])
    _draw_cycle(ax_map, motion, waypoints, nav_paths, plan_ids[-1],
                map_img, map_extent, covered_mask, covered_extent,
                nav_goals=nav_goals, rotate=rotate)
    ax_map.set_title("final state (all cycles)")
    _coverage_curves(ax_p, ax_t, ax_a, motion, meta.get("time_source", "?"),
                     cell_area_m2)
    fig.suptitle(run_dir.name)
    fig.savefig(run_dir / "overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(frame_paths)} cycle frames + overview.png to {run_dir}")

    if args.video:
        try:
            import imageio.v2 as imageio
            import numpy as np
            images = [imageio.imread(fp) for fp in frame_paths]
            # H.264 requires even width/height; crop the last row/column if odd.
            images = [im[: im.shape[0] & ~1, : im.shape[1] & ~1] for im in images]
            out = run_dir / "timelapse.mp4"
            # macro_block_size=None avoids a second silent resize on top of our crop.
            imageio.mimsave(out, images, fps=args.video_fps,
                            codec="libx264", macro_block_size=None)
            print(f"Wrote timelapse.mp4 ({len(images)} frames @ {args.video_fps} fps)")
        except ImportError:
            print("imageio / imageio-ffmpeg not available; skipping video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
