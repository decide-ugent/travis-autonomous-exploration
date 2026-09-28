#!/usr/bin/env python3
"""
A/B — MidPathReplanner replan-ratio (exploration refactor §3).

`MidPathReplanner.check` aborts the current path and replans mid-travel when some
OTHER waypoint's navigable distance drops below `dist_to_target * RATIO`. Today
RATIO is a hardcoded magic `0.5` (only divert when an alternative is ≥2x closer).
This sweeps RATIO to see whether the constant matters and what it should be — WITH
vs WITHOUT diversion — and renders the agent path per ratio so the effect is visible.

Test-first: this does NOT edit exploration/. It monkeypatches
`MidPathReplanner.check` in the SUT layer only (a black-box knob), reusing the
validated machinery from `test_FOV_visited_candidates.py` (independent SLAM
loader + ray-cast scorer, the shipped FOV-aware `_FOV` session at step 4.5, the
run loop, `compute_metrics`). FOV-mark + step 4.5 are treated as fixed.

Ratios swept:
  0.0   -> the `< dist*0.0` test can never fire  => diversion effectively OFF
           (mid-path replan only on target-unreachable), i.e. "WITHOUT"
  0.5   -> production default ("WITH", 2x closer)
  0.75  -> more eager to divert
  1.0   -> divert as soon as ANY waypoint is strictly closer than the target

Usage:
  python test_midpath_replan_ratio.py --maps lab_ghent
  python test_midpath_replan_ratio.py            # all 3 maps, all ratios, + visuals
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import test_FOV_visited_candidates as H
import exploration.execution_strategy as ES
from exploration.explore_costmap_map import navigable_distance_map

OUT_DIR = H.OUT_DIR.parent / "test_midpath_replan_ratio_out"
RATIOS = [0.0, 0.5, 0.75, 1.0]     # 0.0 == diversion OFF ("without")


def _patched_check(ratio: float):
    """Return a MidPathReplanner.check that uses `ratio` instead of the hardcoded
    0.5. Identical logic otherwise (copied from execution_strategy so the sweep is
    an honest black-box knob, not a behaviour rewrite)."""
    def check(self, step, current_target, all_waypoints, navigable_mask):
        if self._last_check is None:
            self._last_check = step
            return False
        import math
        if math.hypot(step[0] - self._last_check[0],
                      step[1] - self._last_check[1]) < self._interval:
            return False
        self._last_check = step
        dist = navigable_distance_map(navigable_mask, step[0], step[1])
        dist_to_target = dist[current_target.row, current_target.col]
        if dist_to_target == np.inf:
            return True
        for wp in all_waypoints:
            if wp is current_target:
                continue
            if dist[wp.row, wp.col] < dist_to_target * ratio:
                return True
        return False
    return check


def _run(map_name: str, ratio: float, max_plans: int):
    ES.MidPathReplanner.check = _patched_check(ratio)   # SUT-layer knob only
    arm = f"ratio_{ratio}"
    H.ARMS[arm] = {"session": H._FOV, "step_m": 4.5}
    world = H.load_true_world(map_name)
    tr = H.run_arm(map_name, world, arm=arm, capture_frames=False, max_plans=max_plans)
    return world, tr, H.compute_metrics(map_name, arm, tr, world)


def _render_paths(map_name: str, world, traces: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bg = np.where(world.occupied, 0.0, np.where(world.free, 1.0, 0.5))
    n = len(traces)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (label, tr) in zip(axes, traces.items()):
        ax.imshow(bg, cmap="gray", origin="upper")
        cov = np.ma.masked_where(~tr.covered, tr.covered)
        ax.imshow(cov, cmap="autumn", alpha=0.45, origin="upper")
        if tr.visited_px:
            xs = [p[0] for p in tr.visited_px]
            ys = [p[1] for p in tr.visited_px]
            ax.plot(xs, ys, "-o", color="deepskyblue", ms=4, lw=1)
            ax.plot(xs[0], ys[0], "s", color="lime", ms=9)
        ax.set_title(f"{map_name} — ratio={label}\n"
                     f"cov={tr.final_coverage:.1%}  path={tr.path_length_m:.0f}m  "
                     f"wps={tr.n_visited}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{map_name}_paths.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+",
                    default=["lab_05", "lab_ghent", "warehouse_amazon"])
    ap.add_argument("--max-plans", type=int, default=200)
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name in args.maps:
        print(f"\n=== {name} — MidPathReplanner ratio sweep (SLAM) ===")
        print("  ratio  cov%   path_m  wps  cov/m   holes%  premat  plans")
        traces = {}
        for ratio in RATIOS:
            world, tr, m = _run(name, ratio, args.max_plans)
            traces[str(ratio)] = tr
            all_metrics.append(m)
            tag = "  (OFF)" if ratio == 0.0 else ("  (prod)" if ratio == 0.5 else "")
            print(f"  {ratio:<5} {m['final_coverage']*100:5.1f}  {m['path_length_m']:6.0f}  "
                  f"{m['n_visited']:3d}  {m['coverage_per_m']:.4f}  "
                  f"{m['coverage_holes_frac']*100:5.1f}   {m['premature_stops']}      "
                  f"{m['n_plans']}{tag}")
        if not args.no_viz:
            _render_paths(name, world, traces)
            print(f"  path visuals -> {OUT_DIR}/{name}_paths.png")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nmetrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
