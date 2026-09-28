#!/usr/bin/env python3
"""
Replan-cadence sweep — `replan_every_n_step` (exploration refactor §11).

Question: under SLAM the node drains the ENTIRE waypoint plan before replanning, so
waypoints 2..N are scored against the map as it was N arrivals ago (stale). Does
replanning every N arrivals — instead of draining the whole plan — shorten the path
/ cut redundant revisiting at equal coverage?

This is a THIN DRIVER. It imports and reuses the already-validated, production-faithful
machinery from `test_FOV_visited_candidates.py` (independent SLAM loader + ray-cast
scorer, the shipped FOV-aware mark = `_FOV`, sampling_step 4.5, the `replan_k` drain
cadence in `run_arm`, and `compute_metrics`). It adds NOTHING to the system-under-test
or the scorer — it only registers arms with different `replan_k` values and tabulates.

The FOV-mark + step-4.5 choice is the DEFAULT starting point, not a frozen constant:
those were validated only under drain-whole-plan, so this sweep is explicitly allowed
to RE-QUESTION them. `--cross-check` re-runs step 3.0 vs 4.5 at the winning N so we can
confirm (or overturn) 4.5 under the new cadence rather than defend it.

Decision metrics (all from compute_metrics):
  primary : path_length_m (down), coverage_per_m (up)
  guards  : final_coverage >= drain baseline, coverage_holes_frac not worse
            n_plans / total_plan_time_s (compute cost must not run away)
            premature_stops == 0, n_visited not inflated, redundant_coverage_frac
            not worse (the old pure execute-1 DITHERING failure mode)

Usage:
  python test_replan_N.py                       # full sweep, 3 maps
  python test_replan_N.py --maps lab_ghent      # one map
  python test_replan_N.py --cross-check         # + step 3.0 vs 4.5 at winning N
"""
from __future__ import annotations

import argparse
import json

import test_FOV_visited_candidates as H   # validated machinery (import, don't fork)

OUT_DIR = H.OUT_DIR.parent / "test_replan_N_out"

# Replan cadences to sweep. None = drain the whole plan (production today = baseline).
# Each arm = the shipped FOV-aware session (_FOV) at step 4.5, differing ONLY in cadence.
_CADENCES = {
    "drain":    {"session": H._FOV, "step_m": 4.5},                  # replan when plan exhausted
    "every_1":  {"session": H._FOV, "step_m": 4.5, "replan_k": 1},   # execute-1-replan
    "every_2":  {"session": H._FOV, "step_m": 4.5, "replan_k": 2},
    "every_3":  {"session": H._FOV, "step_m": 4.5, "replan_k": 3},
    "every_4":  {"session": H._FOV, "step_m": 4.5, "replan_k": 4},
}


def _run(map_name: str, arm: str, spec: dict, max_plans: int, known_map: bool = False):
    H.ARMS[arm] = spec
    world = H.load_true_world(map_name)
    tr = H.run_arm(map_name, world, arm=arm, capture_frames=False,
                   max_plans=max_plans, known_map=known_map)
    return H.compute_metrics(map_name, arm, tr, world)


def _print_row(label: str, m: dict):
    print(f"  {label:10s} cov={m['final_coverage']*100:5.1f}%  wps={m['n_visited']:3d}  "
          f"path={m['path_length_m']:6.0f}m  cov/m={m['coverage_per_m']:.4f}  "
          f"holes={m['coverage_holes_frac']*100:4.1f}%  redund={m['redundant_coverage_frac']*100:3.0f}%  "
          f"premat={m['premature_stops']}  plans={m['n_plans']}  "
          f"plan_t={m['total_plan_time_s']:5.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+",
                    default=["lab_05", "lab_ghent", "warehouse_amazon"])
    ap.add_argument("--max-plans", type=int, default=250,
                    help="high cap: small-N cadences need many more plan cycles")
    ap.add_argument("--cross-check", action="store_true",
                    help="at the best N, re-test step 3.0 vs 4.5 (re-question prior choice)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for name in args.maps:
        print(f"\n=== {name} (SLAM) — replan cadence sweep ===")
        rows = {}
        for arm, spec in _CADENCES.items():
            m = _run(name, f"{arm}", spec, args.max_plans)
            rows[arm] = m
            all_metrics.append(m)
            _print_row(arm, m)

        # Known-map control: with no frontiers, cadence must be a no-op (drain == every_2).
        print(f"  --- {name} known-map control (cadence should not matter) ---")
        for arm in ("drain", "every_2"):
            m = _run(name, f"{arm}_known", _CADENCES[arm], args.max_plans, known_map=True)
            all_metrics.append(m)
            _print_row(f"{arm}(known)", m)

        if args.cross_check:
            # Re-question step 3.0 vs 4.5 under the frequent-replan cadence (every_2).
            print(f"  --- {name} cross-check: step 3.0 vs 4.5 @ every_2 ---")
            for step in (3.0, 4.5):
                spec = {"session": H._FOV, "step_m": step, "replan_k": 2}
                m = _run(name, f"cc_step{step}", spec, args.max_plans)
                all_metrics.append(m)
                _print_row(f"step{step}", m)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nmetrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
