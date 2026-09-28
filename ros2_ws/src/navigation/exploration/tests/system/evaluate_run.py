#!/usr/bin/env python3
"""
Evaluate one exploration run folder (system-test layer, L3).

Pure Python, no ROS. Reads a per-run folder produced by recorder.py plus the
scene's combined config+baseline YAML, then:

  1. computes the KPIs (coverage curve milestones, path length, coverage rate,
     idle time, waypoint count/density, ...),
  2. checks the absolute pass/fail gates from <mode>.gates,
  3. tallies Nav2 goal outcomes (succeeded / aborted / canceled),
  4. writes report.md into the run folder, and prints PASS/FAIL,
  5. optionally saves the run as the scene baseline (--save-baseline) or diffs
     against the saved baseline (--compare).

Time base: timestamp_s in the CSVs is the recorder's node clock — simulation
time when the recorder ran with use_sim_time (meta.yaml time_source: sim).
Time KPIs are therefore comparable across simulators with different real-time
factors; the report states the time source.

Areas: computed from the run's own grid metadata (covered_mask_meta.yaml
written by recorder.py) — no hard-coded resolution. navigable_area_m2 comes
from the free cells of the recorded map (map_final.npy), when present.

Usage:
    # grade a run against the scene gates
    evaluate_run.py runs/lab05_known_map_run1_2026... \
        --config baselines/baseline_lab05.yaml --mode known_map

    # save this run (e.g. the manual reference) as the baseline
    evaluate_run.py runs/lab05_known_map_run0_manual \
        --config baselines/baseline_lab05.yaml --mode known_map \
        --save-baseline --source manual_exploration

    # compare a new run against the saved baseline
    evaluate_run.py runs/lab05_known_map_run5_... \
        --config baselines/baseline_lab05.yaml --mode known_map --compare
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np
import yaml

from stuck_analysis import stuck_for_run

# Human-readable "why the run stopped" for each stop_reason from the exploration
# log. Used when a run did NOT terminate normally (coverage_complete).
_STOP_REASON_TEXT = {
    "stuck_in_inflation":
        "stopped: robot wedged in Nav2's inflation band past plan_timeout_s "
        "(Nav2 could not plan from that pose)",
    "wedged_plans_failed":
        "stopped: every waypoint was inaccessible for enough consecutive plans "
        "(robot wedged, recovery spins did not free it)",
    "candidate_pool_exhausted":
        "stopped: no waypoints could be planned and the candidate pool could "
        "not be recovered (raise plan_timeout_s to retry longer)",
    "no_progress":
        "stopped: several consecutive waypoints each added < the no-progress "
        "coverage threshold (diminishing returns)",
    "coverage_complete":
        "reached the coverage/no-frontier completion condition",
}


# CSV loading
def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str):
    """Float or None from a CSV cell."""
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# KPI computation
def _path_length_m(motion: list[dict]) -> float:
    """Integrate executed path from successive motion x_m/y_m samples."""
    total = 0.0
    prev = None
    for r in motion:
        x, y = _f(r, "x_m"), _f(r, "y_m")
        if x is None or y is None:
            continue
        if prev is not None:
            total += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
    return total


def _coverage_series(motion: list[dict]) -> list[tuple[float, float, float]]:
    """List of (timestamp_s, cumulative_path_m, coverage) along the run."""
    series = []
    total = 0.0
    prev = None
    for r in motion:
        t = _f(r, "timestamp_s")
        cov = _f(r, "coverage")
        x, y = _f(r, "x_m"), _f(r, "y_m")
        if x is not None and y is not None:
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        if t is not None and cov is not None:
            series.append((t, total, cov))
    return series


def _coverage_series_fixed(motion: list[dict]) -> list[tuple[float, float, float]]:
    """Coverage series against a FIXED denominator: covered_cells / final free cells.

    The recorded `coverage` column is covered / ACHIEVABLE-so-far, where the
    achievable set is only the part of the map SLAM has discovered at that
    instant. Early on, the robot has mapped a few m² around spawn, so covering
    those few cells reads as e.g. 23% at t=1s having moved 0 m — impossible on a
    >1000 m² map with an 8 m lidar. It is 23% of what was seen, not of the
    building.

    covered_cells is monotonic and map_free_cells grows as SLAM discovers the
    map; dividing the running covered_cells by the FINAL free-cell count gives
    the honest progressive fraction of the eventually-known map, so early
    milestones are physically meaningful. Falls back to the recorded ratio when
    the covered_cells / map_free_cells columns are absent.
    """
    denom = 0.0
    for r in motion:
        mf = _f(r, "map_free_cells")
        if mf is not None:
            denom = max(denom, mf)
    if denom <= 0:
        # No covered_cells/map_free_cells columns (older recorder, or known-map
        # runs). Fall back to the recorded `coverage`. In known-map mode that
        # ratio already has a fixed denominator, so it is honest; only live-SLAM
        # runs suffer the growing-denominator distortion this function corrects.
        return _coverage_series(motion), False

    series = []
    total = 0.0
    prev = None
    for r in motion:
        t = _f(r, "timestamp_s")
        cc = _f(r, "covered_cells")
        x, y = _f(r, "x_m"), _f(r, "y_m")
        if x is not None and y is not None:
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        if t is not None and cc is not None:
            series.append((t, total, cc / denom))
    return series, True


def _milestone(series, target_cov, idx):
    """First (path_m or time_s) at which coverage >= target. idx: 1=path, 0=time."""
    for t, path, cov in series:
        if cov >= target_cov:
            return path if idx == 1 else t
    return None


def _milestone_curve(series, levels=None):
    """Full path/time-to-coverage curve, not just 50/90%.

    A run limited by Nav2 obstacle inflation can never physically reach the whole
    map, so 90% is often unreachable and the two-point 50/90 summary hides where
    the run actually plateaued. This returns one row per coverage level plus the
    peak the run truly hit, so the plateau is visible.

    Returns (rows, max_cov, path_at_max, time_at_max) where each row is
    (level, path_m_or_None, time_s_or_None); path/time is None for levels the
    run never reached.
    """
    if levels is None:
        levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                  0.70, 0.80, 0.90, 0.95]
    rows = [(lvl, _milestone(series, lvl, 1), _milestone(series, lvl, 0))
            for lvl in levels]
    if series:
        max_cov = max(cov for _, _, cov in series)
        # path/time at the sample where the peak coverage was first reached
        path_at_max = _milestone(series, max_cov, 1)
        time_at_max = _milestone(series, max_cov, 0)
    else:
        max_cov = path_at_max = time_at_max = None
    return rows, max_cov, path_at_max, time_at_max


def _idle_time_s(motion: list[dict], move_eps_m: float = 0.01) -> float:
    """Time across samples where the robot did not move (plan latency /
    human hesitation)."""
    idle = 0.0
    prev = None
    for r in motion:
        t, x, y = _f(r, "timestamp_s"), _f(r, "x_m"), _f(r, "y_m")
        if t is None or x is None or y is None:
            continue
        if prev is not None:
            dt = t - prev[0]
            moved = math.hypot(x - prev[1], y - prev[2])
            if dt > 0 and moved < move_eps_m:
                idle += dt
        prev = (t, x, y)
    return idle


def _area_stats(run_dir: Path):
    """(covered_area_m2, navigable_area_m2) from the run's own recorded grids.

    Resolution comes from covered_mask_meta.yaml (written by recorder.py), never
    hard-coded. navigable_area_m2 counts the free cells (value 0) of the
    recorded map (map_final.npy); None when the run has no map snapshot."""
    covered_area = navigable_area = None

    mask_path = run_dir / "covered_mask_final.npy"
    mask_meta = _load_yaml(run_dir / "covered_mask_meta.yaml")
    if mask_path.is_file() and mask_meta.get("resolution"):
        mask = np.load(mask_path)
        res = float(mask_meta["resolution"])
        covered_area = int((mask > 0).sum()) * res ** 2

    map_path = run_dir / "map_final.npy"
    map_meta = _load_yaml(run_dir / "map_meta.yaml")
    if map_path.is_file() and map_meta.get("resolution"):
        grid = np.load(map_path)
        res = float(map_meta["resolution"])
        navigable_area = int((grid == 0).sum()) * res ** 2

    return covered_area, navigable_area


def _count_waypoints(waypoints: list[dict]) -> int | None:
    """Distinct planned waypoints across the run.

    The node republishes the full waypoint MarkerArray every cycle, so raw row
    counts are inflated; count unique (plan_id, x, y) instead. None when the
    run published no waypoints at all (e.g. a manual run)."""
    seen = {(r.get("plan_id"), r.get("x_m"), r.get("y_m"))
            for r in waypoints if r.get("kind") == "waypoint"}
    return len(seen) or None


def _planning_durations_s(plans: list[dict]) -> list[float]:
    """Duration (s) of each PLANNING phase from the state timeline.

    plans.csv logs the state every timer tick; a planning cycle spans from the
    first PLANNING row to the first row of the next (non-PLANNING) state. The
    last cycle is closed by the final row if the run ended while planning.
    Empty for manual runs (state is only MANUAL)."""
    durations = []
    start_t = None
    for r in plans:
        state = (r.get("state") or "").upper()
        t = _f(r, "timestamp_s")
        if t is None:
            continue
        if state == "PLANNING":
            if start_t is None:
                start_t = t
        elif start_t is not None:
            durations.append(t - start_t)
            start_t = None
    if start_t is not None and plans:
        last_t = _f(plans[-1], "timestamp_s")
        if last_t is not None and last_t > start_t:
            durations.append(last_t - start_t)
    return durations


def _tally_nav_goals(nav_goals: list[dict]) -> dict:
    counts = {"succeeded": 0, "aborted": 0, "canceled": 0, "other": 0,
              "total": 0}
    for g in nav_goals:
        counts["total"] += 1
        status = (g.get("status") or "").upper()
        key = status.lower()
        counts[key if key in counts else "other"] += 1
    return counts


# Metrics assembly
def compute_metrics(run_dir: Path, meta: dict,
                    baseline_area_m2: float | None = None) -> dict:
    motion = _read_csv(run_dir / "motion.csv")
    plans = _read_csv(run_dir / "plans.csv")
    waypoints = _read_csv(run_dir / "published_waypoints.csv")
    nav_goals = _read_csv(run_dir / "nav_goals.csv")

    series = _coverage_series(motion)
    # Honest progressive coverage (fixed final denominator) for the milestone
    # curve, so early % are physical rather than "% of what SLAM had seen so far".
    # fixed_denom is False when the run lacks the cell-count columns and we fell
    # back to the recorded ratio (already fixed-denominator in known-map mode).
    series_fixed, fixed_denom = _coverage_series_fixed(motion)
    final_cov = series[-1][2] if series else 0.0
    path_len = _path_length_m(motion)

    total_waypoints = _count_waypoints(waypoints)

    covered_area, navigable_area = _area_stats(run_dir)
    density = (total_waypoints / navigable_area
               if total_waypoints and navigable_area else None)

    # Coverage referenced to the BASELINE's covered area instead of this run's
    # own SLAM-discovered map. The published `coverage` ratio has a per-run
    # denominator (whatever SLAM had discovered), so a run that under-discovers
    # the map gets a flattering ratio and runs are not comparable to each other.
    # Rescaling every run onto one fixed area (the baseline's covered_area_m2)
    # makes final coverage and the 50/90% milestones mean the same thing in
    # every run. Falls back to the run's own ratio when either area is missing.
    cov_series = series
    final_cov_base = None
    # The published ratio is covered/ACHIEVABLE (see coverage_ratio in
    # explore_costmap_map), so this run's achievable area is recovered from its
    # own end state: achievable = covered_area_m2 / final_coverage. Multiplying
    # the ratio by that converts it to absolute m² observed; dividing by the
    # baseline's covered area expresses every run on one fixed scale.
    # NOTE: navigable_area_m2 (free cells of map_final.npy) is NOT the same
    # quantity and must not be used here.
    if baseline_area_m2 and covered_area and final_cov > 0:
        achievable_area = covered_area / final_cov
        scale = achievable_area / baseline_area_m2
        cov_series = [(t, p, c * scale) for t, p, c in series]
        final_cov_base = round(cov_series[-1][2], 4) if cov_series else None

    # Whether the run reached the COMPLETE state (the must_terminate gate uses
    # this). total_run_time_s is ALWAYS filled: the duration up to the last
    # recorded sample, whether the run completed or was stopped by Ctrl-C. For
    # a completed run that last sample is the COMPLETE row, so it also equals
    # the time-to-completion; for a manual/aborted run it is the stop time.
    terminated = any(
        (p.get("state") or "").upper() == "COMPLETE" for p in plans)
    last_times = [t for t in (_f(p, "timestamp_s") for p in plans) if t is not None]
    if not last_times:
        last_times = [s[0] for s in series]  # fall back to motion timeline
    total_run_time = max(last_times) if last_times else None

    coverage_rate = (final_cov / path_len) if path_len > 0 else None

    # Planning-cycle timing (autonomous runs only; empty for manual).
    plan_durs = _planning_durations_s(plans)
    n_plans = len(plan_durs)
    plan_mean = statistics.fmean(plan_durs) if plan_durs else None
    plan_std = statistics.stdev(plan_durs) if len(plan_durs) > 1 else (
        0.0 if plan_durs else None)

    return {
        "final_coverage": round(final_cov, 4),
        # Same run, measured against the baseline's covered area (comparable
        # across runs); None when this run or the baseline lacks an area.
        "final_coverage_vs_baseline": final_cov_base,
        "covered_area_m2": _round(covered_area),
        "navigable_area_m2": _round(navigable_area),
        "path_length_m": round(path_len, 2),
        "coverage_rate": round(coverage_rate, 5) if coverage_rate else None,
        # Milestones use the baseline-referenced series when available, so "50%"
        # is the same absolute area in every run rather than 50% of whatever
        # that run happened to discover.
        "path_to_50pct_coverage_m": _round(_milestone(cov_series, 0.50, 1)),
        "path_to_90pct_coverage_m": _round(_milestone(cov_series, 0.90, 1)),
        "time_to_50pct_coverage_s": _round(_milestone(cov_series, 0.50, 0)),
        "time_to_90pct_coverage_s": _round(_milestone(cov_series, 0.90, 0)),
        # Full milestone curve (10..95% + the peak actually reached) so a run
        # capped by unreachable area shows where it plateaued, not just 50/90.
        # Uses the fixed-denominator series so early % reflect the whole map, not
        # the tiny area SLAM had discovered at that instant.
        # Non-scalar: rendered as its own report section, skipped by KPI tables.
        # (curve, fixed_denom) — fixed_denom flags which coverage definition was
        # used so the report note is accurate.
        "coverage_milestones": (_milestone_curve(series_fixed), fixed_denom),
        "terminated": terminated,
        "total_run_time_s": _round(total_run_time),
        "idle_time_s": round(_idle_time_s(motion), 2),
        "n_planning_cycles": n_plans,
        "planning_time_mean_s": _round(plan_mean, 3),
        "planning_time_std_s": _round(plan_std, 3),
        "total_waypoints": total_waypoints,
        "waypoints_per_m2": round(density, 4) if density else None,
        "nav_goals": _tally_nav_goals(nav_goals),
    }


def _round(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) else None


# The gate keys check_gates() actually honours. Used to prune dead gates from
# an old-schema YAML on save, so a hand-copied file converges to this schema.
_KNOWN_GATE_KEYS = (
    "final_coverage_min", "must_terminate", "time_to_complete_max_s",
    "nav_aborted_max",
)


# Gates — every gate here is actually checked; keep the YAML in sync.
# Each result is (name, ok, measured, comparator, threshold): measured and
# threshold are always shown side by side so a FAIL states the actual value
# against the required one, not just a broken assertion.
def check_gates(metrics: dict, gates: dict,
                baseline_time_s: float | None = None,
                time_tolerance_pct: float = 25.0,
                baseline_coverage: float | None = None,
                coverage_tolerance_abs: float = 0.02) -> list[tuple]:
    results = []

    def add(name, ok, measured, comparator, threshold):
        results.append((name, bool(ok), measured, comparator, threshold))

    # Coverage target: the baseline's ACHIEVED maximum coverage (minus a
    # tolerance), not an absolute 90%. A human driving the same scene is the
    # realistic ceiling — part of the map is geometrically unreachable under the
    # sensor model and Nav2's inflation radius, so 90% may be unattainable by
    # anyone and grades the strategy against an impossible target.
    #
    # The comparison is area-aware: final_coverage_vs_baseline expresses this
    # run's swept area on the BASELINE's covered-area scale (see
    # compute_metrics), so a run whose SLAM discovered more or less map than the
    # baseline is still measured against the same absolute area. Because that
    # metric is already normalised to the baseline (1.0 == matched it), the
    # threshold on that scale is 1.0 - tolerance. Falls back to the raw ratio vs
    # the config's final_coverage_min when no baseline coverage exists.
    fc_base = metrics.get("final_coverage_vs_baseline")
    if baseline_coverage and fc_base is not None:
        thr = 1.0 - coverage_tolerance_abs
        add("final_coverage", fc_base >= thr, f"{fc_base:.1%} of baseline",
            ">=", f"{thr:.0%} of baseline "
                  f"(baseline reached {baseline_coverage:.1%}, "
                  f"-{coverage_tolerance_abs:.0%})")
    elif baseline_coverage:
        # Areas missing (so no normalised metric) — compare the raw ratios
        # directly, still against the baseline's max rather than a flat 90%.
        fc = metrics["final_coverage"]
        thr = baseline_coverage - coverage_tolerance_abs
        add("final_coverage", fc >= thr, f"{fc:.1%}", ">=",
            f"{thr:.1%} (baseline {baseline_coverage:.1%} "
            f"-{coverage_tolerance_abs:.0%})")
    else:
        fc = metrics["final_coverage"]
        thr = gates.get("final_coverage_min", 0.90)
        add("final_coverage", fc >= thr, f"{fc:.1%}", ">=", f"{thr:.0%}")

    if gates.get("must_terminate", True):
        term = metrics["terminated"]
        add("must_terminate", term,
            "COMPLETE" if term else "no COMPLETE", "==", "COMPLETE")

    # Time budget: purely a time comparison, fully independent of whether the run
    # reached COMPLETE (that is the must_terminate gate's concern). Lower is better,
    # so any run within budget passes. Only missing timing data fails here.
    #
    # The budget is derived from the BASELINE's run time (a real reference for the
    # scene) plus a tolerance, not a hand-picked constant: budget =
    # baseline_time_s * (1 + time_tolerance_pct/100). The baseline's
    # total_run_time_s is filled even for a non-completed baseline (it is the
    # duration up to the last sample), so an interrupted baseline still gives a
    # usable reference. Only when no baseline time exists at all do we fall back
    # to the explicit time_to_complete_max_s gate value.
    trt = metrics["total_run_time_s"]
    if baseline_time_s:
        budget = baseline_time_s * (1.0 + time_tolerance_pct / 100.0)
        thr_label = f"{budget:.0f}s (baseline {baseline_time_s:.0f}s +{time_tolerance_pct:.0f}%)"
    else:
        budget = gates.get("time_to_complete_max_s")
        thr_label = f"{budget:.0f}s" if budget is not None else None
    if budget is not None:
        if trt is not None:
            add("time_to_complete", trt <= budget,
                f"{trt:.0f}s", "<=", thr_label)
        else:
            add("time_to_complete", False, "no data", "<=", thr_label)

    max_aborted = gates.get("nav_aborted_max")
    if max_aborted is not None:
        na = metrics["nav_goals"]["aborted"]
        add("nav_aborted", na <= max_aborted, str(na), "<=", str(max_aborted))

    return results


# Report
def write_report(run_dir: Path, meta: dict, metrics: dict, gate_results,
                 baseline_cmp, baseline: dict, plot_path) -> bool:
    passed = all(ok for _, ok, *_ in gate_results)
    base_src = baseline.get("source") or "?"
    base_folder = baseline.get("run_folder") or "?"

    lines = [f"# Exploration run report: {run_dir.name}", ""]
    lines.append(f"- scene: **{meta.get('scene', '?')}**  mode: **{meta.get('mode', '?')}**")
    lines.append(f"- run folder: `{run_dir.name}`")
    lines.append(f"- time source: **{meta.get('time_source', '?')}** "
                 "(sim = simulation clock, wall = laptop clock)")
    if baseline_cmp:
        lines.append(f"- baseline: **{base_src}** from `{base_folder}`")
    lines.append(f"- overall: **{'PASS' if passed else 'FAIL'}**")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| gate | measured | | threshold | result |")
    lines.append("|---|---|:-:|---|---|")
    for name, ok, measured, comparator, threshold in gate_results:
        lines.append(f"| {name} | {measured} | {comparator} | {threshold} "
                     f"| {'PASS' if ok else 'FAIL'} |")
    lines.append("")

    # Stuck episodes + why the run stopped, from logs/exploration.log (only
    # present when the run was launched via run_with_log.sh).
    stuck = stuck_for_run(run_dir)
    lines.append("## Stuck & termination")
    lines.append("")
    if stuck is None:
        lines.append("_No `logs/exploration.log` for this run "
                     "(not launched via run_with_log.sh) — stuck/stop data "
                     "unavailable._")
    else:
        lines.append(f"- human unstuck assists: **{stuck['human_unstuck']}** "
                     "(robot could not free itself — moved to clear space by a human)")
        lines.append(f"- self-recovered (Nav2): **{stuck['self_recovered']}** "
                     "(got stuck but freed itself, e.g. via a recovery spin)")
        if stuck["ended_stuck"]:
            lines.append(f"- ended while stuck: **{stuck['ended_stuck']}** "
                         "(run stopped wedged, never freed)")
        if not metrics.get("terminated"):
            reason = stuck.get("stop_reason")
            expl = _STOP_REASON_TEXT.get(
                reason, "externally interrupted (Ctrl-C / recorder stopped "
                        "first) — no internal stop condition was logged")
            lines.append(f"- **did not complete normally** — {expl}")
    lines.append("")

    # KPI table: show the baseline value next to each run value when available,
    # so the absolute numbers are always visible, not just pass/fail.
    lines.append("## Measured KPIs")
    lines.append("")
    if baseline_cmp:
        lines.append(f"| metric | run (`{run_dir.name}`) | baseline (`{base_folder}`) |")
        lines.append("|---|---|---|")
        for k, v in metrics.items():
            if k in ("nav_goals", "coverage_milestones"):
                continue
            bv = baseline.get(k, "")
            lines.append(f"| {k} | {v} | {bv} |")
    else:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in metrics.items():
            if k in ("nav_goals", "coverage_milestones"):
                continue
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # -- fine-grained coverage progression (more than 50/90) ----------------
    milestones = metrics.get("coverage_milestones")
    if milestones:
        curve, fixed_denom = milestones
        rows, max_cov, path_at_max, time_at_max = curve
        lines.append("## Coverage milestones (full curve)")
        lines.append("")
        if fixed_denom:
            denom_note = ("Coverage here is covered_cells / final map free-cells (a "
                          "FIXED denominator), so early percentages reflect the whole "
                          "eventually-known map, not the tiny area SLAM had discovered "
                          "at that instant. ")
        else:
            denom_note = ("Coverage is the recorded ratio against the known map (fixed "
                          "denominator in known-map mode). ")
        lines.append("Path and time to reach each coverage level. " + denom_note +
                     "A run capped by Nav2 obstacle inflation cannot cover the whole "
                     "map, so levels above the plateau are never reached (`—`); the "
                     "peak actually reached is the last row.")
        lines.append("")
        lines.append("| coverage | path (m) | time (s) |")
        lines.append("|---|---|---|")
        for lvl, path_m, time_s in rows:
            pm = f"{path_m:.1f}" if path_m is not None else "—"
            ts = f"{time_s:.0f}" if time_s is not None else "—"
            lines.append(f"| {lvl:.0%} | {pm} | {ts} |")
        if max_cov is not None:
            pm = f"{path_at_max:.1f}" if path_at_max is not None else "—"
            ts = f"{time_at_max:.0f}" if time_at_max is not None else "—"
            lines.append(f"| **peak {max_cov:.1%}** | **{pm}** | **{ts}** |")
        lines.append("")

    ng = metrics["nav_goals"]
    lines.append("## Nav2 goal outcomes")
    lines.append("")
    lines.append(f"- total: {ng['total']}  succeeded: {ng['succeeded']}")
    lines.append(f"- aborted: {ng['aborted']}  canceled: {ng['canceled']}  "
                 f"other: {ng['other']}")
    lines.append("")

    if baseline_cmp:
        lines.append("## Comparison vs baseline")
        lines.append("")
        lines.append(f"Run `{run_dir.name}`  vs  baseline `{base_folder}` "
                     f"(source: {base_src}).")
        lines.append("")
        lines.append("| metric | run | baseline | delta | %delta | flag |")
        lines.append("|---|---|---|---|---|---|")
        for name, run_v, base_v, delta, pct, flag in baseline_cmp:
            lines.append(f"| {name} | {run_v} | {base_v} | {delta} | {pct} | {flag} |")
        lines.append("")
        if plot_path is not None:
            lines.append(f"![run vs baseline]({plot_path.name})")
            lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return passed


# Baseline save / compare
def _baseline_keys():
    return [
        "final_coverage", "covered_area_m2", "navigable_area_m2",
        "path_length_m", "coverage_rate", "path_to_50pct_coverage_m",
        "path_to_90pct_coverage_m", "time_to_50pct_coverage_s",
        "time_to_90pct_coverage_s", "total_run_time_s", "idle_time_s",
        "n_planning_cycles", "planning_time_mean_s", "planning_time_std_s",
        "total_waypoints", "waypoints_per_m2",
    ]


def save_baseline(config_path: Path, mode: str, run_dir: Path,
                  metrics: dict, source: str) -> None:
    doc = yaml.safe_load(config_path.read_text()) or {}
    section = doc.setdefault(mode, {})

    # Self-healing: rebuild the baseline block from EXACTLY the current keys, so
    # a file copied in the old schema is migrated on save (dead keys like
    # n_observation_poses / path_efficiency_ratio are dropped, not left as
    # stale clutter). The block is rewritten wholesale rather than updated.
    prev = section.get("baseline") or {}
    base = {"source": source, "run_folder": str(run_dir)}
    for k in _baseline_keys():
        base[k] = metrics.get(k)
    section["baseline"] = base

    # Also prune gates that check_gates() does not honour, so the file never
    # advertises a gate that is silently ignored (e.g. old fidelity_gap_max).
    gates = section.get("gates")
    if isinstance(gates, dict):
        dead = [k for k in gates if k not in _KNOWN_GATE_KEYS]
        if dead:
            for k in dead:
                del gates[k]
            print(f"[save-baseline] pruned unenforced gate(s) from "
                  f"{config_path.name} [{mode}]: {', '.join(dead)}")
    if prev and set(prev) - set(base):
        print(f"[save-baseline] migrated baseline schema in "
              f"{config_path.name} [{mode}] (dropped stale keys).")

    config_path.write_text(yaml.safe_dump(doc, sort_keys=False))

    # Copy the reference covered_mask next to the config.
    mask = run_dir / "covered_mask_final.npy"
    if mask.is_file():
        scene = section.get("run", {}).get("scene", "scene")
        dst = config_path.parent / f"baseline_{scene}_{mode}_covered_mask.npy"
        np.save(dst, np.load(mask))


def compare_baseline(metrics: dict, baseline: dict, tolerances: dict):
    """Compare the run against the baseline across every comparable KPI.

    Returns rows: (name, run_v, base_v, abs_delta, pct_delta, flag). flag is
    'ok' / 'OVER' / 'REGRESSED' for the three tolerance-gated KPIs, and 'info'
    for the rest (shown for context so you can see HOW the run differs, not just
    whether it tripped a tolerance).
    """
    path_tol = tolerances.get("path_length_tolerance_pct", 25)
    wp_tol = tolerances.get("waypoints_tolerance_pct", 25)
    cov_tol = tolerances.get("coverage_tolerance_abs", 0.02)

    rows = []
    for key in _baseline_keys():
        run_v, base_v = metrics.get(key), baseline.get(key)
        if not isinstance(run_v, (int, float)) or not isinstance(base_v, (int, float)):
            continue
        abs_delta = run_v - base_v
        pct = (100.0 * abs_delta / base_v) if base_v not in (0, None) else None
        pct_str = f"{pct:+.1f}%" if pct is not None else "n/a"

        if key == "path_length_m":
            flag = "OVER" if pct is not None and pct > path_tol else "ok"
        elif key == "total_waypoints":
            flag = "OVER" if pct is not None and pct > wp_tol else "ok"
        elif key == "final_coverage":
            flag = "REGRESSED" if (base_v - run_v) > cov_tol else "ok"
        else:
            flag = "info"  # tracked for context, not tolerance-gated
        rows.append((key, run_v, base_v, round(abs_delta, 4), pct_str, flag))
    return rows


def plot_comparison(run_dir: Path, baseline_cmp, scene: str, mode: str,
                    run_label: str, base_label: str) -> Path | None:
    """Bar chart of run vs baseline per KPI so failures are visible at a glance.

    Each KPI is shown normalised (run / baseline) so KPIs on very different
    scales sit on one axis; a baseline of 1.0 is the reference line and flagged
    KPIs are red. The run and baseline source folders are named in the title and
    axis labels. Returns the PNG path, or None if matplotlib is unavailable or
    there is nothing to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not baseline_cmp:
        return None

    names, run_norm, colours, labels = [], [], [], []
    for name, run_v, base_v, _abs, pct_str, flag in baseline_cmp:
        if base_v in (0, None):
            continue
        names.append(name)
        run_norm.append(run_v / base_v)
        colours.append("tab:red" if flag in ("OVER", "REGRESSED") else
                       ("tab:blue" if flag == "info" else "tab:green"))
        labels.append(f"{run_v:g} vs {base_v:g} ({pct_str})")
    if not names:
        return None

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names))))
    y = range(len(names))
    ax.barh(list(y), run_norm, color=colours)
    ax.axvline(1.0, color="0.3", ls="--", lw=1, label=f"baseline: {base_label}")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"run / baseline (1.0 = baseline)\nrun: {run_label}")
    ax.set_title(f"Run vs baseline KPIs: {scene} [{mode}]")
    for yi, lab in zip(y, labels):
        ax.text(0.02, yi, lab, va="center", ha="left", fontsize=7, color="0.2")
    ax.legend(loc="lower right", fontsize=8)
    out = run_dir / "comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _default_scene_doc(scene: str) -> dict:
    """Fresh scene YAML (both mode sections, empty baseline) for a new world.

    Mirrors baselines/baseline_lab05.yaml and recorder._default_scene_doc; gates
    start at standard values and time_to_complete_max_s must be tuned once the
    baseline exists. Kept in sync with recorder.py by hand (importing recorder
    here would drag in rclpy/ROS message deps this grading script doesn't need).
    """
    def section(mode: str, budget_s: int) -> dict:
        return {
            "run": {
                "scene": scene,
                "mode": mode,
                "simulator": "",
                "out_dir": "/ros2_ws/src/navigation/exploration/tests/system/runs",
                "record_bag": True,
                "sample_rate_hz": 5.0,
                "start_pose": [0.0, 0.0, 0.0],
            },
            "gates": {
                "final_coverage_min": 0.90,
                "must_terminate": True,
                "time_to_complete_max_s": budget_s,
                "nav_aborted_max": 0,
            },
            "tolerances": {
                "path_length_tolerance_pct": 25,
                "waypoints_tolerance_pct": 25,
                "coverage_tolerance_abs": 0.02,
            },
            "baseline": {"source": "", "run_folder": ""},
        }
    return {"known_map": section("known_map", 1200),
            "slam": section("slam", 1800)}


# Main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one exploration run folder")
    parser.add_argument("run_dir", help="Path to a per-run folder from recorder.py")
    parser.add_argument("--config", required=True,
                        help="Path to baselines/baseline_<scene>.yaml")
    parser.add_argument("--mode", required=True, choices=["known_map", "slam"])
    parser.add_argument("--save-baseline", action="store_true",
                        help="Store this run as the scene/mode baseline")
    parser.add_argument("--source", default="strategy",
                        help="Baseline source tag (e.g. manual_exploration)")
    parser.add_argument("--compare", action="store_true",
                        help="Diff this run against the saved baseline")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    if not config_path.is_file():
        if not args.save_baseline:
            parser.error(
                f"--config {config_path} not found. Pass an existing scene "
                f"config, or add --save-baseline to scaffold a new one.")
        # First baseline for a new scene: scaffold the combined config+baseline
        # YAML (both mode sections, empty baseline) so --save-baseline has a
        # file to populate. Scene name is derived from baseline_<scene>.yaml.
        stem = config_path.stem
        scene = stem[len("baseline_"):] if stem.startswith("baseline_") else stem
        if not scene:
            parser.error(f"Cannot derive a scene name from '{config_path.name}'")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(_default_scene_doc(scene), sort_keys=False))
        print(f"[evaluate_run] No scene config at '{config_path}'. Created a "
              f"fresh one for scene '{scene}' — review its gates and set "
              "run.simulator.")
    doc = yaml.safe_load(config_path.read_text()) or {}
    section = doc.get(args.mode, {})
    gates = section.get("gates", {})
    tolerances = section.get("tolerances", {})
    baseline = section.get("baseline", {})

    meta = _load_yaml(run_dir / "meta.yaml")
    if meta.get("time_source") == "wall":
        print("WARNING: run was recorded on the wall clock "
              "(use_sim_time was off); time KPIs are laptop time, "
              "not simulation time.")

    # Reference coverage to the baseline's covered area so 50/90% milestones and
    # final coverage mean the same absolute area in every run (see compute_metrics).
    metrics = compute_metrics(run_dir, meta, baseline.get("covered_area_m2"))

    # Time-gate budget reference: this mode's baseline run time, else the OTHER
    # mode's baseline time for the same scene (e.g. hospital has only a slam
    # baseline, so known_map runs are timed against the slam baseline). None when
    # the scene has no baseline time at all -> the gate falls back to the config's
    # time_to_complete_max_s inside check_gates.
    baseline_time = baseline.get("total_run_time_s")
    if not baseline_time:
        for other in ("slam", "known_map"):
            ot = ((doc.get(other) or {}).get("baseline") or {}).get("total_run_time_s")
            if ot:
                baseline_time = ot
                break
    time_tol = tolerances.get("time_tolerance_pct", 25)
    # Coverage-gate reference: the coverage this scene's baseline actually
    # reached. Same fallback chain as the time budget — this mode's baseline
    # first, then the other mode's for the same scene, else None (gate reverts
    # to the config's absolute final_coverage_min inside check_gates).
    baseline_cov = baseline.get("final_coverage")
    if not baseline_cov:
        for other in ("slam", "known_map"):
            oc = ((doc.get(other) or {}).get("baseline") or {}).get("final_coverage")
            if oc:
                baseline_cov = oc
                break
    cov_tol_gate = tolerances.get("coverage_tolerance_abs", 0.02)
    gate_results = check_gates(metrics, gates, baseline_time, time_tol,
                               baseline_cov, cov_tol_gate)

    baseline_cmp = None
    plot_path = None
    if args.compare and baseline and baseline.get("final_coverage") is not None:
        baseline_cmp = compare_baseline(metrics, baseline, tolerances)
        plot_path = plot_comparison(
            run_dir, baseline_cmp,
            scene=meta.get("scene", section.get("run", {}).get("scene", "?")),
            mode=args.mode,
            run_label=run_dir.name,
            base_label=baseline.get("run_folder", baseline.get("source", "baseline")),
        )

    passed = write_report(run_dir, meta, metrics, gate_results, baseline_cmp,
                          baseline, plot_path)

    if args.save_baseline:
        # A baseline must be measured against ITSELF, never against the previous
        # baseline's area — otherwise each save would re-reference the last one
        # and the stored milestones would drift with baseline history.
        base_metrics = compute_metrics(run_dir, meta)
        save_baseline(config_path, args.mode, run_dir, base_metrics, args.source)
        print(f"Saved baseline to {config_path} [{args.mode}]")

    print(f"{'PASS' if passed else 'FAIL'}  {run_dir.name}  "
          f"(coverage={metrics['final_coverage']:.1%}, "
          f"path={metrics['path_length_m']:.1f} m); report.md written")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
