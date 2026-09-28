"""
Integration / simulation tests for navigation/exploration/explore_costmap_map.py

These tests simulate the full plan, observe, re-plan loop against the real
maps shipped under the repo's ``assets/`` directory. No ROS2 required.

Each test prints progress to stdout so you can follow along with `pytest -s`.

Run with:
    pytest tests/test_integration_navigation_exploration_explore_costmap_map.py -v -s
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "navigation"))

from scipy.ndimage import label

from exploration.explore_costmap_map import (
    load_map,
    pixel_to_world,
    plan_waypoints,
    update_covered_mask,
    world_to_pixel,
)

logger = logging.getLogger(__name__)

INFLATION_M = 0.3
DETECTION_RANGE_M = 6.0


# Map discovery
def _find_assets_dir() -> Path:
    """Walk up from this file until a directory containing ``assets/*/map.yaml``
    is found. The maps are read-only inputs and are referenced in place, never
    copied into the tests tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        assets = parent / "assets"
        if assets.is_dir() and any(assets.glob("*/map.yaml")):
            return assets
    raise FileNotFoundError(
        "No assets/*/map.yaml found walking up from "
        f"{here}. Expected the repo's top-level assets/ with map folders."
    )


def _discover_maps() -> list[Path]:
    """Every map folder under assets/ that has both map.yaml and map.pgm."""
    assets = _find_assets_dir()
    maps = sorted(
        d for d in assets.glob("*")
        if (d / "map.yaml").is_file() and (d / "map.pgm").is_file()
    )
    if not maps:
        raise FileNotFoundError(f"No usable map folders under {assets}")
    return maps


# Parametrize every behavioural test over each discovered map. The id is the
# folder name (e.g. "lab_ghent") so failures point straight at the scene.
_MAP_DIRS = _discover_maps()
_MAP_PARAMS = [pytest.param(d, id=d.name) for d in _MAP_DIRS]


def _default_config(max_range: float = DETECTION_RANGE_M) -> dict:
    return {
        "max_detection_range": max_range,
        "fov_horizontal": 87.0,
        "observation_rotation_increment": 30.0,
        "sampling_step_m": 3.0,
        "num_rays": 360,
        "frontier_weight": 1.0,
        "coverage_weight": 1.0,
        "exploration_completion_threshold": 0.90,
        "planner_coverage_warning_threshold": 0.50,
    }


def _load(map_dir: Path):
    return load_map(map_dir / "map.pgm", map_dir / "map.yaml", INFLATION_M)


def _robot_start(md) -> tuple[float, float]:
    """Robot start world (x, y).

    The map's YAML ``origin`` encodes where world (0, 0) — the robot's pose when
    the map was recorded — sits in the grid, so world (0, 0) is the real start.
    Use it when it lands on a navigable cell; otherwise fall back to the centre
    of the largest connected navigable component (a sensible 'middle' that avoids
    stranding the robot on a disconnected LiDAR-glitch island).
    """
    H = md.navigable_mask.shape[0]
    col, row = world_to_pixel(0.0, 0.0, md.resolution, md.origin_x, md.origin_y, H)
    if 0 <= row < H and 0 <= col < md.navigable_mask.shape[1] \
            and md.navigable_mask[row, col]:
        return 0.0, 0.0

    # Fallback: centre of the largest navigable component.
    labelled, n = label(md.navigable_mask)
    if n == 0:
        raise RuntimeError("Map has no navigable cells.")
    main = int(np.argmax([(labelled == i).sum() for i in range(1, n + 1)])) + 1
    rows, cols = np.where(labelled == main)
    ccol, crow = int(cols.mean()), int(rows.mean())
    k = int(np.argmin((cols - ccol) ** 2 + (rows - crow) ** 2))
    return pixel_to_world(int(cols[k]), int(rows[k]), md.resolution,
                          md.origin_x, md.origin_y, H)


def _bar(ratio: float, width: int = 30) -> str:
    """ASCII progress bar: e.g. [==========          ] 50.0%"""
    filled = int(ratio * width)
    return f"[{'=' * filled}{' ' * (width - filled)}] {ratio:.1%}"


# Measured-metrics report. The waypoint count and density are MEASURED, never
# capped: this file is the L2 counterpart of the Isaac system-test report.md,
# so the same numbers (total waypoints, navigable area, waypoints/m2, final
# coverage) can be compared across maps and against the sim runs.
_REPORT_PATH = Path(__file__).parent / "integration_metrics_report.json"


def _record_metrics(map_name: str, metrics: dict) -> None:
    """Merge one map's measured metrics into the shared JSON report."""
    import json

    report: dict = {}
    if _REPORT_PATH.exists():
        try:
            report = json.loads(_REPORT_PATH.read_text())
        except (ValueError, OSError):
            report = {}
    report[map_name] = metrics
    _REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))


@pytest.mark.parametrize("map_dir", _MAP_PARAMS)
class TestIntegrationSimulation:

    def test_plan_execute_cycle_reaches_90_percent(self, map_dir: Path):
        """
        Verifies: plan-then-execute cycle achieves >= 90% visual coverage
        within 5 full plan cycles.

        Each cycle executes the complete waypoint plan before re-planning.
        This matches the BehaviourTree flow:
            PlanExplorationWaypoints, ExploreLoop (all waypoints), re-plan.

        Input: each assets/* map, default config, max 5 cycles.
        Measures: final coverage_ratio >= 0.90; ratio per cycle logged.
        """
        print(f"\n--- test_plan_execute_cycle_reaches_90_percent [{map_dir.name}] ---")
        t0 = time.perf_counter()

        print("  Loading map...", flush=True)
        md = _load(map_dir)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        config = _default_config()
        final_ratio = 0.0
        MAX_CYCLES = 5

        for cycle in range(MAX_CYCLES):
            t_plan = time.perf_counter()
            print(f"  [cycle {cycle}/{MAX_CYCLES-1}] planning waypoints...", flush=True)
            waypoints, ratio, no_frontiers, _ = plan_waypoints(md, config)
            final_ratio = ratio
            dt_plan = time.perf_counter() - t_plan

            print(
                f"  [cycle {cycle}/{MAX_CYCLES-1}] plan done in {dt_plan:.1f}s | "
                f"{len(waypoints)} waypoints | {_bar(ratio)} | no_frontiers={no_frontiers}",
                flush=True,
            )
            logger.info(
                f"[{map_dir.name}][cycle {cycle}] coverage_ratio={ratio:.3f}, "
                f"waypoints={len(waypoints)}, no_frontiers={no_frontiers}, "
                f"plan_time={dt_plan:.1f}s"
            )

            if not waypoints:
                print(f"  [cycle {cycle}] no waypoints returned, stopping early.", flush=True)
                break

            t_obs = time.perf_counter()
            for i, wp in enumerate(waypoints):
                for heading in (wp.headings or list(range(0, 360, 30))):
                    update_covered_mask(
                        md, wp.col, wp.row, float(heading), 87.0, max_range_px
                    )
                if (i + 1) % 5 == 0 or i == len(waypoints) - 1:
                    print(
                        f"    observed {i+1}/{len(waypoints)} waypoints "
                        f"({(i+1)/len(waypoints):.0%})...",
                        flush=True,
                    )
            dt_obs = time.perf_counter() - t_obs
            print(
                f"  [cycle {cycle}] observation done in {dt_obs:.1f}s",
                flush=True,
            )

        total = time.perf_counter() - t0
        print(f"\n  FINAL: {_bar(final_ratio)}  (total {total:.1f}s)", flush=True)
        logger.info(
            f"[{map_dir.name}][final] coverage_ratio={final_ratio:.3f}, total_time={total:.1f}s"
        )
        assert final_ratio >= 0.90, f"[{map_dir.name}] Coverage {final_ratio:.1%} below 90%"

    def test_reactive_replan_after_each_waypoint_reaches_90_percent(self, map_dir: Path):
        """
        Verifies: re-planning after each individual waypoint visit achieves >= 90%
        visual coverage within 60 total waypoint visits.

        After each camera observation the covered_mask grows, so the planner
        immediately drops waypoints made redundant by adjacent observations.
        Robot position is updated after each visit so the TSP start follows
        the robot through the map.

        Input: each assets/* map, default config, max 60 visits.
        Measures: final coverage_ratio >= 0.90; ratio and visit count logged per step.
        """
        print(f"\n--- test_reactive_replan_after_each_waypoint [{map_dir.name}] ---")
        t0 = time.perf_counter()

        print("  Loading map...", flush=True)
        md = _load(map_dir)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        config = _default_config()
        ratio = 0.0
        total_visited = 0
        # Seed the robot at its real start (world origin (0,0) per the map YAML),
        # falling back to the main navigable region if that isn't navigable.
        robot_x, robot_y = _robot_start(md)
        visited: set[tuple[int, int]] = set()
        MAX_VISITS = 60
        LOG_EVERY = 5  # print a summary line every N visits

        print(f"  Starting reactive loop (max {MAX_VISITS} visits)...", flush=True)

        while total_visited < MAX_VISITS:
            waypoints, ratio, _, _ = plan_waypoints(
                md, config, robot_x, robot_y, visited
            )
            if not waypoints:
                print(f"  No waypoints returned at visit {total_visited}, done.", flush=True)
                break

            wp = waypoints[0]
            for heading in (wp.headings or list(range(0, 360, 30))):
                update_covered_mask(
                    md, wp.col, wp.row, float(heading), 87.0, max_range_px
                )
            robot_x, robot_y = wp.x, wp.y
            visited.add((wp.col, wp.row))
            total_visited += 1

            logger.info(
                f"[{map_dir.name}][visit {total_visited:3d}/{MAX_VISITS}] ratio={ratio:.3f} "
                f"wp=({wp.col},{wp.row})"
            )
            if total_visited % LOG_EVERY == 0 or total_visited == 1:
                elapsed = time.perf_counter() - t0
                remaining_est = (
                    (elapsed / total_visited) * (MAX_VISITS - total_visited)
                    if total_visited > 0 else 0
                )
                print(
                    f"  [visit {total_visited:3d}/{MAX_VISITS}] {_bar(ratio)} | "
                    f"wp=({wp.col:4d},{wp.row:4d}) | "
                    f"elapsed={elapsed:.0f}s est_remaining~{remaining_est:.0f}s",
                    flush=True,
                )

        total = time.perf_counter() - t0
        print(
            f"\n  FINAL: {_bar(ratio)}  after {total_visited} visits  (total {total:.1f}s)",
            flush=True,
        )
        logger.info(
            f"[{map_dir.name}][final] coverage_ratio={ratio:.3f} after {total_visited} visits, "
            f"total_time={total:.1f}s"
        )
        assert ratio >= 0.90, (
            f"[{map_dir.name}] Visual exploration reached only {ratio:.1%} "
            f"after {total_visited} waypoints"
        )

    def test_coverage_ratio_is_monotonically_non_decreasing(self, map_dir: Path):
        """
        Verifies: [pattern 10, accumulation] coverage_ratio never decreases between
        re-planning cycles, covered cells are never lost.

        Once a cell is marked covered it must remain covered because covered_mask
        only grows. A decrease would indicate a bug where the mask is reset or
        overwritten during re-planning.

        Input: each assets/* map, default config, 5 plan-then-execute cycles.
        Measures: ratio(t+1) >= ratio(t) for every consecutive cycle pair.
        """
        print(f"\n--- test_coverage_ratio_is_monotonically_non_decreasing [{map_dir.name}] ---")
        t0 = time.perf_counter()

        print("  Loading map...", flush=True)
        md = _load(map_dir)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        config = _default_config()
        previous_ratio = 0.0
        MAX_CYCLES = 5

        for cycle in range(MAX_CYCLES):
            print(f"  [cycle {cycle}/{MAX_CYCLES-1}] planning...", flush=True)
            waypoints, ratio, _, _ = plan_waypoints(md, config)
            delta = ratio - previous_ratio
            elapsed = time.perf_counter() - t0

            print(
                f"  [cycle {cycle}/{MAX_CYCLES-1}] {_bar(ratio)} | "
                f"delta={delta:+.3f} | {len(waypoints)} waypoints | elapsed={elapsed:.0f}s",
                flush=True,
            )
            logger.info(
                f"[{map_dir.name}][cycle {cycle}] coverage_ratio={ratio:.3f}, delta={delta:+.3f}, "
                f"elapsed={elapsed:.1f}s"
            )

            assert ratio >= previous_ratio - 1e-9, (
                f"[{map_dir.name}] Coverage decreased at cycle {cycle}: "
                f"{previous_ratio:.3f} to {ratio:.3f} (delta={delta:+.3f})"
            )
            previous_ratio = ratio

            if not waypoints:
                print(f"  [cycle {cycle}] no waypoints, stopping early.", flush=True)
                break

            for i, wp in enumerate(waypoints):
                for heading in (wp.headings or list(range(0, 360, 30))):
                    update_covered_mask(
                        md, wp.col, wp.row, float(heading), 87.0, max_range_px
                    )
                if (i + 1) % 5 == 0 or i == len(waypoints) - 1:
                    print(
                        f"    observed {i+1}/{len(waypoints)} waypoints...",
                        flush=True,
                    )

        total = time.perf_counter() - t0
        print(
            f"\n  DONE, monotonicity verified over {MAX_CYCLES} cycles  (total {total:.1f}s)",
            flush=True,
        )

    def test_smaller_detection_range_yields_more_waypoints(self, map_dir: Path):
        """
        Verifies: reducing detection range increases the number of planned waypoints
        because each position covers less area and more positions are needed.

        Input: each assets/* map, max_range=3.0m vs max_range=6.0m.
        Measures: len(wps_short) > len(wps_long), strictly; both counts logged.
        The inequality is strict because equal counts are the exact symptom of max_detection_range being ignored, which a >= assertion would pass.
        Measured margins are wide on every shipped map (2.9x to 3.5x), so strictness carries no flakiness risk.
        """
        print(f"\n--- test_smaller_detection_range_yields_more_waypoints [{map_dir.name}] ---")

        print("  Loading maps...", flush=True)
        md_short = _load(map_dir)
        md_long = _load(map_dir)

        print("  Planning with max_range=3.0m...", flush=True)
        t = time.perf_counter()
        wps_short, _, _, _ = plan_waypoints(md_short, _default_config(max_range=3.0))
        print(f"  -> {len(wps_short)} waypoints in {time.perf_counter()-t:.1f}s", flush=True)

        print("  Planning with max_range=6.0m...", flush=True)
        t = time.perf_counter()
        wps_long, _, _, _ = plan_waypoints(md_long, _default_config(max_range=6.0))
        print(f"  -> {len(wps_long)} waypoints in {time.perf_counter()-t:.1f}s", flush=True)

        delta = len(wps_short) - len(wps_long)
        print(
            f"\n  [deviation] short={len(wps_short)}, long={len(wps_long)}, delta={delta:+d}",
            flush=True,
        )
        logger.info(
            f"[{map_dir.name}][deviation] waypoints: short_range={len(wps_short)}, "
            f"long_range={len(wps_long)}, delta={delta:+d}"
        )
        assert len(wps_short) > len(wps_long), (
            f"[{map_dir.name}] Expected strictly more waypoints with 3m range ({len(wps_short)}) "
            f"than 6m range ({len(wps_long)}); equality means the range parameter was ignored"
        )


# ---------------------------------------------------------------------------
# Benchmark, deliberately NOT a test
#
# This has no assertions: it measures and records, so as a test_ function it
# reported a pass that could never fail and ran 8 plan cycles per map to do it.
# Named benchmark_ so pytest does not collect it. Run it explicitly with:
#     python -c "from tests.test_integration_navigation_exploration_explore_costmap_map \
#         import benchmark_waypoint_count_and_density as b; b()"
# ---------------------------------------------------------------------------

def benchmark_waypoint_count_and_density(map_dirs: list[Path] | None = None) -> dict:
    """
    Measures (Option-A efficiency): the total number of waypoints a full plan-then-execute exploration to completion selects, and the resulting waypoint density (waypoints per m2 of navigable area).

    These are MEASURED, not capped: the count is the headline efficiency signal for the max_waypoints_per_plan work tracked in project_option_a_waypoints.
    Lower is better for a given coverage, but a good value is scene-dependent, so the numbers are written to integration_metrics_report.json (the L2 counterpart of the Isaac system-test report.md) for cross-map and cross-run comparison rather than asserted against a threshold.

    There is deliberately no assertion here; completion is already guarded by test_plan_execute_cycle_reaches_90_percent.

    Input: every assets/* map by default, default config, run to completion (<= 8 cycles).
    Records: total_waypoints, navigable_area_m2, waypoints_per_m2, final_coverage, cycles.
    Returns: the per-map metrics dict it also writes to disk.
    """
    all_metrics: dict = {}
    for map_dir in (map_dirs if map_dirs is not None else _MAP_DIRS):
        print(f"\n--- benchmark_waypoint_count_and_density [{map_dir.name}] ---")
        md = _load(map_dir)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        config = _default_config()

        navigable_area_m2 = float(md.navigable_mask.sum()) * (md.resolution ** 2)

        total_selected = 0
        cycles = 0
        ratio = 0.0
        MAX_CYCLES = 8
        for cycle in range(MAX_CYCLES):
            waypoints, ratio, _, _ = plan_waypoints(md, config)
            cycles = cycle + 1
            if not waypoints:
                break
            total_selected += len(waypoints)
            for wp in waypoints:
                for heading in (wp.headings or list(range(0, 360, 30))):
                    update_covered_mask(
                        md, wp.col, wp.row, float(heading), 87.0, max_range_px
                    )
            if ratio >= config["exploration_completion_threshold"]:
                break

        density = total_selected / navigable_area_m2 if navigable_area_m2 else 0.0
        metrics = {
            "total_waypoints": total_selected,
            "navigable_area_m2": round(navigable_area_m2, 1),
            "waypoints_per_m2": round(density, 4),
            "final_coverage": round(ratio, 4),
            "cycles": cycles,
        }
        print(
            f"  navigable={navigable_area_m2:.0f} m2 | total_waypoints={total_selected} "
            f"| {density:.3f} wp/m2 | final_coverage={ratio:.1%} | cycles={cycles}",
            flush=True,
        )
        logger.info(f"[{map_dir.name}][option-a] {metrics}")
        _record_metrics(map_dir.name, metrics)
        all_metrics[map_dir.name] = metrics

    return all_metrics


# SLAM-mode termination (single map; not parametrized)
def test_slam_mode_reaches_no_frontiers():
    """
    Verifies (SLAM mode): when the map is revealed incrementally (unknown cells
    converted to free/occupied as the robot observes), the planner eventually
    reports ``no_frontiers=True``, i.e. exploration terminates rather than
    chasing frontiers forever.

    We emulate SLAM growth the way visual_demo_slam does: start with most of the
    map unknown, then on each cycle reveal the cells the robot can see, and feed
    the grown map back into the planner. Success is no_frontiers becoming True
    within a bounded number of cycles.

    Input: lab_ghent map (or first discovered map), default config, <= 25 cycles.
    Measures: no_frontiers flips True before the cycle budget is exhausted.
    """
    print("\n--- test_slam_mode_reaches_no_frontiers ---")
    map_dir = next((d for d in _MAP_DIRS if d.name == "lab_ghent"), _MAP_DIRS[0])
    print(f"  Using map: {map_dir.name}", flush=True)

    md = _load(map_dir)
    max_range_px = int(DETECTION_RANGE_M / md.resolution)
    config = _default_config()

    # Ground truth to reveal toward.
    truth_free = md.free_mask.copy()
    truth_occ = md.occupied_mask.copy()

    # Start near the centre of the known-free region.
    free_rows, free_cols = np.where(truth_free)
    start_row = int(np.median(free_rows))
    start_col = int(np.median(free_cols))

    def reveal(disc_col: int, disc_row: int, radius_px: int) -> None:
        """Convert unknown cells within radius to their ground-truth class."""
        H, W = md.unknown_mask.shape
        r0, r1 = max(0, disc_row - radius_px), min(H, disc_row + radius_px + 1)
        c0, c1 = max(0, disc_col - radius_px), min(W, disc_col + radius_px + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        within = (rr - disc_row) ** 2 + (cc - disc_col) ** 2 <= radius_px ** 2
        sub = (slice(r0, r1), slice(c0, c1))
        newly = within & md.unknown_mask[sub]
        md.free_mask[sub][newly] = truth_free[sub][newly]
        md.occupied_mask[sub][newly] = truth_occ[sub][newly]
        md.unknown_mask[sub][newly] = False

    # Hide the whole map, then reveal the start disc.
    md.unknown_mask[:] = True
    md.free_mask[:] = False
    md.occupied_mask[:] = False
    reveal(start_col, start_row, max_range_px)
    md.navigable_mask[:] = md.free_mask

    no_frontiers = False
    MAX_CYCLES = 25
    for cycle in range(MAX_CYCLES):
        waypoints, ratio, no_frontiers, _ = plan_waypoints(md, config)
        print(
            f"  [slam cycle {cycle}] {_bar(ratio)} | {len(waypoints)} wps | "
            f"no_frontiers={no_frontiers} | unknown={md.unknown_mask.mean():.1%}",
            flush=True,
        )
        if no_frontiers:
            break
        if not waypoints:
            break
        for wp in waypoints:
            for heading in (wp.headings or list(range(0, 360, 30))):
                update_covered_mask(
                    md, wp.col, wp.row, float(heading), 87.0, max_range_px
                )
            reveal(wp.col, wp.row, max_range_px)
        md.navigable_mask[:] = md.free_mask

    assert no_frontiers, (
        f"SLAM exploration did not reach no_frontiers within {MAX_CYCLES} cycles "
        f"(unknown still {md.unknown_mask.mean():.1%})"
    )
