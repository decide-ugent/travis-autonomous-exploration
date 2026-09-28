"""
Execution strategy for waypoint-based exploration.

Provides two classes:

  MidPathReplanner
      Stateless per-path helper. Called at each travel step; returns True when
      the robot should abort the current path and replan from its position.

  ExplorationSession
      Stateful session object. Owns the full exploration lifecycle:
        - candidate generation and regeneration (for SLAM / growing maps)
        - visited-candidate tracking (spatial exclusion radius)
        - mid-path replanning via MidPathReplanner
        - unreachable-waypoint handling

      Intended call sequence (ROS2 node or demo):

          session = ExplorationSession(map_data, config)
          robot_col, robot_row = session.nearest_start(cx, cy)

          while True:
              waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(robot_x, robot_y)
              completion = config["exploration_completion_threshold"]
              if no_frontiers and ratio >= completion:
                  break  # exploration complete

              wp = waypoints[0]
              path = find_path(navigable_mask, (robot_col, robot_row), (wp.col, wp.row))

              if session.on_unreachable(wp, path):
                  continue  # skip : no path exists, don't mark visited

              mid_replan = False
              for step in path[1:]:
                  heading = ...
                  ratio   = update_covered_mask(...)

                  if session.on_step(step, wp, waypoints):
                      robot_col, robot_row = step
                      mid_replan = True
                      break

              if not mid_replan:
                  # rotate at waypoint ...
                  session.on_arrive(wp)
                  robot_col, robot_row = wp.col, wp.row
"""
from __future__ import annotations

import math
import numpy as np

from exploration.explore_costmap_map import (
    navigable_distance_map,
    generate_candidates,
    plan_waypoints,
    compute_visibility,
)

# FOV-aware visited-mark threshold: a candidate is excluded from future planning
# once the camera has observed at least this fraction of the cells it could ever
# see. Relative (not "all cells") so a small residual sliver behind an occluder
# does not keep a near-saturated viewpoint alive forever.
OBSERVED_FRACTION_THRESHOLD = 0.95


# ── MidPathReplanner ──────────────────────────────────────────────────────────

class MidPathReplanner:
    """Triggers mid-path replanning when a detour makes another waypoint cheaper.

    Args:
        check_interval_px: Minimum pixels traveled between BFS checks.
                           Use visit_radius_px so the cadence matches visited-candidate marking.
        replan_ratio:      Abort the current path and replan when another waypoint's navigable distance drops below this fraction of the
                           distance to the current target. Lower = reluctant to
                           divert (fewer mid-path replans); higher = eager.0 (never divert) gives the longest path on all maps;
                           0.75 shortens the path at held coverage on all three and is the chosen default. The old hardcoded value was 0.5.
    """

    def __init__(self, check_interval_px: float, replan_ratio: float = 0.75) -> None:
        self._interval = max(1.0, check_interval_px)
        self._replan_ratio = replan_ratio
        self._last_check: tuple[int, int] | None = None

    def check(
        self,
        step: tuple[int, int],
        current_target: object,
        all_waypoints: list,
        navigable_mask: np.ndarray,
        inflation_px: float = 0.0,
    ) -> bool:
        """Return True if the robot should abort current path and replan.

        Args:
            step:           Current robot pixel position (col, row).
            current_target: The Waypoint the robot is currently heading to.
            all_waypoints:  All waypoints in the current plan.
            navigable_mask: Boolean navigable grid for BFS.

        Returns:
            True → break travel loop and trigger outer replan.
            False → continue traveling.
        """
        if self._last_check is None:
            self._last_check = step
            return False

        if math.hypot(step[0] - self._last_check[0],
                      step[1] - self._last_check[1]) < self._interval:
            return False

        self._last_check = step

        dist = navigable_distance_map(navigable_mask, step[0], step[1], inflation_px)

        dist_to_target = dist[current_target.row, current_target.col]
        if dist_to_target == np.inf:
            return True  # target unreachable : replan immediately

        for wp in all_waypoints:
            if wp is current_target:
                continue
            if dist[wp.row, wp.col] < dist_to_target * self._replan_ratio:
                return True

        return False


# ── ExplorationSession ────────────────────────────────────────────────────────

class ExplorationSession:
    """Stateful exploration session : owns visited-candidate tracking and replanning.

    This is the object the ROS2 exploration node instantiates once per exploration run. It wraps plan_waypoints and manages all state that must persist across re-plan cycles: which grid positions have been visited, the spatial exclusion radius, candidate regeneration for growing maps, and mid-path replanning.

    Args:
        map_data: Initial MapData (from load_map or build_map_data). When a new
                  map arrives (SLAM), call set_map() to rebind it — that refreshes
                  the candidate pool from the new navigable_mask.
        config:   Flat config dict (same as passed to plan_waypoints).
    """

    def __init__(self, map_data, config: dict) -> None:
        self._config = config

        max_range_m: float  = config.get("max_detection_range", 6.0)
        sampling_step_m: float = config.get("sampling_step_m", 3.0)

        self._sampling_step_px: int = max(1, int(sampling_step_m / map_data.resolution))
        self.visit_radius_px: float = max(1.0, max_range_m / 2.0 / map_data.resolution)

        # OBSERVED viewpoints only: the camera actually saw >= OBSERVED_FRACTION_THRESHOLD  of what this position could ever see (set by _mark/on_arrive). This is a permanent, monotonic fact about coverage.
        self.visited_candidates: set[tuple[int, int]] = set()
        self._replanner: MidPathReplanner | None = None
        # Per-waypoint Nav2 ABORT counter (keyed on pixel position). Once the count hits  _abort_blacklist_after the waypoint goes into _unreachable so re-plans stop  re-picking it (the livelock where every plan re-selects the same aborting  waypoint 0).
        self._abort_counts: dict[tuple[int, int], int] = {}
        self._abort_blacklist_after: int = int(
            config.get("abort_blacklist_after", 3))
        # NOT-REACHED viewpoints: Nav2 repeatedly failed to drive here. This is NOT the  same fact as visited_candidates, the camera never observed these, so the area  is still UNCOVERED and must be retried once circumstances change. Keeping it  separate is what stops "could not reach" from being mistaken for "already seen"  (which silently froze coverage and starved the candidate pool). Transient by  design: restored by clear_unreachable() when the candidate pool would otherwise be empty (starved). Deliberately NOT restored on every arrival, that livelocks on permanently unreachable waypoints.
        self._unreachable: set[tuple[int, int]] = set()

        # set_map binds _md and generates the initial candidate list.
        self._candidates: list[tuple[int, int]] = []
        self.set_map(map_data)

    # ── Public interface ──────────────────────────────────────────────────────

    def set_map(self, map_data) -> None:
        """Bind (or rebind) the map and regenerate candidates from its navigable_mask.

        Called at construction and again whenever a new map arrives (node: _on_map
        rebuilds MapData and calls this). The map-change event is the single point
        where candidates must be refreshed so newly revealed cells (SLAM) are
        included — regenerating here (not per-plan, not per-read) keeps _candidates
        in sync with navigable_mask without a cache-invalidation key.
        """
        self._md = map_data
        self._candidates = generate_candidates(
            map_data.navigable_mask, self._sampling_step_px, map_data.resolution
        )

    def nearest_start(self, cx: int, cy: int) -> tuple[int, int]:
        """Return the navigable candidate nearest to pixel (cx, cy)."""
        if not self._candidates:
            raise RuntimeError("No navigable candidates : check map loading.")
        return min(self._candidates, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

    def plan_waypoints_raw(self, robot_x: float, robot_y: float):
        """Generate the next waypoint plan from the current robot position.

        Returns the full (waypoints, ratio, no_frontiers, candidate_records)
        tuple. Candidates are refreshed on map change via set_map(), not here.
        The caller
        applies the completion check (no_frontiers and ratio >= threshold →
        exploration complete). Arms a fresh MidPathReplanner for this path.

        Args:
            robot_x, robot_y: Current robot world position (metres).

        Returns:
            (waypoints, ratio, no_frontiers, candidate_records)
        """

    
        # Exclude both the genuinely-observed viewpoints and the ones Nav2 currently  cannot reach. The union keeps the abort livelock suppressed, while the two sets stay separable so _unreachable can be restored later (clear_unreachable).
        result = plan_waypoints(
            self._md, self._config, robot_x, robot_y,
            self.visited_candidates | self._unreachable,
        )
        self._replanner = MidPathReplanner(
            self.visit_radius_px,
            replan_ratio=self._config.get("mid_path_replan_ratio", 0.75),
        )
        return result

    def clear_unreachable(self) -> int:
        """Restore all waypoints previously abandoned as unreachable.

        An abort verdict is conditional on circumstances (robot pose, transient local
        costmap, Nav2 state) — none of which are permanent — so a parked waypoint must
        stay retryable. `visited_candidates` is deliberately NOT touched: genuinely
        observed viewpoints stay observed.

        CALL ONLY WHEN THE CANDIDATE POOL IS STARVED (no waypoint can be planned). Restoring at any other moment (e.g. after every successful arrival) creates a
        LIVELOCK: a permanently unreachable waypoint (a frontier inside a wall, a room behind a door the robot cannot fit through) gets restored, is immediately re-selected as the highest-gain candidate, aborts N times, is re-parked, and the cycle repeats forever while real progress is still possible elsewhere. Worse, plans are never empty in that loop, so the node's empty-plan timeout can never fire and the run churns unbounded.

        Restricting the restore to starvation keeps it strictly beneficial: it only runs when the alternative is doing nothing at all, and if the restored waypoints abort
        again the pool starves again, which IS observable, so the node's escalation (warn → retry → plan_timeout_s → stop) still bounds the loop.

        Returns:
            Number of waypoints restored (0 if there was nothing to restore), so the
            caller can log the recovery instead of it happening silently.
        """
        n = len(self._unreachable)
        self._unreachable.clear()
        self._abort_counts.clear()
        return n

    def on_unreachable(self, wp, path: list) -> bool:
        """Check if a waypoint is unreachable (find_path returned only the start).

        Does NOT mark the waypoint visited : it may be reachable from a different
        robot position in a future plan cycle.

        Args:
            wp:   The target Waypoint.
            path: The path returned by find_path.

        Returns:
            True if unreachable (caller should skip/continue).
        """
        if not path:
            return True   # no path at all → unreachable

        bfs_px = len(path) - 1
        return bfs_px == 0 and (wp.col != path[0][0] or wp.row != path[0][1])

    def on_nav_aborted(self, wp) -> bool:
        """Record a Nav2 ABORT for this waypoint; blacklist it if it persists.

        Called by the node whenever Nav2 aborts a NavigateToPose goal. Counts aborts per
        waypoint pixel; once a waypoint has been aborted _abort_blacklist_after times it
        goes into `_unreachable` so future plans exclude it, this breaks the livelock
        where an unreachable frontier is regenerated as waypoint 0 on every re-plan.

        It is deliberately NOT added to `visited_candidates`: the robot never got there, so the camera never observed it and that area is still uncovered. Recording it as
        "visited" would tell the planner the area is done and permanently freeze coverage.

        Returns True when the waypoint was blacklisted this call (so the caller can log it), False otherwise.
        """
        key = (wp.col, wp.row)
        self._abort_counts[key] = self._abort_counts.get(key, 0) + 1
        if self._abort_counts[key] >= self._abort_blacklist_after:
            self._unreachable.add(key)
            return True
        return False

    def on_arrive_clear_aborts(self, wp) -> None:
        """Reset the abort counter for a waypoint that was successfully reached.

        A real arrival proves THIS waypoint is reachable, so earlier transient aborts for it must not accumulate toward blacklisting.

        Deliberately does NOT restore the whole `_unreachable` set. Doing that on every arrival livelocks on permanently-unreachable waypoints: they would be restored,
        instantly re-selected as highest-gain, abort again, and be re-parked, forever, while plans never go empty so no timeout could catch it. The restore runs only when the pool is genuinely starved (see clear_unreachable).
        """
        self._abort_counts.pop((wp.col, wp.row), None)

    def on_step(
        self,
        step: tuple[int, int],
        current_wp,
        all_waypoints: list,
    ) -> bool:
        """Call at each pixel step during travel. Runs the mid-path replan check.

        Args:
            step:          Current robot pixel position (col, row).
            current_wp:    The Waypoint the robot is heading toward.
            all_waypoints: Full current waypoint plan.

        Returns:
            True → mid-path replan triggered; caller should break the travel loop.
            False → continue traveling.
        """
        # Mid-path replan check
        if self._replanner is not None:
            return self._replanner.check(step, current_wp, all_waypoints,
                                         self._md.navigable_mask,
                                         self._md.inflation_px)
        return False

    def on_arrive(self, wp) -> None:
        """Call when the robot successfully reaches a waypoint.

        Marks the arrival position and any candidate the camera has now genuinely
        observed as visited, so the planner won't return there.
        """
        self._mark((wp.col, wp.row))

    # ── Internal ─────────────────────────────────────────────────────────────

    def _mark(self, pos: tuple[int, int]) -> None:
        """Mark visited candidates using an FOV/occlusion-aware observation test.

        A candidate is excluded from future planning once the camera has observed
        at least OBSERVED_FRACTION_THRESHOLD of the cells that viewpoint could ever
        see. compute_visibility already subtracts covered_mask, so it returns the
        still-uncovered cells; the full footprint (the denominator) is the same
        360° cast with ignore_covered=True. Using a fraction (not "all cells")
        means a small residual sliver behind an occluder does not keep a
        near-saturated viewpoint alive forever.

        Covered_mask is the FOV-aware source of truth (grown only by the real rotation), so gating on it keeps re-visit suppression honest.

        The arrival position itself is always marked (the robot rotated there, so
        it is observed by definition).
        """
        self.visited_candidates.add(pos)

        max_range_px = max(
            1, int(self._config.get("max_detection_range", 6.0) / self._md.resolution)
        )
        num_rays = self._config.get("num_rays", 360)
        for c in self._candidates:
            if c in self.visited_candidates:
                continue
            remaining, _ = compute_visibility(c, self._md, max_range_px, num_rays)
            # Full footprint: same cast with covered_mask ignored. Must NOT be
            # done by swapping self._md.covered_mask — an exception (or any
            # re-entrancy) mid-swap would leave an empty mask installed on the
            # shared MapData, silently resetting coverage (seen in the
            # 2026-07-16 warehouse run).
            full, _ = compute_visibility(c, self._md, max_range_px, num_rays,
                                         ignore_covered=True)
            if not full:
                # Boxed in by walls / nothing observable → treat as fully seen.
                self.visited_candidates.add(c)
                continue
            observed_fraction = 1.0 - len(remaining) / len(full)
            if observed_fraction >= OBSERVED_FRACTION_THRESHOLD:
                self.visited_candidates.add(c)
