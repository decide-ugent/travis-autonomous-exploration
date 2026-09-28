"""
Tests for navigation/exploration/exploration/execution_strategy.py

Each test targets one measurable behaviour. Docstrings state:
  - what is being tested
  - what inputs are used
  - what the assertion measures

All tests use small synthetic numpy boolean arrays,  no PGM files, no matplotlib.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.execution_strategy import (
    OBSERVED_FRACTION_THRESHOLD,
    ExplorationSession,
    MidPathReplanner,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_map_data(
    H: int = 20,
    W: int = 20,
    resolution: float = 1.0,
    navigable: np.ndarray | None = None,
) -> SimpleNamespace:
    """Return a minimal MapData-like object accepted by ExplorationSession."""
    if navigable is None:
        navigable = np.ones((H, W), dtype=bool)
    md = SimpleNamespace(
        navigable_mask=navigable,
        free_mask=navigable.copy(),
        occupied_mask=~navigable,
        unknown_mask=np.zeros((H, W), dtype=bool),
        covered_mask=np.zeros((H, W), dtype=bool),
        pgm_array=np.full((H, W), 254, dtype=np.uint8),
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        height=H,
        width=W,
    )
    return md


def _base_config(**overrides) -> dict:
    cfg = {
        "max_detection_range": 6.0,
        "fov_horizontal": 87.0,
        "observation_rotation_increment": 30.0,
        "sampling_step_m": 5.0,
        "num_rays": 360,
        "frontier_weight": 1.0,
        "coverage_weight": 1.0,
        "exploration_completion_threshold": 0.90,
        "planner_coverage_warning_threshold": 0.70,
    }
    cfg.update(overrides)
    return cfg


def _fake_wp(col: int, row: int) -> SimpleNamespace:
    return SimpleNamespace(col=col, row=row, headings=[])


# ---------------------------------------------------------------------------
# ExplorationSession,  nearest_start
# ---------------------------------------------------------------------------

class TestNearestStart:

    def test_nearest_start_does_not_mark(self):
        """
        Verifies: nearest_start only picks a start candidate; it does NOT mark
        anything visited. Marking is arrival-only and FOV-aware (see on_arrive).
        Input: 20x20 all-navigable map, resolution=1.0, sampling_step_m=5.0.
        Measures: visited_candidates stays empty after nearest_start.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        assert len(session.visited_candidates) == 0
        session.nearest_start(0, 0)
        assert len(session.visited_candidates) == 0

    def test_returns_nearest_candidate(self):
        """
        Verifies: nearest_start returns the candidate minimising squared distance to (cx, cy).
        The expectation is derived from the actual candidate pool rather than hardcoded.
        A change to the sampling grid (step, phase, or the isolated-region pass) therefore cannot fail this test spuriously; only a change to nearest_start's own selection rule can.
        Input: 20x20 all-navigable, call nearest_start(12, 7).
        Measures: result == argmin over session._candidates.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        got = session.nearest_start(12, 7)
        expected = min(session._candidates,
                       key=lambda c: (c[0] - 12) ** 2 + (c[1] - 7) ** 2)
        assert got == expected

    def test_degenerate_single_navigable_cell(self):
        """
        Verifies: nearest_start works when only one navigable cell exists.
        Input: 20x20 grid all False except [10, 10]=True, sampling_step_m=1.0.
        Measures: returns (10, 10).
        """
        nav = np.zeros((20, 20), dtype=bool)
        nav[10, 10] = True
        md = _make_map_data(navigable=nav)
        session = ExplorationSession(md, _base_config(sampling_step_m=1.0))
        col, row = session.nearest_start(0, 0)
        assert (col, row) == (10, 10)


# ---------------------------------------------------------------------------
# ExplorationSession,  on_step
# ---------------------------------------------------------------------------

class TestOnStep:

    def _session_at_origin(self) -> ExplorationSession:
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        session.nearest_start(0, 0)
        session._replanner = None   # isolate from MidPathReplanner for these tests
        return session

    def test_on_step_does_not_mark(self):
        """
        Verifies: on_step performs ONLY the mid-path replan check; it never marks
        candidates visited. Continuous/displacement-based marking was removed,
        marking is arrival-only and FOV-aware (see on_arrive / _mark). A candidate
        is credited as visited only after a real observation, never for merely
        driving near it.
        Input: base map; after nearest_start(0,0); several steps of increasing
               displacement.
        Measures: visited_candidates stays empty across all steps.
        """
        session = self._session_at_origin()
        for step in [(1, 0), (4, 0), (5, 0), (8, 0), (10, 0)]:
            session.on_step(step, _fake_wp(15, 15), [])
            assert len(session.visited_candidates) == 0

    def test_returns_false_without_replanner(self):
        """
        Verifies: on_step returns False when _replanner is None (no mid-path replan possible).
        Input: base map; replanner explicitly set to None.
        Measures: return value is False.
        """
        session = self._session_at_origin()
        result = session.on_step((10, 0), _fake_wp(15, 15), [])
        assert result is False


# ---------------------------------------------------------------------------
# ExplorationSession,  on_arrive
# ---------------------------------------------------------------------------

class TestOnArrive:

    def test_marks_waypoint_visited(self):
        """
        Verifies: on_arrive adds the waypoint's position to visited_candidates.
        Input: base map; nearest_start(0,0); on_arrive(wp) with wp.col=15, wp.row=15.
        Measures: (15, 15) in visited_candidates.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        session.nearest_start(0, 0)
        wp = _fake_wp(15, 15)
        session.on_arrive(wp)
        assert (15, 15) in session.visited_candidates

    def test_unobserved_candidates_not_marked(self):
        """
        Verifies: on_arrive marks the arrival position but does NOT mark other
        candidates whose visible footprint has not been observed (covered_mask is
        empty). This is the FOV-aware contract that replaced the wall-blind
        Euclidean disc, a candidate is excluded only once genuinely observed, so
        the pool is not sterilised by mere proximity.
        Input: 20x20 all-navigable map with covered_mask all-False; on_arrive at
               (10,10).
        Measures: only (10,10) is in visited_candidates; no neighbour is pulled in.
        """
        md = _make_map_data()  # covered_mask is all False
        session = ExplorationSession(md, _base_config())
        session.nearest_start(0, 0)
        session.on_arrive(_fake_wp(10, 10))
        assert session.visited_candidates == {(10, 10)}

    def test_fully_observed_map_marks_every_candidate(self):
        """
        Verifies: on_arrive marks a candidate visited once its visible footprint is already fully in covered_mask (i.e. the camera has observed everything that viewpoint could add).
        This is the degenerate end of the range: with the whole grid covered every candidate passes the gate trivially.
        The threshold itself is pinned by test_candidate_below_observed_threshold_is_not_marked and test_candidate_above_observed_threshold_is_marked.
        Input: 20x20 all-navigable map; mark the ENTIRE grid covered, so every candidate's footprint is observed; on_arrive at (10,10).
        Measures: every candidate, not just the arrival position, is marked visited.
        """
        md = _make_map_data()
        md.covered_mask = np.ones_like(md.covered_mask)  # everything observed
        session = ExplorationSession(md, _base_config())
        session.nearest_start(0, 0)
        session.on_arrive(_fake_wp(10, 10))
        # Every candidate is fully observed, so all get marked, not just (10,10).
        assert len(session.visited_candidates) == len(session._candidates)

    def _cover_fraction_of_footprint(self, md, target, frac):
        """Mark `frac` of everything `target` could ever see as covered.

        The denominator is the same 360 degree cast _mark uses (ignore_covered=True), so the fraction here is exactly the one the gate computes.
        Returns the achieved fraction, which differs slightly from `frac` because cells are discrete.
        """
        from exploration.explore_costmap_map import compute_visibility
        full, _ = compute_visibility(target, md, 6, 360, ignore_covered=True)
        cells = sorted(full)
        for col, row in cells[:int(len(cells) * frac)]:
            md.covered_mask[row, col] = True
        remaining, _ = compute_visibility(target, md, 6, 360)
        return 1.0 - len(remaining) / len(full)

    def test_candidate_below_observed_threshold_is_not_marked(self):
        """
        Verifies: a candidate whose footprint is only partially observed stays in the pool.
        This pins the boundary OBSERVED_FRACTION_THRESHOLD actually sets; without it the constant could be changed to any value and no test would notice.
        Input: 20x20 all-navigable; ~90% of (5,5)'s footprint covered; arrive elsewhere.
        Measures: (5,5) is NOT in visited_candidates.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        achieved = self._cover_fraction_of_footprint(md, (5, 5), 0.90)
        assert achieved < OBSERVED_FRACTION_THRESHOLD, "fixture must sit below the gate"
        session.on_arrive(_fake_wp(15, 15))
        assert (5, 5) not in session.visited_candidates

    def test_candidate_above_observed_threshold_is_marked(self):
        """
        Verifies: once past the threshold the candidate is suppressed, so a residual sliver behind an occluder cannot keep a near-saturated viewpoint alive forever.
        Input: 20x20 all-navigable; ~97% of (5,5)'s footprint covered; arrive elsewhere.
        Measures: (5,5) IS in visited_candidates.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        achieved = self._cover_fraction_of_footprint(md, (5, 5), 0.97)
        assert achieved >= OBSERVED_FRACTION_THRESHOLD, "fixture must sit above the gate"
        session.on_arrive(_fake_wp(15, 15))
        assert (5, 5) in session.visited_candidates


# ---------------------------------------------------------------------------
# ExplorationSession,  on_unreachable
# ---------------------------------------------------------------------------

class TestOnUnreachable:
    """on_unreachable reads only len(path) and path[0]; it never consults the map.
    The fixtures below are therefore shape-only, not walkable 8-connected paths.
    """

    def test_true_for_empty_path_start_not_goal(self):
        """
        Verifies: on_unreachable returns True when find_path returned only the start (goal unreachable).
        Input: wp.col=5, wp.row=5; path=[(0,0)] (robot stayed put, goal ≠ start).
        Measures: returns True.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        wp = _fake_wp(5, 5)
        assert session.on_unreachable(wp, [(0, 0)]) is True

    def test_false_for_valid_path(self):
        """
        Verifies: on_unreachable returns False when find_path reached the goal.
        Input: wp.col=5, wp.row=5; path contains multiple steps ending at goal.
        Measures: returns False.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        wp = _fake_wp(5, 5)
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 5)]
        assert session.on_unreachable(wp, path) is False

    def test_false_when_start_equals_goal(self):
        """
        Verifies: on_unreachable returns False when robot is already at the waypoint.
        Input: wp.col=0, wp.row=0; path=[(0,0)].
        Measures: returns False (robot IS at the goal).
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        wp = _fake_wp(0, 0)
        assert session.on_unreachable(wp, [(0, 0)]) is False


# ---------------------------------------------------------------------------
# ExplorationSession,  plan / visited exclusion
# ---------------------------------------------------------------------------

class TestPlanExclusion:

    def test_all_candidates_visited_returns_empty_plan(self):
        """
        Verifies: planning returns no waypoints when all candidates are already visited.
        Input: base map; manually set visited_candidates = set(_candidates).
        Measures: plan_waypoints_raw(0.0, 0.0) yields an empty waypoint list.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        session.nearest_start(0, 0)
        session.visited_candidates = set(session._candidates)
        waypoints, _ratio, _no_frontiers, _records = session.plan_waypoints_raw(0.0, 0.0)
        assert waypoints == []


# ---------------------------------------------------------------------------
# ExplorationSession,  SLAM candidate pool growth
# ---------------------------------------------------------------------------

class TestSLAMCandidateGrowth:

    def test_candidate_pool_grows_after_mask_expansion(self):
        """
        Verifies: _candidates regenerated from the current navigable_mask when a new
                  map is bound via set_map(), so newly revealed SLAM cells enter the pool.
        Input: 20x20 grid, left half navigable; then a new map with the right half
               also navigable, bound via set_map() (as the node does on _on_map).
        Measures: len(_candidates) after set_map > before.
        """
        nav = np.zeros((20, 20), dtype=bool)
        nav[:, :10] = True
        md = _make_map_data(navigable=nav)
        session = ExplorationSession(md, _base_config(sampling_step_m=5.0))
        session.nearest_start(5, 10)

        candidates_before = len(session._candidates)

        # Simulate SLAM revealing the right half: a new MapData is built and bound
        # via set_map() (the node rebuilds MapData in _on_map, never mutates in place).
        nav_expanded = np.ones((20, 20), dtype=bool)
        md_expanded = _make_map_data(navigable=nav_expanded)
        session.set_map(md_expanded)

        candidates_after = len(session._candidates)
        assert candidates_after > candidates_before


# ---------------------------------------------------------------------------
# MidPathReplanner
# ---------------------------------------------------------------------------

class TestMidPathReplanner:

    def _all_nav_mask(self, H: int = 20, W: int = 20) -> np.ndarray:
        return np.ones((H, W), dtype=bool)

    def test_no_fire_on_first_step(self):
        """
        Verifies: check() returns False on the very first call (no prior position stored).
        Input: interval=5.0; step=(0,0); navigable 20x20 mask.
        Measures: returns False; _last_check set to (0,0).
        """
        rp = MidPathReplanner(check_interval_px=5.0)
        wp = _fake_wp(18, 10)
        result = rp.check((0, 0), wp, [wp], self._all_nav_mask())
        assert result is False
        assert rp._last_check == (0, 0)

    def test_no_fire_below_interval(self):
        """
        Verifies: check() returns False when displacement from last check < interval.
        Input: interval=5.0; first call at (0,0), second call at (3,0) (displacement=3 < 5).
        Measures: second call returns False.
        """
        rp = MidPathReplanner(check_interval_px=5.0)
        mask = self._all_nav_mask()
        wp = _fake_wp(18, 10)
        rp.check((0, 0), wp, [wp], mask)
        result = rp.check((3, 0), wp, [wp], mask)
        assert result is False

    def test_fires_when_cheaper_waypoint_exists(self):
        """
        Verifies: check() returns True when an alternative waypoint costs < 50% of current target.
        Input: 20x20 navigable; interval=1.0; robot at (10,10); current target at (18,10)
               (BFS dist≈8); alternative wp at (11,10) (BFS dist≈1 < 8x0.5=4).
        Measures: returns True.
        """
        mask = self._all_nav_mask()
        rp = MidPathReplanner(check_interval_px=1.0)
        current_wp = _fake_wp(18, 10)
        cheap_wp   = _fake_wp(11, 10)

        # Arm last_check far enough away that the next step triggers a BFS check
        rp._last_check = (0, 0)

        result = rp.check((10, 10), current_wp, [current_wp, cheap_wp], mask)
        assert result is True

    def test_no_fire_when_no_waypoint_is_cheaper(self):
        """
        Verifies: check() returns False when no alternative drops below
                  dist_to_target * replan_ratio (default 0.75).
        Input: 20x20 navigable; interval=1.0; robot at (2,0); target at (5,0)
               (geodesic dist=3); alternative at (5,3) (3 diagonals ≈ 4.24,
               NOT < 3x0.75=2.25).
        Measures: returns False.
        """
        mask = self._all_nav_mask()
        rp = MidPathReplanner(check_interval_px=1.0)
        current_wp = _fake_wp(5, 0)
        far_alt    = _fake_wp(5, 3)   # dist from (2,0) ≈ 3+ (not < 3*0.75)

        rp._last_check = (0, 0)

        result = rp.check((2, 0), current_wp, [current_wp, far_alt], mask)
        assert result is False

    def test_replan_ratio_is_configurable(self):
        """
        Verifies: the replan_ratio argument controls the diversion threshold,  the
                  SAME geometry fires with an eager ratio and not with a strict one.
        Input: robot at (2,0); target (5,0) dist=3; alternative (4,0) dist=2.
               2 < 3*0.75 → fires at default; 2 < 3*0.5=1.5 is False → no fire at 0.5.
        Measures: check() True at ratio 0.75, False at ratio 0.5.
        """
        mask = self._all_nav_mask()
        current_wp = _fake_wp(5, 0)
        near_alt   = _fake_wp(4, 0)

        eager = MidPathReplanner(check_interval_px=1.0, replan_ratio=0.75)
        eager._last_check = (0, 0)
        assert eager.check((2, 0), current_wp, [current_wp, near_alt], mask) is True

        strict = MidPathReplanner(check_interval_px=1.0, replan_ratio=0.5)
        strict._last_check = (0, 0)
        assert strict.check((2, 0), current_wp, [current_wp, near_alt], mask) is False


# ---------------------------------------------------------------------------
# Pipeline consistency: visited_candidates grows monotonically
# ---------------------------------------------------------------------------

class TestMonotonicVisited:

    def test_visited_candidates_never_shrinks(self):
        """
        Verifies: visited_candidates is non-decreasing across nearest_start, on_step x N, on_arrive.
        Note: visited_candidates is append-only (_mark only ever adds), so this cannot fail today.
        It exists as a guard against a future .discard() or reassignment, not as a test of on_step.
        Input: base map; steps at x=0,3,6,9,12,15; on_arrive at (15,15).
        Measures: each observation of len(visited_candidates) >= previous.
        """
        md = _make_map_data()
        session = ExplorationSession(md, _base_config())
        session._replanner = None

        session.nearest_start(0, 0)
        prev = len(session.visited_candidates)

        for x in [3, 6, 9, 12, 15]:
            session.on_step((x, 0), _fake_wp(15, 15), [])
            curr = len(session.visited_candidates)
            assert curr >= prev, f"visited_candidates shrank at x={x}: {curr} < {prev}"
            prev = curr

        session.on_arrive(_fake_wp(15, 15))
        curr = len(session.visited_candidates)
        assert curr >= prev, f"visited_candidates shrank on on_arrive: {curr} < {prev}"


# ---------------------------------------------------------------------------
# "Could not reach" must never be recorded as "visited"
#
# Regression guard for the hospital stall: an aborted waypoint was blacklisted into visited_candidates, which told the planner that area was already covered. Coverage then froze (56.6 %) and the candidate pool drained until no plan could be produced, leaving the node retrying forever. Reaching and observing are different facts.
# ---------------------------------------------------------------------------

class TestUnreachableIsNotVisited:

    def _aborting_session(self, n_aborts: int):
        md = _make_map_data()
        session = ExplorationSession(md, _base_config(abort_blacklist_after=3))
        wp = _fake_wp(10, 10)
        blacklisted = False
        for _ in range(n_aborts):
            blacklisted = session.on_nav_aborted(wp)
        return session, wp, blacklisted

    def test_aborted_waypoint_never_enters_visited_candidates(self):
        """
        Verifies: a waypoint Nav2 repeatedly aborts is NOT recorded as visited.
        Input: 3 aborts on (10,10) with abort_blacklist_after=3.
        Measures: (10,10) is in _unreachable but absent from visited_candidates.
        """
        session, wp, blacklisted = self._aborting_session(3)
        assert blacklisted is True
        assert (10, 10) in session._unreachable
        assert (10, 10) not in session.visited_candidates, (
            "an unreached waypoint was marked 'visited', the planner would treat that "
            "area as already covered and never return to it"
        )

    def test_aborted_waypoint_is_still_excluded_from_planning(self):
        """
        Verifies: the abort livelock stays suppressed,  a blacklisted waypoint is still excluded from planning (via the visited|unreachable union), just not as 'visited'.
        Input: 3 aborts on (10,10).
        Measures: the exclusion set passed to the planner contains it.
        """
        session, _, _ = self._aborting_session(3)
        exclusion = session.visited_candidates | session._unreachable
        assert (10, 10) in exclusion

    def test_below_threshold_aborts_do_not_blacklist(self):
        """
        Verifies: fewer aborts than the threshold leave the waypoint fully live.
        Input: 2 aborts with abort_blacklist_after=3.
        Measures: not blacklisted, and in neither set.
        """
        session, _, blacklisted = self._aborting_session(2)
        assert blacklisted is False
        assert (10, 10) not in session._unreachable
        assert (10, 10) not in session.visited_candidates

    def test_clear_unreachable_restores_and_reports_count(self):
        """
        Verifies: clear_unreachable() restores abandoned waypoints and returns how many, without touching genuinely observed ones.
        Input: 3 aborts on (10,10) + an observed waypoint via on_arrive.
        Measures: return value == 1, _unreachable empty, visited_candidates preserved.
        """
        session, _, _ = self._aborting_session(3)
        session.on_arrive(_fake_wp(15, 15))
        observed_before = set(session.visited_candidates)
        assert observed_before, "on_arrive should have recorded an observed viewpoint"

        restored = session.clear_unreachable()

        assert restored == 1
        assert session._unreachable == set()
        assert session.visited_candidates == observed_before, (
            "clear_unreachable() must not discard genuinely observed viewpoints"
        )

    def test_arrival_does_not_restore_unreachable(self):
        """
        Verifies: a successful arrival does NOT bulk-restore the unreachable set.

        Restoring on every arrival livelocks: a permanently unreachable waypoint would be restored, instantly re-selected as highest-gain, abort, be re-parked, and repeat
        forever, and because plans never go empty in that loop, no timeout could catch
        it. The restore is reserved for genuine pool starvation (clear_unreachable).
        Input: 3 aborts on (10,10), then on_arrive_clear_aborts for a DIFFERENT waypoint.
        Measures: (10,10) stays parked in _unreachable.
        """
        session, _, _ = self._aborting_session(3)
        session.on_arrive_clear_aborts(_fake_wp(15, 15))
        assert (10, 10) in session._unreachable, (
            "arrival must not bulk-restore unreachable waypoints (livelock risk)"
        )

    def test_arrival_clears_only_its_own_abort_counter(self):
        """
        Verifies: arriving at a waypoint forgives ITS transient aborts, so they do not accumulate toward blacklisting on a waypoint that is demonstrably reachable.
        Input: 2 aborts on (10,10) (below threshold), then arrive at (10,10).
        Measures: its counter is gone, so the next abort starts from 1 (not 3).
        """
        session, wp, _ = self._aborting_session(2)
        assert session._abort_counts.get((10, 10)) == 2
        session.on_arrive_clear_aborts(wp)
        assert (10, 10) not in session._abort_counts
        assert session.on_nav_aborted(wp) is False   # restarted at 1, not blacklisted
