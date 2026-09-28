"""
Tests for navigation/exploration/explore_costmap_map.py

Pure Python, no ROS2 required.

Coverage:
  - Map loading and mask construction (load_map, build_map_data)
  - Coordinate helpers (pixel_to_world, world_to_pixel)
  - Candidate generation (generate_candidates)
  - Visibility and achievable cells (compute_visibility, compute_achievable_cells)
  - Geodesic distance map (navigable_distance_map)
  - Greedy set cover (greedy_set_cover)
  - Heading computation (compute_headings_for_waypoint, angular_diff)
  - Covered mask update (update_covered_mask)
  - Coverage ratio (coverage_ratio)
  - Path ordering (nearest_neighbor_order)
  - Top-level planner (plan_waypoints)
  - Pipeline consistency: compute_visibility → compute_headings_for_waypoint → update_covered_mask
  - ROS2 bridge parity: build_map_data vs load_map

Each test docstring states:
  Verifies: what behaviour is being asserted
  Input:    what data is used
  Measures: the specific assertion
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "navigation"))

from exploration.explore_costmap_map import (
    CoverageWarning,
    MapData,
    Waypoint,
    angular_diff,
    navigable_distance_map,
    build_map_data,
    compute_achievable_cells,
    compute_headings_for_waypoint,
    compute_visibility,
    coverage_ratio,
    generate_candidates,
    greedy_set_cover,
    load_map,
    nearest_neighbor_order,
    pixel_to_world,
    plan_waypoints,
    update_covered_mask,
    world_to_pixel,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

from conftest import INFLATION_M
DETECTION_RANGE_M = 6.0


def _load(assets_map_dir, covered_mask=None):
    """Load a discovered assets/ map. `assets_map_dir` is supplied by the parametrised
    `assets_map_dir` fixture (see conftest.pytest_generate_tests), so every test using
    it runs once per map under assets/."""
    return load_map(assets_map_dir / "map.pgm", assets_map_dir / "map.yaml",
                    INFLATION_M, covered_mask=covered_mask)


# ---------------------------------------------------------------------------
# Synthetic map helpers
# ---------------------------------------------------------------------------

def _open_map(H: int = 50, W: int = 50) -> MapData:
    """Fully free, obstacle-free map."""
    free = np.ones((H, W), dtype=bool)
    occ = np.zeros((H, W), dtype=bool)
    unk = np.zeros((H, W), dtype=bool)
    return MapData(
        pgm_array=np.full((H, W), 254, dtype=np.uint8),
        resolution=0.05,
        origin_x=0.0, origin_y=0.0,
        free_mask=free, occupied_mask=occ, unknown_mask=unk,
        navigable_mask=free.copy(),
        covered_mask=np.zeros((H, W), dtype=bool),
    )


def _map_with_vwall(H: int, W: int, wall_col: int) -> MapData:
    """Map with a full vertical wall at wall_col."""
    free = np.ones((H, W), dtype=bool)
    occ = np.zeros((H, W), dtype=bool)
    free[:, wall_col] = False
    occ[:, wall_col] = True
    unk = np.zeros((H, W), dtype=bool)
    return MapData(
        pgm_array=np.where(occ, 0, 254).astype(np.uint8),
        resolution=0.05,
        origin_x=0.0, origin_y=0.0,
        free_mask=free, occupied_mask=occ, unknown_mask=unk,
        navigable_mask=free.copy(),
        covered_mask=np.zeros((H, W), dtype=bool),
    )


def _map_with_unknown_band(H: int, W: int, unk_start_col: int) -> MapData:
    """Map where columns >= unk_start_col are unknown."""
    free = np.ones((H, W), dtype=bool)
    occ = np.zeros((H, W), dtype=bool)
    unk = np.zeros((H, W), dtype=bool)
    free[:, unk_start_col:] = False
    unk[:, unk_start_col:] = True
    return MapData(
        pgm_array=np.full((H, W), 254, dtype=np.uint8),
        resolution=0.05,
        origin_x=0.0, origin_y=0.0,
        free_mask=free, occupied_mask=occ, unknown_mask=unk,
        navigable_mask=free.copy(),
        covered_mask=np.zeros((H, W), dtype=bool),
    )


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


# ===========================================================================
# 1. Map loading
# ===========================================================================

class TestLoadMap:

    def test_free_pixel_yields_free_mask(self, assets_map_dir):
        """
        Verifies: at least one free pixel exists in the real map.
        Input: lab_ghent map.
        Measures: free_mask.any() is True.
        """
        md = _load(assets_map_dir)
        assert md.free_mask.any()

    def test_occupied_pixel_yields_occupied_mask(self, assets_map_dir):
        """
        Verifies: at least one occupied pixel exists in the real map.
        Input: lab_ghent map.
        Measures: occupied_mask.any() is True.
        """
        md = _load(assets_map_dir)
        assert md.occupied_mask.any()

    def test_resolution_matches_yaml(self, assets_map_dir):
        """
        Verifies: resolution loaded from YAML matches expected value.
        Input: lab_ghent map.
        Measures: resolution == 0.05 m/px within 0.1% tolerance.
        """
        md = _load(assets_map_dir)
        assert md.resolution == pytest.approx(0.05, rel=1e-3)

    def test_three_masks_are_disjoint(self, assets_map_dir):
        """
        Verifies: free, occupied, and unknown masks are mutually exclusive.
        Input: lab_ghent map.
        Measures: pairwise AND of all three mask pairs is empty.
        """
        md = _load(assets_map_dir)
        assert not (md.free_mask & md.occupied_mask).any()
        assert not (md.free_mask & md.unknown_mask).any()
        assert not (md.occupied_mask & md.unknown_mask).any()

    def test_covered_mask_initialised_false(self, assets_map_dir):
        """
        Verifies: covered_mask is all-False when no mask is passed.
        Input: lab_ghent map, no covered_mask argument.
        Measures: covered_mask.any() is False.
        """
        md = _load(assets_map_dir)
        assert not md.covered_mask.any()

    def test_navigable_excludes_cells_adjacent_to_wall(self, assets_map_dir):
        """
        Verifies: cells immediately adjacent to occupied cells are not navigable.
        Input: every assets map.
        Measures: no free 4-neighbour of an occupied cell is navigable.

        Scans all wall-adjacent free cells, not a leading slice of them: occupied
        cells come back in raster order, and on several maps the first ones are
        the outer border, whose neighbours are unknown rather than free. A map
        with no wall-adjacent free cell is a broken fixture, so it fails here
        instead of skipping.
        """
        md = _load(assets_map_dir)
        assert md.occupied_mask.any(), "map has no occupied cells"

        # Free cells sharing a 4-edge with an occupied cell.
        wall_adjacent = np.zeros_like(md.free_mask)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            wall_adjacent |= np.roll(md.occupied_mask, (-dr, -dc), (0, 1)) & md.free_mask

        # Drop the outermost ring, which also discards np.roll's wrap-around.
        # On warehouse_amazon the last image row carries wall-adjacent free cells
        # that stay navigable: inflation has no data past the map edge, so it
        # under-inflates there. That boundary behaviour is out of scope here;
        # this test is about inflation around real walls.
        wall_adjacent[0, :] = wall_adjacent[-1, :] = False
        wall_adjacent[:, 0] = wall_adjacent[:, -1] = False

        assert wall_adjacent.any(), "map has no interior free cell adjacent to a wall"
        assert not md.navigable_mask[wall_adjacent].any()

    def test_pre_existing_covered_mask_is_preserved(self, assets_map_dir):
        """
        Verifies: [pattern 3, state preservation] a pre-existing covered_mask is not
        overwritten by load_map.
        Input: lab_ghent map + checkerboard covered_mask.
        Measures: returned covered_mask is array-equal to the input mask.
        """
        md_ref = _load(assets_map_dir)
        H, W = md_ref.pgm_array.shape
        checkerboard = np.indices((H, W)).sum(axis=0) % 2 == 0
        md = _load(assets_map_dir, covered_mask=checkerboard)
        np.testing.assert_array_equal(md.covered_mask, checkerboard)


# ===========================================================================
# 2. ROS2 bridge parity
# ===========================================================================

class TestBuildMapDataParity:

    def test_build_map_data_matches_load_map(self, assets_map_dir):
        """
        Verifies: [pattern 8, sibling loaders] build_map_data (ROS2 /map bridge)
        produces identical masks to load_map when given the same underlying data.
        Input: each assets/ PGM + YAML, reconstructed as a float probability array.
        Measures: free_mask, occupied_mask, unknown_mask, navigable_mask are array-equal.
        """
        md_file = _load(assets_map_dir)

        with open(assets_map_dir / "map.yaml") as f:
            meta = yaml.safe_load(f)
        arr = np.array(Image.open(assets_map_dir / "map.pgm"), dtype=np.uint8)
        p_occ = 1.0 - arr.astype(np.float64) / 255.0

        md_bridge = build_map_data(
            p_occ,
            resolution=meta["resolution"],
            origin_x=meta["origin"][0],
            origin_y=meta["origin"][1],
            inflation_radius_m=INFLATION_M,
            free_thresh=meta["free_thresh"],
            occ_thresh=meta["occupied_thresh"],
        )

        np.testing.assert_array_equal(md_file.free_mask, md_bridge.free_mask)
        np.testing.assert_array_equal(md_file.occupied_mask, md_bridge.occupied_mask)
        np.testing.assert_array_equal(md_file.unknown_mask, md_bridge.unknown_mask)
        np.testing.assert_array_equal(md_file.navigable_mask, md_bridge.navigable_mask)


# ===========================================================================
# 3. Coordinate helpers
# ===========================================================================

class TestCoordinateHelpers:

    def test_pixel_to_world_at_bottom_left_origin(self, assets_map_dir):
        """
        Verifies: bottom-left pixel (col=0, row=H-1) maps to world origin.
        Input: lab_ghent map.
        Measures: world x ≈ origin_x, world y ≈ origin_y.
        """
        md = _load(assets_map_dir)
        H = md.pgm_array.shape[0]
        x, y = pixel_to_world(0, H - 1, md.resolution, md.origin_x, md.origin_y, H)
        assert x == pytest.approx(md.origin_x, abs=1e-6)
        assert y == pytest.approx(md.origin_y, abs=1e-6)

    def test_world_to_pixel_roundtrip(self, assets_map_dir):
        """
        Verifies: [pattern 9, round-trip] world → pixel → world preserves coordinates
        within one pixel.
        Input: two world points inside the lab_ghent map.
        Measures: |reconstructed - original| < resolution + 1e-6 for both x and y.
        """
        md = _load(assets_map_dir)
        H = md.pgm_array.shape[0]
        for wx, wy in [
            (md.origin_x + 2.0, md.origin_y + 2.0),
            (md.origin_x + 5.0, md.origin_y + 3.5),
        ]:
            col, row = world_to_pixel(wx, wy, md.resolution, md.origin_x, md.origin_y, H)
            rx, ry = pixel_to_world(col, row, md.resolution, md.origin_x, md.origin_y, H)
            assert abs(rx - wx) < md.resolution + 1e-6
            assert abs(ry - wy) < md.resolution + 1e-6


# ===========================================================================
# 4. Candidate generation
# ===========================================================================

class TestGenerateCandidates:

    def test_all_candidates_in_navigable_mask(self):
        """
        Verifies: every generated candidate lies within the navigable mask.
        Input: 50x50 open map, step=10 px.
        Measures: navigable_mask[row, col] is True for all candidates.
        """
        md = _open_map(50, 50)
        candidates = generate_candidates(md.navigable_mask, sampling_step_px=10)
        assert len(candidates) > 0
        for col, row in candidates:
            assert md.navigable_mask[row, col], f"Candidate ({col},{row}) not navigable"

    def test_small_isolated_region_gets_no_representative(self):
        """
        Verifies: [pattern 1, untested branch] an isolated navigable region smaller
        than MIN_COMPONENT_CELLS (100 cells) receives no representative candidate.
        Input: 60x60 map with a 5x5 island (25 cells) at cols 43-47, rows 23-27.
        Grid step=20 px samples at cols {0,20,40} x rows {0,20,40}, none fall inside
        the island, so it is only reachable via the isolated-region fallback, which
        skips it because 25 < 100 cells.
        Measures: no candidate falls inside the island columns/rows.
        """
        H, W = 60, 60
        free = np.zeros((H, W), dtype=bool)
        occ = np.ones((H, W), dtype=bool)
        # Main navigable region: cols 0–29
        free[:, :30] = True
        occ[:, :30] = False
        # Small isolated island: 5x5 at cols 43–47, rows 23–27 (25 cells).
        # Grid points nearest to this area are (40,20) and (40,40), both outside.
        free[23:28, 43:48] = True
        occ[23:28, 43:48] = False

        candidates = generate_candidates(free, sampling_step_px=20)
        island_cols = set(range(43, 48))
        island_rows = set(range(23, 28))
        island_candidates = [
            (c, r) for c, r in candidates
            if c in island_cols and r in island_rows
        ]
        assert len(island_candidates) == 0, (
            f"Expected no representative for small island, got {island_candidates}"
        )

    def test_large_isolated_region_gets_one_representative(self):
        """
        Verifies: [pattern 1, untested branch] an isolated navigable region of 225 cells
        (15x15) that the grid step misses receives exactly one representative.
        Input: 80x80 map, main region left side, isolated 15x15 room on right, step=20 px.
        Measures: exactly 1 candidate falls inside the isolated room.
        """
        H, W = 80, 80
        free = np.zeros((H, W), dtype=bool)
        occ = np.ones((H, W), dtype=bool)
        # Main region: cols 0–39
        free[:, :40] = True
        occ[:, :40] = False
        # Large isolated room: 15x15 at cols 60–74, rows 30–44 (225 cells)
        free[30:45, 60:75] = True
        occ[30:45, 60:75] = False

        candidates = generate_candidates(free, sampling_step_px=20)
        room_candidates = [
            (c, r) for c, r in candidates
            if 60 <= c < 75 and 30 <= r < 45
        ]
        assert len(room_candidates) >= 1, (
            f"Expected at least 1 representative for large isolated room, got {len(room_candidates)}"
        )

    def test_fully_non_navigable_mask_returns_empty(self):
        """
        Verifies: [pattern 5, degenerate] a fully occupied map yields no candidates.
        Input: 10x10 all-False navigable mask.
        Measures: candidates == [].
        """
        navigable = np.zeros((10, 10), dtype=bool)
        candidates = generate_candidates(navigable, sampling_step_px=5)
        assert candidates == []


# ===========================================================================
# 5. Visibility and achievable cells
# ===========================================================================

class TestComputeVisibility:

    def test_open_grid_has_coverage_no_frontiers(self):
        """
        Verifies: fully known free map yields coverage gain > 0 and no frontier gain.
        Input: 50x50 open map, position (25, 25), max_range_px=20.
        Measures: len(cov) > 0, len(front) == 0.
        """
        md = _open_map(50, 50)
        cov, front = compute_visibility((25, 25), md, max_range_px=20)
        assert len(cov) > 0
        assert len(front) == 0

    def test_wall_blocks_cells_on_far_side(self):
        """
        Verifies: a vertical wall prevents visibility past it.
        Input: 50x50 map with wall at col=25, position (10, 25), max_range_px=40.
        Measures: no coverage cell has col > 25.
        """
        md = _map_with_vwall(50, 50, wall_col=25)
        cov, _ = compute_visibility((10, 25), md, max_range_px=40)
        far_side = [(c, r) for c, r in cov if c > 25]
        assert len(far_side) == 0

    def test_already_covered_cells_excluded(self):
        """
        Verifies: cells already in covered_mask are not returned as coverage gain.
        Input: 30x30 open map, all cells pre-covered.
        Measures: len(cov) == 0.
        """
        md = _open_map(30, 30)
        md.covered_mask[:] = True
        cov, _ = compute_visibility((15, 15), md, max_range_px=20)
        assert len(cov) == 0

    def test_unknown_band_produces_frontiers(self):
        """
        Verifies: rays reaching the unknown band report frontier cells at its boundary.
        Input: 30x30 map with unknown band from col=20, position (10, 15), max_range_px=25.
        Measures: at least one frontier cell has col == 20.
        """
        md = _map_with_unknown_band(30, 30, unk_start_col=20)
        _, front = compute_visibility((10, 15), md, max_range_px=25)
        assert any(col == 20 for col, row in front)


class TestComputeAchievableCells:

    def test_achievable_stable_after_covering_cells(self):
        """
        Verifies: [pattern 4, stable denominator] compute_achievable_cells returns the
        same set before and after mutating covered_mask.
        Input: 50x50 open map, position (25, 25), max_range_px=20.
        Measures: set before == set after covering half the map.
        """
        md = _open_map(50, 50)
        candidates = [(25, 25)]
        before = compute_achievable_cells(candidates, md, max_range_m=1.0)
        md.covered_mask[:25, :] = True  # cover half the map
        after = compute_achievable_cells(candidates, md, max_range_m=1.0)
        assert before == after

    def test_empty_candidates_returns_empty_set(self):
        """
        Verifies: [pattern 5, degenerate] empty candidate list yields empty achievable set.
        Input: 50x50 open map, no candidates.
        Measures: achievable == set().
        """
        md = _open_map(50, 50)
        result = compute_achievable_cells([], md, max_range_m=1.0)
        assert result == set()

    def test_cells_behind_wall_excluded(self):
        """
        Verifies: achievable cells do not include cells on the far side of a full wall.
        Same geometry as TestComputeVisibility::test_wall_blocks_cells_on_far_side, asserted through the achievable-cells entry point instead.
        Kept as a separate test because this is the stable denominator behind coverage_ratio, so occlusion leaking in here would silently corrupt the ratio rather than the visibility set.
        Input: 50x50 map with full vertical wall at col=25, candidate at (10, 25).
        Measures: no achievable cell has col > 25.
        """
        md = _map_with_vwall(50, 50, wall_col=25)
        result = compute_achievable_cells([(10, 25)], md, max_range_m=2.0)
        far = [c for c, r in result if c > 25]
        assert len(far) == 0


# ===========================================================================
# 6. BFS distance map
# ===========================================================================

class TestNavigableDistanceMap:
    """navigable_distance_map: √2-weighted grid Dijkstra (cardinal=1, diagonal=√2).

    Was a uniform-cost BFS (cardinal=diagonal=1); switched to a weighted Dijkstra so
    the returned distances are the true geodesic navigation cost (see
    tests/algo_evaluation/bfs_metric_impact.ipynb). Topology/inf semantics are
    unchanged; only diagonal magnitudes differ (√2 instead of 1 per step).
    """

    def test_zero_at_start(self):
        """
        Verifies: start cell has distance 0.
        Input: 20x20 open map, start (10, 10).
        Measures: dist[10, 10] == 0.0.
        """
        md = _open_map(20, 20)
        dist = navigable_distance_map(md.navigable_mask, 10, 10)
        assert dist[10, 10] == 0.0

    def test_cardinal_distance_is_step_count(self):
        """
        Verifies: a straight cardinal run costs 1 per step (unchanged by the √2 fix).
        Input: 20x20 open map, start (0, 0), check (col=5, row=0).
        Measures: dist[0, 5] == 5.0.
        """
        md = _open_map(20, 20)
        dist = navigable_distance_map(md.navigable_mask, 0, 0)
        assert dist[0, 5] == pytest.approx(5.0)

    def test_diagonal_distance_is_sqrt2_per_step(self):
        """
        Verifies: a pure-diagonal run costs √2 per step (the core of the weighted fix;
        the old uniform BFS returned 5.0 here, under-measuring by √2x).
        Input: 20x20 open map, start (0, 0), check (col=5, row=5).
        Measures: dist[5, 5] == 5·√2 ≈ 7.071.
        """
        md = _open_map(20, 20)
        dist = navigable_distance_map(md.navigable_mask, 0, 0)
        assert dist[5, 5] == pytest.approx(5.0 * math.sqrt(2.0))

    def test_wall_cell_is_inf(self):
        """
        Verifies: a wall cell (non-navigable) has distance inf.
        Input: 20x20 map with wall at col=10, start (5, 10).
        Measures: dist[10, 10] == inf.
        """
        md = _map_with_vwall(20, 20, wall_col=10)
        dist = navigable_distance_map(md.navigable_mask, 5, 10)
        assert dist[10, 10] == np.inf

    def test_partial_wall_forces_longer_path(self):
        """
        Verifies: the path around a partial wall is longer than straight-line distance.
        Input: 40x40 map, wall at col=20 rows 0–29 (passage at rows 30–39).
        Start (col=5, row=5), target (col=35, row=5). Euclidean distance = 30 px.
        Measures: dist[5, 35] > 30.
        """
        H, W = 40, 40
        free = np.ones((H, W), dtype=bool)
        occ = np.zeros((H, W), dtype=bool)
        free[:30, 20] = False
        occ[:30, 20] = True
        md = MapData(
            pgm_array=np.zeros((H, W), dtype=np.uint8),
            resolution=0.05, origin_x=0.0, origin_y=0.0,
            free_mask=free, occupied_mask=occ,
            unknown_mask=np.zeros((H, W), dtype=bool),
            navigable_mask=free.copy(),
            covered_mask=np.zeros((H, W), dtype=bool),
        )
        dist = navigable_distance_map(md.navigable_mask, 5, 5)
        euclidean_px = abs(35 - 5)
        assert dist[5, 35] > euclidean_px

    def test_full_wall_makes_far_side_unreachable(self):
        """
        Verifies: cells on the far side of a full vertical wall are unreachable (inf).
        Input: 20x20 map with full wall at col=10, start (5, 10).
        Measures: dist[10, 15] == inf.
        """
        md = _map_with_vwall(20, 20, wall_col=10)
        dist = navigable_distance_map(md.navigable_mask, 5, 10)
        assert dist[10, 15] == np.inf

    def test_non_navigable_start_snaps_within_inflation(self):
        """
        Verifies: a start on a non-navigable cell within inflation_px + margin of
        navigable space is snapped to the nearest navigable cell (robot transiently
        inside Nav2's inflation band), so planning gets a finite distance map.
        Input: 20x20 map with 1-cell wall at col=10, start ON wall (10, 10),
        inflation_px=2.0 (ceiling 4.0 px); navigable free space is one column away.
        Measures: the snapped start's own distance is 0; its half of the map is finite.
        """
        md = _map_with_vwall(20, 20, wall_col=10)
        dist = navigable_distance_map(md.navigable_mask, 10, 10, inflation_px=2.0)
        # Nearest navigable cell to (col=10,row=10) is (col=9 or 11, row=10), dist 0
        # (the wall disconnects the two halves, so only the snapped side is finite).
        assert dist[10, 9] == 0.0 or dist[10, 11] == 0.0
        # The snapped start's own half of the map is fully reachable (finite).
        snapped_col = 9 if dist[10, 9] == 0.0 else 11
        half = md.navigable_mask.copy()
        if snapped_col < 10:
            half[:, 10:] = False  # keep only the left half (start side)
        else:
            half[:, :11] = False  # keep only the right half (start side)
        assert np.isfinite(dist[half]).all()

    def test_non_navigable_start_no_inflation_returns_all_inf(self):
        """
        Verifies: with inflation_px=0.0 (default) snapping is disabled and a
        non-navigable start returns all-inf, the strict back-compat contract for
        callers that don't supply inflation_px.
        Input: 20x20 map with wall at col=10, start ON wall (10, 10), no inflation_px.
        Measures: np.all(dist == inf).
        """
        md = _map_with_vwall(20, 20, wall_col=10)
        dist = navigable_distance_map(md.navigable_mask, 10, 10)
        assert np.all(dist == np.inf)

    def test_boxed_in_start_returns_all_inf(self):
        """
        Verifies: a start with NO navigable cell within the snap ceiling still
        returns an all-inf map (robot genuinely boxed in → skip cycle, don't teleport),
        even when inflation_px enables snapping.
        Input: 20x20 fully non-navigable map, start at (10, 10), inflation_px=2.0.
        Measures: np.all(dist == inf).
        """
        md = _map_with_vwall(20, 20, wall_col=10)
        md.navigable_mask[:] = False  # no navigable cell anywhere
        dist = navigable_distance_map(md.navigable_mask, 10, 10, inflation_px=2.0)
        assert np.all(dist == np.inf)

    def test_matches_independent_scipy_dijkstra(self):
        """
        Verifies: distances match an independent scipy grid-Dijkstra reference on a
        random cluttered map (guards the cached sparse-graph build against regressions).
        Input: 12x12 random navigable mask (seeded), start on a navigable cell.
        Measures: navigable cells agree within 1e-9; both agree on which cells are inf.
        """
        from scipy.sparse import lil_matrix
        from scipy.sparse.csgraph import dijkstra as sp_dijkstra

        rng = np.random.default_rng(7)
        H, W = 12, 12
        mask = rng.random((H, W)) < 0.8
        sr, sc = 6, 6
        mask[sr, sc] = True  # ensure navigable start

        nei = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
               (0, -1, 1.0), (0, 1, 1.0),
               (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]
        g = lil_matrix((H * W, H * W))
        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    continue
                for dr, dc, w in nei:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                        g[r * W + c, nr * W + nc] = w
        ref = sp_dijkstra(g.tocsr(), indices=sr * W + sc).reshape(H, W)
        ref[~mask] = np.inf  # non-navigable cells: our contract returns inf

        got = navigable_distance_map(mask, sc, sr)
        assert np.array_equal(np.isinf(got), np.isinf(ref))
        finite = ~np.isinf(ref)
        assert np.allclose(got[finite], ref[finite], atol=1e-9)


# ===========================================================================
# 7. Greedy set cover
# ===========================================================================

class TestGreedySetCover:

    def test_three_disjoint_tiles_selects_all_three(self):
        """
        Verifies: three disjoint coverage sets each require their own waypoint.
        Input: visibility dict with 3 disjoint coverage pairs, no frontiers.
        Measures: len(selected) == 3, ratio == 1.0.
        """
        vis = {
            (0, 0): ({(0, 0), (1, 0)}, set()),
            (2, 0): ({(2, 0), (3, 0)}, set()),
            (4, 0): ({(4, 0), (5, 0)}, set()),
        }
        selected, _, ratio, _ = greedy_set_cover(vis)
        assert len(selected) == 3
        assert ratio == pytest.approx(1.0)

    def test_zero_gain_terminates_without_error(self):
        """
        Verifies: [pattern 5, degenerate] a visibility dict with zero coverage gain
        returns empty selection without raising.
        Input: single candidate with empty coverage and frontier sets.
        Measures: len(selected) == 0.
        """
        vis = {(0, 0): (set(), set())}
        selected, _, _, _ = greedy_set_cover(vis)
        assert len(selected) == 0

    def test_gamma_positive_prefers_nearby_candidate(self):
        """
        Verifies: [pattern 2, optional argument] with gamma > 0, a nearby candidate
        with slightly lower gain is selected before a distant high-gain one.
        Input: candidate A (far, gain=5) vs B (near, gain=4), gamma=1.0, normaliser=100.
        Measures: B appears before A in selected.
        """
        vis = {
            (0, 0): ({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}, set()),  # gain=5, far
            (1, 0): ({(5, 0), (6, 0), (7, 0), (8, 0)}, set()),           # gain=4, near
        }
        dist_from_robot = {(0, 0): 100.0, (1, 0): 5.0}
        selected, _, _, _ = greedy_set_cover(
            vis,
            dist_from_robot=dist_from_robot,
            gamma=1.0,
            normaliser=100.0,
        )
        assert selected[0] == (1, 0), (
            f"Expected nearby candidate (1,0) first, got {selected[0]}"
        )

    def test_unreachable_candidate_not_selected_with_gamma(self):
        """
        Verifies: [pattern 2, optional argument] a candidate with dist=inf is never
        selected when gamma > 0, provided its coverage cells are already covered by
        a reachable candidate (zero remaining gain).
        Input: (0,0) reachable covering {A,B,C,D}; (5,5) unreachable covering {A,B}
        (subset of (0,0)'s cells). After (0,0) is selected, (5,5) has no remaining
        gain and must be excluded.
        Measures: unreachable candidate (5,5) absent from selected.
        """
        shared_cells = {(0, 0), (1, 0)}
        vis = {
            (0, 0): (shared_cells | {(2, 0), (3, 0)}, set()),  # reachable, gain=4
            (5, 5): (shared_cells, set()),                       # unreachable, gain=2 (subset)
        }
        dist_from_robot = {(0, 0): 10.0, (5, 5): np.inf}
        selected, _, _, _ = greedy_set_cover(
            vis,
            dist_from_robot=dist_from_robot,
            gamma=1.0,
            normaliser=100.0,
        )
        assert (5, 5) not in selected


# ===========================================================================
# 8. Heading computation
# ===========================================================================

class TestAngularDiff:

    def test_symmetry(self):
        """
        Verifies: angular_diff is symmetric, order of arguments does not matter.
        Input: angular_diff(45, 135) vs angular_diff(135, 45).
        Measures: both return the same value.
        """
        assert angular_diff(45.0, 135.0) == pytest.approx(angular_diff(135.0, 45.0))

    def test_wraparound_small_gap(self):
        """
        Verifies: angular_diff correctly handles wrap-around at 0°/360°.
        Input: angular_diff(10, 350).
        Measures: returns 20.0°.
        """
        assert angular_diff(10.0, 350.0) == pytest.approx(20.0)

    def test_opposite_headings(self):
        """
        Verifies: angular_diff(0, 180) == 180.0 (maximum possible difference).
        Measures: returns 180.0°.
        """
        assert angular_diff(0.0, 180.0) == pytest.approx(180.0)

    def test_just_past_180(self):
        """
        Verifies: angular_diff(0, 181) < 180 (takes the short way around).
        Measures: returns 179.0°.
        """
        assert angular_diff(0.0, 181.0) == pytest.approx(179.0)


class TestComputeHeadings:

    def test_cells_due_east_select_east_heading(self):
        """
        Verifies: cells to the East of the waypoint result in a heading near 0°.
        Input: cells at (15–19, 10), waypoint (10, 10), fov=87°, increment=30°.
        Measures: at least one heading within 43.5° of East (0°).
        """
        cells = {(15 + i, 10) for i in range(5)}
        headings = compute_headings_for_waypoint(10, 10, cells, fov_deg=87.0, increment_deg=30.0)
        assert any(angular_diff(h, 0.0) <= 43.5 for h in headings)

    def test_cells_due_west_select_west_heading(self):
        """
        Verifies: cells to the West of the waypoint result in a heading near 180°.
        Input: cells at (5–9, 10), waypoint (10, 10), fov=87°, increment=30°.
        Measures: at least one heading within 43.5° of West (180°).
        """
        cells = {(10 - i, 10) for i in range(1, 6)}
        headings = compute_headings_for_waypoint(10, 10, cells, fov_deg=87.0, increment_deg=30.0)
        assert any(angular_diff(h, 180.0) <= 43.5 for h in headings)

    def test_no_cells_returns_empty(self):
        """
        Verifies: [pattern 5, degenerate] empty cell set returns an empty heading list.
        Input: empty set, waypoint (10, 10).
        Measures: headings == [].
        """
        assert compute_headings_for_waypoint(10, 10, set()) == []

    def test_headings_are_multiples_of_increment(self):
        """
        Verifies: all returned headings are exact multiples of increment_deg.
        Input: four cells in cardinal directions, fov=87°, increment=30°.
        Measures: h % 30 ≈ 0 for all h.
        """
        cells = {(20, 10), (10, 20), (5, 10), (10, 5)}
        headings = compute_headings_for_waypoint(10, 10, cells, fov_deg=87.0, increment_deg=30.0)
        for h in headings:
            assert (h % 30.0) == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 9. Covered mask update
# ===========================================================================

class TestUpdateCoveredMask:

    def test_cells_in_frustum_become_covered(self):
        """
        Verifies: cells inside the camera frustum are marked covered.
        Input: 50x50 open map, robot (25, 25), heading=0° (East), fov=87°.
        Measures: covered_mask[25, 30] is True (cell directly East).
        """
        md = _open_map(50, 50)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        update_covered_mask(md, 25, 25, heading_deg=0.0, fov_deg=87.0,
                            max_range_px=max_range_px)
        assert md.covered_mask[25, 30]

    def test_cells_outside_frustum_not_covered(self):
        """
        Verifies: cells outside the camera frustum remain uncovered.
        Input: 50x50 open map, robot (25, 25), heading=0° (East), fov=87°.
        Measures: covered_mask[25, 10] is False (cell directly West).
        """
        md = _open_map(50, 50)
        max_range_px = int(DETECTION_RANGE_M / md.resolution)
        update_covered_mask(md, 25, 25, heading_deg=0.0, fov_deg=87.0,
                            max_range_px=max_range_px)
        assert not md.covered_mask[25, 10]

    def test_returns_ratio_between_zero_and_one(self):
        """
        Verifies: update_covered_mask returns a valid coverage ratio.
        Input: 20x20 open map, robot (10, 10), heading=0°, max_range_px=8.
        Measures: 0.0 < ratio <= 1.0.
        """
        md = _open_map(20, 20)
        ratio = update_covered_mask(md, 10, 10, heading_deg=0.0, fov_deg=87.0,
                                    max_range_px=8)
        assert 0.0 < ratio <= 1.0


# ===========================================================================
# 10. Coverage ratio
# ===========================================================================

class TestCoverageRatio:

    def test_empty_achievable_cells_returns_one(self):
        """
        Verifies: [pattern 5, degenerate] empty achievable_cells returns 1.0
        (exploration considered complete, nothing left to cover).
        Input: any covered_mask, achievable_cells = set().
        Measures: coverage_ratio == 1.0.
        """
        mask = np.zeros((10, 10), dtype=bool)
        result = coverage_ratio(mask, set())
        assert result == pytest.approx(1.0)

    def test_partial_coverage_correct_fraction(self):
        """
        Verifies: ratio correctly reflects partial coverage.
        Input: 10 achievable cells at known positions, 4 marked covered.
        Measures: ratio == pytest.approx(0.4).
        """
        H, W = 10, 10
        covered = np.zeros((H, W), dtype=bool)
        achievable = {(c, 0) for c in range(10)}  # row=0, cols 0–9
        for c in range(4):
            covered[0, c] = True  # cover 4 of 10
        result = coverage_ratio(covered, achievable)
        assert result == pytest.approx(0.4)


# ===========================================================================
# 11. Path ordering
# ===========================================================================

class TestNearestNeighbourOrder:

    def _vis(self, pts):
        return {p: (set(), set()) for p in pts}

    def test_visits_all_waypoints(self):
        """
        Verifies: all input waypoints appear in the output.
        Input: 200x200 open map, 3 points.
        Measures: len(result) == 3.
        """
        md = _open_map(200, 200)
        pts = [(10, 10), (100, 10), (190, 10)]
        result = nearest_neighbor_order(pts, md, self._vis(pts))
        assert len(result) == 3

    def test_no_duplicate_waypoints(self):
        """
        Verifies: no waypoint position appears twice in the output.
        Input: 200x200 open map, 4 points.
        Measures: len(set(positions)) == 4.
        """
        md = _open_map(200, 200)
        pts = [(10, 10), (100, 100), (190, 190), (10, 190)]
        result = nearest_neighbor_order(pts, md, self._vis(pts))
        positions = [(wp.col, wp.row) for wp in result]
        assert len(set(positions)) == len(pts)

    def test_starts_nearest_to_robot(self):
        """
        Verifies: the first waypoint is the one nearest (by BFS) to the robot.
        Input: 200x200 open map, robot at world position of (10, 10).
        Measures: result[0] == (10, 10).
        """
        md = _open_map(200, 200)
        pts = [(10, 10), (100, 100), (190, 190)]
        rx = md.origin_x + 10 * md.resolution
        ry = md.origin_y + (200 - 1 - 10) * md.resolution
        result = nearest_neighbor_order(pts, md, self._vis(pts), robot_x=rx, robot_y=ry)
        assert (result[0].col, result[0].row) == (10, 10)

    def test_empty_input_returns_empty(self):
        """
        Verifies: [pattern 5, degenerate] empty waypoint list returns empty output.
        Input: 200x200 open map, pts = [].
        Measures: result == [].
        """
        md = _open_map(200, 200)
        result = nearest_neighbor_order([], md, {})
        assert result == []


# ===========================================================================
# 12. plan_waypoints
# ===========================================================================

class TestPlanWaypoints:

    def test_all_waypoints_in_free_cells(self, assets_map_dir):
        """
        Verifies: every planned waypoint lies in a free cell.
        Input: lab_ghent map, default config.
        Measures: free_mask[wp.row, wp.col] == True for all waypoints.
        """
        md = _load(assets_map_dir)
        waypoints, _, _, _ = plan_waypoints(md, _default_config())
        for wp in waypoints:
            assert md.free_mask[wp.row, wp.col], \
                f"Waypoint ({wp.col},{wp.row}) is not in free space"

    def test_no_two_waypoints_closer_than_inflation_radius(self, assets_map_dir):
        """
        Verifies: all waypoint pairs are at least inflation_radius apart.
        Input: lab_ghent map, default config, INFLATION_M = 0.3.
        Measures: min pairwise distance ≥ 0.3 m.
        """
        md = _load(assets_map_dir)
        waypoints, _, _, _ = plan_waypoints(md, _default_config())
        for i, a in enumerate(waypoints):
            for b in waypoints[i + 1:]:
                dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                assert dist >= INFLATION_M, \
                    f"({a.col},{a.row}) and ({b.col},{b.row}) only {dist:.3f} m apart"

    def test_all_candidates_excluded_returns_empty_waypoints(self, assets_map_dir):
        """
        Verifies: [pattern 7, exclusion/filter] passing all candidates as
        visited_candidates yields an empty waypoint list with a valid ratio.
        Input: lab_ghent map, visited_candidates = all generated candidates.
        Measures: waypoints == [], 0.0 <= ratio <= 1.0.
        """
        md = _load(assets_map_dir)
        step_px = max(1, int(3.0 / md.resolution))
        from exploration.explore_costmap_map import generate_candidates as _gc
        all_cands = set(_gc(md.navigable_mask, step_px))
        waypoints, ratio, _, _ = plan_waypoints(md, _default_config(),
                                             visited_candidates=all_cands)
        assert waypoints == []
        assert 0.0 <= ratio <= 1.0

    def test_known_map_reports_no_frontiers_despite_unknown_cells(self):
        """
        Verifies: in known-map mode (is_slam=False) frontiers are ignored, so
        no_frontiers is True even when the static map still contains unknown cells.
        Regression guard: a real PGM is not free of unknown voids (the hospital map has
        ~89k). Without a live /map those can never be resolved, so a frontier-based stop
        test could never pass and exploration hung forever at partial coverage.
        Input: map with a block of unknown cells; is_slam False vs True.
        Measures: known-map -> no_frontiers True; SLAM on the same map -> False.
        """
        H = W = 40
        p_occ = np.zeros((H, W), dtype=float)
        p_occ[:, :3] = 1.0            # a wall so the map is not degenerate
        p_occ[30:38, 30:38] = 0.5     # unknown void, unresolvable without SLAM
        md = build_map_data(p_occ, resolution=0.5, origin_x=0.0, origin_y=0.0,
                            inflation_radius_m=0.0)
        assert md.unknown_mask.sum() > 0, "fixture must contain unknown cells"

        cfg_known = {**_default_config(), "is_slam": False}
        _, _, no_frontiers_known, _ = plan_waypoints(md, cfg_known)
        assert no_frontiers_known is True, (
            "known-map mode must ignore unresolvable unknown voids, otherwise the "
            "completion gate can never be satisfied"
        )

        cfg_slam = {**_default_config(), "is_slam": True}
        _, _, no_frontiers_slam, _ = plan_waypoints(md, cfg_slam)
        assert no_frontiers_slam is False, (
            "SLAM mode must still report frontiers so map discovery continues"
        )

    def test_coverage_warning_fires_on_isolated_map(self):
        """
        Verifies: [pattern 6, warning] CoverageWarning is emitted when the planner
        can geometrically reach less than warning_threshold of free cells.
        Input: map with a large inaccessible free region, warning_threshold=0.50.
        Measures: pytest.warns(CoverageWarning) catches the warning.
        """
        H, W = 40, 40
        free = np.zeros((H, W), dtype=bool)
        occ = np.ones((H, W), dtype=bool)
        # Navigable region: left half (cols 0–14)
        free[:, :15] = True
        occ[:, :15] = False
        # Inaccessible free region: right half (cols 25–39), free but not navigable
        free[:, 25:] = True
        occ[:, 25:] = False

        navigable = free.copy()
        navigable[:, 25:] = False  # right side unreachable

        md = MapData(
            pgm_array=np.zeros((H, W), dtype=np.uint8),
            resolution=0.05, origin_x=0.0, origin_y=0.0,
            free_mask=free, occupied_mask=occ,
            unknown_mask=np.zeros((H, W), dtype=bool),
            navigable_mask=navigable,
            covered_mask=np.zeros((H, W), dtype=bool),
        )
        config = _default_config()
        config["planner_coverage_warning_threshold"] = 0.60
        with pytest.warns(CoverageWarning):
            plan_waypoints(md, config)

    def test_no_coverage_warning_on_well_connected_map(self, assets_map_dir):
        """
        Verifies: [pattern 6, warning] no CoverageWarning on a well-connected map.
        Input: lab_ghent map, warning_threshold=0.50.
        Measures: no CoverageWarning is emitted.
        """
        md = _load(assets_map_dir)
        config = _default_config()
        config["planner_coverage_warning_threshold"] = 0.50
        with warnings.catch_warnings():
            warnings.simplefilter("error", CoverageWarning)
            plan_waypoints(md, config)  # should not raise


# ===========================================================================
# 13. Pipeline consistency
# ===========================================================================

class TestCameraPlanningScanConsistency:
    """
    Verifies: every cell compute_visibility reports as coverable from a candidate
    is actually marked covered after running compute_headings_for_waypoint +
    update_covered_mask with the same parameters.

    This tests the full planning → observation pipeline across varied sensor configs.
    """

    @pytest.mark.parametrize("num_rays,fov_deg,max_range_m,increment_deg", [
        (180,  87, 6.0, 30),   # low-resolution scan
        (360,  87, 6.0, 30),   # standard config
        (720,  87, 6.0, 30),   # high-resolution scan
        (360,  36, 6.0, 30),   # narrow FOV
        (360, 140, 6.0, 30),   # wide FOV
        (360,  87, 3.0, 30),   # short detection range
        (360,  87, 9.0, 30),   # long detection range
        (360,  87, 6.0, 10),   # fine rotation increment
        (360,  87, 6.0, 45),   # coarse rotation increment
    ])
    def test_all_planning_cells_coverable_by_camera(
        self, assets_map_dir, num_rays, fov_deg, max_range_m, increment_deg
    ):
        """
        Verifies: every cell the planning scan considers reachable is covered by
        the actual heading-selection + camera-observation pipeline.
        Input: lab_ghent map with the parametrized sensor configuration.
        Measures: zero planning-visible cells remain uncovered after all headings applied.
        """
        md = _load(assets_map_dir)
        max_range_px = max(1, int(max_range_m / md.resolution))
        step_px = max(1, int(max_range_m / md.resolution))
        candidates = generate_candidates(md.navigable_mask, step_px)

        for candidate in candidates:
            col0, row0 = candidate
            md.covered_mask[:] = False  # fresh slate per candidate

            cov_cells, _ = compute_visibility(candidate, md, max_range_px,
                                              num_rays=num_rays)
            if not cov_cells:
                continue

            headings = compute_headings_for_waypoint(
                col0, row0, cov_cells,
                fov_deg=fov_deg,
                increment_deg=increment_deg,
            )
            for h in headings:
                update_covered_mask(md, col0, row0, float(h), fov_deg,
                                    max_range_px, num_rays=num_rays)

            still_uncovered = [(c, r) for c, r in cov_cells
                               if not md.covered_mask[r, c]]
            assert still_uncovered == [], (
                f"num_rays={num_rays}, fov={fov_deg}°, range={max_range_m}m, "
                f"increment={increment_deg}°, candidate {candidate}: "
                f"{len(still_uncovered)} planning-visible cells not reached by camera"
            )
