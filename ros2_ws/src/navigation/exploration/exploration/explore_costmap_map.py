"""
Exploration waypoint planner for a known or partially-known occupancy grid map.

Pure Python: no rclpy imports.

Strategy: Iterative Greedy Set Cover with unified frontier + coverage scoring.

    score(candidate) = alpha x frontier_gain + beta x coverage_gain

    frontier_gain: unknown cells revealed by a planning scan (map discovery, SLAM)
    coverage_gain: free, not-yet-camera-observed cells revealed (semantic map)

Both gains come from a single 360° planning scan per candidate position.
This 360° scan is a planning abstraction representing full rotation potential: it answers "if the robot stands here and rotates completely, which cells could it observe?" . The actual camera observation is modelled in update_covered_mask() and compute_headings_for_waypoint().

Known-map mode (config `is_slam: False`): frontier gain is IGNORED (alpha forced to 0 and
no_frontiers forced True) → pure coverage. This is enforced explicitly rather than assumed:
a real static PGM is NOT free of unknown cells (the hospital map has ~89k of them — wall
interiors, voids outside the building). Without a live /map those cells can never be
resolved, so a frontier-based stop test would never pass and exploration would hang forever
at partial coverage, while the voids would also attract waypoints that can never be observed.
SLAM mode: both gains compete → the robot discovers the map and builds the semantic map simultaneously in a single pass.

Stop condition (caller checks):
    no_frontiers  (SLAM: distinct visible frontier cells <= min_frontier_cells;
                   known map: always True, see above)
    AND
    coverage_ratio >= exploration_completion_threshold

Under SLAM no_frontiers stays False until the map is discovered, so the robot never stops early just because enough of the currently-known area is covered.

Coverage ratio denominator = achievable_cells (union of all candidate coverage visibility). Cells behind walls are excluded automatically.

Robot position in planning:
    plan_waypoints() accepts (robot_x, robot_y). The position is used to set the TSP start: the first waypoint is always the one nearest to the robot.
    Additionally, covered_mask encodes everything the robot has already observed; on each re-plan, cells in covered_mask have zero gain and are naturally skipped.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as ndlabel, maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra


class CoverageWarning(UserWarning):
    pass


#--------------------------------------------------------------------------
# Data structures
#--------------------------------------------------------------------------

@dataclass
class MapData:
    pgm_array: np.ndarray       # uint8 (H, W): raw pixel values; source of H, and for the visualiser
    resolution: float           # metres per pixel
    origin_x: float             # world X of bottom-left pixel (col=0, row=H-1)
    origin_y: float             # world Y of bottom-left pixel
    free_mask: np.ndarray       # bool (H, W): cells to visually cover
    occupied_mask: np.ndarray   # bool (H, W): blocks navigation and ray casts
    unknown_mask: np.ndarray    # bool (H, W): not yet mapped; stops rays, potential frontier
    navigable_mask: np.ndarray  # bool (H, W): valid robot standing positions
    covered_mask: np.ndarray    # bool (H, W): camera-observed; mutable, persists across re-plans
    inflation_px: float = 0.0   # Nav2 costmap inflation radius in pixels (inflation_radius_m / resolution).
    # Width of the non-navigable band around obstacles; used to bound the start-snap
    # when the robot sits inside the inflation band. 0.0 (default) → snap disabled.


@dataclass
class Waypoint:
    x: float                    # world metres
    y: float                    # world metres
    col: int                    # pixel column
    row: int                    # pixel row
    headings: list[float] = field(default_factory=list)
    # Planned camera headings in degrees (0=East, 90=North, CCW positive).
    # Minimum set of 87°-wide windows to observe all reachable uncovered cells.
    # Refined at runtime when the robot arrives, based on actual covered_mask state.
    # Planner diagnostic fields — populated by plan_waypoints, None if unavailable.
    frontier_gain: int | None = None   # total unknown cells visible from this position
    coverage_gain: int | None = None   # total uncovered free cells visible from this position
    geodesic_dist_px: float | None = None   # navigable-path (Dijkstra) distance from robot (pixels)
    score: float | None = None         # composite greedy score at selection time


#--------------------------------------------------------------------------
# Map construction: two entry points sharing one internal builder
#--------------------------------------------------------------------------

def load_map(
    pgm_path: str | Path,
    yaml_path: str | Path,
    inflation_radius_m: float,
    covered_mask: np.ndarray | None = None,
) -> MapData:
    """Load an occupancy grid from a PGM + YAML file pair.

    Used for offline planning and tests.
    For live ROS2 use, call build_map_data() after converting the OccupancyGrid message.
    """
    pgm_path = Path(pgm_path)
    yaml_path = Path(yaml_path)

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution: float = float(meta["resolution"])
    origin: list = meta["origin"]
    free_thresh: float = float(meta["free_thresh"])
    occ_thresh: float = float(meta["occupied_thresh"])
    negate: int = int(meta.get("negate", 0))

    arr = np.array(Image.open(pgm_path), dtype=np.uint8)  # (H, W)
    pixel_f = arr.astype(np.float64)
    p_occ = pixel_f / 255.0 if negate else 1.0 - pixel_f / 255.0

    return build_map_data(
        p_occ=p_occ,
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        inflation_radius_m=inflation_radius_m,
        free_thresh=free_thresh,
        occ_thresh=occ_thresh,
        covered_mask=covered_mask,
        pgm_array=arr,
    )


def build_map_data(
    p_occ: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    inflation_radius_m: float,
    free_thresh: float = 0.196,
    occ_thresh: float = 0.65,
    covered_mask: np.ndarray | None = None,
    pgm_array: np.ndarray | None = None,
) -> MapData:
    """Build MapData from a pre-computed occupancy probability array.

    Called by the ROS2 node after decoding a nav_msgs/OccupancyGrid message.
    The node is responsible for the conversion.

    ROS2 OccupancyGrid convention:
        data values: 0 = free, 100 = occupied, -1 = unknown
        layout:      row-major, row 0 = bottom of world (origin)

    Typical conversion in ros_exploration_node.py:
        H = grid_msg.info.height
        W = grid_msg.info.width
        data = np.array(grid_msg.data, dtype=np.float64).reshape(H, W)
        # Flip rows: ROS row 0 = world bottom; image row 0 = world top
        data = np.flipud(data)
        p_occ = np.where(data < 0, 0.5, data / 100.0)   # -1 (unknown) → 0.5
        map_data = build_map_data(
            p_occ,
            resolution=grid_msg.info.resolution,
            origin_x=grid_msg.info.origin.position.x,
            origin_y=grid_msg.info.origin.position.y,
            inflation_radius_m,
        )

    Args:
        p_occ:    Float array (H, W) with values in [0, 1].
                  0 = certainly free, 1 = certainly occupied, 0.5 = unknown.
        free_thresh, occ_thresh: match system_parameters.yaml values.
        pgm_array: Optional raw uint8 (H, W) image. Passed by load_map() to
                   preserve the original PGM pixels for visualisation; when
                   omitted (live ROS2 path) it is derived from p_occ.
    """
    if pgm_array is None:
        pgm_array = (p_occ * 255).astype(np.uint8)

    free_mask = p_occ < free_thresh
    occupied_mask = p_occ > occ_thresh
    unknown_mask = ~free_mask & ~occupied_mask

    inflation_px = inflation_radius_m / resolution
    navigable_mask = free_mask & (distance_transform_edt(~occupied_mask) > inflation_px)

    if covered_mask is None:
        covered_mask = np.zeros_like(free_mask, dtype=bool)

    return MapData(
        pgm_array=pgm_array,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        free_mask=free_mask,
        occupied_mask=occupied_mask,
        unknown_mask=unknown_mask,
        navigable_mask=navigable_mask,
        covered_mask=covered_mask,
        inflation_px=inflation_px,
    )


def reproject_covered_mask(
    old_covered: np.ndarray,
    old_resolution: float,
    old_origin_x: float,
    old_origin_y: float,
    new_shape: tuple[int, int],
    new_resolution: float,
    new_origin_x: float,
    new_origin_y: float,
) -> np.ndarray:
    """Reproject a covered_mask into a new grid geometry (SLAM map growth).

    SLAM rebuilds the map with a new size/origin as it discovers space; the
    accumulated coverage must be carried over by world position, never dropped.
    Both the autonomous node and the manual baseline node use THIS function so
    the behaviour cannot drift between them.

    If the geometry is unchanged the input mask is returned as-is. Vectorised
    world-anchor round-trip (same math as pixel_to_world/world_to_pixel):
        wx = old_ox + col*old_res;   new_col = floor((wx - new_ox)/new_res + .5)
        wy = old_oy + (old_H-1-row)*old_res;
        new_row = new_H-1 - floor((wy - new_oy)/new_res + .5)
    """
    old_H, old_W = old_covered.shape
    new_H, new_W = new_shape
    if (old_H, old_W) == (new_H, new_W) \
            and old_origin_x == new_origin_x and old_origin_y == new_origin_y \
            and old_resolution == new_resolution:
        return old_covered

    old_rows, old_cols = np.where(old_covered)
    wx = old_origin_x + old_cols * old_resolution
    wy = old_origin_y + (old_H - 1 - old_rows) * old_resolution
    new_c = np.floor((wx - new_origin_x) / new_resolution + 0.5).astype(int)
    new_r = new_H - 1 - np.floor(
        (wy - new_origin_y) / new_resolution + 0.5).astype(int)
    inb = (new_c >= 0) & (new_c < new_W) & (new_r >= 0) & (new_r < new_H)
    covered = np.zeros((new_H, new_W), dtype=bool)
    covered[new_r[inb], new_c[inb]] = True
    return covered


#--------------------------------------------------------------------------
# Coordinate helpers
#--------------------------------------------------------------------------

def pixel_to_world(
    col: int,
    row: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    height: int,
) -> tuple[float, float]:
    """Convert pixel (col, row) to world (x, y) in metres.

    ROS convention: row 0 = top of image = maximum world Y.
        x = origin_x + col * resolution
        y = origin_y + (H - 1 - row) * resolution
    """
    return (
        origin_x + col * resolution,
        origin_y + (height - 1 - row) * resolution,
    )


def world_to_pixel(
    x: float,
    y: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
    height: int,
) -> tuple[int, int]:
    """Convert world (x, y) to pixel (col, row). Inverse of pixel_to_world."""
    return (
        int((x - origin_x) / resolution),
        int(height - 1 - (y - origin_y) / resolution),
    )


#--------------------------------------------------------------------------
# Candidate generation
#--------------------------------------------------------------------------

# Minimum physical area (m²) a navigable region must have to earn a representative
# candidate.  Regions smaller than this are inflation slivers / single-pixel artifacts.
# 0.25 m² is the value the historical 100-cell threshold encoded at 0.05 m/px
# (100 x 0.05² = 0.25); expressing it as an area makes it resolution-independent.
MIN_COMPONENT_AREA_M2 = 0.25


def generate_candidates(
    navigable_mask: np.ndarray,
    sampling_step_px: int,
    resolution: float | None = None,
) -> list[tuple[int, int]]:
    """Sample candidate viewpoints on a regular grid over navigable free space.

    After the regular grid pass, each connected navigable region that has no
    grid candidate receives one representative (the pixel deepest inside the
    region — farthest from any wall). This guarantees coverage of narrow aisles
    or rooms that are missed when the grid step is larger than the region's extent.

    Args:
        navigable_mask:   Boolean (H, W) grid of valid standing positions.
        sampling_step_px: Grid spacing in pixels for the regular pass.
        resolution:       Metres per pixel. When given, the minimum-region-size
                          guard is derived from MIN_COMPONENT_AREA_M2 so it means
                          the same physical area at any map resolution. When None,
                          the historical fixed 100-cell threshold is used.
    """
    H, W = navigable_mask.shape
    step = max(1, sampling_step_px)
    grid = {
        (col, row)
        for row in range(0, H, step)
        for col in range(0, W, step)
        if navigable_mask[row, col]
    }

    # One representative per significant connected navigable region not hit by the grid. 
    # Single-pixel artifacts and inflation-edge slivers are skipped via a minimum size guard: a fixed physical area (resolution-aware) when resolution is known, else the 100-cell fallback (good for 0.05m/px resolution).
    if resolution is not None:
        MIN_COMPONENT_CELLS = max(1, round(MIN_COMPONENT_AREA_M2 / (resolution ** 2)))
    else:
        MIN_COMPONENT_CELLS = 100
    labeled, n_components = ndlabel(navigable_mask)
    # Component sizes in a single pass (np.bincount) rather than one full-array
    # np.sum per label.  index 0 = background; reused by pass 3 to filter slivers.
    comp_size = np.bincount(labeled.ravel(), minlength=n_components + 1)

    # Distance-to-wall for every navigable cell.  Computed once here and reused by
    # pass 3, so the transform runs a single time for the whole function.
    dist_to_wall = distance_transform_edt(navigable_mask)

    # Group all foreground pixels by component in one pass (stable sort preserves
    # row-major order within each label) so each region's pixels can be sliced out
    # without re-scanning the whole label array per component.
    fg_rows, fg_cols = np.where(labeled > 0)
    fg_labels = labeled[fg_rows, fg_cols]
    sort_idx = np.argsort(fg_labels, kind="stable")
    fg_rows, fg_cols, fg_labels = fg_rows[sort_idx], fg_cols[sort_idx], fg_labels[sort_idx]
    comp_starts = np.searchsorted(fg_labels, np.arange(1, n_components + 1), side="left")
    comp_ends = np.searchsorted(fg_labels, np.arange(1, n_components + 1), side="right")

    for idx, comp_id in enumerate(range(1, n_components + 1)):
        if comp_size[comp_id] < MIN_COMPONENT_CELLS:
            continue  # artifact / inflation sliver, too small to need a waypoint
        if any(labeled[row, col] == comp_id for col, row in grid):
            continue  # at least one grid candidate already covers this region
        s, e = comp_starts[idx], comp_ends[idx]
        rows_nav, cols_nav = fg_rows[s:e], fg_cols[s:e]
        # Representative = the pixel deepest inside the region (max distance to any
        # wall).  This is wall-safe by construction — unlike a row-major-median or
        # centroid pixel it can never land on an edge of a concave region (L / ring).
        deepest = int(np.argmax(dist_to_wall[rows_nav, cols_nav]))
        grid.add((int(cols_nav[deepest]), int(rows_nav[deepest])))

    # ── Pass 3: medial-axis (distance-transform local maxima) ──────────────
    # Local maxima of the distance-to-wall field form the medial axis —
    # points equidistant from all surrounding walls, landing naturally in
    # aisle centres, room centres, and hallway midpoints regardless of map
    # layout.  Two filter scales are unioned:
    #   - coarse (step//4): strong ridges in narrow corridors
    #   - fine   (step//8): weaker ridges in open areas / wide rooms
    # Spatial grid sampling (one best point per cell_size tile) ensures
    # uniform coverage without row-major ordering bias.
    # Small isolated components (< MIN_COMPONENT_CELLS) are excluded using
    # the same guard as pass 2, so slivers and tiny islands are never added.
    skel_coarse = (dist_to_wall == maximum_filter(dist_to_wall, size=max(3, step // 4)))
    skel_fine   = (dist_to_wall == maximum_filter(dist_to_wall, size=max(3, step // 8)))
    skel_mask = (skel_coarse | skel_fine) & navigable_mask
    cell_size = max(1, step // 2)
    min_spacing = max(1, cell_size // 2)   # minimum pixel gap between skeleton candidates
    grid_list = list(grid)                 # snapshot for proximity checks (grows as we add)
    for r0 in range(0, H, cell_size):
        for c0 in range(0, W, cell_size):
            r1, c1 = min(H, r0 + cell_size), min(W, c0 + cell_size)
            patch = skel_mask[r0:r1, c0:c1]
            if not patch.any():
                continue
            patch_dist = np.where(patch, dist_to_wall[r0:r1, c0:c1], -1.0)
            best = int(np.argmax(patch_dist))
            br, bc = divmod(best, c1 - c0)
            bc_abs, br_abs = c0 + bc, r0 + br
            # Skip if the best point belongs to a small isolated component.
            # comp_size is indexed by label; label 0 (background) has its own count
            # but a skeleton point is always navigable, so its label is >= 1.
            if comp_size[int(labeled[br_abs, bc_abs])] < MIN_COMPONENT_CELLS:
                continue
            # Skip if already too close to any existing candidate, prevents adjacent
            # tiles from producing nearly-duplicate points that violate inflation radius
            too_close = any(
                (bc_abs - gc) ** 2 + (br_abs - gr) ** 2 < min_spacing ** 2
                for gc, gr in grid_list
            )
            if too_close:
                continue
            grid.add((bc_abs, br_abs))
            grid_list.append((bc_abs, br_abs))

    return list(grid)


#--------------------------------------------------------------------------
# BFS distance map : navigable-path distances through the occupancy grid
#--------------------------------------------------------------------------

# 8-connected grid moves with true Euclidean step cost: cardinal = 1, diagonal = √2.
# Using √2 (not a uniform 1) makes the returned distances a true geodesic navigation
# cost, which greedy_set_cover's travel penalty and nearest_neighbor_order's TSP consume as a *magnitude*, a uniform cost under-measures diagonal travel by up to root(2)x and biases
# both which waypoints are chosen and in what order (see tests/algo_evaluation/bfs_metric_impact.ipynb for the measured impact).
_SQRT2 = math.sqrt(2.0)
_GRID_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2), (0, -1, 1.0),
    (0, 1, 1.0), (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2),
)

# Cache the sparse grid-graph + per-cell node index per navigable mask, so repeated
# source queries against the same map are pure C-level scipy Dijkstra calls (the graph
# build is the expensive part and is mask-, not source-, dependent). Keyed by a *content*
# signature rather than id(): CPython recycles id() for garbage-collected arrays, which
# would hand back a stale graph for a different mask. Bounded so a long exploration session
# (the mask changes as the map fills in) can't grow the cache without limit.
_GRAPH_CACHE_MAX = 8
_graph_cache: dict[tuple, tuple[csr_matrix, np.ndarray]] = {}


def _navigable_graph(navigable_mask: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Build (sparse graph over navigable cells, (H,W) int node-index map) for a mask.

    idx[r, c] is the graph node id of navigable cell (r, c), or -1 if non-navigable.
    Edges are added vectorised per neighbour offset (cardinal=1, diagonal=√2).
    """
    H, W = navigable_mask.shape
    idx = -np.ones((H, W), dtype=np.intp)
    ys, xs = np.where(navigable_mask)
    idx[ys, xs] = np.arange(len(ys))
    rows_i: list[np.ndarray] = []
    cols_j: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for dr, dc, w in _GRID_NEIGHBOURS:
        r0, r1 = max(0, -dr), H - max(0, dr)
        c0, c1 = max(0, -dc), W - max(0, dc)
        both = navigable_mask[r0:r1, c0:c1] & navigable_mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        si = idx[r0:r1, c0:c1][both]
        di = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both]
        rows_i.append(si)
        cols_j.append(di)
        vals.append(np.full(si.size, w))
    n = len(ys)
    graph = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_j))),
        shape=(n, n),
    )
    return graph, idx


# Extra margin (pixels) added to inflation_px when bounding the start-snap, to
# absorb rasterisation and the strict ">" in the navigable_mask threshold. The
# snap ceiling itself tracks Nav2's inflation radius (MapData.inflation_px): a
# non-navigable free cell is at most ~inflation_px from a navigable cell by
# construction, so inflation_px + this margin admits the legitimate inflation-band
# case while still rejecting a robot genuinely boxed into a non-navigable island.
_SNAP_MARGIN_PX = 2.0


def _snap_to_navigable(
    navigable_mask: np.ndarray,
    start_col: int,
    start_row: int,
    ceiling_px: float,
) -> tuple[int, int] | None:
    """Nearest navigable cell to (start_col, start_row) within ceiling_px.

    Returns (col, row) of the closest navigable cell in Euclidean pixel distance,
    or None if none exists within the ceiling (caller should treat as unreachable).
    Searches a bounded square window so cost stays O(ceiling²) regardless of map size.
    """
    H, W = navigable_mask.shape
    r = int(math.ceil(ceiling_px))
    r0, r1 = max(0, start_row - r), min(H, start_row + r + 1)
    c0, c1 = max(0, start_col - r), min(W, start_col + r + 1)
    window = navigable_mask[r0:r1, c0:c1]
    ys, xs = np.where(window)
    if ys.size == 0:
        return None
    ys = ys + r0
    xs = xs + c0
    d2 = (ys - start_row) ** 2 + (xs - start_col) ** 2
    best = int(np.argmin(d2))
    if d2[best] > ceiling_px * ceiling_px:
        return None
    return int(xs[best]), int(ys[best])


def navigable_distance_map(
    navigable_mask: np.ndarray,
    start_col: int,
    start_row: int,
    inflation_px: float = 0.0,
) -> np.ndarray:
    """Geodesic distance from (start_col, start_row) over navigable cells.

    Returns a float array (H, W) of grid distances in pixels to each navigable
    cell, np.inf where unreachable or not navigable.

    8-directional movement with true Euclidean step cost (cardinal = 1,
    diagonal = √2), computed as a weighted grid Dijkstra
    (`scipy.sparse.csgraph.dijkstra`). The graph is cached per mask so repeated
    source queries on the same map are fast; this is ~15-20x faster than a pure-Python BFS on the real asset maps.

    Unlike straight-line (Euclidean) distance, this reflects the actual
    navigation cost through the map topology: cells on the other side of a wall
    have a large distance even if they are geometrically close.

    If the start cell is not navigable (robot inside Nav2's inflation band), the
    start is snapped to the nearest navigable cell within inflation_px + margin
    (pass map_data.inflation_px). Beyond that the robot is treated as boxed in and
    an all-inf map is returned. inflation_px=0.0 (default) disables snapping, so
    callers that don't supply it keep the strict "start must be navigable" contract.
    """
    H, W = navigable_mask.shape
    dist = np.full((H, W), np.inf)
    if not (0 <= start_row < H and 0 <= start_col < W):
        return dist
    if not navigable_mask[start_row, start_col]:
        if inflation_px <= 0.0:
            return dist
        ceiling_px = inflation_px + _SNAP_MARGIN_PX
        snapped = _snap_to_navigable(navigable_mask, start_col, start_row, ceiling_px)
        if snapped is None:
            return dist
        start_col, start_row = snapped
    key = (navigable_mask.shape, hash(navigable_mask.tobytes()))
    cached = _graph_cache.get(key)
    if cached is None:
        cached = _navigable_graph(navigable_mask)
        if len(_graph_cache) >= _GRAPH_CACHE_MAX:
            # simple bounded cache: drop the oldest entry (insertion order)
            _graph_cache.pop(next(iter(_graph_cache)))
        _graph_cache[key] = cached
    graph, idx = cached
    dmap = _sp_dijkstra(graph, indices=int(idx[start_row, start_col]))
    ys, xs = np.where(navigable_mask)
    dist[ys, xs] = dmap
    return dist


#--------------------------------------------------------------------------
# Ray-cast primitives (shared by compute_visibility / compute_achievable_cells /
# update_covered_mask). Vectorised over (angle, radius); per-cell semantics stay in the callers.
#--------------------------------------------------------------------------

def _ray_directions(num_rays: int) -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin) per full-circle ray, computed with math.* per angle.
    """
    cos_a = np.array([math.cos(2.0 * math.pi * i / num_rays) for i in range(num_rays)])
    sin_a = np.array([math.sin(2.0 * math.pi * i / num_rays) for i in range(num_rays)])
    return cos_a, sin_a


def _ray_cells(
    col0: int, row0: int, cos_a: np.ndarray, sin_a: np.ndarray, max_range_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer (n_angles, max_range) col/row grids, identical rounding to the scalar loop:
    col = floor(col0 + r·cos + 0.5), row = floor(row0 - r·sin + 0.5)  (-sin: row↓, world y↑)."""
    r = np.arange(1, max_range_px + 1)
    col = np.floor(col0 + np.outer(cos_a, r) + 0.5).astype(np.intp)
    row = np.floor(row0 - np.outer(sin_a, r) + 0.5).astype(np.intp)
    return col, row


def _ray_first_block(
    col: np.ndarray, row: np.ndarray,
    occupied_mask: np.ndarray, unknown_mask: np.ndarray, H: int, W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given (n_angles, max_range) col/row grids, return:
        alive         : cells strictly before the first stop on their ray (free, reachable)
        frontier_here : the unknown cell that stops each ray (first stop AND unknown)
        rr, cc        : clipped row/col for safe mask indexing by the caller
    Stop condition per cell = out-of-bounds OR occupied OR unknown (same as the scalar loops)."""
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    cc = np.clip(col, 0, W - 1)
    rr = np.clip(row, 0, H - 1)
    occ = inb & occupied_mask[rr, cc]
    unk = inb & unknown_mask[rr, cc]
    stop = (~inb) | occ | unk
    cum = np.cumsum(stop, axis=1)
    alive = ((cum - stop) == 0) & (~stop)   # no stop strictly before, and not a stop itself
    frontier_here = (cum == 1) & stop & unk  # first stop on the ray, and it is unknown
    return alive, frontier_here, rr, cc


#--------------------------------------------------------------------------
# Visibility: planning abstraction (360° rotation potential, not camera FOV)
#--------------------------------------------------------------------------

def compute_visibility(
    candidate: tuple[int, int],
    map_data: MapData,
    max_range_px: int,
    num_rays: int = 360,
    ignore_covered: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """360° planning scan from candidate. Returns (coverage_cells, frontier_cells).

    Planning abstraction: represents the full rotation potential of the robot at this position: NOT a single camera shot. Used to rank candidate waypoints.
    The actual camera observation is in update_covered_mask().
    coverage_cells: free AND not-yet-covered cells reachable from this position.
    frontier_cells: first unknown cell per ray (map discovery gain for SLAM).

    Ray stopping rules per step:
        out of bounds → stop
        occupied      → stop (wall)
        unknown   → add to frontier_cells, stop (opaque; potential frontier)
        free      → add to coverage_cells if not already covered; continue

    ignore_covered=True returns the full geometric footprint covered_mask not subtracted), the denominator for observed-fraction tests, without the caller having to mutate map_data.covered_mask.
    """
    col0, row0 = candidate
    H, W = map_data.occupied_mask.shape

    cos_a, sin_a = _ray_directions(num_rays)
    col, row = _ray_cells(col0, row0, cos_a, sin_a, max_range_px)
    alive, frontier_here, rr, cc = _ray_first_block(
        col, row, map_data.occupied_mask, map_data.unknown_mask, H, W)

    # coverage: reachable free cells that are not already covered
    cov_sel = alive if ignore_covered else alive & (~map_data.covered_mask[rr, cc])
    coverage = set(zip(col[cov_sel].tolist(), row[cov_sel].tolist()))
    frontiers = set(zip(col[frontier_here].tolist(), row[frontier_here].tolist()))
    return coverage, frontiers


def compute_all_visibility(
    candidates: list[tuple[int, int]],
    map_data: MapData,
    max_range_m: float,
    num_rays: int = 360,
) -> dict[tuple[int, int], tuple[set, set]]:
    """Compute (coverage_cells, frontier_cells) for every candidate."""
    max_range_px = max(1, int(max_range_m / map_data.resolution))
    return {c: compute_visibility(c, map_data, max_range_px, num_rays) for c in candidates}


def compute_achievable_cells(
    candidates: list[tuple[int, int]],
    map_data: MapData,
    max_range_m: float,
    num_rays: int = 360,
) -> set[tuple[int, int]]:
    """Return all free cells reachable from any candidate, regardless of covered_mask.

    This is the stable denominator for coverage_ratio. Unlike compute_all_visibility, it does NOT filter out already-covered cells so the denominator stays constant across re-plans as more cells are covered.
    Used in plan_waypoints to compute the true coverage ratio:
    ratio = |covered_mask ∩ achievable_cells| / |achievable_cells|
    """
    max_range_px = max(1, int(max_range_m / map_data.resolution))
    H, W = map_data.occupied_mask.shape
    cos_a, sin_a = _ray_directions(num_rays)
    achievable: set[tuple[int, int]] = set()

    for col0, row0 in candidates:
        col, row = _ray_cells(col0, row0, cos_a, sin_a, max_range_px)
        alive, _, _, _ = _ray_first_block(
            col, row, map_data.occupied_mask, map_data.unknown_mask, H, W)
        # Reachable free cells, added regardless of covered state.
        achievable.update(zip(col[alive].tolist(), row[alive].tolist()))

    return achievable


#---------------------------------------------------------------------------
# Greedy set cover
#---------------------------------------------------------------------------

def greedy_set_cover(
    visibility: dict[tuple[int, int], tuple[set, set]],
    alpha: float = 1.0,
    beta: float = 1.0,
    dist_from_robot: dict[tuple[int, int], float] | None = None,
    gamma: float = 0.0,
    normaliser: float = 1.0,
    max_waypoints: int | None = None,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], float, list[dict]]:
    """Select waypoints greedily, maximising utility-density score.

    Score formula (exponential distance decay):
        score(c) = (alpha x frontier_gain + beta x coverage_gain)
                   x exp(-gamma x geodesic_dist / normaliser)

    The exponential factor sharply de-weights far candidates so greedy prefers a tight, well-clustered set the nearest-neighbour router turns into a short tour.
    see tests/algo_evaluation/benchmark_waypoint_scoring_out/FINDINGS_waypoint_scoring.md) explaining the rational of this choice.
    # gamma is the decay rate in max-range units:
    the factor is e^-gamma at one max-range (x0.37 for gamma=1.0). When
    dist_from_robot is None or gamma == 0.0 the factor is 1 and the formula
    reduces to pure-gain scoring.

    Args:
        visibility:       {candidate: (coverage_cells, frontier_cells)} dict.
        alpha:            Weight for map-discovery (frontier) gain.
        beta:             Weight for visual-coverage gain.
        dist_from_robot:  BFS distance in pixels from the robot to each candidate.
                          Pass None to disable travel-cost weighting.
        gamma:            Travel-cost penalty weight (dimensionless when normaliser
                          = max_range_px).  0.0 → pure gain scoring.
        normaliser:       Scale for the distance term; use max_range_px so that
                          gamma is resolution-independent.
        max_waypoints:    Hard cap on the number of selected waypoints.  The
                          greedy loop stops after selecting this many candidates,
                          even if remaining candidates still have positive gain.
                          None (default) means no cap — select until gain = 0.

    Returns:
        selected:           ordered list of selected (col, row) candidates.
        achievable_cells:   union of all coverage_cells across all candidates.
        Denominator for coverage_ratio: excludes cells behind walls that no candidate can geometrically reach.
        coverage_ratio:     fraction of achievable_cells covered by selected waypoints.
        candidate_records:  list of dicts (one per candidate) with scoring data.
                            Fields: col, row, frontier_gain, coverage_gain,                        geodesic_dist_px, score, selection_rank (int rank if selected, else None).
                            Gains are total visibility (not marginal).
                            Scores use the final state of remaining_coverage after
                            all selections, so rejected candidates reflect their
                            residual value. Useful for offline gamma tuning.
    """
    if not visibility:
        # No candidates → nothing to cover; empty selection, full ratio by convention.
        return [], set(), 1.0, []

    achievable_cells: set[tuple[int, int]] = set().union(
        *(cov for cov, _ in visibility.values())
    )
    remaining_coverage = set(achievable_cells)
    remaining_frontiers: set[tuple[int, int]] = set().union(
        *(front for _, front in visibility.values())
    )

    selected: list[tuple[int, int]] = []
    # Exclude candidates unreachable from the robot, BFS dist == inf means
    # no navigable path exists; selecting them only causes teleport / skip.
    if dist_from_robot is not None:
        remaining_candidates = [c for c in visibility if dist_from_robot.get(c, np.inf) < np.inf]
    else:
        remaining_candidates = list(visibility.keys())

    _use_dist = dist_from_robot is not None and gamma > 0.0 and normaliser > 0.0

    def _score(c: tuple[int, int]) -> float:
        gain = (alpha * len(visibility[c][1] & remaining_frontiers) #frontiers gain
                + beta * len(visibility[c][0] & remaining_coverage)) #coverage gain
        if not _use_dist:
            return gain
        dist = dist_from_robot.get(c, 0.0)  # type: ignore[union-attr]
        if dist == np.inf:
            dist = normaliser * 1e6  # unreachable candidate: huge penalty
        return gain * math.exp(-gamma * dist / normaliser)

    while remaining_candidates:
        if max_waypoints is not None and len(selected) >= max_waypoints:
            break  # hard cap reached
        best = max(remaining_candidates, key=_score)
        if (
            not (visibility[best][0] & remaining_coverage)
            and not (visibility[best][1] & remaining_frontiers)
        ):
            break  # no further gain possible

        selected.append(best)
        remaining_coverage -= visibility[best][0]
        remaining_frontiers -= visibility[best][1]
        remaining_candidates.remove(best)

    covered_fraction = (
        1.0 - len(remaining_coverage) / len(achievable_cells)
        if achievable_cells else 1.0
    )

    def _record(c: tuple[int, int], rank: int | None) -> dict:
        # selection_rank is the single selected/rejected signal: an int >=1 for selected
        return {
            "col":            c[0],
            "row":            c[1],
            "frontier_gain":  len(visibility[c][1]),
            "coverage_gain":  len(visibility[c][0]),
            "geodesic_dist_px":    dist_from_robot.get(c) if dist_from_robot else None,
            "score":          _score(c),
            "selection_rank": rank,
        }

    # Selected candidates first (in selection order, rank starts at 1), then the rest.
    selected_set = set(selected)
    candidate_records: list[dict] = [_record(c, rank) for rank, c in enumerate(selected, start=1)]
    candidate_records += [_record(c, None) for c in visibility if c not in selected_set]

    return selected, achievable_cells, covered_fraction, candidate_records


#---------------------------------------------------------------------------
# Heading computation: uses real camera FOV
#---------------------------------------------------------------------------

def compute_headings_for_waypoint(
    goal_col: int,
    goal_row: int,
    uncovered_cells: set[tuple[int, int]],
    fov_deg: float = 87.0,
    increment_deg: float = 30.0,
) -> list[float]:
    """Minimum set of camera headings to observe all uncovered visible cells.

    Uses the real camera FOV and the configured rotation
    increment. Greedy: iteratively picks the heading (multiple of increment_deg) covering the most remaining unobserved bearings.

    Returns:
        Heading angles in degrees (0=East, 90=North, CCW positive).
        Empty list if no uncovered cells → no rotation needed at this waypoint.
    """
    if not uncovered_cells:
        return []

    bearings: list[float] = [
        math.degrees(math.atan2(-(row - goal_row), col - goal_col)) % 360.0
        for col, row in uncovered_cells
        # -row because image row increases downward, world y increases upward
    ]

    half_fov = fov_deg / 2.0
    heading_options = [
        i * increment_deg for i in range(int(round(360.0 / increment_deg)))
    ]

    remaining = list(range(len(bearings)))
    selected: list[float] = []

    while remaining:
        best = max(
            heading_options,
            key=lambda h: sum(
                1 for idx in remaining if angular_diff(bearings[idx], h) <= half_fov
            ),
        )
        newly_covered = [
            idx for idx in remaining if angular_diff(bearings[idx], best) <= half_fov
        ]
        if not newly_covered:
            break
        selected.append(best)
        covered_set = set(newly_covered)
        remaining = [i for i in remaining if i not in covered_set]

    return selected


def angular_diff(a: float, b: float) -> float:
    """Smallest unsigned angular difference between two angles (degrees)."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


#---------------------------------------------------------------------------
# Covered mask update: models real camera frustum
#---------------------------------------------------------------------------

def update_covered_mask(
    map_data: MapData,
    robot_col: int,
    robot_row: int,
    heading_deg: float,
    fov_deg: float,
    max_range_px: int,
    num_rays: int = 360,
) -> float:
    """Mark free cells seen within the camera/lidar frustum as covered.

    Casts rays at integer-degree intervals within the window:
        [heading - fov/2 - leniency, heading + fov/2 + leniency]

    Rays are cast at the same angular step as the planning scan (360°/num_rays),
    within the window [heading - fov/2 - leniency, heading + fov/2 + leniency],
    where leniency = step/2 (half a planning step).  This guarantees that any
    cell the planning scan can see at its discrete angle k·step° is also reached
    here, no aliasing regardless of sensor FOV or planning resolution.

        num_rays=180 → step=2°, leniency=1.0° (~45 rays over 87° FOV)
        num_rays=360 → step=1°, leniency=0.5° (~89 rays over 87° FOV)
        num_rays=720 → step=0.5°, leniency=0.25° (~175 rays over 87° FOV)

    Same stopping rules as compute_visibility.
    Updates map_data.covered_mask in-place.

    Args:
        heading_deg:  Sensor heading (degrees, 0=East, 90=North, CCW positive).
        fov_deg:      Horizontal FOV (87° for RealSense D435; any value for lidar).
        max_range_px: max_detection_range / resolution.
        num_rays:     Must match the num_rays used in compute_visibility / config
                      lidar.num_rays.  Determines leniency = 180° / num_rays.

    Returns:
        Ratio of covered free cells / total free cells after update.
    """
    H, W = map_data.occupied_mask.shape
    step_deg = 360.0 / num_rays          # matches compute_visibility ray spacing
    leniency_deg = step_deg / 2.0        # half a planning step catches boundary cells
    start_f = heading_deg - fov_deg / 2.0 - leniency_deg
    end_f = heading_deg + fov_deg / 2.0 + leniency_deg
    # Integer k-indices of planning rays that fall within the extended frustum.
    k_start = math.ceil(start_f / step_deg)
    k_end = math.floor(end_f / step_deg)

    # math trig per k so the discrete angles match compute_visibility bit-for-bit (see
    # _ray_directions on why np.cos/sin must not be used here). k can be negative when the frustum crosses 0° (e.g. heading 0°, wide FOV-> k in [-70, 70]); normalise the
    # angle to [0, 360) BEFORE the trig so it is the exact same float compute_visibility used for that ray. cos(-30°) equals cos(330°) mathematically but differs by 1 ulp in floating point, which after the floor(x+0.5) rasterisation can shift the ray by one pixel at long range, the camera would then miss cells the planner promised.
    cos_a = np.array([math.cos(math.radians((k * step_deg) % 360.0)) for k in range(k_start, k_end + 1)])
    sin_a = np.array([math.sin(math.radians((k * step_deg) % 360.0)) for k in range(k_start, k_end + 1)])
    col, row = _ray_cells(robot_col, robot_row, cos_a, sin_a, max_range_px)
    alive, _, rr, cc = _ray_first_block(
        col, row, map_data.occupied_mask, map_data.unknown_mask, H, W)
    write = alive & map_data.free_mask[rr, cc]
    map_data.covered_mask[row[write], col[write]] = True

    total_free = int(np.sum(map_data.free_mask))
    return 1.0 if total_free == 0 else float(np.sum(map_data.covered_mask & map_data.free_mask)) / total_free


#--------------------------------------------------------------------------
# Path ordering
#--------------------------------------------------------------------------

def nearest_neighbor_order(
    waypoints_px: list[tuple[int, int]],
    map_data: MapData,
    visibility: dict[tuple[int, int], tuple[set, set]],
    robot_x: float | None = None,
    robot_y: float | None = None,
) -> list[Waypoint]:
    """Order selected waypoints with geodesic nearest-neighbour TSP.

    Uses navigable-path (Dijkstra) distance instead of Euclidean distance so that
    waypoints on the far side of a wall are not mistakenly treated as "close".
    This prevents the robot from ping-ponging across corridors when Euclidean
    and actual navigation distance diverge.

    Algorithm:
        1. Compute the geodesic distance map from the robot's pixel position →
           pick the navigable-nearest first waypoint.
        2. For each subsequent step, compute it from the last selected
           waypoint → pick the navigable-nearest unvisited one.
        O(N x H x W) where N = number of selected waypoints.  With N ≤ 33 and
        H x W ≈ 182 K (lab_ghent) this is fast (~0.2 s total).

    Falls back gracefully: if the robot position is unknown or converts to a
    non-navigable cell, the distance map is all-inf and the function picks
    waypoints in list order for that step only.

    inputs are world metres (robot_x, robot_y); distances operate in pixel space
    internally, so the function is resolution-agnostic, the ordering is correct
    on any map resolution.
    """
    if not waypoints_px:
        return []

    H = map_data.pgm_array.shape[0]
    world_pts = [
        pixel_to_world(col, row, map_data.resolution, map_data.origin_x, map_data.origin_y, H)
        for col, row in waypoints_px
    ]

    def _nearest_by_geodesic(from_col: int, from_row: int,
                        candidates: list[int]) -> int:
        """Return index into waypoints_px of the geodesically-nearest candidate."""
        dist_map = navigable_distance_map(
            map_data.navigable_mask, from_col, from_row, map_data.inflation_px)
        return min(candidates,
                   key=lambda i: dist_map[waypoints_px[i][1], waypoints_px[i][0]])

    # Determine pixel start position for the first BFS
    if robot_x is not None and robot_y is not None:
        start_col, start_row = world_to_pixel(
            robot_x, robot_y, map_data.resolution,
            map_data.origin_x, map_data.origin_y, H,
        )
    else:
        # this else is dead in practice but kept for testing purposes.
        # No robot position: start from the bottom-left-most waypoint (world coords).
        # min over (x, y) tuples = lexicographic: lowest x, then lowest y (deterministic seed).
        fallback = min(range(len(world_pts)), key=lambda i: world_pts[i])
        start_col, start_row = waypoints_px[fallback]

    unvisited = list(range(len(waypoints_px)))
    first = _nearest_by_geodesic(start_col, start_row, unvisited)
    order = [first]
    unvisited.remove(first)

    for _ in range(len(waypoints_px) - 1):
        cur_col, cur_row = waypoints_px[order[-1]]
        nxt = _nearest_by_geodesic(cur_col, cur_row, unvisited)
        order.append(nxt)
        unvisited.remove(nxt)

    # Headings are intentionally left empty here: plan_waypoints fills them once with the
    # configured fov/increment. Computing them here too (with defaults) would just ben 
    # overwritten, and no other caller reads nearest_neighbor_order's headings.
    result = []
    for idx in order:
        col, row = waypoints_px[idx]
        x, y = world_pts[idx]
        result.append(Waypoint(x=x, y=y, col=col, row=row))

    return result


#--------------------------------------------------------------------------
# Coverage ratio
#--------------------------------------------------------------------------

def coverage_ratio(
    covered_mask: np.ndarray,
    achievable_cells: set[tuple[int, int]],
) -> float:
    """Fraction of achievable cells already observed by the camera.
    achievable_cells (from greedy_set_cover) excludes cells behind walls so the ratio is always geometrically achievable.
    """
    if not achievable_cells:
        return 1.0
    return sum(1 for col, row in achievable_cells if covered_mask[row, col]) / len(achievable_cells)


#--------------------------------------------------------------------------
# Top-level entry point
#--------------------------------------------------------------------------

def plan_waypoints(
    map_data: MapData,
    config: dict,
    robot_x: float | None = None,
    robot_y: float | None = None,
    visited_candidates: set[tuple[int, int]] | None = None,
) -> tuple[list[Waypoint], float, bool, list[dict]]:
    """
    Plan the next set of waypoints given current map and coverage state.

    config keys (from system_parameters.yaml):
        camera section:
            max_detection_range            (float, metres)
            fov_horizontal                 (float, degrees)
        exploration section:
            observation_rotation_increment (float, degrees)
            sampling_step_m                (float, metres)
            frontier_weight                (float, alpha)
            coverage_weight                (float, beta)
            exploration_completion_threshold (float)
            planner_coverage_warning_threshold (float)
        lidar section:
            num_rays                       (int, planning scan resolution)

    Args:
        map_data:  Current map state. covered_mask is read and reflects all prior
                   observations. The robot's past coverage is implicitly encoded here:
                   cells already in covered_mask have zero coverage_gain and are
                   naturally excluded from the new plan.
        config:    Flat dict of parameters (see above).
        robot_x, robot_y: Current robot world position (metres). Used to set the
                   TSP start so the first waypoint is always the nearest one.
                   Pass None to fall back to bottom-left ordering.
        visited_candidates: Set of (col, row) pixel positions to EXCLUDE from planning.
                   The caller (ExplorationSession) passes the UNION of two distinct facts
                   it deliberately stores separately:
                     - `visited_candidates`: genuinely OBSERVED viewpoints (the camera saw
                       them) — permanent and monotonic;
                     - `_unreachable`: viewpoints Nav2 repeatedly failed to drive to —
                       never observed, so the area is still uncovered. TRANSIENT: restored
                       by clear_unreachable() when the pool would otherwise be STARVED, because an abort verdict depends on the robot's pose and momentary costmap, not on the map itself. Restoring more eagerly (e.g. on
                       every arrival) livelocks on permanently unreachable waypoints.
                   Conflating the two (recording "could not reach" as "visited") makes the
                   planner believe unreached areas are covered and permanently freezes
                   exploration — do not merge them upstream.

    Returns:
        waypoints:     Ordered Waypoint list (empty when stop condition is met).
        current_ratio: coverage_ratio of covered_mask vs achievable_cells.
        no_frontiers:  True when the total distinct visible frontier cells (union
                       over candidates) is <= min_frontier_cells (0 = strictly none).

    Stop condition (check in the re-planning loop):
        if no_frontiers and current_ratio >= exploration_completion_threshold:
            exploration is complete
    """
    max_range_m: float = config.get("max_detection_range", 6.0)
    # NOTE: fov_horizontal / observation_rotation_increment are intentionally NOT read here —
    # headings are computed at arrival (see the headings note below), not at plan time.
    sampling_step_m: float = config.get("sampling_step_m", 3.0)
    num_rays: int = config.get("num_rays", 360)
    alpha: float = config.get("frontier_weight", 1.0)
    beta: float = config.get("coverage_weight", 1.0)
    gamma: float = config.get("travel_cost_weight", 1.0)
    max_waypoints: int | None = config.get("max_waypoints_per_plan", None)
    completion_threshold: float = config.get("exploration_completion_threshold", 0.90)
    warning_threshold: float = config.get("planner_coverage_warning_threshold", 0.90)
    # SLAM stop tolerance: exploration is "out of frontiers" once the TOTAL number of
    # distinct visible frontier cells (union over candidates) is <= this. 0 reproduces
    # the historical exact test (any candidate seeing >0 frontier cells blocks the stop);
    # a small positive value absorbs phantom/unreachable frontier slivers so the robot
    # does not chase the last wisp forever.
    min_frontier_cells: int = config.get("min_frontier_cells", 0)
    # Map source. Frontiers are UNKNOWN cells, which only a live SLAM map can ever resolve. A static known map keeps its unknown voids forever (wall interiors, outside the building), so frontier gain there is both unachievable and meaningless: it can never reach zero, which would make the completion gate unsatisfiable, and it would attract waypoints to voids that can never be observed. In known-map mode we therefore ignore frontiers entirely and explore on pure coverage (the behaviour this module's header always described but never enforced for real maps).
    is_slam: bool = bool(config.get("is_slam", True))

    sampling_step_px = max(1, int(sampling_step_m / map_data.resolution))
    max_range_px = max(1, int(max_range_m / map_data.resolution))

    all_candidates = generate_candidates(
        map_data.navigable_mask, sampling_step_px, map_data.resolution)

    # achievable_cells uses the FULL candidate set — its size must not change as visited positions are excluded, otherwise the coverage ratio denominator would shrink on each re-plan and produce a misleading (falling) ratio.
    achievable_cells = compute_achievable_cells(all_candidates, map_data, max_range_m, num_rays)

    total_free = int(np.sum(map_data.free_mask))
    if total_free > 0 and achievable_cells:
        geometric_ratio = len(achievable_cells) / total_free
        if geometric_ratio < warning_threshold:
            warnings.warn(
                f"Planner can geometrically reach only {geometric_ratio:.1%} of free cells "
                f"(threshold {warning_threshold:.1%}). Check for isolated map regions.",
                CoverageWarning,
                stacklevel=2,
            )

    # Greedy set cover operates only on unvisited candidates so the planner
    # never returns to a position that has already been observed (or failed).
    candidates = [c for c in all_candidates if c not in (visited_candidates or set())]
    if not candidates:
        # Pool starved (everything observed and/or currently unreachable). Report the TRUE frontier state, never a hardcoded True: claiming "no frontiers" here while frontiers actually remain hides the starvation from the caller, which then cannot tell "exploration finished" from "exploration is stuck", the exact ambiguity that let the node retry silently forever.
        if is_slam:
            vis_all = compute_all_visibility(all_candidates, map_data, max_range_m, num_rays)
            frontier_union_all: set = (
                set().union(*(vis_all[c][1] for c in all_candidates))
                if all_candidates else set()
            )
            starved_no_frontiers = len(frontier_union_all) <= min_frontier_cells
        else:
            # Known map: frontiers are ignored by definition, so completion depends on coverage alone and this flag must not block it.
            starved_no_frontiers = True
        return [], coverage_ratio(map_data.covered_mask, achievable_cells), starved_no_frontiers, []

    # Auto-derive waypoint cap when not set explicitly in config.
    # Physical basis: one viewpoint geometrically "serves" (max_range / step)² candidate-grid cells via its circular field of view.  Dividing the total candidate count by that overlap factor gives the minimum number of viewpoints that can tile the navigable area without redundancy. This scales naturally with map size and camera range, and adapts to unknown maps because |candidates| grows as exploration reveals new area.
    if max_waypoints is None:
        overlap_sq = max(1.0, (max_range_m / sampling_step_m) ** 2)
        max_waypoints = max(5, int(np.ceil(len(candidates) / overlap_sq)))

    # Geodesic distance from robot pixel position. Serves TWO independent purposes:
    #   1. reachability pruning: greedy_set_cover drops candidates with dist == inf (behind walls / disconnected), so they are never sent to Nav2 to fail;
    #   2. the gamma travel penalty on the score.
    # Build it whenever a robot pose is available, NOT gated on gamma. greedy_set_cover applies the gamma penalty separately (its _use_dist checks gamma > 0 on its own), so with gamma = 0 the map feeds only the reachability filter.
    dist_from_robot: dict[tuple[int, int], float] | None = None
    if robot_x is not None and robot_y is not None:
        H = map_data.navigable_mask.shape[0]
        r_col, r_row = world_to_pixel(
            robot_x, robot_y, map_data.resolution,
            map_data.origin_x, map_data.origin_y, H,
        )
        dist_map = navigable_distance_map(
            map_data.navigable_mask, r_col, r_row, map_data.inflation_px)
        dist_from_robot = {c: float(dist_map[c[1], c[0]]) for c in candidates}

    vis = compute_all_visibility(candidates, map_data, max_range_m, num_rays)
    # Known map: unknown cells are permanent voids, so frontier gain must not steer waypoint selection (alpha -> 0). SLAM: frontier gain is the map-discovery driver.
    effective_alpha = alpha if is_slam else 0.0
    selected_px, _, _, candidate_records = greedy_set_cover(
        vis, effective_alpha, beta,
        dist_from_robot=dist_from_robot,
        gamma=gamma,
        normaliser=float(max_range_px),
        max_waypoints=max_waypoints,
    )

    # Count DISTINCT visible frontier cells (union over candidates, so a cell seen by several viewpoints is not double-counted) and treat exploration as out of frontiers once that total is within tolerance. min_frontier_cells=0 → union must be empty → identical to the old `not any(len(vis[c][1]) > 0)` test.
    frontier_union: set = set().union(*(vis[c][1] for c in candidates)) if candidates else set()
    if is_slam:
        no_frontiers = len(frontier_union) <= min_frontier_cells
    else:
        # Known map: the unknown voids are unresolvable without a live /map, so the frontier test can never pass and would block completion forever. Report no_frontiers=True so the caller's `no_frontiers and ratio >= threshold` gate reduces to coverage alone — pure-coverage exploration, as intended.
        no_frontiers = True
    current_ratio = coverage_ratio(map_data.covered_mask, achievable_cells)

    if no_frontiers and current_ratio >= completion_threshold:
        return [], current_ratio, no_frontiers, candidate_records

    waypoints = nearest_neighbor_order(selected_px, map_data, vis, robot_x, robot_y)

    # Populate diagnostic fields on each waypoint from candidate_records
    records_by_pos = {(r["col"], r["row"]): r for r in candidate_records}
    for wp in waypoints:
        rec = records_by_pos.get((wp.col, wp.row), {})
        wp.frontier_gain = rec.get("frontier_gain")
        wp.coverage_gain = rec.get("coverage_gain")
        wp.geodesic_dist_px = rec.get("geodesic_dist_px")
        wp.score         = rec.get("score")

    return waypoints, current_ratio, no_frontiers, candidate_records
