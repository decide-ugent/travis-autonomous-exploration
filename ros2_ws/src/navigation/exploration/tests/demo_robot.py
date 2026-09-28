"""
Shared utilities for the visual exploration demo scripts.

Provides:
  - SceneObject dataclass: object label, position, per-label detection range
  - place_objects:         scatter N objects across the navigable map
  - find_path:             Breadth-First Search (BFS) shortest path on the navigable grid
  - check_detections:      determine which objects the camera sees at current pose
  - reveal_cells:          (SLAM demo) unveil unknown cells as the robot moves

No ROS2 imports; no matplotlib imports.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Load camera / exploration parameters from per-package config YAMLs
# ---------------------------------------------------------------------------

_PERCEPTION_CONFIG = Path(__file__).parent.parent.parent.parent / "perception" / "config" / "perception_system_parameters.yaml"
_EXPLORATION_CONFIG = Path(__file__).parent.parent / "config" / "exploration_system_parameters.yaml"
_NAV2_CONFIG = Path(__file__).parent.parent.parent / "nav2" / "config" / "nav2_mir_jazzy_params.yaml"


def _load_cfg() -> tuple[dict, dict]:
    """Load and return (perception_cfg, exploration_cfg) from per-package YAMLs.

    The exploration YAML is in ROS2 parameter format, so its values live under
    ``exploration_node.ros__parameters`` as flat dotted keys (e.g.
    ``"exploration.sampling_step_m"``). It is unwrapped to that inner dict here
    so callers index it directly by dotted key.
    """
    with open(_PERCEPTION_CONFIG) as f:
        perception = yaml.safe_load(f)
    with open(_EXPLORATION_CONFIG) as f:
        exploration = yaml.safe_load(f)["exploration_node"]["ros__parameters"]
    return perception, exploration


def _global_costmap_inflation_radius() -> float:
    """Read the global costmap inflation_radius (metres) from the nav2 config.

    The nav2 inflation parameter moved out of the exploration YAML into the
    dedicated nav2 costmap config; the demo tracks the active MiR250 params.
    """
    with open(_NAV2_CONFIG) as f:
        nav2 = yaml.safe_load(f)
    params = nav2["global_costmap"]["global_costmap"]["ros__parameters"]
    return float(params["inflation_layer"]["inflation_radius"])


# ---------------------------------------------------------------------------
# Demo config builder
# ---------------------------------------------------------------------------

def build_demo_config() -> dict:
    """Build the planner config dict from the per-package YAML files.

    All values are read verbatim — no overrides. Adjust the per-package YAML
    files to change planner behaviour.
    """
    p, e = _load_cfg()
    return {
        "max_detection_range":            p["camera"]["max_detection_range"],
        "fov_horizontal":                 p["camera"]["fov_horizontal"],
        "observation_rotation_increment": e["exploration.observation_rotation_increment"],
        "sampling_step_m":                e["exploration.sampling_step_m"],
        "num_rays":                       p["lidar"]["num_rays"],
        "frontier_weight":                e["exploration.frontier_weight"],
        "coverage_weight":                e["exploration.coverage_weight"],
        "exploration_completion_threshold":
            e["exploration.exploration_completion_threshold"],
        "planner_coverage_warning_threshold":
            e["exploration.planner_coverage_warning_threshold"],
    }


def _get_perception_constants() -> tuple[float, float, int, float]:
    """Return (MAX_DETECTION_M, FOV_HORIZONTAL, NUM_RAYS, INFLATION_M) from YAMLs."""
    p, _ = _load_cfg()
    return (
        float(p["camera"]["max_detection_range"]),
        float(p["camera"]["fov_horizontal"]),
        int(p["lidar"]["num_rays"]),
        _global_costmap_inflation_radius(),
    )


MAX_DETECTION_M, FOV_HORIZONTAL, NUM_RAYS, INFLATION_M = _get_perception_constants()
OBSERVATION_INCREMENT: float = build_demo_config()["observation_rotation_increment"]

# ---------------------------------------------------------------------------
# Per-label detection ranges and heights
# Calibrated to lab_ghent (32 m × 14 m), camera max range = MAX_DETECTION_M
# Smaller/closer objects require the robot to be nearby to detect them.
# ---------------------------------------------------------------------------

OBJECT_TYPES: dict[str, dict] = {
    "bottle":   {"detection_range_m": 1.5, "z": 0.5},
    "mug":      {"detection_range_m": 2.0, "z": 0.8},
    "laptop":   {"detection_range_m": 3.5, "z": 0.9},
    "plant":    {"detection_range_m": 4.0, "z": 1.2},
    "backpack": {"detection_range_m": 4.0, "z": 0.6},
    "chair":    {"detection_range_m": MAX_DETECTION_M, "z": 0.5},
}

# Fixed sequence of 10 labels (2 each of the 5 smallest, 2 chairs)
_DEFAULT_LABEL_SEQUENCE = [
    "bottle", "mug", "laptop", "plant", "backpack",
    "bottle", "mug", "laptop", "backpack", "chair",
]


# ---------------------------------------------------------------------------
# SceneObject
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    label: str
    col: int               # pixel column
    row: int               # pixel row
    x: float               # world metres
    y: float
    z: float               # height above floor (metres)
    detection_range_m: float
    detected: bool = False


# ---------------------------------------------------------------------------
# Object placement
# ---------------------------------------------------------------------------

def place_objects(
    navigable_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    n: int = 10,
    labels: list[str] | None = None,
    seed: int = 42,
) -> list[SceneObject]:
    """Randomly scatter N objects at distinct navigable positions.

    Args:
        navigable_mask: bool (H, W) — valid robot/object positions.
        resolution:     metres per pixel.
        origin_x/y:     world coords of bottom-left pixel.
        n:              number of objects.
        labels:         list of n label strings; defaults to _DEFAULT_LABEL_SEQUENCE[:n].
        seed:           random seed for reproducibility.

    Returns:
        List of SceneObject, one per object.
    """
    H, W = navigable_mask.shape
    rng = random.Random(seed)

    nav_cells = list(zip(*np.where(navigable_mask)))  # list of (row, col)
    if len(nav_cells) < n:
        n = len(nav_cells)

    chosen_indices = rng.sample(range(len(nav_cells)), n)
    chosen = [nav_cells[i] for i in chosen_indices]

    if labels is None:
        seq = (_DEFAULT_LABEL_SEQUENCE * math.ceil(n / len(_DEFAULT_LABEL_SEQUENCE)))[:n]
        # Shuffle to avoid always placing bottles in top-left corner
        seq = list(seq)
        rng.shuffle(seq)
        labels = seq

    objects: list[SceneObject] = []
    for k, (row, col) in enumerate(chosen):
        label = labels[k % len(labels)]
        props = OBJECT_TYPES.get(label, {"detection_range_m": MAX_DETECTION_M, "z": 0.5})
        x = origin_x + col * resolution
        y = origin_y + (H - 1 - row) * resolution
        objects.append(SceneObject(
            label=label,
            col=col,
            row=row,
            x=x,
            y=y,
            z=props["z"],
            detection_range_m=props["detection_range_m"],
        ))
    return objects


# ---------------------------------------------------------------------------
# Path-finding: Breadth-First Search
# ---------------------------------------------------------------------------

def find_path(
    navigable_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return the shortest pixel path from start to goal on navigable cells.

    Uses Breadth-First Search (BFS): explores the grid level by level,
    guaranteed to find the shortest path. 8-directional movement.

    Args:
        navigable_mask: bool (H, W).
        start:          (col, row) of start position.
        goal:           (col, row) of goal position.

    Returns:
        Ordered list of (col, row) positions from start to goal (inclusive).
        Returns [start] if no path exists (robot stays put).
    """
    if start == goal:
        return [start]

    H, W = navigable_mask.shape
    sc, sr = start
    gc, gr = goal

    # BFS
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),           (0,  1),
                  (1,  -1), (1,  0), (1,  1)]

    while queue:
        col, row = queue.popleft()
        if (col, row) == goal:
            break
        for dc, dr in neighbours:
            nc, nr = col + dc, row + dr
            if 0 <= nc < W and 0 <= nr < H and navigable_mask[nr, nc]:
                if (nc, nr) not in came_from:
                    came_from[(nc, nr)] = (col, row)
                    queue.append((nc, nr))

    if goal not in came_from:
        return [start]   # no path — robot stays put

    # Reconstruct path
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def _bearing_deg(robot_col: int, robot_row: int, obj_col: int, obj_row: int) -> float:
    """Bearing from robot to object in degrees (0=East, 90=North, CCW positive)."""
    dx = obj_col - robot_col
    dy = -(obj_row - robot_row)   # row↓ in image → y↑ in world
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _angular_diff(a: float, b: float) -> float:
    """Smallest unsigned angular difference between two headings (degrees)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _has_line_of_sight(
    occupied_mask: np.ndarray,
    unknown_mask: np.ndarray,
    robot_col: int,
    robot_row: int,
    obj_col: int,
    obj_row: int,
) -> bool:
    """March single ray from robot toward object; return True if unobstructed."""
    H, W = occupied_mask.shape
    dx = obj_col - robot_col
    dy = obj_row - robot_row
    dist = math.hypot(dx, dy)
    if dist == 0:
        return True
    cos_a = dx / dist
    sin_a = dy / dist
    steps = int(dist)
    for r in range(1, steps + 1):
        c = int(math.floor(robot_col + r * cos_a + 0.5))
        rr = int(math.floor(robot_row + r * sin_a + 0.5))
        if c < 0 or c >= W or rr < 0 or rr >= H:
            return False
        if occupied_mask[rr, c] or unknown_mask[rr, c]:
            return False
        if (c, rr) == (obj_col, obj_row):
            return True
    return True


def check_detections(
    occupied_mask: np.ndarray,
    unknown_mask: np.ndarray,
    resolution: float,
    robot_col: int,
    robot_row: int,
    heading_deg: float,
    fov_deg: float,
    objects: list[SceneObject],
) -> list[SceneObject]:
    """Check which undetected objects are within the camera frustum and line-of-sight.

    An object is detected when ALL of:
      1. It is not already detected
      2. Its bearing from the robot falls within [heading ± fov/2]
      3. Its pixel distance ≤ detection_range_m / resolution
      4. No wall or unknown cell blocks the line of sight

    Detected objects have their .detected flag set to True in-place.

    Returns:
        List of newly detected SceneObject instances.
    """
    newly: list[SceneObject] = []
    for obj in objects:
        if obj.detected:
            continue
        bearing = _bearing_deg(robot_col, robot_row, obj.col, obj.row)
        if _angular_diff(bearing, heading_deg) > fov_deg / 2.0:
            continue
        dist_px = math.hypot(obj.col - robot_col, obj.row - robot_row)
        if dist_px * resolution > obj.detection_range_m:
            continue
        if not _has_line_of_sight(occupied_mask, unknown_mask, robot_col, robot_row, obj.col, obj.row):
            continue
        obj.detected = True
        newly.append(obj)
    return newly


# ---------------------------------------------------------------------------
# SLAM: reveal unknown cells as robot moves
# ---------------------------------------------------------------------------

def reveal_cells(
    free_mask: np.ndarray,
    occupied_mask: np.ndarray,
    unknown_mask: np.ndarray,
    navigable_mask: np.ndarray,
    original_free: np.ndarray,
    original_occ: np.ndarray,
    robot_col: int,
    robot_row: int,
    reveal_range_px: int,
    num_rays: int = 360,
    inflation_px: float = 0.0,
) -> bool:
    """Reveal unknown cells within sensor range as the robot moves.

    Casts num_rays rays from the robot position. For each unknown cell
    reached by a ray, restores its true occupancy from the original masks
    and clears the unknown flag.

    Modifies free_mask, occupied_mask, unknown_mask, navigable_mask in-place.
    When inflation_px > 0, navigable_mask is recomputed with proper inflation
    after any new cells are revealed (one distance_transform_edt call per step).

    Args:
        original_free/occ:  copies of the masks before unknown cells were applied.
        reveal_range_px:    how far the robot can sense (pixels).
        inflation_px:       robot inflation radius in pixels; 0 skips recompute.

    Returns:
        True if any new cells were revealed this call.
    """
    from scipy.ndimage import distance_transform_edt
    H, W = unknown_mask.shape
    step_deg = 360.0 / num_rays
    newly_revealed = False
    for i in range(num_rays):
        angle = math.radians(i * step_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for r in range(1, reveal_range_px + 1):
            c = int(math.floor(robot_col + r * cos_a + 0.5))
            rr = int(math.floor(robot_row - r * sin_a + 0.5))
            if c < 0 or c >= W or rr < 0 or rr >= H:
                break
            if original_occ[rr, c]:
                occupied_mask[rr, c] = True
                free_mask[rr, c]     = False
                unknown_mask[rr, c]  = False
                break
            if unknown_mask[rr, c]:
                unknown_mask[rr, c]  = False
                free_mask[rr, c]     = original_free[rr, c]
                occupied_mask[rr, c] = original_occ[rr, c]
                newly_revealed = True
            if original_occ[rr, c]:
                break

    if newly_revealed:
        if inflation_px > 0.0:
            new_nav = free_mask & (distance_transform_edt(~occupied_mask) > inflation_px)
            navigable_mask[:] = new_nav
        else:
            # No inflation info: fall back to marking all newly free cells navigable
            navigable_mask[:] = free_mask & ~occupied_mask

    return newly_revealed
