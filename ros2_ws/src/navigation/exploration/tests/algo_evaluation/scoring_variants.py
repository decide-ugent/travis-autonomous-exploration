"""
Self-contained scoring variants for the waypoint-scoring benchmark.

Production (`exploration.explore_costmap_map`) uses a single hardcoded
exponential-decay travel penalty and carries NO model registry or knobs — so it
can evolve without dragging experimental branches. This module owns the
parameterized scoring path the benchmark needs to A/B-compare the alternatives
(linear baseline, gaussian shape, sequential "fix A", marginal-gain floor). It is
frozen here on purpose: it reproduces the ablation regardless of how production
changes later.

What lives here (the parts that differ per variant):
    DISTANCE_MODELS           — linear / expdecay / gaussian factor functions
    greedy_set_cover_variant  — greedy with distance_model / sequential / gain_floor
    plan_waypoints_variant    — plan_waypoints orchestration calling the above
    VariantSession            — ExplorationSession that plans via the variant path

Everything else (candidate generation, visibility, achievable-cell denominator,
BFS, nearest-neighbour ordering, coverage ratio, Waypoint) is imported from
production unchanged — those primitives are stable and shared.

The "expdecay, gamma=1, no sequential, no floor" configuration reproduces the
production scoring exactly, which the benchmark uses as a cross-check that this
copy has not drifted from prod.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from exploration.explore_costmap_map import (
    MapData,
    Waypoint,
    CoverageWarning,
    generate_candidates,
    compute_all_visibility,
    compute_achievable_cells,
    nearest_neighbor_order,
    navigable_distance_map,
    coverage_ratio,
    world_to_pixel,
)
from exploration.execution_strategy import ExplorationSession


# ---------------------------------------------------------------------------
# Distance-penalty models
# ---------------------------------------------------------------------------
# factor(dist_px, gamma, normaliser) -> float in (0, 1]; score = gain * factor.
#   "linear"   : 1 / (1 + gamma*d/norm)   — original/baseline. Slow decay.
#   "expdecay" : exp(-gamma*d/norm)        — production. Sharp decay.
#   "gaussian" : exp(-(gamma*d/norm)^2)    — flat near robot, hard far cutoff.

def _dist_factor_linear(dist: float, gamma: float, normaliser: float) -> float:
    return 1.0 / (1.0 + gamma * dist / normaliser)


def _dist_factor_expdecay(dist: float, gamma: float, normaliser: float) -> float:
    return math.exp(-gamma * dist / normaliser)


def _dist_factor_gaussian(dist: float, gamma: float, normaliser: float) -> float:
    x = gamma * dist / normaliser
    return math.exp(-x * x)


DISTANCE_MODELS = {
    "linear": _dist_factor_linear,
    "expdecay": _dist_factor_expdecay,
    "gaussian": _dist_factor_gaussian,
}


# ---------------------------------------------------------------------------
# Greedy set cover (parameterized)
# ---------------------------------------------------------------------------

def greedy_set_cover_variant(
    visibility: dict[tuple[int, int], tuple[set, set]],
    alpha: float = 1.0,
    beta: float = 1.0,
    dist_from_robot: dict[tuple[int, int], float] | None = None,
    gamma: float = 0.0,
    normaliser: float = 1.0,
    max_waypoints: int | None = None,
    distance_model: str = "expdecay",
    sequential: bool = False,
    dist_provider: "callable | None" = None,
    gain_floor: float = 0.0,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], float, list[dict]]:
    """Greedy waypoint selection, maximising utility-density score.

        score(c) = (alpha*frontier_gain + beta*coverage_gain)
                   * distance_factor(geodesic_dist, gamma, normaliser)

    distance_factor is chosen by distance_model (DISTANCE_MODELS). When
    dist_from_robot is None or gamma == 0 the factor is 1 (pure-gain scoring).

    sequential (fix A): re-seed distances from each pick via dist_provider so the
    penalty follows the growing tour (one BFS per pick).

    gain_floor: stop once a pick's marginal gain drops below gain_floor * the
    first pick's marginal gain (structural tail cap). 0.0 = off.

    Returns (selected, achievable_cells, covered_fraction, candidate_records) —
    same contract as production greedy_set_cover.
    """
    if not visibility:
        return [], set(), 1.0, []

    achievable_cells: set[tuple[int, int]] = set().union(
        *(cov for cov, _ in visibility.values())
    )
    remaining_coverage = set(achievable_cells)
    remaining_frontiers: set[tuple[int, int]] = set().union(
        *(front for _, front in visibility.values())
    )

    selected: list[tuple[int, int]] = []
    if dist_from_robot is not None:
        remaining_candidates = [c for c in visibility if dist_from_robot.get(c, np.inf) < np.inf]
    else:
        remaining_candidates = list(visibility.keys())

    _use_dist = dist_from_robot is not None and gamma > 0.0 and normaliser > 0.0
    _sequential = sequential and dist_provider is not None and gamma > 0.0 and normaliser > 0.0
    try:
        _factor = DISTANCE_MODELS[distance_model]
    except KeyError:
        raise ValueError(
            f"unknown distance_model {distance_model!r}; "
            f"choose from {sorted(DISTANCE_MODELS)}"
        )

    current_dist: list[dict[tuple[int, int], float] | None] = [dist_from_robot]

    def _marginal_gain(c: tuple[int, int]) -> float:
        return (alpha * len(visibility[c][1] & remaining_frontiers)
                + beta * len(visibility[c][0] & remaining_coverage))

    def _score(c: tuple[int, int]) -> float:
        gain = _marginal_gain(c)
        if not _use_dist:
            return gain
        dmap = current_dist[0]
        dist = dmap.get(c, 0.0) if dmap is not None else 0.0
        if dist == np.inf:
            dist = normaliser * 1e6  # unreachable candidate: huge penalty
        return gain * _factor(dist, gamma, normaliser)

    first_gain: float | None = None
    while remaining_candidates:
        if max_waypoints is not None and len(selected) >= max_waypoints:
            break
        best = max(remaining_candidates, key=_score)
        if (
            not (visibility[best][0] & remaining_coverage)
            and not (visibility[best][1] & remaining_frontiers)
        ):
            break

        if gain_floor > 0.0 and first_gain is not None:
            if _marginal_gain(best) < gain_floor * first_gain:
                break

        selected.append(best)
        if first_gain is None:
            first_gain = _marginal_gain(best)  # before removing its cells
        remaining_coverage -= visibility[best][0]
        remaining_frontiers -= visibility[best][1]
        remaining_candidates.remove(best)

        if _sequential and remaining_candidates:
            current_dist[0] = dist_provider(best[0], best[1])

    covered_fraction = (
        1.0 - len(remaining_coverage) / len(achievable_cells)
        if achievable_cells else 1.0
    )

    current_dist[0] = dist_from_robot  # stable reference for records

    def _record(c: tuple[int, int], rank: int | None) -> dict:
        return {
            "col":            c[0],
            "row":            c[1],
            "frontier_gain":  len(visibility[c][1]),
            "coverage_gain":  len(visibility[c][0]),
            "geodesic_dist_px":    dist_from_robot.get(c) if dist_from_robot else None,
            "score":          _score(c),
            "selection_rank": rank,
        }

    selected_set = set(selected)
    candidate_records: list[dict] = [_record(c, rank) for rank, c in enumerate(selected, start=1)]
    candidate_records += [_record(c, None) for c in visibility if c not in selected_set]

    return selected, achievable_cells, covered_fraction, candidate_records


# ---------------------------------------------------------------------------
# plan_waypoints (variant)
# ---------------------------------------------------------------------------

def plan_waypoints_variant(
    map_data: MapData,
    config: dict,
    robot_x: float | None = None,
    robot_y: float | None = None,
    visited_candidates: set[tuple[int, int]] | None = None,
) -> tuple[list[Waypoint], float, bool, list[dict]]:
    """Copy of production plan_waypoints that scores via greedy_set_cover_variant.

    Reuses all stable production primitives (candidate generation, visibility,
    achievable-cell denominator, nearest-neighbour ordering, coverage ratio).
    Reads the variant knobs distance_model / scoring_sequential / gain_floor from
    config (defaults reproduce production expdecay scoring).
    """
    max_range_m: float = config.get("max_detection_range", 6.0)
    sampling_step_m: float = config.get("sampling_step_m", 3.0)
    num_rays: int = config.get("num_rays", 360)
    alpha: float = config.get("frontier_weight", 1.0)
    beta: float = config.get("coverage_weight", 1.0)
    gamma: float = config.get("travel_cost_weight", 1.0)
    max_waypoints: int | None = config.get("max_waypoints_per_plan", None)
    completion_threshold: float = config.get("exploration_completion_threshold", 0.90)
    warning_threshold: float = config.get("planner_coverage_warning_threshold", 0.90)
    # Variant knobs (default = production expdecay scoring).
    distance_model: str = config.get("distance_model", "expdecay")
    scoring_sequential: bool = bool(config.get("scoring_sequential", False))
    gain_floor: float = float(config.get("gain_floor", 0.0))

    sampling_step_px = max(1, int(sampling_step_m / map_data.resolution))
    max_range_px = max(1, int(max_range_m / map_data.resolution))

    all_candidates = generate_candidates(
        map_data.navigable_mask, sampling_step_px, map_data.resolution)

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

    candidates = [c for c in all_candidates if c not in (visited_candidates or set())]
    if not candidates:
        return [], coverage_ratio(map_data.covered_mask, achievable_cells), True, []

    if max_waypoints is None:
        overlap_sq = max(1.0, (max_range_m / sampling_step_m) ** 2)
        max_waypoints = max(5, int(np.ceil(len(candidates) / overlap_sq)))

    dist_from_robot: dict[tuple[int, int], float] | None = None
    if robot_x is not None and robot_y is not None and gamma > 0.0:
        H = map_data.navigable_mask.shape[0]
        r_col, r_row = world_to_pixel(
            robot_x, robot_y, map_data.resolution,
            map_data.origin_x, map_data.origin_y, H,
        )
        dist_map = navigable_distance_map(map_data.navigable_mask, r_col, r_row)
        dist_from_robot = {c: float(dist_map[c[1], c[0]]) for c in candidates}

    dist_provider = None
    if scoring_sequential and dist_from_robot is not None:
        def dist_provider(col: int, row: int,
                          _mask=map_data.navigable_mask, _cands=candidates):
            dm = navigable_distance_map(_mask, col, row)
            return {c: float(dm[c[1], c[0]]) for c in _cands}

    vis = compute_all_visibility(candidates, map_data, max_range_m, num_rays)
    selected_px, _, _, candidate_records = greedy_set_cover_variant(
        vis, alpha, beta,
        dist_from_robot=dist_from_robot,
        gamma=gamma,
        normaliser=float(max_range_px),
        max_waypoints=max_waypoints,
        distance_model=distance_model,
        sequential=scoring_sequential,
        dist_provider=dist_provider,
        gain_floor=gain_floor,
    )

    no_frontiers = not any(len(vis[c][1]) > 0 for c in candidates)
    current_ratio = coverage_ratio(map_data.covered_mask, achievable_cells)

    if no_frontiers and current_ratio >= completion_threshold:
        return [], current_ratio, no_frontiers, candidate_records

    waypoints = nearest_neighbor_order(selected_px, map_data, vis, robot_x, robot_y)

    records_by_pos = {(r["col"], r["row"]): r for r in candidate_records}
    for wp in waypoints:
        rec = records_by_pos.get((wp.col, wp.row), {})
        wp.frontier_gain = rec.get("frontier_gain")
        wp.coverage_gain = rec.get("coverage_gain")
        wp.geodesic_dist_px = rec.get("geodesic_dist_px")
        wp.score         = rec.get("score")

    return waypoints, current_ratio, no_frontiers, candidate_records


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class VariantSession(ExplorationSession):
    """ExplorationSession that plans via plan_waypoints_variant.

    Reuses all of ExplorationSession's state management (visited-candidate
    tracking, set_map candidate regeneration, mid-path replanner) — only the
    per-plan scoring call is overridden, so the benchmark exercises the real
    session lifecycle with the selected scoring variant.
    """

    def plan_waypoints_raw(self, robot_x: float, robot_y: float):
        from exploration.execution_strategy import MidPathReplanner
        result = plan_waypoints_variant(
            self._md, self._config, robot_x, robot_y, self.visited_candidates
        )
        self._replanner = MidPathReplanner(self.visit_radius_px)
        return result
