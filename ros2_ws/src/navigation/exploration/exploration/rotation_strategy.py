"""
Rotation strategies for waypoint observation.

Provides two building blocks used by the visual demo scripts:

  get_headings(wp, observation_increment, current_heading)
      Returns the ordered list of headings the robot should rotate through
      at a waypoint: the exact minimum-total-angular-travel visiting order
      from the robot's current heading (each move costs the wrapped shortest
      delta, matching the node's Spin goals).

  RotationState
      Tracks per-heading coverage gain and signals early-stop when the area
      around the waypoint is saturated (no new cells for stop_after consecutive
      headings).

Usage in a demo loop::

    from rotation_strategy import get_headings, RotationState

    _rot_state = RotationState(stop_after=3)
    for h in get_headings(wp, OBSERVATION_INCREMENT, heading):
        heading = float(h)
        ratio = update_covered_mask(md, wp.col, wp.row, heading, ...)
        # ... log, vis update, plt.pause ...
        if not _rot_state.update(ratio):
            break  # saturated — no new coverage for 3 consecutive headings

Strategy options
----------------
stop_after=3  (default)   Planner headings + min-travel order + early stop.
                           Best trade-off: 2–6 headings instead of 12,
                           stops early when area is saturated.
stop_after=999             Planner headings + min-travel order, no interruption.
                           Use when you want exactly the planner's set.
wp.headings empty          Falls back to full 360° sweep (12 headings at 30°),
                           still ordered + early-stopped.
"""
from __future__ import annotations

import itertools


def get_headings(
    wp,
    observation_increment: float,
    current_heading: float = 0.0,
) -> list[float]:
    """Return headings to rotate through at *wp*, ordered for minimal angular travel.

    Uses ``wp.headings`` (the planner's pre-computed minimum coverage set,
    typically 2-6 headings) when available.  Falls back to a full 360° sweep
    at *observation_increment* steps when the planner produced no headings.

    The visiting order is the exact minimum-total-angular-travel order from
    *current_heading*, where each move costs the wrapped shortest delta , matching the node's Spin goals (_send_spin_goal sends shortest signed
    deltas).
    Args:
        wp:                   Waypoint dataclass with a ``headings`` attribute.
        observation_increment: Angular step in degrees for the fallback sweep.
        current_heading:      Robot's current heading in degrees at arrival.

    Returns:
        Ordered list of heading angles (degrees, 0=East, CCW positive).
    """
    if wp.headings:
        headings = [float(h) for h in wp.headings]
    else:
        headings = [float(h) for h in range(0, 360, round(observation_increment))]

    return _min_travel_order(headings, current_heading)


def _wrapped_step(frm: float, to: float) -> float:
    """|shortest signed delta| in degrees — the Spin cost of one heading change."""
    return abs((to - frm + 180.0) % 360.0 - 180.0)


def _sweep_cost(order: list[float], start: float) -> float:
    cost, cur = 0.0, start
    for h in order:
        cost += _wrapped_step(cur, h)
        cur = h
    return cost


def _min_travel_order(headings: list[float], current_heading: float) -> list[float]:
    """Exact minimum-total-angular-travel visiting order from *current_heading*.

    Planner heading sets are small (typically 2-6, at most 360/increment), so
    brute force over permutations is exact and cheap up to 7 headings (≤5040
    orders). Larger sets (the 12-heading full-sweep fallback) use the cheaper
    of the clockwise vs counterclockwise monotone sweep — for a dense ring a
    monotone sweep is already optimal, and permuting 12 headings (12! ≈ 5·10⁸)
    is not worth it.
    """
    if len(headings) <= 1:
        return list(headings)

    if len(headings) <= 7:
        best_order, best_cost = None, float("inf")
        for perm in itertools.permutations(headings):
            cost, cur = 0.0, current_heading
            for h in perm:
                cost += _wrapped_step(cur, h)
                cur = h
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost, best_order = cost, list(perm)
        return best_order

    cw = sorted(headings, key=lambda h: (h - current_heading) % 360.0)
    ccw = sorted(headings, key=lambda h: (current_heading - h) % 360.0)
    return cw if _sweep_cost(cw, current_heading) <= _sweep_cost(ccw, current_heading) else ccw


class RotationState:
    """Tracks per-heading coverage gain and signals when to stop rotating.

    Call :meth:`update` after every ``update_covered_mask`` invocation.
    Returns ``True`` while rotation should continue, ``False`` when the area
    is considered saturated (no new cells covered for *stop_after* consecutive
    headings).

    Args:
        stop_after: Number of consecutive zero-gain headings before stopping.
                    Default 3 — stops quickly once the area is saturated while
                    tolerating one or two headings that happen to add nothing.
    """

    def __init__(self, stop_after: int = 3) -> None:
        self.stop_after = stop_after
        self._consecutive_zero = 0
        self._prev_ratio: float | None = None

    def update(self, ratio: float) -> bool:
        """Register the coverage ratio after one heading rotation.

        Args:
            ratio: Value returned by ``update_covered_mask`` for the heading
                   just executed.

        Returns:
            ``True`` if rotation should continue, ``False`` to stop.
        """
        if self._prev_ratio is not None and ratio <= self._prev_ratio:
            self._consecutive_zero += 1
        else:
            self._consecutive_zero = 0
        self._prev_ratio = ratio
        return self._consecutive_zero < self.stop_after
