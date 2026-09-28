"""Unit tests for rotation_strategy.

Covers:
  - get_headings min-total-angular-travel ordering:
    exactness vs a permutation reference for small sets, the wrap-behind case the
    old clockwise sort got wrong, the 12-heading fallback sweep direction choice,
    and set preservation (ordering must never add/drop/duplicate headings).
  - RotationState early-stop contract.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration.rotation_strategy import (  # noqa: E402
    RotationState, get_headings, _min_travel_order, _sweep_cost, _wrapped_step,
)


@dataclass
class _WP:
    headings: list[float] = field(default_factory=list)


def _brute_force_min_cost(headings, start):
    return min(_sweep_cost(list(p), start) for p in itertools.permutations(headings))


class TestWrappedStep:
    def test_shortest_delta(self):
        assert _wrapped_step(10.0, 350.0) == pytest.approx(20.0)   # backwards is shorter
        assert _wrapped_step(350.0, 10.0) == pytest.approx(20.0)
        assert _wrapped_step(0.0, 180.0) == pytest.approx(180.0)
        assert _wrapped_step(90.0, 90.0) == 0.0


class TestGetHeadingsMinTravel:
    def test_wrap_behind_case_beats_clockwise_sort(self):
        """The motivating §12 case: robot at 10°, headings just behind it.
        Old clockwise sort went 330° forward first (320° + more); min-travel
        backs up −20°, −20° for 40° total."""
        wp = _WP(headings=[350.0, 330.0])
        order = get_headings(wp, 30.0, current_heading=10.0)
        assert order == [350.0, 330.0]
        assert _sweep_cost(order, 10.0) == pytest.approx(40.0)

    def test_exact_optimum_small_sets(self):
        """Cost equals the brute-force permutation optimum for all set sizes
        the planner produces (2..7)."""
        rng = random.Random(1234)
        for _ in range(300):
            n = rng.randint(2, 7)
            hs = rng.sample([i * 30.0 for i in range(12)], n)
            cur = rng.uniform(0.0, 360.0)
            order = get_headings(_WP(headings=hs), 30.0, cur)
            assert sorted(order) == sorted(hs)          # same multiset
            assert _sweep_cost(order, cur) == pytest.approx(
                _brute_force_min_cost(hs, cur))

    def test_arbitrary_angles_not_just_increment_multiples(self):
        hs = [13.7, 291.2, 200.0]
        cur = 45.0
        order = get_headings(_WP(headings=hs), 30.0, cur)
        assert _sweep_cost(order, cur) == pytest.approx(_brute_force_min_cost(hs, cur))

    def test_single_heading_passthrough(self):
        assert get_headings(_WP(headings=[123.0]), 30.0, 7.0) == [123.0]

    def test_empty_set_falls_back_to_full_sweep(self):
        order = get_headings(_WP(headings=[]), 30.0, current_heading=95.0)
        assert sorted(order) == [float(h) for h in range(0, 360, 30)]

    def test_full_sweep_picks_cheaper_direction(self):
        """12-heading ring: a monotone sweep is optimal; the direction whose first
        step from current_heading is shorter must be chosen. At 95° the nearest
        slots are 90° (5° behind) and 120° (25° ahead) → counterclockwise-start
        (90° first) is cheaper: 5 + 330 = 335 vs 25 + 330 = 355."""
        order = get_headings(_WP(headings=[]), 30.0, current_heading=95.0)
        assert order[0] == 90.0
        assert _sweep_cost(order, 95.0) == pytest.approx(335.0)

    def test_full_sweep_never_worse_than_old_clockwise_sort(self):
        rng = random.Random(99)
        ring = [float(h) for h in range(0, 360, 30)]
        for _ in range(100):
            cur = rng.uniform(0.0, 360.0)
            order = get_headings(_WP(headings=[]), 30.0, cur)
            cw = sorted(ring, key=lambda h: (h - cur) % 360.0)
            assert _sweep_cost(order, cur) <= _sweep_cost(cw, cur) + 1e-9

    def test_large_set_uses_monotone_sweep(self):
        """>7 headings: order must be one monotone direction (no zigzag)."""
        hs = [i * 40.0 for i in range(9)]
        order = get_headings(_WP(headings=hs), 40.0, current_heading=5.0)
        assert sorted(order) == sorted(hs)
        # monotone: signed steps all same sign (mod wrap)
        signed = [((order[i + 1] - order[i] + 180.0) % 360.0) - 180.0
                  for i in range(len(order) - 1)]
        assert all(s > 0 for s in signed) or all(s < 0 for s in signed)


class TestMinTravelOrderHelper:
    def test_empty_and_singleton(self):
        assert _min_travel_order([], 0.0) == []
        assert _min_travel_order([42.0], 0.0) == [42.0]

    def test_does_not_mutate_input(self):
        hs = [90.0, 0.0, 180.0]
        snapshot = list(hs)
        _min_travel_order(hs, 10.0)
        assert hs == snapshot


class TestRotationState:
    def test_stops_after_n_zero_gain(self):
        rs = RotationState(stop_after=3)
        assert rs.update(0.10) is True      # gain
        assert rs.update(0.10) is True      # 1st zero-gain
        assert rs.update(0.10) is True      # 2nd
        assert rs.update(0.10) is False     # 3rd → stop

    def test_gain_resets_counter(self):
        rs = RotationState(stop_after=2)
        assert rs.update(0.10) is True
        assert rs.update(0.10) is True      # 1st zero-gain
        assert rs.update(0.20) is True      # gain resets
        assert rs.update(0.20) is True      # 1st zero-gain again
        assert rs.update(0.20) is False     # 2nd → stop
