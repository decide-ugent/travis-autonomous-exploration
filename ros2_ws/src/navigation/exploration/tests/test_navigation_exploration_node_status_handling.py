"""
Tests for the Nav2 goal-result handling in
navigation/exploration/exploration/ros2_exploration_node.py.

These exercise the state machine's reaction to each NavigateToPose result status
(SUCCEEDED / ABORTED / CANCELED) and the "inaccessible waypoint" safety net:
try the next waypoint in the current plan, keep the failed spot UNVISITED, and
escalate (recovery Spin, then stop) when a whole plan fails from a stuck pose.

This is the class of bug that produced the field failure where nav2 aborted a
goal ("Start occupied") and the node re-planned to the SAME waypoint forever.

The ROS side-effects (_send_nav_goal, _send_recovery_spin, _clear_rviz_markers,
_arrive_no_spin) are replaced with spies so the tests assert DECISIONS (state,
_wp_index, streak, which action was taken) without needing live action servers.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")
from action_msgs.msg import GoalStatus  # noqa: E402

from exploration.ros2_exploration_node import (  # noqa: E402
    ExplorationNode,
    _State,
    _MAX_FAILED_PLAN_STREAK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _ros():
    """One rclpy context for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _wp(col: int, row: int):
    """Minimal Waypoint-like object (only col/row/x/y are read here)."""
    return SimpleNamespace(col=col, row=row, x=float(col), y=float(row))


def _result_future(status: int):
    """A stand-in for the get_result_async() future passed to _on_nav_result."""
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


@pytest.fixture
def node():
    """An ExplorationNode with all ROS side-effects stubbed out.

    The node is constructed for real (so the actual _on_nav_result / advance
    logic runs), but every method that would touch a live action server or RViz
    is replaced with a spy that records the call. We seed a 3-waypoint plan.
    """
    n = ExplorationNode()

    # Spies for side-effecting methods.
    n._sent_goals = []
    n._recovery_spins = 0
    n._arrivals = 0
    n._send_nav_goal = lambda wp: n._sent_goals.append(wp)          # type: ignore
    n._send_recovery_spin = lambda: setattr(                        # type: ignore
        n, "_recovery_spins", n._recovery_spins + 1)
    n._clear_rviz_markers = lambda: None                           # type: ignore
    n._arrive_no_spin = lambda: setattr(n, "_arrivals", n._arrivals + 1)  # type: ignore
    # SUCCEEDED with see_while_moving True routes to _arrive_no_spin (stubbed).
    n._see_while_moving = True

    n._waypoints = [_wp(10, 10), _wp(20, 20), _wp(30, 30)]
    n._wp_index = 0
    n._plan_all_failed_streak = 0
    n._state = _State.TRAVELING

    yield n
    n.destroy_node()


# ---------------------------------------------------------------------------
# ABORTED, the safety net
# ---------------------------------------------------------------------------

def test_abort_tries_next_waypoint_same_plan(node):
    """ABORTED on waypoint 0 advances to waypoint 1 in the SAME plan (no replan)."""
    node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
    assert node._wp_index == 1
    assert node._sent_goals == [node._waypoints[1]]   # next waypoint sent
    assert node._state != _State.PLANNING             # did not discard the plan
    assert node._plan_all_failed_streak == 0          # not a whole-plan failure yet


def test_abort_does_not_mark_failed_spot_visited(node):
    """The aborted waypoint's cell is NOT added to visited_candidates.

    'Could not reach' and 'observed' are different facts: the robot never got there, so
    the area is still uncovered. Recording an abort as "visited" told the planner that
    region was already covered and permanently froze exploration (hospital stall).
    Repeated aborts may still park the waypoint as *unreachable* (a transient,
    recoverable state), but never as visited.
    """
    # Stub session exposing the real API surface the node uses on an abort.
    node._session = SimpleNamespace(
        visited_candidates=set(),
        unreachable=set(),
        on_nav_aborted=lambda wp: False,   # below the blacklist threshold
    )
    failed = node._waypoints[0]
    node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
    assert (failed.col, failed.row) not in node._session.visited_candidates


def test_abort_last_waypoint_exhausts_plan_then_replans(node):
    """ABORTED on the LAST waypoint (plan exhausted) → back to PLANNING, streak=1."""
    node._wp_index = len(node._waypoints) - 1
    node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
    assert node._state == _State.PLANNING
    assert node._plan_all_failed_streak == 1
    assert node._sent_goals == []                     # nothing left to send


# ---------------------------------------------------------------------------
# Stuck-pose escalation
# ---------------------------------------------------------------------------

def _fail_whole_plan_once(node):
    """Simulate one full plan failing: the LAST waypoint aborts while the robot
    is travelling. (In the real flow the prior waypoints already advanced the
    index; here we jump to the last one and abort it.) Resets state to TRAVELING
    first, as _do_planning/_send_nav_goal would before the next result arrives."""
    node._state = _State.TRAVELING
    node._wp_index = len(node._waypoints) - 1
    node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))


def test_second_full_plan_failure_triggers_recovery_spin(node):
    """A whole plan failing twice in a row triggers the recovery Spin."""
    _fail_whole_plan_once(node)     # streak 1 -> PLANNING
    _fail_whole_plan_once(node)     # streak 2 -> recovery Spin
    assert node._plan_all_failed_streak == 2
    assert node._recovery_spins == 1


def test_rapid_failures_do_not_stop_before_the_timeout(node):
    """A BURST of failed plans must NOT stop the run: the streak alone is not a duration.

    Regression guard for the hospital run: with the robot parked in the inflation band
    nav2 cannot plan from that start pose and rejects every goal in ~30 ms, so a whole
    20-waypoint plan exhausts in <1 s and the 5-plan cap was reached in a couple of
    SECONDS, the run gave up long before anyone could move the robot. The stop now
    additionally requires exploration.plan_timeout_s of elapsed time.
    """
    node._plan_timeout_s = 300.0
    for _ in range(_MAX_FAILED_PLAN_STREAK + 3):
        _fail_whole_plan_once(node)
    assert node._plan_all_failed_streak >= _MAX_FAILED_PLAN_STREAK
    assert node._state != _State.COMPLETE, (
        "instant back-to-back failures must not trigger the stuck-stop; the operator "
        "needs the plan_timeout_s window to free the robot"
    )


def test_persistent_failure_stops_once_timeout_elapsed(node):
    """Beyond _MAX_FAILED_PLAN_STREAK *and* plan_timeout_s the node stops (COMPLETE)."""
    node._plan_timeout_s = 300.0
    for _ in range(_MAX_FAILED_PLAN_STREAK):
        _fail_whole_plan_once(node)
    # Backdate the stuck-since clock so the time gate is satisfied.
    node._plan_all_failed_since -= 301.0
    _fail_whole_plan_once(node)
    assert node._plan_all_failed_streak >= _MAX_FAILED_PLAN_STREAK
    assert node._state == _State.COMPLETE


class TestInflatedPoseGate:
    """Nav2 cannot plan FROM an occupied/inflated start pose: it rejects every goal in
    ~30 ms whatever the goal is. Since an abort immediately sends the next waypoint
    (callback cascade, not tick-limited), a whole plan burns through in <1 s and the
    1 Hz replan repeats it, thousands of doomed goals while the robot is wedged.
    _send_nav_goal must therefore send NOTHING while the pose is un-plannable.
    """

    def _prepare(self, node, navigable: bool):
        import numpy as np
        from types import SimpleNamespace
        mask = np.zeros((50, 50), dtype=bool)
        if navigable:
            mask[25, 25] = True
        node._md = SimpleNamespace(navigable_mask=mask)
        node._get_robot_pixel_pos = lambda: (25, 25)          # type: ignore
        # Restore the REAL _send_nav_goal (the fixture stubs it out).
        node._send_nav_goal = ExplorationNode._send_nav_goal.__get__(node)
        # Non-blocking readiness probe (replaced the old blocking wait_for_server, which
        # froze the single-threaded executor for 5s per attempt). False here means a
        # would-be send fails loudly instead of reaching the real action client.
        node._nav_client = SimpleNamespace(server_is_ready=lambda: False)
        node._inflated_since = None
        node._plan_timeout_s = 300.0

    def test_no_goal_sent_while_inflated(self, node):
        """Verifies: an un-plannable pose short-circuits before the action client is
        even consulted, and the node returns to PLANNING to re-check next tick."""
        self._prepare(node, navigable=False)
        node._state = _State.TRAVELING
        node._send_nav_goal(node._waypoints[0])
        assert node._state == _State.PLANNING
        assert node._inflated_since is not None, "the inflation clock must start"

    def test_inflated_stop_fires_only_after_timeout(self, node):
        """Verifies: the wedged path is time-bounded by plan_timeout_s. It must not stop
        immediately (that would deny the rescue window) but must stop once elapsed."""
        self._prepare(node, navigable=False)
        node._send_nav_goal(node._waypoints[0])
        assert node._state != _State.COMPLETE          # rescue window still open
        node._inflated_since -= 301.0                  # backdate past the timeout
        node._send_nav_goal(node._waypoints[0])
        assert node._state == _State.COMPLETE

    def test_clock_clears_when_pose_becomes_plannable(self, node):
        """Verifies: a brief inflation excursion does not count toward a later episode."""
        self._prepare(node, navigable=False)
        node._send_nav_goal(node._waypoints[0])
        assert node._inflated_since is not None
        self._prepare(node, navigable=True)            # robot back in free space
        node._send_nav_goal(node._waypoints[0])        # proceeds past the gate
        assert node._inflated_since is None


def test_zero_timeout_never_stops(node):
    """plan_timeout_s == 0 means 'never give up', the stuck-stop is disabled."""
    node._plan_timeout_s = 0.0
    for _ in range(_MAX_FAILED_PLAN_STREAK):
        _fail_whole_plan_once(node)
    node._plan_all_failed_since -= 100000.0   # arbitrarily long
    _fail_whole_plan_once(node)
    assert node._state != _State.COMPLETE


# ---------------------------------------------------------------------------
# SUCCEEDED, resets the stuck streak
# ---------------------------------------------------------------------------

def test_success_resets_failed_plan_streak(node):
    """A real arrival clears the failed-plan streak so past failures don't
    accumulate toward the stop threshold across successful legs."""
    node._plan_all_failed_streak = 3
    node._state = _State.TRAVELING
    node._on_nav_result(_result_future(GoalStatus.STATUS_SUCCEEDED))
    assert node._plan_all_failed_streak == 0
    assert node._arrivals == 1                        # routed to _arrive_no_spin


# ---------------------------------------------------------------------------
# CANCELED, deliberate interruption, must be inert
# ---------------------------------------------------------------------------

def test_cancel_is_ignored(node):
    """A CANCELED result (mid-path replan cancels the goal) must NOT advance the
    index, send a goal, touch the streak, or trigger recovery. The canceller
    already set the next state."""
    node._wp_index = 1
    node._plan_all_failed_streak = 0
    node._state = _State.PLANNING          # canceller already moved us here
    node._on_nav_result(_result_future(GoalStatus.STATUS_CANCELED))
    assert node._wp_index == 1             # unchanged
    assert node._sent_goals == []          # no goal sent
    assert node._plan_all_failed_streak == 0
    assert node._recovery_spins == 0


def test_stale_cancel_after_new_goal_does_not_corrupt_index(node):
    """The race the CANCELED branch closes: a stale cancel result arriving AFTER
    a new plan has already started (state back to NAVIGATING, index reset) must
    not advance/corrupt the new plan's index."""
    # Simulate: new plan started, we're navigating waypoint 0 of it.
    node._wp_index = 0
    node._state = _State.NAVIGATING
    node._on_nav_result(_result_future(GoalStatus.STATUS_CANCELED))
    assert node._wp_index == 0             # new plan's index untouched
    assert node._sent_goals == []


# ---------------------------------------------------------------------------
# Arrival failsafe, nav2 failure is not proof the waypoint was missed
# ---------------------------------------------------------------------------

class TestArrivalFailsafe:
    """Nav2 reporting ABORTED does not prove the robot is not at the goal: it may
    have stopped just outside nav2's own goal checker, or a human may drive it the
    last stretch (the teleop rescue in README.md). On failure the node holds in
    VERIFYING and decides from the TF pose, accepting the waypoint within a loose
    position tolerance with orientation ignored entirely, and only declaring failure
    after the robot has been STILL for arrival_verify_timeout_s.

    Pose comes from TF (map -> base_frame), the same frame as the goal. Odometry is
    deliberately not used: it drifts under SLAM and lives in another frame.
    """

    def _prepare(self, node, goal_xy=(10.0, 10.0), tol=0.4, timeout=30.0):
        """Arm the failsafe as if a goal to goal_xy had been sent and then failed."""
        node._current_goal_xy = goal_xy
        node._arrival_tolerance_m = tol
        node._arrival_verify_timeout_s = timeout
        node._arrival_motion_eps_m = 0.05
        # Marking coverage needs a real map; the failsafe path itself does not.
        node._see_while_moving = False
        node._session = SimpleNamespace(
            visited_candidates=set(),
            on_nav_aborted=lambda wp: False,
            on_arrive_clear_aborts=lambda wp: None,
        )

    def _at(self, node, x, y):
        node._get_robot_world_pos = lambda: (x, y)                  # type: ignore
        node._get_robot_pixel_pos = lambda: (25, 25)                # type: ignore
        node._get_robot_heading_deg = lambda: 0.0                   # type: ignore

    def test_failure_enters_verifying_instead_of_aborting(self, node):
        """Verifies the core change: a nav2 abort no longer immediately advances the
        waypoint. The index must be untouched and no next goal sent while verifying."""
        self._prepare(node)
        self._at(node, 99.0, 99.0)          # nowhere near the goal yet
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        assert node._state == _State.VERIFYING
        assert node._wp_index == 0                    # not advanced
        assert node._sent_goals == []                 # no next goal sent
        assert node._plan_all_failed_streak == 0

    def test_pose_within_tolerance_is_accepted_as_arrival(self, node):
        """Verifies the rescue: standing within tolerance of the goal after a nav2
        failure counts as a real arrival, routed through the normal arrival path."""
        self._prepare(node, goal_xy=(10.0, 10.0), tol=0.4)
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        self._at(node, 10.3, 10.0)          # 0.3 m away, inside the 0.4 m tolerance
        node._see_while_moving = True       # route to the stubbed _arrive_no_spin
        node._do_verify_check()
        assert node._arrivals == 1                    # treated as a full arrival
        # The state is left to the stubbed _arrive_no_spin here; in production it
        # continues into _finish_waypoint -> _advance_waypoint, which always either
        # sends the next goal or sets the next state. What matters is that the
        # waypoint was accepted rather than rejected.
        assert node._wp_index == 0                    # not advanced by an abort
        assert node._verify_since is None             # window closed

    def test_orientation_is_ignored_entirely(self, node):
        """Verifies yaw is fully tolerated: arrival is accepted at any heading, since
        nav2 never turned the robot to the goal yaw once it gave up."""
        self._prepare(node, goal_xy=(10.0, 10.0))
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        self._at(node, 10.1, 10.0)
        node._get_robot_heading_deg = lambda: 179.0   # type: ignore  facing away
        node._see_while_moving = True
        node._do_verify_check()
        assert node._arrivals == 1

    def test_arrival_clears_the_failed_plan_streak(self, node):
        """Verifies a pose-verified arrival is indistinguishable from a nav2-reported
        one downstream: it proves the robot is not stuck, so the streak resets."""
        self._prepare(node)
        node._plan_all_failed_streak = 3
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        self._at(node, 10.0, 10.0)
        node._see_while_moving = True
        node._do_verify_check()
        assert node._plan_all_failed_streak == 0

    def test_still_and_far_times_out_into_the_abort_path(self, node):
        """Verifies the window is bounded: a robot parked away from the goal ends up
        on exactly the original abort path, advancing to the next waypoint."""
        self._prepare(node, timeout=30.0)
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._do_verify_check()                       # establishes the stillness clock
        assert node._state == _State.VERIFYING        # not given up yet
        node._verify_since -= 31.0                    # backdate past the timeout
        node._do_verify_check()
        assert node._wp_index == 1                    # advanced, as a plain abort would
        assert node._sent_goals == [node._waypoints[1]]

    def test_motion_keeps_the_window_open_past_the_timeout(self, node):
        """Verifies the timeout is a STILLNESS timeout, not a wall-clock cap: a human
        actively driving the robot must not have the window closed under them."""
        self._prepare(node, timeout=30.0)
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._do_verify_check()
        node._verify_since -= 29.0                    # nearly timed out
        self._at(node, 98.0, 99.0)                    # but the robot moved 1 m
        node._do_verify_check()
        assert node._state == _State.VERIFYING        # clock restarted, still waiting
        assert node._wp_index == 0

    def test_jitter_below_eps_does_not_hold_the_window_open(self, node):
        """Verifies estimator noise is not mistaken for driving: sub-eps pose wobble
        must still time out, or a parked robot would wait forever."""
        self._prepare(node, timeout=30.0)
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._do_verify_check()
        node._verify_since -= 31.0
        self._at(node, 99.001, 99.0)                  # 1 mm of jitter, below eps
        node._do_verify_check()
        assert node._wp_index == 1                    # timed out anyway

    def test_missing_tf_pose_makes_no_verdict(self, node):
        """Verifies a TF outage is inert: it must neither accept an arrival nor count
        as stillness, since the node then knows nothing about where the robot is."""
        self._prepare(node)
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._get_robot_world_pos = lambda: None      # type: ignore  TF unavailable
        node._verify_since -= 31.0                    # even well past the timeout
        node._do_verify_check()
        assert node._state == _State.VERIFYING        # no verdict either way
        assert node._wp_index == 0
        assert node._arrivals == 0

    def test_zero_timeout_disables_the_failsafe(self, node):
        """Verifies the escape hatch: arrival_verify_timeout_s == 0 restores the
        original immediate-abort behaviour with no verification window at all."""
        self._prepare(node, timeout=0.0)
        self._at(node, 10.0, 10.0)                    # even standing ON the goal
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        assert node._state != _State.VERIFYING
        assert node._wp_index == 1                    # aborted immediately
        assert node._sent_goals == [node._waypoints[1]]

    def test_no_recorded_goal_falls_back_to_immediate_abort(self, node):
        """Verifies the failsafe cannot act on an unknown goal: with no recorded goal
        pose there is nothing to measure against, so the legacy path must run."""
        self._prepare(node)
        node._current_goal_xy = None                  # no goal ever sent
        self._at(node, 10.0, 10.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        assert node._state != _State.VERIFYING
        assert node._wp_index == 1

    def test_timed_out_waypoint_is_counted_as_aborted_exactly_once(self, node):
        """Verifies the failsafe does not double-count: the waypoint that survives a
        failed window must register exactly one abort, as a plain failure would."""
        self._prepare(node, timeout=30.0)
        aborts = []
        node._session.on_nav_aborted = lambda wp: (aborts.append(wp), False)[1]
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._do_verify_check()
        node._verify_since -= 31.0
        node._do_verify_check()
        assert len(aborts) == 1

    def test_accepted_arrival_continues_the_plan_like_a_real_arrival(self, node):
        """Verifies VERIFYING is not a dead end: with the real (unstubbed) arrival
        continuation, accepting a waypoint runs _finish_waypoint and advances the plan
        exactly as a nav2-reported arrival does.

        The final _state is not asserted because the fixture stubs _send_nav_goal,
        which is what sets NAVIGATING in production (ros2_exploration_node.py:594);
        sending the next goal is the observable outcome that proves the handoff.
        """
        self._prepare(node, goal_xy=(10.0, 10.0))
        # Skip the observation work (needs a live map) but keep the real continuation
        # that owns the state transition.
        import numpy as np
        node._md = SimpleNamespace(
            navigable_mask=np.ones((50, 50), dtype=bool),
            resolution=1.0, origin_x=0.0, origin_y=0.0,
        )
        node._finish_waypoint = ExplorationNode._finish_waypoint.__get__(node)  # type: ignore
        # Record the waypoint the real _finish_waypoint reports as reached.
        node._arrived_waypoints = []
        node._session.on_arrive = lambda wp: node._arrived_waypoints.append(
            node._waypoints.index(wp))
        node._is_slam = False
        node._replan_every_n_step = -1
        node._coverage = 0.5
        node._arrive_no_spin = lambda: node._finish_waypoint()  # type: ignore
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        assert node._state == _State.VERIFYING
        self._at(node, 10.0, 10.0)
        node._see_while_moving = True
        node._do_verify_check()
        assert node._arrived_waypoints == [0]         # ran the real _finish_waypoint
        assert node._wp_index == 1                    # plan advanced past the waypoint
        assert node._sent_goals == [node._waypoints[1]]
        assert node._verify_since is None             # window closed

    def test_failed_spot_is_not_marked_visited(self, node):
        """Verifies the invariant the original abort path protects is preserved: a
        waypoint the robot never reached stays UNVISITED, so its area is still
        uncovered and can be re-planned from a later pose."""
        self._prepare(node, timeout=30.0)
        failed = node._waypoints[0]
        self._at(node, 99.0, 99.0)
        node._on_nav_result(_result_future(GoalStatus.STATUS_ABORTED))
        node._do_verify_check()
        node._verify_since -= 31.0
        node._do_verify_check()
        assert (failed.col, failed.row) not in node._session.visited_candidates


# ---------------------------------------------------------------------------
# Nav2 silent-death watchdog, a goal that never returns any result at all
# ---------------------------------------------------------------------------

class TestNav2Watchdog:
    """rclpy never fails a pending result future when its action server disappears
    (there is no set_exception anywhere in ActionClient), so a nav2 that dies or is
    restarted mid-goal leaves the node waiting in TRAVELING forever: no SUCCEEDED, no
    ABORTED, no result of any kind. Nothing else bounds that wait, because
    plan_timeout_s only guards the empty-plan and plan-exhaustion paths and the
    mid-path replanner only fires once the robot has MOVED, which a robot with a dead
    nav2 does not do.

    The watchdog trips on the server vanishing (nav2 gone) or on feedback drying up
    while it is still advertised (nav2 wedged), then hands the waypoint to the arrival
    failsafe so a robot that did reach the goal still gets credit.
    """

    def _prepare(self, node, server_ready=True, stall=30.0, enabled=True):
        node._nav2_watchdog_enabled = enabled
        node._nav2_stall_timeout_s = stall
        node._nav_client = SimpleNamespace(server_is_ready=lambda: server_ready)
        node._current_goal_xy = (10.0, 10.0)
        node._arrival_tolerance_m = 0.4
        node._arrival_verify_timeout_s = 30.0
        node._arrival_motion_eps_m = 0.05
        node._see_while_moving = False
        node._session = SimpleNamespace(
            visited_candidates=set(),
            on_nav_aborted=lambda wp: False,
            on_arrive_clear_aborts=lambda wp: None,
        )
        node._get_robot_world_pos = lambda: (99.0, 99.0)            # type: ignore
        node._get_robot_heading_deg = lambda: 0.0                   # type: ignore
        node._goal_handle = SimpleNamespace(
            cancel_goal_async=lambda: pytest.fail(
                "must not cancel a goal on a dead/wedged server: the cancel future "
                "never resolves either, which is the hang the watchdog exists to break"))
        # A goal is in flight and nav2 last spoke right now.
        node._last_nav_progress_t = node.get_clock().now().nanoseconds / 1e9

    def test_server_disappearing_trips_into_verifying(self, node):
        """Verifies the core case: nav2 dies mid-goal, and instead of hanging in
        TRAVELING forever the node abandons the goal and verifies the arrival."""
        self._prepare(node, server_ready=False)
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING
        assert node._goal_handle is None              # stale handle dropped
        assert node._wp_index == 0                    # waypoint not given up on yet
        assert node._sent_goals == []

    def test_feedback_stall_trips_while_server_still_advertised(self, node):
        """Verifies the second half: a nav2 that is up and advertising but has stopped
        producing feedback is presumed wedged, which server liveness alone misses."""
        self._prepare(node, server_ready=True, stall=30.0)
        node._last_nav_progress_t -= 31.0             # silent past the timeout
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING

    def test_fresh_feedback_keeps_resetting_the_clock(self, node):
        """Verifies a healthy nav2 is never disturbed: feedback keeps the stall clock
        current, so the watchdog stays quiet however long the leg takes.

        This is also what protects the STUCK-ROBOT case. NavigateToPose.Feedback carries
        number_of_recoveries because nav2 publishes feedback throughout its BackUp/Spin/
        Wait recovery behaviours, so a robot fighting its way out of an obstacle keeps
        the clock fresh however long recovery takes. The watchdog measures nav2 SILENCE,
        never lack of progress.
        """
        self._prepare(node, server_ready=True, stall=30.0)
        for _ in range(5):
            node._last_nav_progress_t -= 29.0         # nearly stale
            node._on_nav_feedback(object())           # then nav2 speaks again
            node._check_nav2_alive()
            assert node._state == _State.TRAVELING

    def test_no_goal_in_flight_disarms_the_watchdog(self, node):
        """Verifies the watchdog only judges an in-flight goal: with nothing being
        tracked it must stay silent even if the server is missing, so the arrival
        observation and re-planning are not disturbed."""
        self._prepare(node, server_ready=False)
        node._last_nav_progress_t = None              # no goal being tracked
        node._check_nav2_alive()
        assert node._state == _State.TRAVELING

    def test_disabled_watchdog_never_trips(self, node):
        """Verifies the escape hatch: nav2.watchdog_enabled false restores the previous
        behaviour exactly, even with the server gone."""
        self._prepare(node, server_ready=False, enabled=False)
        node._check_nav2_alive()
        assert node._state == _State.TRAVELING

    def test_zero_stall_timeout_keeps_the_liveness_half(self, node):
        """Verifies the two halves are independent: stall timeout 0 disables only the
        feedback half, a server that vanishes is still caught immediately."""
        self._prepare(node, server_ready=True, stall=0.0)
        node._last_nav_progress_t -= 9999.0           # silent forever, must NOT trip
        node._check_nav2_alive()
        assert node._state == _State.TRAVELING
        node._nav_client = SimpleNamespace(server_is_ready=lambda: False)
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING        # liveness half still active

    def test_trip_then_robot_at_goal_accepts_the_waypoint(self, node):
        """Verifies the watchdog reuses the arrival failsafe rather than duplicating
        it: a robot sitting at the goal when nav2 vanished still gets credit."""
        self._prepare(node, server_ready=False)
        node._arrivals = 0
        node._see_while_moving = True                 # route to the stubbed _arrive_no_spin
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING
        node._get_robot_world_pos = lambda: (10.1, 10.0)   # type: ignore  at the goal
        node._do_verify_check()
        assert node._arrivals == 1

    def test_trip_then_still_and_far_falls_through_to_abort(self, node):
        """Verifies a genuinely lost waypoint still fails normally: the watchdog opens
        the window, and a robot parked away from the goal ends on the ordinary abort
        path rather than being credited."""
        self._prepare(node, server_ready=False)
        node._check_nav2_alive()
        node._do_verify_check()
        node._verify_since -= 31.0
        node._do_verify_check()
        assert node._wp_index == 1                    # advanced, as a plain abort would
        assert node._sent_goals == [node._waypoints[1]]

    def test_trip_without_failsafe_still_gives_up_rather_than_hanging(self, node):
        """Verifies the watchdog never leaves the node wedged: with the failsafe
        disabled it must still abandon the goal, since hanging forever is the exact
        bug being fixed."""
        self._prepare(node, server_ready=False)
        node._arrival_verify_timeout_s = 0.0          # failsafe off
        node._check_nav2_alive()
        # The state is left to the stubbed _send_nav_goal here (it sets NAVIGATING in
        # production); advancing the plan is the observable proof the goal was dropped
        # rather than waited on forever.
        assert node._state != _State.VERIFYING        # no window, went straight to abort
        assert node._goal_handle is None              # stale handle dropped
        assert node._wp_index == 1                    # gave the waypoint up
        assert node._sent_goals == [node._waypoints[1]]

    def test_late_result_after_a_trip_does_not_double_accept(self, node):
        """Verifies the restart race: a SUCCEEDED from the OLD nav2 landing after the
        failsafe already resolved the waypoint must not mark it twice or advance the
        plan a second time."""
        self._prepare(node, server_ready=False)
        node._arrivals = 0
        node._see_while_moving = True
        node._check_nav2_alive()
        node._get_robot_world_pos = lambda: (10.0, 10.0)   # type: ignore
        node._do_verify_check()
        assert node._arrivals == 1                    # accepted once
        node._state = _State.PLANNING                 # failsafe moved on
        node._on_nav_result(_result_future(GoalStatus.STATUS_SUCCEEDED))
        assert node._arrivals == 1                    # late result ignored

    def test_send_goal_without_a_server_returns_to_planning_without_blocking(self, node):
        """Verifies re-linking: a missing server must not block the single-threaded
        executor (the old wait_for_server froze TF, /map and the timer for 5s per try)
        and must return to PLANNING so the next 1 Hz tick retries and picks up a nav2
        that has since restarted."""
        import time
        import numpy as np
        node._md = SimpleNamespace(navigable_mask=np.ones((50, 50), dtype=bool))
        node._get_robot_pixel_pos = lambda: (25, 25)               # type: ignore
        node._nav_client = SimpleNamespace(server_is_ready=lambda: False)
        node._send_nav_goal = ExplorationNode._send_nav_goal.__get__(node)  # type: ignore
        node._inflated_since = None
        node._plan_timeout_s = 300.0
        node._state = _State.TRAVELING
        t0 = time.perf_counter()
        node._send_nav_goal(node._waypoints[0])
        elapsed = time.perf_counter() - t0
        assert node._state == _State.PLANNING
        assert elapsed < 1.0, f"send blocked for {elapsed:.1f}s; must be non-blocking"


# ---------------------------------------------------------------------------
# TF listener recovery, surviving a /tf_static publisher replacement
# ---------------------------------------------------------------------------

class TestTfRecovery:
    """When the other container restarts, its /tf_static publishers are destroyed and
    recreated with new participant GUIDs. A latched static sample is delivered only
    once, when a subscription MATCHES a publisher, so a listener that was already
    matched never receives the replacement. Static transforms are never republished,
    so this cannot self-heal: map -> base_frame stops resolving permanently even though
    the dynamic /tf stream is fine and a freshly started tf2_echo still works.

    No cache length and no Buffer.clear() can fix that, because the missing sample will
    never be re-sent to an already-matched subscription. Only a NEW subscription can
    obtain it, which is what the rebuild creates.
    """

    def _prepare(self, node, lookup_ok=False, timeout=30.0, enabled=True):
        node._nav2_watchdog_enabled = enabled
        node._nav2_stall_timeout_s = timeout
        node._base_frame = 'base_link'
        node._unregistered = 0

        def _lookup(target, source, when):
            if lookup_ok:
                return SimpleNamespace(transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.0, y=2.0, z=0.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
            raise RuntimeError('Could not find a connection between map and base_link')

        node._tf_buffer = SimpleNamespace(
            lookup_transform=_lookup,
            can_transform=lambda *a, **k: (False, 'missing static link'),
            all_frames_as_yaml=lambda: 'base_link:\n  parent: odom\n',
        )
        node._tf_listener = SimpleNamespace(
            unregister=lambda: setattr(node, '_unregistered', node._unregistered + 1))
        node._tf_fail_since = None
        node._tf_rebuilds = 0

    def test_failure_starts_the_clock_but_does_not_rebuild_immediately(self, node):
        """Verifies transient TF gaps are tolerated: the node already handles a missing
        pose by retrying, so a brief outage must not thrash the buffer."""
        self._prepare(node)
        assert node._get_robot_world_pos() is None
        assert node._tf_fail_since is not None         # clock started
        node._check_tf_alive()
        assert node._tf_rebuilds == 0                  # too soon to rebuild
        assert node._unregistered == 0

    def test_persistent_failure_rebuilds_the_listener(self, node):
        """Verifies the actual fix: after the threshold the listener is torn down and
        replaced, so fresh subscriptions re-match every current publisher and pull the
        latched /tf_static samples the old subscription can never receive."""
        self._prepare(node, timeout=30.0)
        node._get_robot_world_pos()
        old_buffer = node._tf_buffer
        node._tf_fail_since -= 31.0                    # broken past the threshold
        node._check_tf_alive()
        assert node._unregistered == 1                 # old subscriptions destroyed
        assert node._tf_buffer is not old_buffer       # NEW buffer, not a clear()
        assert node._tf_rebuilds == 1
        assert node._tf_fail_since is None             # re-armed for a fresh episode

    def test_successful_lookup_clears_the_clock(self, node):
        """Verifies a recovered TF tree resets the state, so a later isolated failure
        starts a fresh clock instead of inheriting an old one and rebuilding at once."""
        self._prepare(node, lookup_ok=False)
        node._get_robot_world_pos()
        assert node._tf_fail_since is not None
        self._prepare(node, lookup_ok=True)            # TF healthy again
        node._tf_fail_since = 1.0                      # stale clock from before
        assert node._get_robot_world_pos() == (1.0, 2.0)
        assert node._tf_fail_since is None

    def test_heading_lookup_shares_the_same_bookkeeping(self, node):
        """Verifies the choke point really is shared: the heading helper must feed the
        same health state, or a run that only reads heading would never trigger recovery."""
        self._prepare(node)
        assert node._get_robot_heading_deg() is None
        assert node._tf_fail_since is not None

    def test_disabled_watchdog_never_rebuilds(self, node):
        """Verifies the single escape hatch covers TF too."""
        self._prepare(node, enabled=False)
        node._get_robot_world_pos()
        node._tf_fail_since -= 999.0
        node._check_tf_alive()
        assert node._tf_rebuilds == 0
        assert node._unregistered == 0

    def test_runs_in_waiting_for_map(self, node):
        """Regression guard for the placement decision: the check must sit ABOVE the
        WAITING_FOR_MAP early return in the timer. A static-map run whose TF breaks sits
        in WAITING_FOR_MAP retrying _init_session forever, so a check inside the
        per-state branches would never fire in exactly the case that needs it."""
        self._prepare(node, timeout=30.0)
        node._get_robot_world_pos()
        node._tf_fail_since -= 31.0
        node._state = _State.WAITING_FOR_MAP
        node._is_slam = False
        node._session = None
        node._init_session = lambda: False             # type: ignore  TF still down
        node._publish_status = lambda: None            # type: ignore
        node._exploration_timer()
        assert node._tf_rebuilds == 1, (
            "the TF check must run before the WAITING_FOR_MAP early return")

    def test_repeated_failure_trips_again_after_another_timeout(self, node):
        """Verifies recovery re-arms rather than latching: if a rebuild does not fix it,
        trying again every threshold (and logging the missing link each time) is strictly
        better than the old behaviour of failing silently forever."""
        self._prepare(node, timeout=30.0)
        node._get_robot_world_pos()
        node._tf_fail_since -= 31.0
        node._check_tf_alive()
        assert node._tf_rebuilds == 1
        node._get_robot_world_pos()                    # still broken after the rebuild
        node._tf_fail_since -= 31.0
        node._check_tf_alive()
        assert node._tf_rebuilds == 2

    def test_broken_diagnostics_do_not_prevent_the_rebuild(self, node):
        """Verifies diagnostics never block recovery: a buffer so broken that
        can_transform itself raises is exactly when the rebuild matters most."""
        self._prepare(node, timeout=30.0)
        node._get_robot_world_pos()

        def _boom(*a, **k):
            raise RuntimeError('buffer is wedged')

        node._tf_buffer.can_transform = _boom
        node._tf_buffer.all_frames_as_yaml = _boom
        node._tf_fail_since -= 31.0
        node._check_tf_alive()
        assert node._tf_rebuilds == 1
        assert node._unregistered == 1

    def test_rebuild_does_not_block(self, node):
        """Verifies the diagnostics and rebuild stay off the blocking path: this node
        runs on a single-threaded rclpy.spin, so a blocking can_transform timeout here
        would freeze TF, /map and the 1 Hz timer, the hazard already removed from
        wait_for_server."""
        import time
        self._prepare(node, timeout=30.0)
        node._get_robot_world_pos()
        node._tf_fail_since -= 31.0
        t0 = time.perf_counter()
        node._check_tf_alive()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"TF recovery blocked for {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Watchdog arming, the send-to-accept window that produced the observed hang
# ---------------------------------------------------------------------------

class TestWatchdogArming:
    """The watchdog is armed by _last_nav_progress_t, and it used to be set only on goal
    ACCEPTANCE. send_goal_async's future never fails when the server disappears (rclpy
    has no set_exception), so a nav2 destroyed between the send and the acceptance
    callback left the clock at None and _check_nav2_alive returning at its first line
    forever. The node parked in NAVIGATING with the process alive and no output.

    Observed in the field: the last log line was "Navigating to waypoint 1", the send,
    with nothing after it, and the node sat there for 10+ minutes.
    """

    def _prepare(self, node, server_ready=False):
        node._nav2_watchdog_enabled = True
        node._nav2_stall_timeout_s = 30.0
        node._teleop_enabled = False
        node._nav_client = SimpleNamespace(server_is_ready=lambda: server_ready)
        node._spin_client = SimpleNamespace(server_is_ready=lambda: server_ready)
        node._current_goal_xy = (10.0, 10.0)
        node._arrival_tolerance_m = 0.4
        node._arrival_verify_timeout_s = 30.0
        node._get_robot_world_pos = lambda: (99.0, 99.0)            # type: ignore
        node._get_robot_heading_deg = lambda: 0.0                   # type: ignore
        node._session = SimpleNamespace(
            visited_candidates=set(),
            on_nav_aborted=lambda wp: False,
            on_arrive_clear_aborts=lambda wp: None,
        )
        node._nav2_trips = 0
        node._nav2_rebuilds = 0

    def test_server_dies_between_send_and_accept_still_trips(self, node):
        """THE regression test for the observed hang. The goal is sent and the server
        dies before the acceptance callback ever fires, so nothing after the send runs.
        The watchdog must still trip; before arming at send it could not, because its
        clock was only set by that callback."""
        self._prepare(node, server_ready=False)
        # Arm exactly as _send_nav_goal now does, then never deliver _on_goal_response.
        node._arm_action_watchdog(node._nav_client)
        node._state = _State.NAVIGATING
        node._last_nav_progress_t -= 31.0
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING, (
            "a nav2 destroyed in the send-to-accept window must not hang the node")

    def test_unarmed_watchdog_is_inert(self, node):
        """Verifies the arming flag still gates the watchdog: with no goal tracked at
        all it must stay silent, which is what keeps PLANNING and arrival observation
        from tripping it."""
        self._prepare(node, server_ready=False)
        node._last_nav_progress_t = None
        node._state = _State.NAVIGATING
        node._check_nav2_alive()
        assert node._state == _State.NAVIGATING

    def test_watchdog_probes_the_client_owning_the_goal(self, node):
        """Verifies a spin-phase death is detected: during a rotation the in-flight goal
        belongs to the SPIN server, so probing the nav client would miss it entirely."""
        self._prepare(node, server_ready=True)
        node._spin_client = SimpleNamespace(server_is_ready=lambda: False)  # spin died
        node._arm_action_watchdog(node._spin_client)
        node._state = _State.ROTATING
        node._check_nav2_alive()
        assert node._state == _State.VERIFYING

    def test_rotating_is_watchdog_covered_in_the_timer(self, node):
        """Verifies ROTATING is no longer an uncovered state: an arrival spin runs after
        every waypoint in legacy mode, and a nav2 restart during one hung the same way."""
        self._prepare(node, server_ready=False)
        node._arm_action_watchdog(node._spin_client)
        node._last_nav_progress_t -= 31.0
        node._state = _State.ROTATING
        node._publish_status = lambda: None                        # type: ignore
        node._publish_mask_overlays = lambda: None                 # type: ignore
        node._do_rotation = lambda: pytest.fail(                   # type: ignore
            "must not drive a stale spin after the watchdog tripped")
        node._exploration_timer()
        assert node._state == _State.VERIFYING

    def test_repeated_trips_rebuild_the_action_clients(self, node):
        """Verifies escalation: when re-discovery alone does not restore the binding,
        the clients are destroyed and recreated, mirroring the TF listener rebuild."""
        self._prepare(node, server_ready=False)
        rebuilt = []
        node._rebuild_action_clients = lambda: rebuilt.append(1)   # type: ignore
        for _ in range(2):
            node._arm_action_watchdog(node._nav_client)
            node._state = _State.NAVIGATING
            node._check_nav2_alive()
        assert node._nav2_trips == 2
        assert len(rebuilt) == 1                  # only on the 2nd consecutive trip

    def test_acceptance_resets_the_trip_streak(self, node):
        """Verifies unrelated failures spread over a long run never accumulate into a
        rebuild: a server that accepts a goal is demonstrably alive."""
        self._prepare(node, server_ready=False)
        node._arm_action_watchdog(node._nav_client)
        node._state = _State.NAVIGATING
        node._check_nav2_alive()
        assert node._nav2_trips == 1
        node._on_goal_response(SimpleNamespace(
            result=lambda: SimpleNamespace(accepted=True,
                                           get_result_async=lambda: SimpleNamespace(
                                               add_done_callback=lambda cb: None))))
        assert node._nav2_trips == 0


# ---------------------------------------------------------------------------
# Operator teleop mode
# ---------------------------------------------------------------------------

class TestTeleopMode:
    """A latched Bool on /exploration/teleop_enabled hands the robot to a human. While
    on, the planner keeps running and coverage keeps being marked, but no nav2 goals are
    sent, and driving to within arrival_tolerance_m of the waypoint counts as reaching
    it. Turning it off replans from wherever the operator parked.

    The flag is state, not a heartbeat: publishing once is enough, and a repeat of the
    value already held is ignored.
    """

    def _prepare(self, node):
        node._teleop_enabled = False
        node._arrival_tolerance_m = 0.4
        node._current_goal_xy = (10.0, 10.0)
        node._see_while_moving = False
        node._cancels = 0
        node._goal_handle = SimpleNamespace(
            cancel_goal_async=lambda: setattr(node, '_cancels', node._cancels + 1))
        node._session = SimpleNamespace(
            visited_candidates=set(),
            on_nav_aborted=lambda wp: False,
            on_arrive_clear_aborts=lambda wp: None,
        )
        node._get_robot_world_pos = lambda: (99.0, 99.0)            # type: ignore
        node._get_robot_heading_deg = lambda: 0.0                   # type: ignore
        node._state = _State.TRAVELING

    def test_enabling_cancels_the_in_flight_goal_once(self, node):
        """Verifies control is handed over immediately, so nav2 and the joystick never
        fight over cmd_vel."""
        self._prepare(node)
        node._on_teleop_enabled(SimpleNamespace(data=True))
        assert node._teleop_enabled is True
        assert node._cancels == 1
        assert node._goal_handle is None
        assert node._last_nav_progress_t is None   # watchdog disarmed, silence expected

    def test_repeated_true_is_ignored(self, node):
        """Verifies the one-shot / edge-triggered semantics: publishing the value that is
        already held does nothing, so a latched sample replayed to a late subscriber
        cannot re-cancel or re-log."""
        self._prepare(node)
        node._on_teleop_enabled(SimpleNamespace(data=True))
        assert node._cancels == 1
        node._goal_handle = SimpleNamespace(
            cancel_goal_async=lambda: setattr(node, '_cancels', node._cancels + 1))
        node._on_teleop_enabled(SimpleNamespace(data=True))   # same value again
        assert node._cancels == 1                            # no second cancel
        assert node._goal_handle is not None                 # untouched

    def test_no_goals_are_sent_while_enabled(self, node):
        """Verifies the suppression choke point: the planner may keep producing
        waypoints, but none of them reach nav2."""
        self._prepare(node)
        node._teleop_enabled = True
        sent = []
        node._nav_client = SimpleNamespace(
            server_is_ready=lambda: True,
            send_goal_async=lambda *a, **k: sent.append(1))
        node._publish_current_goal = lambda wp: None               # type: ignore
        node._send_nav_goal = ExplorationNode._send_nav_goal.__get__(node)  # type: ignore
        node._send_nav_goal(node._waypoints[0])
        assert sent == []

    def test_driving_to_the_waypoint_counts_as_arrival(self, node):
        """Verifies the core teleop behaviour: reaching the goal by hand credits the
        waypoint, using the same rule as the arrival failsafe."""
        self._prepare(node)
        node._teleop_enabled = True
        node._arrivals = 0
        node._see_while_moving = True          # routes to the stubbed _arrive_no_spin
        node._get_robot_world_pos = lambda: (10.2, 10.0)           # type: ignore
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        node._do_teleop_check()
        assert node._arrivals == 1

    def test_orientation_is_ignored_while_driving(self, node):
        """Verifies yaw is fully tolerated in teleop, matching the failsafe: a human
        parks facing any direction."""
        self._prepare(node)
        node._teleop_enabled = True
        node._arrivals = 0
        node._see_while_moving = True
        node._get_robot_world_pos = lambda: (10.0, 10.1)           # type: ignore
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        node._get_robot_heading_deg = lambda: 179.0                # type: ignore
        node._do_teleop_check()
        assert node._arrivals == 1

    def test_sitting_far_away_never_times_out(self, node):
        """Verifies the failsafe's stillness timeout does not leak into teleop: the
        operator decides when to move on, so parking away from the goal is not failure."""
        self._prepare(node)
        node._teleop_enabled = True
        node._arrivals = 0
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        for _ in range(100):                    # far away for a long time
            node._do_teleop_check()
        assert node._arrivals == 0
        assert node._wp_index == 0              # never given up on
        assert node._state == _State.TRAVELING

    def test_disabling_replans_from_the_current_position(self, node):
        """Verifies the handback: exploration resumes from wherever the robot was
        parked, which is also the manual replan-from-terminal command."""
        self._prepare(node)
        node._on_teleop_enabled(SimpleNamespace(data=True))
        node._on_teleop_enabled(SimpleNamespace(data=False))
        assert node._teleop_enabled is False
        assert node._state == _State.PLANNING
        assert node._sent_goals == []           # replans rather than resending blindly

    def test_repeated_false_does_not_replan_again(self, node):
        """Verifies edge-triggering in the other direction too."""
        self._prepare(node)
        node._on_teleop_enabled(SimpleNamespace(data=False))   # already disabled
        assert node._state == _State.TRAVELING                 # untouched, no replan

    def test_one_drive_credits_only_one_waypoint(self, node):
        """Regression: reaching waypoint 0 credited the ENTIRE plan from one spot.

        Observed in RViz as every waypoint marker turning green after driving to the
        first one. Two causes: _current_goal_xy was only written in the non-teleop half
        of _send_nav_goal, so it stayed pinned to the last goal actually sent to nav2;
        and accepting an arrival advances to a waypoint that may already be inside the
        tolerance circle. Together they credited a waypoint per 1 Hz tick without the
        robot moving at all.
        """
        self._prepare(node)
        node._teleop_enabled = True
        node._arrivals = 0
        node._see_while_moving = True
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        node._teleop_left_last_goal = True
        # Parked exactly on waypoint 0's recorded goal, and never moving again.
        node._get_robot_world_pos = lambda: (10.0, 10.0)           # type: ignore
        for _ in range(10):
            node._do_teleop_check()
        assert node._arrivals == 1, (
            f"one drive must credit one waypoint, got {node._arrivals}")

    def test_leaving_the_circle_re_arms_the_next_arrival(self, node):
        """Verifies the latch is not a one-shot: after driving away and on to the next
        waypoint, that arrival counts too."""
        self._prepare(node)
        node._teleop_enabled = True
        node._arrivals = 0
        node._see_while_moving = True
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        node._teleop_left_last_goal = True
        node._get_robot_world_pos = lambda: (10.0, 10.0)           # type: ignore
        node._do_teleop_check()
        assert node._arrivals == 1
        # Drive away (outside tolerance), which re-arms.
        node._get_robot_world_pos = lambda: (50.0, 50.0)           # type: ignore
        node._do_teleop_check()
        # New waypoint target recorded there, and the driver arrives.
        node._current_goal_xy = (50.0, 50.0)
        node._do_teleop_check()
        assert node._arrivals == 2

    def test_taking_over_on_a_goal_does_not_credit_it(self, node):
        """Verifies enabling teleop while parked on the current goal is not a free
        arrival: the operator must deliberately drive somewhere."""
        self._prepare(node)
        node._arrivals = 0
        node._see_while_moving = True
        node._get_robot_pixel_pos = lambda: None                   # type: ignore
        node._on_teleop_enabled(SimpleNamespace(data=True))
        node._get_robot_world_pos = lambda: (10.0, 10.0)           # type: ignore  on the goal
        node._do_teleop_check()
        assert node._arrivals == 0

    def test_watchdog_never_trips_while_teleop_is_on(self, node):
        """Verifies the interaction: with goal sending suppressed, nav2 being quiet is
        the expected condition and must not be reported as a fault."""
        self._prepare(node)
        node._teleop_enabled = True
        node._nav2_watchdog_enabled = True
        node._nav2_stall_timeout_s = 30.0
        node._nav_client = SimpleNamespace(server_is_ready=lambda: False)
        node._active_client = node._nav_client
        node._last_nav_progress_t = node.get_clock().now().nanoseconds / 1e9 - 999.0
        node._check_nav2_alive()
        assert node._state == _State.TRAVELING
