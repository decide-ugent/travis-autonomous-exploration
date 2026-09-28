# Exploration Package

This package implements autonomous map exploration for TRAVIS. Given a 2D occupancy grid, it plans the robot's waypoints to maximise coverage of the navigable area, orchestrates the observation rotation at each waypoint, and manages all state that must persist across planning cycles (visited positions, SLAM candidate growth, mid-path replanning). The package is designed to be called by the ROS2 navigation node, but all core logic is pure Python and can be exercised offline through the demo scripts and test suite.

---

## Architecture overview

```
exploration/          ← core logic
    explore_costmap_map.py      planning engine (pure Python, no ROS)
    execution_strategy.py       session state across plan cycles (pure Python, no ROS)
    rotation_strategy.py        heading order + rotation stop (pure Python, no ROS)
    ros2_exploration_node.py    ROS2 node: state machine + Nav2 action clients
    rviz_visualisation.py       RViz marker/overlay builders

launch/
    exploration_nav2.launch.py  full stack (Nav2 + panoramic lidar + exploration + RViz)
    exploration.launch.py       exploration node only (+ RViz)
    manual_exploration.launch.py human-driven baseline run (coverage node + RViz)
    rviz.launch.py              RViz alone, included by the others (use_rviz:=false to skip)

tests/                ← demos, evaluation tools, pytest tests
    demo_robot.py
    visual_demo_coverage.py / visual_demo_semantic.py / visual_demo_slam.py
    annotate_ideal.py
    evaluate_waypoint_positions.py
    conftest.py
    test_navigation_exploration_explore_costmap_map.py
    test_navigation_exploration_execution_strategy.py
    test_navigation_exploration_rotation_strategy.py
    test_navigation_exploration_node_status_handling.py
    test_integration_navigation_exploration_explore_costmap_map.py

    algo_evaluation/  ← offline A/B harnesses that justify the current defaults logic choices
        benchmark_waypoint_scoring.py, scoring_variants.py
        test_see_while_moving.py, test_heading_travel.py, test_replan_N.py
        test_midpath_replan_ratio.py, test_FOV_visited_candidates.py

    system/           ← L3 end-to-end simulator runs (see tests/system/README.md)
        recorder.py, manual_exploration.py, evaluate_run.py
        visualize_run.py, aggregate_runs.py, resume_test.py, baselines/

visualisation/        ← matplotlib visualisers (demo only, not imported by ROS2)
    exploration_visualiser.py
    semantic_exploration_visualiser.py
    visualise_system_steps.py    step-by-step visualisation of the selection pipeline (writes runs here)
```

**Three test levels.** `tests/*.py` are L1/L2 (unit + offline integration, no ROS, run in seconds). `tests/algo_evaluation/` are offline A/B benchmarks used to pick parameter values, not pass/fail gates. `tests/system/` is L3: the real stack in a simulator, recorded then evaluated offline against a per-scene baseline.

---

## Core logic files (`exploration/`)

### `explore_costmap_map.py`

The planning engine. This is the largest and most important file in the package. Reading it in order follows the full data pipeline from map loading to ordered waypoints.

**What it does, step by step:**

1. **Map loading** - `load_map(pgm, yaml, inflation_m)` reads a standard ROS2 PGM+YAML map, applies wall inflation (using a Euclidean distance transform), and returns a `MapData` dataclass holding five boolean masks: `free_mask`, `occupied_mask`, `unknown_mask`, `navigable_mask`, and `covered_mask`. `navigable_mask` is the subset of free cells that are far enough from walls for the robot body to fit. `covered_mask` starts all-False and accumulates cells the camera has observed.

2. **Candidate generation** - `generate_candidates(navigable_mask, step_px)` places a regular grid of candidate waypoints across all navigable cells. The grid spacing is `sampling_step_m / resolution` pixels, matching the robot's detection range so candidates tile the space without gaps.

3. **Visibility computation** - `compute_visibility(candidate, map_data, max_range_px, num_rays)` casts rays from a candidate position (simulating a full 360° rotation of the robot). It returns two sets: `coverage_cells` (free cells that the camera would observe) and `frontier_cells` (the first unknown cell on each ray - relevant for SLAM). `compute_all_visibility` runs this for all candidates.

4. **Greedy set cover** - `greedy_set_cover(candidates, visibility, covered_mask, ...)` selects a minimal subset of candidates that collectively covers the uncovered area. It iterates: pick the candidate with the highest weighted score (frontier gain x alpha + coverage gain x beta - travel cost x Y), mark its cells as covered in a temporary mask, repeat. This is a classic greedy approximation to the NP-hard weighted set cover problem.

5. **Heading computation** - `compute_headings_for_waypoint(candidate, coverage_cells, fov_deg, increment_deg)` finds the minimum set of camera headings that covers all cells visible from a waypoint. This avoids a full 360° sweep at every position.

6. **TSP ordering** - `nearest_neighbor_order(waypoints, map_data, ...)` orders the selected waypoints using a nearest-neighbour heuristic on BFS (navigable-path) distance, not Euclidean distance. This prevents the robot from being sent across a wall.

7. **Top-level planner** - `plan_waypoints(map_data, config, robot_x, robot_y, visited_candidates)` runs the full pipeline (2-6 above) and returns an ordered `Waypoint` list, the current coverage ratio, a `no_frontiers` flag, and per-candidate records for logging.

**Key supporting functions:**

| Function | Purpose |
|---|---|
| `build_map_data(...)` | ROS2 bridge: constructs `MapData` from a `/map` occupancy grid message |
| `reproject_covered_mask(...)` | Re-anchors `covered_mask` onto a resized/shifted SLAM grid, so coverage survives a map growth instead of resetting |
| `update_covered_mask(map_data, col, row, heading, fov_deg, ...)` | Simulates one camera shot; marks newly observed cells in `covered_mask`; returns updated coverage ratio |
| `navigable_distance_map(navigable_mask, start_col, start_row, inflation_px)` | Wall-aware geodesic distance from one position to all reachable cells; used for travel-cost scoring and mid-path replanning |
| `coverage_ratio(covered_mask, achievable_cells)` | Fraction of achievable cells that have been covered so far |
| `pixel_to_world / world_to_pixel` | Coordinate conversion between pixel and ROS2 world metres |


---

### `execution_strategy.py`

The session manager. Owns all state that must survive across planning cycles and provides the interface the ROS2 node (or demo) calls at each step of the exploration loop.

**Why it exists:** the planning logic in `explore_costmap_map.py` is stateless, it takes a map and a config and returns waypoints. Anything that must persist (which positions have been visited, where the robot was last marked, the mid-path replanner instance) lives here instead.

**`MidPathReplanner`**

A lightweight helper instantiated fresh for each planned path. At each pixel step during travel, `check(step, current_target, all_waypoints, navigable_mask, inflation_px)` runs `navigable_distance_map` from the robot's current position and returns `True` if any other waypoint is reachable for less than `mid_path_replan_ratio` of the cost to the current target. This catches detours: if the robot has turned a corner and is now closer to a different waypoint, it replans immediately rather than completing a suboptimal path. It also returns `True` when the current target has become unreachable (distance `inf`).

The ratio is the `mid_path_replan_ratio` parameter, **default 0.75**, 0.75 was chosen with `tests/algo_evaluation/test_midpath_replan_ratio.py`: it gave a shorter path at held coverage on all three test maps, and 0.0 (never divert) gave the longest path on all of them.

`check` fires at most once every `check_interval_px` pixels (set to `visit_radius_px` so the cadence matches candidate marking). The very first call always returns `False` - there is no prior position to measure displacement from.

**`ExplorationSession`**

The main class. Instantiate once per exploration run; call its methods in the loop below.

```
session = ExplorationSession(map_data, config)
robot_col, robot_row = session.nearest_start(cx, cy)

while True:
    waypoints, ratio, no_frontiers, _ = session.plan_waypoints_raw(robot_x, robot_y)
    if no_frontiers and ratio >= config["exploration_completion_threshold"]:
        break                          # exploration complete
    if not waypoints:
        session.clear_unreachable()    # pool starved: retry the parked waypoints
        continue

    wp = waypoints[0]
    path = find_path(navigable_mask, (robot_col, robot_row), (wp.col, wp.row))

    if session.on_unreachable(wp, path):
        continue                       # no path - skip without marking visited

    mid_replan = False
    for step in path[1:]:
        if session.on_step(step, wp, waypoints):
            robot_col, robot_row = step
            mid_replan = True
            break

    if not mid_replan:
        # rotate and observe at wp ...
        session.on_arrive(wp)
        robot_col, robot_row = wp.col, wp.row
```

| Method | When to call | What it does |
|---|---|---|
| `set_map(map_data)` | At construction, and on every new `/map` | Rebinds the map and regenerates candidates from the new `navigable_mask`. The single point where SLAM growth enters the candidate pool |
| `nearest_start(cx, cy)` | Once, before the loop | Returns the navigable candidate nearest to a pixel position |
| `plan_waypoints_raw(robot_x, robot_y)` | At the top of every loop iteration | Calls `plan_waypoints` excluding `visited_candidates ∪ _unreachable`, arms a fresh `MidPathReplanner`, returns `(waypoints, ratio, no_frontiers, records)` |
| `on_unreachable(wp, path)` | After `find_path`, before entering the travel loop | Returns `True` if no path exists - caller should `continue` **without** marking visited |
| `on_nav_aborted(wp)` | On every Nav2 ABORT | Counts aborts per waypoint pixel; blacklists into `_unreachable` after `abort_blacklist_after` (default 3). Returns `True` on the call that blacklists |
| `on_arrive_clear_aborts(wp)` | On Nav2 SUCCESS | Clears that one waypoint's abort count (arrival proves it reachable). Deliberately does **not** restore the whole `_unreachable` set |
| `clear_unreachable()` | Only when the candidate pool is starved | Restores every parked-unreachable waypoint; returns how many, so the recovery can be logged |
| `on_step(step, wp, waypoints)` | At every pixel step during travel | Runs the `MidPathReplanner` check; returns `True` to trigger a mid-path replan |
| `on_arrive(wp)` | On arrival, when not mid-replanning | Marks the waypoint and any now-observed candidate visited |

**Two separate sets, this distinction is the point.**

- `visited_candidates`, viewpoints the camera **genuinely observed**. Permanent and monotonic.
- `_unreachable`, viewpoints **Nav2 could not drive to**. The camera never saw these, so the area is still *uncovered*. Transient by design.

Both are excluded from planning (their union is passed to `plan_waypoints`), but they are *not* the same fact. Recording an abort as "visited" tells the planner the area is done and **permanently freezes coverage**, that was the cause of the hospital known-map stall at 56.6%.

`_unreachable` is restored **only** when the candidate pool would otherwise be starved (`clear_unreachable`, called from the node's `_handle_empty_plan`). Restoring it on every arrival livelocks: a permanently unreachable frontier, one inside a wall, or a room behind a door the robot cannot fit through, is restored, instantly re-picked as the highest-gain waypoint 0, aborts, is re-parked, forever. Because plans never go empty in that loop, no timeout can catch it. Restricting the restore to starvation keeps it strictly beneficial: it only runs when the alternative is doing nothing, and if the restored waypoints abort again the pool starves again, which *is* observable, so the node's escalation still bounds the loop.

**Visited-candidate marking is FOV-aware.** `_mark(pos)` no longer uses a Euclidean disc, that disc was wall-blind and sterilised the candidate pool, terminating exploration at ~13.6% coverage on `lab_ghent`. A candidate is now marked visited once the camera has actually observed ≥ `OBSERVED_FRACTION_THRESHOLD` (0.95) of the cells that viewpoint could ever see, computed with two `compute_visibility` casts: the remaining-uncovered cells, and the full footprint via `ignore_covered=True`. A fraction rather than "all cells" keeps a residual sliver behind an occluder from keeping a saturated viewpoint alive forever.

The full-footprint cast **must** use the `ignore_covered` flag, not a temporary swap of `map_data.covered_mask`: an exception mid-swap leaves an empty mask installed on the shared `MapData` and silently resets all coverage (observed in the 2026-07-16 warehouse run).

---

### `rotation_strategy.py`

Decides which headings to rotate through at each waypoint and when to stop early.

**`get_headings(wp, observation_increment, current_heading)`**
Returns the ordered list of headings for one waypoint. Uses `wp.headings` (the planner's pre-computed minimum coverage set, typically 2–6 headings) when available; falls back to a full 360° sweep in `observation_increment` steps (typically 12 headings at 30°).

Headings are ordered to **minimise total angular travel** from `current_heading`, not simply sorted clockwise. `compute_headings_for_waypoint` minimises the *number* of headings but ignores how far the robot physically turns between them, which is the real cost in time and odometry drift. `_min_travel_order` picks the cheaper sweep direction, which cut travel by 12.6% on `lab_ghent` (`tests/algo_evaluation/test_heading_travel.py`).

Note this only applies on the legacy turn-at-goal path; with `see_while_moving` on (the default) there is no per-waypoint rotation to order.

**`RotationState`**
Tracks per-heading coverage gain. Call `update(ratio)` after every `update_covered_mask` call. Returns `True` while rotation should continue; returns `False` once the area has been saturated (no new cells covered for `stop_after=3` consecutive headings).

```python
_rot_state = RotationState(stop_after=3)
for h in get_headings(wp, OBSERVATION_INCREMENT, heading):
    heading = float(h)
    ratio = update_covered_mask(md, wp.col, wp.row, heading, ...)
    if not _rot_state.update(ratio):
        break   # saturated - stop rotating
```

---

### `ros2_exploration_node.py`

The ROS2 node. Wraps `ExplorationSession` in a 1 Hz timer-driven state machine and drives the robot through the Nav2 `NavigateToPose` and `Spin` actions. This is the only file in `exploration/` that imports ROS.

```
WAITING_FOR_MAP → PLANNING → NAVIGATING → TRAVELING → ROTATING ─┐
                     ↑             │          │                 │
                     │             └──────────┴→ VERIFYING ─────┤
                     └──────────────────────────────────────────┘
                     └→ COMPLETE  (coverage reached, no-progress stop, or wedged)
```

`ROTATING` is skipped entirely when `see_while_moving` is on (the default).

`VERIFYING` is entered when Nav2 gives up on a waypoint, or when the watchdog finds the goal has gone silent. No goal is in flight in this state: the node is deciding from the TF pose whether the robot actually arrived, then leaves for `NAVIGATING` (waypoint credited) or `PLANNING` (waypoint failed). It is also a short time window during which a human can drive the robot the last stretch to the goal, see [Arrival failsafe](#arrival-failsafe-a-nav2-failure-is-not-proof-of-non-arrival).



**Topics.**

| Topic | Dir | Type | Purpose |
|---|---|---|---|
| `/map` | sub | `OccupancyGrid` | live SLAM map, or `map_server` in known-map mode |
| `/exploration/teleop_enabled` | sub | `std_msgs/Bool` | hand the robot to a human operator, see [Manual assisted exploration](#manual-assisted-exploration) |
| `/exploration/status` | pub | `String` | node state, including the current teleop flag |

The node also publishes coverage and RViz visualisation topics (`/exploration/coverage`, the covered/nav masks, waypoint and FOV markers).

**Map input.** Subscribes to `/map` (live SLAM via slam_toolbox, or `map_server` in known-map mode), or loads a static PGM when `map_file_path` is set. `_is_slam` distinguishes the two and gates several behaviours below. On a SLAM map resize the node calls `reproject_covered_mask` so accumulated coverage is re-anchored rather than reset, and `world_to_pixel` refreshes stale waypoint pixel coords.

#### See-while-moving (default: on)

The node used to travel heading-blind and then spin in place at each goal. With `see_while_moving: true` it instead:

1. aims the Nav2 goal **yaw** at the most-uncovered direction (`_best_uncovered_heading`), so the robot arrives already facing what it still needs to see; and
2. marks the camera FOV **continuously along the path**, every `observe_step_m` metres (default 0.5 m) of travel.

On arrival it takes that single look (`_arrive_no_spin`) and moves on, no `Spin` action at all. `tests/algo_evaluation/ test_see_while_moving.py` found this reaches similar or higher coverage than turn-at-goal while skipping every spin, with the
largest gain under SLAM.

> Requires Nav2 `yaw_goal_tolerance` bwith a sensible value, otherwise Nav2 will not actually end the robot facing the requested heading and the aimed look is wasted (it's fine but a pity)

Setting `see_while_moving: false` restores the legacy turn-at-goal behaviour (`ROTATING` state, one `Spin` action per heading from `get_headings`).

#### Failure handling and stop conditions

This is where most of the node's complexity lives; each guard exists because of an observed failure.

| Situation | Node behaviour |
|---|---|
| Nav2 **ABORT** on a waypoint | Enter `VERIFYING` first: Nav2 giving up does not prove the robot is not at the goal. Only if the arrival failsafe cannot rescue it does the node try the *next* waypoint (`_try_next_waypoint`). The failed spot stays **unvisited** so a later pose can still reach it |
| Same waypoint aborts `abort_blacklist_after` times | Blacklisted into `_unreachable`, breaking the livelock where the same in-wall frontier is re-picked as waypoint 0 on every re-plan. Only aborts the arrival failsafe could **not** rescue are counted |
| Nav2 **disappears mid-goal** (killed, restarted, crashed) | `rclpy` never fails a pending result future when its action server vanishes, so the node would wait in `TRAVELING` forever with no result at all. The watchdog bounds that wait using NavigateToPose feedback as a heartbeat, hands the waypoint to the arrival failsafe, and lets the next goal send re-link to the restarted server. After two consecutive trips the action clients are destroyed and recreated |
| `map -> base_frame` stops resolving after a container restart | Latched `/tf_static` samples are delivered only once on subscription match, so a listener already matched never receives the replacement publishers and the transform breaks permanently. Static transforms are never republished, so this cannot self-heal: `_check_tf_alive` rebuilds the `TransformListener` |
| Nav2 **CANCEL** | Ignored. A cancel is *our own* mid-path replan, not an inaccessible waypoint. Must not advance the index or count toward the stuck streak, this also closes the race where a stale cancel lands after a new goal was sent |
| Every waypoint in the plan inaccessible | `_on_plan_exhausted_failed`: streak ≥ 2 → recovery `Spin` to regain clearance; streak ≥ 5 **and** stuck longer than `plan_timeout_s` → stop |
| Plan came back **empty** | `_handle_empty_plan`: after ~5 s, `clear_unreachable()` restores parked waypoints and planning retries with a full pool |
| Robot parked in the costmap inflation band | Nav2 rejects every goal in ~30 ms, so the streak cap alone fired within *seconds*. The `plan_timeout_s` time gate (default 300 s) is what makes this survivable, it holds the run open long enough for an operator to free the robot, logging a warning throughout |

#### Arrival failsafe: a Nav2 failure is not proof of non-arrival

Nav2 reporting ABORTED does not mean the waypoint was missed. The robot often stops just outside Nav2's own goal checker, and a human may drive it the last stretch. Rather than trusting Nav2's verdict, `_do_verify_check` holds the node in `VERIFYING` and decides from the **TF pose** (`map -> base_frame`, the frame the goal is in). Odometry is deliberately not used: it drifts under SLAM and lives in another frame.

- Within `arrival_tolerance_m` of the goal, **orientation ignored entirely**, the waypoint is credited immediately.
- Otherwise the node waits until the robot has been **still** for `arrival_verify_timeout_s` before declaring the waypoint failed.

That second point is a *stillness* timeout, not a wall clock. Any movement beyond `arrival_motion_eps_m` restarts it, which opens **a short time window for a human to drive the robot to the goal without touching any topic**: keep moving and the window keeps extending, arrive and the waypoint counts as reached. Coverage is still marked while this happens, so anything the camera sees during the window is kept. This is distinct from [manual assisted exploration](#manual-assisted-exploration) below, which is an explicit mode with no timeout.

`arrival_motion_eps_m` separates real motion from TF/SLAM jitter: too low and estimator noise alone holds the window open on a parked robot, too high and slow manual driving is mistaken for standing still.

Set `arrival_verify_timeout_s` to `0` to disable the failsafe entirely and restore the immediate-abort behaviour.

#### Manual assisted exploration

A human can take the robot at any time, without killing the run, by publishing to `/exploration/teleop_enabled`:

```bash
# take control: exploration stops sending Nav2 goals, you drive
ros2 topic pub --once /exploration/teleop_enabled std_msgs/Bool "{data: true}"

# give control back: exploration replans from wherever you parked the robot
ros2 topic pub --once /exploration/teleop_enabled std_msgs/Bool "{data: false}"
```

Exploration still runs underneath: the node keeps planning and keeps **marking camera coverage as you drive**, so nothing observed by hand is lost. Driving to within `arrival_tolerance_m` of the current waypoint (orientation ignored, the same rule `_within_arrival_tolerance` applies to the failsafe, so the two can never disagree) counts it as reached and the plan moves on by itself. Publishing `false` cancels the rest of the current plan and replans from the robot's current position via the existing `force_replan` path.

Unlike the failsafe window above, this mode has **no timeout**: `_do_teleop_check` never gives up on a waypoint, because in teleop the operator decides when to move on. The switch is **edge-triggered state, not a heartbeat**: publish once and the mode persists with no republishing, and re-publishing a value already held is ignored.

> The subscription is deliberately VOLATILE, not transient-local. A TRANSIENT_LOCAL subscription is the one durability combination that refuses to match a VOLATILE publisher, and `ros2 topic pub` is VOLATILE unless told otherwise, so latching here would leave the one-liner above waiting for a matching subscription forever.

#### "Stuck", and why the node waits for a human

The node **does not relocalise or self-rescue** a wedged robot lost with nav2. There is no pose reset, no AMCL re-initialisation, no costmap-clearing beyond Nav2's own behaviour tree. When it decides the robot is stuck, its only recovery is a `Spin` to try to regain clearance; if that fails, it holds the run open and waits for a human, then resumes on its own once the pose is plannable again. The human can simply free the robot and let it continue, or take over deliberately with [manual assisted exploration](#manual-assisted-exploration).

"Stuck" is one of two concrete conditions, both measured by the same `plan_timeout_s` clock (default 300 s in sim-seconds):

1. **Every waypoint in the plan is inaccessible.** Nav2 aborts on all of them, so `_on_plan_exhausted_failed` runs. On the 2nd consecutive fully-failed plan it issues a recovery `Spin`; it only *stops* once the failure streak has held for longer than `plan_timeout_s`.
2. **The robot is sitting inside the costmap inflation band.** Nav2 cannot plan *from* a start pose it considers occupied, so it rejects every goal in ~30 ms. Here the node deliberately sends **nothing** until the pose is plannable again (spamming doomed goals is pointless), and tracks how long the pose has been un-plannable with the same clock.

Both paths need the time gate for the same reason: the failure *count* races to its cap in seconds (a 20-waypoint plan burns through in under a second when every goal is rejected instantly), so the count alone cannot tell "briefly clipped an obstacle" from "genuinely wedged". Only elapsed wall-clock (sim-clock) time can. Throughout the wait the node logs a warning roughly every 30 s so an operator knows intervention is needed.

What clears the stuck state:

- **A human frees the robot.** Once the pose is plannable again, the next plan cycle sends goals normally; a single successful arrival clears the failure streak and resets the stuck clock, so the timer always measures the *current* episode, not a past one.
- **The timeout elapses first.** Exploration stops with a clear log line. This is a real stop, not a crash: it is the node giving up after nobody intervened.

`plan_timeout_s` is the knob for how long to wait. **Set it to `0` to never give up** (retry forever, useful for unattended runs where no human will come); raise it to give an operator a longer window. It is deliberately *not* zero by default so an unsupervised robot does not spin against a wall indefinitely.

**Completion.** Exploration stops when `no_frontiers` **and** coverage ≥ `exploration_completion_threshold`. Two SLAM-only relaxations keep the robot from chasing the last unreachable wisp forever:

- `min_frontier_cells`, treat "no frontiers" as *total visible frontier cells ≤ this*, rather than exactly zero, absorbing phantom slivers.
- `no_progress_streak` / `no_progress_eps` , also stop after N consecutive arrivals that each add < 0.5% new coverage, catching persistently unreachable frontiers the tolerance alone will not.

Both are gated to live SLAM: on a static known map "no progress" is the *expected* steady state once covered, so applying them there would stop the run early.

**Replan cadence.** Under live SLAM the node returns to `PLANNING` every `replan_every_n_step` arrivals (2) instead of draining a plan whose later waypoints were scored against a map several arrivals stale. A static map always drains the whole plan.

#### Parameters for the failsafe, teleop and watchdog

Values live in `config/exploration_system_parameters.yaml`, documented inline; the table says what each one governs.

| Parameter | Governs |
|---|---|
| `exploration.arrival_tolerance_m` | how close counts as arrived, orientation ignored. Shared by the failsafe and teleop so they cannot disagree |
| `exploration.arrival_verify_timeout_s` | how long the robot must be **still** before a failed waypoint is given up. Set to `0` to disable the failsafe |
| `exploration.arrival_motion_eps_m` | pose change per tick that counts as real motion rather than TF/SLAM jitter |
| `exploration.abort_blacklist_after` | consecutive unrescued aborts before a waypoint is blacklisted |
| `nav2.watchdog_enabled` | master switch for **both** the Nav2 action watchdog and the TF listener recovery. `false` restores the old hang-until-killed behaviour |
| `nav2.watchdog_stall_timeout_s` | silence before a still-advertising Nav2 is presumed wedged; doubles as how long `map -> base_frame` must fail before the TF listener is rebuilt |

---

## Test and demo files (`tests/`)

### `demo_robot.py`

Shared utilities used by all three visual demo scripts. Import this, not matplotlib or ROS2. Contains:

- **`SceneObject`** - dataclass for a detectable object (label, pixel position, world position, per-label detection range).
- **`place_objects(navigable_mask, resolution, origin, n, seed)`** - randomly scatter N objects on navigable cells with reproducible seeding.
- **`find_path(navigable_mask, start, goal)`** - BFS shortest path on the navigable grid (8-directional). Returns `[start]` if the goal is unreachable.
- **`check_detections(occupied_mask, unknown_mask, resolution, robot_col, robot_row, heading_deg, fov_deg, objects)`** - determines which undetected objects fall within the camera frustum and line of sight at the current pose.
- **`reveal_cells(free_mask, occupied_mask, unknown_mask, navigable_mask, original_free, original_occ, robot_col, robot_row, reveal_range_px, num_rays, inflation_px)`** - SLAM simulation: casts rays from the robot, reveals unknown cells, and recomputes the inflated navigable mask when new cells appear.
- **`build_demo_config()`** - reads the per-package YAML files and assembles the planner config dict.

---

### `visual_demo_coverage.py`

Runs a complete simulated exploration on the lab map, coverage mode only (no objects, no SLAM). Produces a matplotlib animation showing the robot path, covered area, and waypoints in real time, plus CSV logs in `tests/visual_demo_coverage/coverage_demo_log_N/`.

**When to run:** after any change to the planner or session logic, to visually verify that the robot actually covers the map, does not oscillate, and terminates correctly.

```bash
python3 tests/visual_demo_coverage.py
```

---

### `visual_demo_semantic.py`

Same as coverage but adds randomly placed objects. The robot detects them as it rotates at each waypoint. The visualiser overlays detection events and object positions.

**When to run:** to verify that detection logic and the session loop work together - that objects are found in reasonable order without the robot being sent to already-detected positions.

```bash
python3 tests/visual_demo_semantic.py
```

---

### `visual_demo_slam.py`

Same as semantic but starts with only the left half of the map known. Unknown cells are revealed as the robot moves (`reveal_cells`), and the navigable mask is recomputed with inflation at each step. The planner regenerates candidates from the growing map each cycle.

**When to run:** to verify that SLAM-mode candidate regeneration works - the robot should eventually cross into the right half and cover it, not oscillate on the known side.

```bash
python3 tests/visual_demo_slam.py
```

---

### `annotate_ideal.py`

Post-run annotation tool. After running `visual_demo_coverage.py`, open this tool and click on the map to define the positions where a human expert would have placed waypoints. Saves `ideal_waypoints.csv` in the log directory for use by `evaluate_waypoint_positions.py`.

```bash
python3 tests/annotate_ideal.py                    # auto-detects latest log dir
python3 tests/annotate_ideal.py path/to/log_dir    # explicit
```

Controls: left-click to add, right-click / Z to undo, Enter or close to save.

---

### `evaluate_waypoint_positions.py`

Quantitative evaluator. Loads up to three `ideal_waypoints_N.csv` files, clusters them by consensus (positions confirmed by ≥ 2 annotators within 20 px), runs `plan_waypoints` from the map centre, and reports a hit-rate table: for each ideal position, the nearest actual waypoint and whether it is within 1 m (20 px at 0.05 m/px). Saves `evaluation_result.png`.

```bash
python3 tests/evaluate_waypoint_positions.py
```

Run this after tuning `frontier_weight`, `coverage_weight`, or `sampling_step_m` to measure whether the planner is selecting the right positions.

---

### `visualisation/visualise_system_steps.py`

Explains the *selection logic* step by step, for docs / slides / a live walkthrough.
Runs the real planner **once** on a map and renders each pipeline stage as a labelled PNG panel plus an animation (`exploration_steps.mp4`, via imageio's bundled ffmpeg; falls back to a GIF only when imageio is unavailable). All animated panels share one fixed frame size. (Lives under `visualisation/`; it still imports `demo_robot`/`conftest` from `tests/`.)

1. occupancy masks → 2. candidate viewpoints → 3. the score field (every candidate coloured by its score) → 4. the greedy set-cover picks, **three sub-frames per round** (score → pick with its 360° planning disc and an arrow from the previous pick → re-score) → 5. the geodesic nearest-neighbour tour vs the naive Euclidean order → 6. the final ordered plan.

Unlike the recorded `candidates.csv` (final-state scores only), this captures the per-iteration marginal score that actually drives each greedy pick, and writes it to `steps.csv`. The instrumented greedy loop is asserted equal to `greedy_set_cover`'s selection order and the ordered result equal to `plan_waypoints`', so the pictures cannot drift from production.

```bash
python3 visualisation/visualise_system_steps.py                    # reference lab map
python3 visualisation/visualise_system_steps.py --pgm m.pgm --yaml m.yaml
python3 visualisation/visualise_system_steps.py --no-anim          # PNGs + steps.csv only
```

Output lands in `visualisation/visualise_system_steps/run[_N]/`.

---

## Test files

### `conftest.py`

Adds all TRAVIS package roots (`navigation`, `perception`, `travis_brain`, `speech`) to `sys.path` so tests can import packages without a ROS2 install. Pytest loads this automatically - no action needed.

---

### `test_navigation_exploration_explore_costmap_map.py` - 165 unit tests

Unit tests for every function in `explore_costmap_map.py`. All tests use small synthetic numpy arrays (no PGM files, no matplotlib). Several classes are heavily parametrised, which is why the counts are far above the number of distinct scenarios, and why this file, not the node tests, dominates suite runtime.

| Class | Tests | What is verified |
|---|---|---|
| `TestLoadMap` | 35 | Map loading: free/occupied/unknown masks are correct; navigable cells exclude wall neighbours; pre-existing `covered_mask` is preserved |
| `TestBuildMapDataParity` | 5 | `build_map_data()` (ROS2 bridge) produces identical masks to `load_map()` for the same map data |
| `TestCoordinateHelpers` | 10 | `pixel_to_world` at the origin; `world_to_pixel` round-trip |
| `TestGenerateCandidates` | 4 | Candidates lie in navigable cells; small isolated regions get no candidate; fully non-navigable maps return empty |
| `TestComputeVisibility` | 4 | Open grids have coverage but no frontiers; walls block cells on the far side; already-covered cells are excluded; unknown bands produce frontiers |
| `TestComputeAchievableCells` | 3 | Achievable set is stable after covering cells (denominator does not shrink); empty candidates; cells behind walls excluded |
| `TestNavigableDistanceMap` | 11 | Distance zero at start; distances increase correctly; walls unreachable (inf); partial walls force longer paths; diagonals cost √2 |
| `TestGreedySetCover` | 9 | Three disjoint tiles all selected; zero-gain terminates cleanly; real map exceeds coverage threshold; positive γ prefers nearby candidates; unreachable candidates not selected |
| `TestAngularDiff` | 4 | Symmetry; wraparound; opposite headings; just past 180° |
| `TestComputeHeadings` | 4 | East/west cells select correct heading; no cells returns empty; headings are multiples of increment |
| `TestUpdateCoveredMask` | 3 | Cells in frustum become covered; cells outside frustum unchanged; ratio between 0 and 1 |
| `TestCoverageRatio` | 2 | Empty achievable set returns 1.0; partial coverage returns correct fraction |
| `TestNearestNeighbourOrder` | 4 | All waypoints returned; no duplicates; first waypoint nearest to robot; empty input returns empty |
| `TestPlanWaypoints` | 22 | All waypoints in free cells; no two closer than inflation radius; all-visited returns empty; coverage warning fires on isolated map; no warning on well-connected map |
| `TestCameraPlanningScanConsistency` | 45 | Every cell the planner labels coverable is actually coverable by the camera model |

```bash
# Run from the exploration package root
pytest tests/test_navigation_exploration_explore_costmap_map.py -v
```

---

### `test_navigation_exploration_execution_strategy.py` - 26 unit tests

Unit tests for `ExplorationSession` and `MidPathReplanner`. All tests use 20×20 synthetic numpy arrays with `resolution=1.0 m/px` so distances and candidate positions are easy to reason about without running the full planner.

| Class | Tests | What is verified |
|---|---|---|
| `TestNearestStart` | 3 | Returns the geometrically nearest candidate; works with a single navigable cell |
| `TestOnStep` | 2 | Runs the mid-path replan check; returns `False` without a replanner armed |
| `TestOnArrive` | 3 | Waypoint position added to `visited_candidates`; FOV-aware marking of surrounding candidates |
| `TestOnUnreachable` | 3 | Returns `True` when no path exists; `False` for a valid multi-step path; `False` when the robot is already at the goal |
| `TestPlanExclusion` | 2 | Planning returns `[]` when all candidates are visited; visited candidates from one cycle are excluded from the next |
| `TestSLAMCandidateGrowth` | 1 | Candidate pool grows after `navigable_mask` is expanded mid-session |
| `TestMidPathReplanner` | 5 | No fire on first step; no fire below displacement interval; fires when a waypoint costs < `replan_ratio` of the current target; no fire when nothing is that much cheaper |
| `TestMonotonicVisited` | 1 | `visited_candidates` never shrinks across a full session sequence |
| `TestUnreachableIsNotVisited` | 6 | **An aborted waypoint is never recorded as visited**; blacklisting after N aborts; `clear_unreachable` restores the parked set; a real arrival clears only that waypoint's abort count |

```bash
pytest tests/test_navigation_exploration_execution_strategy.py -v
```

---

### `test_navigation_exploration_rotation_strategy.py` - 13 unit tests

Covers the min-travel heading order added to `get_headings`, plus `RotationState`.

| Class | Tests | What is verified |
|---|---|---|
| `TestGetHeadingsMinTravel` | 8 | Returned headings are the same *set* as before, ordered to minimise total turn from the current heading |
| `TestMinTravelOrderHelper` | 2 | `_min_travel_order` picks the cheaper sweep direction |
| `TestRotationState` | 2 | Continues while coverage grows; stops after `stop_after` gainless headings |
| `TestWrappedStep` | 1 | Angular step wraps correctly across 0°/360° |

---

### `test_navigation_exploration_node_status_handling.py` - 61 unit tests

Tests the ROS2 node's Nav2 result handling **without a running ROS graph**, the
failure paths that caused the observed livelocks and premature stops, plus the
arrival failsafe, teleop mode and the Nav2/TF watchdogs.

The base fixture leaves the arrival failsafe **off** (`arrival_verify_timeout_s: 0`), so the legacy rows below exercise the immediate-abort path directly; the failsafe classes opt in by setting the tolerance and timeout themselves.

| Test / class | What is verified |
|---|---|
| `test_abort_tries_next_waypoint_same_plan` | With the failsafe off, an ABORT advances within the current plan rather than re-planning |
| `test_abort_does_not_mark_failed_spot_visited` | **An aborted waypoint is not recorded as observed** |
| `test_abort_last_waypoint_exhausts_plan_then_replans` | Exhausting the plan by failure returns to PLANNING |
| `test_second_full_plan_failure_triggers_recovery_spin` | Streak ≥ 2 issues a recovery `Spin` |
| `test_rapid_failures_do_not_stop_before_the_timeout` | The streak cap alone cannot stop the run in seconds |
| `test_persistent_failure_stops_once_timeout_elapsed` | Stop fires once `plan_timeout_s` has genuinely elapsed |
| `TestInflatedPoseGate` | The inflated-pose stop fires only after the timeout; the clock clears when the pose becomes plannable again |
| `test_zero_timeout_never_stops` | `plan_timeout_s: 0` retries forever |
| `test_success_resets_failed_plan_streak` | A real arrival clears the streak and its stuck-since clock |
| `test_cancel_is_ignored` | A CANCEL (our own mid-path replan) does not advance the index or count as a failure |
| `test_stale_cancel_after_new_goal_does_not_corrupt_index` | A late cancel result landing after a new goal cannot corrupt `_wp_index` |
| `TestArrivalFailsafe` (13) | A Nav2 failure enters `VERIFYING` instead of aborting outright; arriving within tolerance credits the waypoint whatever the yaw; motion restarts the stillness clock so a human driving keeps the window open; only stillness for the full timeout fails the waypoint; `arrival_verify_timeout_s: 0` restores immediate abort |
| `TestTeleopMode` (9) | While teleop is on no Nav2 goals are sent, coverage is still marked as the human drives, driving to the waypoint counts as reaching it, disabling replans from the current pose, and the watchdog never trips |
| `TestNav2Watchdog` (11) | A Nav2 that vanishes or goes silent mid-goal trips the watchdog instead of hanging in `TRAVELING`; the waypoint is handed to the failsafe; repeated trips rebuild the action clients |
| `TestTfRecovery` (9) | A `map -> base_frame` outage lasting past the threshold rebuilds the `TransformListener`; transient gaps do not |
| `TestWatchdogArming` (6) | The watchdog clock is armed at goal send (not acceptance) and disarmed once a goal resolves, so nothing is left tracked during arrival observation |

```bash
pytest tests/test_navigation_exploration_rotation_strategy.py \
       tests/test_navigation_exploration_node_status_handling.py -v
```

---

### `test_integration_navigation_exploration_explore_costmap_map.py` - 6 integration tests

End-to-end simulation tests that run the full plan → travel → observe loop on the real lab map. These are slow (30 s – 3 min each) because they execute the complete planning pipeline for every waypoint. Run them with `-s` to see per-cycle progress output.

| Test | What is verified |
|---|---|
| `test_plan_execute_cycle_reaches_90_percent` | A single plan followed by full execution of all waypoints reaches ≥ 90% coverage |
| `test_reactive_replan_after_each_waypoint_reaches_90_percent` | Replanning after every waypoint (as the ROS2 node does) also reaches ≥ 90% |
| `test_coverage_ratio_is_monotonically_non_decreasing` | Coverage ratio never goes down across waypoints |
| `test_smaller_detection_range_yields_more_waypoints` | Halving `max_detection_range` produces more waypoints (sanity check on planner scaling) |
| `test_measure_waypoint_count_and_density` | Reports waypoint count and spatial density for the map (regression tripwire on planner scaling) |
| `test_slam_mode_reaches_no_frontiers` | A SLAM-mode run on a growing map terminates with `no_frontiers` rather than churning |

```bash
# -s is required to see cycle-by-cycle progress and the final coverage bar
pytest tests/test_integration_navigation_exploration_explore_costmap_map.py -v -s
```

**When to run:** before merging any change to `explore_costmap_map.py` or `execution_strategy.py` that affects planning or coverage logic.

---

## Offline A/B harnesses (`tests/algo_evaluation/`)

These are **not** pass/fail tests, they are experiments that produced the current
parameter defaults. Each script's docstring states the question it answers, and each
writes plots/CSVs next to itself. Re-run one when you want to revisit its decision.

| Script | Question it answered | Outcome |
|---|---|---|
| `test_see_while_moving.py` | Observe along the path vs spin at each goal? | `see_while_moving: true` (A/B/C: turn-at-goal vs see-while-moving vs both) |
| `test_heading_travel.py` | Does minimising heading *count* ignore angular *travel*? | Min-travel ordering in `get_headings`; −12.6% travel on `lab_ghent` |
| `test_replan_N.py` | Drain the whole plan, or replan every N arrivals under SLAM? | `replan_every_n_step: 2` (SLAM-gated) |
| `test_midpath_replan_ratio.py` | What should the hardcoded 0.5 divert ratio be? | `mid_path_replan_ratio: 0.75` |
| `test_FOV_visited_candidates.py` | Can the wall-blind Euclidean visited-disc be removed? | Yes, replaced by FOV/occlusion-aware `covered_mask` marking |
| `benchmark_waypoint_scoring.py` + `scoring_variants.py` | How do scoring-function changes compare on fixed maps? | Identified the uniform-cost BFS distance as ~100% of plan time → Dijkstra swap |



---

## System tests (`tests/system/`)

L3 end-to-end runs of the real stack in a simulator: **record** live → **evaluate** offline → **visualise** offline. Only the recorder needs ROS, so run folders can be copied off the simulation machine and analysed anywhere. Gates are per (scene, mode) in `baselines/baseline_<scene>.yaml`.

Always pass `use_sim_time:=true`, all time KPIs are in *simulation* seconds and wall-clock numbers are not comparable between simulators.

See [tests/system/README.md](tests/system/README.md) for the full workflow.

---

## When to run which test

| Situation | Command |
|---|---|
| Quick sanity check after any edit | `pytest tests/test_navigation_exploration_*.py -v` (265 tests, ~2.5 min, the heavily parametrised map-loading and camera-consistency classes dominate) |
| Changing Nav2 result / failure handling | `pytest tests/test_navigation_exploration_node_status_handling.py -v` |
| Before merging to master | Add the integration suite: `pytest tests/ -v -s` |
| After changing planner weights or sampling step | Run `evaluate_waypoint_positions.py` and inspect the hit-rate table |
| Visually checking a planner or session change | Run the appropriate `visual_demo_*.py` and watch the animation |
| SLAM-specific changes | Run `visual_demo_slam.py`; robot must reach the right half of the map |
| Revisiting a tuned parameter | Re-run the matching `tests/algo_evaluation/` harness |
| Validating on the real stack | Record an L3 run and compare against the scene baseline |
