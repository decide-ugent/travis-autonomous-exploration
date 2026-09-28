# TRAVIS Exploration, V2.0 Overview

A description of what the **exploration** package does, why it exists, and how it works at a conceptual level.

V2.0 keeps V1.0's planning core and changes how the robot *observes* while executing a plan, plus what happens when the world does not cooperate. If you know V1.0, the short version is: the robot no longer stops and spins to look around, and a run no longer ends because Nav2 had a bad day.

## What exploration v2.0 does

A robot is placed inside an indoor space (a warehouse aisle, a lab, an office floor). The exploration decides, on its own, where the robot should drive and which way it should point its camera, so that it observes as much of the floor as possible without a human steering it. The output is a sequence of stops that the robot then executes, observing continuously as it moves.

Manually mapping or inspecting a space is slow, repetitive, and easy to do incompletely (corners get missed, aisles get skipped). We want the robot to handle that decision itself, given either a pre-recorded floor plan or no map at all. In TRAVIS, the exploration package is the component responsible for the "go look around" behaviour, before any task that requires understanding what is in the environment.

This is **coverage planning**, not classic frontier exploration. The goal is to *visually observe* the floor with the camera, so the planner reasons about which cells the camera has actually seen, not merely where the map boundary is. Frontiers are one term in the score, and they matter mainly in SLAM mode.

## The big picture, in three questions

The whole algorithm answers three questions a human would also ask if dropped into an unfamiliar room with the same job:

1. **Where could I usefully stand?**
   The software lays a regular grid of *candidate* positions over every spot the robot's body actually fits.
2. **Which of those standing spots, taken together, let me see the most?**
   It picks the smallest useful subset of candidates whose camera views, combined, cover (almost) the whole space.
3. **In what order do I visit them, and which way do I look on the way?**
   It orders the chosen spots so total walking distance stays short, and aims the camera at what is still unseen while the robot drives.

The third question is where V2.0 differs. V1.0 answered it with "arrive, then rotate through a list of headings". V2.0 answers it with "observe continuously along the path, and arrive already facing the right way".

### The planner logic

The planner logic relies on:

- **Greedy set cover**: at each step, pick the spot that adds the most new coverage, then repeat until enough of the floor is covered.
- **Wall-aware distance**: distance measured by walking around walls (not a straight line), so the robot is never told to go through a wall.
- **Nearest-neighbour ordering (a TSP heuristic)**: visit the chosen spots in an order that keeps total walking short, without trying every possible order.

## The map layers

The logic is built over several "map" overlays:

| Term | Meaning |
|---|---|
| **Map** | A 2D top-down picture of the floor, with three kinds of cells: free, wall, unknown. |
| **Navigable area** | The free cells that are far enough from walls for the robot's body to fit. |
| **Candidate / Waypoint** | A point on the floor the robot could stand at. A *candidate* is any such point; a *waypoint* is one the planner has actually selected. |
| **Coverage** | The fraction of the floor the camera has already observed. Grows as the robot drives. |
| **Frontier** | The boundary between known and unknown areas. Only relevant when the map is being built live (SLAM mode). |
| **Heading** | The direction the camera is pointed at a given moment. |

![Map layers](images/exploration_step_1_masks.png)

## Two operating modes

The package runs in one of two modes, decided by the launch configuration.

- **Known-map mode**: a floor plan is provided up front. The planner's job is to maximise camera coverage of that known floor.
- **SLAM mode**: the map starts mostly unknown and grows as the robot drives. The planner re-runs regularly, turning frontiers into newly reachable territory.

Several behaviours below are deliberately gated to SLAM mode, because on a static map they would misfire. Where that is the case it is called out explicitly.

## The planner logic, in detail

Each step takes the previous step's output as input.

### 1 Building the map layers

Before any planning happens, the incoming occupancy grid is turned into a stack of boolean masks: free, wall, unknown, and **navigable** (free cells far enough from walls for the robot's body to fit, computed by inflating walls with a distance transform). A separate **covered_mask** starts empty and accumulates every cell the camera has actually observed. This `covered_mask` persists across plan cycles, which is what stops the robot from re-visiting areas it has already seen.

In SLAM mode the map itself changes shape as new area is discovered. When it does, the accumulated coverage is **re-anchored onto the new grid** rather than reset, so a map resize does not silently erase the run's progress.

Code: `load_map`, `build_map_data`, `reproject_covered_mask` in `explore_costmap_map.py`.

### 2 Generating candidates

Candidate positions (places the robot *could* stand) are produced by a three-pass union over the navigable mask:

1. A regular grid at spacing `sampling_step_m`.
2. One representative per connected navigable region, so isolated rooms are not missed if the grid happens to skip them.
3. Skeleton points along the medial axis of corridors (local maxima of the distance-to-walls field), so candidates land in the middle of corridors instead of hugging walls.

A typical indoor map of 280 square metres produces 30 to 150 candidates.

### 3 Visibility, what each candidate would see

For every candidate, the planner simulates a full 360 degree turn by casting rays (360 by default, one per degree). Each ray walks outward up to `max_detection_range` and stops on the first wall or unknown cell it hits. Along the way, it records two distinct sets:

- **Coverage cells**: free cells the camera would observe from there, that are not already in `covered_mask`. This is how much of the *known* floor that spot would help reveal.
- **Frontier cells**: the first unknown cell hit on each ray. This is how much of the *unknown* the spot would help discover. Only meaningful in SLAM mode.

The two sets are stored separately so the planner can weight them independently.

Code: `compute_visibility`, `compute_all_visibility`.

### 4 Scoring a candidate

Each candidate gets a score that combines its two gains and penalises distance, with the distance penalty applied as an **exponential decay**:

```
score  =  (alpha * frontier_gain  +  beta * coverage_gain)  *  exp(-gamma * distance / max_range)
```

Where:

- `alpha` = `frontier_weight`, `beta` = `coverage_weight`, `gamma` = `travel_cost_weight` (the YAML parameters).
- `distance` is the walking distance from the robot, in pixels, measured around walls (see step 7), not in a straight line.
- `max_range` is the camera's `max_detection_range` in pixels. Dividing by it makes `gamma` resolution-independent.

> **Changed since V1.0.** V1.0 divided the gain by `1 + gamma * distance`. V2.0 multiplies it by `exp(-gamma * distance)`. The exponential falls off faster nearby and never lets a very distant candidate keep a small but non-zero share of its gain, which in practice stops the planner from crossing the building for a mediocre viewpoint. A candidate that cannot be reached at all is given an effectively infinite distance, so it sorts to the bottom instead of being selected and then failing.

Quick intuition for the extremes:

- `gamma = 0`: distance is ignored, the planner picks whatever scores highest no matter how far.
- `alpha = 0`: the robot only cares about looking at known floor (inspection mode).
- `beta = 0`: the robot only cares about pushing into the unknown (pure SLAM exploration).

Code: scoring loop inside `greedy_set_cover`.

### 5 Greedy set cover, picking the waypoints

The planner now needs the smallest set of candidates whose visibility, taken together, covers the floor. This is the classic **set cover** problem, which is NP-hard, so a greedy approximation is used:

1. Score every remaining candidate.
2. Pick the highest-scoring one and add it to the chosen set.
3. Subtract its coverage and frontier cells from what is still "remaining".
4. Re-score and repeat.

![Greedy round, scoring every candidate](images/step_4_03a_score.png)

![Greedy round, picking the best](images/step_4_03b_pick.png)

![Greedy round, re-scoring what is left](images/step_4_03c_rescore.png)

The loop stops when one of three things happens: the best remaining candidate adds zero new coverage and zero new frontier (saturation), no candidates are left, or the cap `max_waypoints_per_plan` is reached. That cap is normally left at `-1`, which means "derive it from the map and the camera range", so it adapts automatically as a SLAM map grows.

### 6 Choosing where to look

At each waypoint the planner works out which direction has the most still-unseen floor, and the goal is sent to Nav2 **with that direction as its target orientation**. The robot therefore arrives already facing what it still needs to see, and takes a single look rather than a rotation sequence.

Meanwhile, the camera's field of view is marked as observed **continuously along the path**, every `observe_step_m` of travel. Coverage accrues while the robot is driving, not only when it is parked.

> **Changed since V1.0.** V1.0 drove heading-blind and then rotated through a set-cover-selected list of 2 to 6 headings at each waypoint. V2.0 observes while moving and aims the arrival heading. Benchmarking (`tests/algo_evaluation/test_see_while_moving.py`) found this reaches similar or higher coverage while skipping every spin, with the largest gain under SLAM.
>
> The old behaviour is still available: set `see_while_moving: false` to restore turn-at-goal. That path also improved since V1.0. The headings are now ordered to minimise total rotation travel, and the early-stop rotation logic that V1.0 listed as planned-but-unimplemented is now wired in, so the robot stops rotating once consecutive headings stop adding coverage instead of always completing the list.

This depends on Nav2 actually honouring the requested orientation. If Nav2's `yaw_goal_tolerance` is too loose, the robot ends facing an arbitrary direction and the aimed look is wasted.

![Final plan](images/step_6_final_plan.png)

### 7 Ordering the waypoints

The chosen waypoints still need a visiting order. The planner uses **nearest-neighbour over wall-aware distance**, not Euclidean:

1. From the robot's current position, measure distance to every waypoint, pick the closest as the first stop.
2. From that stop, measure again, pick the closest unvisited waypoint as the second stop.
3. Repeat until all are ordered.

Wall-aware distance matters because two points 5 metres apart through a wall require walking around it, perhaps 20 metres. Euclidean ordering would happily zigzag the robot across walls; this ordering follows the corridors.

> **Changed since V1.0.** The distance field is computed with Dijkstra over the navigable mask, with diagonal steps costed at √2, rather than a plain breadth-first search that treated every neighbour as one step. BFS systematically underestimated diagonal travel and distorted both the ordering and the distance penalty in the score.

Code: `nearest_neighbor_order`.

### 8 While the robot drives, mid-path replanning

The plan is not frozen once the robot starts moving. As the robot travels along its current segment, a lightweight check runs periodically:

> If, from where I am right now, *any* other waypoint has become substantially cheaper (in wall-aware distance) than the target I am currently heading to, abandon the current segment and replan from here.

The threshold is `mid_path_replan_ratio`, and it is intentionally lenient: small detours do not trigger a replan, but a real shortcut (the robot turned a corner and is now much closer to a different waypoint) does.

Under live SLAM the node additionally returns to planning every `replan_every_n_step` arrivals, so later waypoints are chosen from the freshly revealed map instead of being drained from a plan scored several arrivals ago. On a static map the whole plan is drained, since nothing new can appear.

Code: `MidPathReplanner.check`.

### 9 Growing candidates as the SLAM map grows

In SLAM mode, every plan cycle regenerates the candidate set from the current navigable mask. Newly-mapped corridors immediately get new candidates. A `visited_candidates` set persists across cycles, and candidates near a position the robot has already observed from are treated as visited, so the planner never circles back to slightly-offset duplicates.

Crucially, **a waypoint Nav2 failed to reach is not recorded as visited**. "Could not reach" and "observed" are different facts: the robot never got there, so the area is still uncovered. Failed waypoints go to a separate *unreachable* set, which is restored only when the planner would otherwise run out of candidates.

Code: `ExplorationSession.plan_waypoints_raw`, `_mark`.

### 10 When does exploration declare itself done?

Exploration finishes when the coverage ratio is at or above `exploration_completion_threshold` **and** there are no remaining frontiers. The denominator for the coverage ratio is computed once over all candidates, so the percentage is stable as cells are progressively marked covered.

Two SLAM-only relaxations stop the robot chasing the last unreachable wisp forever:

- `min_frontier_cells`: treat "no frontiers" as *total frontier cells at or below this*, rather than exactly zero, absorbing phantom slivers.
- `no_progress_streak` / `no_progress_eps`: also stop after N consecutive arrivals that each add almost no new coverage.

Both are gated to live SLAM: on a static known map "no progress" is the *expected* steady state once covered, so applying them there would stop a run early.

Code: `compute_achievable_cells`, `coverage_ratio`.

## When things go wrong

This is the other half of what V2.0 adds. A real run does not stop being useful the moment Nav2 misbehaves.

### The robot cannot reach a waypoint

A Nav2 failure does **not** prove the robot is not at the goal: it often stops just outside Nav2's own goal checker. Rather than trusting the verdict, the node checks the robot's actual measured position and credits the waypoint if it is close enough, orientation ignored.

If the robot is genuinely not there, the node waits before giving up, and that wait is a **stillness** timeout rather than a stopwatch: any real motion restarts it. That opens a short window in which a human can drive the robot the last stretch and have the waypoint count as reached, with anything the camera sees during the intervention kept. Only failures that this check could *not* rescue count towards blacklisting a waypoint as genuinely unreachable.

### A human wants to take over

Publishing to `/exploration/teleop_enabled` hands the robot to an operator at any time. Exploration keeps running underneath: it keeps planning and keeps marking camera coverage as the human drives, so nothing observed by hand is lost. Driving to a waypoint counts it as reached. Handing control back replans from wherever the robot was left. Unlike the window above, this mode has no timeout: the operator decides when to move on.

### Nav2 dies, or is restarted, mid-run

Nav2 can be restarted, or crash outright, without ending the run. Two distinct failures are handled: a Nav2 that disappears mid-goal (which would otherwise leave the node waiting forever for a result that will never arrive), and a transform tree that stops resolving after the other container restarts (which cannot repair itself, because the transforms involved are published only once). In both cases the node detects the silence, recovers, and reconnects.

### The robot is genuinely wedged

The node does **not** attempt self-rescue: no pose reset, no re-localisation. Its only automatic recovery is a rotation in place to regain clearance. Beyond that it holds the run open, warns that intervention is needed, and resumes on its own once the pose is plannable again. This is deliberate: an autonomous escape attempt with a bad pose estimate can corrupt the map and turn a recoverable stop into a lost run.

The full parameter-level treatment of all of the above is in the [package README](README.md).

## What you see when it runs

When you watch a live run (in RViz or in the offline matplotlib demo), the screen shows:

- The **map** in greyscale (white = free, black = wall, grey = unknown).
- The **navigable area** highlighted in one colour, narrower than the free area because it accounts for the robot's body.
- The **candidates** as small dots over the navigable area.
- The **chosen waypoints** as larger markers, connected in visiting order.
- A **heading wedge** at the robot's current position, showing the camera's field of view.
- The **covered area** filling in progressively as the robot drives.

![Run overview, house SLAM](images/overview_house_slam_run11.png)

## The knobs that matter

Most users only ever touch these:

| Var | What it controls | Trade-off |
|---|---|---|
| `sampling_step_m` | How dense the candidate grid is. | Denser grid: more thorough, but slower planning. |
| `frontier_weight` | Preference for discovering unknown map area. | Higher: more SLAM-like, robot pushes into the unknown. |
| `coverage_weight` | Preference for seeing more of the already-known area. | Higher: more inspection-like, robot revisits gaps. |
| `travel_cost_weight` | Penalty for placing a waypoint far from the robot. | Higher: shorter paths, but possibly less coverage. |
| `exploration_completion_threshold` | The coverage fraction at which the run is declared "done". | Lower: robot stops sooner, leaves more unseen. |
| `see_while_moving` | Observe along the path (V2.0) or turn at each goal (V1.0). | Off: legacy behaviour, slower, more spins. |
| `plan_timeout_s` | How long to keep a stuck run open waiting for a human. | `0` never gives up, useful for unattended runs. |

The full list, with every parameter documented inline, lives in `config/exploration_system_parameters.yaml`. Camera and LiDAR parameters live in `perception/config/perception_system_parameters.yaml`.

## How it plugs into TRAVIS

The exploration package is one component among several. From outside, it has a simple boundary:

- **Inputs**: a 2D occupancy map and the robot's current pose.
- **Outputs**: position commands sent to **Nav2** (the standard ROS2 navigation stack), which is responsible for actually steering the robot, avoiding moving obstacles, and rotating in place.

```
   map  +  robot pose
            │
            ▼
      ┌─────────────┐
      │ exploration │  ◄── operator can take over at any time
      └─────────────┘
            │
            ▼
          Nav2  ◄── may be restarted; exploration reconnects
            │
            ▼
       robot motors
```

Exploration decides *where to go and which way to look*. Nav2 decides *how to physically get there*. Object recognition and task-level reasoning live in other TRAVIS components.

## Testing it

```bash
# 1. Unit tests, no ROS needed
pytest tests/test_navigation_exploration_*.py -v

# 2. Offline demo: no robot, no ROS2, just matplotlib
python tests/visual_demo_coverage.py

# 3. Live, on a real robot or in simulation
ros2 launch exploration exploration.launch.py
```

A third level exists: scripted system runs that record a full session and compare it against a saved baseline. See `tests/system/README.md`.

## What V2.0 does *not* do

- **No multi-robot coordination.** One robot at a time.
- **Dynamic obstacles are delegated to Nav2's local planner.** The exploration planner sees a static snapshot of the map; people walking through the corridor are handled by the layer below, and the exploration layer simply re-plans later from whatever the map then says.
- **Semantic / object-driven goal selection lives elsewhere in TRAVIS.** This package optimises *coverage* of the floor; it does not decide *what to look for*.
- **Planning assumes the pose estimate is trustworthy.** Localisation drift is not modelled or corrected by this package.
- **Not validated in NVIDIA Isaac Sim.** V2.0 has been exercised on recorded maps, in Gazebo simulation, and on a real robot in the Ghent lab; Isaac-Sim validation remains outstanding.
