# Exploration system tests (L3)

End-to-end evaluation of the exploration stack running in a simulator(Gazebo 11 / humble locally, Isaac Sim on the remote laptop, or the jazzy docker). One world runs in exactly ONE simulator; the `run.simulator` field in the scene YAML documents which.

The pipeline is: **record** a run live → **evaluate** it offline →
**visualise** it offline. Only the recorder needs ROS; evaluation and
visualisation are pure Python, so run folders can be copied off the remote
laptop and analysed anywhere.

```
recorder.py            ROS node, run alongside the exploration stack
manual_exploration.py  ROS node, human-driven reference run (baseline)
evaluate_run.py        offline: KPIs, gates, PASS/FAIL, report.md, baseline
visualize_run.py       offline: per-cycle frames, overview.png, mp4
aggregate_runs.py      offline: mean/std over N runs, normalized vs baseline
baselines/             one baseline_<scene>.yaml per environment
```

## 0. Prerequisites

- The simulations are launched from their own launch files (outside this repo folder)
- Nav2 + exploration stack is launched from exploration. RViz is decoupled:
  `exploration.launch.py` includes `launch/rviz.launch.py` (`use_rviz:=false`
  for headless runs), and for manual baseline runs
  `manual_exploration.launch.py` starts the coverage node + RViz with the same
  assembled parameters as the autonomous node (see §2).
- **Always pass `use_sim_time:=true` to the recorder.** All time KPIs and
  time gates are in SIMULATION seconds; Gazebo and Isaac publish `/clock` at
  different real-time factors, and wall-clock numbers are not comparable.
  The recorder writes `time_source: sim|wall` into `meta.yaml` and
  `evaluate_run.py` warns loudly on `wall`.
- One `baselines/baseline_<scene>.yaml` per world. Each file has a
  `known_map:` and a `slam:` section so gates, tolerances and the saved
  baseline are per (scene, mode). **You do not need to create it by hand**:
  when the recorder is given a `--config` that does not exist yet, it creates
  a fresh scene file from the default template (scene name taken from the
  filename `baseline_<scene>.yaml`) and continues — review its gates and set
  `run.simulator` afterwards. `--config` paths are resolved as given first,
  then against the installed `share/exploration/baselines/`; keep the
  source-tree copy (`tests/system/baselines/`) as the versioned one that
  `--save-baseline` targets.

## 1. Record a run

Start the recorder BEFORE unpausing/starting exploration, so t=0 covers the
whole run:

```bash
ros2 run exploration recorder.py \
    --config baselines/baseline_lab05.yaml --mode known_map \
    --ros-args -p use_sim_time:=true \
               -p lidar.scan_topic:=/panoramic/scan \
               -p lidar.base_frame:=base_footprint
```

`--mode` is `known_map` or `slam` and selects the matching section of the
scene YAML. Stop the recorder with Ctrl-C once exploration is done; it then
writes the final artefacts and closes the bag.

Each run produces one self-contained folder under `run.out_dir`:

```
<out_dir>/<scene>_<mode>_run<N>_<timestamp>/
    motion.csv               5 Hz pose (TF map frame) + coverage; no blank cells
    plans.csv                exploration state machine timeline (incl. VERIFYING,
                             entered when Nav2 fails a waypoint and the node
                             checks the TF pose before giving it up)
    nav_goals.csv            every Nav2 goal (tracked by goal UUID); includes
                             status + recoveries + distance_remaining + a derived
                             `reason` for ABORTED goals (see "Why a goal failed")
    published_waypoints.csv  strategy waypoints + current goal
    nav2_paths.csv           Nav2 planned paths
    covered_mask_final.npy   + covered_mask_meta.yaml (resolution/origin)
    map_final.npy            + map_meta.yaml (last /map seen, if published)
    maps_completion/         per-plan-cycle covered_mask + /map snapshots (.npy)
    meta.yaml                scene/mode/time_source/params
    bag/                     full ros2 bag (fallback ground truth)
    logs/                    per-process console + rcl logs, if launched via
                             run_with_log.sh (see "Separated logs")
```

### Why a Nav2 goal failed

This Nav2 (`nav2_msgs` 1.1.20) returns `std_msgs/Empty` as the action result — no
error code. The recorder instead subscribes to the action **feedback** and, for
each terminal goal, records the last `recoveries` and `distance_remaining` seen,
plus a heuristic `reason` for ABORTED goals in `nav_goals.csv`:

| reason              | meaning                                                    |
|---------------------|------------------------------------------------------------|
| `no_valid_path`     | aborted before any feedback — planner found no path start  |
| `stuck_no_progress` | recoveries fired and still far from goal — could not advance|
| `failed_near_goal`  | aborted close to goal — e.g. goal in an inflated cell       |
| `aborted_unknown`   | aborted but feedback did not match the above                |

The raw `recoveries` / `distance_remaining` columns are kept so the label can be
second-guessed. `when` = `t_result`; `where` = `goal_x_m` / `goal_y_m`.

### Separated logs

`exploration` and `nav2` run in their own terminals; wrap each launch with
`run_with_log.sh <tag> <command...>` to tee its console to `<tag>.log` and route
its rcl node logs to `<tag>_rcl/`.

All wrappers write to ONE fixed staging dir (`/tmp/exprun_stage`) — **no env var
to export in each terminal**. Because nav2/exploration start before the recorder
(its run folder does not exist yet), logs stage there first; the recorder copies
the staging dir into `runs/<run>/logs/` at shutdown, then clears it so the next
run starts clean. Just run the wrapper in each terminal:

```bash
# terminal 1 — nav2 (SLAM: omit map; known-map: add map:=<abs path to map.yaml>):
./run_with_log.sh nav2 ros2 launch nav2 nav2.launch.py robot:=turtle3
# known-map example:
#   ./run_with_log.sh nav2 ros2 launch nav2 nav2.launch.py robot:=turtle3 \
#       map:=/ros2_ws/src/assets/house_amazon/map.yaml

# terminal 2 — recorder (unwrapped: it owns the run folder and does the copying):
ros2 run exploration recorder.py \
    --config baselines/baseline_house_aws.yaml --mode slam \
    --ros-args -p use_sim_time:=true \
               -p lidar.scan_topic:=/panoramic/scan \
               -p lidar.base_frame:=base_footprint

# terminal 3 — exploration (start a few seconds after nav2 is up):
# SLAM: omit map_dir (reads the live /map topic).
# known-map: add map_dir:=<abs path to the map FOLDER> (contains .pgm + .yaml).
./run_with_log.sh exploration ros2 launch exploration exploration.launch.py
# known-map example:
#   ./run_with_log.sh exploration ros2 launch exploration exploration.launch.py \
#       map_dir:=/ros2_ws/src/assets/house_amazon
```

### One-shot launcher (three terminals)

`launch_run.sh` opens the three above in separate `gnome-terminal` windows, each
attached to the container (`docker exec -it TRAVIS_ros2jazzy`), with the recorder
and exploration waiting `DELAY` seconds so Nav2 comes up first:

```bash
./launch_run.sh                          # defaults: TRAVIS_ros2jazzy, 5s delay,
                                         # baseline_house_aws.yaml, slam
DELAY=6 ./launch_run.sh                  # longer nav2 head-start
CONTAINER=other CONFIG=baselines/baseline_lab05.yaml MODE=known_map ./launch_run.sh
```

Each window drops to a shell after its process exits so you can inspect it. Run
it from the host (it needs `gnome-terminal`), not inside the container.

Result: `runs/<run>/logs/{nav2,exploration,recorder}.log` + their `_rcl/` dirs.
The recorder copies (not moves), so still-running nav2/exploration keep logging;
the run folder holds a snapshot up to the recorder's shutdown. Override the
staging path with `RUN_LOG_STAGE` if you ever need to (honoured by both sides).

All positions in all CSVs are in the **map frame** (robot pose from TF
`map -> base_frame`), so everything overlays without frame juggling.

## 2. Take the baseline (once per scene+mode)

A human drives a careful reference run with the strategy OFF, using the SAME
motion layer as an autonomous run (Nav2 goals clicked in RViz — not teleop),
so the baseline includes the same controller behaviour (acceleration limits,
inflation detours, recovery pauses) and the comparison isolates the strategy's
choice of where to go.

Launch, in order:

1. the simulator (its own launch file, outside this repo);
2. Nav2 (+ AMCL/map_server in known-map mode, or SLAM in slam mode) — same
   launch as for an autonomous run;
3. the manual coverage node + RViz — this replaces `exploration.launch.py`
   and assembles the exact same parameters (camera/lidar/inflation/map), so
   the coverage measurement is identical to the autonomous node's:

   ```bash
   ros2 launch exploration manual_exploration.launch.py \
       map_dir:=/abs/path/to/map_folder use_sim_time:=true
   # slam mode: just omit map_dir (the node then reads /map)
   ```

   The mode is deduced: `map_dir` given → known_map, omitted → slam. A
   `map_dir` that doesn't exist aborts with an error (typo protection) rather
   than silently switching to SLAM. `use_sim_time` defaults to **false** (real
   robot), so pass `use_sim_time:=true` for these sim-based baseline runs — it
   must match the simulator and Nav2 or TF lookups fail.

   RViz starts from this launch (`use_rviz:=true` by default; it can also be
   started alone with `ros2 launch exploration rviz.launch.py`);
4. **the recorder — the run is lost without it.** Same command as §1:

   ```bash
   ros2 run exploration recorder.py \
       --config baselines/baseline_lab05.yaml --mode known_map \
       --ros-args -p use_sim_time:=true \
                  -p lidar.scan_topic:=/panoramic/scan \
                  -p lidar.base_frame:=base_footprint
   ```

Only start driving once the recorder logs `Recording to <run folder>`: it
observes passively, so anything driven before that moment is simply not in the
data and cannot be recovered — the whole manual run would have to be redone.

Drive the robot by clicking "2D Goal Pose" in RViz until the space is swept
(the coverage overlay `/exploration/covered_mask` shows live in RViz what has
been credited). Stop the recorder (Ctrl-C) and persist the run as the scene
baseline:

```bash
./evaluate_run.py runs/lab05_known_map_run1_... \
    --config baselines/baseline_lab05.yaml --mode known_map \
    --save-baseline --source manual_exploration
```

This fills the `baseline:` block of the scene YAML and copies the reference
covered_mask next to it. After the baseline exists, tune the scene's
`gates.time_to_complete_max_s` to a sensible budget.

## 3. Evaluate a run

```bash
./evaluate_run.py runs/lab05_known_map_run2_... \
    --config baselines/baseline_lab05.yaml --mode known_map --compare
```

Writes `report.md` (+ `comparison.png` with `--compare`) into the run folder
and prints PASS/FAIL. Exit code 0 = PASS.

Gates (all actually checked, defined per scene+mode in the YAML):

| gate | meaning |
|---|---|
| `final_coverage_min` | final coverage ratio must reach the threshold |
| `must_terminate` | the run must reach the COMPLETE state |
| `time_to_complete_max_s` | completion within the sim-time budget (fails if never completed) |
| `nav_aborted_max` | max Nav2 goals allowed to ABORT |

`--compare` additionally diffs every KPI against the saved baseline and flags
`path_length_m` / `total_waypoints` / `final_coverage` against the
`tolerances:` block.

## 4. Visualise a run

```bash
./visualize_run.py runs/lab05_known_map_run2_...            # frames + overview
./visualize_run.py runs/... --video                         # + timelapse.mp4 (pausable)
./visualize_run.py runs/... --map-yaml /path/to/lab05.yaml  # static map underlay
```

Renders one PNG per planning cycle (`frames/`) plus `overview.png`: final
trajectory over the map, coverage-vs-path and coverage-vs-time curves. The map
underlay comes from the run's own `map_final.npy` when `/map` was published
(SLAM, or a map_server); for known-map runs without `/map`, pass the
map_server YAML with `--map-yaml`.

## 5. Aggregate several runs (mean ± std vs baseline)

Exploration is stochastic; grade the strategy on a GROUP of runs of one
(scene, mode), not a single run:

```bash
./aggregate_runs.py runs/lab05_known_map_run* \
    --config baselines/baseline_lab05.yaml --mode known_map \
    --out summaries/lab05_known_map
```

Writes into `--out`:

- `summary.md`, per-KPI mean / std / min / max over the runs, plus each
  mean as a ratio to the scene baseline (`mean/baseline`),
- `curves_<scene>_<mode>.png`, all runs' coverage-vs-time and
  coverage-vs-path curves overlaid (semi-transparent) with the mean curve,
- `normalized_kpis.png` (multi-group form), cross-scene comparison.

Several scenes can go on one normalized figure; pair one `--config` per
`--group`, in order:

```bash
./aggregate_runs.py \
    --group runs/lab05_known_map_run*     --config baselines/baseline_lab05.yaml \
    --group runs/warehouse_known_map_run* --config baselines/baseline_warehouse.yaml \
    --mode known_map --out summaries/known_map_all
```

Absolute KPIs are never compared across environments, the cross-scene
figure shows only the normalized ratios (run mean / that scene's baseline,
error bars = std/baseline), so "1.15x baseline in lab05 vs 1.6x in the
warehouse" is a fair statement of WHERE the strategy degrades. Runs of
different scenes mixed into one group are rejected. If you record several
manual baseline runs, aggregate those folders the same way to get a
baseline mean ± std.

## Typical full loop (checklist before burning Isaac/remote time)

1. Launch sim + stack locally (Gazebo humble).
2. Record a short run; Ctrl-C; check the run folder has non-empty
   `motion.csv`, `covered_mask_final.npy`, and `meta.yaml` says
   `time_source: sim`.
3. `evaluate_run.py` → read `report.md`; `visualize_run.py` → eyeball
   `overview.png`.
4. Take + save the manual baseline, tune gates.
5. Only then run the same procedure on Isaac / the jazzy docker.
6. Once several runs exist per scene, `aggregate_runs.py` for the
   mean ± std picture.
