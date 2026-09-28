# nav2

Nav2 configuration, launch files and map tooling for TRAVIS.

## Launch modes

`nav2.launch.py` picks a mode from whether `map:=` is given:

| Invocation | What starts |
|---|---|
| no `map:=` | SLAM mapping: nav2 bringup + slam_toolbox `online_async` |
| `map:=<map.yaml>` | known map, **AMCL** localisation (default, unchanged) |
| `map:=<map.yaml> localization:=slam_toolbox serialized_map:=<stem>` | known map, slam_toolbox localisation |

### Why a slam_toolbox localisation option exists

AMCL matches the live scan against a rasterized occupancy grid using a particle
filter. slam_toolbox in `localization` mode instead scan-matches against the
**stored scans** of a serialized pose-graph, which is generally tighter. It
publishes `map -> odom` exactly as AMCL does, so nav2 and the exploration node
need no change to use it.

AMCL remains the default. Nothing about an existing invocation changes.

### Per-robot arguments

`base_frame` and `scan_topic` reach slam_toolbox only (AMCL takes both from its
own params file). They default to the simulator's values and must be overridden
on the ROSbot XL:

```bash
ros2 launch nav2 nav2.launch.py \
    map:=/ros2_ws/src/assets/lab_ghent/map.yaml \
    localization:=slam_toolbox \
    serialized_map:=/ros2_ws/src/assets/lab_ghent/lab_ghent_20260814 \
    base_frame:=base_link scan_topic:=/scan
```

Note `serialized_map` is the pose-graph **stem with no extension**: the launch
file checks for `<stem>.posegraph` and fails early with a clear message if it is
missing, rather than letting slam_toolbox come up with no map.

## Saving a map from a SLAM run

Two formats are always written together, because two different consumers need
different things and **both are mandatory**:

| Files | Consumer | Why |
|---|---|---|
| `.posegraph` + `.data` | slam_toolbox localisation | the stored scans it matches against; cannot be regenerated |
| `.pgm` + `.yaml` | the exploration node | it plans on an occupancy grid |

The exploration node decides static-vs-live from
`_is_slam = not bool(map_file_path)`, and that path is found by globbing
`map_dir` for a `.pgm` (see `exploration.launch.py:_resolve_map_paths`). A
graph-only folder therefore either fails that glob or, worse, leaves exploration
believing it is doing live SLAM with the wrong stopping rules.

### Automatically, on every recorded SLAM run

`recorder.py --mode slam` calls the script below at shutdown and writes the
result to `navigation/nav2/maps/<scene>_<timestamp>/`. The run's `meta.yaml`
records `saved_map_dir`; the map folder's `source.txt` names the run that
produced it. Set `MAP_STORE` in the environment to write elsewhere.

### By hand, at any point during a run

```bash
ros2 run nav2 save_map.sh <out_dir> [stem] [source_run]
```

Serialising mid-run is safe and does not disturb mapping. Exit codes:
`0` saved, `1` bad usage, `2` slam_toolbox not running (nothing to save, not an
error), `3` a service call failed or the files did not land.

### Promoting a map

`navigation/nav2/maps/` is **gitignored working output**: every SLAM run drops a
folder there, including bad runs. When a map proves good, copy it by hand into
`assets/<scene>/`, which is where a committed, reusable map belongs and what the
system tests expect (`tests/conftest.py`).

Promotion is deliberately manual so a bad run can never overwrite a known-good
lab map.

### The one trap worth knowing

slam_toolbox resolves the output path **in its own process**, not in the shell
that calls the service. Under Docker the directory must be valid inside the
*slam_toolbox* container; a path that only exists where the caller runs yields
`RESULT_FAILED_TO_WRITE_FILE`, or worse, reports success while writing somewhere
invisible. `save_map.sh` therefore verifies all four files exist and are
non-empty before reporting success. On the ROSbot `/ros2_ws/src` is the shared
bind mount, so paths under it resolve identically in both containers.

A map folder must also contain exactly **one** `.yaml`. This is why provenance
is written as `source.txt` and not `source.yaml`: exploration takes
`glob('*.yaml')[0]` and glob order is not guaranteed, so a second YAML could be
loaded as the map descriptor.

## Testing

```bash
python3 -m pytest ../exploration/tests/system/test_save_map.py -q
```

Covers the save script (via a fake `ros2` on `PATH`), the recorder's SLAM-only
gating, and the launch branching, including that a known map still defaults to
AMCL.
