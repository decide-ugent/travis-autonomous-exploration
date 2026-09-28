#!/usr/bin/env bash
# Launch one exploration run across three gnome-terminal windows, each attaching to the running Jazzy container (docker exec) and starting one process:

#   1. nav2         (immediately)
#   2. recorder     (after $DELAY seconds — Nav2 needs a few seconds to come up)
#   3. exploration  (after $DELAY seconds)

# nav2 and exploration are wrapped with run_with_log.sh so their console + rcl logs stage to /tmp/exprun_stage and the recorder copies them into runs/<run>/logs at shutdown. The recorder runs UNwrapped (it creates the run folder and does the copying; its own diagnostics already land in that folder).

# Usage:
#   ./launch_run.sh                 # default container/config/delay
#   DELAY=6 ./launch_run.sh         # wait 6s before recorder+exploration
#   CONTAINER=other ./launch_run.sh # attach to a different container
#
# Everything below is configurable via env vars (see the defaults).
set -euo pipefail

CONTAINER=${CONTAINER:-TRAVIS_ros2jazzy}
DELAY=${DELAY:-5}                       # seconds nav2 gets before the other two
# System-test dir INSIDE the container (repo mounts ./ros2_ws/src -> /ros2_ws/src).
SYS_DIR=${SYS_DIR:-/ros2_ws/src/navigation/exploration/tests/system}
CONFIG=${CONFIG:-baselines/baseline_house_aws.yaml}
MODE=${MODE:-slam}

# Each terminal opens its own `docker exec` into the container, so a stopped container would fail all three windows with a cryptic error. Check once, upfront.
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "error: container '$CONTAINER' is not running." >&2
    echo "start it first, or set CONTAINER=<name> (running: $(docker ps --format '{{.Names}}' | paste -sd, -))" >&2
    exit 1
fi

# The three commands, run from $SYS_DIR inside the container.
NAV2_CMD='./run_with_log.sh nav2 ros2 launch nav2 nav2.launch.py robot:=turtle3'

RECORDER_CMD="ros2 run exploration recorder.py \
    --config ${CONFIG} --mode ${MODE} \
    --ros-args -p use_sim_time:=true \
               -p lidar.scan_topic:=/panoramic/scan \
               -p lidar.base_frame:=base_footprint"

EXPLORE_CMD='./run_with_log.sh exploration ros2 launch exploration exploration.launch.py'

# Wrap a command so it: attaches to the container with an interactive login shell (so ~/.bashrc sources ROS), cds to the system-test dir, optionally sleeps, runs the command, then drops to an interactive prompt so the window stays open (and you can Ctrl-C / inspect) instead of closing when the process exits.
container_cmd() {
    local sleep_s=$1 inner=$2
    local pre=""
    [[ "$sleep_s" -gt 0 ]] && pre="echo 'waiting ${sleep_s}s for nav2...'; sleep ${sleep_s}; "
    printf 'docker exec -it %q bash -ic %q' "$CONTAINER" \
        "cd ${SYS_DIR}; ${pre}${inner}; echo; echo '[process exited — shell kept open]'; exec bash"
}

# Pick a terminal emulator that actually launches. gnome-terminal returns
# immediately (it just messages gnome-terminal-server over D-Bus), so `&` cannot
# tell us it failed — and in a ROS/snap shell the leaked LD_LIBRARY_PATH breaks
# gnome-terminal's and terminator's dynamic linker. So: strip the host
# LD_LIBRARY_PATH for the terminal process (the command inside runs via
# `docker exec`, which uses the container's own env — the host libs are
# irrelevant to it) and probe each emulator by actually running `true` in it.
# TERM_EMU=<name> forces a specific one.
_run_emu() {  # _run_emu <emu> <title> <cmd...>  — launches, backgrounded
    local emu=$1 title=$2 full=$3
    case "$emu" in
        gnome-terminal) env -u LD_LIBRARY_PATH gnome-terminal --title="$title" -- bash -c "$full" & ;;
        terminator)     env -u LD_LIBRARY_PATH terminator --title="$title" -x bash -c "$full" & ;;
        xterm)          env -u LD_LIBRARY_PATH xterm -T "$title" -e bash -c "$full" & ;;
        *) return 127 ;;
    esac
}

_probe_emu() {  # returns 0 if <emu> can open a window here
    local emu=$1
    command -v "$emu" >/dev/null 2>&1 || return 1
    # Actually try to open+close a trivial window; suppress its output.
    timeout 6 env -u LD_LIBRARY_PATH "$emu" -e true >/dev/null 2>&1 \
        || timeout 6 env -u LD_LIBRARY_PATH "$emu" -x true >/dev/null 2>&1 \
        || timeout 6 env -u LD_LIBRARY_PATH "$emu" -- true >/dev/null 2>&1
}

# Resolve the emulator once: honour $TERM_EMU, else first that actually works.
EMU=${TERM_EMU:-}
if [[ -z "$EMU" ]]; then
    for cand in gnome-terminal terminator xterm; do
        if _probe_emu "$cand"; then EMU=$cand; break; fi
    done
fi
if [[ -z "$EMU" ]]; then
    echo "error: no working terminal emulator found (tried gnome-terminal, terminator, xterm)." >&2
    echo "install one, or set TERM_EMU=<name>. Is DISPLAY set? DISPLAY='${DISPLAY:-}'." >&2
    exit 1
fi

open_term() {
    local title=$1 full=$2
    _run_emu "$EMU" "$title" "$full"
}

echo "Launching run: container=$CONTAINER delay=${DELAY}s config=$CONFIG mode=$MODE term=$EMU"

open_term "nav2"        "$(container_cmd 0      "$NAV2_CMD")"
open_term "recorder"    "$(container_cmd "$DELAY" "$RECORDER_CMD")"
open_term "exploration" "$(container_cmd "$DELAY" "$EXPLORE_CMD")"

echo "Opened 3 $EMU terminals (nav2 now; recorder + exploration after ${DELAY}s)."
