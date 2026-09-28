#!/usr/bin/env bash
# Launch a ROS2 command with its console output tee'd to a named log file, and
# its per-node rcl logs routed to a matching sub-directory, so the exploration
# and nav2 processes (started in separate terminals) produce clearly separated
# logs instead of the default timestamp-named ~/.ros/log dirs.
#
# nav2/exploration start (in their own terminals) BEFORE the recorder, so the
# recorder's run folder does not exist yet at launch time. All wrappers therefore
# write to ONE fixed STAGING dir; the recorder copies it into runs/<run>/logs and
# then clears it, so the next run starts clean (see recorder._adopt_staged_logs).
# The fixed path means NO env var to export in each terminal — just run:
#
#   # terminal 1:
#   ./run_with_log.sh nav2        ros2 launch nav2 nav2.launch.py robot:=turtle3 ...
#   # terminal 2 (a few seconds later):
#   ./run_with_log.sh exploration ros2 launch exploration exploration.launch.py ...
#   # terminal 3:
#   ./run_with_log.sh recorder    ros2 run exploration recorder.py --config ...
#
# Staging layout (copied into runs/<run>/logs/ by the recorder at shutdown):
#   <stage>/<tag>.log           full console output (stdout+stderr, tee'd live)
#   <stage>/<tag>_rcl/          ROS per-node .log files (ROS_LOG_DIR for this proc)
#
# Override the staging path with RUN_LOG_STAGE if you ever need to (both the
# wrapper and the recorder honour it); by default it is /tmp/exprun_stage.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <tag> <command...>" >&2
    echo "  e.g. $0 nav2 ros2 launch nav2 nav2.launch.py robot:=turtle3" >&2
    exit 2
fi

tag=$1
shift

log_root=${RUN_LOG_STAGE:-/tmp/exprun_stage}
mkdir -p "$log_root/${tag}_rcl"
console_log="$log_root/${tag}.log"

# Route ROS's own per-node logs to a tag-specific dir (differentiates nav2 vs
# exploration node logs, which otherwise land in one shared timestamped dir).
export ROS_LOG_DIR="$log_root/${tag}_rcl"

echo "[run_with_log] tag=$tag  console=$console_log  rcl=$ROS_LOG_DIR"
echo "[run_with_log] cmd: $*"

# stdbuf keeps the tee live (line-buffered) so the console still updates in real
# time. 2>&1 merges stderr so warnings/errors are in the same ordered log.
stdbuf -oL -eL "$@" 2>&1 | tee "$console_log"
