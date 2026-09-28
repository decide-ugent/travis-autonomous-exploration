#!/usr/bin/env bash
# Persist the live slam_toolbox session as a REUSABLE map, in both formats TRAVIS needs.
#
# Why two formats, not one: they feed two different consumers and both are mandatory.
#   .posegraph + .data  slam_toolbox localization mode scan-matches against the STORED SCANS in this graph. This is the artifact that makes localization better than AMCL, and it is the one that cannot be regenerated later.
#   .pgm + .yaml        the exploration node plans on an occupancy grid. It decides static-vs-live from `_is_slam = not bool(map_file_path)` and that path is found by globbing map_dir for a .pgm (see exploration.launch.py:_resolve_map_paths). A graph-only folder therefore either fails the glob or, worse, leaves exploration believing it is doing live SLAM with the wrong stop rules.
# Both are written from the same instant so the grid and the graph always agree.
#
# Usage:
#   ros2 run nav2 save_map.sh <out_dir> [stem] [source_run]
#
#   out_dir     directory to write into, created if missing
#   stem        basename without extension (default: map)
#   source_run  optional provenance string recorded in source.txt
#
# IMPORTANT, the single sharpest deployment risk here: slam_toolbox resolves <out_dir> in ITS OWN process, not in this shell. Under Docker the path must be valid inside the slam_toolbox container; a path that only exists where this script runs yields RESULT_FAILED_TO_WRITE_FILE while this script sees a perfectly good local directory. On the ROSbot /ros2_ws/src is the shared bind mount, so paths under it resolve identically in both containers.
#
# Exit codes: 0 all good, 1 bad usage, 2 slam_toolbox not running (nothing to save), 3 a service call failed or produced no files.
set -uo pipefail

OUT_DIR=${1:-}
STEM=${2:-map}
SOURCE_RUN=${3:-}

if [[ -z "$OUT_DIR" ]]; then
    echo "usage: save_map.sh <out_dir> [stem] [source_run]" >&2
    exit 1
fi

# Service names are namespaced by the slam_toolbox node, matching nav2.launch.py's online_async include.
SERIALIZE_SRV=/slam_toolbox/serialize_map
SAVE_SRV=/slam_toolbox/save_map
# ros2 service call has no timeout option of its own, so each call is bounded externally. Generous because serializing a large graph genuinely takes a while, and a truncated write is worse than a slow one.
CALL_TIMEOUT=${SAVE_MAP_TIMEOUT:-60}

# A known_map run has no slam_toolbox at all. That is an ordinary situation, not a failure, so it exits distinctly (2) and the recorder can treat it as "nothing to do" rather than as an error.
if ! ros2 service list 2>/dev/null | grep -qx "$SERIALIZE_SRV"; then
    echo "[save_map] $SERIALIZE_SRV not available; slam_toolbox is not running. Nothing to save." >&2
    exit 2
fi

mkdir -p "$OUT_DIR" || { echo "[save_map] cannot create $OUT_DIR" >&2; exit 3; }
# slam_toolbox appends its own extensions to whatever it is given, so both services take a path WITHOUT one.
BASE="$(cd "$OUT_DIR" && pwd)/$STEM"

# Returns 0 only when the response carries result=0. Both services define RESULT_SUCCESS=0, and slam_toolbox reports failures in the response rather than by failing the call, so a call that "succeeded" with result=255 must not be mistaken for a save.
call_and_check() {
    local srv=$1 type=$2 request=$3 label=$4 out
    out=$(timeout "$CALL_TIMEOUT" ros2 service call "$srv" "$type" "$request" 2>&1)
    local rc=$?
    if [[ $rc -eq 124 ]]; then
        echo "[save_map] $label timed out after ${CALL_TIMEOUT}s" >&2
        return 1
    fi
    if [[ $rc -ne 0 ]]; then
        echo "[save_map] $label call failed (rc=$rc)" >&2
        echo "$out" >&2
        return 1
    fi
    # The response prints as e.g. "response:\nslam_toolbox.srv.SaveMap_Response(result=0)".
    if [[ "$out" =~ result=([0-9]+) ]]; then
        local code=${BASH_REMATCH[1]}
        if [[ "$code" != "0" ]]; then
            # 255 is RESULT_FAILED_TO_WRITE_FILE for serialize and RESULT_UNDEFINED_FAILURE for save; 1 is RESULT_NO_MAP_RECEIEVD (upstream's spelling). A write failure almost always means the path is wrong inside slam_toolbox's container, per the note at the top of this file.
            echo "[save_map] $label returned result=$code (non-zero means failure)" >&2
            return 1
        fi
    else
        echo "[save_map] $label produced no parseable result field" >&2
        echo "$out" >&2
        return 1
    fi
    echo "[save_map] $label ok"
    return 0
}

status=0

# The pose-graph goes first: it is the artifact that cannot be rebuilt from anything else, whereas the occupancy grid can always be re-rasterized from a graph later.
call_and_check "$SERIALIZE_SRV" slam_toolbox/srv/SerializePoseGraph \
    "{filename: '$BASE'}" "serialize_map" || status=3

# SaveMap wraps its name in a std_msgs/String, unlike SerializePoseGraph's bare string.
call_and_check "$SAVE_SRV" slam_toolbox/srv/SaveMap \
    "{name: {data: '$BASE'}}" "save_map" || status=3

# Verify the files actually landed. The services can report success while writing nothing if the path resolved elsewhere in slam_toolbox's filesystem, and that failure is invisible without this check.
missing=()
for ext in posegraph data pgm yaml; do
    [[ -s "$BASE.$ext" ]] || missing+=("$STEM.$ext")
done
if (( ${#missing[@]} )); then
    echo "[save_map] MISSING or empty after save: ${missing[*]}" >&2
    echo "[save_map] If the service calls reported ok, the path resolved differently inside the slam_toolbox container. See the note at the top of this script." >&2
    status=3
fi

# Provenance is deliberately .txt and NOT .yaml: exploration's _resolve_map_paths takes glob('*.yaml')[0] and glob order is not guaranteed, so a second YAML here could be loaded as the map descriptor instead of map.yaml.
{
    echo "saved_at: $(date -Iseconds)"
    echo "stem: $STEM"
    [[ -n "$SOURCE_RUN" ]] && echo "source_run: $SOURCE_RUN"
    echo "host: $(hostname)"
} > "$OUT_DIR/source.txt"

if [[ $status -eq 0 ]]; then
    echo "[save_map] wrote $STEM.{posegraph,data,pgm,yaml} into $OUT_DIR"
fi
exit $status
