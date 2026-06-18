#!/usr/bin/env bash
set -u

# Collect one lightweight Ray/Linux debugging snapshot.
#
# This script is intended to be run on the Ray head node first. If a problem is
# suspected on worker nodes, run it there as well: Ray CLI outputs are cluster
# scoped, while top_rss/smaps outputs are local-node scoped.
#
# Example:
#
#   WORK_DIR=/path/to/ray/work_dir \
#   OUT_ROOT=/path/to/debug_output \
#   SMAPS_TOP_N=30 \
#   RAY_STATE_LIST_LIMIT=100 \
#   bash .dev_scripts/debug_rl_snapshot.sh
#
# Long-running tmux loop:
#
#   while true; do
#     WORK_DIR=/path/to/ray/work_dir OUT_ROOT=/path/to/debug_output \
#       bash .dev_scripts/debug_rl_snapshot.sh
#     sleep 180
#   done
#
# Optional environment variables:
#   WORK_DIR                Directory to cd into before running Ray CLI.
#                           Default: current directory.
#   OUT_ROOT                Parent directory for mem_YYYYMMDD_HHMMSS snapshots.
#                           Default: WORK_DIR. Falls back to /tmp if not
#                           writable.
#   SMAPS_TOP_N             Number of highest-RSS local processes for which to
#                           copy /proc/<pid>/smaps_rollup. Default: 20.
#   DEBUG_PIDS              Extra whitespace-separated PIDs to include in smaps
#                           collection even if they are not in the top N.
#   RAY_STATE_LIST_LIMIT    Limit for `ray list ... --detail` JSON outputs.
#                           Default: 100. Increase only when debugging task,
#                           actor, or object skew in detail.
#   INCLUDE_OBJECT_DETAILS  When 1, also collect `ray list objects --detail`.
#                           Default: 0 because object details can be large.

WORK_DIR=${WORK_DIR:-$(pwd)}
OUT_ROOT=${OUT_ROOT:-${WORK_DIR}}
SMAPS_TOP_N=${SMAPS_TOP_N:-20}
INCLUDE_OBJECT_DETAILS=${INCLUDE_OBJECT_DETAILS:-0}
RAY_STATE_LIST_LIMIT=${RAY_STATE_LIST_LIMIT:-100}

cd "$WORK_DIR"

current_time=$(date "+%Y%m%d_%H%M%S")
output_dir="${OUT_ROOT}/mem_${current_time}"
if ! mkdir -p "$output_dir" 2>/dev/null; then
    OUT_ROOT="/tmp/xtuner_ray_debug_${USER:-unknown}"
    output_dir="${OUT_ROOT}/mem_${current_time}"
    mkdir -p "$output_dir"
fi

run_capture() {
    local output_file=$1
    shift

    {
        echo "# command: $*"
        echo "# started_at: $(date --iso-8601=seconds)"
        "$@"
        status=$?
        echo "# finished_at: $(date --iso-8601=seconds)"
        echo "# exit_status: ${status}"
        return "${status}"
    } > "${output_file}" 2>&1 || true
}

collect_top_rss() {
    ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss && return 0

    # Fallback for environments where ps cannot sort or has a reduced feature
    # set. This keeps the script usable in minimal containers.
    echo "PID PPID RSS VSZ COMM ARGS"
    for status_file in /proc/[0-9]*/status; do
        pid=${status_file#/proc/}
        pid=${pid%/status}
        ppid=$(awk '/^PPid:/ { print $2 }' "${status_file}" 2>/dev/null)
        rss=$(awk '/^VmRSS:/ { print $2 }' "${status_file}" 2>/dev/null)
        vsz=$(awk '/^VmSize:/ { print $2 }' "${status_file}" 2>/dev/null)
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null)
        args=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)
        [[ -z "${rss}" ]] && rss=0
        [[ -z "${vsz}" ]] && vsz=0
        printf '%s %s %s %s %s %s\n' \
            "${pid}" "${ppid:-0}" "${rss}" "${vsz}" "${comm:-unknown}" "${args:-}"
    done | sort -k3,3nr
}

{
    echo "timestamp=${current_time}"
    echo "date=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "include_object_details=${INCLUDE_OBJECT_DETAILS}"
    echo "smaps_top_n=${SMAPS_TOP_N}"
    echo "ray_state_list_limit=${RAY_STATE_LIST_LIMIT}"
} > "${output_dir}/meta.txt"

run_capture "${output_dir}/ray_status.txt" ray status
run_capture "${output_dir}/summary_objects.txt" ray summary objects
run_capture "${output_dir}/memory.txt" ray memory --stats-only
run_capture "${output_dir}/summary_tasks.txt" ray summary tasks
run_capture "${output_dir}/summary_actors.txt" ray summary actors
run_capture "${output_dir}/tasks.json" ray list tasks --detail --format=json --limit "${RAY_STATE_LIST_LIMIT}"
run_capture "${output_dir}/actors.json" ray list actors --detail --format=json --limit "${RAY_STATE_LIST_LIMIT}"

if [[ "${INCLUDE_OBJECT_DETAILS}" == "1" ]]; then
    run_capture "${output_dir}/objects.json" ray list objects --detail --format=json --limit "${RAY_STATE_LIST_LIMIT}"
fi

run_capture "${output_dir}/top_rss.txt" collect_top_rss

awk -v n="${SMAPS_TOP_N}" '$1 ~ /^[0-9]+$/ && count < n { print $1; count++ }' \
    "${output_dir}/top_rss.txt" > "${output_dir}/smaps_pids.txt"

if [[ -n "${DEBUG_PIDS:-}" ]]; then
    printf '%s\n' ${DEBUG_PIDS} >> "${output_dir}/smaps_pids.txt"
fi

sort -n -u "${output_dir}/smaps_pids.txt" -o "${output_dir}/smaps_pids.txt"

while read -r pid; do
    [[ -z "${pid}" ]] && continue
    if [[ -r "/proc/${pid}/smaps_rollup" ]]; then
        cp "/proc/${pid}/smaps_rollup" "${output_dir}/smaps_${pid}.txt"
    else
        echo "Cannot read /proc/${pid}/smaps_rollup" > "${output_dir}/smaps_${pid}.txt"
    fi
done < "${output_dir}/smaps_pids.txt"

echo "Saved to ${output_dir}/"
