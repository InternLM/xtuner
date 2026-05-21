#!/bin/bash
set -euo pipefail

# ========================== 配置区 ==========================
CONFIG="${CONFIG:-autotest/config/qwen3_moe_30BA3_ep8.py}"
CKPT_MODE="${CKPT_MODE:-sync}"          # sync 或 async
ASYNC_CHECKPOINT_TYPE="${ASYNC_CHECKPOINT_TYPE:-process}"  # thread 或 process，仅 async 模式生效
NPROC=8
NUM_CORES="${NUM_CORES:-12}"
TOTAL_STEPS="${TOTAL_STEPS:-40}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1}"
CKPT_PROCESS_CONTROL="${CKPT_PROCESS_CONTROL:-1}"
CKPT_PROCESS_NUM_CPUS="${CKPT_PROCESS_NUM_CPUS:-0}"
CKPT_PROCESS_CPUS="${CKPT_PROCESS_CPUS:-}"
CKPT_PROCESS_NICE="${CKPT_PROCESS_NICE:-19}"
CKPT_PROCESS_IONICE_CLASS="${CKPT_PROCESS_IONICE_CLASS:-2}"
CKPT_PROCESS_IONICE_LEVEL="${CKPT_PROCESS_IONICE_LEVEL:-7}"
ASYNC_DCP_SHARE_MEMORY="${ASYNC_DCP_SHARE_MEMORY:-1}"
ASYNC_DCP_SHARE_MEMORY_PINNED="${ASYNC_DCP_SHARE_MEMORY_PINNED:-0}"
ASYNC_DCP_THREAD_COUNT="${ASYNC_DCP_THREAD_COUNT:-1}"
ASYNC_DCP_FILE_WRITE_LOCK_SLOTS="${ASYNC_DCP_FILE_WRITE_LOCK_SLOTS:-4}"
ASYNC_DCP_FILE_WRITE_LOCK_KEY="${ASYNC_DCP_FILE_WRITE_LOCK_KEY:-}"

QWEN3_MOE_PATH=/mnt/shared-storage-user/llmrazor-share/,/model/Qwen3-30B-A3B
ALPACA_PATH=/mnt/shared-storage-user/llmrazor-share/,/data/alpaca
# ============================================================

resolve_cpu_affinity_layout() {
    python - "$NUM_CORES" "$CKPT_PROCESS_CONTROL" "$CKPT_PROCESS_NUM_CPUS" "$CKPT_PROCESS_CPUS" <<'PY'
import os
import sys


def parse_cpu_list(spec):
    cpus = set()
    if not spec:
        return cpus
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_cpu_list(cpus):
    cpus = sorted(cpus)
    if not cpus:
        return ""
    ranges = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


num_cores = int(sys.argv[1])
ckpt_control = sys.argv[2] == "1"
ckpt_num = int(sys.argv[3])
requested_ckpt = sys.argv[4]

available = sorted(os.sched_getaffinity(0))
available_set = set(available)
if not available:
    raise SystemExit("no available CPUs from os.sched_getaffinity(0)")

if requested_ckpt:
    ckpt = parse_cpu_list(requested_ckpt)
    missing = sorted(ckpt - available_set)
    if missing:
        raise SystemExit(f"CKPT_PROCESS_CPUS contains CPUs outside current affinity: {missing}")
elif ckpt_control and ckpt_num > 0:
    if len(available) <= ckpt_num:
        raise SystemExit(
            f"not enough CPUs to reserve checkpoint cores: total {len(available)}, requested {ckpt_num}"
        )
    ckpt = set(available[-ckpt_num:])
else:
    ckpt = set()

candidates = [cpu for cpu in available if cpu not in ckpt]
if num_cores <= 0:
    train = set(candidates)
elif len(candidates) >= num_cores:
    train = set(candidates[:num_cores])
else:
    raise SystemExit(
        f"not enough CPUs for training: need {num_cores}, available {len(candidates)} after checkpoint reserve"
    )

if ckpt and train & ckpt:
    raise SystemExit(f"training CPUs and checkpoint CPUs overlap: {sorted(train & ckpt)}")

print(f"CPU_CORES={format_cpu_list(train)}")
print(f"CKPT_PROCESS_CPUS={format_cpu_list(ckpt)}")
print(f"NUM_CORES={len(train)}")
print(f"AVAILABLE_CPUS={format_cpu_list(available)}")
PY
}
eval "$(resolve_cpu_affinity_layout)"

ASYNC_FLAG=0
[[ "$CKPT_MODE" == "async" ]] && ASYNC_FLAG=1

CONFIG_NAME=$(basename "$CONFIG" .py)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
if [[ "$CKPT_MODE" == "async" ]]; then
    RUN_NAME="${CONFIG_NAME}-${CKPT_MODE}-${ASYNC_CHECKPOINT_TYPE}-${NUM_CORES}cores-${TOTAL_STEPS}steps-ckpt${CKPT_INTERVAL}"
    if [[ "$ASYNC_CHECKPOINT_TYPE" == "process" && "$CKPT_PROCESS_CONTROL" == "1" ]]; then
        if [[ -n "$CKPT_PROCESS_CPUS" ]]; then
            RUN_NAME="${RUN_NAME}-ckptctrl-cpus${CKPT_PROCESS_CPUS//,/_}-nice${CKPT_PROCESS_NICE}-ionice${CKPT_PROCESS_IONICE_CLASS}${CKPT_PROCESS_IONICE_LEVEL}"
        else
            RUN_NAME="${RUN_NAME}-ckptctrl-nice${CKPT_PROCESS_NICE}-ionice${CKPT_PROCESS_IONICE_CLASS}${CKPT_PROCESS_IONICE_LEVEL}"
        fi
    fi
    RUN_NAME="${RUN_NAME}-${TIMESTAMP}"
    ASYNC_DCP_FILE_WRITE_LOCK_KEY="${ASYNC_DCP_FILE_WRITE_LOCK_KEY:-$RUN_NAME}"
else
    RUN_NAME="${CONFIG_NAME}-${CKPT_MODE}-${NUM_CORES}cores-${TOTAL_STEPS}steps-ckpt${CKPT_INTERVAL}-${TIMESTAMP}"
fi
LOG_DIR="logs/${RUN_NAME}"
WORK_DIR="work_dirs/${RUN_NAME}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}-train.log"
MONITOR_CSV="${LOG_DIR}/${RUN_NAME}-monitor.csv"
CKPT_PROCESS_CONTROL_LOG="${LOG_DIR}/${RUN_NAME}-ckpt-process-control.log"

mkdir -p "$LOG_DIR" "$WORK_DIR"
chmod 777 "$LOG_DIR" "$WORK_DIR"

echo ""
echo "================================================================"
echo "[INFO] 开始 ${CKPT_MODE} 训练"
echo "[INFO] ASYNC_CHECKPOINT_TYPE = $ASYNC_CHECKPOINT_TYPE"
echo "[INFO] CKPT_PROCESS_CONTROL = $CKPT_PROCESS_CONTROL"
echo "[INFO] AVAILABLE_CPUS = $AVAILABLE_CPUS"
echo "[INFO] CPU_CORES = $CPU_CORES"
echo "[INFO] NUM_CORES = $NUM_CORES"
if [[ "$CKPT_PROCESS_CONTROL" == "1" ]]; then
    echo "[INFO] CKPT_PROCESS_NUM_CPUS = $CKPT_PROCESS_NUM_CPUS"
    echo "[INFO] CKPT_PROCESS_CPUS = $CKPT_PROCESS_CPUS"
    echo "[INFO] CKPT_PROCESS_NICE = $CKPT_PROCESS_NICE"
    echo "[INFO] CKPT_PROCESS_IONICE_CLASS = $CKPT_PROCESS_IONICE_CLASS"
    echo "[INFO] CKPT_PROCESS_IONICE_LEVEL = $CKPT_PROCESS_IONICE_LEVEL"
fi
echo "[INFO] RUN_NAME  = $RUN_NAME"
echo "[INFO] WORK_DIR  = $WORK_DIR"
echo "[INFO] TRAIN_LOG = $TRAIN_LOG"
echo "[INFO] MONITOR   = $MONITOR_CSV"
echo "================================================================"

# ---------------------- 后台监控函数 ----------------------

collect_descendants() {
    local root_pid=$1
    local children child
    children=$(pgrep -P "$root_pid" 2>/dev/null || true)
    for child in $children; do
        echo "$child"
        collect_descendants "$child"
    done
}

checkpoint_process_controller() {
    local train_pid=$1
    local log_file=$2

    echo "timestamp,pid,action,result" > "$log_file"

    declare -A configured_pids=()

    while kill -0 "$train_pid" 2>/dev/null; do
        local descendants p cmdline
        descendants=$(collect_descendants "$train_pid")
        for p in $descendants; do
            if [[ -n "${configured_pids[$p]:-}" ]]; then
                continue
            fi

            if [[ ! -r "/proc/$p/cmdline" ]]; then
                continue
            fi
            cmdline=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
            if [[ -z "$cmdline" || "$cmdline" != *"spawn_main"* ]]; then
                continue
            fi

            configured_pids[$p]=1
            echo "$(date +%s),$p,detected,\"$cmdline\"" >> "$log_file"

            if [[ -n "$CKPT_PROCESS_CPUS" ]]; then
                if taskset -pc "$CKPT_PROCESS_CPUS" "$p" >> "$log_file" 2>&1; then
                    echo "$(date +%s),$p,taskset,ok" >> "$log_file"
                else
                    echo "$(date +%s),$p,taskset,failed" >> "$log_file"
                fi
            fi

            if [[ -n "$CKPT_PROCESS_NICE" ]]; then
                if renice -n "$CKPT_PROCESS_NICE" -p "$p" >> "$log_file" 2>&1; then
                    echo "$(date +%s),$p,renice,ok" >> "$log_file"
                else
                    echo "$(date +%s),$p,renice,failed" >> "$log_file"
                fi
            fi

            if [[ -n "$CKPT_PROCESS_IONICE_CLASS" ]]; then
                local ionice_cmd=(ionice -c "$CKPT_PROCESS_IONICE_CLASS")
                if [[ -n "$CKPT_PROCESS_IONICE_LEVEL" ]]; then
                    ionice_cmd+=(-n "$CKPT_PROCESS_IONICE_LEVEL")
                fi
                ionice_cmd+=(-p "$p")
                if "${ionice_cmd[@]}" >> "$log_file" 2>&1; then
                    echo "$(date +%s),$p,ionice,ok" >> "$log_file"
                else
                    echo "$(date +%s),$p,ionice,failed" >> "$log_file"
                fi
            fi
        done

        sleep 1
    done
}

process_tree_monitor() {
    local train_pid=$1
    local csv_file=$2
    local clk_tck
    clk_tck=$(getconf CLK_TCK)

    echo "elapsed_s,rss_gib,pss_gib,cpu_pct" > "$csv_file"

    local start_time
    start_time=$(date +%s.%N)
    local prev_ticks=0
    local prev_time=$start_time

    sleep 1

    while kill -0 "$train_pid" 2>/dev/null; do
        local all_pids
        all_pids=$(collect_descendants "$train_pid")
        all_pids="$train_pid $all_pids"

        local total_rss_kb=0 total_pss_kb=0 total_ticks=0

        for p in $all_pids; do
            local rss_kb
            rss_kb=$(awk '/^VmRSS:/{print $2}' /proc/"$p"/status 2>/dev/null || echo 0)
            total_rss_kb=$((total_rss_kb + rss_kb))

            local pss_kb
            pss_kb=$(awk '/^Pss:/{sum+=$2} END{print sum+0}' /proc/"$p"/smaps_rollup 2>/dev/null || echo 0)
            total_pss_kb=$((total_pss_kb + pss_kb))

            local utime stime
            read -r utime stime <<< "$(awk '{print $14, $15}' /proc/"$p"/stat 2>/dev/null || echo '0 0')"
            total_ticks=$((total_ticks + utime + stime))
        done

        local now
        now=$(date +%s.%N)
        local elapsed
        elapsed=$(awk "BEGIN {printf \"%.1f\", $now - $start_time}")
        local dt
        dt=$(awk "BEGIN {printf \"%.6f\", $now - $prev_time}")

        local cpu_pct="0.0"
        local dticks=$((total_ticks - prev_ticks))
        if [[ $(awk "BEGIN {print ($dt > 0.01)}") == "1" ]]; then
            cpu_pct=$(awk "BEGIN {printf \"%.1f\", $dticks / $clk_tck / $dt * 100}")
        fi

        local rss_gib pss_gib
        rss_gib=$(awk "BEGIN {printf \"%.2f\", $total_rss_kb / 1048576}")
        pss_gib=$(awk "BEGIN {printf \"%.2f\", $total_pss_kb / 1048576}")

        echo "${elapsed},${rss_gib},${pss_gib},${cpu_pct}" >> "$csv_file"

        prev_ticks=$total_ticks
        prev_time=$now

        sleep "$MONITOR_INTERVAL"
    done
}

echo "ASYNC_DCP_SHARE_MEMORY=$ASYNC_DCP_SHARE_MEMORY"
echo "ASYNC_DCP_SHARE_MEMORY_PINNED=$ASYNC_DCP_SHARE_MEMORY_PINNED"
echo "ASYNC_DCP_THREAD_COUNT=$ASYNC_DCP_THREAD_COUNT"
echo "ASYNC_DCP_FILE_WRITE_LOCK_SLOTS=$ASYNC_DCP_FILE_WRITE_LOCK_SLOTS"
echo "ASYNC_DCP_FILE_WRITE_LOCK_KEY=$ASYNC_DCP_FILE_WRITE_LOCK_KEY"
# ---------------------- 启动训练 ----------------------
taskset -c "$CPU_CORES" env -u PYTHONPATH \
    PYTHONPATH=$PWD \
    MALLOC_ARENA_MAX=2 \
    XTUNER_GC_ENABLE=1 \
    XTUNER_DISABLE_HF_SAVE=1 \
    QWEN3_MOE_PATH="$QWEN3_MOE_PATH" \
    ALPACA_PATH="$ALPACA_PATH" \
    WORK_DIR="$WORK_DIR" \
    ASYNC_CHECKPOINT="$ASYNC_FLAG" \
    ASYNC_CHECKPOINT_TYPE="$ASYNC_CHECKPOINT_TYPE" \
    ASYNC_DCP_SHARE_MEMORY="$ASYNC_DCP_SHARE_MEMORY" \
    ASYNC_DCP_SHARE_MEMORY_PINNED="$ASYNC_DCP_SHARE_MEMORY_PINNED" \
    ASYNC_DCP_THREAD_COUNT="$ASYNC_DCP_THREAD_COUNT" \
    ASYNC_DCP_FILE_WRITE_LOCK_SLOTS="$ASYNC_DCP_FILE_WRITE_LOCK_SLOTS" \
    ASYNC_DCP_FILE_WRITE_LOCK_KEY="$ASYNC_DCP_FILE_WRITE_LOCK_KEY" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    CKPT_INTERVAL="$CKPT_INTERVAL" \
    torchrun --nproc_per_node="$NPROC" xtuner/v1/train/cli/sft.py \
    --config "$CONFIG" \
    > >(tee "$TRAIN_LOG") 2>&1 &

TRAIN_PID=$!

# 启动监控
process_tree_monitor "$TRAIN_PID" "$MONITOR_CSV" &
MONITOR_PID=$!

CKPT_PROCESS_CONTROLLER_PID=""
if [[ "$CKPT_MODE" == "async" && "$ASYNC_CHECKPOINT_TYPE" == "process" && "$CKPT_PROCESS_CONTROL" == "1" ]]; then
    checkpoint_process_controller "$TRAIN_PID" "$CKPT_PROCESS_CONTROL_LOG" &
    CKPT_PROCESS_CONTROLLER_PID=$!
fi

# 等待训练结束
wait "$TRAIN_PID"
EXIT_CODE=$?

# 停止监控
kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null || true
if [[ -n "$CKPT_PROCESS_CONTROLLER_PID" ]]; then
    kill "$CKPT_PROCESS_CONTROLLER_PID" 2>/dev/null
    wait "$CKPT_PROCESS_CONTROLLER_PID" 2>/dev/null || true
fi

# # 删除 work_dir
# echo ""
# echo "[INFO] 删除 work_dir: $WORK_DIR"
# rm -rf "$WORK_DIR"

echo ""
echo "========== 训练完成 =========="
echo "  模式:     $CKPT_MODE"
echo "  退出码:   $EXIT_CODE"
echo "  训练日志: $TRAIN_LOG"
echo "  监控数据: $MONITOR_CSV"
if [[ -f "$CKPT_PROCESS_CONTROL_LOG" ]]; then
    echo "  CKPT控制: $CKPT_PROCESS_CONTROL_LOG"
fi
echo "=============================="
