#!/usr/bin/env bash
#
# diagnose_hang.sh — collect one snapshot of state for a hung xtuner/v1 RL job.
#
# Run this on ANY cluster node (head or worker). It looks only at local PIDs —
# so to get full coverage you'd run it on every node and concatenate outputs.
# Everything is written to a single directory under $OUT_DIR so you can tail /
# grep it later.
#
# Usage:
#   bash diagnose_hang.sh                          # auto-pick output dir
#   bash diagnose_hang.sh /path/to/out_dir         # explicit output dir
#
# Requires: py-spy (in the training venv), nvidia-smi, ray CLI.

set -u  # not -e: we want to collect whatever we can even if a step fails.

OUT_DIR="${1:-/tmp/xtuner_hang_$(hostname)_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
echo "[diagnose_hang] writing to $OUT_DIR"

# --------------------------------------------------------------------------
# 0. Meta — node identity + wall clock
# --------------------------------------------------------------------------
{
  echo "host=$(hostname)"
  echo "ip=$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo "date=$(date -Is)"
  echo "uptime=$(uptime)"
} > "$OUT_DIR/00_meta.txt"

# --------------------------------------------------------------------------
# 1. GPU — utilization + per-PID memory. All-zero util = clear hang signal.
# --------------------------------------------------------------------------
nvidia-smi > "$OUT_DIR/01_nvidia_smi.txt" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv > "$OUT_DIR/01_gpu_procs.csv" 2>&1
# 3 samples of util so you can see if GPUs do ANYTHING.
nvidia-smi dmon -s u -c 3 > "$OUT_DIR/01_gpu_util_sample.txt" 2>&1

# --------------------------------------------------------------------------
# 2. Ray — cluster state, actors, tasks, object store.
#    These only work on the head node (with RAY_ADDRESS set); on worker nodes
#    they'll error out and that's fine, we have py-spy below.
# --------------------------------------------------------------------------
ray status > "$OUT_DIR/02_ray_status.txt" 2>&1
ray list actors --detail --limit 500 > "$OUT_DIR/02_ray_actors.txt" 2>&1
ray list tasks --state RUNNING --limit 500 > "$OUT_DIR/02_ray_tasks_running.txt" 2>&1
ray list tasks --state PENDING_ARGS_AVAIL --limit 500 > "$OUT_DIR/02_ray_tasks_pending_args.txt" 2>&1
ray list tasks --state PENDING_OBJ_STORE_MEM_AVAIL --limit 500 > "$OUT_DIR/02_ray_tasks_pending_plasma.txt" 2>&1
ray summary tasks > "$OUT_DIR/02_ray_summary.txt" 2>&1
ray memory --stats-only > "$OUT_DIR/02_ray_memory.txt" 2>&1

# --------------------------------------------------------------------------
# 3. RoutedExpertStore stats — expected ~0 live keys between train steps;
#    steadily growing or wedged = suspect. Only runs if the store actor exists.
# --------------------------------------------------------------------------
python - > "$OUT_DIR/03_routed_expert_store.txt" 2>&1 <<'PY'
import sys
try:
    import ray
    ray.init(address="auto", namespace="routed_expert_store", ignore_reinit_error=True)
    actor = ray.get_actor("routed_expert_store_v1")
    print("stats:", ray.get(actor.stats.remote(), timeout=10))
except Exception as e:
    print(f"store probe failed: {e}", file=sys.stderr)
PY

# --------------------------------------------------------------------------
# 4. Process inventory — everything that could belong to our job.
# --------------------------------------------------------------------------
ps -eo pid,ppid,user,pcpu,pmem,rss,stat,etime,cmd \
  --sort=-rss \
  > "$OUT_DIR/04_ps_all.txt" 2>&1

# Narrow to candidates we actually care about.
ps -eo pid,stat,etime,cmd --sort=-etime \
  | grep -E "xtuner|lmdeploy|ray::|TrainingWorker|RolloutWorker|LMDeploy|Environment|TokenizeController|Judger|RoutedExpertStore" \
  | grep -v grep \
  > "$OUT_DIR/04_ps_filtered.txt" 2>&1

awk '{print $1}' "$OUT_DIR/04_ps_filtered.txt" | sort -u > "$OUT_DIR/04_pids.txt"

# --------------------------------------------------------------------------
# 5. py-spy dumps — the money shot. One file per PID so you can diff easily.
#    py-spy may be in a venv; try a couple of common locations.
# --------------------------------------------------------------------------
PYSPY="$(command -v py-spy || true)"
if [ -z "$PYSPY" ]; then
  for cand in \
    /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl/bin/py-spy \
    /mnt/shared-storage-user/liukuikun/miniconda3/envs/xtuner_dev/bin/py-spy; do
    if [ -x "$cand" ]; then PYSPY="$cand"; break; fi
  done
fi

mkdir -p "$OUT_DIR/05_pyspy"
if [ -z "$PYSPY" ]; then
  echo "py-spy not found; install with: pip install py-spy" \
    > "$OUT_DIR/05_pyspy/_MISSING.txt"
else
  echo "using py-spy at $PYSPY" > "$OUT_DIR/05_pyspy/_tool.txt"
  while read -r pid; do
    [ -z "$pid" ] && continue
    # Blocking dump (no --nonblocking) so we see the real async task stack
    # in AsyncIO Thread / uvloop — the interesting hangs live there, not in
    # MainThread. Pauses the target for a few ms; fine for diagnostics.
    # --subprocesses catches worker subprocesses (e.g. spawn-mode dataloader).
    timeout 20 "$PYSPY" dump --pid "$pid" --subprocesses \
      > "$OUT_DIR/05_pyspy/pid_${pid}.txt" 2>&1 \
      || echo "[timeout or failure for pid $pid]" >> "$OUT_DIR/05_pyspy/pid_${pid}.txt"
  done < "$OUT_DIR/04_pids.txt"
fi

# --------------------------------------------------------------------------
# 6. Heuristic summary — grep the stack dumps for the usual suspects so you
#    don't have to manually open every file.
# --------------------------------------------------------------------------
{
  echo "=== hang-pattern summary (counts across all py-spy dumps) ==="
  for pat in \
    "dist.barrier" \
    "ncclAllReduce" \
    "c10d::" \
    "ray\.get" \
    "ray\.wait" \
    "RayWaiter" \
    "plasma" \
    "aiohttp" \
    "http\.client" \
    "asyncio\.wait" \
    "asyncio\.Event" \
    "threading\.Event" \
    "Queue\.get" \
    "store_keys_to_release" \
    "_add_rollout_routed_experts" \
    "release_many" \
    "/v1/chat/completions" \
    "routed_expert_store"; do
    n=$(grep -rF "$pat" "$OUT_DIR/05_pyspy/" 2>/dev/null | wc -l)
    printf "  %6d  %s\n" "$n" "$pat"
  done
} > "$OUT_DIR/06_summary.txt"

# --------------------------------------------------------------------------
# 7. Training log tail — the last 500 lines can give you the "what happened
#    right before it went quiet" context. Edit the pattern if your log dir
#    layout differs.
# --------------------------------------------------------------------------
LATEST_LOG=$(ls -1t /mnt/shared-storage-user/llmit1/user/liukuikun/delivery/*/training_log.txt 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
  echo "# source: $LATEST_LOG" > "$OUT_DIR/07_training_log_tail.txt"
  tail -n 500 "$LATEST_LOG" >> "$OUT_DIR/07_training_log_tail.txt"
fi

# --------------------------------------------------------------------------
# 8. Done — print pointer + a tiny preview so the runner knows what they got.
# --------------------------------------------------------------------------
{
  echo ""
  echo "=== done. output under: $OUT_DIR ==="
  echo ""
  echo "--- 06_summary.txt ---"
  cat "$OUT_DIR/06_summary.txt"
  echo ""
  echo "--- ls $OUT_DIR ---"
  ls -la "$OUT_DIR"
} | tee "$OUT_DIR/_RESULT.txt"
