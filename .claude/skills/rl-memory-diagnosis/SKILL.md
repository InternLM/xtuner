---
name: rl-memory-diagnosis
description: Diagnose XTuner RL training memory issues - driver heap growth, Ray object store pressure, GC death spiral, and RolloutState retention. Use when RL training hangs or slows down, driver logs stop, or memory usage is abnormal during rollout/training.
---

# RL Training Memory Diagnosis

Diagnose four failure modes: GC death spiral, object store pressure, driver heap bloat, ABORTED
inventory retention.

**Scope**: `[MemReport]` lines and the periodic JSONL sample the *calling process* — in RL the single
driver process on the head node, which is where the ReplayBuffer/RolloutState heap lives. Worker-side
memory is separate; see "Worker-side memory". Event lines are always emitted on the `debug_mem`
branch; the periodic JSONL needs `XTUNER_MEMORY_WATCH_ENABLED=1`.

**Files** (paths relative to `$RUN`):

```text
driver.log                        # [MemReport], steps, buffer counts
lmdeploy.log
nodes/0/ray/session_*/logs/       # python-core-driver-*.log (GC), raylet.out (store)
memory/rank_unknown.jsonl         # periodic sampler (opt-in; RL driver has no RANK)
mem_*/actor_memory.json           # per-actor RSS/GPU on the local node (XTUNER_RL_MEM_DIR)
```

## Step 1: Hung or slow?

```bash
tail -5 $RUN/driver.log            # last timestamp vs now
# driver pid alive? check the python process with huge RSS on the head node
```

- Log stopped + process alive → GC death spiral / asyncio starvation
- Log stopped + process dead → crash (find traceback)
- Log slow → inspect produce_batch timing trend

## Step 2: LocalGC statistics

```bash
awk '/core_worker.cc:454: Event stats/{ts=$1" "$2} /grpc_server.LocalGC - /{print ts" || "$0}' \
    $RUN/nodes/0/ray/session_*/logs/python-core-driver-*.log | tail -10
```

| Pattern | Diagnosis |
|---|---|
| count < 10, mean < 100 ms, active = 0 | Normal (periodic GC) |
| count growing fast, mean > 1 s | GC slowing down |
| count > 100, mean > 60 s, active > 0 | **Death spiral** |
| mean grows linearly | Heap is growing → Step 4 |

**Threshold**: `mean_execution_time / interval > 0.5` → process spends >50% time in GC.

## Step 3: Object store water level

```bash
grep -a "object_store.cc:37" $RUN/nodes/0/ray/session_*/logs/raylet.out | tail -10
grep -a "Allowing the Plasma store" $RUN/nodes/0/ray/session_*/logs/raylet.out   # capacity
```

| Usage | Risk |
|---|---|
| < 50% | Safe |
| 50–70% | Warning |
| > 70% | **Danger** — global GC every 30 s on all nodes |

Default trigger: `plasma_store_usage_trigger_gc_threshold = 0.7`.

## Step 4: Driver process memory

Two `[MemReport]` lines per train step, always on (driver only): after `take_train_batch`
(`tag=agent_loop_manager`) and at step end (`tag=rl_trainer`).

```text
[MemReport][rl_trainer][pid=12345][train_step=3] rss=1.42GiB anon=1.11 shmem=0.05 file=0.26 swap=0.00 hwm=1.51
```

| Field | /proc source | Meaning |
|---|---|---|
| `rss` | `VmRSS` | Resident set size: all pages currently in RAM (anon + shmem + file) |
| `anon` | `RssAnon` | Anonymous private pages — Python heap/objects (the GC-cost driver) |
| `shmem` | `RssShmem` | Shared-memory pages, e.g. plasma object-store mappings, tmpfs |
| `file` | `RssFile` | File-backed pages: mmap'd libraries, weights files, page cache |
| `swap` | `VmSwap` | Pages currently swapped out (expect 0 on GPU nodes) |
| `hwm` | `VmHWM` | High-water mark: peak RSS since process start; never decreases |

All values are GiB. `rss` carries the `GiB` suffix; the others are bare numbers (legacy format from
the original `mem_report`). `hwm` is cumulative — use it to confirm a past peak, not current usage.

```bash
grep "\[MemReport\]" $RUN/driver.log | tail -10

# Optional periodic trace (start run with XTUNER_MEMORY_WATCH_ENABLED=1 XTUNER_MEMORY_WATCH_INTERVAL=30)
jq -c 'select(.event=="periodic") | [.time_ns, .process.RssAnon, .process.VmRSS]' \
    $RUN/memory/rank_unknown.jsonl | tail -20

# Fallback without instrumentation (head node only)
cat /proc/<driver_pid>/status | grep -E "VmRSS|RssAnon|RssShmem|RssFile"
```

JSONL values are raw bytes; `process.RssAnon` equals `anon=` in the event line. The sampler writes
`memory/rank_<RANK>.jsonl`; the RL driver has no `RANK` → `rank_unknown.jsonl`. `host_*.jsonl` is
only written when `LOCAL_RANK=0`, which RL drivers normally do not set.

| Metric | Normal | Concerning | Critical |
|---|---|---|---|
| RssAnon | < 10 GB | 10–50 GB | > 100 GB |
| RssShmem | < 5 GB | 5–20 GB | > 40 GB |
| VmRSS | < 20 GB | 20–80 GB | > 150 GB |

Estimate GC time: `gc_seconds ≈ RssAnon_GB × 2.5` (~0.25 µs/token × ~150 B/token).

## Worker-side memory (multi-node)

Driver lines cannot see worker processes. With `XTUNER_RL_MEM_DIR` set, `actor_memory.json` records
the local node's actors:

```bash
jq -c 'select(.TrainingWorker or .RayWorkerWrapper) | [.time, .TrainingWorker.mem_gb, .RayWorkerWrapper.mem_gb]' \
    $RUN/mem_*/actor_memory.json | tail -5
```

`TrainingWorker` holds training state and CPU-offloaded weights; `LMDeployWorker` is a thin ~1 GB
wrapper (**not** the engine); `RayEngineWorker` / `RayWorkerWrapper` are the real engine (large GPU
numbers appear there). Caveats: `mem_gb` is the actor's own RSS (children only counted for GPU), and
only the current node is covered. Cross-node reads require a process on that node (Ray task) or
node-level monitoring (node_exporter / Ray dashboard).

For per-rank host memory, call `log_memory_event(tag=f"train_rank{rank}", marker=...)` from the
worker's step hook (e.g. `RoleWorker` step log); it always samples the *calling* process.

## Step 5: RolloutState retention (ABORTED inventory)

```bash
grep -a "get_batch from buffer" $RUN/driver.log | grep -o "leftover_aborted=[0-9]*"
grep -a "produce_batch done\|Start produce_batch" $RUN/driver.log | tail -10
grep -a "batch_target=\|oversample_budget=" $RUN/driver.log | head -3
```

- Healthy: `leftover_aborted` bounded (< 2× batch_target), consumed next step.
- Unhealthy: grows monotonically → ABORTED RolloutStates stay in driver heap.

## Step 6: Conclusion

```text
GC:         normal / slowing / death spiral  (mean=XXs, active=N)
obj_store:  XX% (safe / warning / over 70% threshold)
driver_anon:XX GB (stable / growing / critical)
aborted:    N per step (bounded / accumulating)
conclusion: <one sentence>
```

## Step 7: Remediation (priority order)

Deep dive: `downloads/dev/driver-ray-gc-hang/gc-and-memory-model.md`. The trigger measures plasma
bytes, while `gc.collect()` cost scales with container/reference edges in the process heap; when
resident tokens are billions of `list[int]`, each pressure-triggered GC burns minutes of GIL and
releases nothing (data is normally referenced, not cyclic).

1. **Enlarge the store** so usage never crosses 70%: `RAY_object_store_memory=128000000000` (check
   /dev/shm headroom).
2. **Raise the trigger / slow the cadence**: `RAY_plasma_store_usage_trigger_gc_threshold=0.95`,
   `RAY_local_gc_min_interval_s=300`, `RAY_global_gc_min_interval_s=300` (spill-to-disk at ~0.9 is
   the backstop).
3. **Reduce resident tokens** (`driver_anon ≈ samples × actual_seq_len × 150 B`): lower
   `OVER_SAMPLE_THRESHOLD`, `MAX_RESPONSE_LENGTH` (keep `PACK_MAX_LENGTH` in sync), `TRAIN_BATCH_SIZE`,
   `PROMPT_REPEAT_K`; set `TAIL_BATCH_TRIGGER_SIZE` negative to drop expired groups.
