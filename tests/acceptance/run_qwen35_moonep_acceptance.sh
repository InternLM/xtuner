#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <deepep|moonep> <mtp:0|1> <pack-length> [acceptance-root]" >&2
    exit 2
fi

backend=$1
mtp=$2
pack_length=$3
acceptance_root=${4:-work_dirs/moonep_qwen35_acceptance}
# Resolve the checkout that owns this script so commit-level comparisons can
# run in isolated worktrees without importing the developer's dirty checkout.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
gpu_lock=/mnt/shared-storage-user/zhaopenghao/github/xtuner/zdev/gpu_lock.sh

if [[ $backend != "deepep" && $backend != "moonep" ]]; then
    echo "backend must be deepep or moonep" >&2
    exit 2
fi
if [[ $mtp != "0" && $mtp != "1" ]]; then
    echo "mtp must be 0 or 1" >&2
    exit 2
fi
if ! [[ $pack_length =~ ^[1-9][0-9]*$ ]]; then
    echo "pack-length must be a positive integer" >&2
    exit 2
fi

# Re-enter the exact same command while holding the repository-wide 8-GPU
# lock. The marker prevents recursively acquiring the non-reentrant lock.
if [[ ${MOONEP_ACCEPTANCE_LOCK_HELD:-0} != "1" ]]; then
    exec "$gpu_lock" env MOONEP_ACCEPTANCE_LOCK_HELD=1 "$0" "$@"
fi

source /mnt/shared-storage-user/zhaopenghao/miniconda3/etc/profile.d/conda.sh
conda activate pt212_cu132
cd "$repo_root"

run_dir="$acceptance_root/${backend}_mtp${mtp}_pack${pack_length}"
if [[ ${MOONEP_ACCEPTANCE_MICRO_BATCH:-1} != "1" ]]; then
    run_dir="${run_dir}_micro${MOONEP_ACCEPTANCE_MICRO_BATCH}"
fi
if [[ -e $run_dir ]]; then
    echo "refusing to mix acceptance attempts in existing directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"

export PYTHONPATH="$repo_root"
export MOONEP_ACCEPTANCE_BACKEND=$backend
export MOONEP_ACCEPTANCE_MTP=$mtp
export MOONEP_ACCEPTANCE_PACK_LENGTH=$pack_length
export MOONEP_ACCEPTANCE_WORK_DIR=$run_dir
export MOONEP_ACCEPTANCE_MODEL_PATH=${MOONEP_ACCEPTANCE_MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-35B-A3B}
export MOONEP_ACCEPTANCE_DATA_PATH=${MOONEP_ACCEPTANCE_DATA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/alpaca}
export MODEL_COMPILE=1
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-0}"
# FA3's Hopper backward has no deterministic path for head_dim 256 (Qwen3.5),
# so the FA3 comparison runs non-deterministically; keep the default otherwise.
export XTUNER_DETERMINISTIC="${XTUNER_DETERMINISTIC:-true}"
export XTUNER_ACTIVATION_OFFLOAD=0
export XTUNER_COMPILE_NO_INPLACE_BUFFERS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROFILE_RANKS=${PROFILE_RANKS:-0}
export TRITON_CACHE_DIR="$run_dir/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$run_dir/torchinductor_cache"
unset XTUNER_USE_CUTLASS_GROUP_GEMM
unset GROUPED_GEMM_USE_CUTLASS

config=tests/acceptance/sft_qwen35_moonep_acceptance.py
python -m xtuner._testing.moonep_acceptance capture \
    --config "$config" \
    --output "$run_dir/acceptance_manifest.json"

# VMM/NCCL allocations are not all visible to PyTorch's allocator counters.
# Sample all eight devices while the same GPU lock covers the training job.
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits \
    --loop-ms=500 > "$run_dir/device_memory.csv" &
memory_monitor_pid=$!
trap 'kill "$memory_monitor_pid" 2>/dev/null || true; wait "$memory_monitor_pid" 2>/dev/null || true' EXIT

torchrun \
    --nproc-per-node 8 \
    --master-port "${MOONEP_ACCEPTANCE_MASTER_PORT:-29618}" \
    xtuner/v1/train/cli/sft.py \
    --config "$config" \
    2>&1 | tee "$run_dir/stdout.log"
