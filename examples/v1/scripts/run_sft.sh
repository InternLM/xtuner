#!/usr/bin/env bash
set -euo pipefail

# QWEN3_5_MOE_PATH and ALPACA_PATH can be initialized by sourcing zdev/env.sh.
# Set DATA_PATH and MEDIA_ROOT instead when training multimodal data.
: "${QWEN3_5_MOE_PATH:?QWEN3_5_MOE_PATH is required}"
: "${DATA_PATH:=${ALPACA_PATH:?DATA_PATH or ALPACA_PATH is required}}"
export QWEN3_5_MOE_PATH DATA_PATH
export MEDIA_ROOT="${MEDIA_ROOT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/examples/v1/config/sft_qwen3p5.py}"
export WORK_DIR="${2:-${WORK_DIR:-work_dirs/qwen3p5_sft}}"

XTUNER_PATH="${XTUNER_PATH:-${REPO_ROOT}}"
export PYTHONPATH="${XTUNER_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

export XTUNER_ACTIVATION_OFFLOAD="${XTUNER_ACTIVATION_OFFLOAD:-0}"
export XTUNER_GC_ENABLE="${XTUNER_GC_ENABLE:-1}"
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-1}"
# FA3 does not support deterministic backward for Qwen3.5's head_dim=256.
if [[ "${XTUNER_USE_FA3}" == "1" ]]; then
  export XTUNER_DETERMINISTIC=false
fi
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NNODES="${NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"

ulimit -n 65536
mkdir -p "${WORK_DIR}"

torchrun \
  --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --tee 3 \
  -m xtuner.v1.train.cli.sft \
  --config "${CONFIG_PATH}" \
  2>&1 | tee -a "${WORK_DIR}/node_${NODE_RANK}.txt"
