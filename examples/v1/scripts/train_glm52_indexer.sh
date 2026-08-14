#!/usr/bin/env bash
set -euo pipefail

# Train GLM-5.2 and its main-stack source indexers jointly. Activate the
# intended Python environment before invoking this script.
: "${GLM5_2_MODEL_PATH:?GLM5_2_MODEL_PATH is required}"

export DATASET_TYPE="${DATASET_TYPE:-alpaca}"
case "${DATASET_TYPE}" in
  alpaca)
    : "${ALPACA_PATH:?ALPACA_PATH is required when DATASET_TYPE=alpaca}"
    ;;
  alpaca_long)
    : "${ALPACA_LONG_PATH:?ALPACA_LONG_PATH is required when DATASET_TYPE=alpaca_long}"
    ;;
  *)
    echo "Unsupported DATASET_TYPE=${DATASET_TYPE}; use alpaca or alpaca_long." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/examples/v1/config/sft_glm5p2.py}"
export WORK_DIR="${2:-${WORK_DIR:-work_dirs/glm52_indexer_sft}}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export TRAIN_DSA_INDEXER="${TRAIN_DSA_INDEXER:-1}"
export INDEXER_LOSS_COEFF="${INDEXER_LOSS_COEFF:-1.0}"
export INDEXER_ONLY="${INDEXER_ONLY:-0}"
export INDEXER_DEBUG_INTERVAL="${INDEXER_DEBUG_INTERVAL:-0}"
export SPARSE_MLA_BACKEND="${SPARSE_MLA_BACKEND:-cudnn_dsa}"

# These values reflect the constraints enforced by sft_glm5p2.py while
# source-indexer training is enabled.
export SP_SIZE="${SP_SIZE:-1}"
export INTRA_LAYER_MICRO_BATCH="${INTRA_LAYER_MICRO_BATCH:-1}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0}"
export MODEL_COMPILE="${MODEL_COMPILE:-0}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"

export EP_SIZE="${EP_SIZE:-4}"
export TOTAL_STEP="${TOTAL_STEP:-300}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export DO_CLIP="${DO_CLIP:-1}"

export DATASET_SAMPLE_RATIO="${DATASET_SAMPLE_RATIO:-1.0}"
export SAMPLE_MAX_LENGTH="${SAMPLE_MAX_LENGTH:-4096}"
export PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-4096}"
export CACHE_TAG="${CACHE_TAG:-glm52_indexer_4096}"

export FP8="${FP8:-1}"
export DEBUG_SKIP_SAVE="${DEBUG_SKIP_SAVE:-0}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
export HF_INTERVAL="${HF_INTERVAL:-${TOTAL_STEP}}"
export HF_MAX_KEEP="${HF_MAX_KEEP:-1}"
export PROFILE_TIME="${PROFILE_TIME:-0}"
export PROFILE_MEMORY="${PROFILE_MEMORY:-0}"

NNODES="${NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${REPO_ROOT}"
test -f "${CONFIG_PATH}"
mkdir -p "${WORK_DIR}"
ulimit -n 65536

command=(
  torchrun
  "--nproc-per-node=${NPROC_PER_NODE}"
  "--master-addr=${MASTER_ADDR}"
  "--master-port=${MASTER_PORT}"
  "--nnodes=${NNODES}"
  "--node-rank=${NODE_RANK}"
  --tee 3
  -m xtuner.v1.train.cli.sft
  --config "${CONFIG_PATH}"
)

if [[ "${DRY_RUN:-0}" != "0" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

"${command[@]}" 2>&1 | tee -a "${WORK_DIR}/node_${NODE_RANK}.txt"
