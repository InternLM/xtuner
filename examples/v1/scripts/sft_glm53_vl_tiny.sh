#!/usr/bin/env bash
# Before running, activate a Python environment with the training dependencies and torchrun.
# Export GLM5_3_MODEL_PATH and VL_DATA_PATH; set VL_MEDIA_ROOT if media uses another root.
# Set GPU visibility and other site-specific runtime settings externally.
# XTUNER_PATH defaults to this repository and is prepended to PYTHONPATH.
# This launcher does not set machine-specific paths or activate an environment.
set -euo pipefail

# GLM-5.3-Flash VL (image/video) end-to-end SFT smoke, the multimodal counterpart of
# sft_glm53_tiny.sh. Same knobs and the same acceptance matrix (default / SP2 / EP8 /
# XTUNER_ACTIVATION_OFFLOAD=0 / FP8=1 / MODEL_COMPILE=1), so the two modalities stay comparable.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
XTUNER_PATH="${XTUNER_PATH:-${REPO_ROOT}}"
export PYTHONPATH="${XTUNER_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

export CONFIG_PATH="${CONFIG_PATH:-examples/v1/config/sft_glm53_vl.py}"
export WORK_DIR="${WORK_DIR:-work_dirs/sft_glm53_vl/tiny}"
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29681}"

export EP_SIZE="${EP_SIZE:-4}"
export SP_SIZE="${SP_SIZE:-1}"
export DISPATCHER="${DISPATCHER:-all2all}"
export SPARSE_MLA_BACKEND="${SPARSE_MLA_BACKEND:-flash_mla_cudnn}"
export VISION_ATTN_IMPL="${VISION_ATTN_IMPL:-flash_attention}"

export DATASET_SAMPLE_RATIO="${DATASET_SAMPLE_RATIO:-1.0}"
# Unlike the text launcher's 4096, this matches PACK_MAX_LENGTH: truncated visual spans are
# dropped, and video samples can exceed 4096 tokens.
export SAMPLE_MAX_LENGTH="${SAMPLE_MAX_LENGTH:-16384}"
export PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-16384}"
# Caps the per-image patch count so one packed sample keeps room for text; the placeholder span
# has to fit inside PACK_MAX_LENGTH or the collator's invariant check fires.
export VL_MAX_PIXELS="${VL_MAX_PIXELS:-1048576}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NNODES * NPROC_PER_NODE / SP_SIZE))}"
export INTRA_LAYER_MICRO_BATCH="${INTRA_LAYER_MICRO_BATCH:-1}"
export TOTAL_STEP="${TOTAL_STEP:-20}"

export FP8="${FP8:-0}"
export MODEL_COMPILE="${MODEL_COMPILE:-0}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-2048}"
export SWAP_OPTIMIZER="${SWAP_OPTIMIZER:-0}"
export STRICT_LOAD="${STRICT_LOAD:-0}"
export DEBUG_SKIP_SAVE="${DEBUG_SKIP_SAVE:-1}"

export XTUNER_ACTIVATION_OFFLOAD="${XTUNER_ACTIVATION_OFFLOAD:-1}"
export XTUNER_GC_ENABLE="${XTUNER_GC_ENABLE:-1}"

export PROFILE_TIME="${PROFILE_TIME:-0}"
export PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
export PROFILE_STEP="${PROFILE_STEP:-8}"

mkdir -p "${WORK_DIR}"
if [[ "${NODE_RANK}" == "0" ]]; then
  cp -f "${SCRIPT_PATH}" "${WORK_DIR}/$(basename "${SCRIPT_PATH}")"
  cp -f "${CONFIG_PATH}" "${WORK_DIR}/$(basename "${CONFIG_PATH}")"
fi

current_time=$(date "+%m%d%H%M%S")
env | grep -E '^(GLM5_3_MODEL_PATH|VL_DATA_PATH|VL_MEDIA_ROOT|VL_MAX_PIXELS|CONFIG_PATH|WORK_DIR|NNODES|NPROC_PER_NODE|NODE_RANK|MASTER_ADDR|MASTER_PORT|SAMPLE_MAX_LENGTH|PACK_MAX_LENGTH|GLOBAL_BATCH_SIZE|INTRA_LAYER_MICRO_BATCH|TOTAL_STEP|EP_SIZE|SP_SIZE|DISPATCHER|SPARSE_MLA_BACKEND|VISION_ATTN_IMPL|FP8|MODEL_COMPILE|TORCH_COMPILE|LOSS_CHUNK_SIZE|SWAP_OPTIMIZER|STRICT_LOAD|DEBUG_SKIP_SAVE|PROFILE_[A-Z0-9_]+)=' | sort

torchrun \
  --nnodes "${NNODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  xtuner/v1/train/cli/sft.py \
  --config "${CONFIG_PATH}" \
  2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}_${NODE_RANK}.txt"
