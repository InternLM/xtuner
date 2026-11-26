#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export STUDENT_MODEL_PATH=${1:?"Usage: $0 STUDENT_MODEL_PATH TEACHER_MODEL_PATH DATA_PATH"}
export MODEL_PATH="${STUDENT_MODEL_PATH}"
export TEACHER_MODEL_PATH=${2:?"Usage: $0 STUDENT_MODEL_PATH TEACHER_MODEL_PATH DATA_PATH"}
export DATA_PATH=${3:?"Usage: $0 STUDENT_MODEL_PATH TEACHER_MODEL_PATH DATA_PATH"}

export OPD_CONFIG_PATH="${OPD_CONFIG_PATH:-recipe/on_policy_distillation/config/rl_qwen35_35b_topk_mopd_train_teacher_disk.py}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
export OPD_CONFIG_FILE="${REPO_ROOT}/${OPD_CONFIG_PATH}"
export WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work_dirs/qwen35_35b_topk_mopd_train_teacher_disk}"
export PYTHONUNBUFFERED=1

export XTUNER_RL_NUM_WORKERS="${XTUNER_RL_NUM_WORKERS:-8}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export GRPC_ENABLE_HTTP_PROXY="${GRPC_ENABLE_HTTP_PROXY:-0}"
export XTUNER_USE_LMDEPLOY="${XTUNER_USE_LMDEPLOY:-1}"
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PROMPT_REPEAT_K="${PROMPT_REPEAT_K:-4}"
export EVALUATE_STEP="${EVALUATE_STEP:-5}"
export TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-5}"
export TEACHER_PREFETCH="${TEACHER_PREFETCH:-1}"

unset NVSHMEM_DEBUG
export NVSHMEM_INFO="${NVSHMEM_INFO:-0}"

mkdir -p "${WORK_DIR}"
cd "${REPO_ROOT}"

# Keep the retry behavior of run_rl.sh when Ray's resource table is temporarily empty.
bash examples/v1/scripts/run_rl.sh \
    "${OPD_CONFIG_FILE}" \
    lmdeploy \
    "${STUDENT_MODEL_PATH}" \
    "${DATA_PATH}" \
    "${EVAL_DATA_PATH}"
