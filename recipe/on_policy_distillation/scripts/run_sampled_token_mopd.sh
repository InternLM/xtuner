#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/launch_teacher_utils.sh"

export STUDENT_MODEL_PATH=${1:?"Usage: $0 STUDENT_MODEL_PATH DATA_PATH"}
export MODEL_PATH="${STUDENT_MODEL_PATH}"
export DATA_PATH=${2:?"Usage: $0 STUDENT_MODEL_PATH DATA_PATH"}
export GSM8K_TEACHER_MODEL_PATH=${GSM8K_TEACHER_MODEL_PATH:?"GSM8K_TEACHER_MODEL_PATH is required"}
export GEO3K_TEACHER_MODEL_PATH=${GEO3K_TEACHER_MODEL_PATH:?"GEO3K_TEACHER_MODEL_PATH is required"}

export OPD_CONFIG_PATH="${OPD_CONFIG_PATH:-recipe/on_policy_distillation/config/rl_dapo_math_mopd.py}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
export TEACHER_STARTUP_TIMEOUT_S="${TEACHER_STARTUP_TIMEOUT_S:-1200}"
export PYTHONUNBUFFERED=1

USE_SGLANG=${XTUNER_USE_SGLANG:-0}
USE_LMDEPLOY=${XTUNER_USE_LMDEPLOY:-0}
USE_VLLM=${XTUNER_USE_VLLM:-0}

if [[ "${USE_SGLANG}" == "1" && "${USE_LMDEPLOY}" == "0" && "${USE_VLLM}" == "0" ]]; then
    OPD_BACKEND="sglang"
elif [[ "${USE_SGLANG}" == "0" && "${USE_LMDEPLOY}" == "1" && "${USE_VLLM}" == "0" ]]; then
    OPD_BACKEND="lmdeploy"
else
    echo "Exactly one of XTUNER_USE_SGLANG and XTUNER_USE_LMDEPLOY must be set to 1; XTUNER_USE_VLLM must be 0." >&2
    exit 1
fi

export XTUNER_USE_SGLANG="${USE_SGLANG}"
export XTUNER_USE_LMDEPLOY="${USE_LMDEPLOY}"
export XTUNER_USE_VLLM="${USE_VLLM}"

export WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work_dirs/dapo_math_mopd}"
export OPD_CONFIG_FILE="${REPO_ROOT}/${OPD_CONFIG_PATH}"

TRAINING_STARTED=0

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    stop_teacher_servers

    if (( TRAINING_STARTED )); then
        ray stop --force >/dev/null 2>&1 || true
    fi

    exit "${exit_code}"
}

trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM

start_teacher_servers "${OPD_CONFIG_FILE}" "${OPD_BACKEND}" "${WORK_DIR}"
wait_for_teacher_servers "${TEACHER_STARTUP_TIMEOUT_S}"

echo "All ${#TEACHER_NAMES[@]} teachers are ready."
echo "Starting MOPD training with student GPUs: ${STUDENT_CUDA_VISIBLE_DEVICES}"

cd "${REPO_ROOT}"
TRAINING_STARTED=1
CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
    bash -o pipefail examples/v1/scripts/run_rl.sh \
    "${OPD_CONFIG_FILE}" \
    "${OPD_BACKEND}" \
    "${STUDENT_MODEL_PATH}" \
    "${DATA_PATH}" \
    "${EVAL_DATA_PATH}"
