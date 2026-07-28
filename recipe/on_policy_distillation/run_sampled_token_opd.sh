#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

STUDENT_MODEL_PATH="$1"
TEACHER_MODEL_PATH="$2"
DATA_PATH="$3"

STUDENT_CUDA_VISIBLE_DEVICES="0,1,2,3"
TEACHER_CUDA_VISIBLE_DEVICES="7"
TEACHER_HOST="127.0.0.1"
TEACHER_PORT="13141"
TEACHER_TP_SIZE="1"
TEACHER_CHUNKED_PREFILL_SIZE="4096"
TEACHER_GPU_MEMORY_UTILIZATION="0.6"
TEACHER_STARTUP_TIMEOUT_S="1200"

USE_SGLANG=${XTUNER_USE_SGLANG:-0}
USE_LMDEPLOY=${XTUNER_USE_LMDEPLOY:-0}
USE_VLLM=${XTUNER_USE_VLLM:-0}

if [[ "${USE_SGLANG}" == "1" && "${USE_LMDEPLOY}" == "0" && "${USE_VLLM}" == "0" ]]; then
    OPD_BACKEND="sglang"
    TEACHER_HEALTH_PATH="health_generate"
    TEACHER_MODEL_INFO_PATH="get_model_info"
elif [[ "${USE_SGLANG}" == "0" && "${USE_LMDEPLOY}" == "1" && "${USE_VLLM}" == "0" ]]; then
    OPD_BACKEND="lmdeploy"
    TEACHER_HEALTH_PATH="health"
    TEACHER_MODEL_INFO_PATH="v1/models"
else
    echo "Exactly one of XTUNER_USE_SGLANG and XTUNER_USE_LMDEPLOY must be set to 1; XTUNER_USE_VLLM must be 0." >&2
    exit 1
fi

export XTUNER_USE_SGLANG="${USE_SGLANG}"
export XTUNER_USE_LMDEPLOY="${USE_LMDEPLOY}"
export XTUNER_USE_VLLM="${USE_VLLM}"

WORK_DIR="${REPO_ROOT}/work_dirs/dapo_math_opd"
TEACHER_ENDPOINT="http://${TEACHER_HOST}:${TEACHER_PORT}"
TEACHER_LOG_FILE="${WORK_DIR}/teacher.log"

mkdir -p "${WORK_DIR}"

TEACHER_PID=""
TRAINING_STARTED=0

cleanup() {
    local exit_code=$?
    local attempt

    trap - EXIT INT TERM

    if [[ -n "${TEACHER_PID}" ]]; then
        kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || true
        for ((attempt = 0; attempt < 30; attempt++)); do
            if ! kill -0 -- "-${TEACHER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 -- "-${TEACHER_PID}" 2>/dev/null; then
            kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || true
        fi
        wait "${TEACHER_PID}" 2>/dev/null || true
    fi

    if (( TRAINING_STARTED )); then
        ray stop --force >/dev/null 2>&1 || true
    fi

    exit "${exit_code}"
}

wait_for_teacher() {
    local deadline=$((SECONDS + TEACHER_STARTUP_TIMEOUT_S))

    while (( SECONDS < deadline )); do
        if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
            echo "Teacher process exited before becoming ready." >&2
            tail -n 50 "${TEACHER_LOG_FILE}" >&2 || true
            return 1
        fi
        if curl -sf --max-time 2 "${TEACHER_ENDPOINT}/${TEACHER_HEALTH_PATH}" >/dev/null; then
            return 0
        fi
        echo "Waiting for teacher service at ${TEACHER_ENDPOINT}..."
        sleep 5
    done

    echo "Teacher service did not become ready within ${TEACHER_STARTUP_TIMEOUT_S} seconds." >&2
    tail -n 50 "${TEACHER_LOG_FILE}" >&2 || true
    return 1
}

trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM

echo "Starting teacher model: ${TEACHER_MODEL_PATH}"
echo "Teacher backend: ${OPD_BACKEND}"
echo "Teacher GPUs: ${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Teacher log: ${TEACHER_LOG_FILE}"

setsid env \
    CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
    PYTHONUNBUFFERED=1 \
    TEACHER_HOST="${TEACHER_HOST}" \
    TEACHER_PORT="${TEACHER_PORT}" \
    TEACHER_TP_SIZE="${TEACHER_TP_SIZE}" \
    TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE}" \
    TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    bash "${SCRIPT_DIR}/start_pg_opd_teacher.sh" "${TEACHER_MODEL_PATH}" \
    >"${TEACHER_LOG_FILE}" 2>&1 &
TEACHER_PID=$!

wait_for_teacher
curl -sS --max-time 10 "${TEACHER_ENDPOINT}/${TEACHER_MODEL_INFO_PATH}"
echo
echo "Teacher service is ready at ${TEACHER_ENDPOINT}"
echo "Starting Pure PG-OPD training with student GPUs: ${STUDENT_CUDA_VISIBLE_DEVICES}"

export WORK_DIR
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
TRAINING_STARTED=1
CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
    bash -o pipefail examples/v1/scripts/run_rl.sh \
    recipe/on_policy_distillation/rl_dapo_math_opd.py \
    "${OPD_BACKEND}" \
    "${STUDENT_MODEL_PATH}" \
    "${DATA_PATH}"
