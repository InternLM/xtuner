#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if (( $# != 0 )); then
    echo "This script does not accept positional arguments." >&2
    echo "Set STUDENT_MODEL_PATH, TEACHER_MODEL_PATH, and DATA_PATH before running it." >&2
    exit 2
fi

: "${STUDENT_MODEL_PATH:?STUDENT_MODEL_PATH is required}"
: "${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH is required}"
: "${DATA_PATH:?DATA_PATH is required}"

STUDENT_CUDA_VISIBLE_DEVICES="0,1,2,3"
TEACHER_CUDA_VISIBLE_DEVICES="7"
TEACHER_HOST="127.0.0.1"
TEACHER_PORT="13141"
TEACHER_TP_SIZE="1"
TEACHER_CHUNKED_PREFILL_SIZE="4096"
TEACHER_GPU_MEMORY_UTILIZATION="0.6"
TEACHER_STARTUP_TIMEOUT_S="1200"

WORK_DIR="${REPO_ROOT}/work_dirs/dapo_math_opd"
TEACHER_ENDPOINT="http://${TEACHER_HOST}:${TEACHER_PORT}"
TEACHER_LOG_FILE="${WORK_DIR}/teacher.log"

if [[ ! -d "${STUDENT_MODEL_PATH}" ]]; then
    echo "Student model directory does not exist: ${STUDENT_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
    echo "Teacher model directory does not exist: ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Training data file does not exist: ${DATA_PATH}" >&2
    exit 1
fi

for required_command in python curl ray setsid; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command is not available: ${required_command}" >&2
        exit 1
    fi
done

python -c "import sglang" >/dev/null
mkdir -p "${WORK_DIR}"

if curl -sf --max-time 2 "${TEACHER_ENDPOINT}/health_generate" >/dev/null; then
    echo "A teacher service is already running at ${TEACHER_ENDPOINT}" >&2
    exit 1
fi

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
        if curl -sf --max-time 2 "${TEACHER_ENDPOINT}/health_generate" >/dev/null; then
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
echo "Teacher GPUs: ${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Teacher log: ${TEACHER_LOG_FILE}"

setsid env \
    CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
    PYTHONUNBUFFERED=1 \
    python -m sglang.launch_server \
    --model-path "${TEACHER_MODEL_PATH}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP_SIZE}" \
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${TEACHER_GPU_MEMORY_UTILIZATION}" \
    >"${TEACHER_LOG_FILE}" 2>&1 &
TEACHER_PID=$!

wait_for_teacher
curl -sS --max-time 10 "${TEACHER_ENDPOINT}/get_model_info"
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
    sglang \
    "${STUDENT_MODEL_PATH}" \
    "${DATA_PATH}"
