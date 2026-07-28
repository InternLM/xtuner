#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

TEACHER_MODEL_PATH=${1:?"Usage: $0 TEACHER_MODEL_PATH"}
TEACHER_HOST=${TEACHER_HOST:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
TEACHER_CHUNKED_PREFILL_SIZE=${TEACHER_CHUNKED_PREFILL_SIZE:-4096}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.6}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
LMDEPLOY_PATH=${LMDEPLOY_PATH:-"${REPO_ROOT}/work_dirs/lmdeploy"}

USE_SGLANG=${XTUNER_USE_SGLANG:-0}
USE_LMDEPLOY=${XTUNER_USE_LMDEPLOY:-0}
USE_VLLM=${XTUNER_USE_VLLM:-0}

if [[ "${USE_SGLANG}" == "1" && "${USE_LMDEPLOY}" == "0" && "${USE_VLLM}" == "0" ]]; then
    exec "${PYTHON_EXECUTABLE}" -m sglang.launch_server \
        --model-path "${TEACHER_MODEL_PATH}" \
        --host "${TEACHER_HOST}" \
        --port "${TEACHER_PORT}" \
        --tp "${TEACHER_TP_SIZE}" \
        --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
        --mem-fraction-static "${TEACHER_GPU_MEMORY_UTILIZATION}"
fi

if [[ "${USE_SGLANG}" == "0" && "${USE_LMDEPLOY}" == "1" && "${USE_VLLM}" == "0" ]]; then
    export PYTHONPATH="${LMDEPLOY_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${PYTHON_EXECUTABLE}" -m lmdeploy serve api_server "${TEACHER_MODEL_PATH}" \
        --backend pytorch \
        --role Hybrid \
        --logprobs-mode raw_logprobs \
        --server-name "${TEACHER_HOST}" \
        --server-port "${TEACHER_PORT}" \
        --tp "${TEACHER_TP_SIZE}" \
        --max-prefill-token-num "${TEACHER_CHUNKED_PREFILL_SIZE}" \
        --cache-max-entry-count "${TEACHER_GPU_MEMORY_UTILIZATION}"
fi

echo "Exactly one of XTUNER_USE_SGLANG and XTUNER_USE_LMDEPLOY must be set to 1; XTUNER_USE_VLLM must be 0." >&2
exit 1
