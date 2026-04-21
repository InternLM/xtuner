#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi

IFS=',' read -r -a XTUNER_TEST_VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
XTUNER_TEST_GPU_NUM="${#XTUNER_TEST_VISIBLE_GPUS[@]}"
if [[ "${XTUNER_TEST_GPU_NUM}" -ne 4 && "${XTUNER_TEST_GPU_NUM}" -ne 8 ]]; then
  echo "run_test.sh expects 4 or 8 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

XTUNER_TEST_GPU0="${XTUNER_TEST_VISIBLE_GPUS[0]//[[:space:]]/}"
for XTUNER_TEST_GPU in "${XTUNER_TEST_VISIBLE_GPUS[@]}"; do
  XTUNER_TEST_GPU="${XTUNER_TEST_GPU//[[:space:]]/}"
  if [[ ! "${XTUNER_TEST_GPU}" =~ ^[0-9]+$ ]]; then
    echo "run_test.sh expects numeric CUDA_VISIBLE_DEVICES entries, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
done

source ./zdev/env.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py312-pt28-raw

export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/xrt_${XTUNER_TEST_GPU0}g${XTUNER_TEST_GPU_NUM}_$$"
export XTUNER_DIST_PORT_BASE="$((35000 + XTUNER_TEST_GPU0 * 1024))"
export XTUNER_TEST_NUM_WORKERS="${XTUNER_TEST_GPU_NUM}"

echo "run_test.sh: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "run_test.sh: XTUNER_DIST_PORT_BASE=${XTUNER_DIST_PORT_BASE}"

pytest --durations=20 \
  tests/rl/test_agent_loop_utils.py \
  tests/rl/test_staleness_policy.py \
  tests/rl/test_replay_buffer.py \
  tests/rl/test_multi_task_agent_loop_manager.py \
  tests/rl/test_producer.py \
  tests/rl/test_async_rollout.py::TestOversampling \
  tests/rl/test_async_rollout.py::TestPartialRollout \
  tests/rl/test_async_rollout.py::TestTailBatch \
  tests/rl/test_rl_colocate_trainer.py \
  tests/rl/test_rl_disaggregated_trainer.py \
  tests/rl/test_agent_loop.py::TestAgentLoop::test_gsm8k_agent_loop_manager
