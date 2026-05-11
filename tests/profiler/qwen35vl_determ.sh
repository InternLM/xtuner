#!/usr/bin/env bash
# 使用后 4 张 GPU 连跑两次 Qwen3.5-VL 最小 step，并比较不同 TRITON_CACHE_DIR 下的梯度。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 默认使用现有训练脚本相同的 conda 环境；需要外部环境时可设置 SKIP_CONDA_ACTIVATE=1。
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-fla}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC="${NPROC:-4}"
export NUM_LAYERS="${1:-${NUM_LAYERS:-8}}"
EXTRA_ARGS=("${@:2}")

export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307}"
export DATA_PATH="${DATA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/ci_vl}"
export MEDIA_ROOT="${MEDIA_ROOT:-${DATA_PATH}}"

# 这些开关与 zdev/sft_qwen35_mengke.sh 保持一致。
export XTUNER_DETERMINISTIC="${XTUNER_DETERMINISTIC:-true}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-0}"
export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK="${TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d%H%M%S)}"
OUT_DIR="${OUT_DIR:-work_dirs/qwen35vl_acc/qwen35vl_determ_layer${NUM_LAYERS}_gpu${NPROC}/${RUN_TAG}}"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/torch_custom_triton_cache}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
SCRIPT="tests/profiler/qwen35vl_determ.py"

mkdir -p "${OUT_DIR}" "${TRITON_CACHE_ROOT}"

echo "[qwen35vl_determ.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[qwen35vl_determ.sh] NUM_LAYERS=${NUM_LAYERS} OUT_DIR=${OUT_DIR}"

run_once() {
  local run_id="$1"
  local record_path="${OUT_DIR}/run${run_id}/grads"
  local cache_dir
  if [[ -n "${TRITON_CACHE_DIR_RUN1:-}" && "${run_id}" == "1" ]]; then
    cache_dir="${TRITON_CACHE_DIR_RUN1}"
  elif [[ -n "${TRITON_CACHE_DIR_RUN2:-}" && "${run_id}" == "2" ]]; then
    cache_dir="${TRITON_CACHE_DIR_RUN2}"
  elif [[ "${SAME_TRITON_CACHE:-0}" == "1" ]]; then
    cache_dir="${TRITON_CACHE_ROOT}/qwen35vl_determ_layer${NUM_LAYERS}_same_${RUN_TAG}"
  elif [[ "${TRITON_CACHE_MODE:-zdev}" == "zdev" ]]; then
    # 默认复用 zdev/sft_qwen35_mengke.sh 的 cache 命名，便于直接复现 exp1/exp2。
    cache_dir="${TRITON_CACHE_ROOT}/tmp_layer${NUM_LAYERS}_gpu${NPROC}_exp${run_id}"
  else
    cache_dir="${TRITON_CACHE_ROOT}/qwen35vl_determ_layer${NUM_LAYERS}_run${run_id}_${RUN_TAG}"
  fi
  if [[ "${CLEAR_TRITON_CACHE:-0}" == "1" ]]; then
    rm -rf "${cache_dir}"
  fi
  mkdir -p "$(dirname "${record_path}")" "${cache_dir}"
  export TRITON_CACHE_DIR="${cache_dir}"

  echo "[qwen35vl_determ.sh] run${run_id}: TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  torchrun --nproc-per-node "${NPROC}" --master-port "${MASTER_PORT}" "${SCRIPT}" \
    --record-path "${record_path}" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --media-root "${MEDIA_ROOT}" \
    --num-layers "${NUM_LAYERS}" \
    --deterministic \
    "${@:2}"
}

run_once 1 "${EXTRA_ARGS[@]}"
run_once 2 "${EXTRA_ARGS[@]}" --compare "${OUT_DIR}/run1/grads"
