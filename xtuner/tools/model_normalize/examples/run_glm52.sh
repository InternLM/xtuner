#!/usr/bin/env bash
set -Eeuo pipefail

if (($# < 1)); then
  echo "usage: $0 {bf16|fp8}" >&2
  exit 2
fi
MODE="$1"
SOURCE_DIR="${SOURCE_DIR:?set SOURCE_DIR to the source HF directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the output root}"
SHARD_SIZE_GB="${SHARD_SIZE_GB:-4}"
MAX_SAVE_WORKERS="${MAX_SAVE_WORKERS:-4}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../run_model_normalize.sh"

EXTRA_ARGS=()
if [[ -n "${BASE_MODEL_DIR}" ]]; then
  EXTRA_ARGS+=(--base-model-dir "${BASE_MODEL_DIR}")
fi

case "${MODE}" in
  bf16)
    exec bash "${RUNNER}" repack \
      --source "${SOURCE_DIR}" \
      --output "${OUTPUT_ROOT}/20_hf_bf16_mtp" \
      --shard-size-gb "${SHARD_SIZE_GB}" \
      "${EXTRA_ARGS[@]}"
    ;;
  fp8)
    : "${REFERENCE_DIR:?set REFERENCE_DIR for FP8, or use the generic CLI with --policy heuristic}"
    exec bash "${RUNNER}" to-fp8 \
      --source "${SOURCE_DIR}" \
      --output "${OUTPUT_ROOT}/20_hf_fp8_mtp" \
      --reference "${REFERENCE_DIR}" \
      --shard-size-gb "${SHARD_SIZE_GB}" \
      --max-save-workers "${MAX_SAVE_WORKERS}" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "unsupported mode: ${MODE}; expected bf16 or fp8" >&2
    exit 2
    ;;
esac
