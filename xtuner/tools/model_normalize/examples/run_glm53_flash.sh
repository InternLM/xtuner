#!/usr/bin/env bash
# One-command GLM-5.3-Flash normalization: BF16 repack and/or FP8 conversion.
#
#   bash run_glm53_flash.sh {bf16|fp8|both}
#
# Required environment:
#   SOURCE_DIR       input HF directory (training export)
#   OUTPUT_ROOT      output root; products land in
#                    ${OUTPUT_ROOT}/20_hf_bf16_mtp and ${OUTPUT_ROOT}/20_hf_fp8_mtp
#   REFERENCE_DIR    fp8/both only: directory holding the reference release's
#                    model.safetensors.index.json (and config.json)
#
# Optional environment:
#   SHARD_SIZE_GB    default 5 (matches the GLM-5.3-Flash release shard count)
#   MAX_SAVE_WORKERS default 4 (parallel FP8 shard-save workers)
#   DTYPE_MAP        JSON map of tensor name -> dtype used to align storage
#                    dtypes with the reference release. Defaults to
#                    <staging>/reference/glm53_flash_bf16_dtypes.json when that
#                    file exists; set DTYPE_MAP= to disable alignment.
#   BASE_MODEL_DIR   default unset; when set, LICENSE/generation_config are
#                    stamped into the product (not used for the
#                    official-aligned GLM-5.3-Flash release)
set -Eeuo pipefail

MODE="${1:-}"
case "${MODE}" in
  bf16 | fp8 | both) ;;
  "")
    echo "usage: $0 {bf16|fp8|both}" >&2
    exit 2
    ;;
  *)
    echo "unsupported mode: ${MODE}; expected bf16, fp8 or both" >&2
    exit 2
    ;;
esac

SOURCE_DIR="${SOURCE_DIR:?set SOURCE_DIR to the source HF directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the output root}"
SHARD_SIZE_GB="${SHARD_SIZE_GB:-5}"
MAX_SAVE_WORKERS="${MAX_SAVE_WORKERS:-4}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"

if [[ "${MODE}" != "bf16" && -z "${REFERENCE_DIR:-}" ]]; then
  echo "error: set REFERENCE_DIR for '${MODE}' to the directory holding the reference model.safetensors.index.json" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../run_model_normalize.sh"

if [[ -z "${DTYPE_MAP+x}" ]]; then
  DEFAULT_DTYPE_MAP="$(cd "${SCRIPT_DIR}/../../../.." 2>/dev/null && pwd)/reference/glm53_flash_bf16_dtypes.json"
  if [[ -f "${DEFAULT_DTYPE_MAP}" ]]; then
    DTYPE_MAP="${DEFAULT_DTYPE_MAP}"
  else
    DTYPE_MAP=""
  fi
fi

EXTRA_ARGS=()
if [[ -n "${BASE_MODEL_DIR}" ]]; then
  EXTRA_ARGS+=(--base-model-dir "${BASE_MODEL_DIR}")
fi
if [[ -n "${DTYPE_MAP}" ]]; then
  [[ -f "${DTYPE_MAP}" ]] || { echo "error: DTYPE_MAP not found: ${DTYPE_MAP}" >&2; exit 2; }
  EXTRA_ARGS+=(--dtype-map "${DTYPE_MAP}")
fi

echo "[run_glm53_flash] mode=${MODE}"
echo "[run_glm53_flash] SOURCE_DIR=${SOURCE_DIR}"
echo "[run_glm53_flash] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run_glm53_flash] SHARD_SIZE_GB=${SHARD_SIZE_GB}"
if [[ "${MODE}" != "bf16" ]]; then
  echo "[run_glm53_flash] REFERENCE_DIR=${REFERENCE_DIR}"
  echo "[run_glm53_flash] MAX_SAVE_WORKERS=${MAX_SAVE_WORKERS}"
fi
if [[ -n "${DTYPE_MAP}" ]]; then
  echo "[run_glm53_flash] DTYPE_MAP=${DTYPE_MAP}"
else
  echo "[run_glm53_flash] DTYPE_MAP=disabled (no alignment)"
fi
if [[ -n "${BASE_MODEL_DIR}" ]]; then
  echo "[run_glm53_flash] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
fi

run_bf16() {
  echo "[run_glm53_flash] repack -> ${OUTPUT_ROOT}/20_hf_bf16_mtp"
  bash "${RUNNER}" repack \
    --source "${SOURCE_DIR}" \
    --output "${OUTPUT_ROOT}/20_hf_bf16_mtp" \
    --shard-size-gb "${SHARD_SIZE_GB}" \
    "${EXTRA_ARGS[@]}"
}

run_fp8() {
  echo "[run_glm53_flash] to-fp8 -> ${OUTPUT_ROOT}/20_hf_fp8_mtp"
  bash "${RUNNER}" to-fp8 \
    --source "${SOURCE_DIR}" \
    --output "${OUTPUT_ROOT}/20_hf_fp8_mtp" \
    --reference "${REFERENCE_DIR}" \
    --shard-size-gb "${SHARD_SIZE_GB}" \
    --max-save-workers "${MAX_SAVE_WORKERS}" \
    "${EXTRA_ARGS[@]}"
}

case "${MODE}" in
  bf16) run_bf16 ;;
  fp8) run_fp8 ;;
  both)
    run_bf16
    run_fp8
    ;;
esac
