#!/usr/bin/env bash
# Run the EP+TP training unit test.
# Requires 4 GPUs (EP=2 * TP=2 * DP=1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-fla}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

XTUNER_USE_CUTLASS_GROUP_GEMM="${XTUNER_USE_CUTLASS_GROUP_GEMM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29533}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES
export XTUNER_USE_CUTLASS_GROUP_GEMM

cd "${REPO_ROOT}"
python -m pytest \
  tests/engine/test_moe_train_engine_tpep.py \
  -v \
  -x \
  --no-header
