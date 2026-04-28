#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认使用用户指定的 fla 环境；需要切换时可在命令前覆盖 CONDA_ENV。
CONDA_ENV="${CONDA_ENV:-fla}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 本脚本固定验证 EP=2, TP=2。
EP_SIZE="${EP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$((EP_SIZE * TP_SIZE * DP_SIZE))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29532}"
XTUNER_USE_CUTLASS_GROUP_GEMM="${XTUNER_USE_CUTLASS_GROUP_GEMM:-1}"

# 显式使用当前仓库代码，避免导入 conda 环境或其他目录下安装的 xtuner。
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES
export EP_SIZE
export TP_SIZE
export DP_SIZE
export XTUNER_USE_CUTLASS_GROUP_GEMM

cd "${REPO_ROOT}"
torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  .dev_scripts/validate_moeblock_tpep_vs_single.py
