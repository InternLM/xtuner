#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认使用用户指定的 fla 环境；需要切换时可在命令前覆盖 CONDA_ENV。
CONDA_ENV="${CONDA_ENV:-fla}"
CONDA_SH="${CONDA_SH:-~/miniconda3/etc/profile.d/conda.sh}"

# xtuner_ep.md 的示例固定为 EP=2；默认额外验证 4 份 DP replica。
EP_SIZE="${EP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$((EP_SIZE * DP_SIZE))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29531}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# 显式使用当前仓库代码，避免导入 conda 环境或其他目录下安装的 xtuner。
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES
export EP_SIZE
export DP_SIZE

cd "${REPO_ROOT}"
torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  .dev_scripts/validate_xtuner_ep_md.py
