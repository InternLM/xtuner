#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA toolkit}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
task_python="${PYTHON:-python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
task_output="${PWD}/work_dirs/glm53_block_profile"
mkdir -p "${task_output}"
export TILELANG_CACHE_DIR="${task_output}/tilelang_cache"
export TILELANG_TMP_DIR="${task_output}/tilelang_tmp"
export TRITON_CACHE_DIR="${task_output}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${task_output}/inductor_cache"
export CUDA_CACHE_PATH="${task_output}/cuda_cache"
export XDG_CACHE_HOME="${task_output}/xdg_cache"
export TMPDIR="${task_output}/tmp"
mkdir -p "${TMPDIR}"
exec "${task_python}" \
    -m torch.distributed.run --nproc-per-node=8 --master-port=29763 tools/compare_glm53_sp_indexer_balance.py "$@"
