#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA toolkit}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
task_python="${PYTHON:-python}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export XTUNER_DETERMINISTIC=true
task_output="${PWD}/work_dirs/glm53_cooperative_comparison"
mkdir -p "${task_output}"
export TILELANG_CACHE_DIR="${task_output}/tilelang_cache"
export TILELANG_TMP_DIR="${task_output}/tilelang_tmp"
export TRITON_CACHE_DIR="${task_output}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${task_output}/inductor_cache"

"${task_python}" -m pytest tests/ops/test_cooperative_kpool.py \
    tests/model/test_glm53_dsa.py \
    -k 'Kpool or tied_scores or packed_sharded' -q
"${task_python}" -u tools/compare_glm53_cooperative_indexer.py "$@"
