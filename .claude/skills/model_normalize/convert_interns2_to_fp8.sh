#!/usr/bin/env bash
set -euo pipefail

# Quantize InternS2PreviewCandidate5 to FP8 with a Qwen3.5-FP8-style rule set:
# only big Linear weights in the LM / MTP / experts are quantized; vision tower,
# all norms, routers, and linear_attn control-flow projections are kept bf16.

SOURCE=$1
TARGET=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/hf_to_fp8.py" \
    "${SOURCE}" \
    "${TARGET}" \
    -p '^(?:model\.language_model\.layers|mtp\.layers)\.\d+\.self_attn\.[qkvo]_proj\.weight$' \
    -p '^(?:model\.language_model\.layers|mtp\.layers)\.\d+\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)\.weight$' \
    -p '^(?:model\.language_model\.layers|mtp\.layers)\.\d+\.mlp\.shared_expert\.(?:gate|up|down)_proj\.weight$' \
    -p '^model\.language_model\.layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)$' \
    -p '^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$'
