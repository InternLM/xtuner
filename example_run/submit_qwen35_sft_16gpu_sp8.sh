#!/usr/bin/env bash
set -euo pipefail
set -x

# 16 GPUs = 2 nodes x 8 GPUs per node.
#
# Muon currently does not support EP > 1 in this XTuner version, so use
# EP=1 and shard the language model over all 16 ranks with FSDP. SP=8 splits
# every packed 256K sequence over 8 ranks (32K tokens per rank).
gpu_group="${GPU_GROUP:?Set GPU_GROUP to the rjob charged group}"
namespace="${NAMESPACE:?Set NAMESPACE to the rjob namespace}"
gpus_per_node=8
num_nodes=2

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xtuner_path="$(cd -- "${script_dir}/.." && pwd)"
# The Python config reads all world-size-dependent values from the environment,
# so the same config can be used for both 16-GPU and 32-GPU submissions.
config_file="${script_dir}/sft_qwen35_35b_256k_32gpu.py"
meta_data_path="${META_DATA_PATH:-${script_dir}/meta.json}"

model_path="${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5 model snapshot directory}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the training output directory}"
tokenizer_cache_dir="${TOKENIZER_CACHE_DIR:?Set TOKENIZER_CACHE_DIR to the tokenizer cache directory}"
log_dir="${LOG_DIR:-${output_root}/logs}"
ceph_config="${CEPH_CONFIG:-}"

image="${IMAGE:?Set IMAGE to the training image}"
meta_name=$(basename "${meta_data_path}")
meta_name=${meta_name%.*}
run_tag="${meta_name}-16gpu-ep1-sp8-muon-$(date +%Y%m%d-%H%M%S)"
job_name="qwen35-sft-${run_tag}"
work_dir="${output_root}/${run_tag}"

required_files=(
    "${config_file}"
    "${meta_data_path}"
    "${model_path}/config.json"
    "${model_path}/tokenizer_config.json"
    "${model_path}/model.safetensors.index.json"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file does not exist: ${required_file}" >&2
        exit 2
    fi
done

submit_mode_args=()
if [[ "${PREDICT_ONLY:-false}" == "true" ]]; then
    submit_mode_args+=(--predict-only=true)
fi

mount_args=()
if [[ -n "${RJOB_MOUNTS:-}" ]]; then
    IFS=',' read -r -a mount_specs <<< "${RJOB_MOUNTS}"
    for mount_spec in "${mount_specs[@]}"; do
        [[ -n "${mount_spec}" ]] && mount_args+=(--mount="${mount_spec}")
    done
fi

rjob submit \
    "${submit_mode_args[@]}" \
    --name="${job_name}" \
    --task_name t0 \
    --gpu="${gpus_per_node}" \
    --memory=1500000 \
    --cpu=50 \
    --charged-group="${gpu_group}" \
    --namespace="${namespace}" \
    --private-machine=group \
    -P "${num_nodes}" \
    --image="${image}" \
    "${mount_args[@]}" \
    --host-network=true \
    --gang-start=true \
    --custom-resources=rdma/mlnx_shared=8 \
    --custom-resources=mellanox.com/mlnx_rdma=1 \
    -e DISTRIBUTED_JOB=true \
    -e XTUNER_PATH="${xtuner_path}" \
    -e CONFIG_FILE="${config_file}" \
    -e MODEL_PATH="${model_path}" \
    -e META_DATA_PATH="${meta_data_path}" \
    -e CEPH_CONFIG="${ceph_config}" \
    -e WORK_DIR="${work_dir}" \
    -e TOKENIZER_CACHE_DIR="${tokenizer_cache_dir}" \
    -e XTUNER_TOKENIZE_DEBUG_SAMPLES="${XTUNER_TOKENIZE_DEBUG_SAMPLES:-0}" \
    -e LOG_DIR="${log_dir}" \
    -e GPUS_PER_NODE="${gpus_per_node}" \
    -e TORCHRUN_NNODES="${num_nodes}" \
    -e SAMPLE_MAX_LENGTH=262144 \
    -e PACK_MAX_LENGTH=262144 \
    -e GLOBAL_BATCH_SIZE=8 \
    -e SP_SIZE=8 \
    -e TP_SIZE=1 \
    -e EP_SIZE=1 \
    -e NUM_WORKERS=4 \
    -e PACK_EXTRA_BUFFER_SIZE=20 \
    -e RAND_VIDEO_MAX_FRAMES=24 \
    -e MAX_PIXELS=16777216 \
    -e LR=2e-5 \
    -e LR_MIN=1e-6 \
    -e WEIGHT_DECAY=0.05 \
    -e WARMUP_RATIO=0.1 \
    -e RECOMPUTE_RATIO=1.0 \
    -e LOSS_REDUCTION=square \
    -e TORCH_COMPILE=true \
    -e TOTAL_EPOCH=1 \
    -e HF_INTERVAL=500 \
    -e HF_MAX_KEEP=2 \
    -e CHECKPOINT_INTERVAL=500 \
    -e CHECKPOINT_MAXKEEP=2 \
    -- bash -lc '
        set -euo pipefail
        set -x

        export PYTHONPATH="${XTUNER_PATH}:${PYTHONPATH:-}"
        export TORCHRUN_NODE_RANK="${NODE_RANK:-${RANK:-}}"
        export MASTER_PORT="${MASTER_PORT:-29500}"
        export LOG_FILE="${LOG_DIR}/qwen35-sft-16gpu-ep1-sp8-muon-node${TORCHRUN_NODE_RANK:-unknown}.log"

        if [ -z "${TORCHRUN_NODE_RANK}" ] || [ -z "${MASTER_ADDR:-}" ]; then
            echo "ERROR: NODE_RANK/RANK and MASTER_ADDR are required."
            env | sort | grep -E \
                "^(NODE_RANK|RANK|NODE_COUNT|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|HOSTNAME)=" \
                || true
            exit 2
        fi

        if [ "${TORCHRUN_NNODES}" != "1" ] && {
            [ "${MASTER_ADDR}" = "127.0.0.1" ] ||
                [ "${MASTER_ADDR}" = "localhost" ]
        }; then
            echo "ERROR: MASTER_ADDR=${MASTER_ADDR} is invalid for a multi-node job."
            exit 2
        fi

        mkdir -p "${LOG_DIR}"
        exec > >(tee "${LOG_FILE}") 2>&1

        cd "${XTUNER_PATH}"
        ls -l "${CONFIG_FILE}" "${META_DATA_PATH}"
        python -c "import xtuner; print(\"xtuner import:\", xtuner.__file__)"

        torchrun \
            --nproc-per-node="${GPUS_PER_NODE}" \
            --nnodes="${TORCHRUN_NNODES}" \
            --node_rank="${TORCHRUN_NODE_RANK}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            xtuner/v1/train/cli/sft.py \
            --config="${CONFIG_FILE}"
    '
