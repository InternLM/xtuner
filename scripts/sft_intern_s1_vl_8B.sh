set -ex
export XTUNER_PACK_WORKERS=8
export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export PYTHONPATH="$(pwd)"
export HF_HOME="$(pwd)/"

# config_file=examples/v1/internvl/sft_internvl3.5_8B_config.py
# work_dir="/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_saved_model/internvl3.5/internvl3.5-8B-sft-bs512-maxsteps8000-lr8e-5"

# config_file=examples/v1/interns1/sft_intern_s1_8B_config.py
# work_dir="/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_saved_model/interns1/interns1-mini-sft-bs512-maxsteps8000-lr8e-5"

config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny.py
work_dir="/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_saved_model/internvl3.5/internvl3.5-8B-sft-based-3.5tiny-data-bs32-epoch1-lr2e-5"

# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --nproc-per-node 8  \
#     xtuner/v1/train/cli/sft.py --config ${config_file}

current_time=$(date "+%m%d%H")
if [ ! -d "$work_dir" ]; then
  mkdir -p "$work_dir"
fi
cp "$config_file" "${work_dir}"
torchrun \
  --nnodes=${NODE_COUNT} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${PROC_PER_NODE} \
    xtuner/v1/train/cli/sft.py --config ${config_file}  \
    2>&1 | tee -a "${work_dir}/training_log_${current_time}.txt"