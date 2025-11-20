set -ex
export XTUNER_PACK_WORKERS=8
export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export PYTHONPATH="$(pwd)"
export HF_HOME="$(pwd)/"

config_file=$1

torchrun \
  --nnodes=${NODE_COUNT} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${PROC_PER_NODE} \
    xtuner/v1/train/cli/sft.py --config ${config_file}