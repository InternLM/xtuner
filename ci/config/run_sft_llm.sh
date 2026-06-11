#!/usr/bin/env bash

WORK_DIR=$(dirname $(dirname $(dirname "${BASH_SOURCE[0]}")))
cd $WORK_DIR

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export HCCL_RDMA_TC=132
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=512
export TORCH_HCCL_ZERO_COPY=1
export GPUS_PER_NODE=$(echo $ASCEND_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export GLOO_SOCKET_IFNAME=eth0
export ASCEND_WORK_PATH=/tmp
export XTUNER_GC_ENABLE=1
export XTUNER_ACTIVATION_OFFLOAD=1
export PYTHONPATH="$WORK_DIR"

torchrun --master-port 12384 --nproc-per-node 16 --nnodes=$WORLD_SIZE --master-addr $MASTER_ADDR --master-port $MASTER_PORT --node-rank $RANK -m xtuner.v1.train.cli.sft --config ci/config/qwen3_235B_text.py
