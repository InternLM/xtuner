#!/usr/bin/env bash
config="configs/guanaco/gunaco_llama_7B.py"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 --master_port 10000 tools/train.py ${config} \
