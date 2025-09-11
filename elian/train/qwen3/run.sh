#!/bin/bash
# ==========================================
#   Training Script for Multi-GPU Training
# ==========================================
# step1: cd /data1/nuist_llm/TrainLLM/SFT-elian/xtuner
# step2: YOUR_ENV_PATH='/home/202312150002/anaconda3/envs/llm/lib/python3.10/site-packages'
# step3: cp -r ./xtuner $YOUR_ENV_PATH
# step4: bash ./elian/train/qwen3/run.sh

# conda
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate xtuner
TRAIN_PATH=/data1/nuist_llm/TrainLLM/SFT-elian/xtuner/elian/train/qwen3
cd $TRAIN_PATH || exit 1

# cuda 
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-12.4"
echo "CUDA version used: $($CUDA_HOME/bin/nvcc --version | grep 'release' | awk '{print $6}')"

# node
NUM_NODES=1
GPU_LIST="0,3"
NUM_GPUS_PER_NODE=$(echo $GPU_LIST | awk -F',' '{print NF}')
export NPROC_PER_NODE=2
export NNODES=1
export NODE_RANK=0
export PORT=10171
export NODE_0_ADDR=172.16.107.15
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo,eth0 

# train params
TRAIN_SCRIPT="$TRAIN_PATH/main.py"
export CUDA_VISIBLE_DEVICES=$GPU_LIST
echo "Elian-Xtuner-V0.2.0 (used GPUs: ${CUDA_VISIBLE_DEVICES})"
export XTUNER_DETERMINISTIC=true # torch.use_deterministic_algorithms

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
         --nnodes=$NUM_NODES \
         --node_rank=$NODE_RANK \
         --master_addr=$NODE_0_ADDR \
         --master_port=$PORT \
         $TRAIN_SCRIPT