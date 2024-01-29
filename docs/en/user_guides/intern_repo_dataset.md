**Note: The primary aim of this document is to provide detailed instructions on how to train models based on the data format provided by the InternLM repository, rather than to train the InternLM model itself.**

## Tutorial

### Step 1, Export the Template Config File

you can export the config named \`internlm_7b_full_intern_repo_dataset_template\`\` to the current directory using the following command:

```bash
xtuner copy-cfg internlm_7b_full_intern_repo_dataset_template .
```

### Step 2, Modify the Template Config File

You only need to modify the corresponding part of the above interface in the Config file.

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'
use_varlen_attn = True

# Data
- dataset_folder = '/path/to/your/dataset'
+ dataset_folder = '/real/dataset/path'
max_length = 8192
pack_to_max_length = True
...
```

### Step 3, Start training

Slurm:

```
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

Aliyun DLC:

```diff
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

export NCCL_BUFFSIZE=2097152
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
- export EXP_NAME=debug
+ export EXP_NAME=your_exp_name
export PYTHONPATH='.':$PYTHONPATH
source ~/.bashrc
+ cd /path/to/xtuner
+ conda activate conda_env_name

echo ${KUBERNETES_CONTAINER_RESOURCE_GPU}
echo ${WORLD_SIZE}
echo ${MASTER_PORT}
echo ${MASTER_ADDR}
echo ${RANK}
python -m torch.distributed.launch \
    --nproc_per_node=${KUBERNETES_CONTAINER_RESOURCE_GPU} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    xtuner/tools/train.py \
    internlm_7b_full_intern_repo_dataset_template_copy.py \
    --deepspeed deepspeed_zero1 \
    --launcher pytorch \
    --work-dir work_dirs/${EXP_NAME}
```

## Dataset Format

The training dataset of [InternLM](https://github.com/InternLM/InternLM) is pre-tokenized, and is formatted as follows:

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

Among them, tokens with negative values are not involved in the calculation of loss during the training process.
