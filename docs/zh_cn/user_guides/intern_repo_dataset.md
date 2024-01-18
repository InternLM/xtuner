**注意：本文档的主要目标是详细说明如何根据 InternLM 仓库所提供的数据格式进行模型训练，而非如何训练 InternLM 模型。**

## 使用教程

### Step 1, 导出模板 config 文件

可以通过下列命令将名为 internlm_7b_full_intern_repo_dataset_template 的 config 导出至当前目录下：

```
xtuner copy-cfg internlm_7b_full_intern_repo_dataset_template .
```

### Step 2, 修改模板 config 文件

只需修改 Config 文件中上述接口对应部分即可。

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'

# Data
- dataset_folder = '/path/to/your/dataset'
+ dataset_folder = '/real/dataset/path'
max_length = 2048
pack_to_max_length = True
...
```

### Step 3, 启动训练

在 slurm 集群调度系统中可以通过以下命令启动训练：

```
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

在阿里云 DLC 中可通过以下命令启动训练：

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

## 数据集格式

[InternLM](https://github.com/InternLM/InternLM) 仓库所使用的训练数据集会被预先 token 化，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。
