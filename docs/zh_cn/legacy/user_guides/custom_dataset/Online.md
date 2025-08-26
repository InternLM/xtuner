# 在线处理数据集

与离线处理数据集相比，在线处理数据集操作较为简单，适用于数据量不是特别大的训练情况。

## Step 1, 导出模板 config 文件

XTuner 中提供了用于自定义数据集微调的模板 config ，与其他基于 huggingface hub 上的数据集微调的 config 相比，只有数据部分进行了微小的修改，例如：

```diff
+ data_files = ['/path/to/json/file.json']
train_dataset = dict(
    ...,
-   dataset=dict(type=load_dataset, path='tatsu-lab/alpaca'),
+   dataset=dict(type=load_dataset, path='json', data_files=data_files),
    ...
)
```

可使用以下命令查看 XTuner 中提供的用于自定义数据集微调的模板 config：

```
xtuner list-cfg -p custom_dataset
```

若想基于 Internlm2 进行全量微调，可从上述命令输出结果中选择 `internlm2_7b_full_finetune_custom_dataset_e1` 并导出至当前目录下：

```
xtuner copy-cfg internlm2_7b_full_finetune_custom_dataset_e1 .
```

## Step 2, 修改模板 config 文件

首先，需要修改 Step 1 中导出的模板 config 中的训练数据路径部分：

```diff
- data_files = ['/path/to/json/file.json']
+ data_files = ['/path/to/your/json/file1.json',
+               '/path/to/your/json/file2.json', ...]
```

其次，需要修改 config 模板中的数据格式对应部分。若数据集满足以下格式，则不需修改：

```
[
    {
        "conversation": [
            {
                "system": "",
                "input": "xxx",
                "output": "xxx"
            },
            {
                "input": "xxx",
                "output": "xxx"
            }
        ]
    },
...
]
```

若不满足，则可以通过 `xtuner list-dataset-format` 命令查看 XTuner 中支持的数据集格式，并修改 config 模板中的数据格式对应部分。例如自定义数据集满足 Alpaca 格式，则可以修改：

```diff
+ from xtuner.dataset.map_fns import alpaca_map_fn
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    ...,
-   dataset_map_fn=None,
+   dataset_map_fn=alpaca_map_fn，
    ...
)
```

## Step 3, 启动训练

在 slurm 集群调度系统中可以通过以下命令启动训练：

```
srun ${SRUN_ARGS} xtuner train internlm2_7b_full_finetune_custom_dataset_e1_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

若出现 OOM 现象，可尝试使用 zero2 或 zero3。以下命令可以使用 zero 3 显存优化策略进行训练：

```
srun ${SRUN_ARGS} xtuner train internlm2_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero3
```

若训练数据集较大，可能需要在训练前设置环境变量 `XTUNER_DATASET_TIMEOUT` 为一个更大的数（默认为 60 分钟超时，可以酌情将其调大，如：120）：

```
XTUNER_DATASET_TIMEOUT=120 srun ${SRUN_ARGS} xtuner train internlm2_7b_full_finetune_custom_dataset_e1_copy.py --launcher slurm --deepspeed deepspeed_zero1
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

export NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
export PORT=${MASTER_PORT}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export ADDR=${MASTER_ADDR}

echo ${KUBERNETES_CONTAINER_RESOURCE_GPU}
echo ${WORLD_SIZE}
echo ${MASTER_PORT}
echo ${MASTER_ADDR}
echo ${RANK}
xtuner train internlm2_7b_full_finetune_custom_dataset_e1_copy.py \
    --deepspeed deepspeed_zero1 \
    --work-dir work_dirs/${EXP_NAME}
```

同理，对于大数据集，需要酌情调整环境变量 `XTUNER_DATASET_TIMEOUT`。

## Step 4, 转模型

deepspeed 转 hf：

```
python xtuner/tools/model_converters/pth_to_hf.py internlm2_7b_full_finetune_custom_dataset_e1_copy.py /src/model/path /hf/dst/model/path
```

hf 转 Turbomind：

```
lmdeploy convert internlm2-chat-7b /hf/dst/model/path --dst-path /turbomind/dst/model/path
```
