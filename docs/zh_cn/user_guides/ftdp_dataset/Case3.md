# 使用 Processed 普通对话数据集训任意模型

使用尚未 token 化的 ftdp 数据进行训练，保持待训练模型的对话模板不变，且不需要进行离线处理的场景。

## Step 1, 导出模板 config 文件

XTuner 中目前提供了训练 Internlm2 的模板 config，使用命令：

```
xtuner copy-cfg internlm2_7b_w_untokenized_dataset .
```

可将训练 Internlm2 的模板 config 导出至当前目录下。

## Step 2, 修改模板 config 文件

修改模板 config 文件中的训练数据路径为真实数据路径，路径中的所有以 `.json` 为后缀的数据将会作为训练数据：

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-7b'
use_varlen_attn = True

# Data
- dataset_folder = '/mnt/petrelfs/share_data/caoweihan/v1_sample_with_legal_cate'
+ dataset_folder = '/path/to/untokenized/data'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 32768
pack_to_max_length = True
...
```

## Step 3, 获取数据顺序 （可选）

运行下面的代码可获取数据顺序，并存为 txt 文件：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/untokenized/data \
    --save-folder /folder/to/save/data/order \
    --file-type ${file_type}
```

其中，`--file-type ${file_type}` 表示需要统计所有以 `${file_type}` 为文件名后缀的文件的顺序。

例如，需要获取 `/path/to/untokenized/data` 路径下所有以 `.json` 结尾的文件的顺序，并保存在当前路径下，那么上述命令需要改为：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/untokenized/data \
    --save-folder . \
    --file-type .json
```

同时，需要进一步修改 Step 2 中的 Config 文件，并设置数据顺序文件路径：

```diff
...
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    dataset_cfg=dict(
        type=load_intern_repo_tokenized_dataset,
-       data_order_path=None,
+       data_order_path='/folder/to/save/data/order/'+'data_order.txt',
        folder=dataset_folder,
        min_length=0,
-       file_type='.bin'  # 指定 data_order_path 后，file_type 参数就不需要设置了
    ),
    packed_length=max_length,
    seed=1024)
```

## Step 4, 启动训练

在 slurm 集群调度系统中可以通过以下命令启动训练：

```
srun ${SRUN_ARGS} xtuner train internlm2_7b_w_untokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

若出现 OOM 现象，可尝试使用 zero2 或 zero3。以下命令可以使用 zero 3 显存优化策略进行训练：

```
srun ${SRUN_ARGS} xtuner train internlm2_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero3
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
xtuner train internlm2_7b_w_untokenized_dataset_copy.py \
    --deepspeed deepspeed_zero1 \
    --work-dir work_dirs/${EXP_NAME}

```

## Step 5, 转模型

deepspeed 转 hf：

```
python xtuner/tools/model_converters/pth_to_hf.py internlm2_7b_w_untokenized_dataset_copy.py /src/model/path /hf/dst/model/path
```

hf 转 Turbomind：

```
lmdeploy convert internlm2-chat-7b /hf/dst/model/path --dst-path /turbomind/dst/model/path
```
