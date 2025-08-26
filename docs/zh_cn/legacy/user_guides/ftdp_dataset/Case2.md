# 使用 Processed 数据集训练非 InternLM2 模型

使用尚未 token 化的 ftdp 数据训练其他模型（以 Mistral 为例），且需要用 Internlm2 对话模板覆盖原有对话模板以便让模型掌握 agent 、tool 能力。

## Step 1, 离线处理数据集

ftdp 把 sft 任务的数据处理划分为三个类型，原始数据（origin）、预处理数据（processed）和 token 过的数据（tokenized）。我们需要将预处理过的、具有统一格式的 ftdp 数据 token 化得到直接可以用于训练的格式。其中，预处理数据需要满足以下目录结构：

```
|-- processed-dir
    |-- data1
    |   |-- processed
    |       |-- sft_chat
    |           |-- data1.jsonl
    |-- data2
    |   |-- processed
    |       |-- sft_chat
    |           |-- data2.jsonl
```

使用以下命令可离线 token 化 ftdp 格式的预处理数据（processed）数据集：

```
python xtuner/tools/tokenize_ftdp_datasets.py \
    --processed-dir /path/to/preprocessed/data \
    --tokenized-dir /path/to/tokenized/data \
    --tokenizer-path pretrained_model_name_or_path \
    --tokenizer-w-special-tokens-save-dir /path/to/save/new/tokenizer
```

上述命令中：

1. `--processed-dir` 需要指定预处理后的，具有 ftdp 标准格式的数据路径（同 Case 1）；
2. `--tokenized-dir` 需要指定为 token 化后的数据存储路径（同 Case 1）；
3. `--tokenizer-path pretrained_model_name_or_path` 中的 `pretrained_model_name_or_path` 同 `from_pretrained` 接口中的 `pretrained_model_name_or_path`（同 Case 1）；
4. 由于除 Internlm2 外的其他模型（如 mistral 等）没有 internlm2-chat 模型的智能体、工具调用等功能的对话模板，因此对于非 internlm2 模型，需要将 internlm2-chat 对话模板中的一些特殊字符（如：\<|im_start|>、\<|plugin|>等）加入到新模型的 tokenizer 的 special tokens 中，需要通过 `--tokenizer-w-special-tokens-save-dir` 指定新 tokenizer 的存储路径。**同时，后续训练过程需要使用新保存的 tokenizer 而非原始 tokenizer。**

## Step 2, 导出模板 config 文件

XTuner 中目前提供了训练 Mistral 的模板 config，使用命令：

```
xtuner copy-cfg mistral_7b_w_tokenized_dataset .
```

可将训练 Mistral 的模板 config 导出至当前目录下。

## Step 3, 修改模板 config 文件

1. 修改模板 config 文件中的训练数据路径为真实数据路径，其中 `/path/to/tokenized/data` 需要基于 Step 1 中的 `/path/to/tokenized/data` 进一步指定 train folder，即 `/path/to/tokenized/data/chatml_llamav13_32k/train/` 。
2. 需要修改 tokenizer 路径为 Step 1 保存的路径 `/path/to/save/new/tokenizer`。
3. 由于 Step 1 扩充了 tokenizer 的词表，因此需要将新 tokenizer 传入 `SupervisedFinetune` 中，以扩展 llm model 的词表大小。

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'mistralai/Mistral-7B-v0.1'
# 已经使用 Internlm2 的对话模板覆盖了 Mistral 的原有模板，new tokenizer 中已经
# 添加了 Internlm2 对话模板中的特殊字符。
# 请参考 docs/zh_cn/user_guides/finetune_custom_dataset.md
- tokenizer_path = '/new/tokenizer/path'
+ tokenizer_path = '/path/to/save/new/tokenizer'
use_varlen_attn = True

# Data
- dataset_folder = '/path/to/sft/data/folder'
+ dataset_folder = '/path/to/tokenized/data/chatml_llamav13_32k/train'
# 已经使用 Internlm2 的对话模板覆盖了 Mistral 的原有模板
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 32768
pack_to_max_length = True
...

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
model = dict(
+   tokenizer=tokenizer,
    ...)
```

在使用 DeepSpeed 训练模型时，如需在保存 checkpoint 时只保存模型权重，而不保存优化器状态，可参考以下步骤：

1. 确保 mmengine 版本大于等于 0.10.3

```
pip install 'mmengine>=0.10.3'
```

2. 修改 Config 文件，CheckpointHook 增加 save_optimizer=False

```diff
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=1),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
   checkpoint=dict(
        type=CheckpointHook,
+       save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)
```

需要注意，经过以上设置后，训练过程不可 resume 。

## Step 4, 获取数据顺序 （可选）

运行下面的代码可获取数据顺序，并存为 txt 文件：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/tokenized/data \
    --save-folder /folder/to/save/data/order \
    --file-type ${file_type}
```

其中，`--file-type ${file_type}` 表示需要统计所有以 `${file_type}` 为文件名后缀的文件的顺序。

例如，需要获取 `/path/to/tokenized/data` 路径下所有以 `.bin` 结尾的文件的顺序，并保存在当前路径下，那么上述命令需要改为：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/tokenized/data \
    --save-folder . \
    --file-type .bin
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
        file_type='.bin'
    ),
    packed_length=max_length,
    seed=1024)
```

## Step 5, 启动训练

注：训练前期（几十个 iters）loss 偏高是正常现象，因为模型需要时间学习 Internlm2 的对话模板。

在 slurm 集群调度系统中可以通过以下命令启动训练：

```
srun ${SRUN_ARGS} xtuner train mistral_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero1
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
xtuner train mistral_7b_w_tokenized_dataset_copy.py \
    --deepspeed deepspeed_zero1 \
    --work-dir work_dirs/${EXP_NAME}

```

## Step 6, 转模型

deepspeed 转 hf：

```
python xtuner/tools/model_converters/pth_to_hf.py mistral_7b_w_tokenized_dataset_copy.py /src/model/path /hf/dst/model/path
```

hf 转 Turbomind：

```
lmdeploy convert internlm2-chat-7b /hf/dst/model/path --dst-path /turbomind/dst/model/path
```

## Step 7，Turbomind 评测

评测前需要按照[Opencompass 使用文档](https://aicarrier.feishu.cn/wiki/PR28wWg3tiY2xCkuysccRBNenIf#RNcbdEVZ9oulPQxFz9gcOxwjnff)准备环境。

使用内部版 Opencompass 的 ca949db74502a68c8a900afdf751c584fb7c7655 这个 commit id 进行评测。在 `configs/sft_cfg/7B/Ampere_chatml_v053/` 目录下添加如下 config ：

```diff
import os.path as osp
from copy import deepcopy

from mmengine.config import read_base

with read_base():
    # datasets
    from ...dataset_collections.medium_chat_sft_v053 import \
        base_datasets, longtext_datasets, math_agent_datasets, cibench_datasets, plugin_eval_datasets
    # summarizer
    from ...summarizers.medium_chat_sft_v053 import summarizer
    # clusters
    from ...clusters.slurm_llmit2 import infer, eval
    # lark robot
    from ...lark import lark_bot_url
    # base models cfg
    from .base_model.base_model_turbomind import base_model_cfg, base_longtext_model_cfg, base_agent_llm_cfg, base_math_agent_cfg, \
        base_cibench_agent_cfg, base_plugin_eval_model_cfg

# ------------------ change here ↓ ------------------
models_path = [
+     '/path/to/turbomind_model'
]

# users can set `auto`, `spot`, or `reserved`. Defaults to `auto`.
infer['runner']['quotatype'] = 'auto'
infer['runner']['max_num_workers'] = 32
infer['runner']['partition'] = 'llmit2'

eval['runner']['quotatype'] = 'auto'
eval['runner']['max_num_workers'] = 64
eval['runner']['partition'] = 'llmit2'
# ------------------ change end ------------------

# ------------------ default settings ↓ ------------------
# careful to change the following settings

# add different eval models
base_models = []
longtext_models = []
math_agent_models = []
cibench_agent_models = []
plugin_eval_models = []
for model_path in models_path:
    if model_path.endswith('/'):
        model_path = model_path[:-1]
    abbr = osp.split(osp.split(model_path)[0])[-1]
    ckpt_iter = osp.split(model_path)[-1]

    summarizer_abbr = f"{abbr}@{ckpt_iter}"

    tmp_base_model_cfg = deepcopy(base_model_cfg)
    tmp_base_model_cfg['abbr'] = f"{abbr}@{ckpt_iter}"
    tmp_base_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_base_model_cfg['path'] = model_path

    # process base model
    base_models.append(tmp_base_model_cfg)

    # process longtext model
    tmp_longtext_model_cfg = deepcopy(base_longtext_model_cfg)
    tmp_longtext_model_cfg['abbr'] = f"{abbr}@{ckpt_iter}-longtext"
    tmp_longtext_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_longtext_model_cfg['path'] = model_path
    longtext_models.append(tmp_longtext_model_cfg)

    # set agent model cfg
    tmp_agent_llm_cfg = deepcopy(base_agent_llm_cfg)
    tmp_agent_llm_cfg['path'] = model_path

    # process math agent model
    tmp_math_agent_cfg = deepcopy(base_math_agent_cfg)
    tmp_math_agent_cfg['abbr'] = f"{abbr}@{ckpt_iter}-math-react"
    tmp_math_agent_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_math_agent_cfg['llm'] = tmp_agent_llm_cfg
    math_agent_models.append(tmp_math_agent_cfg)

    # process cibench agent model
    tmp_cibench_agent_cfg = deepcopy(base_cibench_agent_cfg)
    tmp_cibench_agent_cfg['abbr'] = f"{abbr}@{ckpt_iter}-cibench-react"
    tmp_cibench_agent_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_cibench_agent_cfg['llm'] = tmp_agent_llm_cfg
    cibench_agent_models.append(tmp_cibench_agent_cfg)

    # process plugin eval model
    tmp_plugin_eval_model_cfg = deepcopy(base_plugin_eval_model_cfg)
    tmp_plugin_eval_model_cfg['abbr'] = f"{abbr}@{ckpt_iter}-plugin-eval"
    tmp_plugin_eval_model_cfg['summarizer_abbr'] = summarizer_abbr
    tmp_plugin_eval_model_cfg['path'] = model_path
    plugin_eval_models.append(tmp_plugin_eval_model_cfg)

del tmp_base_model_cfg, tmp_longtext_model_cfg, tmp_agent_llm_cfg, \
    tmp_math_agent_cfg, tmp_cibench_agent_cfg, tmp_plugin_eval_model_cfg

# set all models
model_dataset_combinations = []
models = []
datasets = []

# The agent test is relatively slow, so they placed first.
# process longtext datasets
model_dataset_combinations.append(dict(models=longtext_models, datasets=longtext_datasets))
models.extend(longtext_models)
datasets.extend(longtext_datasets)
# process math agent datasets
model_dataset_combinations.append(dict(models=math_agent_models, datasets=math_agent_datasets))
models.extend(math_agent_models)
datasets.extend(math_agent_datasets)
# process cibench agent datasets
model_dataset_combinations.append(dict(models=cibench_agent_models, datasets=cibench_datasets))
models.extend(cibench_agent_models)
datasets.extend(cibench_datasets)
# process plugin eval datasets
model_dataset_combinations.append(dict(models=plugin_eval_models, datasets=plugin_eval_datasets))
models.extend(plugin_eval_models)
datasets.extend(plugin_eval_datasets)

# process base datasets
model_dataset_combinations.append(dict(models=base_models, datasets=base_datasets))
models.extend(base_models)
datasets.extend(base_datasets)

# ------------------ default settings end ------------------

```
