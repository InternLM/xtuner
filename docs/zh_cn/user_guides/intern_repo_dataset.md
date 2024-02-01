**注意：本文档的主要目标是详细说明如何根据 InternLM 仓库所提供的数据格式进行模型训练，而非如何训练 InternLM 模型。**

# 使用 tokenized 数据集进行训练

## 使用教程

### Step 1, 导出模板 config 文件

可以通过下列命令将名为 internlm2_7b_w_tokenized_dataset 的 config 导出至当前目录下：

```
xtuner copy-cfg internlm2_7b_w_tokenized_dataset .
```

### Step 2, 修改模板 config 文件

修改 Config 文件中上述接口对应部分。

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-7b'
use_varlen_attn = True

# Data
- dataset_folder = '/path/to/sft/data/folder'  # noqa: E501
+ dataset_folder = '/real/dataset/path'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 32768
pack_to_max_length = True
...
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

### Step 3，获取数据顺序 （可选）

运行下面的代码可获取数据顺序，并存为 txt 文件：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/your/data \
    --save-folder /folder/to/save/data/order \
    --file-type ${file_type}
```

其中，`--file-type ${file_type}` 表示需要统计所有以 `${file_type}` 为文件名后缀的文件的顺序。

例如，需要获取 `/path/to/your/data` 路径下所有以 `.bin` 结尾的文件的顺序，并保存在当前路径下，那么上述命令需要改为：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/your/data \
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

### Step 4, 启动训练

在 slurm 集群调度系统中可以通过以下命令启动训练：

```
srun ${SRUN_ARGS} xtuner train internlm2_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero1
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
xtuner train internlm2_7b_w_tokenized_dataset_copy.py \
    --deepspeed deepspeed_zero1 \
    --work-dir work_dirs/${EXP_NAME}

```

### Step 5，转模型

deepspeed 转 hf：

```
python xtuner/tools/model_converters/pth_to_hf.py internlm2_7b_w_tokenized_dataset_copy.py /src/model/path /hf/dst/model/path
```

hf 转 Turbomind：

```
lmdeploy convert internlm2-chat-7b /hf/dst/model/path --dst-path /turbomind/dst/model/path
```

### Step 6，Turbomind 评测

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

## 数据集格式

[InternLM](https://github.com/InternLM/InternLM) 仓库所使用的训练数据集会被预先 token 化，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。

# 使用 untokenized 数据集进行训练

## 使用教程

### Step 1, 导出模板 config 文件

可以通过下列命令将名为 internlm2_7b_w_untokenized_dataset 的 config 导出至当前目录下：

```
xtuner copy-cfg internlm2_7b_w_untokenized_dataset .
```

### Step 2, 修改模板 config 文件

修改 Config 文件中上述接口对应部分。

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-7b'
use_varlen_attn = True

# Data
- dataset_folder = '/mnt/petrelfs/share_data/caoweihan/v1_sample_with_legal_cate'  # noqa: E501
+ dataset_folder = '/real/dataset/path'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 32768
pack_to_max_length = True
...
```

### Step 3，获取数据顺序 （可选）

运行下面的代码可获取数据顺序，并存为 txt 文件：

```
python xtuner/tools/get_data_order.py \
    --data-folder /path/to/your/data \
    --save-folder /folder/to/save/data/order \
    --file-type .json
```

其中，`--file-type .json` 表示需要获取所有以 `.json` 为结尾的文件的顺序。

同时，需要修改 Step 2 中的 Config 文件，并设置数据顺序文件路径：

```diff
...
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    dataset_cfg=dict(
        type=load_intern_repo_untokenized_dataset,
-       data_order_path=None,
+       data_order_path='/folder/to/save/data/order/'+'data_order.txt',
        folder=dataset_folder,
        tokenizer=tokenizer,
        max_length=max_length,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        file_type='.json'),
    packed_length=max_length,
    seed=1024)
```

### Step 4，离线 token 化并处理原数据集 （可选）

对于大数据集，将原始数据集 token 化，并添加对话模板的过程可能较为耗时，因此可以先离线处理好，每次使用时直接读取处理好的数据集。

运行以下代码对原始数据集进行离线处理：

```
python xtuner/tools/process_untokenized_datasets.py \
    --data-folder /path/to/data/folder \
    --save-folder ./processed \
    --tokenizer-path pretrained_model_name_or_path \
    --prompt-template internlm2_chat \
    --dataset-format ftdp
```

其中 `pretrained_model_name_or_path` 同 `from_pretrained` 接口中的 `pretrained_model_name_or_path`，`--prompt-template` 表示对话模板的种类，其他可选对话模板可参考 [templates](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/prompt_template.md)。untokenized internlm repo 格式的数据集（别名 ftdp 格式）满足以下格式：

```
[
    {
        'role': 'user',
        'content': 'xxx'
    },
    {
        'role': 'assistant',
        'content': 'xxx'
    },
    ...
]
```

`--dataset-format` 一项需要设为 `ftdp`。

使用离线处理好的数据集进行训练，需要额外修改 Step 2 中的 Config 文件，并设置存放离线处理后的数据集路径：

```diff
...
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    dataset_cfg=dict(
        type=load_intern_repo_untokenized_dataset,
+       processed_dataset_dict_path=/folder/to/save/processed/data,
-       data_order_path=None,
-       folder=dataset_folder,
-       tokenizer=tokenizer,
-       max_length=max_length,
-       template_map_fn=dict(
-           type=template_map_fn_factory, template=prompt_template),
-       file_type='.json'),
    packed_length=max_length,
    seed=1024)
...
```

### Step 4, 5, 6, 7，同上

## 数据集格式

untokenized internlm repo 格式的数据集（别名 ftdp 格式）满足以下格式：

```
[
    {
        'role': 'user',
        'content': 'xxx'
    },
    {
        'role': 'assistant',
        'content': 'xxx'
    },
    ...
]
[
    {
        'role': 'user',
        'content': 'xxx'
    },
    {
        'role': 'assistant',
        'content': 'xxx'
    },
    ...
]
```

其中 user 对应的内容在训练过程中不参与 loss 的计算。
