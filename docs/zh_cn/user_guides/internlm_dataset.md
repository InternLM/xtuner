## 数据集格式

Internlm 训练数据集是已经被 tokenized 过的，格式如下所示：

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

其中，数值为负数的 tokens 在训练过程中不参与 loss 计算。

## 接口介绍

为了在 XTuner 中训练 InternLM-Chat 模型，需要将原数据格式转换为 XTuner 标准数据集格式。处理 InternLM 格式数据集的核心函数如下：

```python
def process(dataset_folder=None,
            cached_folder=None,
            split='train',
            pack_to_max_length=False,
            max_length=2048,
            shuffle_before_pack=True,
            num_proc=32):
    ......
```

其中：

1. dataset_folder：表示训练数据集所在路径，路径下的所有以 `.bin` 结尾的文件都会被当做训练数据。由于处理后的数据会被缓存下来，因此**只有第一次处理原始数据的时候才需要提供 dataset_folder 这一字段**。
2. cached_folder：表示处理后的数据缓存至 cached_folder 文件夹下。
3. split：通过hf datasets 读取的数据集通常是一个 DatasetDict ，需要通过split变量拿到对应的Dataset，一般使用默认值 train 即可
4. pack_to_max_length：是否将多条数据拼接为一条数据进行训练。
5. max_length：表示数据处理过程会将多条训练数据 pack 成一条长度为max_length 的数据。只有当pack_to_max_length=True时生效。
6. shuffle_before_pack：在pack前是否对数据集进行shuffle，一般使用默认的True即可。只有当pack_to_max_length=True时生效。
7. num_proc：启用多进程进行数据处理，可根据情况增加 num_proc 的数值，集群上可以设为 96 。

## 使用教程

只需修改 Config 文件中上述接口对应部分即可。以下方代码为例：

```diff
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import process_internlm_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.engine import DatasetInfoHook, ThroughputHook, EvaluateChatHook
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE
import torch

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'

# Data
- dataset_folder = './data'
- cached_folder = './packed_processed_dataset'
+ dataset_folder = '/path/to/your/dataset'
+ cached_folder = '/path/to/cache/your/processed/dataset'
max_length = 2048
pack_to_max_length = True
num_proc = 96

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 4  # 1bs * 4acc * 32gpu = 128 batchsize
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.95)
weight_decay = 0.01
max_norm = 1  # grad clip

# Evaluate the generation performance during the training
prompt_template = PROMPT_TEMPLATE.internlm_chat
evaluation_freq = 2000
SYSTEM = ''
evaluation_inputs = [
    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #q
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        torch_dtype=torch.bfloat16,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_internlm_dataset,
    dataset_folder=dataset_folder,
    cached_folder=cached_folder,
    max_length=max_length,
    pack_to_max_length=pack_to_max_length,
    num_proc=num_proc)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    )

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = dict(
    type=CosineAnnealingLR,
    eta_min=lr * 0.1,
    by_epoch=True,
    T_max=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template),
    dict(
        type=ThroughputHook
    )
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

log_processor = dict(mean_pattern=r'.*(loss|time|data_time|grad_norm|tflops).*')
```
