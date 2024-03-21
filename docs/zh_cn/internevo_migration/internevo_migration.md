# 模型

XTuner 支持基于 Huggingface Hub 上的模型进行训练，如下修改 config 内容即可切换模型：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
- pretrained_model_name_or_path = 'internlm/internlm2-7b'
+ pretrained_model_name_or_path = 'internlm/internlm2-20b'

```

# 数据

InternEvo 在训练过程中通常会把多条数据拼接为一个特定的最大长度，随后输入模型训练。其配置往往满足以下形式：

```python
data = dict(
    seq_len=SEQ_LEN,
    pack_sample_into_one=False,
    min_length=MIN_LENGTH,
    train_folder=TRAIN_FOLDER,
    dataset_weights=DATASET_WEIGHTS,
    ...)
```

其中，数据配比 (`dataset_weights=DATASET_WEIGHTS`) 功能 XTuner 尚未支持。`TRAIN_FOLDER` 中的训练数据需要满足 ftdp tokenized 数据集格式：

```
|-- TRAIN_FOLDER
    |-- cn
    |   |-- dataset1
    |   |   |-- data1.bin
    |   |   |-- data1.bin.meta
    |   |-- dataset2
    |   |   |-- data2.bin
    |   |   |-- data2.bin.meta
```

在 XTuner 中实现在线数据集拼接策略需要参考 `xtuner/configs/internlm/internlm2_7b/internlm2_7b_w_tokenized_dataset.py` 文件中的配置：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Data
- dataset_folder = '/path/to/sft/data/folder'
+ dataset_folder = TRAIN_FOLDER
- max_length = 32768
+ max_length = SEQ_LEN

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    dataset_cfg=dict(
        type=load_intern_repo_tokenized_dataset,
        data_order_path=None,
        folder=dataset_folder,
-       min_length=0,
+       min_length=MIN_LENGTH,
        file_type='.bin'),
    packed_length=max_length,
    seed=1024)
```

> \[!IMPORTANT\]
> 需要注意，由于训练数据喂给模型的先后顺序可能对训练结果造成影响，因此建议不要轻易修改上述配置中的 `seed` 选项。同时，可参考[文档todo](./ftdp_dataset/Case4.md#step-3-获取数据顺序-可选)进一步固定数据顺序。

# 训练策略

## 变长注意力 (Variable Length Flash Attention)

InternEvo 通过设置[数据配置](https://github.com/InternLM/InternEvo/blob/develop/doc/usage.md#%E6%95%B0%E6%8D%AE%E9%85%8D%E7%BD%AE)中的 `pack_sample_into_one` 参数为 False 来使用“变长注意力机制”（见下图右侧）。

<div align="center">
  <img src="https://github.com/InternLM/InternEvo/blob/develop/doc/imgs/pack_into_one.png?raw=true" width="800"/>
  <br /><br />
</div>

在 XTuner 中使用这一功能需要设置 config 中的 `use_varlen_attn` 配置为 True，即可保证训练行为与 InternEvo 一致：

```diff
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-7b'
- use_varlen_attn = False
+ use_varlen_attn = True
...
```

> \[!IMPORTANT\]
> 需要注意，当设置 `use_varlen_attn = True` 后，请确保 `batch_size` 被设置为 1，且 `pack_to_max_length` 被设置为 True。

## batch_size 与 accumulative_counts

在 InternEvo 的配置中，与 batch_size 和 accumulative_counts 相关的配置有如下几个：

```python
data = dict(
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=MICRO_NUM,
    # MICRO_BATCH_SIZE * SEQ_LEN = PACKED_LENGTH
    micro_bsz=MICRO_BATCH_SIZE,
    total_steps=TOTAL_STEP,
    # 梯度累计，默认等于MICRO_NUM（BS）
    gradient_accumulation=GRADIENT_ACCUMULATION,
    ...)
```

其中：

1. `micro_num` 与 `gradient_accumulation` 通常具有相同含义，其数值默认相等。
2. `total_steps` 在 XTuner 中可以不手动指定，可通过 `max_epochs` 指定。
3. XTuner 目前只支持 `micro_bsz = 1` 。

为对齐以上配置，可参考 XTuner 中 `xtuner/configs/internlm/internlm2_7b/internlm2_7b_w_tokenized_dataset.py` 文件中的配置，并进行如下修改：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Scheduler & Optimizer
- accumulative_counts = 1
+ accumulative_counts = MICRO_NUM # or GRADIENT_ACCUMULATION
- max_epochs = 1
+ max_epochs = MAX_EPOCHS
```

## 并行训练

### ZeRO 系列显存优化

XTuner 支持使用 ZeRO 系列显存优化降低训练过程中的显存消耗：

```shell
  # 单卡
  xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
  # 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train ${CONFIG_NAME_OR_PATH} --launcher slurm --deepspeed deepspeed_zero2
```

- `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 。

### 序列并行

InternEvo 中支持了 Data Parallel、Tensor Parallel、Pipeline Parallel 和 Sequence Parallel 四种并行策略。XTuner 目前支持了 Data Parallel 和 Sequence Parallel 两种并行策略，可满足基本全部的训练需求（搭配 zero3 显存优化策略可支持 70B 模型 256K 上下文训练）。

假定 InternEvo 训练过程中：tp_world_size = TP, pp_world_size = PP, sequence_parallel = True。则在 XTuner 的配置文件中进行如下修改可保证训练行为一致：

```diff
+ from xtuner.parallel.sequence import SequenceParallelSampler

+ sequence_parallel_size = TP * PP

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataloader = dict(
-   sampler=dict(type=DefaultSampler, shuffle=True),
+   sampler=dict(type=SequenceParallelSampler, seed=1024, shuffle=True),
    ...)
```

XTuner 序列并行的详细用法请参考 [序列并行文档](../training/training_extreme_long_sequence.md)。
